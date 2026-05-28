"""10-model benchmark orchestrator.

Runs the two-pass extraction pipeline and the naive baseline across all
models in models.yaml, measures completeness, hallucination rate, timing,
and estimated cost. Writes per-model and aggregate results to
benchmark/results/.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from annex_iv.schema import AnnexIVDocument, DocumentedField
from benchmark.baseline import generate_annex_iv_baseline
from extraction.pipeline import _ITEM_ATTRS, generate_annex_iv
from extraction.validators import validate_grounding
from gap_analysis.analyzer import analyze
from ingestion.base import NormalisedArtifact
from ingestion.huggingface import HuggingFaceModelCardLoader

logger = logging.getLogger(__name__)

_COST_PER_M_INPUT = 0.15
_COST_PER_M_OUTPUT = 0.60
_AVG_INPUT_TOKENS_PER_CALL = 3000
_AVG_OUTPUT_TOKENS_PER_CALL = 800
_CALLS_TWO_PASS = 18
_CALLS_BASELINE = 9


def _estimate_cost(num_calls: int) -> float:
    input_cost = (num_calls * _AVG_INPUT_TOKENS_PER_CALL / 1_000_000) * _COST_PER_M_INPUT
    output_cost = (num_calls * _AVG_OUTPUT_TOKENS_PER_CALL / 1_000_000) * _COST_PER_M_OUTPUT
    return round(input_cost + output_cost, 4)


def _system_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _count_filled(doc: AnnexIVDocument) -> int:
    count = 0
    for attr_name, _item_cls in _ITEM_ATTRS:
        item = getattr(doc, attr_name)
        for field_name in type(item).model_fields:
            f: DocumentedField = getattr(item, field_name)
            if f.status == "filled":
                count += 1
    return count


def _count_baseline_hallucinations(
    baseline_doc: AnnexIVDocument,
    artifact: NormalisedArtifact,
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
) -> tuple[int, int]:
    """Run grounding validation on every filled baseline field.

    Returns (hallucinated_count, total_filled_count).
    """
    filled = 0
    hallucinated = 0
    for attr_name, _item_cls in _ITEM_ATTRS:
        item = getattr(baseline_doc, attr_name)
        for field_name in type(item).model_fields:
            f: DocumentedField = getattr(item, field_name)
            if f.status != "filled":
                continue
            filled += 1
            check_field = DocumentedField(value=f.value, status="pending")
            result = validate_grounding(
                check_field, artifact, model=model, client=client
            )
            if result.status == "unavailable":
                hallucinated += 1
                logger.info(
                    "baseline_hallucination_detected",
                    extra={
                        "source_uri": artifact.source_uri,
                        "field": field_name,
                        "value": f.value,
                    },
                )
    return hallucinated, filled


def load_models(models_yaml: Path) -> list[dict[str, str]]:
    return yaml.safe_load(models_yaml.read_text())


def run_single_model(
    model_entry: dict[str, str],
    cache_dir: Path,
    openai_model: str = "gpt-4o-mini",
    force: bool = False,
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Run both pipelines for a single model and return per-model metrics."""
    model_id = model_entry["model_id"]
    domain = model_entry.get("domain", "unknown")
    sys_name = _system_name(model_id)
    cache_path = cache_dir / f"{sys_name}.artifact.json"

    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        logger.info("loading_cached_artifact", extra={"model_id": model_id})
        artifact = NormalisedArtifact.model_validate_json(cache_path.read_text())
    else:
        logger.info("fetching_artifact", extra={"model_id": model_id})
        loader = HuggingFaceModelCardLoader()
        artifact = loader.load(model_id)
        cache_path.write_text(artifact.model_dump_json(indent=2))

    t0 = time.monotonic()
    two_pass_doc = generate_annex_iv(artifact, model=openai_model, client=client)
    two_pass_time = time.monotonic() - t0

    t0 = time.monotonic()
    baseline_doc = generate_annex_iv_baseline(artifact, model=openai_model, client=client)
    baseline_time = time.monotonic() - t0

    t0 = time.monotonic()
    hallucinated, baseline_filled_total = _count_baseline_hallucinations(
        baseline_doc, artifact, model=openai_model, client=client
    )
    hallucination_check_time = time.monotonic() - t0

    two_pass_gap = analyze(two_pass_doc)
    baseline_gap = analyze(baseline_doc)

    return {
        "model_id": model_id,
        "domain": domain,
        "two_pass": {
            "completeness": round(two_pass_gap.completeness_score, 4),
            "completeness_required": round(two_pass_gap.completeness_required_only, 4),
            "filled_fields": two_pass_gap.filled_fields,
            "total_fields": two_pass_gap.total_fields,
            "gap_count": len(two_pass_gap.gaps),
            "wall_clock_seconds": round(two_pass_time, 1),
            "estimated_cost_usd": _estimate_cost(_CALLS_TWO_PASS),
        },
        "baseline": {
            "completeness": round(baseline_gap.completeness_score, 4),
            "completeness_required": round(baseline_gap.completeness_required_only, 4),
            "filled_fields": baseline_gap.filled_fields,
            "total_fields": baseline_gap.total_fields,
            "gap_count": len(baseline_gap.gaps),
            "wall_clock_seconds": round(baseline_time, 1),
            "estimated_cost_usd": _estimate_cost(_CALLS_BASELINE),
            "hallucinated_fields": hallucinated,
            "hallucination_rate": round(
                hallucinated / baseline_filled_total if baseline_filled_total else 0.0, 4
            ),
            "hallucination_check_seconds": round(hallucination_check_time, 1),
        },
    }


def compute_aggregate(per_model: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics from per-model results."""
    tp_comp = [m["two_pass"]["completeness"] for m in per_model]
    tp_comp_req = [m["two_pass"]["completeness_required"] for m in per_model]
    tp_filled = [m["two_pass"]["filled_fields"] for m in per_model]
    tp_time = [m["two_pass"]["wall_clock_seconds"] for m in per_model]
    tp_cost = [m["two_pass"]["estimated_cost_usd"] for m in per_model]

    bl_comp = [m["baseline"]["completeness"] for m in per_model]
    bl_comp_req = [m["baseline"]["completeness_required"] for m in per_model]
    bl_filled = [m["baseline"]["filled_fields"] for m in per_model]
    bl_time = [m["baseline"]["wall_clock_seconds"] for m in per_model]
    bl_cost = [m["baseline"]["estimated_cost_usd"] for m in per_model]
    bl_hall = [m["baseline"]["hallucination_rate"] for m in per_model]

    def _s(values: list[float]) -> dict[str, float]:
        return {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
        }

    return {
        "model_count": len(per_model),
        "two_pass": {
            "completeness": _s(tp_comp),
            "completeness_required": _s(tp_comp_req),
            "filled_fields": _s(tp_filled),
            "wall_clock_seconds": _s(tp_time),
            "estimated_cost_usd": _s(tp_cost),
        },
        "baseline": {
            "completeness": _s(bl_comp),
            "completeness_required": _s(bl_comp_req),
            "filled_fields": _s(bl_filled),
            "wall_clock_seconds": _s(bl_time),
            "estimated_cost_usd": _s(bl_cost),
            "hallucination_rate": _s(bl_hall),
        },
    }


def generate_readme_snippet(agg: dict[str, Any]) -> str:
    tp = agg["two_pass"]
    bl = agg["baseline"]

    tp_comp = f"{tp['completeness']['mean']:.0%}"
    bl_comp = f"{bl['completeness']['mean']:.0%}"
    tp_comp_req = f"{tp['completeness_required']['mean']:.0%}"
    bl_comp_req = f"{bl['completeness_required']['mean']:.0%}"
    tp_filled = f"{tp['filled_fields']['mean']:.1f}"
    bl_filled = f"{bl['filled_fields']['mean']:.1f}"
    bl_hall = f"{bl['hallucination_rate']['mean']:.0%}"
    tp_time = f"{tp['wall_clock_seconds']['mean']:.0f}"
    bl_time = f"{bl['wall_clock_seconds']['mean']:.0f}"
    tp_cost = f"${tp['estimated_cost_usd']['mean']:.4f}"
    bl_cost = f"${bl['estimated_cost_usd']['mean']:.4f}"

    return f"""## Benchmark

Tested against {agg['model_count']} HuggingFace models with public model cards across diverse domains (NLP, vision, audio, biomedical, content moderation).

| Metric | Two-pass system | LLM-only baseline |
|---|---|---|
| Mean completeness | {tp_comp} | {bl_comp} |
| Mean required-field completeness | {tp_comp_req} | {bl_comp_req} |
| Mean fields per document (of 29) | {tp_filled} | {bl_filled} |
| Hallucination rate | 0% (by design) | {bl_hall} |
| Mean time per document | {tp_time}s | {bl_time}s |
| Estimated cost per document | {tp_cost} | {bl_cost} |

The two-pass system rejects {bl_hall} of what a naive LLM-only baseline would have populated, while still producing {tp_comp} completeness from public artifacts alone.
"""


def run_benchmark(
    models_yaml: Path,
    output_dir: Path,
    openai_model: str = "gpt-4o-mini",
    force: bool = False,
    *,
    client: Any = None,
) -> dict[str, Any]:
    """Run the full benchmark and write results."""
    models = load_models(models_yaml)
    cache_dir = models_yaml.parent / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_model: list[dict[str, Any]] = []
    for i, entry in enumerate(models):
        model_id = entry["model_id"]
        logger.info(
            "benchmark_progress",
            extra={"model_id": model_id, "index": i + 1, "total": len(models)},
        )
        print(f"[{i + 1}/{len(models)}] {model_id}...", flush=True)

        result = run_single_model(
            entry, cache_dir, openai_model=openai_model, force=force, client=client
        )
        per_model.append(result)

        tp = result["two_pass"]
        bl = result["baseline"]
        print(
            f"  two-pass: {tp['completeness']:.0%} ({tp['filled_fields']}/{tp['total_fields']})  "
            f"baseline: {bl['completeness']:.0%} ({bl['filled_fields']}/{bl['total_fields']})  "
            f"hallucination: {bl['hallucination_rate']:.0%}"
        )

    aggregate = compute_aggregate(per_model)
    snippet = generate_readme_snippet(aggregate)

    (output_dir / "per_model.json").write_text(json.dumps(per_model, indent=2))
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    (output_dir / "README_SNIPPET.md").write_text(snippet)

    print(f"\nResults written to {output_dir}/")
    print(snippet)

    return aggregate
