"""Tests for the benchmark system (baseline, orchestrator, aggregation, CLI).

All tests mock LLM calls — no real API tokens spent. The actual 10-model
benchmark is too expensive (~$0.20, ~30 min) for CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from annex_iv.schema import (
    AnnexIVDocument,
    AnnexIVItem1General,
    DocumentedField,
)
from benchmark.baseline import generate_annex_iv_baseline
from benchmark.run_benchmark import compute_aggregate, generate_readme_snippet
from cli import app
from extraction.pipeline import FieldCandidate
from extraction.validators import GroundingCheck
from ingestion.base import NormalisedArtifact

runner = CliRunner()


def _build_artifact(raw_text: str = "the system is a credit-scoring model") -> NormalisedArtifact:
    return NormalisedArtifact(
        artifact_type="huggingface_model_card",
        source_uri="huggingface://test/fake-model",
        structured_content={"description": "a test"},
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Fake instructor client (same pattern as test_extraction.py)
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, parent: "_FakeInstructorClient") -> None:
        self._parent = parent

    def create(
        self,
        *,
        response_model: type[Any],
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self._parent.calls.append(
            {"response_model": response_model, "messages": messages, "model": model}
        )
        return self._parent.dispatch(response_model, messages, model)


class _FakeChat:
    def __init__(self, parent: "_FakeInstructorClient") -> None:
        self.completions = _FakeCompletions(parent)


class _FakeInstructorClient:
    def __init__(self, dispatcher) -> None:
        self.dispatch = dispatcher
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


def _baseline_dispatcher(
    response_model: type[Any], messages: list[dict[str, str]], model: str | None
) -> Any:
    """Dispatcher that marks the first field of each item as found."""
    if response_model is GroundingCheck:
        return GroundingCheck(grounded=True, locator="line 1", reasoning="ok")
    kwargs: dict[str, FieldCandidate] = {}
    for i, name in enumerate(response_model.model_fields):
        if i == 0:
            kwargs[name] = FieldCandidate(value=f"value-for-{name}", found_in_source=True)
        else:
            kwargs[name] = FieldCandidate(value=None, found_in_source=False)
    return response_model(**kwargs)


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_returns_valid_document(self) -> None:
        artifact = _build_artifact()
        client = _FakeInstructorClient(_baseline_dispatcher)

        doc = generate_annex_iv_baseline(artifact, client=client)

        assert isinstance(doc, AnnexIVDocument)
        assert doc.target_system_name == "huggingface://test/fake-model"
        assert doc.generator_version == "0.1.0-baseline"

    def test_filled_fields_have_provenance(self) -> None:
        artifact = _build_artifact()
        client = _FakeInstructorClient(_baseline_dispatcher)

        doc = generate_annex_iv_baseline(artifact, client=client)

        assert doc.item_1_general.intended_purpose.status == "filled"
        prov = doc.item_1_general.intended_purpose.provenance
        assert prov is not None
        assert prov.source_artifact == "huggingface://test/fake-model"
        assert prov.locator == "(no grounding check)"

    def test_unavailable_fields_have_no_provenance(self) -> None:
        artifact = _build_artifact()
        client = _FakeInstructorClient(_baseline_dispatcher)

        doc = generate_annex_iv_baseline(artifact, client=client)

        assert doc.item_1_general.provider_name.status == "unavailable"
        assert doc.item_1_general.provider_name.provenance is None

    def test_only_pass1_calls_no_grounding(self) -> None:
        artifact = _build_artifact()
        client = _FakeInstructorClient(_baseline_dispatcher)

        generate_annex_iv_baseline(artifact, client=client)

        for call in client.calls:
            assert call["response_model"] is not GroundingCheck, (
                "Baseline should not call grounding validation"
            )

    def test_baseline_uses_9_calls(self) -> None:
        artifact = _build_artifact()
        client = _FakeInstructorClient(_baseline_dispatcher)

        generate_annex_iv_baseline(artifact, client=client)

        assert len(client.calls) == 9


# ---------------------------------------------------------------------------
# Aggregate computation tests
# ---------------------------------------------------------------------------


def _make_per_model_result(
    model_id: str = "test/model",
    tp_comp: float = 0.3,
    tp_comp_req: float = 0.35,
    tp_filled: int = 9,
    tp_time: float = 60.0,
    bl_comp: float = 0.5,
    bl_comp_req: float = 0.55,
    bl_filled: int = 15,
    bl_time: float = 30.0,
    bl_hall_rate: float = 0.2,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "domain": "test",
        "two_pass": {
            "completeness": tp_comp,
            "completeness_required": tp_comp_req,
            "filled_fields": tp_filled,
            "total_fields": 29,
            "gap_count": 29 - tp_filled,
            "wall_clock_seconds": tp_time,
            "estimated_cost_usd": 0.0013,
        },
        "baseline": {
            "completeness": bl_comp,
            "completeness_required": bl_comp_req,
            "filled_fields": bl_filled,
            "total_fields": 29,
            "gap_count": 29 - bl_filled,
            "wall_clock_seconds": bl_time,
            "estimated_cost_usd": 0.0006,
            "hallucinated_fields": int(bl_filled * bl_hall_rate),
            "hallucination_rate": bl_hall_rate,
            "hallucination_check_seconds": 20.0,
        },
    }


class TestAggregateComputation:
    def test_mean_completeness(self) -> None:
        results = [
            _make_per_model_result(tp_comp=0.3, bl_comp=0.5),
            _make_per_model_result(tp_comp=0.5, bl_comp=0.7),
        ]
        agg = compute_aggregate(results)

        assert agg["two_pass"]["completeness"]["mean"] == pytest.approx(0.4, abs=0.01)
        assert agg["baseline"]["completeness"]["mean"] == pytest.approx(0.6, abs=0.01)

    def test_median_completeness(self) -> None:
        results = [
            _make_per_model_result(tp_comp=0.2),
            _make_per_model_result(tp_comp=0.3),
            _make_per_model_result(tp_comp=0.8),
        ]
        agg = compute_aggregate(results)

        assert agg["two_pass"]["completeness"]["median"] == pytest.approx(0.3, abs=0.01)

    def test_model_count(self) -> None:
        results = [_make_per_model_result() for _ in range(5)]
        agg = compute_aggregate(results)

        assert agg["model_count"] == 5

    def test_hallucination_rate_mean(self) -> None:
        results = [
            _make_per_model_result(bl_hall_rate=0.1),
            _make_per_model_result(bl_hall_rate=0.3),
        ]
        agg = compute_aggregate(results)

        assert agg["baseline"]["hallucination_rate"]["mean"] == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------
# README snippet generation
# ---------------------------------------------------------------------------


class TestReadmeSnippet:
    def test_contains_table(self) -> None:
        results = [_make_per_model_result() for _ in range(3)]
        agg = compute_aggregate(results)
        snippet = generate_readme_snippet(agg)

        assert "## Benchmark" in snippet
        assert "| Metric |" in snippet
        assert "Mean completeness" in snippet
        assert "Hallucination rate" in snippet
        assert "0% (by design)" in snippet

    def test_model_count_appears(self) -> None:
        results = [_make_per_model_result() for _ in range(10)]
        agg = compute_aggregate(results)
        snippet = generate_readme_snippet(agg)

        assert "10 HuggingFace models" in snippet


# ---------------------------------------------------------------------------
# CLI benchmark --help
# ---------------------------------------------------------------------------


class TestBenchmarkCli:
    def test_help_exits_0(self) -> None:
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output.lower()

    def test_help_lists_in_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.output
