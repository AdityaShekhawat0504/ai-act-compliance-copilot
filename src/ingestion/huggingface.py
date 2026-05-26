"""HuggingFace model card parser.

Fetches and parses model cards from the HuggingFace Hub into NormalisedArtifact
instances. This is the first and primary ingestion source — the benchmark runs
against 10 public HuggingFace models.

See CLAUDE.md: "Start with huggingface.py (model cards)"
"""

from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import ModelCard

from ingestion.base import ArtifactLoader, NormalisedArtifact

_SECTION_ALIASES: dict[str, list[str]] = {
    "intended_use": [
        "intended use",
        "intended uses",
        "intended uses & limitations",
        "intended usage",
        "usage",
        "uses",
    ],
    "limitations": [
        "limitations",
        "limitations and bias",
        "limitations and risks",
        "known limitations",
        "usage and limitations",
    ],
    "training_data": [
        "training data",
        "training dataset",
        "training details",
        "training set",
        "model data",
        "data",
        "dataset",
    ],
    "training_procedure": [
        "training procedure",
        "training",
        "training hyperparameters",
        "training process",
        "pretraining",
        "pre-training",
        "implementation information",
    ],
    "evaluation": [
        "evaluation",
        "evaluation results",
        "benchmark results",
        "results",
        "performance",
    ],
    "ethical_considerations": [
        "ethical considerations",
        "ethical considerations and risks",
        "ethics",
        "ethics and safety",
        "bias",
        "bias, risks, and limitations",
    ],
    "environmental_impact": [
        "environmental impact",
        "carbon footprint",
        "compute",
    ],
    "citation": [
        "citation",
        "bibtex entry and citation info",
        "bibtex",
        "how to cite",
        "citing",
    ],
    "description": [
        "description",
        "model description",
        "overview",
        "summary",
        "about",
    ],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _SECTION_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias] = canonical


def _normalise_heading(heading: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", heading.lower()).strip()


def _parse_markdown_sections(text: str) -> dict[str, str]:
    """Split markdown into sections keyed by normalised heading."""
    pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        heading = _normalise_heading(match.group(2))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        canonical = _ALIAS_TO_CANONICAL.get(heading)
        if canonical and canonical not in sections:
            sections[canonical] = body
        elif heading not in sections:
            sections[heading] = body

    return sections


def _extract_frontmatter(card: ModelCard) -> dict:
    """Extract YAML frontmatter fields from a ModelCard."""
    if card.data is None:
        return {}
    raw = card.data.to_dict()
    return {k: v for k, v in raw.items() if v is not None}


class HuggingFaceModelCardLoader(ArtifactLoader):
    """Loads HuggingFace model cards into NormalisedArtifact instances.

    Accepts either a HF model ID (e.g. ``"google/gemma-2b"``) or a local
    file path to a model card markdown file.
    """

    def load(self, source: str) -> NormalisedArtifact:
        """Load a model card from HuggingFace Hub or a local file.

        Missing sections produce ``None`` values — no exceptions for
        incomplete cards.
        """
        local_path = Path(source)
        if local_path.exists():
            card = ModelCard.load(local_path)
            source_uri = f"file://{local_path.resolve()}"
        else:
            card = ModelCard.load(source)
            source_uri = f"huggingface://{source}"

        raw_text = card.text
        frontmatter = _extract_frontmatter(card)
        sections = _parse_markdown_sections(raw_text)

        structured: dict = {}

        structured["model_id"] = frontmatter.get("model_id") or (
            source if not local_path.exists() else None
        )

        for key in (
            "license",
            "language",
            "library_name",
            "pipeline_tag",
            "tags",
            "datasets",
            "metrics",
            "model-index",
        ):
            structured[key] = frontmatter.get(key)

        for key in (
            "intended_use",
            "limitations",
            "training_data",
            "training_procedure",
            "evaluation",
            "ethical_considerations",
            "environmental_impact",
            "citation",
            "description",
        ):
            structured[key] = sections.get(key)

        return NormalisedArtifact(
            artifact_type="huggingface_model_card",
            source_uri=source_uri,
            structured_content=structured,
            raw_text=raw_text,
        )
