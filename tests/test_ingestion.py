"""Tests for artifact ingestion parsers (src/ingestion/).

Three real HuggingFace models are fetched live on first run, then cached
in tests/fixtures/huggingface/ so subsequent runs are offline-safe.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.base import ArtifactLoader, NormalisedArtifact
from ingestion.huggingface import HuggingFaceModelCardLoader

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "huggingface"

MODELS = {
    "nlp_thorough": "google/gemma-2b",
    "vision": "google/vit-base-patch16-224",
    "minimal": "prajjwal1/bert-tiny",
}


def _cache_path(label: str) -> Path:
    return FIXTURES_DIR / f"{label}.json"


def _fetch_and_cache(label: str, model_id: str) -> NormalisedArtifact:
    path = _cache_path(label)
    if path.exists():
        data = json.loads(path.read_text())
        return NormalisedArtifact.model_validate(data)
    loader = HuggingFaceModelCardLoader()
    artifact = loader.load(model_id)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2))
    return artifact


@pytest.fixture(scope="session")
def nlp_artifact() -> NormalisedArtifact:
    return _fetch_and_cache("nlp_thorough", MODELS["nlp_thorough"])


@pytest.fixture(scope="session")
def vision_artifact() -> NormalisedArtifact:
    return _fetch_and_cache("vision", MODELS["vision"])


@pytest.fixture(scope="session")
def minimal_artifact() -> NormalisedArtifact:
    return _fetch_and_cache("minimal", MODELS["minimal"])


# ---------------------------------------------------------------------------
# Base class contract
# ---------------------------------------------------------------------------


class TestBaseClasses:
    def test_loader_is_abstract(self) -> None:
        assert issubclass(HuggingFaceModelCardLoader, ArtifactLoader)

    def test_normalised_artifact_fields(self) -> None:
        a = NormalisedArtifact(
            artifact_type="test",
            source_uri="test://x",
            structured_content={"a": 1},
            raw_text="hello",
        )
        assert a.artifact_type == "test"
        assert a.raw_text == "hello"


# ---------------------------------------------------------------------------
# NLP model with thorough card (google/gemma-2b)
# ---------------------------------------------------------------------------


class TestNlpThorough:
    def test_artifact_type(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.artifact_type == "huggingface_model_card"

    def test_source_uri(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.source_uri == "huggingface://google/gemma-2b"

    def test_raw_text_non_empty(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.raw_text
        assert len(nlp_artifact.raw_text) > 1000

    def test_has_intended_use(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.structured_content["intended_use"] is not None

    def test_has_training_data(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.structured_content["training_data"] is not None

    def test_has_evaluation(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.structured_content["evaluation"] is not None

    def test_has_description(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.structured_content["description"] is not None

    def test_has_limitations(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.structured_content["limitations"] is not None

    def test_has_license_in_frontmatter(
        self, nlp_artifact: NormalisedArtifact
    ) -> None:
        assert nlp_artifact.structured_content["license"] is not None

    def test_model_id_populated(self, nlp_artifact: NormalisedArtifact) -> None:
        assert nlp_artifact.structured_content["model_id"] == "google/gemma-2b"


# ---------------------------------------------------------------------------
# Vision model (google/vit-base-patch16-224)
# ---------------------------------------------------------------------------


class TestVision:
    def test_source_uri(self, vision_artifact: NormalisedArtifact) -> None:
        assert vision_artifact.source_uri == "huggingface://google/vit-base-patch16-224"

    def test_raw_text_non_empty(self, vision_artifact: NormalisedArtifact) -> None:
        assert vision_artifact.raw_text
        assert len(vision_artifact.raw_text) > 500

    def test_has_training_data(self, vision_artifact: NormalisedArtifact) -> None:
        assert vision_artifact.structured_content["training_data"] is not None

    def test_has_evaluation(self, vision_artifact: NormalisedArtifact) -> None:
        assert vision_artifact.structured_content["evaluation"] is not None

    def test_has_datasets_frontmatter(
        self, vision_artifact: NormalisedArtifact
    ) -> None:
        datasets = vision_artifact.structured_content["datasets"]
        assert datasets is not None
        assert isinstance(datasets, list)

    def test_has_citation(self, vision_artifact: NormalisedArtifact) -> None:
        assert vision_artifact.structured_content["citation"] is not None


# ---------------------------------------------------------------------------
# Minimal card (prajjwal1/bert-tiny)
# ---------------------------------------------------------------------------


class TestMinimal:
    def test_raw_text_non_empty(self, minimal_artifact: NormalisedArtifact) -> None:
        assert minimal_artifact.raw_text
        assert len(minimal_artifact.raw_text) > 0

    def test_missing_sections_are_none(
        self, minimal_artifact: NormalisedArtifact
    ) -> None:
        sc = minimal_artifact.structured_content
        assert sc["intended_use"] is None
        assert sc["training_data"] is None
        assert sc["evaluation"] is None
        assert sc["ethical_considerations"] is None

    def test_frontmatter_still_parsed(
        self, minimal_artifact: NormalisedArtifact
    ) -> None:
        sc = minimal_artifact.structured_content
        assert sc["language"] is not None
        assert sc["tags"] is not None

    def test_no_exception_on_missing_sections(self) -> None:
        loader = HuggingFaceModelCardLoader()
        artifact = _fetch_and_cache("minimal", MODELS["minimal"])
        assert artifact is not None


# ---------------------------------------------------------------------------
# Local file loading
# ---------------------------------------------------------------------------


class TestLocalFile:
    def test_load_from_local_path(self, tmp_path: Path) -> None:
        card_text = """\
---
license: mit
language: en
---

# Test Model

## Model description

A test model for unit testing.

## Training data

Trained on synthetic data.
"""
        card_file = tmp_path / "README.md"
        card_file.write_text(card_text)

        loader = HuggingFaceModelCardLoader()
        artifact = loader.load(str(card_file))

        assert artifact.source_uri.startswith("file://")
        assert artifact.structured_content["license"] == "mit"
        assert artifact.structured_content["description"] is not None
        assert artifact.structured_content["training_data"] is not None
        assert artifact.raw_text is not None
