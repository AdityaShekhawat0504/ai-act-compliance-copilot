"""NormalisedArtifact and ArtifactLoader base classes.

NormalisedArtifact is a Pydantic model (not a dataclass) because everything
downstream — schema, extraction, output — uses Pydantic for validation and
serialisation. Keeping the ingestion layer on the same foundation means we
get free JSON round-tripping, type coercion, and consistent `.model_dump()`
across the entire pipeline.

See CLAUDE.md: "Artifact ingestion (src/ingestion/)"
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class NormalisedArtifact(BaseModel):
    """Internal representation produced by all artifact parsers.

    Consumed by the extraction layer to populate Annex IV fields.
    """

    artifact_type: str
    """e.g. 'huggingface_model_card', 'mlflow_run', 'eval_report'."""

    source_uri: str
    """Canonical identifier, e.g. 'huggingface://google/gemma-2b'."""

    structured_content: dict
    """Parsed sections as a dict. Keys vary by artifact type."""

    raw_text: str | None = None
    """Original text for downstream grounding lookups."""


class ArtifactLoader(ABC):
    """Base class for artifact parsers.

    Each concrete loader fetches or reads one artifact type and returns
    a NormalisedArtifact.
    """

    @abstractmethod
    def load(self, source: str) -> NormalisedArtifact:
        """Load an artifact from *source* and return a normalised representation.

        *source* is loader-specific: a HuggingFace model ID, an MLflow run URI,
        a file path, etc.
        """
