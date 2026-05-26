"""Two-pass extraction orchestrator.

Pass 1: extracts candidate Annex IV field values from normalised artifacts
using structured-output LLM calls via instructor.
Pass 2: validates each candidate against the source artifact and attaches
provenance, or drops it. Explicit refusal to fabricate values not
grounded in the source.

See CLAUDE.md: "LLM-assisted field extraction (src/extraction/)"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, create_model

from annex_iv.schema import (
    AnnexIVDocument,
    AnnexIVItem1General,
    AnnexIVItem2Development,
    AnnexIVItem3MonitoringControl,
    AnnexIVItem4PerformanceMetrics,
    AnnexIVItem5RiskManagement,
    AnnexIVItem6LifecycleChanges,
    AnnexIVItem7HarmonisedStandards,
    AnnexIVItem8DeclarationOfConformity,
    AnnexIVItem9PostMarketMonitoring,
    DocumentedField,
)
from extraction.prompts import ITEM_PROMPTS
from extraction.validators import validate_grounding
from ingestion.base import NormalisedArtifact

logger = logging.getLogger(__name__)

_GENERATOR_VERSION = "0.1.0"

_ITEM_ATTRS: list[tuple[str, type[BaseModel]]] = [
    ("item_1_general", AnnexIVItem1General),
    ("item_2_development", AnnexIVItem2Development),
    ("item_3_monitoring_control", AnnexIVItem3MonitoringControl),
    ("item_4_performance_metrics", AnnexIVItem4PerformanceMetrics),
    ("item_5_risk_management", AnnexIVItem5RiskManagement),
    ("item_6_lifecycle_changes", AnnexIVItem6LifecycleChanges),
    ("item_7_harmonised_standards", AnnexIVItem7HarmonisedStandards),
    ("item_8_declaration_of_conformity", AnnexIVItem8DeclarationOfConformity),
    ("item_9_post_market_monitoring", AnnexIVItem9PostMarketMonitoring),
]


# ---------------------------------------------------------------------------
# Lazy client construction
# ---------------------------------------------------------------------------

_client_singleton: Any = None


def _get_client() -> Any:
    """Lazily construct an instructor-wrapped OpenAI client.

    Lazy so importing this module does not require OPENAI_API_KEY — tests
    that inject their own client never hit this path.
    """
    global _client_singleton
    if _client_singleton is None:
        import instructor
        from openai import OpenAI

        _client_singleton = instructor.from_openai(OpenAI())
    return _client_singleton


# ---------------------------------------------------------------------------
# Pass 1 — candidate extraction
# ---------------------------------------------------------------------------


class FieldCandidate(BaseModel):
    """One field's candidate from the Pass 1 LLM call.

    Kept separate from DocumentedField so the LLM never has to populate a
    Provenance (that's Pass 2's job) and so the schema's
    'filled requires provenance' invariant cannot be violated.
    """

    value: str | None = Field(
        description="The extracted value as it appears (or close to it) in "
        "the source. Null if not in source."
    )
    found_in_source: bool = Field(
        description="True if the value is directly stated in the source; "
        "false otherwise. Set false when you set value to null."
    )


_candidate_model_cache: dict[type[BaseModel], type[BaseModel]] = {}


def _candidate_model_for(item_cls: type[BaseModel]) -> type[BaseModel]:
    """Build (and cache) a Pydantic model where every field on *item_cls* is
    replaced by a FieldCandidate. This is the response_model handed to
    instructor — keeps the LLM's task simple and validation-friendly.
    """
    if item_cls in _candidate_model_cache:
        return _candidate_model_cache[item_cls]

    field_defs: dict[str, Any] = {
        name: (FieldCandidate, ...) for name in item_cls.model_fields
    }
    model = create_model(f"{item_cls.__name__}Candidates", **field_defs)
    _candidate_model_cache[item_cls] = model
    return model


def extract_candidates(
    artifact: NormalisedArtifact,
    target_item: type[BaseModel],
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
) -> BaseModel:
    """Pass 1: extract candidate field values for one Annex IV item.

    Returns an instance of *target_item* populated with DocumentedField
    values:
    - status="pending" + value set when the LLM marked found_in_source=True.
      Pass 2 will promote these to "filled" or drop them to "unavailable".
    - status="unavailable" + value=None when the LLM said the field is not
      in the source.
    """
    if target_item not in ITEM_PROMPTS:
        raise ValueError(
            f"No extraction prompt registered for {target_item.__name__}"
        )

    if client is None:
        client = _get_client()

    prompt = ITEM_PROMPTS[target_item]
    candidate_cls = _candidate_model_for(target_item)

    source_text = artifact.raw_text or ""
    user_message = (
        f"Source artifact: {artifact.source_uri}\n\n"
        f"SOURCE TEXT:\n{source_text}"
    )

    candidates = client.chat.completions.create(
        model=model,
        response_model=candidate_cls,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
    )

    documented_fields: dict[str, DocumentedField] = {}
    for field_name in target_item.model_fields:
        candidate: FieldCandidate = getattr(candidates, field_name)
        if candidate.found_in_source and candidate.value:
            documented_fields[field_name] = DocumentedField(
                value=candidate.value,
                status="pending",
            )
            logger.info(
                "pass1_candidate_found",
                extra={
                    "source_uri": artifact.source_uri,
                    "item": target_item.__name__,
                    "field": field_name,
                },
            )
        else:
            documented_fields[field_name] = DocumentedField(
                value=None,
                status="unavailable",
            )
            logger.info(
                "pass1_unavailable",
                extra={
                    "source_uri": artifact.source_uri,
                    "item": target_item.__name__,
                    "field": field_name,
                },
            )

    return target_item(**documented_fields)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_annex_iv(
    artifact: NormalisedArtifact,
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
) -> AnnexIVDocument:
    """Run the full two-pass extraction over a single artifact.

    For each of the nine Annex IV items, runs Pass 1 (candidate extraction),
    then Pass 2 (grounding validation) on every candidate field. Composes
    the validated items into an AnnexIVDocument.
    """
    if client is None:
        client = _get_client()

    now = datetime.now(timezone.utc)
    validated_items: dict[str, BaseModel] = {}

    for attr_name, item_cls in _ITEM_ATTRS:
        logger.info(
            "starting_item_extraction",
            extra={"source_uri": artifact.source_uri, "item": item_cls.__name__},
        )
        candidate_item = extract_candidates(
            artifact, item_cls, model=model, client=client
        )

        validated_fields: dict[str, DocumentedField] = {}
        for field_name in item_cls.model_fields:
            candidate_field = getattr(candidate_item, field_name)
            validated_fields[field_name] = validate_grounding(
                candidate_field, artifact, model=model, client=client
            )

        validated_items[attr_name] = item_cls(**validated_fields)

    return AnnexIVDocument(
        created_at=now,
        last_updated=now,
        generator_version=_GENERATOR_VERSION,
        target_system_name=artifact.source_uri,
        **validated_items,
    )
