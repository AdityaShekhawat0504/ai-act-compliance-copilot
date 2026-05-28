"""Single-pass baseline extraction WITHOUT grounding validation.

For each Annex IV item, calls the LLM once with the same per-item prompt
used by Pass 1, but trusts whatever it returns — no Pass 2 grounding check.
Any value the LLM populates is stamped as status='filled' with a synthetic
Provenance (locator='(no grounding check)').

This is the naive-prompting counterfactual against which the two-pass
system's hallucination rejection is measured.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from annex_iv.schema import (
    AnnexIVDocument,
    DocumentedField,
    Provenance,
)
from extraction.pipeline import (
    FieldCandidate,
    _ITEM_ATTRS,
    _candidate_model_for,
    _get_client,
)
from extraction.prompts import ITEM_PROMPTS
from ingestion.base import NormalisedArtifact

logger = logging.getLogger(__name__)

_GENERATOR_VERSION = "0.1.0-baseline"


def generate_annex_iv_baseline(
    artifact: NormalisedArtifact,
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
) -> AnnexIVDocument:
    """Single-pass extraction WITHOUT grounding validation.

    For each item, calls the LLM once with the same per-item prompt
    used by Pass 1, but trusts whatever it returns. Any value populated
    by the LLM is stamped as status='filled' with a synthetic Provenance
    object pointing at the artifact source_uri (no locator, no grounding
    check). This is the 'naive prompting' counterfactual.
    """
    if client is None:
        client = _get_client()

    now = datetime.now(timezone.utc)
    items: dict[str, BaseModel] = {}

    for attr_name, item_cls in _ITEM_ATTRS:
        logger.info(
            "baseline_extracting",
            extra={"source_uri": artifact.source_uri, "item": item_cls.__name__},
        )

        candidate_cls = _candidate_model_for(item_cls)
        prompt = ITEM_PROMPTS[item_cls]
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
        for field_name in item_cls.model_fields:
            candidate: FieldCandidate = getattr(candidates, field_name)
            if candidate.found_in_source and candidate.value:
                provenance = Provenance(
                    source_artifact=artifact.source_uri,
                    locator="(no grounding check)",
                    extracted_at=now,
                    extraction_model=model,
                )
                documented_fields[field_name] = DocumentedField(
                    value=candidate.value,
                    provenance=provenance,
                    status="filled",
                )
            else:
                documented_fields[field_name] = DocumentedField(
                    value=None,
                    status="unavailable",
                )

        items[attr_name] = item_cls(**documented_fields)

    return AnnexIVDocument(
        created_at=now,
        last_updated=now,
        generator_version=_GENERATOR_VERSION,
        target_system_name=artifact.source_uri,
        **items,
    )
