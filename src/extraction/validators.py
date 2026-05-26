"""Grounding validation for extracted field values.

Implements Pass 2 of the extraction pipeline: verifies that each candidate
value is actually grounded in the source artifact. Attaches Provenance
objects to validated fields; drops ungrounded candidates to the gap report.

See CLAUDE.md: "Provenance is non-negotiable" and "No fabrication"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from annex_iv.schema import DocumentedField, Provenance
from extraction.prompts import GROUNDING_VALIDATOR_PROMPT
from ingestion.base import NormalisedArtifact

logger = logging.getLogger(__name__)


class GroundingCheck(BaseModel):
    """Structured response from the Pass 2 grounding validator."""

    grounded: bool
    locator: str | None
    reasoning: str


def validate_grounding(
    field: DocumentedField,
    artifact: NormalisedArtifact,
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
) -> DocumentedField:
    """Pass 2: verify a candidate field value is grounded in the artifact.

    - Fields with status="pending" and a value are checked against the source.
      If grounded, return status="filled" with a fresh Provenance.
      If not grounded, return status="unavailable" and log the dropped value.
    - Fields already marked "unavailable" pass through unchanged.
    - Fields with status="filled" are assumed validated (should not happen in
      normal pipeline flow but handled defensively).
    """
    if field.status == "unavailable":
        return field

    if field.status == "filled":
        return field

    if field.value is None:
        return DocumentedField(value=None, status="unavailable")

    if client is None:
        client = _get_client()

    source_text = artifact.raw_text or ""

    check: GroundingCheck = client.chat.completions.create(
        model=model,
        response_model=GroundingCheck,
        messages=[
            {"role": "system", "content": GROUNDING_VALIDATOR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"EXTRACTED VALUE:\n{field.value}\n\n"
                    f"SOURCE TEXT (from {artifact.source_uri}):\n{source_text}"
                ),
            },
        ],
    )

    if check.grounded:
        provenance = Provenance(
            source_artifact=artifact.source_uri,
            locator=check.locator or "(no specific locator)",
            extracted_at=datetime.now(timezone.utc),
            extraction_model=model,
            confidence=None,
        )
        return DocumentedField(
            value=field.value,
            provenance=provenance,
            status="filled",
        )

    logger.info(
        "pass2_dropped_ungrounded_field",
        extra={
            "source_uri": artifact.source_uri,
            "dropped_value": field.value,
            "reasoning": check.reasoning,
        },
    )
    return DocumentedField(value=None, status="unavailable")


def _get_client() -> Any:
    from extraction.pipeline import _get_client as _shared_get_client

    return _shared_get_client()
