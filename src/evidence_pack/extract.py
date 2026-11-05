"""Fact extraction from product documents with the existing two-pass pipeline.

Pass 1 asks the LLM for a value and a verbatim supporting quote per fact
(structured output via instructor, same client as the Annex IV pipeline).
Then, before any second LLM call, code checks that the quote appears verbatim
in the document. Pass 2 is the existing ``extraction.validators
.validate_grounding`` check. A fact is ``known`` only if the quote passes the
code check AND Pass 2 marks it grounded. Everything else stays ``unknown``.

The LLM extracts facts. It never classifies: classification is done in code
from data/rules.yaml.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, create_model

from annex_iv.schema import DocumentedField
from evidence_pack.documents import SourceDocument, verify_quote
from evidence_pack.facts import FACT_SPECS, Fact, FactSet, coerce_value
from evidence_pack.rules import catalogue_ids
from extraction.validators import validate_grounding
from ingestion.base import NormalisedArtifact

logger = logging.getLogger(__name__)


class FactCandidate(BaseModel):
    value: str | None = Field(
        description="Answer as stated in the source: 'yes'/'no' for yes-no questions, "
        "a comma-separated list of ids for list questions, short text otherwise. "
        "Null if the source does not say."
    )
    quote: str | None = Field(
        description="An exact, copy-pasted sentence or phrase from the source that "
        "supports the value. Must appear character for character in the source. "
        "Null if value is null."
    )


FactCandidates = create_model(
    "FactCandidates",
    **{spec.name: (FactCandidate, ...) for spec in FACT_SPECS},  # type: ignore[call-overload]
)


def _system_prompt() -> str:
    lines = [
        "You extract facts about a software product's AI feature from one document.",
        "Rules:",
        "- Only answer from the document. If it does not clearly say, return null for value and quote.",
        "- Every non-null value needs a quote copied exactly from the document.",
        "- Do not decide legal classifications. Just report what the document states.",
        "",
        "Questions (field name: question):",
    ]
    for spec in FACT_SPECS:
        q = f"- {spec.name}: {spec.question}"
        if spec.kind == "bool":
            q += " Answer yes or no."
        if spec.choices_from:
            q += " Allowed ids: " + ", ".join(catalogue_ids(spec.choices_from)) + "."
        lines.append(q)
    return "\n".join(lines)


class ExtractionLog(BaseModel):
    document: str
    fact: str
    outcome: str
    """'known', 'no_value', 'quote_not_found', 'bad_value', or 'not_grounded'."""


def extract_facts_from_document(
    doc: SourceDocument,
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
) -> tuple[dict[str, Fact], list[ExtractionLog]]:
    if client is None:
        from extraction.pipeline import _get_client

        client = _get_client()

    candidates = client.chat.completions.create(
        model=model,
        response_model=FactCandidates,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": f"DOCUMENT ({doc.name}):\n{doc.text}"},
        ],
    )
    artifact = NormalisedArtifact(
        artifact_type="product_document",
        source_uri=f"document://{doc.name}",
        structured_content={},
        raw_text=doc.text,
    )

    facts: dict[str, Fact] = {}
    log: list[ExtractionLog] = []
    for spec in FACT_SPECS:
        cand: FactCandidate = getattr(candidates, spec.name)
        if not cand.value:
            log.append(ExtractionLog(document=doc.name, fact=spec.name, outcome="no_value"))
            continue
        try:
            value = coerce_value(spec, cand.value)
            if spec.kind == "list" and spec.choices_from and value is not None:
                allowed = set(catalogue_ids(spec.choices_from))
                if any(v not in allowed for v in value):
                    raise ValueError("value outside catalogue")
        except ValueError:
            log.append(ExtractionLog(document=doc.name, fact=spec.name, outcome="bad_value"))
            continue
        if value is None:
            log.append(ExtractionLog(document=doc.name, fact=spec.name, outcome="no_value"))
            continue
        # Code check first: a quote that is not verbatim in the source is
        # rejected without spending a second LLM call.
        if not verify_quote(cand.quote, doc.text):
            log.append(ExtractionLog(document=doc.name, fact=spec.name, outcome="quote_not_found"))
            continue
        claim = (
            f"Question: {spec.question}\nAnswer: {cand.value}\n"
            f"Supporting quote: \"{cand.quote}\""
        )
        checked = validate_grounding(
            DocumentedField(value=claim, status="pending"), artifact, model=model, client=client
        )
        if checked.status != "filled":
            log.append(ExtractionLog(document=doc.name, fact=spec.name, outcome="not_grounded"))
            continue
        facts[spec.name] = Fact(
            name=spec.name, value=value, status="known", origin="document",
            source=doc.name, quote=cand.quote, quote_verified=True,
        )
        log.append(ExtractionLog(document=doc.name, fact=spec.name, outcome="known"))
    return facts, log


def extract_facts(
    documents: list[SourceDocument],
    model: str = "gpt-4o-mini",
    *,
    client: Any = None,
) -> tuple[FactSet, list[ExtractionLog]]:
    """Extract facts from every document. The first document that grounds a
    fact wins; a later document that disagrees is recorded as a conflict."""
    merged = FactSet()
    full_log: list[ExtractionLog] = []
    for doc in documents:
        facts, log = extract_facts_from_document(doc, model=model, client=client)
        full_log.extend(log)
        for name, fact in facts.items():
            existing = merged.facts.get(name)
            if existing is None:
                merged.facts[name] = fact
            elif existing.value != fact.value:
                merged.notes.append(
                    f"{name}: {existing.source} says {existing.value!r} but "
                    f"{fact.source} says {fact.value!r}; kept the first, please confirm"
                )
    return merged, full_log


def merge(document_facts: FactSet, questionnaire_facts: FactSet) -> FactSet:
    """Questionnaire answers override document facts (they are explicit human
    input); disagreements are recorded as notes for review."""
    out = FactSet(
        facts=dict(document_facts.facts),
        notes=list(document_facts.notes) + list(questionnaire_facts.notes),
    )
    for name, fact in questionnaire_facts.facts.items():
        doc_fact = out.facts.get(name)
        if doc_fact is not None and doc_fact.value != fact.value:
            out.notes.append(
                f"{name}: questionnaire answer {fact.value!r} overrides "
                f"{doc_fact.source} ({doc_fact.value!r}); please confirm"
            )
        out.facts[name] = fact
    return out
