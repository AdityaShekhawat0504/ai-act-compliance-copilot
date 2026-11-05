"""End-to-end entry point: inputs -> facts -> classification -> evidence pack."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from evidence_pack.documents import SourceDocument, load_document
from evidence_pack.extract import ExtractionLog, extract_facts, merge
from evidence_pack.facts import FactSet
from evidence_pack.questionnaire import load_questionnaire
from evidence_pack.report import EvidencePack, build_evidence_pack


def run(
    questionnaire: Path | None = None,
    documents: list[Path] | None = None,
    *,
    use_llm: bool = False,
    model: str = "gpt-4o-mini",
    client: Any = None,
    as_of: dt.date | None = None,
    now: dt.datetime | None = None,
) -> tuple[EvidencePack, list[ExtractionLog]]:
    """Build an evidence pack.

    - questionnaire only: fully offline and deterministic.
    - documents with ``use_llm=True``: facts are extracted with the two-pass
      pipeline; questionnaire answers (if any) override them.
    - documents without ``use_llm``: used only to verify questionnaire quotes.
    """
    docs: dict[str, SourceDocument] = {}
    q_facts = FactSet()
    label = None
    if questionnaire is not None:
        q_facts, docs, raw = load_questionnaire(questionnaire, documents)
        label = raw.get("label")
    else:
        for p in documents or []:
            d = load_document(p)
            docs[d.name] = d

    log: list[ExtractionLog] = []
    facts = q_facts
    if use_llm and docs:
        doc_facts, log = extract_facts(list(docs.values()), model=model, client=client)
        facts = merge(doc_facts, q_facts)

    return build_evidence_pack(facts, as_of=as_of, now=now, label=label), log
