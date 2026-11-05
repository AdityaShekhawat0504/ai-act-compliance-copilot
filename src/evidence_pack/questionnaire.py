"""Structured questionnaire input: the offline path into the evidence pack.

Format (YAML or JSON)::

    documents: [README.md, privacy_policy.md]   # optional, relative to the file
    answers:
      company_name: Example GmbH
      interacts_with_people: yes                # short form
      generates_synthetic_content:              # long form with evidence
        value: yes
        evidence: {file: README.md, quote: "drafts replies with AI"}

Unanswered questions stay unknown. An evidence quote must appear verbatim in
the named document; otherwise the quote is dropped and a note is recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from evidence_pack.documents import SourceDocument, load_document, verify_quote
from evidence_pack.facts import FACT_SPECS, SPECS_BY_NAME, Fact, FactSet, coerce_value
from evidence_pack.rules import catalogue_ids


class QuestionnaireError(ValueError):
    pass


def _validate_list(name: str, value: list[str], catalogue: str) -> None:
    allowed = set(catalogue_ids(catalogue))
    bad = [v for v in value if v not in allowed]
    if bad:
        raise QuestionnaireError(
            f"{name}: unknown value(s) {bad}; allowed: {sorted(allowed)}"
        )


def facts_from_answers(
    answers: Mapping[str, Any],
    documents: Mapping[str, SourceDocument] | None = None,
) -> FactSet:
    """Turn questionnaire answers into a FactSet, verifying evidence quotes."""
    documents = documents or {}
    facts = FactSet()
    for name, raw in answers.items():
        if name not in SPECS_BY_NAME:
            raise QuestionnaireError(f"unknown question {name!r}")
        spec = SPECS_BY_NAME[name]
        evidence = None
        if isinstance(raw, Mapping):
            evidence = raw.get("evidence")
            raw = raw.get("value")
        try:
            value = coerce_value(spec, raw)
        except ValueError as exc:
            raise QuestionnaireError(str(exc)) from exc
        if value is None:
            continue
        if spec.kind == "list" and spec.choices_from:
            _validate_list(name, value, spec.choices_from)

        fact = Fact(name=name, value=value, status="known", origin="questionnaire", source="questionnaire")
        if evidence:
            file_name = str(evidence.get("file", ""))
            quote = evidence.get("quote")
            doc = documents.get(file_name)
            if doc is None:
                facts.notes.append(
                    f"{name}: evidence file {file_name!r} was not provided, quote ignored"
                )
            elif verify_quote(quote, doc.text):
                fact = Fact(
                    name=name, value=value, status="known", origin="questionnaire",
                    source=file_name, quote=quote, quote_verified=True,
                )
            else:
                facts.notes.append(
                    f"{name}: quoted evidence not found verbatim in {file_name}, quote ignored"
                )
        facts.facts[name] = fact
    return facts


def load_questionnaire(path: Path, extra_documents: list[Path] | None = None) -> tuple[FactSet, dict[str, SourceDocument], dict[str, Any]]:
    """Load a questionnaire file and the documents it lists.

    Returns (facts, documents by file name, raw file content).
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping) or not isinstance(data.get("answers", {}), Mapping):
        raise QuestionnaireError(f"{path}: expected a mapping with an 'answers' section")
    docs: dict[str, SourceDocument] = {}
    for rel in data.get("documents", []) or []:
        doc = load_document(path.parent / rel)
        docs[doc.name] = doc
    for extra in extra_documents or []:
        doc = load_document(extra)
        docs[doc.name] = doc
    facts = facts_from_answers(data.get("answers", {}) or {}, docs)
    return facts, docs, dict(data)


def template() -> str:
    """A blank questionnaire with every question as a comment."""
    lines = [
        "# EU AI Act evidence pack questionnaire.",
        "# Leave a line out (or answer 'unknown') if you do not know; the pack lists it as a gap.",
        "# Optional evidence: {value: yes, evidence: {file: README.md, quote: \"exact text\"}}",
        "documents: []",
        "answers:",
    ]
    group = None
    for spec in FACT_SPECS:
        if spec.group != group:
            group = spec.group
            lines.append(f"  # --- {group} ---")
        hint = {"bool": "yes | no", "text": "\"...\"", "list": "[]"}[spec.kind]
        if spec.choices_from:
            hint += "  # choose from: " + ", ".join(catalogue_ids(spec.choices_from))
        lines.append(f"  # {spec.question}")
        lines.append(f"  # {spec.name}: {hint}")
    return "\n".join(lines) + "\n"
