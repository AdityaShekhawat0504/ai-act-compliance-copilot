"""Tests for evidence pack inputs: documents, quote verification, facts,
questionnaire loading, and the two-pass extraction (LLM mocked)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from evidence_pack.documents import SourceDocument, document_from_bytes, load_document, verify_quote
from evidence_pack.extract import FactCandidate, FactCandidates, extract_facts, merge
from evidence_pack.facts import FACT_SPECS, SPECS_BY_NAME, Fact, FactSet, coerce_value
from evidence_pack.questionnaire import QuestionnaireError, facts_from_answers, load_questionnaire, template
from extraction.validators import GroundingCheck

# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


class TestVerifyQuote:
    TEXT = "Our assistant tells every user\nthat they are talking to an AI.  It never   pretends to be human."

    def test_exact_substring(self) -> None:
        assert verify_quote("It never pretends to be human.", self.TEXT)

    def test_whitespace_and_line_breaks_normalised(self) -> None:
        assert verify_quote("tells every user that they are talking to an AI.", self.TEXT)

    def test_curly_quotes_normalised(self) -> None:
        assert verify_quote("the user’s data is kept", "the user's data is kept for 30 days")

    def test_paraphrase_rejected(self) -> None:
        assert not verify_quote("It never claims to be human.", self.TEXT)

    def test_case_must_match(self) -> None:
        assert not verify_quote("it never pretends to be human.", self.TEXT)

    def test_short_quote_rejected(self) -> None:
        assert not verify_quote("an AI", self.TEXT)

    def test_empty_quote_rejected(self) -> None:
        assert not verify_quote(None, self.TEXT)
        assert not verify_quote("", self.TEXT)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class TestDocuments:
    def test_markdown_and_text(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# Title\nHello world", encoding="utf-8")
        (tmp_path / "b.txt").write_text("plain text", encoding="utf-8")
        assert "Hello world" in load_document(tmp_path / "a.md").text
        assert load_document(tmp_path / "b.txt").name == "b.txt"

    def test_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.pdf"
        c = canvas.Canvas(str(path), pagesize=A4)
        c.drawString(72, 750, "Customer data is never used for model training.")
        c.save()
        doc = load_document(path)
        assert verify_quote("Customer data is never used for model training.", doc.text)

    def test_unsupported_type(self) -> None:
        with pytest.raises(ValueError):
            document_from_bytes("slides.pptx", b"x")


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


class TestFacts:
    def test_names_unique(self) -> None:
        assert len({s.name for s in FACT_SPECS}) == len(FACT_SPECS)

    @pytest.mark.parametrize("raw,expected", [(True, True), ("yes", True), ("No", False), ("unknown", None), (None, None)])
    def test_coerce_bool(self, raw: Any, expected: Any) -> None:
        assert coerce_value(SPECS_BY_NAME["interacts_with_people"], raw) == expected

    def test_coerce_bool_rejects_garbage(self) -> None:
        with pytest.raises(ValueError):
            coerce_value(SPECS_BY_NAME["interacts_with_people"], "maybe")

    def test_coerce_list(self) -> None:
        spec = SPECS_BY_NAME["annex_iii_areas"]
        assert coerce_value(spec, "employment, education") == ["employment", "education"]
        assert coerce_value(spec, "none") == []
        assert coerce_value(spec, []) == []

    def test_known_document_fact_requires_verified_quote(self) -> None:
        with pytest.raises(ValueError):
            Fact(name="interacts_with_people", value=True, status="known", origin="document", source="a.md")

    def test_known_fact_requires_value(self) -> None:
        with pytest.raises(ValueError):
            Fact(name="interacts_with_people", status="known", origin="questionnaire")

    def test_missing_fact_is_unknown(self) -> None:
        fs = FactSet()
        assert not fs.get("interacts_with_people").is_known
        assert fs.value("interacts_with_people") is None
        with pytest.raises(KeyError):
            fs.get("not_a_fact")


# ---------------------------------------------------------------------------
# Questionnaire
# ---------------------------------------------------------------------------


DOC = SourceDocument(name="README.md", text="The chat widget answers customer questions with AI.")


class TestQuestionnaire:
    def test_short_and_long_forms(self) -> None:
        fs = facts_from_answers({
            "uses_ai_internally": "no",
            "interacts_with_people": {
                "value": True,
                "evidence": {"file": "README.md", "quote": "The chat widget answers customer questions with AI."},
            },
        }, {"README.md": DOC})
        assert fs.value("uses_ai_internally") is False
        f = fs.get("interacts_with_people")
        assert f.quote_verified and f.source == "README.md"
        assert not fs.notes

    def test_bad_quote_dropped_with_note(self) -> None:
        fs = facts_from_answers({
            "interacts_with_people": {"value": True,
                                      "evidence": {"file": "README.md", "quote": "Our chatbot is fully disclosed."}},
        }, {"README.md": DOC})
        f = fs.get("interacts_with_people")
        assert f.is_known and not f.quote_verified and f.quote is None
        assert "not found verbatim" in fs.notes[0]

    def test_missing_evidence_file_noted(self) -> None:
        fs = facts_from_answers({"interacts_with_people": {"value": True,
                                                           "evidence": {"file": "x.md", "quote": "something long enough"}}})
        assert "was not provided" in fs.notes[0]

    def test_unknown_answers_skipped(self) -> None:
        fs = facts_from_answers({"interacts_with_people": "unknown"})
        assert not fs.get("interacts_with_people").is_known

    def test_unknown_question_rejected(self) -> None:
        with pytest.raises(QuestionnaireError):
            facts_from_answers({"is_awesome": True})

    def test_list_values_validated_against_catalogue(self) -> None:
        with pytest.raises(QuestionnaireError):
            facts_from_answers({"annex_iii_areas": ["astrology"]})

    def test_load_file_with_documents(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text(DOC.text, encoding="utf-8")
        q = tmp_path / "q.yaml"
        q.write_text(yaml.safe_dump({
            "label": "FICTIONAL",
            "documents": ["README.md"],
            "answers": {"interacts_with_people": {"value": True, "evidence": {
                "file": "README.md", "quote": "answers customer questions with AI"}}},
        }), encoding="utf-8")
        facts, docs, raw = load_questionnaire(q)
        assert "README.md" in docs and raw["label"] == "FICTIONAL"
        assert facts.get("interacts_with_people").quote_verified

    def test_template_is_valid_yaml_and_lists_every_question(self) -> None:
        text = template()
        data = yaml.safe_load(text)
        assert data["answers"] is None and data["documents"] == []
        for spec in FACT_SPECS:
            assert f"# {spec.name}:" in text


# ---------------------------------------------------------------------------
# Two-pass extraction with a mocked LLM client
# ---------------------------------------------------------------------------


class FakeClient:
    """Mimics instructor's client.chat.completions.create.

    Pass 1 (response_model=FactCandidates) returns *candidates*; Pass 2
    (response_model=GroundingCheck) returns *grounded*.
    """

    def __init__(self, candidates: dict[str, FactCandidate], grounded: bool = True) -> None:
        self.candidates = candidates
        self.grounded = grounded
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, response_model: Any, messages: list) -> Any:
        if response_model is FactCandidates:
            self.calls.append("pass1")
            empty = FactCandidate(value=None, quote=None)
            return FactCandidates(**{s.name: self.candidates.get(s.name, empty) for s in FACT_SPECS})
        if response_model is GroundingCheck:
            self.calls.append("pass2")
            return GroundingCheck(grounded=self.grounded, locator="README.md", reasoning="test")
        raise AssertionError(f"unexpected response_model {response_model}")


README = SourceDocument(
    name="README.md",
    text="Acme Chat answers customer questions in a chat window. It uses GPT-4o from a third-party provider.",
)


class TestExtraction:
    def test_grounded_fact_with_verbatim_quote_is_known(self) -> None:
        client = FakeClient({"interacts_with_people": FactCandidate(
            value="yes", quote="Acme Chat answers customer questions in a chat window.")})
        facts, log = extract_facts([README], client=client)
        f = facts.get("interacts_with_people")
        assert f.is_known and f.value is True and f.quote_verified and f.origin == "document"
        assert client.calls == ["pass1", "pass2"]
        assert any(e.outcome == "known" for e in log)

    def test_fabricated_quote_rejected_before_pass2(self) -> None:
        client = FakeClient({"interacts_with_people": FactCandidate(
            value="yes", quote="Acme Chat clearly discloses that it is an AI.")})
        facts, log = extract_facts([README], client=client)
        assert not facts.get("interacts_with_people").is_known
        assert client.calls == ["pass1"]
        assert [e.outcome for e in log if e.fact == "interacts_with_people"] == ["quote_not_found"]

    def test_pass2_rejection_leaves_fact_unknown(self) -> None:
        client = FakeClient({"uses_gpai_model": FactCandidate(
            value="yes", quote="It uses GPT-4o from a third-party provider.")}, grounded=False)
        facts, log = extract_facts([README], client=client)
        assert not facts.get("uses_gpai_model").is_known
        assert [e.outcome for e in log if e.fact == "uses_gpai_model"] == ["not_grounded"]

    def test_bad_value_rejected(self) -> None:
        client = FakeClient({"annex_iii_areas": FactCandidate(
            value="chatting", quote="Acme Chat answers customer questions in a chat window.")})
        facts, log = extract_facts([README], client=client)
        assert not facts.get("annex_iii_areas").is_known
        assert [e.outcome for e in log if e.fact == "annex_iii_areas"] == ["bad_value"]

    def test_conflicting_documents_recorded(self) -> None:
        other = SourceDocument(name="faq.md", text="Acme Chat does not talk to customers directly at all.")
        client = _SequencedClient([
            {"interacts_with_people": FactCandidate(
                value="yes", quote="Acme Chat answers customer questions in a chat window.")},
            {"interacts_with_people": FactCandidate(
                value="no", quote="Acme Chat does not talk to customers directly at all.")},
        ])
        merged, _ = extract_facts([README, other], client=client)
        assert merged.value("interacts_with_people") is True
        assert any("please confirm" in n for n in merged.notes)

    def test_questionnaire_overrides_document(self) -> None:
        doc_facts = FactSet(facts={"uses_ai_internally": Fact(
            name="uses_ai_internally", value=True, status="known", origin="document",
            source="README.md", quote="we use it ourselves every day", quote_verified=True)})
        q = facts_from_answers({"uses_ai_internally": False})
        merged = merge(doc_facts, q)
        assert merged.value("uses_ai_internally") is False
        assert any("overrides" in n for n in merged.notes)

    def test_prompt_never_asks_for_classification(self) -> None:
        from evidence_pack.extract import _system_prompt

        prompt = _system_prompt()
        assert "Do not decide legal classifications" in prompt
        assert "risk_tier" not in prompt


class _SequencedClient(FakeClient):
    def __init__(self, per_document: list[dict[str, FactCandidate]]) -> None:
        super().__init__({})
        self._queue = list(per_document)

    def _create(self, *, model: str, response_model: Any, messages: list) -> Any:
        if response_model is FactCandidates:
            self.candidates = self._queue.pop(0)
        return super()._create(model=model, response_model=response_model, messages=messages)
