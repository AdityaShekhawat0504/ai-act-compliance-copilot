"""Tests for evidence pack outputs, the CLI, the demo and the eval script."""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
from pathlib import Path

import pytest
import yaml
from pypdf import PdfReader

from evidence_pack import cli
from evidence_pack.export import CSV_COLUMNS, answers_csv, answers_markdown, write_all
from evidence_pack.pipeline import run
from evidence_pack.questionnaire import facts_from_answers
from evidence_pack.report import build_evidence_pack
from tests.test_ep_inputs import FakeClient
from tests.test_ep_rules import BASE

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "brightloop" / "questionnaire.yaml"
AS_OF = dt.date(2026, 11, 3)
NOW = dt.datetime(2026, 11, 3, 12, 0, tzinfo=dt.timezone.utc)


def _pdf_text(path: Path) -> str:
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)


@pytest.fixture(scope="module")
def demo_pack():
    pack, _ = run(DEMO, as_of=AS_OF, now=NOW)
    return pack


class TestDemoCompany:
    def test_classification(self, demo_pack) -> None:
        c = demo_pack.classification
        assert c.role == "both"
        assert c.risk_tier == "high_risk"
        assert c.trigger_ids == ["chatbot", "synthetic_content"]
        assert c.gpai_use == "integrates_third_party"
        assert c.ai_literacy_applies

    def test_labelled_fictional(self, demo_pack) -> None:
        assert "FICTIONAL" in demo_pack.label

    def test_all_demo_quotes_verified(self, demo_pack) -> None:
        assert demo_pack.gaps and not [g for g in demo_pack.gaps if g.kind == "evidence_note"]
        quoted = [f for f in demo_pack.facts if f.quote]
        assert quoted and all(f.quote_verified for f in quoted)

    def test_annex_iv_draft_for_high_risk_provider(self, demo_pack) -> None:
        assert demo_pack.annex_iv is not None
        f = demo_pack.annex_iv.item_1_general.provider_name
        assert f.status == "filled" and f.provenance.source_artifact == "questionnaire"
        q = demo_pack.annex_iv.item_3_monitoring_control.capabilities_limitations
        assert q.provenance.source_artifact == "model_card.md" and q.provenance.locator.startswith("quote:")
        assert demo_pack.annex_iv_gaps.gaps  # items 5-9 have nothing to fill them

    def test_gaps_include_unanswered_evidence(self, demo_pack) -> None:
        items = " ".join(g.item for g in demo_pack.gaps)
        assert "Art. 9" in items and "Art. 17" in items and "Art. 4" in items

    def test_obligations_sorted_by_deadline(self, demo_pack) -> None:
        dates = [o.deadline_date for o in demo_pack.obligations]
        assert dates == sorted(dates)

    def test_notice_present(self, demo_pack) -> None:
        assert demo_pack.notice.startswith("Not legal advice")
        assert "Official Journal" in demo_pack.notice


class TestExports:
    def test_write_all(self, demo_pack, tmp_path: Path) -> None:
        paths = write_all(demo_pack, tmp_path)
        for key in ["pdf", "markdown", "csv", "json", "annex_iv_pdf", "annex_iv_json"]:
            assert paths[key].exists() and paths[key].stat().st_size > 0
        text = _pdf_text(paths["pdf"])
        for needle in ["EU AI Act Evidence Pack", "Not legal advice", "FICTIONAL", "Obligations checklist",
                       "Suggested disclosure texts", "Gap list", "Customer questionnaire answers",
                       "Annex IV technical documentation draft", "Appendix A"]:
            assert needle in text, needle
        assert "Annex IV" in _pdf_text(paths["annex_iv_pdf"])
        data = json.loads(paths["json"].read_text())
        assert data["classification"]["risk_tier"] == "high_risk"

    def test_csv(self, demo_pack) -> None:
        rows = list(csv.DictReader(io.StringIO(answers_csv(demo_pack))))
        assert list(rows[0]) == CSV_COLUMNS
        assert len(rows) == len(demo_pack.customer_answers) == 12
        assert {r["status"] for r in rows} <= {"answered", "needs_input"}

    def test_markdown(self, demo_pack) -> None:
        md = answers_markdown(demo_pack)
        assert md.startswith("> **FICTIONAL")
        assert "| Q01 |" in md and "## Gaps" in md and "Not legal advice" in md

    def test_minimal_system_has_no_annex_iv(self, tmp_path: Path) -> None:
        pack = build_evidence_pack(facts_from_answers(BASE), as_of=AS_OF, now=NOW)
        assert pack.classification.risk_tier == "minimal_risk"
        assert pack.annex_iv is None and not pack.disclosures
        paths = write_all(pack, tmp_path)
        assert "annex_iv_pdf" not in paths
        assert "No Article 50 trigger identified" in _pdf_text(paths["pdf"])

    def test_empty_questionnaire_still_renders(self, tmp_path: Path) -> None:
        pack = build_evidence_pack(facts_from_answers({}), as_of=AS_OF, now=NOW)
        assert pack.classification.role == "none"
        assert any(g.kind == "classification_input" for g in pack.gaps)
        assert all(a.status == "needs_input" for a in pack.customer_answers if a.id in {"Q01", "Q02"})
        write_all(pack, tmp_path)


class TestCli:
    def test_demo(self, tmp_path: Path, capsys) -> None:
        assert cli.main(["demo", "-o", str(tmp_path), "--as-of", "2026-11-03"]) == 0
        out = capsys.readouterr().out
        assert "High-risk" in out and "Not legal advice" in out
        assert (tmp_path / "evidence_pack.pdf").exists()

    def test_build(self, tmp_path: Path) -> None:
        assert cli.main(["build", "-q", str(DEMO), "-o", str(tmp_path)]) == 0
        assert (tmp_path / "questionnaire_answers.csv").exists()

    def test_template(self, capsys) -> None:
        assert cli.main(["template"]) == 0
        assert "answers:" in capsys.readouterr().out

    def test_no_input_is_an_error(self, tmp_path: Path) -> None:
        assert cli.main(["build", "-o", str(tmp_path)]) == 1

    def test_llm_without_key_is_an_error(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert cli.main(["build", "-d", str(DEMO.parent / "README.md"), "--llm", "-o", str(tmp_path)]) == 1

    def test_bad_questionnaire_is_an_error(self, tmp_path: Path) -> None:
        q = tmp_path / "q.yaml"
        q.write_text("answers:\n  not_a_question: yes\n", encoding="utf-8")
        assert cli.main(["build", "-q", str(q), "-o", str(tmp_path / "out")]) == 1


class TestLlmPath:
    def test_documents_only_with_mocked_llm(self) -> None:
        from evidence_pack.extract import FactCandidate

        client = FakeClient({
            "interacts_with_people": FactCandidate(
                value="yes",
                quote="A chat assistant on the careers page answers applicants' questions about open roles"),
            "annex_iii_areas": FactCandidate(
                value="employment",
                quote="The system ranks incoming applications against the job requirements"),
        })
        pack, log = run(None, [DEMO.parent / "README.md"], use_llm=True, client=client, as_of=AS_OF, now=NOW)
        assert pack.classification.risk_tier == "high_risk"
        assert "chatbot" in pack.classification.trigger_ids
        assert sum(e.outcome == "known" for e in log) == 2
        # Role was not stated in the document, so it is an open gap.
        assert "places_on_market_under_own_name" in pack.classification.unknown_inputs


class TestEval:
    def test_eval_runs_and_reports_every_field(self, capsys) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        acc, _mismatches, n = mod.evaluate(ROOT / "eval" / "systems")
        assert n == 15
        assert set(acc) == {"role", "risk_tier", "art50_triggers", "gpai_use", "ai_literacy"}
        assert mod.main([]) == 0
        assert "accuracy" in capsys.readouterr().out

    def test_eval_systems_are_labelled_fictional(self) -> None:
        for path in sorted((ROOT / "eval" / "systems").glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            assert text.startswith("# FICTIONAL SYSTEM"), path.name
            assert yaml.safe_load(text)["name"].startswith("Fictional:"), path.name
