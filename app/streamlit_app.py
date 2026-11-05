"""Streamlit UI for the EU AI Act evidence pack.

    uv run --with streamlit streamlit run app/streamlit_app.py

All logic lives in the evidence_pack package; this file only collects input
and offers the outputs for download.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evidence_pack.documents import document_from_bytes, load_document  # noqa: E402
from evidence_pack.export import answers_csv, answers_markdown, write_all  # noqa: E402
from evidence_pack.extract import extract_facts, merge  # noqa: E402
from evidence_pack.facts import FACT_SPECS  # noqa: E402
from evidence_pack.questionnaire import QuestionnaireError, facts_from_answers  # noqa: E402
from evidence_pack.report import GPAI_TEXT, ROLE_TEXT, TIER_TEXT, build_evidence_pack  # noqa: E402
from evidence_pack.rules import catalogue_ids  # noqa: E402

DEMO_PATH = ROOT / "examples" / "brightloop" / "questionnaire.yaml"

st.set_page_config(page_title="AI Act evidence pack", layout="wide")
st.title("EU AI Act evidence pack")
st.warning(
    "Not legal advice. Classification is done by fixed rules in code; dates must be re-checked "
    "against the Official Journal of the European Union."
)

use_demo = st.sidebar.checkbox("Load the fictional demo company (Brightloop)", value=False)
mode = "demo" if use_demo else "blank"
demo = yaml.safe_load(DEMO_PATH.read_text(encoding="utf-8")) if use_demo else {"answers": {}, "documents": []}
demo_answers = demo.get("answers") or {}
if use_demo:
    st.sidebar.info(demo.get("label", "Fictional example"))


def _demo_value(name: str):
    raw = demo_answers.get(name)
    if isinstance(raw, dict):
        raw = raw.get("value")
    return raw


tab_docs, tab_form, tab_pack = st.tabs(["1. Documents", "2. Questionnaire", "3. Evidence pack"])

with tab_docs:
    st.write("Upload product documents (README, model card, privacy policy). They are used to verify quoted "
             "evidence and, optionally, to extract facts with the two-pass LLM pipeline.")
    uploads = st.file_uploader("Documents", type=["md", "txt", "pdf"], accept_multiple_files=True)
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    use_llm = st.checkbox("Extract facts from documents with the LLM (needs OPENAI_API_KEY)",
                          value=False, disabled=not has_key)
    if use_demo:
        st.caption("Demo documents loaded: " + ", ".join(demo.get("documents", [])))

with tab_form:
    st.write("Answer what you know. Anything left as 'unknown' shows up as a gap.")
    form_answers: dict = {}
    group = None
    for spec in FACT_SPECS:
        if spec.group != group:
            group = spec.group
            st.subheader(group)
        key = f"q_{mode}_{spec.name}"
        default = _demo_value(spec.name)
        if spec.kind == "bool":
            options = ["unknown", "yes", "no"]
            idx = 0 if default is None or default == "unknown" else (1 if default is True else 2)
            choice = st.selectbox(spec.question, options, index=idx, key=key, help=spec.help or None)
            if choice != "unknown":
                form_answers[spec.name] = choice == "yes"
        elif spec.kind == "list":
            answered = st.checkbox(f"{spec.question} (tick once answered; empty = none)",
                                   value=default is not None, key=key + "_answered")
            picked = st.multiselect("Select all that apply", catalogue_ids(spec.choices_from),
                                    default=default or [], key=key)
            if answered:
                form_answers[spec.name] = picked
        else:
            text = st.text_input(spec.question, value=str(default) if default else "", key=key)
            if text.strip():
                form_answers[spec.name] = text.strip()

with tab_pack:
    as_of = st.date_input("Assess deadlines as of")
    if st.button("Generate evidence pack", type="primary"):
        docs = {}
        if use_demo:
            for rel in demo.get("documents", []):
                d = load_document(DEMO_PATH.parent / rel)
                docs[d.name] = d
        for up in uploads or []:
            try:
                d = document_from_bytes(up.name, up.getvalue())
                docs[d.name] = d
            except ValueError as exc:
                st.error(str(exc))

        # Keep demo evidence quotes for answers the user did not change.
        answers = {}
        for name, value in form_answers.items():
            raw = demo_answers.get(name)
            if isinstance(raw, dict) and raw.get("evidence") and _demo_value(name) == value:
                answers[name] = {"value": value, "evidence": raw["evidence"]}
            else:
                answers[name] = value
        try:
            facts = facts_from_answers(answers, docs)
        except QuestionnaireError as exc:
            st.error(str(exc))
            st.stop()
        if use_llm and docs:
            with st.spinner("Extracting facts (two-pass, quotes verified in code)..."):
                doc_facts, _log = extract_facts(list(docs.values()))
            facts = merge(doc_facts, facts)

        pack = build_evidence_pack(facts, as_of=as_of, label=demo.get("label") if use_demo else None)
        st.session_state["pack"] = pack

    pack = st.session_state.get("pack")
    if pack is not None:
        c = pack.classification
        cols = st.columns(4)
        cols[0].metric("Role", ROLE_TEXT[c.role])
        cols[1].metric("Risk tier", TIER_TEXT[c.risk_tier])
        cols[2].metric("Article 50 triggers", len(c.art50_triggers))
        cols[3].metric("Gaps", len(pack.gaps))
        st.write(f"General-purpose AI: {GPAI_TEXT[c.gpai_use]}. "
                 f"AI literacy (Art. 4): {'applies' if c.ai_literacy_applies else 'does not apply'}.")

        st.subheader("Obligations checklist")
        st.table([{"Article": o.article, "Obligation": o.title, "Party": o.party,
                   "Applies from": o.deadline_date.isoformat(), "Evidence": o.evidence_status}
                  for o in pack.obligations])
        if pack.disclosures:
            st.subheader("Suggested disclosure texts")
            for t in pack.disclosures:
                st.markdown(f"**{t.article}: {t.label}**")
                st.info(t.disclosure)
        st.subheader("Gaps")
        for g in pack.gaps:
            st.write(f"- **{g.item}**: {g.detail}")

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_all(pack, Path(tmp))
            pdf_bytes = paths["pdf"].read_bytes()
        st.download_button("Download PDF report", pdf_bytes, "evidence_pack.pdf", "application/pdf")
        st.download_button("Download answer sheet (Markdown)", answers_markdown(pack),
                           "questionnaire_answers.md", "text/markdown")
        st.download_button("Download answer sheet (CSV)", answers_csv(pack),
                           "questionnaire_answers.csv", "text/csv")
        st.download_button("Download JSON", pack.model_dump_json(indent=2), "evidence_pack.json",
                           "application/json")
