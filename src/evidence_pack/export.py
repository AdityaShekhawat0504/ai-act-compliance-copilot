"""Export an EvidencePack as PDF report, Markdown and CSV answer sheets, JSON."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from evidence_pack.report import GPAI_TEXT, ROLE_TEXT, TIER_TEXT, EvidencePack

CSV_COLUMNS = ["id", "question", "answer", "basis", "evidence", "status"]

_EVIDENCE_TEXT = {
    "provided": "evidence provided",
    "missing": "evidence missing",
    "not_tracked": "document to prepare",
}


# ---------------------------------------------------------------------------
# Customer questionnaire answer sheet
# ---------------------------------------------------------------------------


def answers_csv(pack: EvidencePack) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for a in pack.customer_answers:
        writer.writerow(a.model_dump(include=set(CSV_COLUMNS)))
    return buf.getvalue()


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def answers_markdown(pack: EvidencePack) -> str:
    c = pack.classification
    lines = []
    if pack.label:
        lines += [f"> **{pack.label}**", ""]
    lines += [
        f"# EU AI Act questionnaire answers: {pack.system_name}",
        "",
        f"Company: {pack.company_name}  ",
        f"Generated: {pack.generated_at:%Y-%m-%d} (dates assessed as of {pack.as_of.isoformat()})  ",
        f"Rules version {c.rules_version}, dates version {c.deadlines_version}",
        "",
        f"> {pack.notice}",
        "",
        "## Summary",
        "",
        f"- Role: {ROLE_TEXT[c.role]} ({c.role_article})",
        f"- Risk tier: {TIER_TEXT[c.risk_tier]} ({c.risk_article})",
        f"- Article 50 triggers: {', '.join(t.article for t in c.art50_triggers) or 'none'}",
        f"- General-purpose AI: {GPAI_TEXT[c.gpai_use]}",
        f"- AI literacy (Art. 4) applies: {'yes' if c.ai_literacy_applies else 'no'}",
        "",
        "## Answers",
        "",
        "| # | Question | Answer | Basis | Evidence | Status |",
        "|---|---|---|---|---|---|",
    ]
    for a in pack.customer_answers:
        lines.append(
            "| " + " | ".join(_md_cell(x) for x in [a.id, a.question, a.answer, a.basis, a.evidence, a.status]) + " |"
        )
    lines += ["", "## Obligations and deadlines", "",
              "| Article | Obligation | Party | Applies from | Evidence |", "|---|---|---|---|---|"]
    for o in pack.obligations:
        lines.append(
            "| " + " | ".join(_md_cell(x) for x in [
                o.article, o.title, o.party.replace("_", " "),
                o.deadline_date.isoformat() + (" (in force)" if o.applies_now else ""),
                _EVIDENCE_TEXT[o.evidence_status],
            ]) + " |"
        )
    if pack.disclosures:
        lines += ["", "## Suggested disclosure texts", ""]
        for t in pack.disclosures:
            lines += [f"**{t.article}: {t.label}**", "", f"> {t.disclosure}", ""]
    lines += ["", "## Gaps", ""]
    if pack.gaps:
        lines += [f"- [{g.kind.replace('_', ' ')}] {g.item}: {g.detail}" for g in pack.gaps]
    else:
        lines.append("No gaps identified.")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=20, leading=25),
        "h1": ParagraphStyle("h1", parent=base["Heading2"], fontSize=14, spaceBefore=6, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading4"], fontSize=10.5, spaceBefore=6),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5, leading=13),
        "cell": ParagraphStyle("c", parent=base["BodyText"], fontSize=8, leading=10),
        "quote": ParagraphStyle("q", parent=base["BodyText"], fontName="Helvetica-Oblique",
                                fontSize=9.5, leading=13, leftIndent=10),
        "banner": ParagraphStyle("ban", parent=base["BodyText"], fontName="Helvetica-Bold",
                                 fontSize=10, textColor=colors.HexColor("#9A3412")),
        "notice": ParagraphStyle("n", parent=base["BodyText"], fontSize=9, leading=12,
                                 backColor=colors.HexColor("#FEF3C7"), borderPadding=6),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text)), style)


def _table(header: list[str], rows: list[list[str]], widths: list[float], s: dict) -> Table:
    data = [[_p(h, s["cell"]) for h in header]] + [[_p(c, s["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9CA3AF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def write_pdf(pack: EvidencePack, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = _styles()
    c = pack.classification
    story: list = []

    if pack.label:
        story += [_p(pack.label, s["banner"]), Spacer(1, 3 * mm)]
    story += [
        _p("EU AI Act Evidence Pack", s["title"]),
        _p(f"{pack.system_name} by {pack.company_name}", s["h1"]),
        _p(f"Generated {pack.generated_at:%Y-%m-%d}; deadlines assessed as of {pack.as_of.isoformat()}. "
           f"Rules v{c.rules_version}, dates v{c.deadlines_version}, generator v{pack.generator_version}.",
           s["body"]),
        Spacer(1, 4 * mm),
        _p(pack.notice, s["notice"]),
        Spacer(1, 6 * mm),
        _p("1. Classification", s["h1"]),
    ]
    rows = [
        ["Role", ROLE_TEXT[c.role], c.role_article],
        ["Risk tier", TIER_TEXT[c.risk_tier], c.risk_article],
        ["High-risk basis", "; ".join(c.high_risk_basis) or "-", "Art. 6"],
        ["Prohibited practices", "; ".join(c.prohibited_practices) or "none reported", "Art. 5"],
        ["Article 50 triggers", "; ".join(f"{t.article} {t.label}" for t in c.art50_triggers) or "none", "Art. 50"],
        ["General-purpose AI", GPAI_TEXT[c.gpai_use], c.gpai_article],
        ["AI literacy", "applies" if c.ai_literacy_applies else "does not apply", "Art. 4"],
    ]
    story.append(_table(["Decision", "Result", "Article"], rows, [35 * mm, 105 * mm, 30 * mm], s))
    story.append(Spacer(1, 2 * mm))
    story.append(_p("Every result above comes from fixed, versioned rules applied in code to the facts in "
                    "Appendix A. Unknown facts are treated conservatively and listed as gaps.", s["cell"]))

    story += [_p("2. Obligations checklist", s["h1"])]
    if pack.obligations:
        rows = [[o.article, o.title, o.party.replace("_", " "),
                 o.deadline_date.isoformat() + (" (in force)" if o.applies_now else " (upcoming)"),
                 _EVIDENCE_TEXT[o.evidence_status]] for o in pack.obligations]
        story.append(_table(["Article", "Obligation", "Party", "Applies from", "Evidence"], rows,
                            [27 * mm, 75 * mm, 22 * mm, 22 * mm, 24 * mm], s))
    else:
        story.append(_p("No obligations identified from the answers given.", s["body"]))

    story += [_p("3. Suggested disclosure texts", s["h1"])]
    if pack.disclosures:
        for t in pack.disclosures:
            story.append(KeepTogether([_p(f"{t.article}: {t.label}", s["h2"]), _p(t.disclosure, s["quote"])]))
        story.append(_p("Adapt wording to your product and language. Show it at the latest at the first "
                        "interaction or exposure (Art. 50(5)).", s["cell"]))
    else:
        story.append(_p("No Article 50 trigger identified; no disclosure text needed.", s["body"]))

    story += [_p("4. Gap list: missing evidence", s["h1"])]
    if pack.gaps:
        rows = [[g.kind.replace("_", " "), g.item, g.detail] for g in pack.gaps]
        story.append(_table(["Type", "Item", "What is missing"], rows, [28 * mm, 70 * mm, 72 * mm], s))
    else:
        story.append(_p("No gaps identified.", s["body"]))

    story += [PageBreak(), _p("5. Customer questionnaire answers", s["h1"])]
    rows = [[a.id, a.question, a.answer, a.basis, a.status.replace("_", " ")] for a in pack.customer_answers]
    story.append(_table(["#", "Question", "Answer", "Basis", "Status"], rows,
                        [10 * mm, 45 * mm, 80 * mm, 18 * mm, 17 * mm], s))

    if pack.annex_iv is not None and pack.annex_iv_gaps is not None:
        story += [PageBreak(), _p("6. Annex IV technical documentation draft", s["h1"]),
                  _p(pack.annex_iv_gaps.summary, s["body"]), Spacer(1, 2 * mm)]
        rows = []
        for item_attr in type(pack.annex_iv).model_fields:
            if not item_attr.startswith("item_"):
                continue
            item = getattr(pack.annex_iv, item_attr)
            number = item_attr.split("_")[1]
            for fname in type(item).model_fields:
                f = getattr(item, fname)
                if f.status == "filled":
                    rows.append([number, fname, str(f.value), f.provenance.source_artifact if f.provenance else ""])
                else:
                    rows.append([number, fname, "Not documented yet", ""])
        story.append(_table(["Item", "Field", "Content", "Source"], rows,
                            [12 * mm, 45 * mm, 85 * mm, 28 * mm], s))
        story.append(_p("A full Annex IV PDF in the original generator's format is written next to this "
                        "report (annex_iv.pdf).", s["cell"]))

    story += [PageBreak(), _p("Appendix A: Facts and evidence", s["h1"])]
    rows = []
    for f in pack.facts:
        if f.is_known:
            val = ", ".join(f.value) if isinstance(f.value, list) else (
                ("yes" if f.value else "no") if isinstance(f.value, bool) else str(f.value))
            ev = f'"{f.quote}" ({f.source}, verified verbatim)' if f.quote_verified else "self-declared in questionnaire"
        else:
            val, ev = "unknown", "-"
        rows.append([f.name, val or "none", ev])
    story.append(_table(["Fact", "Value", "Evidence"], rows, [50 * mm, 50 * mm, 70 * mm], s))

    SimpleDocTemplate(
        str(path), pagesize=A4, title=f"AI Act evidence pack: {pack.system_name}",
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
    ).build(story)
    return path


def write_all(pack: EvidencePack, out_dir: Path, *, annex_iv_pdf: bool = True) -> dict[str, Path]:
    """Write every output file into *out_dir*; return name -> path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": out_dir / "evidence_pack.pdf",
        "markdown": out_dir / "questionnaire_answers.md",
        "csv": out_dir / "questionnaire_answers.csv",
        "json": out_dir / "evidence_pack.json",
    }
    write_pdf(pack, paths["pdf"])
    paths["markdown"].write_text(answers_markdown(pack), encoding="utf-8")
    paths["csv"].write_text(answers_csv(pack), encoding="utf-8")
    paths["json"].write_text(pack.model_dump_json(indent=2), encoding="utf-8")
    if annex_iv_pdf and pack.annex_iv is not None and pack.annex_iv_gaps is not None:
        from output.pdf import generate_pdf

        paths["annex_iv_pdf"] = generate_pdf(pack.annex_iv, pack.annex_iv_gaps, out_dir / "annex_iv.pdf")
        paths["annex_iv_json"] = out_dir / "annex_iv.json"
        paths["annex_iv_json"].write_text(pack.annex_iv.model_dump_json(indent=2), encoding="utf-8")
    return paths
