"""PDF documentation package for an AnnexIVDocument.

Layout: cover page (target system, completeness scores), table of contents,
one section per Annex IV item (regulation text as a blockquote, then every
field with its status and provenance), Appendix A (gap report) and
Appendix B (provenance index of every filled field).

Pure rendering over already-validated data: no LLM calls.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from annex_iv.schema import AnnexIVDocument
from gap_analysis.analyzer import GapReport

_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "annex_iv" / "reference"

ITEM_SECTIONS: list[tuple[int, str, str]] = [
    (1, "item_1_general", "General description of the AI system"),
    (2, "item_2_development", "Elements of the AI system and its development process"),
    (3, "item_3_monitoring_control", "Monitoring, functioning and control"),
    (4, "item_4_performance_metrics", "Appropriateness of the performance metrics"),
    (5, "item_5_risk_management", "Risk management system (Article 9)"),
    (6, "item_6_lifecycle_changes", "Relevant changes through the lifecycle"),
    (7, "item_7_harmonised_standards", "Harmonised standards applied"),
    (8, "item_8_declaration_of_conformity", "EU declaration of conformity (Article 47)"),
    (9, "item_9_post_market_monitoring", "Post-market monitoring system (Article 72)"),
]

PLACEHOLDER = "Not documented in source artifacts."


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=22, leading=28),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=16, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading3"], fontSize=11, spaceBefore=8),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=9.5, leading=13),
        "quote": ParagraphStyle(
            "q",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            leftIndent=10,
            textColor=colors.HexColor("#444444"),
        ),
        "small": ParagraphStyle(
            "s", parent=base["BodyText"], fontSize=8, leading=10,
            textColor=colors.HexColor("#555555"),
        ),
    }


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def _regulation_text(item_number: int) -> str:
    path = _REFERENCE_DIR / f"item_{item_number}.md"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    # Keep the regulation body, drop the markdown heading and source line.
    if "---" in text:
        text = text.split("---", 1)[1]
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _table(rows: list[list[str]], widths: list[float], style: ParagraphStyle) -> Table:
    data = [[_p(str(cell), style) for cell in row] for row in rows]
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAF0")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def generate_pdf(document: AnnexIVDocument, gap_report: GapReport, output_path: Path) -> Path:
    """Render *document* and *gap_report* to a PDF at *output_path*."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s = _styles()
    story: list = []

    # Cover page
    name = document.target_system_name or "Unnamed AI system"
    story += [
        Spacer(1, 40 * mm),
        _p("Annex IV Technical Documentation", s["title"]),
        _p("Regulation (EU) 2024/1689 (EU AI Act), Article 11(1)", s["body"]),
        Spacer(1, 10 * mm),
        _p(f"Target system: {name}", s["h2"]),
        _p(f"Generator version: {document.generator_version or 'unknown'}", s["body"]),
        _p(
            f"Created: {document.created_at.isoformat() if document.created_at else 'unknown'}",
            s["body"],
        ),
        Spacer(1, 6 * mm),
        _p(f"Completeness (all fields): {gap_report.completeness_score:.0%}", s["body"]),
        _p(
            f"Completeness (required fields): {gap_report.completeness_required_only:.0%}",
            s["body"],
        ),
        _p(
            f"Fields filled: {gap_report.filled_fields} of {gap_report.total_fields}; "
            f"gaps: {len(gap_report.gaps)}",
            s["body"],
        ),
        Spacer(1, 10 * mm),
        _p(
            "Every filled field carries provenance to its source artifact. Fields that "
            "could not be grounded in the source are left empty and listed in Appendix A. "
            "This document is not legal advice.",
            s["small"],
        ),
        PageBreak(),
    ]

    # Table of contents
    story.append(_p("Contents", s["h1"]))
    for number, _attr, title in ITEM_SECTIONS:
        story.append(_p(f"Item {number}: {title}", s["body"]))
    story.append(_p("Appendix A: Gap analysis", s["body"]))
    story.append(_p("Appendix B: Provenance index", s["body"]))
    story.append(PageBreak())

    # One section per item
    provenance_rows: list[list[str]] = []
    for number, attr, title in ITEM_SECTIONS:
        story.append(_p(f"Item {number}: {title}", s["h1"]))
        reg = _regulation_text(number)
        if reg:
            story.append(_p(reg, s["quote"]))
            story.append(Spacer(1, 4 * mm))
        item = getattr(document, attr)
        for field_name in type(item).model_fields:
            field = getattr(item, field_name)
            story.append(_p(field_name.replace("_", " ").capitalize(), s["h2"]))
            if field.status == "filled" and field.value is not None:
                story.append(_p(str(field.value), s["body"]))
                prov = field.provenance
                if prov is not None:
                    story.append(
                        _p(
                            f"Source: {prov.source_artifact} | locator: {prov.locator} | "
                            f"model: {prov.extraction_model}",
                            s["small"],
                        )
                    )
                    provenance_rows.append(
                        [f"{number}", field_name, prov.source_artifact, prov.locator]
                    )
            else:
                story.append(_p(f"[{field.status}] {PLACEHOLDER}", s["small"]))
        story.append(PageBreak())

    # Appendix A: gaps
    story.append(_p("Appendix A: Gap analysis", s["h1"]))
    story.append(_p(gap_report.summary, s["body"]))
    story.append(Spacer(1, 4 * mm))
    if gap_report.gaps:
        gaps = sorted(gap_report.gaps, key=lambda g: (g.annex_iv_item, not g.required))
        rows = [["Item", "Field", "Required", "Suggested source"]]
        rows += [
            [
                str(g.annex_iv_item),
                g.field_name,
                "yes" if g.required else "no",
                g.suggested_artifact_type,
            ]
            for g in gaps
        ]
        story.append(_table(rows, [14 * mm, 55 * mm, 18 * mm, 83 * mm], s["small"]))
    else:
        story.append(_p("No gaps identified.", s["body"]))
    story.append(PageBreak())

    # Appendix B: provenance
    story.append(_p("Appendix B: Provenance index", s["h1"]))
    if provenance_rows:
        rows = [["Item", "Field", "Source artifact", "Locator"]] + provenance_rows
        story.append(_table(rows, [14 * mm, 50 * mm, 56 * mm, 50 * mm], s["small"]))
    else:
        story.append(_p("No fields populated.", s["body"]))

    SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        title=f"Annex IV documentation: {name}",
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    ).build(story)
    return output_path
