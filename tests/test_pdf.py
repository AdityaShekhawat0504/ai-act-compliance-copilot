"""Tests for PDF generation (src/output/pdf.py) and CLI --pdf flags.

All tests use synthetic AnnexIVDocument fixtures — no LLM or network calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pypdf import PdfReader
from typer.testing import CliRunner

from annex_iv.schema import (
    AnnexIVDocument,
    AnnexIVItem1General,
    AnnexIVItem2Development,
    AnnexIVItem3MonitoringControl,
    AnnexIVItem4PerformanceMetrics,
    AnnexIVItem5RiskManagement,
    AnnexIVItem6LifecycleChanges,
    AnnexIVItem7HarmonisedStandards,
    AnnexIVItem8DeclarationOfConformity,
    AnnexIVItem9PostMarketMonitoring,
    DocumentedField,
    Provenance,
)
from cli import app
from gap_analysis.analyzer import analyze
from output.pdf import generate_pdf

runner = CliRunner()

_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

_PROV = Provenance(
    source_artifact="huggingface://google/gemma-2b",
    locator="README.md#description",
    extracted_at=_NOW,
    extraction_model="gpt-4o-mini",
)


def _filled(val: str = "test value") -> DocumentedField:
    return DocumentedField(value=val, provenance=_PROV, status="filled")


def _unavailable() -> DocumentedField:
    return DocumentedField(value=None, status="unavailable")


def _make_partial_document() -> AnnexIVDocument:
    """Document with a mix of filled and unavailable fields."""
    return AnnexIVDocument(
        created_at=_NOW,
        last_updated=_NOW,
        generator_version="0.1.0",
        target_system_name="huggingface://google/gemma-2b",
        item_1_general=AnnexIVItem1General(
            intended_purpose=_filled("General-purpose language model for text generation"),
            provider_name=_filled("Google"),
            provider_address=_unavailable(),
            authorised_representative=_unavailable(),
            system_version=_filled("2b"),
            software_hardware_versions=_unavailable(),
            distribution_forms=_filled("Open weights via HuggingFace Hub"),
            hardware_requirements=_unavailable(),
            product_images=_unavailable(),
            user_interface_description=_unavailable(),
            instructions_of_use=_filled("Use via transformers library"),
        ),
        item_2_development=AnnexIVItem2Development(
            development_methods=_filled("Pre-trained transformer model"),
            design_specifications=_filled("Decoder-only transformer with 2B parameters"),
            system_architecture=_unavailable(),
            data_requirements=_filled("Trained on large web corpus"),
            human_oversight_assessment=_unavailable(),
            predetermined_changes=_unavailable(),
            validation_testing=_filled("Benchmarked on MMLU, HellaSwag, etc."),
            cybersecurity_measures=_unavailable(),
        ),
        item_3_monitoring_control=AnnexIVItem3MonitoringControl(
            capabilities_limitations=_filled("Strong on English text generation"),
            unintended_outcomes_risks=_unavailable(),
            human_oversight_measures=_unavailable(),
            input_data_specifications=_filled("Text input, max 8192 tokens"),
        ),
        item_4_performance_metrics=AnnexIVItem4PerformanceMetrics(
            metrics_appropriateness=_unavailable(),
        ),
        item_5_risk_management=AnnexIVItem5RiskManagement(
            risk_management_description=_unavailable(),
        ),
        item_6_lifecycle_changes=AnnexIVItem6LifecycleChanges(
            lifecycle_changes=_unavailable(),
        ),
        item_7_harmonised_standards=AnnexIVItem7HarmonisedStandards(
            harmonised_standards=_unavailable(),
        ),
        item_8_declaration_of_conformity=AnnexIVItem8DeclarationOfConformity(
            declaration_of_conformity=_unavailable(),
        ),
        item_9_post_market_monitoring=AnnexIVItem9PostMarketMonitoring(
            post_market_monitoring=_unavailable(),
        ),
    )


def _make_all_filled_document() -> AnnexIVDocument:
    """Document with every field filled — no gaps."""
    return AnnexIVDocument(
        created_at=_NOW,
        last_updated=_NOW,
        generator_version="0.1.0",
        target_system_name="test-system-all-filled",
        item_1_general=AnnexIVItem1General(
            intended_purpose=_filled(),
            provider_name=_filled(),
            provider_address=_filled(),
            authorised_representative=_filled(),
            system_version=_filled(),
            software_hardware_versions=_filled(),
            distribution_forms=_filled(),
            hardware_requirements=_filled(),
            product_images=_filled(),
            user_interface_description=_filled(),
            instructions_of_use=_filled(),
        ),
        item_2_development=AnnexIVItem2Development(
            development_methods=_filled(),
            design_specifications=_filled(),
            system_architecture=_filled(),
            data_requirements=_filled(),
            human_oversight_assessment=_filled(),
            predetermined_changes=_filled(),
            validation_testing=_filled(),
            cybersecurity_measures=_filled(),
        ),
        item_3_monitoring_control=AnnexIVItem3MonitoringControl(
            capabilities_limitations=_filled(),
            unintended_outcomes_risks=_filled(),
            human_oversight_measures=_filled(),
            input_data_specifications=_filled(),
        ),
        item_4_performance_metrics=AnnexIVItem4PerformanceMetrics(
            metrics_appropriateness=_filled(),
        ),
        item_5_risk_management=AnnexIVItem5RiskManagement(
            risk_management_description=_filled(),
        ),
        item_6_lifecycle_changes=AnnexIVItem6LifecycleChanges(
            lifecycle_changes=_filled(),
        ),
        item_7_harmonised_standards=AnnexIVItem7HarmonisedStandards(
            harmonised_standards=_filled(),
        ),
        item_8_declaration_of_conformity=AnnexIVItem8DeclarationOfConformity(
            declaration_of_conformity=_filled(),
        ),
        item_9_post_market_monitoring=AnnexIVItem9PostMarketMonitoring(
            post_market_monitoring=_filled(),
        ),
    )


def _make_all_empty_document() -> AnnexIVDocument:
    """Document with every field unavailable — no filled fields."""
    return AnnexIVDocument(
        created_at=_NOW,
        last_updated=_NOW,
        generator_version="0.1.0",
        target_system_name="test-system-all-empty",
    )


def _extract_all_text(reader: PdfReader) -> str:
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------------------------------------------------------------------------
# Core PDF generation tests
# ---------------------------------------------------------------------------


class TestPdfGeneration:
    def test_produces_nonempty_file(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"

        result = generate_pdf(doc, gap, out)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_valid_pdf(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        reader = PdfReader(out)
        assert len(reader.pages) >= 1

    def test_contains_system_name(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        assert "google/gemma-2b" in text

    def test_contains_annex_iv(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        assert "Annex IV" in text

    def test_contains_all_item_headers(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        for i in range(1, 10):
            assert f"Item {i}" in text, f"Item {i} header not found"

    def test_contains_appendices(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        assert "Appendix A" in text
        assert "Appendix B" in text

    def test_contains_completeness_percentage(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        pct = f"{gap.completeness_score:.0%}"
        assert pct in text, f"Expected completeness '{pct}' not found in PDF text"

    def test_minimum_page_count(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        reader = PdfReader(out)
        assert len(reader.pages) >= 11, (
            f"Expected at least 11 pages, got {len(reader.pages)}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPdfEdgeCases:
    def test_mostly_unavailable_contains_placeholder(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        assert "Not documented in source artifacts" in text

    def test_no_gaps_shows_no_gaps_message(self, tmp_path: Path) -> None:
        doc = _make_all_filled_document()
        gap = analyze(doc)
        assert len(gap.gaps) == 0
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        assert "No gaps identified" in text

    def test_no_filled_fields_shows_no_fields_message(self, tmp_path: Path) -> None:
        doc = _make_all_empty_document()
        gap = analyze(doc)
        out = tmp_path / "test.pdf"
        generate_pdf(doc, gap, out)

        text = _extract_all_text(PdfReader(out))
        assert "No fields populated" in text


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCliPdfFlags:
    def test_generate_pdf_flag_produces_pdf(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        out = tmp_path / "out"

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("cli.HuggingFaceModelCardLoader") as MockLoader,
            patch("cli.generate_annex_iv", return_value=doc),
        ):
            MockLoader.return_value.load.return_value = "artifact-sentinel"
            result = runner.invoke(
                app, ["generate", "google/gemma-2b", "-o", str(out), "--pdf"]
            )

        assert result.exit_code == 0, result.output
        pdf_path = out / "google__gemma-2b.pdf"
        assert pdf_path.exists()
        reader = PdfReader(pdf_path)
        assert len(reader.pages) >= 1

    def test_generate_no_pdf_flag_skips_pdf(self, tmp_path: Path) -> None:
        doc = _make_partial_document()
        out = tmp_path / "out"

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("cli.HuggingFaceModelCardLoader") as MockLoader,
            patch("cli.generate_annex_iv", return_value=doc),
        ):
            MockLoader.return_value.load.return_value = "artifact-sentinel"
            result = runner.invoke(
                app, ["generate", "google/gemma-2b", "-o", str(out), "--no-pdf"]
            )

        assert result.exit_code == 0, result.output
        pdf_path = out / "google__gemma-2b.pdf"
        assert not pdf_path.exists()
