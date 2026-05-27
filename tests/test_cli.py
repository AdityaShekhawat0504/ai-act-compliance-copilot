"""Tests for the CLI (src/cli.py).

All tests mock the LLM and HuggingFace calls — no network or API tokens needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
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

runner = CliRunner()

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)

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


def _make_document() -> AnnexIVDocument:
    return AnnexIVDocument(
        created_at=_NOW,
        last_updated=_NOW,
        generator_version="0.1.0",
        target_system_name="huggingface://google/gemma-2b",
        item_1_general=AnnexIVItem1General(
            intended_purpose=_filled(),
            provider_name=_filled(),
            provider_address=_unavailable(),
            authorised_representative=_unavailable(),
            system_version=_filled(),
            software_hardware_versions=_unavailable(),
            distribution_forms=_filled(),
            hardware_requirements=_unavailable(),
            product_images=_unavailable(),
            user_interface_description=_unavailable(),
            instructions_of_use=_filled(),
        ),
        item_2_development=AnnexIVItem2Development(
            development_methods=_filled(),
            design_specifications=_filled(),
            system_architecture=_unavailable(),
            data_requirements=_filled(),
            human_oversight_assessment=_unavailable(),
            predetermined_changes=_unavailable(),
            validation_testing=_filled(),
            cybersecurity_measures=_unavailable(),
        ),
        item_3_monitoring_control=AnnexIVItem3MonitoringControl(
            capabilities_limitations=_filled(),
            unintended_outcomes_risks=_unavailable(),
            human_oversight_measures=_unavailable(),
            input_data_specifications=_filled(),
        ),
        item_4_performance_metrics=AnnexIVItem4PerformanceMetrics(
            metrics_appropriateness=_filled(),
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


# ---- generate ----


def test_generate_hf_model(tmp_path: Path) -> None:
    doc = _make_document()
    out = tmp_path / "out"

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("cli.HuggingFaceModelCardLoader") as MockLoader,
        patch("cli.generate_annex_iv", return_value=doc),
    ):
        MockLoader.return_value.load.return_value = "artifact-sentinel"
        result = runner.invoke(app, ["generate", "google/gemma-2b", "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert (out / "google__gemma-2b.annex_iv.json").exists()
    assert (out / "google__gemma-2b.gap_report.json").exists()
    assert (out / "google__gemma-2b.summary.txt").exists()

    loaded = AnnexIVDocument.model_validate_json(
        (out / "google__gemma-2b.annex_iv.json").read_text()
    )
    assert loaded.target_system_name == doc.target_system_name


def test_generate_local_file(tmp_path: Path) -> None:
    local_md = tmp_path / "my_model.md"
    local_md.write_text("---\nlicense: mit\n---\n# Model Card\nA local model.\n")
    out = tmp_path / "out"
    doc = _make_document()

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("cli.HuggingFaceModelCardLoader") as MockLoader,
        patch("cli.generate_annex_iv", return_value=doc),
    ):
        MockLoader.return_value.load.return_value = "artifact-sentinel"
        result = runner.invoke(app, ["generate", str(local_md), "-o", str(out)])

    assert result.exit_code == 0, result.output
    assert (out / "my_model.annex_iv.json").exists()
    assert (out / "my_model.gap_report.json").exists()
    assert (out / "my_model.summary.txt").exists()


def test_generate_missing_api_key(tmp_path: Path) -> None:
    env = {k: v for k, v in __import__("os").environ.items() if k != "OPENAI_API_KEY"}

    with patch.dict("os.environ", env, clear=True):
        result = runner.invoke(app, ["generate", "google/gemma-2b", "-o", str(tmp_path)])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
    assert "Traceback" not in result.output


def test_generate_hf_404(tmp_path: Path) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("cli.HuggingFaceModelCardLoader") as MockLoader,
    ):
        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status_code = 404
        mock_response.json.return_value = {}
        MockLoader.return_value.load.side_effect = RepositoryNotFoundError(
            "not found", response=mock_response
        )
        result = runner.invoke(
            app, ["generate", "nonexistent/model", "-o", str(tmp_path)]
        )

    assert result.exit_code == 1
    assert "not found" in result.output.lower()
    assert "Traceback" not in result.output


def test_generate_no_gap_report(tmp_path: Path) -> None:
    doc = _make_document()
    out = tmp_path / "out"

    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
        patch("cli.HuggingFaceModelCardLoader") as MockLoader,
        patch("cli.generate_annex_iv", return_value=doc),
    ):
        MockLoader.return_value.load.return_value = "artifact-sentinel"
        result = runner.invoke(
            app, ["generate", "google/gemma-2b", "-o", str(out), "--no-gap-report"]
        )

    assert result.exit_code == 0, result.output
    assert (out / "google__gemma-2b.annex_iv.json").exists()
    assert not (out / "google__gemma-2b.gap_report.json").exists()
    assert (out / "google__gemma-2b.summary.txt").exists()


# ---- report ----


def test_report_valid_json(tmp_path: Path) -> None:
    doc = _make_document()
    annex_file = tmp_path / "test.annex_iv.json"
    annex_file.write_text(doc.model_dump_json(indent=2))

    result = runner.invoke(app, ["report", str(annex_file)])
    assert result.exit_code == 0, result.output

    gap_file = tmp_path / "test.gap_report.json"
    assert gap_file.exists()
    gap_data = json.loads(gap_file.read_text())
    assert 0.0 <= gap_data["completeness_score"] <= 1.0


def test_report_nonexistent_file() -> None:
    result = runner.invoke(app, ["report", "nonexistent.json"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_report_invalid_json(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json at all")

    result = runner.invoke(app, ["report", str(bad_file)])
    assert result.exit_code == 1
    assert "not a valid AnnexIVDocument" in result.output


# ---- version ----


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() != ""


# ---- help ----


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "generate" in result.output
    assert "report" in result.output
    assert "version" in result.output
