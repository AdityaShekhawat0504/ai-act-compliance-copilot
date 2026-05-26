"""Tests for the Annex IV gap analyzer (src/gap_analysis/analyzer.py)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from annex_iv.schema import (
    AnnexIVDocument,
    DocumentedField,
    Provenance,
)
from gap_analysis.analyzer import (
    FIELD_REGULATION_TEXT,
    FieldGap,
    GapReport,
    analyze,
    required_for,
    suggested_artifact_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ITEM_TO_ATTR: dict[int, str] = {
    1: "item_1_general",
    2: "item_2_development",
    3: "item_3_monitoring_control",
    4: "item_4_performance_metrics",
    5: "item_5_risk_management",
    6: "item_6_lifecycle_changes",
    7: "item_7_harmonised_standards",
    8: "item_8_declaration_of_conformity",
    9: "item_9_post_market_monitoring",
}


def _provenance() -> Provenance:
    return Provenance(
        source_artifact="huggingface://acme/test-model",
        locator="README.md#section",
        extracted_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
        extraction_model="gpt-4o-mini-2024-07-18",
        confidence=0.9,
    )


def _filled(value: str = "value") -> DocumentedField[str]:
    return DocumentedField[str](
        value=value, status="filled", provenance=_provenance()
    )


def _unavailable() -> DocumentedField[str]:
    return DocumentedField[str](value=None, status="unavailable", provenance=None)


def _all_schema_fields() -> list[tuple[int, str]]:
    doc = AnnexIVDocument()
    pairs: list[tuple[int, str]] = []
    for item_number, attr in _ITEM_TO_ATTR.items():
        item_model = getattr(doc, attr)
        for field_name in type(item_model).model_fields:
            pairs.append((item_number, field_name))
    return pairs


def _fill_every_field(doc: AnnexIVDocument) -> None:
    for item_number, attr in _ITEM_TO_ATTR.items():
        item_model = getattr(doc, attr)
        for field_name in type(item_model).model_fields:
            setattr(item_model, field_name, _filled(f"filled-{field_name}"))


# ---------------------------------------------------------------------------
# Empty document
# ---------------------------------------------------------------------------


class TestEmptyDocument:
    def test_empty_document_has_zero_completeness(self) -> None:
        report = analyze(AnnexIVDocument())
        assert report.completeness_score == 0.0
        assert report.completeness_required_only == 0.0

    def test_empty_document_all_fields_pending(self) -> None:
        report = analyze(AnnexIVDocument())
        assert report.pending_fields == report.total_fields
        assert report.unavailable_fields == 0
        assert report.filled_fields == 0

    def test_empty_document_gaps_cover_every_schema_field(self) -> None:
        report = analyze(AnnexIVDocument())
        all_pairs = set(_all_schema_fields())
        gap_pairs = {(g.annex_iv_item, g.field_name) for g in report.gaps}
        assert gap_pairs == all_pairs
        assert all(g.status == "pending" for g in report.gaps)


# ---------------------------------------------------------------------------
# Fully-filled document
# ---------------------------------------------------------------------------


class TestFullyFilledDocument:
    def test_fully_filled_document_is_complete(self) -> None:
        doc = AnnexIVDocument(target_system_name="CreditScore-EU-v3")
        _fill_every_field(doc)

        report = analyze(doc)

        assert report.completeness_score == 1.0
        assert report.completeness_required_only == 1.0
        assert report.gaps == []
        assert report.filled_fields == report.total_fields
        assert report.target_system_name == "CreditScore-EU-v3"
        assert "100%" in report.summary


# ---------------------------------------------------------------------------
# Mixed document — hand-computed expected values
# ---------------------------------------------------------------------------


class TestMixedDocument:
    """A hand-constructed document with a known mix of filled/unavailable/
    pending fields. Counts and scores are checked against values computed
    directly from the schema (so the test does not duplicate the
    implementation's required/total bookkeeping)."""

    def _build(self) -> AnnexIVDocument:
        doc = AnnexIVDocument(target_system_name="MixedFixture")
        # Fill four required fields across items 1, 2, 3, 5.
        doc.item_1_general.intended_purpose = _filled("automated credit scoring")
        doc.item_2_development.system_architecture = _filled("transformer encoder")
        doc.item_3_monitoring_control.capabilities_limitations = _filled(
            "F1=0.82 on holdout"
        )
        doc.item_5_risk_management.risk_management_description = _filled(
            "Article 9 RMS in place"
        )
        # Mark three fields explicitly unavailable.
        doc.item_1_general.product_images = _unavailable()
        doc.item_2_development.cybersecurity_measures = _unavailable()
        doc.item_9_post_market_monitoring.post_market_monitoring = _unavailable()
        return doc

    def test_counts_match_hand_computed(self) -> None:
        doc = self._build()
        report = analyze(doc)

        total = len(_all_schema_fields())
        assert report.total_fields == total
        assert report.filled_fields == 4
        assert report.unavailable_fields == 3
        assert report.pending_fields == total - 4 - 3

    def test_completeness_scores(self) -> None:
        doc = self._build()
        report = analyze(doc)

        total = len(_all_schema_fields())
        assert report.completeness_score == pytest.approx(4 / total)

        total_required = sum(
            1 for item, fld in _all_schema_fields() if required_for(item, fld)
        )
        # All four filled fields are required ones (intended_purpose,
        # system_architecture, capabilities_limitations,
        # risk_management_description).
        assert report.completeness_required_only == pytest.approx(
            4 / total_required
        )

    def test_gap_items_match_unfilled_set(self) -> None:
        doc = self._build()
        report = analyze(doc)

        unfilled_pairs = set(_all_schema_fields()) - {
            (1, "intended_purpose"),
            (2, "system_architecture"),
            (3, "capabilities_limitations"),
            (5, "risk_management_description"),
        }
        gap_pairs = {(g.annex_iv_item, g.field_name) for g in report.gaps}
        assert gap_pairs == unfilled_pairs

    def test_gap_status_matches_documented_field(self) -> None:
        doc = self._build()
        report = analyze(doc)

        gap_status = {(g.annex_iv_item, g.field_name): g.status for g in report.gaps}
        assert gap_status[(1, "product_images")] == "unavailable"
        assert gap_status[(2, "cybersecurity_measures")] == "unavailable"
        assert gap_status[(9, "post_market_monitoring")] == "unavailable"
        # Untouched fields should still be pending.
        assert gap_status[(1, "provider_name")] == "pending"
        assert gap_status[(4, "metrics_appropriateness")] == "pending"


# ---------------------------------------------------------------------------
# Score ordering invariant
# ---------------------------------------------------------------------------


class TestScoreOrdering:
    """``completeness_required_only`` must be ≥ ``completeness_score`` when
    no conditional fields are filled — the required denominator is a strict
    subset of the total denominator while the numerator is unchanged."""

    def test_empty_document(self) -> None:
        report = analyze(AnnexIVDocument())
        assert report.completeness_required_only >= report.completeness_score

    def test_fully_filled_document(self) -> None:
        doc = AnnexIVDocument()
        _fill_every_field(doc)
        report = analyze(doc)
        assert report.completeness_required_only >= report.completeness_score

    def test_only_required_fields_filled(self) -> None:
        doc = AnnexIVDocument()
        for item_number, attr in _ITEM_TO_ATTR.items():
            item_model = getattr(doc, attr)
            for field_name in type(item_model).model_fields:
                if required_for(item_number, field_name):
                    setattr(item_model, field_name, _filled())
        report = analyze(doc)
        assert report.completeness_required_only == 1.0
        assert report.completeness_required_only >= report.completeness_score


# ---------------------------------------------------------------------------
# Metadata completeness — every schema field is covered
# ---------------------------------------------------------------------------


class TestMetadataCoverage:
    def test_every_field_has_regulation_text(self) -> None:
        for pair in _all_schema_fields():
            assert pair in FIELD_REGULATION_TEXT, pair
            assert FIELD_REGULATION_TEXT[pair].strip() != ""

    def test_every_field_has_suggested_artifact(self) -> None:
        for item, field in _all_schema_fields():
            assert suggested_artifact_for(item, field).strip() != ""

    def test_every_field_has_required_flag(self) -> None:
        for item, field in _all_schema_fields():
            value = required_for(item, field)
            assert isinstance(value, bool)

    def test_no_orphan_metadata_entries(self) -> None:
        """Metadata dicts must not contain stale entries that no longer
        correspond to schema fields."""
        all_pairs = set(_all_schema_fields())
        assert set(FIELD_REGULATION_TEXT.keys()) == all_pairs

    def test_items_4_to_9_are_all_required(self) -> None:
        for item, field in _all_schema_fields():
            if item >= 4:
                assert required_for(item, field) is True

    def test_known_conditional_fields_are_not_required(self) -> None:
        # These four fields are conditioned on applicability in the
        # regulation text and must therefore be required=False.
        assert required_for(1, "authorised_representative") is False
        assert required_for(1, "product_images") is False
        assert required_for(2, "data_requirements") is False
        assert required_for(2, "predetermined_changes") is False
        assert required_for(3, "input_data_specifications") is False


# ---------------------------------------------------------------------------
# FieldGap content sanity
# ---------------------------------------------------------------------------


class TestFieldGapContent:
    def test_every_gap_has_non_empty_metadata(self) -> None:
        doc = AnnexIVDocument()
        report = analyze(doc)
        assert report.gaps  # sanity: empty document yields gaps
        for gap in report.gaps:
            assert isinstance(gap, FieldGap)
            assert gap.regulation_text.strip() != ""
            assert gap.suggested_artifact_type.strip() != ""
            assert gap.status in ("pending", "unavailable")
            assert 1 <= gap.annex_iv_item <= 9

    def test_gap_metadata_matches_lookup_helpers(self) -> None:
        doc = AnnexIVDocument()
        report = analyze(doc)
        for gap in report.gaps:
            assert gap.suggested_artifact_type == suggested_artifact_for(
                gap.annex_iv_item, gap.field_name
            )
            assert gap.required == required_for(
                gap.annex_iv_item, gap.field_name
            )
            assert gap.regulation_text == FIELD_REGULATION_TEXT[
                (gap.annex_iv_item, gap.field_name)
            ]


# ---------------------------------------------------------------------------
# Summary snapshot
# ---------------------------------------------------------------------------


class TestSummarySnapshot:
    def test_summary_matches_expected_format(self) -> None:
        doc = AnnexIVDocument(target_system_name="SnapshotFixture")
        # Two filled, two unavailable; the rest pending.
        doc.item_1_general.intended_purpose = _filled("classify spam")
        doc.item_5_risk_management.risk_management_description = _filled("RMS")
        doc.item_8_declaration_of_conformity.declaration_of_conformity = (
            _unavailable()
        )
        doc.item_9_post_market_monitoring.post_market_monitoring = _unavailable()

        report = analyze(doc)

        total = len(_all_schema_fields())
        expected_pct = f"{2 / total:.0%}"
        total_required = sum(
            1 for item, fld in _all_schema_fields() if required_for(item, fld)
        )
        expected_req_pct = f"{2 / total_required:.0%}"
        expected_gap_count = total - 2  # everything except the two filled
        # Item 5 has a single field which is now filled, so it drops out of
        # the gap-items list. All other items still have at least one gap.
        expected_gap_items = [1, 2, 3, 4, 6, 7, 8, 9]

        assert f"{2}/{total} fields filled" in report.summary
        assert expected_pct in report.summary
        assert expected_req_pct in report.summary
        assert f"{expected_gap_count} gaps identified" in report.summary
        assert str(expected_gap_items) in report.summary

    def test_summary_when_no_gaps(self) -> None:
        doc = AnnexIVDocument()
        _fill_every_field(doc)
        report = analyze(doc)
        assert "0 gaps identified" in report.summary
        assert "100%" in report.summary


# ---------------------------------------------------------------------------
# Lookup-helper error handling
# ---------------------------------------------------------------------------


class TestLookupHelpers:
    def test_suggested_artifact_for_unknown_field_raises(self) -> None:
        with pytest.raises(KeyError):
            suggested_artifact_for(1, "not_a_real_field")

    def test_required_for_unknown_field_raises(self) -> None:
        with pytest.raises(KeyError):
            required_for(9, "not_a_real_field")


# ---------------------------------------------------------------------------
# GapReport schema sanity
# ---------------------------------------------------------------------------


def test_gap_report_round_trips_through_json() -> None:
    doc = AnnexIVDocument(target_system_name="RoundTrip")
    doc.item_1_general.intended_purpose = _filled("x")
    report = analyze(doc)

    json_str = report.model_dump_json()
    restored = GapReport.model_validate_json(json_str)

    assert restored.target_system_name == "RoundTrip"
    assert restored.total_fields == report.total_fields
    assert len(restored.gaps) == len(report.gaps)
    assert restored.summary == report.summary
