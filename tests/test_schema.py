"""Tests for Annex IV Pydantic schema models (src/annex_iv/schema.py)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

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


def _make_provenance(**overrides: object) -> Provenance:
    defaults = {
        "source_artifact": "huggingface://google/gemma-2b",
        "locator": "README.md#intended-use",
        "extracted_at": datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        "extraction_model": "gpt-4o-mini-2024-07-18",
        "confidence": 0.92,
    }
    defaults.update(overrides)
    return Provenance(**defaults)


# ---------------------------------------------------------------------------
# Empty document construction
# ---------------------------------------------------------------------------


class TestEmptyDocument:
    def test_empty_document_is_valid(self) -> None:
        doc = AnnexIVDocument()
        assert doc.item_1_general is not None
        assert doc.item_9_post_market_monitoring is not None

    def test_empty_document_metadata_defaults_to_none(self) -> None:
        doc = AnnexIVDocument()
        assert doc.created_at is None
        assert doc.last_updated is None
        assert doc.generator_version is None
        assert doc.target_system_name is None

    def test_empty_document_all_fields_pending(self) -> None:
        doc = AnnexIVDocument()
        assert doc.item_1_general.intended_purpose.status == "pending"
        assert doc.item_2_development.design_specifications.status == "pending"


# ---------------------------------------------------------------------------
# DocumentedField validation
# ---------------------------------------------------------------------------


class TestDocumentedField:
    def test_filled_requires_provenance(self) -> None:
        with pytest.raises(ValidationError, match="requires provenance"):
            DocumentedField[str](
                value="some value",
                status="filled",
                provenance=None,
            )

    def test_filled_with_provenance_is_valid(self) -> None:
        field = DocumentedField[str](
            value="Credit scoring model for consumer lending",
            status="filled",
            provenance=_make_provenance(),
        )
        assert field.value == "Credit scoring model for consumer lending"
        assert field.provenance is not None
        assert field.provenance.confidence == 0.92

    def test_unavailable_with_none_value_and_provenance(self) -> None:
        field = DocumentedField[str](
            value=None,
            status="unavailable",
            provenance=None,
        )
        assert field.status == "unavailable"
        assert field.value is None
        assert field.provenance is None

    def test_pending_is_default(self) -> None:
        field = DocumentedField[str]()
        assert field.status == "pending"
        assert field.value is None
        assert field.provenance is None


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestSerialisationRoundTrip:
    def test_document_json_round_trip(self) -> None:
        doc = AnnexIVDocument(
            created_at=datetime(2025, 5, 1, tzinfo=timezone.utc),
            last_updated=datetime(2025, 5, 20, tzinfo=timezone.utc),
            generator_version="0.1.0",
            target_system_name="CreditScore-EU-v3",
        )
        doc.item_1_general.intended_purpose = DocumentedField[str](
            value="Automated credit scoring for consumer lending in the EU",
            status="filled",
            provenance=_make_provenance(),
        )
        doc.item_3_monitoring_control.capabilities_limitations = DocumentedField[str](
            value=None,
            status="unavailable",
            provenance=None,
        )

        json_str = doc.model_dump_json()
        restored = AnnexIVDocument.model_validate_json(json_str)

        assert restored.target_system_name == "CreditScore-EU-v3"
        assert restored.item_1_general.intended_purpose.value == (
            "Automated credit scoring for consumer lending in the EU"
        )
        assert restored.item_1_general.intended_purpose.status == "filled"
        assert restored.item_1_general.intended_purpose.provenance is not None
        assert (
            restored.item_3_monitoring_control.capabilities_limitations.status
            == "unavailable"
        )

    def test_round_trip_preserves_all_metadata(self) -> None:
        prov = _make_provenance(confidence=None)
        doc = AnnexIVDocument(
            generator_version="0.1.0",
            target_system_name="TestSystem",
        )
        doc.item_5_risk_management.risk_management_description = DocumentedField[str](
            value="Full risk management per Article 9",
            status="filled",
            provenance=prov,
        )

        json_str = doc.model_dump_json()
        restored = AnnexIVDocument.model_validate_json(json_str)
        restored_prov = (
            restored.item_5_risk_management.risk_management_description.provenance
        )

        assert restored_prov is not None
        assert restored_prov.confidence is None
        assert restored_prov.extraction_model == "gpt-4o-mini-2024-07-18"
        assert restored_prov.source_artifact == "huggingface://google/gemma-2b"


# ---------------------------------------------------------------------------
# Independent item construction
# ---------------------------------------------------------------------------


_ITEM_CLASSES = [
    AnnexIVItem1General,
    AnnexIVItem2Development,
    AnnexIVItem3MonitoringControl,
    AnnexIVItem4PerformanceMetrics,
    AnnexIVItem5RiskManagement,
    AnnexIVItem6LifecycleChanges,
    AnnexIVItem7HarmonisedStandards,
    AnnexIVItem8DeclarationOfConformity,
    AnnexIVItem9PostMarketMonitoring,
]


@pytest.mark.parametrize("cls", _ITEM_CLASSES, ids=lambda c: c.__name__)
def test_item_constructible_independently(cls: type) -> None:
    instance = cls()
    assert instance is not None
    for field_name in cls.model_fields:
        field_val = getattr(instance, field_name)
        assert isinstance(field_val, DocumentedField)
        assert field_val.status == "pending"
