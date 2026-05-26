"""Tests for LLM-assisted field extraction pipeline (src/extraction/).

Most tests use a FakeInstructor client that returns deterministic responses
keyed off the response_model passed in. One integration test runs the real
pipeline against a cached HuggingFace fixture if OPENAI_API_KEY is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from annex_iv.schema import (
    AnnexIVDocument,
    AnnexIVItem1General,
    AnnexIVItem2Development,
    DocumentedField,
)
from extraction.pipeline import (
    FieldCandidate,
    _candidate_model_for,
    extract_candidates,
    generate_annex_iv,
)
from extraction.validators import GroundingCheck, validate_grounding
from ingestion.base import NormalisedArtifact


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "huggingface"


# ---------------------------------------------------------------------------
# Fake instructor client
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, parent: "FakeInstructorClient") -> None:
        self._parent = parent

    def create(
        self,
        *,
        response_model: type[Any],
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        self._parent.calls.append(
            {
                "response_model": response_model,
                "messages": messages,
                "model": model,
            }
        )
        return self._parent.dispatch(response_model, messages, model)


class _FakeChat:
    def __init__(self, parent: "FakeInstructorClient") -> None:
        self.completions = _FakeCompletions(parent)


class FakeInstructorClient:
    """Minimal stand-in for an instructor-wrapped OpenAI client.

    Construct with a dispatcher callable that returns an instance of the
    requested response_model based on (response_model, messages, model).
    """

    def __init__(
        self,
        dispatcher: Callable[[type[Any], list[dict[str, str]], str | None], Any],
    ) -> None:
        self.dispatch = dispatcher
        self.calls: list[dict[str, Any]] = []
        self.chat = _FakeChat(self)


def _build_artifact(raw_text: str = "irrelevant source text") -> NormalisedArtifact:
    return NormalisedArtifact(
        artifact_type="huggingface_model_card",
        source_uri="huggingface://test/fake-model",
        structured_content={"description": "a test"},
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Pass 1 tests
# ---------------------------------------------------------------------------


class TestPass1ExtractCandidates:
    def test_pass1_returns_pending_for_found_fields(self) -> None:
        artifact = _build_artifact()

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            kwargs: dict[str, FieldCandidate] = {}
            for name in response_model.model_fields:
                if name == "intended_purpose":
                    kwargs[name] = FieldCandidate(
                        value="a credit-scoring model", found_in_source=True
                    )
                else:
                    kwargs[name] = FieldCandidate(value=None, found_in_source=False)
            return response_model(**kwargs)

        client = FakeInstructorClient(dispatcher)
        item = extract_candidates(
            artifact, AnnexIVItem1General, client=client
        )

        assert isinstance(item, AnnexIVItem1General)
        assert item.intended_purpose.status == "pending"
        assert item.intended_purpose.value == "a credit-scoring model"
        assert item.intended_purpose.provenance is None
        assert item.provider_name.status == "unavailable"
        assert item.provider_name.value is None

    def test_pass1_uses_correct_prompt(self) -> None:
        artifact = _build_artifact()

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            return response_model(
                **{
                    name: FieldCandidate(value=None, found_in_source=False)
                    for name in response_model.model_fields
                }
            )

        client = FakeInstructorClient(dispatcher)
        extract_candidates(artifact, AnnexIVItem2Development, client=client)

        assert len(client.calls) == 1
        system_msg = client.calls[0]["messages"][0]["content"]
        assert "item 2 of Regulation (EU) 2024/1689" in system_msg
        assert "data_requirements" in system_msg

    def test_pass1_passes_model_to_client(self) -> None:
        artifact = _build_artifact()

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            return response_model(
                **{
                    name: FieldCandidate(value=None, found_in_source=False)
                    for name in response_model.model_fields
                }
            )

        client = FakeInstructorClient(dispatcher)
        extract_candidates(
            artifact, AnnexIVItem1General, model="gpt-4o", client=client
        )

        assert client.calls[0]["model"] == "gpt-4o"

    def test_pass1_empty_value_treated_as_unavailable(self) -> None:
        artifact = _build_artifact()

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            return response_model(
                **{
                    name: FieldCandidate(value="", found_in_source=True)
                    for name in response_model.model_fields
                }
            )

        client = FakeInstructorClient(dispatcher)
        item = extract_candidates(
            artifact, AnnexIVItem1General, client=client
        )
        assert item.intended_purpose.status == "unavailable"
        assert item.intended_purpose.value is None


# ---------------------------------------------------------------------------
# Pass 2 tests
# ---------------------------------------------------------------------------


class TestPass2Validation:
    def test_grounded_field_receives_provenance(self) -> None:
        artifact = _build_artifact("the system is a credit-scoring model for the EU")
        field = DocumentedField[str](
            value="credit-scoring model for the EU", status="pending"
        )

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            assert response_model is GroundingCheck
            return GroundingCheck(
                grounded=True,
                locator="line 1",
                reasoning="direct match",
            )

        client = FakeInstructorClient(dispatcher)
        result = validate_grounding(field, artifact, client=client)

        assert result.status == "filled"
        assert result.value == "credit-scoring model for the EU"
        assert result.provenance is not None
        assert result.provenance.source_artifact == "huggingface://test/fake-model"
        assert result.provenance.locator == "line 1"
        assert result.provenance.extraction_model == "gpt-4o-mini"

    def test_ungrounded_field_dropped_to_unavailable(self) -> None:
        artifact = _build_artifact("the system is a credit-scoring model")
        field = DocumentedField[str](
            value="a deep neural network trained on 6 trillion tokens",
            status="pending",
        )

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            return GroundingCheck(
                grounded=False,
                locator=None,
                reasoning="source contains no token-count claim",
            )

        client = FakeInstructorClient(dispatcher)
        result = validate_grounding(field, artifact, client=client)

        assert result.status == "unavailable"
        assert result.value is None
        assert result.provenance is None

    def test_unavailable_field_passes_through(self) -> None:
        artifact = _build_artifact()
        field = DocumentedField[str](value=None, status="unavailable")

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            raise AssertionError("Pass 2 should not call LLM for unavailable fields")

        client = FakeInstructorClient(dispatcher)
        result = validate_grounding(field, artifact, client=client)

        assert result.status == "unavailable"
        assert result.value is None

    def test_pending_with_none_value_becomes_unavailable(self) -> None:
        artifact = _build_artifact()
        field = DocumentedField[str](value=None, status="pending")

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            raise AssertionError("Should not call LLM when value is None")

        client = FakeInstructorClient(dispatcher)
        result = validate_grounding(field, artifact, client=client)
        assert result.status == "unavailable"

    def test_grounded_with_no_locator_uses_placeholder(self) -> None:
        artifact = _build_artifact("source text containing the value")
        field = DocumentedField[str](value="the value", status="pending")

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            return GroundingCheck(
                grounded=True, locator=None, reasoning="matches"
            )

        client = FakeInstructorClient(dispatcher)
        result = validate_grounding(field, artifact, client=client)
        assert result.status == "filled"
        assert result.provenance is not None
        assert result.provenance.locator == "(no specific locator)"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_generate_annex_iv_runs_both_passes(self) -> None:
        artifact = _build_artifact("the system is a credit-scoring model for the EU")

        pass1_call_count = 0
        pass2_call_count = 0

        def dispatcher(
            response_model: type[Any], messages: list[dict[str, str]], model: str | None
        ) -> Any:
            nonlocal pass1_call_count, pass2_call_count
            if response_model is GroundingCheck:
                pass2_call_count += 1
                return GroundingCheck(
                    grounded=True, locator="somewhere", reasoning="ok"
                )
            pass1_call_count += 1
            # Mark only the first field of each item as found
            kwargs: dict[str, FieldCandidate] = {}
            for i, name in enumerate(response_model.model_fields):
                if i == 0:
                    kwargs[name] = FieldCandidate(
                        value=f"value-for-{name}", found_in_source=True
                    )
                else:
                    kwargs[name] = FieldCandidate(value=None, found_in_source=False)
            return response_model(**kwargs)

        client = FakeInstructorClient(dispatcher)
        doc = generate_annex_iv(artifact, client=client)

        assert isinstance(doc, AnnexIVDocument)
        assert pass1_call_count == 9
        assert pass2_call_count == 9
        assert doc.item_1_general.intended_purpose.status == "filled"
        assert doc.item_1_general.intended_purpose.provenance is not None
        assert doc.item_1_general.provider_name.status == "unavailable"
        assert doc.target_system_name == "huggingface://test/fake-model"
        assert doc.generator_version == "0.1.0"


# ---------------------------------------------------------------------------
# Candidate model construction
# ---------------------------------------------------------------------------


class TestCandidateModelConstruction:
    def test_candidate_model_mirrors_item_fields(self) -> None:
        cls = _candidate_model_for(AnnexIVItem1General)
        assert set(cls.model_fields) == set(AnnexIVItem1General.model_fields)

    def test_candidate_model_is_cached(self) -> None:
        a = _candidate_model_for(AnnexIVItem1General)
        b = _candidate_model_for(AnnexIVItem1General)
        assert a is b


# ---------------------------------------------------------------------------
# Live integration test
# ---------------------------------------------------------------------------


_HAS_API_KEY = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="OPENAI_API_KEY not set; skipping live integration test",
)
def test_live_pipeline_against_cached_fixture() -> None:
    fixture_path = FIXTURES_DIR / "nlp_thorough.json"
    if not fixture_path.exists():
        pytest.skip(
            "fixture not present; run tests/test_ingestion.py first to "
            "populate fixtures/huggingface/"
        )

    data = json.loads(fixture_path.read_text())
    artifact = NormalisedArtifact.model_validate(data)

    doc = generate_annex_iv(artifact)

    assert isinstance(doc, AnnexIVDocument)
    assert doc.target_system_name == artifact.source_uri

    item1_filled = [
        name
        for name in AnnexIVItem1General.model_fields
        if getattr(doc.item_1_general, name).status == "filled"
    ]
    item2_filled = [
        name
        for name in AnnexIVItem2Development.model_fields
        if getattr(doc.item_2_development, name).status == "filled"
    ]

    assert item1_filled, "expected item 1 to have at least one filled field"
    assert item2_filled, "expected item 2 to have at least one filled field"

    for field_name in item1_filled:
        field = getattr(doc.item_1_general, field_name)
        assert field.provenance is not None
        assert field.provenance.source_artifact == artifact.source_uri
