"""Tests for the evidence pack rules engine, classification and obligations.

Every rule in src/evidence_pack/data/rules.yaml has at least one test case
here; test_every_rule_has_a_case fails if a rule is added without one.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from evidence_pack.classify import classify
from evidence_pack.facts import SPECS_BY_NAME
from evidence_pack.questionnaire import facts_from_answers
from evidence_pack.report import build_obligations
from evidence_pack.rules import RuleError, deadline, evaluate, load_deadlines, load_rules

# A known, "nothing special" provider: every classification input answered no.
BASE: dict[str, Any] = {
    "company_name": "Test Co (fictional)",
    "system_name": "Test system",
    "places_on_market_under_own_name": True,
    "rebrands_or_substantially_modifies": False,
    "uses_ai_internally": False,
    "prohibited_practices": [],
    "annex_iii_areas": [],
    "annex_iii_derogation_claimed": False,
    "performs_profiling": False,
    "safety_component_annex_i_product": False,
    "interacts_with_people": False,
    "generates_synthetic_content": False,
    "emotion_recognition": False,
    "biometric_categorisation": False,
    "generates_deepfakes": False,
    "publishes_public_interest_text": False,
    "human_editorial_review": False,
    "on_eu_market_before_2026_08_02": False,
    "uses_gpai_model": False,
    "trains_own_gpai_model": False,
}


def cls_for(**overrides: Any):
    answers = {**BASE, **overrides}
    answers = {k: v for k, v in answers.items() if v is not None}
    return classify(facts_from_answers(answers))


def obligations_for(as_of: dt.date = dt.date(2026, 11, 3), **overrides: Any):
    answers = {k: v for k, v in {**BASE, **overrides}.items() if v is not None}
    facts = facts_from_answers(answers)
    return {o.id: o for o in build_obligations(classify(facts), facts, as_of)}


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_eq(self) -> None:
        assert evaluate({"field": "a", "eq": True}, {"a": True})
        assert not evaluate({"field": "a", "eq": True}, {"a": False})

    def test_nonempty(self) -> None:
        assert evaluate({"field": "a", "nonempty": True}, {"a": ["x"]})
        assert not evaluate({"field": "a", "nonempty": True}, {"a": []})

    def test_intersects(self) -> None:
        assert evaluate({"field": "a", "intersects": ["x", "y"]}, {"a": ["y"]})
        assert not evaluate({"field": "a", "intersects": ["x"]}, {"a": ["z"]})

    def test_intersects_requires_list(self) -> None:
        with pytest.raises(RuleError):
            evaluate({"field": "a", "intersects": ["x"]}, {"a": "x"})

    def test_unknown_never_matches_and_is_reported(self) -> None:
        unknowns: set[str] = set()
        assert not evaluate({"field": "a", "eq": True}, {"a": None}, unknowns)
        assert not evaluate({"field": "a", "eq": False}, {"a": None}, unknowns)
        assert unknowns == {"a"}

    def test_not_of_unknown_is_true_and_reports(self) -> None:
        unknowns: set[str] = set()
        assert evaluate({"not": {"field": "a", "eq": True}}, {"a": None}, unknowns)
        assert unknowns == {"a"}

    def test_all_settled_by_known_false_hides_other_unknowns(self) -> None:
        unknowns: set[str] = set()
        cond = {"all": [{"field": "a", "eq": True}, {"field": "b", "eq": True}]}
        assert not evaluate(cond, {"a": False, "b": None}, unknowns)
        assert unknowns == set()

    def test_any_settled_by_known_true_hides_other_unknowns(self) -> None:
        unknowns: set[str] = set()
        cond = {"any": [{"field": "a", "eq": True}, {"field": "b", "eq": True}]}
        assert evaluate(cond, {"a": True, "b": None}, unknowns)
        assert unknowns == set()

    def test_unsettled_all_reports_unknowns(self) -> None:
        unknowns: set[str] = set()
        cond = {"all": [{"field": "a", "eq": True}, {"field": "b", "eq": True}]}
        assert not evaluate(cond, {"a": True, "b": None}, unknowns)
        assert unknowns == {"b"}

    def test_always(self) -> None:
        assert evaluate({"always": True}, {})

    @pytest.mark.parametrize(
        "cond",
        [{}, {"bogus": 1}, {"field": "a"}, {"field": "a", "eq": 1, "nonempty": True}, {"field": "missing", "eq": 1}],
    )
    def test_malformed_conditions_raise(self, cond: dict) -> None:
        with pytest.raises(RuleError):
            evaluate(cond, {"a": 1})


# ---------------------------------------------------------------------------
# Rules and dates files
# ---------------------------------------------------------------------------


def _walk_fields(cond: Any):
    if isinstance(cond, dict):
        if "field" in cond:
            yield cond["field"]
        for v in cond.values():
            yield from _walk_fields(v)
    elif isinstance(cond, list):
        for v in cond:
            yield from _walk_fields(v)


class TestRulesFile:
    def test_versioned(self) -> None:
        assert load_rules()["version"]
        assert load_deadlines()["version"]

    def test_every_rule_has_article(self) -> None:
        rules = load_rules()
        for section in ["derived", "roles", "transparency_triggers", "risk_tiers", "gpai", "obligations"]:
            for entry in rules[section]:
                assert entry.get("article"), f"{section}.{entry['id']} has no article"
        for catalogue in ["prohibited_practices", "annex_iii_areas"]:
            for entry in rules[catalogue]:
                assert entry.get("article")

    def test_conditions_only_reference_known_names(self) -> None:
        rules = load_rules()
        known = set(SPECS_BY_NAME) | {d["id"] for d in rules["derived"]} | {
            "transparency_triggered", "ai_literacy_applies"}
        for section in ["derived", "roles", "transparency_triggers", "risk_tiers", "gpai", "obligations"]:
            for entry in rules[section]:
                for name in _walk_fields(entry["when"]):
                    assert name in known, f"{section}.{entry['id']} refers to {name!r}"

    def test_obligation_deadlines_and_evidence_exist(self) -> None:
        keys = set(load_deadlines()["deadlines"]) | {"high_risk"}
        for ob in load_rules()["obligations"]:
            assert ob["deadline"] in keys
            if ob.get("deadline_if_legacy"):
                assert ob["deadline_if_legacy"] in keys
            if ob.get("evidence"):
                assert ob["evidence"] in SPECS_BY_NAME

    def test_ordered_sections_end_with_catch_all(self) -> None:
        rules = load_rules()
        for section in ["roles", "risk_tiers", "gpai"]:
            assert rules[section][-1]["when"] == {"always": True}

    @pytest.mark.parametrize(
        "key,date",
        [
            ("prohibited_practices", dt.date(2025, 2, 2)),
            ("ai_literacy", dt.date(2025, 2, 2)),
            ("gpai_obligations", dt.date(2025, 8, 2)),
            ("transparency", dt.date(2026, 8, 2)),
            ("transparency_marking_legacy", dt.date(2026, 12, 2)),
            ("high_risk_annex_iii", dt.date(2027, 12, 2)),
            ("high_risk_annex_i", dt.date(2028, 8, 2)),
        ],
    )
    def test_configured_dates(self, key: str, date: dt.date) -> None:
        assert deadline(key)["date"] == date

    def test_dates_file_carries_recheck_warning(self) -> None:
        from evidence_pack.rules import DEADLINES_PATH

        assert "Official Journal" in DEADLINES_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# One case per classification rule
# ---------------------------------------------------------------------------

# rule id -> (overrides on BASE, check on the Classification)
CLASSIFICATION_CASES: dict[str, tuple[dict[str, Any], Any]] = {
    # derived
    "is_provider": ({}, lambda c: c.derived["is_provider"]),
    "is_deployer": ({"uses_ai_internally": True}, lambda c: c.derived["is_deployer"]),
    "is_prohibited": ({"prohibited_practices": ["social_scoring"]}, lambda c: c.derived["is_prohibited"]),
    "high_risk_annex_i": ({"safety_component_annex_i_product": True}, lambda c: c.derived["high_risk_annex_i"]),
    "annex_iii_derogation": (
        {"annex_iii_areas": ["education"], "annex_iii_derogation_claimed": True, "performs_profiling": False},
        lambda c: c.derived["annex_iii_derogation"] and not c.derived["high_risk_annex_iii"],
    ),
    "high_risk_annex_iii": ({"annex_iii_areas": ["employment"]}, lambda c: c.derived["high_risk_annex_iii"]),
    "is_high_risk": ({"annex_iii_areas": ["credit_scoring"]}, lambda c: c.derived["is_high_risk"]),
    # roles
    "both": ({"uses_ai_internally": True}, lambda c: c.role == "both"),
    "provider": ({}, lambda c: c.role == "provider"),
    "deployer": ({"places_on_market_under_own_name": False, "uses_ai_internally": True}, lambda c: c.role == "deployer"),
    "none": ({"places_on_market_under_own_name": False}, lambda c: c.role == "none"),
    # Art. 50 triggers
    "chatbot": ({"interacts_with_people": True}, lambda c: c.trigger_ids == ["chatbot"]),
    "synthetic_content": ({"generates_synthetic_content": True}, lambda c: c.trigger_ids == ["synthetic_content"]),
    "emotion_biometric": ({"emotion_recognition": True}, lambda c: c.trigger_ids == ["emotion_biometric"]),
    "deepfake": ({"generates_deepfakes": True}, lambda c: c.trigger_ids == ["deepfake"]),
    "ai_text_public_interest": (
        {"publishes_public_interest_text": True, "human_editorial_review": False},
        lambda c: c.trigger_ids == ["ai_text_public_interest"],
    ),
    # risk tiers
    "prohibited": ({"prohibited_practices": ["untargeted_face_scraping"]}, lambda c: c.risk_tier == "prohibited"),
    "high_risk": ({"annex_iii_areas": ["law_enforcement"]}, lambda c: c.risk_tier == "high_risk"),
    "limited_risk": ({"interacts_with_people": True}, lambda c: c.risk_tier == "limited_risk"),
    "minimal_risk": ({}, lambda c: c.risk_tier == "minimal_risk"),
    # GPAI
    "provides_gpai_model": ({"trains_own_gpai_model": True}, lambda c: c.gpai_use == "provides_gpai_model"),
    "integrates_third_party": ({"uses_gpai_model": True}, lambda c: c.gpai_use == "integrates_third_party"),
    "gpai_none": ({}, lambda c: c.gpai_use == "none"),
}


@pytest.mark.parametrize("rule_id", sorted(CLASSIFICATION_CASES))
def test_classification_rule(rule_id: str) -> None:
    overrides, check = CLASSIFICATION_CASES[rule_id]
    c = cls_for(**overrides)
    assert check(c), f"rule {rule_id} not satisfied: {c.model_dump()}"


class TestClassificationEdges:
    def test_rebranding_makes_provider(self) -> None:
        c = cls_for(places_on_market_under_own_name=False, rebrands_or_substantially_modifies=True)
        assert c.role == "provider"

    def test_derogation_blocked_by_profiling(self) -> None:
        c = cls_for(annex_iii_areas=["employment"], annex_iii_derogation_claimed=True, performs_profiling=True)
        assert c.risk_tier == "high_risk"

    def test_unknown_derogation_stays_high_risk_and_is_a_gap(self) -> None:
        c = cls_for(annex_iii_areas=["education"], annex_iii_derogation_claimed=None, performs_profiling=None)
        assert c.risk_tier == "high_risk"
        assert "annex_iii_derogation_claimed" in c.unknown_inputs

    def test_editorial_review_exempts_public_interest_text(self) -> None:
        c = cls_for(publishes_public_interest_text=True, human_editorial_review=True)
        assert "ai_text_public_interest" not in c.trigger_ids

    def test_unknown_editorial_review_triggers_conservatively(self) -> None:
        c = cls_for(publishes_public_interest_text=True, human_editorial_review=None)
        assert "ai_text_public_interest" in c.trigger_ids
        assert "human_editorial_review" in c.unknown_inputs

    def test_irrelevant_unknown_not_reported(self) -> None:
        c = cls_for(human_editorial_review=None)
        assert "human_editorial_review" not in c.unknown_inputs

    def test_biometric_categorisation_triggers_art50_3(self) -> None:
        assert cls_for(biometric_categorisation=True).trigger_ids == ["emotion_biometric"]

    def test_prohibited_beats_high_risk(self) -> None:
        c = cls_for(prohibited_practices=["emotion_recognition_work_education"], annex_iii_areas=["employment"])
        assert c.risk_tier == "prohibited"
        assert c.prohibited_practices and "Art. 5(1)(f)" in c.prohibited_practices[0]

    def test_high_risk_beats_limited(self) -> None:
        c = cls_for(annex_iii_areas=["employment"], interacts_with_people=True)
        assert c.risk_tier == "high_risk"
        assert c.trigger_ids == ["chatbot"]

    def test_high_risk_basis_lists_annex_iii_point(self) -> None:
        c = cls_for(annex_iii_areas=["employment"])
        assert any("Annex III, point 4" in b for b in c.high_risk_basis)

    def test_ai_literacy_applies_to_provider_and_deployer_only(self) -> None:
        assert cls_for().ai_literacy_applies
        assert cls_for(places_on_market_under_own_name=False, uses_ai_internally=True).ai_literacy_applies
        assert not cls_for(places_on_market_under_own_name=False).ai_literacy_applies

    def test_unknown_role_inputs_reported(self) -> None:
        c = cls_for(places_on_market_under_own_name=None, uses_ai_internally=None)
        assert c.role == "none"
        assert "places_on_market_under_own_name" in c.unknown_inputs

    def test_disclosure_text_uses_company_name(self) -> None:
        c = cls_for(interacts_with_people=True)
        assert "Test Co (fictional)" in c.art50_triggers[0].disclosure

    def test_versions_reported(self) -> None:
        c = cls_for()
        assert c.rules_version == str(load_rules()["version"])
        assert c.deadlines_version == str(load_deadlines()["version"])


# ---------------------------------------------------------------------------
# One case per obligation rule
# ---------------------------------------------------------------------------

HIGH_RISK = {"annex_iii_areas": ["employment"], "performs_profiling": True}

OBLIGATION_CASES: dict[str, dict[str, Any]] = {
    "art4_ai_literacy": {},
    "art5_stop_prohibited": {"prohibited_practices": ["social_scoring"]},
    "art50_1_disclose_interaction": {"interacts_with_people": True},
    "art50_2_machine_readable_marking": {"generates_synthetic_content": True},
    "art50_3_inform_exposed_persons": {"places_on_market_under_own_name": False, "uses_ai_internally": True,
                                       "emotion_recognition": True},
    "art50_4_label_deepfakes": {"uses_ai_internally": True, "generates_deepfakes": True},
    "art50_4_label_public_interest_text": {"uses_ai_internally": True, "publishes_public_interest_text": True},
    "gpai_downstream_documentation": {"uses_gpai_model": True},
    "gpai_provider_obligations": {"trains_own_gpai_model": True},
    "art6_4_document_derogation": {"annex_iii_areas": ["education"], "annex_iii_derogation_claimed": True},
    "art9_risk_management": HIGH_RISK,
    "art10_data_governance": HIGH_RISK,
    "art11_technical_documentation": HIGH_RISK,
    "art12_logging": HIGH_RISK,
    "art14_human_oversight": HIGH_RISK,
    "art17_quality_management": HIGH_RISK,
    "art43_conformity_assessment": HIGH_RISK,
    "art72_post_market_monitoring": HIGH_RISK,
    "art26_deployer_duties": {**HIGH_RISK, "uses_ai_internally": True},
    "art26_7_inform_workers": {**HIGH_RISK, "uses_ai_internally": True},
    "art27_fundamental_rights_impact": {"annex_iii_areas": ["credit_scoring"], "uses_ai_internally": True},
}


@pytest.mark.parametrize("rule_id", sorted(OBLIGATION_CASES))
def test_obligation_rule_fires(rule_id: str) -> None:
    assert rule_id in obligations_for(**OBLIGATION_CASES[rule_id])


@pytest.mark.parametrize("rule_id", sorted(set(OBLIGATION_CASES) - {"art4_ai_literacy"}))
def test_obligation_rule_silent_on_baseline(rule_id: str) -> None:
    assert rule_id not in obligations_for()


def test_every_rule_has_a_case() -> None:
    rules = load_rules()
    ids = {e["id"] for s in ["derived", "roles", "transparency_triggers", "risk_tiers"] for e in rules[s]}
    ids |= {f"gpai_{e['id']}" if e["id"] == "none" else e["id"] for e in rules["gpai"]}
    assert ids <= set(CLASSIFICATION_CASES), ids - set(CLASSIFICATION_CASES)
    ob_ids = {e["id"] for e in rules["obligations"]}
    assert ob_ids == set(OBLIGATION_CASES), ob_ids ^ set(OBLIGATION_CASES)


class TestObligationDetails:
    def test_deployer_only_gets_no_provider_duties(self) -> None:
        obs = obligations_for(places_on_market_under_own_name=False, uses_ai_internally=True, **HIGH_RISK)
        assert "art9_risk_management" not in obs
        assert "art26_deployer_duties" in obs

    def test_provider_chatbot_without_deployer_role_has_no_50_4(self) -> None:
        obs = obligations_for(generates_deepfakes=True)
        assert "art50_4_label_deepfakes" not in obs

    def test_high_risk_deadline_annex_iii(self) -> None:
        assert obligations_for(**HIGH_RISK)["art9_risk_management"].deadline_date == dt.date(2027, 12, 2)

    def test_high_risk_deadline_annex_i(self) -> None:
        ob = obligations_for(safety_component_annex_i_product=True)["art9_risk_management"]
        assert ob.deadline_date == dt.date(2028, 8, 2)

    def test_high_risk_deadline_takes_earliest_when_both(self) -> None:
        ob = obligations_for(safety_component_annex_i_product=True, **HIGH_RISK)["art9_risk_management"]
        assert ob.deadline_date == dt.date(2027, 12, 2)

    def test_marking_grace_period_for_legacy_systems(self) -> None:
        new = obligations_for(generates_synthetic_content=True)["art50_2_machine_readable_marking"]
        legacy = obligations_for(generates_synthetic_content=True, on_eu_market_before_2026_08_02=True)[
            "art50_2_machine_readable_marking"]
        assert new.deadline_date == dt.date(2026, 8, 2)
        assert legacy.deadline_date == dt.date(2026, 12, 2)

    def test_applies_now_depends_on_as_of(self) -> None:
        early = obligations_for(as_of=dt.date(2026, 1, 1), interacts_with_people=True)
        late = obligations_for(as_of=dt.date(2026, 8, 2), interacts_with_people=True)
        assert not early["art50_1_disclose_interaction"].applies_now
        assert late["art50_1_disclose_interaction"].applies_now
        assert early["art4_ai_literacy"].applies_now

    def test_evidence_status(self) -> None:
        missing = obligations_for(interacts_with_people=True)["art50_1_disclose_interaction"]
        provided = obligations_for(interacts_with_people=True, discloses_ai_interaction=True)[
            "art50_1_disclose_interaction"]
        untracked = obligations_for(**HIGH_RISK)["art43_conformity_assessment"]
        assert missing.evidence_status == "missing"
        assert provided.evidence_status == "provided"
        assert untracked.evidence_status == "not_tracked"

    def test_fria_not_for_employment(self) -> None:
        obs = obligations_for(uses_ai_internally=True, **HIGH_RISK)
        assert "art27_fundamental_rights_impact" not in obs
