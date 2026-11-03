"""Deterministic EU AI Act classification from facts.

Every decision here comes from data/rules.yaml through rules.evaluate. No LLM
is involved. Unknown inputs are treated conservatively by the rules (for
example, an unknown Art. 6(3) derogation means the system stays high-risk)
and are listed so the gap report can ask for them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from evidence_pack.facts import FACT_SPECS, FactSet
from evidence_pack.rules import catalogue_entry, evaluate, load_deadlines, load_rules


class Trigger(BaseModel):
    id: str
    article: str
    label: str
    disclosure: str


class Classification(BaseModel):
    role: str
    role_article: str
    risk_tier: str
    risk_article: str
    high_risk_basis: list[str]
    prohibited_practices: list[str]
    art50_triggers: list[Trigger]
    gpai_use: str
    gpai_article: str
    ai_literacy_applies: bool
    derived: dict[str, bool]
    unknown_inputs: dict[str, list[str]]
    """Fact name -> decisions that looked at it while it was unknown."""
    rules_version: str
    deadlines_version: str

    @property
    def trigger_ids(self) -> list[str]:
        return [t.id for t in self.art50_triggers]


def _first_match(entries: list[dict[str, Any]], values: dict[str, Any], decision: str,
                 unknown_inputs: dict[str, set[str]]) -> dict[str, Any]:
    for entry in entries:
        unknowns: set[str] = set()
        matched = evaluate(entry["when"], values, unknowns)
        for name in unknowns:
            unknown_inputs.setdefault(name, set()).add(decision)
        if matched:
            return entry
    raise ValueError(f"no rule matched for {decision}; the last rule must be 'always: true'")


def _fill(template: str, facts: FactSet) -> str:
    return " ".join(
        template.format(
            company_name=facts.text("company_name", "[company name]"),
            system_name=facts.text("system_name", "[system name]"),
        ).split()
    )


def fact_values(facts: FactSet) -> dict[str, Any]:
    return {spec.name: facts.value(spec.name) for spec in FACT_SPECS}


def classify(facts: FactSet) -> Classification:
    rules = load_rules()
    values = fact_values(facts)
    unknown_inputs: dict[str, set[str]] = {}

    derived: dict[str, bool] = {}
    for entry in rules["derived"]:
        unknowns: set[str] = set()
        derived[entry["id"]] = evaluate(entry["when"], values, unknowns)
        values[entry["id"]] = derived[entry["id"]]
        for name in unknowns:
            if name in derived:
                continue  # derived values are never unknown themselves
            unknown_inputs.setdefault(name, set()).add(entry["id"])

    role = _first_match(rules["roles"], values, "role", unknown_inputs)

    triggers: list[Trigger] = []
    for entry in rules["transparency_triggers"]:
        unknowns = set()
        if evaluate(entry["when"], values, unknowns):
            triggers.append(
                Trigger(
                    id=entry["id"],
                    article=entry["article"],
                    label=entry["label"],
                    disclosure=_fill(entry["disclosure"], facts),
                )
            )
        for name in unknowns:
            unknown_inputs.setdefault(name, set()).add("Article 50 transparency")
    values["transparency_triggered"] = bool(triggers)
    derived["transparency_triggered"] = bool(triggers)

    tier = _first_match(rules["risk_tiers"], values, "risk tier", unknown_inputs)
    gpai = _first_match(rules["gpai"], values, "GPAI use", unknown_inputs)

    lit_unknowns: set[str] = set()
    literacy = evaluate(rules["ai_literacy"]["when"], values, lit_unknowns)
    values["ai_literacy_applies"] = literacy
    derived["ai_literacy_applies"] = literacy

    basis: list[str] = []
    if derived["high_risk_annex_i"]:
        basis.append("Annex I product (Art. 6(1))")
    if derived["high_risk_annex_iii"]:
        for area in values["annex_iii_areas"] or []:
            entry = catalogue_entry("annex_iii_areas", area)
            basis.append(f"{entry['article']}: {entry['label']}")

    prohibited = [
        f"{catalogue_entry('prohibited_practices', p)['article']}: "
        f"{catalogue_entry('prohibited_practices', p)['label']}"
        for p in (values["prohibited_practices"] or [])
    ]

    return Classification(
        role=role["id"],
        role_article=role["article"],
        risk_tier=tier["id"],
        risk_article=tier["article"],
        high_risk_basis=basis,
        prohibited_practices=prohibited,
        art50_triggers=triggers,
        gpai_use=gpai["id"],
        gpai_article=gpai["article"],
        ai_literacy_applies=literacy,
        derived=derived,
        unknown_inputs={k: sorted(v) for k, v in sorted(unknown_inputs.items())},
        rules_version=str(rules["version"]),
        deadlines_version=str(load_deadlines()["version"]),
    )
