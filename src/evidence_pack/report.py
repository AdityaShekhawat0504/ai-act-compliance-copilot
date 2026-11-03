"""Build the evidence pack: obligations checklist, disclosures, gaps,
customer questionnaire answers and (for high-risk systems) an Annex IV draft.

All content is derived from facts and the deterministic classification.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Literal

from pydantic import BaseModel

from annex_iv.schema import AnnexIVDocument, DocumentedField, Provenance
from evidence_pack.classify import Classification, Trigger, classify, fact_values
from evidence_pack.facts import SPECS_BY_NAME, Fact, FactSet
from evidence_pack.rules import deadline, evaluate, load_rules
from gap_analysis.analyzer import GapReport, analyze

GENERATOR_VERSION = "2.0.0"

NOTICE = (
    "Not legal advice. This evidence pack is generated automatically from the "
    "information provided and a fixed set of rules. It is a starting point for "
    "your own assessment and for review by qualified counsel. Classification "
    "rules (version {rules}) and application dates (version {dates}) must be "
    "re-checked against the Official Journal of the European Union."
)

ROLE_TEXT = {
    "provider": "Provider",
    "deployer": "Deployer",
    "both": "Provider and deployer",
    "none": "Neither provider nor deployer (based on the answers given)",
}
TIER_TEXT = {
    "prohibited": "Prohibited practice",
    "high_risk": "High-risk",
    "limited_risk": "Limited risk (transparency obligations)",
    "minimal_risk": "Minimal risk",
}
GPAI_TEXT = {
    "provides_gpai_model": "Provides its own general-purpose AI model",
    "integrates_third_party": "Integrates a third-party general-purpose AI model",
    "none": "No general-purpose AI model reported",
}


class ObligationItem(BaseModel):
    id: str
    article: str
    party: str
    title: str
    deadline_key: str
    deadline_date: dt.date
    deadline_label: str
    applies_now: bool
    evidence_fact: str | None
    evidence_status: Literal["provided", "missing", "not_tracked"]
    evidence_source: str | None = None


class GapItem(BaseModel):
    kind: Literal["classification_input", "obligation_evidence", "evidence_note", "annex_iv"]
    item: str
    detail: str


class CustomerAnswer(BaseModel):
    id: str
    question: str
    answer: str
    basis: str
    evidence: str
    status: Literal["answered", "needs_input"]


class EvidencePack(BaseModel):
    generator_version: str = GENERATOR_VERSION
    generated_at: dt.datetime
    as_of: dt.date
    company_name: str
    system_name: str
    label: str | None = None
    """Free-text banner, e.g. 'FICTIONAL EXAMPLE'."""
    notice: str
    classification: Classification
    obligations: list[ObligationItem]
    disclosures: list[Trigger]
    gaps: list[GapItem]
    customer_answers: list[CustomerAnswer]
    facts: list[Fact]
    annex_iv: AnnexIVDocument | None = None
    annex_iv_gaps: GapReport | None = None


# ---------------------------------------------------------------------------
# Obligations
# ---------------------------------------------------------------------------


def _resolve_deadline(rule: dict, cls: Classification, facts: FactSet) -> dict:
    key = rule["deadline"]
    if key == "high_risk":
        options = []
        if cls.derived.get("high_risk_annex_iii"):
            options.append(deadline("high_risk_annex_iii"))
        if cls.derived.get("high_risk_annex_i"):
            options.append(deadline("high_risk_annex_i"))
        if not options:
            options.append(deadline("high_risk_annex_iii"))
        return min(options, key=lambda d: d["date"])
    if rule.get("deadline_if_legacy") and facts.value("on_eu_market_before_2026_08_02") is True:
        return deadline(rule["deadline_if_legacy"])
    return deadline(key)


def build_obligations(cls: Classification, facts: FactSet, as_of: dt.date) -> list[ObligationItem]:
    values = fact_values(facts)
    values.update(cls.derived)
    items: list[ObligationItem] = []
    for rule in load_rules()["obligations"]:
        if not evaluate(rule["when"], values):
            continue
        d = _resolve_deadline(rule, cls, facts)
        ev_name = rule.get("evidence")
        if ev_name is None:
            status, source = "not_tracked", None
        else:
            fact = facts.get(ev_name)
            status = "provided" if fact.truthy else "missing"
            source = fact.source if fact.truthy else None
        items.append(
            ObligationItem(
                id=rule["id"],
                article=rule["article"],
                party=rule["party"],
                title=rule["title"],
                deadline_key=d["key"],
                deadline_date=d["date"],
                deadline_label=d["label"],
                applies_now=d["date"] <= as_of,
                evidence_fact=ev_name,
                evidence_status=status,
                evidence_source=source,
            )
        )
    items.sort(key=lambda o: (o.deadline_date, _article_number(o.article), o.article))
    return items


def _article_number(article: str) -> int:
    match = re.search(r"\d+", article)
    return int(match.group()) if match else 0


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


def build_gaps(cls: Classification, obligations: list[ObligationItem], facts: FactSet,
               annex_gaps: GapReport | None) -> list[GapItem]:
    gaps: list[GapItem] = []
    for name, affects in cls.unknown_inputs.items():
        gaps.append(
            GapItem(
                kind="classification_input",
                item=SPECS_BY_NAME[name].question,
                detail=f"Unknown. Affects: {', '.join(affects)}. The result above assumes the conservative case.",
            )
        )
    for ob in obligations:
        if ob.evidence_status == "missing":
            gaps.append(
                GapItem(
                    kind="obligation_evidence",
                    item=f"{ob.article}: {ob.title}",
                    detail=f"No evidence yet. Provide: {SPECS_BY_NAME[ob.evidence_fact].question}",
                )
            )
        elif ob.evidence_status == "not_tracked":
            gaps.append(
                GapItem(
                    kind="obligation_evidence",
                    item=f"{ob.article}: {ob.title}",
                    detail="Prepare and file the supporting document; this pack cannot verify it.",
                )
            )
    for note in facts.notes:
        gaps.append(GapItem(kind="evidence_note", item=note.split(":", 1)[0], detail=note))
    if annex_gaps is not None and annex_gaps.gaps:
        required = [g for g in annex_gaps.gaps if g.required]
        gaps.append(
            GapItem(
                kind="annex_iv",
                item="Annex IV technical documentation draft",
                detail=f"{annex_gaps.summary} {len(required)} required fields still empty.",
            )
        )
    return gaps


# ---------------------------------------------------------------------------
# Customer questionnaire answers
# ---------------------------------------------------------------------------


def _evidence(facts: FactSet, *names: str) -> str:
    parts = []
    for n in names:
        f = facts.get(n)
        if not f.is_known:
            continue
        if f.quote_verified:
            parts.append(f'{f.source}: "{f.quote}"')
        else:
            parts.append(f"self-declared ({n})")
    return "; ".join(parts) if parts else "none"


def _bool_answer(facts: FactSet, name: str, yes: str, no: str) -> tuple[str, bool]:
    v = facts.value(name)
    if v is None:
        return "To be confirmed.", False
    return (yes if v else no), True


def build_customer_answers(cls: Classification, obligations: list[ObligationItem],
                           facts: FactSet) -> list[CustomerAnswer]:
    answers: list[CustomerAnswer] = []

    def add(qid: str, question: str, answer: str, basis: str, evidence: str, ok: bool) -> None:
        answers.append(
            CustomerAnswer(id=qid, question=question, answer=answer, basis=basis,
                           evidence=evidence, status="answered" if ok else "needs_input")
        )

    desc = facts.value("system_description")
    add("Q01", "Does your product use AI? What does the AI feature do?",
        f"Yes. {desc}" if desc else "Yes. Description to be provided.",
        "Art. 3(1)", _evidence(facts, "system_description"), bool(desc))

    role_ok = not ({"places_on_market_under_own_name", "uses_ai_internally"} & set(cls.unknown_inputs))
    add("Q02", "What is your role under the EU AI Act (Regulation (EU) 2024/1689)?",
        ROLE_TEXT[cls.role] + ".", cls.role_article,
        _evidence(facts, "places_on_market_under_own_name", "uses_ai_internally"), role_ok)

    tier_answer = TIER_TEXT[cls.risk_tier]
    if cls.risk_tier == "high_risk":
        tier_answer += ". Basis: " + "; ".join(cls.high_risk_basis)
    if cls.risk_tier == "prohibited":
        tier_answer += ". Matches: " + "; ".join(cls.prohibited_practices)
    add("Q03", "How is the AI system classified under the AI Act?",
        tier_answer + ".", cls.risk_article,
        _evidence(facts, "annex_iii_areas", "prohibited_practices", "safety_component_annex_i_product"),
        "annex_iii_areas" not in cls.unknown_inputs and "prohibited_practices" not in cls.unknown_inputs)

    pp = facts.value("prohibited_practices")
    add("Q04", "Is the system used for any practice prohibited by Article 5?",
        "To be confirmed." if pp is None else ("Yes: " + "; ".join(cls.prohibited_practices) + "." if pp else "No."),
        "Art. 5(1)", _evidence(facts, "prohibited_practices"), pp is not None)

    if cls.art50_triggers:
        lines = []
        for t in cls.art50_triggers:
            ob = [o for o in obligations if o.article.startswith(t.article.split(",")[0])]
            state = ", ".join(f"{o.party}: evidence {o.evidence_status.replace('_', ' ')}" for o in ob) or "obligation sits with the other party in the chain"
            lines.append(f"{t.article} ({t.label}): {state}")
        t_answer = "Applies. " + " | ".join(lines)
    else:
        t_answer = "No Article 50 transparency trigger identified."
    add("Q05", "Which Article 50 transparency obligations apply, and how do you meet them?",
        t_answer, "Art. 50",
        _evidence(facts, "discloses_ai_interaction", "marks_output_machine_readable", "labels_ai_generated_content"),
        not any("Article 50 transparency" in a for a in cls.unknown_inputs.values()))

    names = facts.value("gpai_model_names")
    g_answer = GPAI_TEXT[cls.gpai_use] + (f": {names}." if names else ".")
    add("Q06", "Do you use general-purpose AI models (for example large language models)? Which ones?",
        g_answer, cls.gpai_article, _evidence(facts, "uses_gpai_model", "gpai_model_names"),
        facts.value("uses_gpai_model") is not None)

    a, ok = _bool_answer(facts, "trains_on_customer_data",
                         "Yes, customer data is used to train or fine-tune models.",
                         "No, customer data is not used to train or fine-tune models.")
    add("Q07", "Is customer data used to train or improve your AI models?", a, "Art. 10; GDPR",
        _evidence(facts, "trains_on_customer_data"), ok)

    ho = facts.value("human_oversight_description")
    add("Q08", "How is human oversight of the AI output ensured?",
        str(ho) if ho else "To be confirmed.", "Art. 14; Art. 26",
        _evidence(facts, "human_oversight_description"), bool(ho))

    a, ok = _bool_answer(facts, "ai_literacy_programme",
                         "Staff who build and operate the AI feature receive AI literacy training.",
                         "No AI literacy programme is in place yet.")
    add("Q09", "What AI literacy measures are in place (Article 4)?", a, "Art. 4",
        _evidence(facts, "ai_literacy_programme"), ok)

    lim = facts.value("known_limitations")
    met = facts.value("performance_metrics")
    lim_answer = " ".join(x for x in [str(lim) if lim else "", f"Metrics: {met}" if met else ""] if x)
    add("Q10", "What are the known limitations and how is accuracy measured?",
        lim_answer or "To be confirmed.", "Art. 13; Art. 15",
        _evidence(facts, "known_limitations", "performance_metrics"), bool(lim or met))

    contact = facts.value("ai_governance_contact")
    add("Q11", "Who is the contact for AI governance and AI Act questions?",
        str(contact) if contact else "To be confirmed.", "-",
        _evidence(facts, "ai_governance_contact"), bool(contact))

    open_now = [o for o in obligations if o.applies_now and o.evidence_status != "provided"]
    upcoming = [o for o in obligations if not o.applies_now]
    plan = (
        f"{len(obligations)} obligations identified; {len(open_now)} already apply and still need evidence; "
        f"{len(upcoming)} apply from a future date (earliest "
        f"{min(o.deadline_date for o in upcoming).isoformat() if upcoming else 'n/a'})."
    )
    add("Q12", "What is your AI Act compliance status and timeline?", plan,
        "Art. 113", "obligations checklist in this pack", True)
    return answers


# ---------------------------------------------------------------------------
# Annex IV draft (high-risk only)
# ---------------------------------------------------------------------------

ANNEX_IV_MAP: list[tuple[str, str, str]] = [
    ("item_1_general", "intended_purpose", "system_description"),
    ("item_1_general", "provider_name", "company_name"),
    ("item_1_general", "provider_address", "provider_address"),
    ("item_1_general", "system_version", "system_version"),
    ("item_1_general", "distribution_forms", "distribution_form"),
    ("item_2_development", "development_methods", "gpai_model_names"),
    ("item_2_development", "data_requirements", "training_data_description"),
    ("item_2_development", "human_oversight_assessment", "human_oversight_description"),
    ("item_3_monitoring_control", "capabilities_limitations", "known_limitations"),
    ("item_3_monitoring_control", "human_oversight_measures", "human_oversight_description"),
    ("item_4_performance_metrics", "metrics_appropriateness", "performance_metrics"),
    ("item_5_risk_management", "risk_management_description", "risk_management_description"),
    ("item_9_post_market_monitoring", "post_market_monitoring", "post_market_monitoring_description"),
]


def build_annex_iv_draft(facts: FactSet, now: dt.datetime) -> AnnexIVDocument:
    """Fill the Annex IV schema from known text facts. Each filled field keeps
    provenance (file and verified quote, or 'questionnaire'). Every other
    field is unavailable and shows up in the Annex IV gap analysis."""
    items: dict[str, dict[str, DocumentedField]] = {
        attr: {name: DocumentedField(value=None, status="unavailable")
               for name in AnnexIVDocument.model_fields[attr].annotation.model_fields}
        for attr in AnnexIVDocument.model_fields
        if attr.startswith("item_")
    }
    for item_attr, field_name, fact_name in ANNEX_IV_MAP:
        fact = facts.get(fact_name)
        if not (fact.is_known and fact.value):
            continue
        locator = f'quote: "{fact.quote}"' if fact.quote_verified else f"questionnaire answer: {fact_name}"
        items[item_attr][field_name] = DocumentedField(
            value=facts.text(fact_name),
            status="filled",
            provenance=Provenance(
                source_artifact=fact.source or "questionnaire",
                locator=locator,
                extracted_at=now,
                extraction_model=(
                    "two-pass extraction + verbatim quote check"
                    if fact.origin == "document" else "questionnaire"
                ),
            ),
        )
    built = {
        attr: AnnexIVDocument.model_fields[attr].annotation(**fields)
        for attr, fields in items.items()
    }
    return AnnexIVDocument(
        created_at=now, last_updated=now, generator_version=GENERATOR_VERSION,
        target_system_name=facts.text("system_name", "AI system"), **built,
    )


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def build_evidence_pack(facts: FactSet, *, as_of: dt.date | None = None,
                        now: dt.datetime | None = None, label: str | None = None) -> EvidencePack:
    now = now or dt.datetime.now(dt.timezone.utc)
    as_of = as_of or now.date()
    cls = classify(facts)
    obligations = build_obligations(cls, facts, as_of)

    annex = annex_gaps = None
    if cls.derived.get("is_high_risk") and cls.derived.get("is_provider"):
        annex = build_annex_iv_draft(facts, now)
        annex_gaps = analyze(annex)

    return EvidencePack(
        generated_at=now,
        as_of=as_of,
        company_name=facts.text("company_name", "Unnamed company"),
        system_name=facts.text("system_name", "Unnamed AI system"),
        label=label,
        notice=NOTICE.format(rules=cls.rules_version, dates=cls.deadlines_version),
        classification=cls,
        obligations=obligations,
        disclosures=cls.art50_triggers,
        gaps=build_gaps(cls, obligations, facts, annex_gaps),
        customer_answers=build_customer_answers(cls, obligations, facts),
        facts=[facts.get(name) for name in SPECS_BY_NAME],
        annex_iv=annex,
        annex_iv_gaps=annex_gaps,
    )
