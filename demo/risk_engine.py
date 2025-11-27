# risk_engine.py

"""
Advanced risk scoring engine for AI Act-style classification.

We use 5 pillars:
- Domain risk (Annex III style sectors)
- Data risk (personal / sensitive / scale / third-party)
- Model risk (complexity, black-box, generative, online learning)
- Deployment impact (legal rights, automation, safety, real-time, EU users)
- Governance controls (reduces risk: oversight, logging, docs, monitoring)

TOTAL SCORE = domain + data + model + deployment - governance

Risk category:
  0-4   : Low Risk
  5-9   : Limited Risk
  10-14 : High Risk
  15+   : Very High Risk
"""

def _domain_risk(domain: str):
    domain = (domain or "").lower()
    score = 0
    reasons = []

    # Mapping roughly aligned with Annex III categories
    if domain in ("finance", "credit_insurance"):
        score = 7
        reasons.append("Finance / credit / insurance impacts access to essential economic services (Annex III area).")
    elif domain in ("employment_hr", "hr"):
        score = 7
        reasons.append("Employment / HR affects access to jobs and career opportunities (Annex III area).")
    elif domain in ("education",):
        score = 6
        reasons.append("Education / vocational training impacts access to education (Annex III area).")
    elif domain in ("healthcare_medical", "health"):
        score = 7
        reasons.append("Healthcare / medical applications impact health and safety, often linked to regulated products.")
    elif domain in ("biometrics",):
        score = 8
        reasons.append("Biometric identification / categorisation is explicitly listed as high-risk.")
    elif domain in ("law_enforcement_public_safety", "public_safety", "law_enforcement"):
        score = 8
        reasons.append("Law enforcement / public safety systems are explicitly listed as high-risk.")
    elif domain in ("migration_border_asylum",):
        score = 8
        reasons.append("Migration / border / asylum management is explicitly listed as high-risk.")
    elif domain in ("justice_democracy", "justice_democratic_processes"):
        score = 8
        reasons.append("Justice / democratic processes are explicitly high-impact for fundamental rights.")
    else:
        score = 3
        reasons.append("Domain not explicitly high-risk, defaulting to moderate base risk.")

    return score, reasons


def _data_risk(answers):
    score = 0
    reasons = []

    if answers.get("uses_personal_data"):
        score += 2
        reasons.append("Uses personal data (identifiable individuals).")

    if answers.get("uses_special_categories"):
        score += 3
        reasons.append("Uses sensitive / special category data (health, ethnicity, religion, etc.).")

    if answers.get("large_scale_data"):
        score += 2
        reasons.append("Large-scale data processing increases potential impact and exposure.")

    if answers.get("third_party_data"):
        score += 1
        reasons.append("Uses third-party data where quality and legality may be harder to control.")

    return score, reasons


def _model_risk(answers):
    score = 0
    reasons = []

    model_type = (answers.get("model_type") or "").lower()

    if model_type in ("rules", "simple_linear"):
        score += 1
        reasons.append("Model is relatively simple / interpretable (rules or simple linear).")
    elif model_type in ("tree_ensemble", "random_forest", "gradient_boosting"):
        score += 3
        reasons.append("Tree ensemble models are more complex and less transparent by default.")
    elif model_type in ("neural_network", "deep_learning"):
        score += 4
        reasons.append("Neural networks / deep learning are highly complex and opaque.")
    elif model_type == "external_api":
        score += 4
        reasons.append("Relies on external / third-party AI API, harder to fully audit and control.")
    else:
        score += 2
        reasons.append("Model type unspecified / generic; assuming moderate complexity.")

    if answers.get("uses_generative_ai"):
        score += 2
        reasons.append("Uses generative AI (LLMs, generative models) which are harder to predict and control.")

    if answers.get("online_learning"):
        score += 2
        reasons.append("Model adapts over time (online learning), increasing governance complexity.")

    return score, reasons


def _deployment_risk(answers):
    score = 0
    reasons = []

    if answers.get("fully_automated"):
        score += 3
        reasons.append("Decisions are fully automated without mandatory human confirmation.")

    if answers.get("affects_legal_rights"):
        score += 4
        reasons.append("System affects individuals' legal or economic rights (e.g., credit, jobs, housing).")

    if answers.get("safety_critical"):
        score += 4
        reasons.append("System is safety-critical (e.g., health, physical safety, critical infrastructure).")

    if answers.get("real_time"):
        score += 1
        reasons.append("System is used in real-time, leaving limited time for review or intervention.")

    if answers.get("eu_users"):
        score += 1
        reasons.append("System impacts users in the EU, clearly bringing it into scope of EU AI regulation.")

    return score, reasons


def _governance_score(answers):
    """
    Governance reduces risk: higher governance -> more subtraction.
    """
    score = 0
    reasons = []

    if answers.get("has_human_oversight"):
        score += 3
        reasons.append("Human-in-the-loop oversight is in place for important decisions.")

    if answers.get("has_logging"):
        score += 2
        reasons.append("Technical logging and audit trail are implemented.")

    if answers.get("has_documentation"):
        score += 2
        reasons.append("Documentation exists for data, model, and limitations.")

    if answers.get("has_monitoring"):
        score += 2
        reasons.append("Ongoing monitoring (performance, drift, incidents) is in place.")

    return score, reasons


def _map_score_to_level(total_score: int):
    if total_score <= 4:
        return "Low Risk"
    elif total_score <= 9:
        return "Limited Risk"
    elif total_score <= 14:
        return "High Risk"
    else:
        return "Very High Risk"


def compute_risk_score(answers: dict):
    """
    Main entry point.

    answers is a dict expected to contain keys:
      - domain: one of the supported domain strings
      - uses_personal_data: bool
      - uses_special_categories: bool
      - large_scale_data: bool
      - third_party_data: bool
      - model_type: "rules"/"simple_linear"/"tree_ensemble"/"neural_network"/"external_api"/...
      - uses_generative_ai: bool
      - online_learning: bool
      - fully_automated: bool
      - affects_legal_rights: bool
      - safety_critical: bool
      - real_time: bool
      - eu_users: bool
      - has_human_oversight: bool
      - has_logging: bool
      - has_documentation: bool
      - has_monitoring: bool
    """
    domain_score, domain_reasons = _domain_risk(answers.get("domain"))
    data_score, data_reasons = _data_risk(answers)
    model_score, model_reasons = _model_risk(answers)
    deploy_score, deploy_reasons = _deployment_risk(answers)
    gov_score, gov_reasons = _governance_score(answers)

    raw_score = domain_score + data_score + model_score + deploy_score
    total_score = raw_score - gov_score
    if total_score < 0:
        total_score = 0  # clamp at 0 for sanity

    level = _map_score_to_level(total_score)

    reasons_by_pillar = {
        "domain": domain_reasons,
        "data": data_reasons,
        "model": model_reasons,
        "deployment": deploy_reasons,
        "governance": gov_reasons
    }

    breakdown = {
        "domain_score": domain_score,
        "data_score": data_score,
        "model_score": model_score,
        "deployment_score": deploy_score,
        "governance_score": gov_score,
        "raw_score": raw_score,
        "total_score": total_score
    }

    flat_reasons = domain_reasons + data_reasons + model_reasons + deploy_reasons
    flat_reasons += [f"Governance: {r}" for r in gov_reasons]

    return {
        "score": total_score,
        "level": level,
        "breakdown": breakdown,
        "reasons_by_pillar": reasons_by_pillar,
        "reasons": flat_reasons
    }

    