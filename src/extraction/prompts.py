"""Per-item extraction prompts for Annex IV fields.

Each prompt targets a specific Annex IV item (1-9) and is designed to extract
structured values from normalised artifacts. Prompts enforce grounding
requirements — the LLM must return "unavailable" rather than guess.

Prompts are constants composed at module load time from the regulation text
in src/annex_iv/reference/item_N.md and a per-item field specification.
Artifact content is NOT injected at module level — that happens inside the
extraction call.

See CLAUDE.md: "No bare LLM calls" and "No fabrication"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from annex_iv.schema import (
    AnnexIVItem1General,
    AnnexIVItem2Development,
    AnnexIVItem3MonitoringControl,
    AnnexIVItem4PerformanceMetrics,
    AnnexIVItem5RiskManagement,
    AnnexIVItem6LifecycleChanges,
    AnnexIVItem7HarmonisedStandards,
    AnnexIVItem8DeclarationOfConformity,
    AnnexIVItem9PostMarketMonitoring,
)

_REFERENCE_DIR = Path(__file__).parent.parent / "annex_iv" / "reference"


def _load_reference(item_number: int) -> str:
    return (_REFERENCE_DIR / f"item_{item_number}.md").read_text().strip()


_BASE_INSTRUCTIONS = """\
You extract structured information from a model card or MLOps artifact to
populate EU AI Act Annex IV technical documentation. Your output is consumed
by a regulatory compliance pipeline that ships generated text to lawyers and
auditors.

CRITICAL RULES:
1. ONLY use information that is explicitly stated in the source artifact you
   are given. If a field is not directly stated, set value=null and
   found_in_source=false.
2. NEVER fabricate, infer, extrapolate, or guess. Do not fill in plausible
   defaults. Do not use external knowledge about the model or its provider.
3. A second-pass grounding check will cross-reference every populated value
   against the source. Ungrounded values will be silently dropped, so
   over-claiming gives you no benefit — it costs you accuracy.
4. Light paraphrasing of the source is acceptable. Adding new claims is not.
5. If the source mentions only part of what a field asks for, extract the
   part that is stated. Do not invent the rest.
"""


def _compose_prompt(
    item_number: int,
    field_specs: list[tuple[str, str]],
    positive_example: str,
    negative_example: str,
) -> str:
    regulation = _load_reference(item_number)
    field_lines = "\n".join(
        f"- {name}: {desc}" for name, desc in field_specs
    )
    return f"""{_BASE_INSTRUCTIONS}

REGULATION TEXT YOU ARE DOCUMENTING (Annex IV, item {item_number} of Regulation (EU) 2024/1689):

{regulation}

FIELDS TO EXTRACT:
{field_lines}

For each field, return a structured object:
- value: the extracted text (string), or null if not in source
- found_in_source: true if the value is directly stated in the source, false otherwise

POSITIVE EXAMPLE (extraction grounded in source):
{positive_example}

NEGATIVE EXAMPLE (refusal — value not in source):
{negative_example}
"""


# ---------------------------------------------------------------------------
# Item 1 — General description
# ---------------------------------------------------------------------------

_ITEM_1_FIELDS: list[tuple[str, str]] = [
    ("intended_purpose", "1(a): the intended purpose of the AI system"),
    ("provider_name", "1(a): the name of the provider"),
    ("provider_address", "1(a): the address of the provider"),
    (
        "authorised_representative",
        "1(a): name and address of the authorised representative, if applicable",
    ),
    (
        "system_version",
        "1(b): the version or versions of the AI system to which the documentation refers",
    ),
    (
        "software_hardware_versions",
        "1(c): software versions or hardware versions in which the system is embedded",
    ),
    (
        "distribution_forms",
        "1(d): the forms in which the system is placed on the market (embedded, downloads, APIs, etc.)",
    ),
    (
        "hardware_requirements",
        "1(e): the hardware on which the AI system is intended to run",
    ),
    (
        "product_images",
        "1(f): photographs or illustrations of products containing the system, if applicable",
    ),
    (
        "user_interface_description",
        "1(g): a basic description of the deployer-facing user interface",
    ),
    (
        "instructions_of_use",
        "1(h): instructions of use and installation instructions for deployers",
    ),
]

ITEM_1_PROMPT = _compose_prompt(
    1,
    _ITEM_1_FIELDS,
    positive_example=(
        'Source contains: "Gemma is a family of lightweight, state-of-the-art '
        'open models from Google, built from the same research and technology '
        'used to create the Gemini models."\n'
        'For intended_purpose: {"value": "lightweight open model from Google, '
        'built from the same research as the Gemini models", "found_in_source": true}'
    ),
    negative_example=(
        "Source contains no mention of a postal address for the provider.\n"
        'For provider_address: {"value": null, "found_in_source": false}\n'
        'DO NOT WRITE: {"value": "Mountain View, CA (assumed Google '
        'headquarters)", "found_in_source": true}'
    ),
)


# ---------------------------------------------------------------------------
# Item 2 — Development process
# ---------------------------------------------------------------------------

_ITEM_2_FIELDS: list[tuple[str, str]] = [
    (
        "development_methods",
        "2(a): methods and steps of development, including use of pre-trained components or third-party tools",
    ),
    (
        "design_specifications",
        "2(b): general logic of the system and algorithms; key design choices, rationale, classification choices, optimisation targets, expected output",
    ),
    (
        "system_architecture",
        "2(c): system architecture, software component interactions, computational resources for develop/train/test/validate",
    ),
    (
        "data_requirements",
        "2(d): training datasets — provenance, scope, characteristics, selection, labelling, cleaning methodologies",
    ),
    (
        "human_oversight_assessment",
        "2(e): assessment of human oversight measures needed per Article 14",
    ),
    (
        "predetermined_changes",
        "2(f): pre-determined changes to the system and its performance, if applicable",
    ),
    (
        "validation_testing",
        "2(g): validation and testing procedures, validation/testing data, accuracy/robustness metrics, test logs",
    ),
    (
        "cybersecurity_measures",
        "2(h): cybersecurity measures put in place",
    ),
]

ITEM_2_PROMPT = _compose_prompt(
    2,
    _ITEM_2_FIELDS,
    positive_example=(
        'Source contains: "These models were trained on a dataset of text '
        'data that includes a wide variety of sources, totaling 6 trillion '
        'tokens." and a separate section listing web documents, code, and '
        "mathematics as the source mix.\n"
        'For data_requirements: {"value": "trained on a dataset of text data '
        "from a wide variety of sources totaling 6 trillion tokens, including "
        'web documents, code, and mathematics", "found_in_source": true}'
    ),
    negative_example=(
        "Source describes the training data and architecture but says nothing "
        "about cybersecurity controls.\n"
        'For cybersecurity_measures: {"value": null, "found_in_source": false}\n'
        'DO NOT WRITE: {"value": "Standard cybersecurity practices apply", '
        '"found_in_source": true}'
    ),
)


# ---------------------------------------------------------------------------
# Item 3 — Monitoring, functioning and control
# ---------------------------------------------------------------------------

_ITEM_3_FIELDS: list[tuple[str, str]] = [
    (
        "capabilities_limitations",
        "3(a): capabilities and limitations in performance, including accuracy levels overall and for specific groups",
    ),
    (
        "unintended_outcomes_risks",
        "3(b): foreseeable unintended outcomes and sources of risks to health, safety, fundamental rights, and discrimination",
    ),
    (
        "human_oversight_measures",
        "3(c): human oversight measures per Article 14, including technical measures to facilitate interpretation of outputs",
    ),
    (
        "input_data_specifications",
        "3(d): specifications on input data, as appropriate",
    ),
]

ITEM_3_PROMPT = _compose_prompt(
    3,
    _ITEM_3_FIELDS,
    positive_example=(
        'Source contains a "Limitations" section: "These models have certain '
        "limitations that users should be aware of. Open Vocabulary, Open-ended "
        "Generation: The breadth of possible inputs and the open-ended nature "
        'of the responses can lead to undesirable outputs."\n'
        'For capabilities_limitations: {"value": "limitations include open '
        "vocabulary and open-ended generation, where the breadth of possible "
        'inputs can lead to undesirable outputs", "found_in_source": true}'
    ),
    negative_example=(
        "Source describes model performance benchmarks but says nothing about "
        "human oversight or interpretation of outputs.\n"
        'For human_oversight_measures: {"value": null, "found_in_source": false}'
    ),
)


# ---------------------------------------------------------------------------
# Item 4 — Performance metrics
# ---------------------------------------------------------------------------

_ITEM_4_FIELDS: list[tuple[str, str]] = [
    (
        "metrics_appropriateness",
        "4: description of the appropriateness of the chosen performance metrics for this specific AI system",
    ),
]

ITEM_4_PROMPT = _compose_prompt(
    4,
    _ITEM_4_FIELDS,
    positive_example=(
        'Source contains: "We report top-1 and top-5 accuracy on ImageNet-1k. '
        "These metrics are standard for image classification and allow direct "
        'comparison against the original ViT and ResNet baselines."\n'
        'For metrics_appropriateness: {"value": "top-1 and top-5 accuracy on '
        "ImageNet-1k, chosen as standard image-classification metrics that "
        'allow direct comparison against ViT and ResNet baselines", '
        '"found_in_source": true}'
    ),
    negative_example=(
        "Source lists metric values (e.g. accuracy=0.87) but never discusses "
        "why those particular metrics were chosen.\n"
        'For metrics_appropriateness: {"value": null, "found_in_source": false}'
    ),
)


# ---------------------------------------------------------------------------
# Item 5 — Risk management
# ---------------------------------------------------------------------------

_ITEM_5_FIELDS: list[tuple[str, str]] = [
    (
        "risk_management_description",
        "5: detailed description of the risk management system per Article 9",
    ),
]

ITEM_5_PROMPT = _compose_prompt(
    5,
    _ITEM_5_FIELDS,
    positive_example=(
        'Source contains a section titled "Risk evaluation": "Our approach to '
        "evaluating and mitigating the risks of Gemma is structured. For each "
        "issue, we define policies, then evaluate models against those policies "
        'using a combination of human and automated red-teaming."\n'
        'For risk_management_description: {"value": "structured risk approach: '
        "policies are defined per issue, then models are evaluated against those "
        'policies using human and automated red-teaming", "found_in_source": true}'
    ),
    negative_example=(
        "Source describes model performance and training but contains no "
        "section on risk management or Article 9 compliance.\n"
        'For risk_management_description: {"value": null, "found_in_source": false}'
    ),
)


# ---------------------------------------------------------------------------
# Item 6 — Lifecycle changes
# ---------------------------------------------------------------------------

_ITEM_6_FIELDS: list[tuple[str, str]] = [
    (
        "lifecycle_changes",
        "6: relevant changes made to the system through its lifecycle (versions, retraining, updates)",
    ),
]

ITEM_6_PROMPT = _compose_prompt(
    6,
    _ITEM_6_FIELDS,
    positive_example=(
        'Source contains: "v1.0 was released in February 2024. v1.1 was '
        "released in May 2024 with improved instruction-following from "
        'additional RLHF training."\n'
        'For lifecycle_changes: {"value": "v1.0 released February 2024; v1.1 '
        "released May 2024 with improved instruction-following from "
        'additional RLHF training", "found_in_source": true}'
    ),
    negative_example=(
        "Source mentions only the current version of the model with no "
        "history of changes or retraining events.\n"
        'For lifecycle_changes: {"value": null, "found_in_source": false}'
    ),
)


# ---------------------------------------------------------------------------
# Item 7 — Harmonised standards
# ---------------------------------------------------------------------------

_ITEM_7_FIELDS: list[tuple[str, str]] = [
    (
        "harmonised_standards",
        "7: list of harmonised standards applied (or, if none, description of alternative solutions)",
    ),
]

ITEM_7_PROMPT = _compose_prompt(
    7,
    _ITEM_7_FIELDS,
    positive_example=(
        'Source contains: "This system conforms to ISO/IEC 23053:2022 '
        '(Framework for AI systems using ML) and ISO/IEC 42001:2023 (AI '
        'management systems)."\n'
        'For harmonised_standards: {"value": "ISO/IEC 23053:2022 (Framework '
        "for AI systems using ML) and ISO/IEC 42001:2023 (AI management "
        'systems)", "found_in_source": true}'
    ),
    negative_example=(
        "Source mentions general best practices but does not name specific "
        "harmonised standards (ISO, IEC, CEN, CENELEC, ETSI).\n"
        'For harmonised_standards: {"value": null, "found_in_source": false}'
    ),
)


# ---------------------------------------------------------------------------
# Item 8 — EU declaration of conformity
# ---------------------------------------------------------------------------

_ITEM_8_FIELDS: list[tuple[str, str]] = [
    (
        "declaration_of_conformity",
        "8: copy or summary of the EU declaration of conformity per Article 47",
    ),
]

ITEM_8_PROMPT = _compose_prompt(
    8,
    _ITEM_8_FIELDS,
    positive_example=(
        'Source contains: "EU Declaration of Conformity — Provider: Example '
        "Corp; System: CreditModel v3; Conformity assessment per Article 43; "
        'Signed by J. Smith, 2025-03-12."\n'
        'For declaration_of_conformity: {"value": "EU Declaration of '
        "Conformity present: Provider Example Corp, System CreditModel v3, "
        'conformity assessment per Article 43, signed J. Smith 2025-03-12", '
        '"found_in_source": true}'
    ),
    negative_example=(
        "Source contains a model card with no EU declaration of conformity "
        "or any reference to Article 47.\n"
        'For declaration_of_conformity: {"value": null, "found_in_source": false}'
    ),
)


# ---------------------------------------------------------------------------
# Item 9 — Post-market monitoring
# ---------------------------------------------------------------------------

_ITEM_9_FIELDS: list[tuple[str, str]] = [
    (
        "post_market_monitoring",
        "9: detailed description of the post-market monitoring system per Article 72, including the monitoring plan",
    ),
]

ITEM_9_PROMPT = _compose_prompt(
    9,
    _ITEM_9_FIELDS,
    positive_example=(
        'Source contains: "After release, we collect aggregated usage '
        "telemetry, monitor for jailbreak attempts, and run quarterly safety "
        'evaluations on a held-out evaluation set."\n'
        'For post_market_monitoring: {"value": "post-release monitoring '
        "includes aggregated usage telemetry, jailbreak-attempt detection, "
        'and quarterly safety evaluations on a held-out set", '
        '"found_in_source": true}'
    ),
    negative_example=(
        "Source describes initial release but says nothing about ongoing "
        "monitoring or post-market evaluation.\n"
        'For post_market_monitoring: {"value": null, "found_in_source": false}'
    ),
)


# ---------------------------------------------------------------------------
# Item-class -> prompt mapping
# ---------------------------------------------------------------------------

ITEM_PROMPTS: dict[type[Any], str] = {
    AnnexIVItem1General: ITEM_1_PROMPT,
    AnnexIVItem2Development: ITEM_2_PROMPT,
    AnnexIVItem3MonitoringControl: ITEM_3_PROMPT,
    AnnexIVItem4PerformanceMetrics: ITEM_4_PROMPT,
    AnnexIVItem5RiskManagement: ITEM_5_PROMPT,
    AnnexIVItem6LifecycleChanges: ITEM_6_PROMPT,
    AnnexIVItem7HarmonisedStandards: ITEM_7_PROMPT,
    AnnexIVItem8DeclarationOfConformity: ITEM_8_PROMPT,
    AnnexIVItem9PostMarketMonitoring: ITEM_9_PROMPT,
}


# ---------------------------------------------------------------------------
# Grounding validator prompt (Pass 2)
# ---------------------------------------------------------------------------

GROUNDING_VALIDATOR_PROMPT = """\
You are a grounding validator for an EU AI Act Annex IV compliance pipeline.

You receive an EXTRACTED VALUE that an earlier pass claims to have extracted
from a SOURCE TEXT. Your job is to determine whether that value is actually
supported by the source.

DEFINITION OF "GROUNDED":
- The value is grounded if a careful reader of the source would arrive at
  substantially the same statement. Light paraphrasing is acceptable.
- The value is NOT grounded if it introduces claims, details, numbers, names,
  or interpretations that the source does not contain.
- Partial grounding (where some of the value is in the source and some is
  invented) counts as NOT GROUNDED — return false.

OUTPUT:
- grounded (bool): true if supported by source, false otherwise.
- locator (string or null): if grounded, a short reference to where in the
  source it comes from (a section heading, a quoted phrase, or a line range).
  Null if not grounded.
- reasoning (string): a one-sentence explanation of your decision.

Err on the side of dropping anything ambiguous. Downstream this is fed to
auditors — a false positive (claiming grounded when it isn't) is much worse
than a false negative.
"""
