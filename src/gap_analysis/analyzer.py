"""Gap analysis for unfilled Annex IV fields.

For each Annex IV field whose ``DocumentedField`` is not in status ``filled``,
surface what is missing, which Annex IV item it maps to, the verbatim
regulation clause that defines it, and the artifact type that would typically
fill it. Pure deterministic analysis over an
:class:`annex_iv.schema.AnnexIVDocument` — no LLM calls.

See CLAUDE.md: "Gap analysis (src/gap_analysis/)".
"""

from __future__ import annotations

from typing import Iterator, Literal

from pydantic import BaseModel

from annex_iv.schema import AnnexIVDocument, DocumentedField


# ---------------------------------------------------------------------------
# Item mapping
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


# ---------------------------------------------------------------------------
# FIELD_REGULATION_TEXT — verbatim Annex IV clauses per (item, field)
# ---------------------------------------------------------------------------


FIELD_REGULATION_TEXT: dict[tuple[int, str], str] = {
    # Item 1 — General description of the AI system
    (1, "intended_purpose"): (
        "1(a): its intended purpose, the name and address of the provider "
        "and, if applicable, of the authorised representative."
    ),
    (1, "provider_name"): "1(a): name of the provider.",
    (1, "provider_address"): "1(a): address of the provider.",
    (1, "authorised_representative"): (
        "1(a): if applicable, name and address of the authorised "
        "representative."
    ),
    (1, "system_version"): (
        "1(b): the version or versions of the AI system to which the "
        "technical documentation refers."
    ),
    (1, "software_hardware_versions"): (
        "1(c): for software, the software versions or, for hardware, the "
        "hardware versions in which the AI system is embedded."
    ),
    (1, "distribution_forms"): (
        "1(d): a description of any forms in which the AI system is placed "
        "on the market or put into service, such as software packages "
        "embedded into hardware, downloads, or APIs."
    ),
    (1, "hardware_requirements"): (
        "1(e): a description of the hardware on which the AI system is "
        "intended to run."
    ),
    (1, "product_images"): (
        "1(f): where the AI system is a component of products, photographs "
        "or illustrations showing external features, marking and internal "
        "layout of those products."
    ),
    (1, "user_interface_description"): (
        "1(g): a basic description of the user interface provided to the "
        "deployer."
    ),
    (1, "instructions_of_use"): (
        "1(h): instructions of use for the deployer and, where applicable, "
        "installation instructions."
    ),
    # Item 2 — Development process
    (2, "development_methods"): (
        "2(a): the methods and steps performed for the development of the AI "
        "system, including, where relevant, recourse to pre-trained systems "
        "or tools provided by third parties and how those have been used, "
        "integrated or modified by the provider."
    ),
    (2, "design_specifications"): (
        "2(b): the design specifications of the system, namely the general "
        "logic of the AI system and of the algorithms; the key design "
        "choices including the rationale and assumptions made; the main "
        "classification choices; what the system is designed to optimise "
        "for; the description of the expected output and output quality of "
        "the system; trade-off decisions made to comply with Chapter III, "
        "Section 2."
    ),
    (2, "system_architecture"): (
        "2(c): the description of the system architecture explaining how "
        "software components build on or feed into each other and integrate "
        "into the overall processing; the computational resources used to "
        "develop, train, test and validate the AI system."
    ),
    (2, "data_requirements"): (
        "2(d): where relevant, the data requirements in terms of datasheets "
        "describing the training methodologies and techniques and the "
        "training data sets used, including a general description of these "
        "data sets, information on their provenance, scope and main "
        "characteristics; how the data was obtained and selected; labelling "
        "procedures and data cleaning methodologies."
    ),
    (2, "human_oversight_assessment"): (
        "2(e): assessment of the human oversight measures needed in "
        "accordance with Article 14, including an assessment of the "
        "technical measures needed to facilitate the interpretation of the "
        "outputs of AI systems by the deployers, in accordance with "
        "Article 13(3), point (d)."
    ),
    (2, "predetermined_changes"): (
        "2(f): where applicable, a detailed description of pre-determined "
        "changes to the AI system and its performance, together with all "
        "the relevant information related to the technical solutions "
        "adopted to ensure continuous compliance with the relevant "
        "requirements set out in Chapter III, Section 2."
    ),
    (2, "validation_testing"): (
        "2(g): the validation and testing procedures used, including "
        "information about the validation and testing data used and their "
        "main characteristics; metrics used to measure accuracy, robustness "
        "and compliance with other relevant requirements set out in "
        "Chapter III, Section 2 as well as potentially discriminatory "
        "impacts; test logs and all test reports dated and signed by the "
        "responsible persons."
    ),
    (2, "cybersecurity_measures"): "2(h): cybersecurity measures put in place.",
    # Item 3 — Monitoring, functioning and control
    (3, "capabilities_limitations"): (
        "3(a): its capabilities and limitations in performance, including "
        "the degrees of accuracy for specific persons or groups of persons "
        "on which the system is intended to be used and the overall "
        "expected level of accuracy in relation to its intended purpose."
    ),
    (3, "unintended_outcomes_risks"): (
        "3(b): the foreseeable unintended outcomes and sources of risks to "
        "health and safety, fundamental rights and discrimination in view "
        "of the intended purpose of the AI system."
    ),
    (3, "human_oversight_measures"): (
        "3(c): the human oversight measures needed in accordance with "
        "Article 14, including the technical measures put in place to "
        "facilitate the interpretation of the outputs of AI systems by the "
        "deployers."
    ),
    (3, "input_data_specifications"): (
        "3(d): specifications on input data, as appropriate."
    ),
    # Item 4 — Performance metrics
    (4, "metrics_appropriateness"): (
        "4: a description of the appropriateness of the performance metrics "
        "for the specific AI system."
    ),
    # Item 5 — Risk management
    (5, "risk_management_description"): (
        "5: a detailed description of the risk management system in "
        "accordance with Article 9."
    ),
    # Item 6 — Lifecycle changes
    (6, "lifecycle_changes"): (
        "6: a description of relevant changes made to the system through "
        "its lifecycle."
    ),
    # Item 7 — Harmonised standards
    (7, "harmonised_standards"): (
        "7: a list of the harmonised standards applied in full or in part "
        "the references of which have been published in the Official "
        "Journal of the European Union; where no such harmonised standards "
        "have been applied, a detailed description of the solutions adopted "
        "to meet the requirements set out in Chapter III, Section 2, "
        "including a list of other relevant standards and technical "
        "specifications applied."
    ),
    # Item 8 — EU declaration of conformity
    (8, "declaration_of_conformity"): (
        "8: a copy of the EU declaration of conformity referred to in "
        "Article 47."
    ),
    # Item 9 — Post-market monitoring
    (9, "post_market_monitoring"): (
        "9: a detailed description of the system in place to evaluate the "
        "AI system performance in the post-market phase in accordance with "
        "Article 72, including the post-market monitoring plan referred to "
        "in Article 72(3)."
    ),
}


# ---------------------------------------------------------------------------
# SUGGESTED_ARTIFACT — artifact type that would typically fill each field
# ---------------------------------------------------------------------------


_SUGGESTED_ARTIFACT: dict[tuple[int, str], str] = {
    (1, "intended_purpose"): "model card intended use section",
    (1, "provider_name"): "model card YAML frontmatter or repository owner",
    (1, "provider_address"): (
        "model card contact section or provider organisation registration"
    ),
    (1, "authorised_representative"): (
        "provider's EU regulatory filings or legal documentation"
    ),
    (1, "system_version"): (
        "model card version metadata, git tag, or release notes"
    ),
    (1, "software_hardware_versions"): (
        "model card dependencies section, requirements.txt, or "
        "environment manifest"
    ),
    (1, "distribution_forms"): (
        "model card deployment section, API documentation, or distribution "
        "manifest"
    ),
    (1, "hardware_requirements"): (
        "model card hardware section or inference benchmarks"
    ),
    (1, "product_images"): (
        "product datasheets, marketing materials, or hardware design "
        "documentation"
    ),
    (1, "user_interface_description"): (
        "user-facing documentation or UI screenshots"
    ),
    (1, "instructions_of_use"): (
        "model card usage section or end-user manual"
    ),
    (2, "development_methods"): (
        "model card training section or training code repository"
    ),
    (2, "design_specifications"): (
        "model card architecture section or design document"
    ),
    (2, "system_architecture"): (
        "architecture diagram, model card architecture section, or training "
        "code"
    ),
    (2, "data_requirements"): (
        "dataset card or training data documentation"
    ),
    (2, "human_oversight_assessment"): (
        "Article 14 human oversight assessment document"
    ),
    (2, "predetermined_changes"): (
        "model lifecycle plan or change management documentation"
    ),
    (2, "validation_testing"): (
        "MLflow run, evaluation report, or model card evaluation section"
    ),
    (2, "cybersecurity_measures"): (
        "threat model document or security review"
    ),
    (3, "capabilities_limitations"): (
        "model card limitations section or evaluation report"
    ),
    (3, "unintended_outcomes_risks"): (
        "model card bias and risks section or fundamental rights impact "
        "assessment"
    ),
    (3, "human_oversight_measures"): (
        "deployment runbook or Article 14 oversight plan"
    ),
    (3, "input_data_specifications"): (
        "model card input specification or API schema"
    ),
    (4, "metrics_appropriateness"): (
        "evaluation report or model card metrics justification"
    ),
    (5, "risk_management_description"): (
        "Article 9 risk management system documentation"
    ),
    (6, "lifecycle_changes"): "model changelog or version history",
    (7, "harmonised_standards"): (
        "compliance documentation listing applied standards (e.g. "
        "ISO/IEC 42001) or rationale where none apply"
    ),
    (8, "declaration_of_conformity"): (
        "EU declaration of conformity per Article 47"
    ),
    (9, "post_market_monitoring"): (
        "post-market monitoring plan per Article 72"
    ),
}


# ---------------------------------------------------------------------------
# _REQUIRED — required vs conditional per (item, field)
# ---------------------------------------------------------------------------
#
# A field is marked required=False when the regulation conditions it on
# applicability ("if applicable", "where applicable", "where relevant",
# "as appropriate"). Single-field items (4-9) are all required=True.


_REQUIRED: dict[tuple[int, str], bool] = {
    # Item 1
    (1, "intended_purpose"): True,
    (1, "provider_name"): True,
    (1, "provider_address"): True,
    (1, "authorised_representative"): False,  # "if applicable"
    (1, "system_version"): True,
    (1, "software_hardware_versions"): True,
    (1, "distribution_forms"): True,
    (1, "hardware_requirements"): True,
    (1, "product_images"): False,  # "where the AI system is a component of products"
    (1, "user_interface_description"): True,
    (1, "instructions_of_use"): True,
    # Item 2
    (2, "development_methods"): True,
    (2, "design_specifications"): True,
    (2, "system_architecture"): True,
    (2, "data_requirements"): False,  # "where relevant"
    (2, "human_oversight_assessment"): True,
    (2, "predetermined_changes"): False,  # "where applicable"
    (2, "validation_testing"): True,
    (2, "cybersecurity_measures"): True,
    # Item 3
    (3, "capabilities_limitations"): True,
    (3, "unintended_outcomes_risks"): True,
    (3, "human_oversight_measures"): True,
    (3, "input_data_specifications"): False,  # "as appropriate"
    # Items 4-9 are single-field items; all required
    (4, "metrics_appropriateness"): True,
    (5, "risk_management_description"): True,
    (6, "lifecycle_changes"): True,
    (7, "harmonised_standards"): True,
    (8, "declaration_of_conformity"): True,
    (9, "post_market_monitoring"): True,
}


# ---------------------------------------------------------------------------
# Public lookup helpers (functions backed by module-level dicts)
# ---------------------------------------------------------------------------


def suggested_artifact_for(item_number: int, field_name: str) -> str:
    """Return the artifact type that would typically fill ``field_name``.

    Used to populate :attr:`FieldGap.suggested_artifact_type` for unfilled
    Annex IV fields.
    """
    try:
        return _SUGGESTED_ARTIFACT[(item_number, field_name)]
    except KeyError as exc:
        raise KeyError(
            f"No suggested artifact registered for Annex IV item "
            f"{item_number} field {field_name!r}"
        ) from exc


def required_for(item_number: int, field_name: str) -> bool:
    """Return whether ``field_name`` is mandatory under Annex IV.

    ``False`` for fields whose regulation clause conditions them on
    applicability (``where applicable`` / ``where relevant`` /
    ``if applicable`` / ``as appropriate``).
    """
    try:
        return _REQUIRED[(item_number, field_name)]
    except KeyError as exc:
        raise KeyError(
            f"No required-flag registered for Annex IV item "
            f"{item_number} field {field_name!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class FieldGap(BaseModel):
    """A single unfilled Annex IV field surfaced by gap analysis."""

    annex_iv_item: int
    field_name: str
    regulation_text: str
    status: Literal["unavailable", "pending"]
    suggested_artifact_type: str
    required: bool


class GapReport(BaseModel):
    """Structured summary of Annex IV completeness for a document."""

    target_system_name: str | None
    total_fields: int
    filled_fields: int
    unavailable_fields: int
    pending_fields: int
    completeness_score: float
    completeness_required_only: float
    gaps: list[FieldGap]
    summary: str


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _walk_fields(
    document: AnnexIVDocument,
) -> Iterator[tuple[int, str, DocumentedField]]:
    """Yield ``(item_number, field_name, documented_field)`` for every field
    in every Annex IV item of ``document``."""
    for item_number, attr in _ITEM_TO_ATTR.items():
        item_model = getattr(document, attr)
        for field_name in type(item_model).model_fields:
            yield item_number, field_name, getattr(item_model, field_name)


def analyze(document: AnnexIVDocument) -> GapReport:
    """Build a :class:`GapReport` describing the document's completeness.

    Walks every field of every Annex IV item, counts statuses, and emits a
    :class:`FieldGap` for each field whose status is ``pending`` or
    ``unavailable``.
    """
    total = 0
    filled = 0
    unavailable = 0
    pending = 0
    total_required = 0
    filled_required = 0
    gaps: list[FieldGap] = []

    for item_number, field_name, doc_field in _walk_fields(document):
        total += 1
        is_required = required_for(item_number, field_name)
        if is_required:
            total_required += 1

        status = doc_field.status
        if status == "filled":
            filled += 1
            if is_required:
                filled_required += 1
            continue

        if status == "unavailable":
            unavailable += 1
        else:
            pending += 1

        gaps.append(
            FieldGap(
                annex_iv_item=item_number,
                field_name=field_name,
                regulation_text=FIELD_REGULATION_TEXT[(item_number, field_name)],
                status=status,
                suggested_artifact_type=suggested_artifact_for(
                    item_number, field_name
                ),
                required=is_required,
            )
        )

    completeness_score = filled / total if total else 0.0
    completeness_required_only = (
        filled_required / total_required if total_required else 0.0
    )

    gap_item_numbers = sorted({gap.annex_iv_item for gap in gaps})
    if gaps:
        summary = (
            f"Annex IV compliance: {filled}/{total} fields filled "
            f"({completeness_score:.0%}). "
            f"Required-field completeness: {completeness_required_only:.0%}. "
            f"{len(gaps)} gaps identified across items {gap_item_numbers}."
        )
    else:
        summary = (
            f"Annex IV compliance: {filled}/{total} fields filled "
            f"({completeness_score:.0%}). "
            f"Required-field completeness: {completeness_required_only:.0%}. "
            f"0 gaps identified."
        )

    return GapReport(
        target_system_name=document.target_system_name,
        total_fields=total,
        filled_fields=filled,
        unavailable_fields=unavailable,
        pending_fields=pending,
        completeness_score=completeness_score,
        completeness_required_only=completeness_required_only,
        gaps=gaps,
        summary=summary,
    )
