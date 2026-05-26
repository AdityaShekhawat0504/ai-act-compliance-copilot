"""Pydantic v2 models encoding all nine items of Annex IV as typed fields with provenance metadata.

The foundation of the entire pipeline — every other module reads from or writes to
these models. Each field carries a Provenance object tracking source artifact,
locator, extraction timestamp, and model used. Fields without provenance are
treated as gaps, not content.

See CLAUDE.md: "Annex IV schema (src/annex_iv/schema.py)"
"""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, model_validator

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Tracks where a field value was extracted from."""

    source_artifact: str
    """e.g. 'huggingface://google/gemma-2b'"""

    locator: str
    """e.g. 'README.md#training-data'"""

    extracted_at: datetime

    extraction_model: str
    """e.g. 'gpt-4o-mini-2024-07-18'"""

    confidence: float | None = None


# ---------------------------------------------------------------------------
# DocumentedField — generic wrapper for every schema field
# ---------------------------------------------------------------------------


class DocumentedField(BaseModel, Generic[T]):
    """Wraps every Annex IV field with value, provenance, and status.

    - status="filled": value and provenance must both be present.
    - status="unavailable": value and provenance may be None.
    - status="pending": extraction has not yet been attempted.
    """

    value: T | None = None
    provenance: Provenance | None = None
    status: Literal["filled", "unavailable", "pending"] = "pending"

    @model_validator(mode="after")
    def _filled_requires_provenance(self) -> DocumentedField[T]:
        if self.status == "filled" and self.provenance is None:
            raise ValueError(
                "DocumentedField with status='filled' requires provenance"
            )
        return self


# ---------------------------------------------------------------------------
# Item 1 — General description of the AI system
# ---------------------------------------------------------------------------


class AnnexIVItem1General(BaseModel):
    """Annex IV, item 1: A general description of the AI system."""

    intended_purpose: DocumentedField[str] = DocumentedField()
    """1(a): its intended purpose, the name and address of the provider and,
    if applicable, of the authorised representative."""

    provider_name: DocumentedField[str] = DocumentedField()
    """1(a): name of the provider."""

    provider_address: DocumentedField[str] = DocumentedField()
    """1(a): address of the provider."""

    authorised_representative: DocumentedField[str] = DocumentedField()
    """1(a): if applicable, name and address of the authorised representative."""

    system_version: DocumentedField[str] = DocumentedField()
    """1(b): the version or versions of the AI system to which the technical
    documentation refers."""

    software_hardware_versions: DocumentedField[str] = DocumentedField()
    """1(c): for software, the software versions or, for hardware, the hardware
    versions in which the AI system is embedded."""

    distribution_forms: DocumentedField[str] = DocumentedField()
    """1(d): a description of any forms in which the AI system is placed on
    the market or put into service, such as software packages embedded into
    hardware, downloads, or APIs."""

    hardware_requirements: DocumentedField[str] = DocumentedField()
    """1(e): a description of the hardware on which the AI system is intended
    to run."""

    product_images: DocumentedField[str] = DocumentedField()
    """1(f): where the AI system is a component of products, photographs or
    illustrations showing external features, marking and internal layout of
    those products."""

    user_interface_description: DocumentedField[str] = DocumentedField()
    """1(g): a basic description of the user interface provided to the deployer."""

    instructions_of_use: DocumentedField[str] = DocumentedField()
    """1(h): instructions of use for the deployer and, where applicable,
    installation instructions."""


# ---------------------------------------------------------------------------
# Item 2 — Development process
# ---------------------------------------------------------------------------


class AnnexIVItem2Development(BaseModel):
    """Annex IV, item 2: A detailed description of the elements of the AI
    system and of the process for its development."""

    development_methods: DocumentedField[str] = DocumentedField()
    """2(a): the methods and steps performed for the development of the AI
    system, including, where relevant, recourse to pre-trained systems or
    tools provided by third parties and how those have been used, integrated
    or modified by the provider."""

    design_specifications: DocumentedField[str] = DocumentedField()
    """2(b): the design specifications of the system, namely the general logic
    of the AI system and of the algorithms; the key design choices including
    the rationale and assumptions made, also with regard to persons or groups
    of persons on which the system is intended to be used; the main
    classification choices; what the system is designed to optimise for, and
    the relevance of the different parameters; the description of the expected
    output and output quality of the system; the decisions about any possible
    trade-off made regarding the technical solutions adopted to comply with
    the requirements set out in Chapter III, Section 2."""

    system_architecture: DocumentedField[str] = DocumentedField()
    """2(c): the description of the system architecture explaining how software
    components build on or feed into each other and integrate into the overall
    processing; the computational resources used to develop, train, test and
    validate the AI system."""

    data_requirements: DocumentedField[str] = DocumentedField()
    """2(d): where relevant, the data requirements in terms of datasheets
    describing the training methodologies and techniques and the training data
    sets used, including a general description of these data sets, information
    on their provenance, scope and main characteristics; how the data was
    obtained and selected; labelling procedures (e.g. for supervised learning),
    data cleaning methodologies (e.g. outliers detection)."""

    human_oversight_assessment: DocumentedField[str] = DocumentedField()
    """2(e): assessment of the human oversight measures needed in accordance
    with Article 14, including an assessment of the technical measures needed
    to facilitate the interpretation of the outputs of AI systems by the
    deployers, in accordance with Article 13(3), point (d)."""

    predetermined_changes: DocumentedField[str] = DocumentedField()
    """2(f): where applicable, a detailed description of pre-determined changes
    to the AI system and its performance, together with all the relevant
    information related to the technical solutions adopted to ensure continuous
    compliance of the AI system with the relevant requirements set out in
    Chapter III, Section 2."""

    validation_testing: DocumentedField[str] = DocumentedField()
    """2(g): the validation and testing procedures used, including information
    about the validation and testing data used and their main characteristics;
    metrics used to measure accuracy, robustness and compliance with other
    relevant requirements set out in Chapter III, Section 2 as well as
    potentially discriminatory impacts; test logs and all test reports dated
    and signed by the responsible persons, including with regard to
    pre-determined changes as referred to in point (f)."""

    cybersecurity_measures: DocumentedField[str] = DocumentedField()
    """2(h): cybersecurity measures put in place."""


# ---------------------------------------------------------------------------
# Item 3 — Monitoring, functioning and control
# ---------------------------------------------------------------------------


class AnnexIVItem3MonitoringControl(BaseModel):
    """Annex IV, item 3: Detailed information about the monitoring, functioning
    and control of the AI system."""

    capabilities_limitations: DocumentedField[str] = DocumentedField()
    """3(a): its capabilities and limitations in performance, including the
    degrees of accuracy for specific persons or groups of persons on which the
    system is intended to be used and the overall expected level of accuracy
    in relation to its intended purpose."""

    unintended_outcomes_risks: DocumentedField[str] = DocumentedField()
    """3(b): the foreseeable unintended outcomes and sources of risks to health
    and safety, fundamental rights and discrimination in view of the intended
    purpose of the AI system."""

    human_oversight_measures: DocumentedField[str] = DocumentedField()
    """3(c): the human oversight measures needed in accordance with Article 14,
    including the technical measures put in place to facilitate the
    interpretation of the outputs of AI systems by the deployers."""

    input_data_specifications: DocumentedField[str] = DocumentedField()
    """3(d): specifications on input data, as appropriate."""


# ---------------------------------------------------------------------------
# Item 4 — Performance metrics
# ---------------------------------------------------------------------------


class AnnexIVItem4PerformanceMetrics(BaseModel):
    """Annex IV, item 4: A description of the appropriateness of the
    performance metrics for the specific AI system."""

    metrics_appropriateness: DocumentedField[str] = DocumentedField()
    """4: a description of the appropriateness of the performance metrics
    for the specific AI system."""


# ---------------------------------------------------------------------------
# Item 5 — Risk management system
# ---------------------------------------------------------------------------


class AnnexIVItem5RiskManagement(BaseModel):
    """Annex IV, item 5: A detailed description of the risk management system
    in accordance with Article 9."""

    risk_management_description: DocumentedField[str] = DocumentedField()
    """5: a detailed description of the risk management system in accordance
    with Article 9."""


# ---------------------------------------------------------------------------
# Item 6 — Lifecycle changes
# ---------------------------------------------------------------------------


class AnnexIVItem6LifecycleChanges(BaseModel):
    """Annex IV, item 6: A description of relevant changes made to the system
    through its lifecycle."""

    lifecycle_changes: DocumentedField[str] = DocumentedField()
    """6: a description of relevant changes made to the system through its
    lifecycle."""


# ---------------------------------------------------------------------------
# Item 7 — Harmonised standards
# ---------------------------------------------------------------------------


class AnnexIVItem7HarmonisedStandards(BaseModel):
    """Annex IV, item 7: A list of the harmonised standards applied in full or
    in part, or a description of alternative solutions adopted."""

    harmonised_standards: DocumentedField[str] = DocumentedField()
    """7: a list of the harmonised standards applied in full or in part the
    references of which have been published in the Official Journal of the
    European Union; where no such harmonised standards have been applied, a
    detailed description of the solutions adopted to meet the requirements
    set out in Chapter III, Section 2, including a list of other relevant
    standards and technical specifications applied."""


# ---------------------------------------------------------------------------
# Item 8 — EU declaration of conformity
# ---------------------------------------------------------------------------


class AnnexIVItem8DeclarationOfConformity(BaseModel):
    """Annex IV, item 8: A copy of the EU declaration of conformity referred
    to in Article 47."""

    declaration_of_conformity: DocumentedField[str] = DocumentedField()
    """8: a copy of the EU declaration of conformity referred to in
    Article 47."""


# ---------------------------------------------------------------------------
# Item 9 — Post-market monitoring
# ---------------------------------------------------------------------------


class AnnexIVItem9PostMarketMonitoring(BaseModel):
    """Annex IV, item 9: A detailed description of the system in place to
    evaluate the AI system performance in the post-market phase."""

    post_market_monitoring: DocumentedField[str] = DocumentedField()
    """9: a detailed description of the system in place to evaluate the AI
    system performance in the post-market phase in accordance with Article 72,
    including the post-market monitoring plan referred to in
    Article 72(3)."""


# ---------------------------------------------------------------------------
# Top-level document
# ---------------------------------------------------------------------------


class AnnexIVDocument(BaseModel):
    """Complete Annex IV technical documentation package.

    Composes all nine items of Annex IV (Regulation (EU) 2024/1689) plus
    document-level metadata.
    """

    created_at: datetime | None = None
    last_updated: datetime | None = None
    generator_version: str | None = None
    target_system_name: str | None = None

    item_1_general: AnnexIVItem1General = AnnexIVItem1General()
    item_2_development: AnnexIVItem2Development = AnnexIVItem2Development()
    item_3_monitoring_control: AnnexIVItem3MonitoringControl = (
        AnnexIVItem3MonitoringControl()
    )
    item_4_performance_metrics: AnnexIVItem4PerformanceMetrics = (
        AnnexIVItem4PerformanceMetrics()
    )
    item_5_risk_management: AnnexIVItem5RiskManagement = (
        AnnexIVItem5RiskManagement()
    )
    item_6_lifecycle_changes: AnnexIVItem6LifecycleChanges = (
        AnnexIVItem6LifecycleChanges()
    )
    item_7_harmonised_standards: AnnexIVItem7HarmonisedStandards = (
        AnnexIVItem7HarmonisedStandards()
    )
    item_8_declaration_of_conformity: AnnexIVItem8DeclarationOfConformity = (
        AnnexIVItem8DeclarationOfConformity()
    )
    item_9_post_market_monitoring: AnnexIVItem9PostMarketMonitoring = (
        AnnexIVItem9PostMarketMonitoring()
    )
