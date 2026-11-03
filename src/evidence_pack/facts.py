"""Facts about a company and its AI feature, and where each fact came from.

A fact is either ``known`` (it has a value and a source) or ``unknown``.
Facts extracted from documents are only ``known`` when their supporting quote
was found verbatim in the source by code (see ``documents.verify_quote``).
Questionnaire answers are self-declared; an evidence quote attached to one is
verified the same way, and dropped (with a gap) if it is not in the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

from pydantic import BaseModel, model_validator

FactValue = Union[bool, str, list[str], None]
FactKind = Literal["bool", "text", "list"]


@dataclass(frozen=True)
class FactSpec:
    name: str
    kind: FactKind
    question: str
    group: str
    choices_from: str | None = None  # catalogue key in rules.yaml for list facts
    help: str = ""
    examples: tuple[str, ...] = field(default_factory=tuple)


FACT_SPECS: list[FactSpec] = [
    # --- Company and system ------------------------------------------------
    FactSpec("company_name", "text", "Company (provider or deployer) name", "System"),
    FactSpec("system_name", "text", "Name of the AI feature or system", "System"),
    FactSpec("system_description", "text", "What does the AI feature do and what is its intended purpose?", "System"),
    FactSpec("provider_address", "text", "Registered address of the company", "System"),
    FactSpec("system_version", "text", "Version of the AI feature", "System"),
    FactSpec("distribution_form", "text", "How is it delivered to customers (SaaS web app, API, on-premise)?", "System"),
    FactSpec("ai_governance_contact", "text", "Contact for AI governance questions", "System"),
    # --- Role ----------------------------------------------------------------
    FactSpec("places_on_market_under_own_name", "bool", "Do you offer the AI feature to customers under your own name or brand?", "Role",
             help="Developing an AI system (or having it developed) and placing it on the market under your own name makes you a provider (Art. 3(3))."),
    FactSpec("rebrands_or_substantially_modifies", "bool", "Do you put your name on, substantially modify, or change the purpose of someone else's high-risk AI system?", "Role",
             help="Art. 25(1) turns a distributor, importer or deployer into a provider in these cases."),
    FactSpec("uses_ai_internally", "bool", "Does your company itself use the AI system in its own professional operations?", "Role",
             help="Using an AI system under your authority makes you a deployer (Art. 3(4))."),
    # --- Risk ----------------------------------------------------------------
    FactSpec("prohibited_practices", "list", "Does the system do any of these prohibited practices? (empty list = none)", "Risk",
             choices_from="prohibited_practices"),
    FactSpec("annex_iii_areas", "list", "Is the system used in any of these Annex III areas? (empty list = none)", "Risk",
             choices_from="annex_iii_areas"),
    FactSpec("annex_iii_derogation_claimed", "bool", "Annex III only: does the system only perform a narrow procedural or preparatory task without materially influencing decisions (Art. 6(3))?", "Risk"),
    FactSpec("performs_profiling", "bool", "Does the system profile natural persons (automated evaluation of personal aspects)?", "Risk"),
    FactSpec("safety_component_annex_i_product", "bool", "Is the AI a safety component of, or itself, a product under Annex I legislation (e.g. medical devices, machinery, toys)?", "Risk"),
    # --- Transparency --------------------------------------------------------
    FactSpec("interacts_with_people", "bool", "Does the system interact directly with people (chatbot, voice assistant)?", "Transparency"),
    FactSpec("generates_synthetic_content", "bool", "Does it generate synthetic text, images, audio or video?", "Transparency"),
    FactSpec("emotion_recognition", "bool", "Does it recognise or infer emotions from biometric data?", "Transparency"),
    FactSpec("biometric_categorisation", "bool", "Does it assign people to categories based on biometric data?", "Transparency"),
    FactSpec("generates_deepfakes", "bool", "Does it generate or manipulate image, audio or video resembling real people, places or events (deepfakes)?", "Transparency"),
    FactSpec("publishes_public_interest_text", "bool", "Is AI-generated text published to inform the public on matters of public interest?", "Transparency"),
    FactSpec("human_editorial_review", "bool", "Is that text reviewed by a human who holds editorial responsibility?", "Transparency"),
    FactSpec("on_eu_market_before_2026_08_02", "bool", "Was the generative feature already on the EU market before 2 August 2026?", "Transparency"),
    # --- GPAI ----------------------------------------------------------------
    FactSpec("uses_gpai_model", "bool", "Does the feature use a third-party general-purpose AI model (e.g. a large language model API)?", "GPAI"),
    FactSpec("gpai_model_names", "text", "Which general-purpose AI models are used?", "GPAI"),
    FactSpec("trains_own_gpai_model", "bool", "Do you train and release your own general-purpose AI model?", "GPAI"),
    # --- Evidence you already have ------------------------------------------
    FactSpec("ai_literacy_programme", "bool", "Do staff who build or use the system get AI literacy training?", "Evidence"),
    FactSpec("discloses_ai_interaction", "bool", "Does the product tell users they are interacting with AI?", "Evidence"),
    FactSpec("marks_output_machine_readable", "bool", "Is generated output marked in a machine-readable way (metadata, watermark, C2PA)?", "Evidence"),
    FactSpec("informs_affected_persons", "bool", "Are people exposed to emotion recognition or biometric categorisation informed?", "Evidence"),
    FactSpec("labels_ai_generated_content", "bool", "Is published AI-generated content visibly labelled?", "Evidence"),
    FactSpec("has_gpai_provider_documentation", "bool", "Do you hold the GPAI model provider's documentation for downstream providers?", "Evidence"),
    FactSpec("trains_on_customer_data", "bool", "Is customer data used to train or fine-tune models?", "Evidence"),
    FactSpec("training_data_description", "text", "Describe the training, validation and test data", "Evidence"),
    FactSpec("known_limitations", "text", "Known limitations and accuracy of the system", "Evidence"),
    FactSpec("performance_metrics", "text", "Which performance metrics do you measure and why?", "Evidence"),
    FactSpec("human_oversight_description", "text", "How do humans oversee or override the system's output?", "Evidence"),
    FactSpec("risk_management_description", "text", "Describe your AI risk management process", "Evidence"),
    FactSpec("has_event_logging", "bool", "Does the system automatically log events (inputs, outputs, decisions)?", "Evidence"),
    FactSpec("has_quality_management_system", "bool", "Do you run a documented quality management system for the AI feature?", "Evidence"),
    FactSpec("post_market_monitoring_description", "text", "How do you monitor the system after release?", "Evidence"),
]

SPECS_BY_NAME: dict[str, FactSpec] = {s.name: s for s in FACT_SPECS}


class Fact(BaseModel):
    """One fact with its value and evidence."""

    name: str
    value: FactValue = None
    status: Literal["known", "unknown"] = "unknown"
    origin: Literal["document", "questionnaire"] | None = None
    source: str | None = None
    """File name for document facts or evidence quotes; 'questionnaire' otherwise."""
    quote: str | None = None
    """Verbatim text from ``source`` supporting the value; verified by code."""
    quote_verified: bool = False

    @model_validator(mode="after")
    def _invariants(self) -> Fact:
        if self.status == "known" and self.value is None:
            raise ValueError(f"known fact {self.name!r} needs a value")
        if self.status == "known" and self.origin == "document" and not self.quote_verified:
            raise ValueError(
                f"document fact {self.name!r} cannot be known without a verified quote"
            )
        if self.quote_verified and not self.quote:
            raise ValueError(f"fact {self.name!r} marked verified without a quote")
        return self

    @property
    def is_known(self) -> bool:
        return self.status == "known"

    @property
    def truthy(self) -> bool:
        """Known and affirmative: True, non-empty text or a non-empty list."""
        if not self.is_known:
            return False
        if isinstance(self.value, bool):
            return self.value
        return bool(self.value)


class FactSet(BaseModel):
    """All facts for one system, keyed by name. Missing names are unknown."""

    facts: dict[str, Fact] = {}
    notes: list[str] = []
    """Messages about dropped evidence and conflicts, shown in the gap list."""

    def get(self, name: str) -> Fact:
        if name not in SPECS_BY_NAME:
            raise KeyError(f"unknown fact name {name!r}")
        return self.facts.get(name) or Fact(name=name)

    def value(self, name: str) -> FactValue:
        fact = self.get(name)
        return fact.value if fact.is_known else None

    def text(self, name: str, default: str = "") -> str:
        v = self.value(name)
        if v is None:
            return default
        if isinstance(v, list):
            return ", ".join(v)
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)


def coerce_value(spec: FactSpec, raw: object) -> FactValue:
    """Convert a questionnaire or LLM answer into the fact's type.

    Returns None for "unknown"-style answers. Raises ValueError for answers
    that cannot be read as the expected type.
    """
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "unknown", "n/a", "?", "null", "none given"}:
        return None
    if spec.kind == "bool":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in {"yes", "true", "y", "1"}:
            return True
        if s in {"no", "false", "n", "0"}:
            return False
        raise ValueError(f"{spec.name}: expected yes/no, got {raw!r}")
    if spec.kind == "list":
        if isinstance(raw, str):
            if raw.strip().lower() in {"none", "[]"}:
                return []
            items = [x.strip() for x in raw.split(",")]
        elif isinstance(raw, (list, tuple)):
            items = [str(x).strip() for x in raw]
        else:
            raise ValueError(f"{spec.name}: expected a list, got {raw!r}")
        return [x for x in items if x]
    return str(raw).strip()
