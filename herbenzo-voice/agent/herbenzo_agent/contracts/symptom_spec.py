"""SymptomSpec v1 — the typed handoff contract from Stage F (intake) to Stage A (Recommender).

This module is the single source of truth. The JSON Schema in
`herbenzo-voice/contracts/` is generated from it by `export_schema.py`.

Semantics:
- A list value of `[]` means the user said "none"; a field that is `None` means "not captured".
- Every user-derived value is wrapped in `Captured[T]` with its own confidence, capture mode and
  the transcript turn ids it came from (provenance).
- `confidence_floor` is the minimum confidence over the applicable required slots. Downstream
  stages may lower it, never raise it.

Keep this module pure: no settings, no I/O. Policy values (e.g. adult age) arrive through the
pydantic validation context: `SymptomSpec.model_validate(data, context={"min_adult_age": 18})`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Generic, Literal, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

SPEC_VERSION = "1.0.0"
DEFAULT_MIN_ADULT_AGE = 18
PREGNANCY_AGE_RANGE = (12, 55)

T = TypeVar("T")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CaptureMode(StrEnum):
    stated = "stated"
    confirmed = "confirmed"
    inferred = "inferred"
    declined = "declined"


class Captured(Strict, Generic[T]):
    value: T | None
    confidence: float = Field(ge=0.0, le=1.0)
    capture: CaptureMode
    turn_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _declined_is_empty(self):
        if self.capture is CaptureMode.declined:
            if self.value is not None or self.confidence != 0.0:
                raise ValueError("E010: declined capture requires value=None and confidence=0.0")
        elif self.value is None:
            raise ValueError("E010: a non-declined capture requires a value")
        return self


class DurationUnit(StrEnum):
    hours = "hours"
    days = "days"
    weeks = "weeks"
    months = "months"
    years = "years"


class Duration(Strict):
    value: float = Field(gt=0)
    unit: DurationUnit
    raw_text: str


class BodySystem(StrEnum):
    digestive = "digestive"
    respiratory = "respiratory"
    musculoskeletal = "musculoskeletal"
    skin = "skin"
    sleep_mental = "sleep_mental"
    metabolic = "metabolic"
    womens_health = "womens_health"
    urinary = "urinary"
    ent = "ent"
    general = "general"
    other = "other"


class SymptomDetail(Strict):
    name: Captured[str]
    duration: Captured[Duration]
    severity_0_10: Captured[int] | None = None
    frequency_pattern: Captured[str] | None = None
    location: Captured[str] | None = None
    character: Captured[str] | None = None
    aggravating_factors: Captured[list[str]] | None = None
    relieving_factors: Captured[list[str]] | None = None
    associated_symptoms: Captured[list[str]] | None = None


class Medication(Strict):
    name: str = Field(min_length=1)
    dose_text: str | None = None
    kind: Literal["prescription", "otc", "herbal_or_ayurvedic", "supplement", "unknown"] = "unknown"


class SexAtBirth(StrEnum):
    female = "female"
    male = "male"
    intersex = "intersex"
    prefer_not_to_say = "prefer_not_to_say"


class PregnancyStatus(StrEnum):
    pregnant = "pregnant"
    breastfeeding = "breastfeeding"
    trying_to_conceive = "trying_to_conceive"
    none = "none"
    unknown = "unknown"


class Subject(Strict):
    age_years: Captured[int]
    sex_at_birth: Captured[SexAtBirth]
    pregnancy_status: Captured[PregnancyStatus] | None = None


class SafetyProfile(Strict):
    current_medications: Captured[list[Medication]]
    allergies: Captured[list[str]]
    chronic_conditions: Captured[list[str]]
    recent_or_planned_surgery: Captured[bool] | None = None


DietType = Literal["vegetarian", "non_vegetarian", "vegan", "eggetarian", "other"]


class AyurvedicContext(Strict):
    appetite: Captured[str] | None = None
    digestion: Captured[str] | None = None
    bowel_pattern: Captured[str] | None = None
    sleep_quality: Captured[str] | None = None
    stress_level: Captured[str] | None = None
    diet_type: Captured[DietType] | None = None


class RedFlag(Strict):
    code: str = Field(pattern=r"^RF_[A-Z_]+$")
    source: Literal["rule", "model"]
    evidence_turn_id: str
    matched_text: str


class RedFlagScreen(Strict):
    screened_every_turn: bool
    flags: list[RedFlag]
    outcome: Literal["clear", "escalated"]


class Consent(Strict):
    granted: bool
    consent_text_version: str
    granted_at: datetime | None = None
    turn_id: str | None = None

    @model_validator(mode="after")
    def _granted_has_evidence(self):
        if self.granted and (self.granted_at is None or self.turn_id is None):
            raise ValueError("E011: granted consent requires granted_at and turn_id")
        return self


class ChiefComplaint(Strict):
    verbatim: str = Field(min_length=1)
    summary: Captured[str]
    body_system: Captured[BodySystem]


class Provenance(Strict):
    agent_version: str
    graph_version: str
    llm_models: dict[str, str]
    prompt_sha256: dict[str, str]
    guardrails_config_sha256: str
    asr_model: str | None = None
    tts_model: str | None = None
    transcript_ref: str


class SpecStatus(StrEnum):
    complete = "complete"
    incomplete = "incomplete"
    escalated_red_flag = "escalated_red_flag"
    out_of_scope = "out_of_scope"


def pregnancy_applicable(sex: SexAtBirth | None, age: int | None) -> bool:
    if sex not in (SexAtBirth.female, SexAtBirth.intersex) or age is None:
        return False
    lo, hi = PREGNANCY_AGE_RANGE
    return lo <= age <= hi


class SymptomSpec(Strict):
    spec_version: Literal["1.0.0"] = SPEC_VERSION
    spec_id: UUID = Field(default_factory=uuid4)
    session_id: str = Field(min_length=1)
    status: SpecStatus
    status_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    channel: Literal["voice", "chat"]
    language: str = "en-IN"
    jurisdiction: str = Field(default="IN", pattern=r"^[A-Z]{2}$")
    reporter_role: Literal["self", "caregiver", "practitioner"]
    consent: Consent
    red_flag_screen: RedFlagScreen
    subject: Subject | None = None
    chief_complaint: ChiefComplaint | None = None
    symptoms: list[SymptomDetail] = Field(default_factory=list)
    safety_profile: SafetyProfile | None = None
    ayurvedic_context: AyurvedicContext | None = None
    confidence_floor: float = Field(ge=0.0, le=1.0)
    provenance: Provenance

    def required_slots(self) -> dict[str, Captured]:
        """Applicable required slots that are present, keyed by dotted path."""
        slots: dict[str, Captured] = {}
        if self.subject:
            slots["subject.age_years"] = self.subject.age_years
            slots["subject.sex_at_birth"] = self.subject.sex_at_birth
            if self.subject.pregnancy_status is not None:
                slots["subject.pregnancy_status"] = self.subject.pregnancy_status
        if self.chief_complaint:
            slots["chief_complaint.summary"] = self.chief_complaint.summary
            slots["chief_complaint.body_system"] = self.chief_complaint.body_system
        if self.symptoms:
            slots["symptoms[0].name"] = self.symptoms[0].name
            slots["symptoms[0].duration"] = self.symptoms[0].duration
        if self.safety_profile:
            slots["safety_profile.current_medications"] = self.safety_profile.current_medications
            slots["safety_profile.allergies"] = self.safety_profile.allergies
            slots["safety_profile.chronic_conditions"] = self.safety_profile.chronic_conditions
        return slots

    def compute_confidence_floor(self) -> float:
        slots = self.required_slots()
        return min((c.confidence for c in slots.values()), default=0.0)

    def is_handoff_eligible(self, min_confidence: float) -> bool:
        return self.status is SpecStatus.complete and self.confidence_floor >= min_confidence

    @model_validator(mode="after")
    def _contract_rules(self, info: ValidationInfo):
        ctx = info.context or {}
        min_adult_age = ctx.get("min_adult_age", DEFAULT_MIN_ADULT_AGE)
        errors: list[str] = []
        complete = self.status is SpecStatus.complete

        if complete and not self.consent.granted:
            errors.append("E001: status=complete requires consent.granted")
        if complete and self.red_flag_screen.outcome != "clear":
            errors.append("E002: status=complete requires red_flag_screen.outcome=clear")
        if self.status is SpecStatus.escalated_red_flag and (
            not self.red_flag_screen.flags or self.red_flag_screen.outcome != "escalated"
        ):
            errors.append("E003: status=escalated_red_flag requires >=1 flag and outcome=escalated")
        if self.red_flag_screen.outcome == "escalated" and self.status is not SpecStatus.escalated_red_flag:
            errors.append("E003: red_flag_screen.outcome=escalated requires status=escalated_red_flag")

        if complete and (
            self.subject is None
            or self.chief_complaint is None
            or self.safety_profile is None
            or not self.symptoms
        ):
            errors.append(
                "E004: status=complete requires subject, chief_complaint, safety_profile and >=1 symptom"
            )

        if complete:
            inferred = [k for k, c in self.required_slots().items() if c.capture is CaptureMode.inferred]
            if inferred:
                errors.append(f"E005: required slots cannot be inferred: {', '.join(inferred)}")

        if self.subject:
            age = self.subject.age_years.value
            applicable = pregnancy_applicable(self.subject.sex_at_birth.value, age)
            if complete and applicable and self.subject.pregnancy_status is None:
                errors.append("E006: pregnancy_status is required for this sex and age")
            if not applicable and self.subject.pregnancy_status is not None:
                errors.append("E006: pregnancy_status must be omitted for this sex and age")
            if age is not None and not 0 <= age <= 120:
                errors.append("E007: age_years must be within 0..120")
            if complete and age is not None and age < min_adult_age:
                errors.append(f"E008: age below {min_adult_age} cannot be status=complete (use out_of_scope)")

        for i, s in enumerate(self.symptoms):
            if s.severity_0_10 and s.severity_0_10.value is not None and not 0 <= s.severity_0_10.value <= 10:
                errors.append(f"E007: symptoms[{i}].severity_0_10 must be within 0..10")

        expected_floor = self.compute_confidence_floor()
        if abs(self.confidence_floor - expected_floor) > 1e-9:
            errors.append(f"E009: confidence_floor {self.confidence_floor} != computed {expected_floor}")

        if errors:
            raise ValueError("; ".join(errors))
        return self
