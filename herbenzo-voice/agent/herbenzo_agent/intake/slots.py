"""Slot registry: what the intake asks, in what order, and how answers are typed.

The planner walks `SLOTS` in order and asks the first applicable, unresolved slot. The extractor
may fill any slot at any time. Base questions are the canonical wording: the responder may
rephrase them but not change their meaning, and they are the fallback when the LLM fails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, TypeAdapter

from herbenzo_agent.contracts.symptom_spec import (
    BodySystem,
    DietType,
    DurationUnit,
    Medication,
    PregnancyStatus,
    SexAtBirth,
    pregnancy_applicable,
)
from herbenzo_agent.intake.models import IntakePolicy, IntakeSession

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class DurationValue(BaseModel):
    value: float = Field(gt=0)
    unit: DurationUnit
    raw_text: NonEmptyStr


def _medications(raw: Any) -> list[dict]:
    items = raw if isinstance(raw, list) else [raw]
    meds = [{"name": m} if isinstance(m, str) else m for m in items]
    return [Medication.model_validate(m).model_dump() for m in meds]


@dataclass(frozen=True)
class Slot:
    key: str
    required: bool
    question: str
    adapter: TypeAdapter | None = None
    parse_fn: Callable[[Any], Any] | None = None
    askable: bool = True
    max_asks: int = 2
    applies: Callable[[IntakeSession, IntakePolicy], bool] = lambda s, p: True
    about_other: str | None = None  # question wording when reporter is not the subject
    # For safety slots: an off-target "none" or decline must quote one of these words (see planner).
    topic_keywords: tuple[str, ...] = ()

    def parse(self, raw: Any) -> Any:
        if self.parse_fn is not None:
            return self.parse_fn(raw)
        assert self.adapter is not None
        value = self.adapter.validate_python(raw)
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [v.value if hasattr(v, "value") else v for v in value]
        return value.value if hasattr(value, "value") else value

    def question_for(self, session: IntakeSession) -> str:
        role = session.value("reporter_role")
        if role in ("caregiver", "practitioner") and self.about_other:
            return self.about_other
        return self.question


def _str() -> TypeAdapter:
    return TypeAdapter(NonEmptyStr)


def _str_list() -> TypeAdapter:
    return TypeAdapter(list[NonEmptyStr])


def _pregnancy_applies(s: IntakeSession, p: IntakePolicy) -> bool:
    sex = s.value("subject.sex_at_birth")
    age = s.value("subject.age_years")
    return pregnancy_applicable(SexAtBirth(sex) if sex else None, age)


def _ayurvedic(s: IntakeSession, p: IntakePolicy) -> bool:
    return p.ayurvedic_context_enabled


SLOTS: tuple[Slot, ...] = (
    Slot(
        "reporter_role",
        True,
        "Are you telling me about your own health, someone you care for, or a patient you treat?",
        TypeAdapter(Literal["self", "caregiver", "practitioner"]),
    ),
    Slot(
        "chief_complaint.verbatim",
        True,
        "What's been bothering you? Please tell me in your own words.",
        _str(),
        about_other="What's been bothering them? Please describe it in your own words.",
    ),
    Slot("chief_complaint.summary", True, "In a few words, what is the main problem?", _str()),
    Slot(
        "chief_complaint.body_system",
        True,
        "Which part of the body is this mostly affecting?",
        TypeAdapter(BodySystem),
    ),
    Slot("symptoms[0].name", True, "What would you call the main symptom?", _str(), askable=False),
    Slot("symptoms[0].duration", True, "How long has this been going on?", TypeAdapter(DurationValue)),
    Slot(
        "symptoms[0].severity_0_10",
        False,
        "On a scale of zero to ten, how strong is it at its worst?",
        TypeAdapter(Annotated[int, Field(ge=0, le=10)]),
    ),
    Slot(
        "symptoms[0].frequency_pattern",
        False,
        "Is it there all the time, or does it come and go?",
        _str(),
    ),
    Slot("symptoms[0].location", False, "Where exactly do you feel it?", _str(), askable=False),
    Slot("symptoms[0].character", False, "How would you describe the feeling?", _str(), askable=False),
    Slot(
        "symptoms[0].aggravating_factors",
        False,
        "Does anything make it worse, like certain foods, stress, or activity?",
        _str_list(),
    ),
    Slot("symptoms[0].relieving_factors", False, "Does anything make it better?", _str_list()),
    Slot(
        "symptoms[0].associated_symptoms",
        False,
        "Have you noticed anything else happening along with it?",
        _str_list(),
    ),
    Slot(
        "subject.age_years",
        True,
        "How old are you?",
        TypeAdapter(Annotated[int, Field(ge=0, le=120)]),
        about_other="How old is the person?",
    ),
    Slot(
        "subject.sex_at_birth",
        True,
        "What sex were you assigned at birth? You can also say you'd prefer not to say.",
        TypeAdapter(SexAtBirth),
        about_other="What sex was the person assigned at birth?",
    ),
    Slot(
        "subject.pregnancy_status",
        True,
        "Are you currently pregnant, breastfeeding, or trying to conceive?",
        TypeAdapter(PregnancyStatus),
        applies=_pregnancy_applies,
        about_other="Is the person currently pregnant, breastfeeding, or trying to conceive?",
    ),
    Slot(
        "safety_profile.current_medications",
        True,
        "Are you taking any medicines right now, including herbal or Ayurvedic products or supplements?",
        parse_fn=_medications,
        topic_keywords=(
            "medic",
            "tablet",
            "pill",
            "drug",
            "taking",
            "take",
            "supplement",
            "herbal",
            "ayurvedic",
        ),
        about_other="Are they taking any medicines right now, including herbal or Ayurvedic products or supplements?",
    ),
    Slot(
        "safety_profile.allergies",
        True,
        "Do you have any allergies to medicines, herbs, or foods?",
        _str_list(),
        topic_keywords=("allerg",),
        about_other="Do they have any allergies to medicines, herbs, or foods?",
    ),
    Slot(
        "safety_profile.chronic_conditions",
        True,
        "Do you have any long-term health conditions, like diabetes, high blood pressure, or liver or kidney problems?",
        _str_list(),
        topic_keywords=(
            "condition",
            "chronic",
            "diabetes",
            "sugar",
            "pressure",
            "bp",
            "thyroid",
            "asthma",
            "liver",
            "kidney",
            "heart",
            "long-term",
            "long term",
            "illness",
            "disease",
            "health problem",
        ),
        about_other="Do they have any long-term health conditions, like diabetes, high blood pressure, or liver or kidney problems?",
    ),
    Slot(
        "safety_profile.recent_or_planned_surgery",
        False,
        "Have you had surgery recently, or is any surgery planned?",
        TypeAdapter(bool),
        topic_keywords=("surg", "operation", "operated"),
    ),
    Slot(
        "ayurvedic_context.appetite", False, "How has your appetite been lately?", _str(), applies=_ayurvedic
    ),
    Slot(
        "ayurvedic_context.digestion", False, "How is your digestion generally?", _str(), applies=_ayurvedic
    ),
    Slot(
        "ayurvedic_context.bowel_pattern",
        False,
        "How regular are your bowel movements?",
        _str(),
        applies=_ayurvedic,
    ),
    Slot(
        "ayurvedic_context.sleep_quality",
        False,
        "How well have you been sleeping?",
        _str(),
        applies=_ayurvedic,
    ),
    Slot(
        "ayurvedic_context.stress_level",
        False,
        "How would you describe your stress levels these days?",
        _str(),
        applies=_ayurvedic,
    ),
    Slot(
        "ayurvedic_context.diet_type",
        False,
        "Is your diet vegetarian, non-vegetarian, vegan, or something else?",
        TypeAdapter(DietType),
        applies=_ayurvedic,
    ),
)

SLOT_INDEX: dict[str, Slot] = {s.key: s for s in SLOTS}


def get_slot(key: str) -> Slot | None:
    return SLOT_INDEX.get(key)
