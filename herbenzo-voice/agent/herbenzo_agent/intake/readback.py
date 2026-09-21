"""Templated readback built only from stored slots, so what the user confirms is exactly what is saved."""

from __future__ import annotations

from typing import Any

from herbenzo_agent.contracts.symptom_spec import CaptureMode
from herbenzo_agent.intake.models import IntakeSession

_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

_DECLINED_LABELS = {
    "subject.age_years": "age",
    "subject.sex_at_birth": "sex at birth",
    "subject.pregnancy_status": "pregnancy status",
    "symptoms[0].duration": "how long it has lasted",
    "safety_profile.current_medications": "medicines",
    "safety_profile.allergies": "allergies",
    "safety_profile.chronic_conditions": "long-term conditions",
}


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


class _Voice:
    def __init__(self, about_self: bool):
        self.about_self = about_self

    def __getattr__(self, name: str) -> str:
        forms = {
            "subj": ("you", "they"),
            "Subj": ("You", "They"),
            "are": ("You're", "They're"),
            "are_l": ("you're", "they're"),
            "have": ("you've", "they've"),
            "has": ("you have", "they have"),
            "Has": ("You have", "They have"),
            "poss": ("your", "their"),
        }
        pair = forms[name]
        return pair[0] if self.about_self else pair[1]


def _declined(session: IntakeSession, key: str) -> bool:
    slot = session.slots.get(key)
    return slot is not None and slot.capture is CaptureMode.declined


def _duration_clause(raw: dict[str, Any] | None) -> str:
    if not raw:
        return ""
    text = str(raw.get("raw_text") or f"{raw['value']:g} {raw['unit']}").strip()
    lowered = text.lower()
    if lowered.startswith("for "):
        text = text[4:]
    elif lowered.startswith("since "):
        return f" {text}"
    return f" for {text}"


def build_readback(session: IntakeSession) -> list[str]:
    """Return spoken chunks (each at most two sentences). The last chunk asks for confirmation."""
    v = _Voice(session.value("reporter_role") in (None, "self"))
    chunks: list[str] = []

    # Chunk 1: who and what.
    summary = (
        session.value("chief_complaint.summary")
        or session.value("chief_complaint.verbatim")
        or "this problem"
    )
    age = session.value("subject.age_years")
    lead = f"{v.are} {age}, and {v.have} had" if age is not None else f"{v.Subj} have had"
    if not v.about_self and age is None:
        lead = "They have had"
    sentence = (
        f"Here's what I have. {lead} {summary}{_duration_clause(session.value('symptoms[0].duration'))}"
    )
    extras = []
    if worse := session.value("symptoms[0].aggravating_factors"):
        extras.append(f"worse with {_join(worse)}")
    if better := session.value("symptoms[0].relieving_factors"):
        extras.append(f"better with {_join(better)}")
    severity = session.value("symptoms[0].severity_0_10")
    if severity is not None:
        extras.append(f"about {_NUMBER_WORDS[severity]} out of ten at its worst")
    if extras:
        sentence += ", " + ", ".join(extras)
    chunks.append(sentence + ".")

    # Chunk 2: safety profile.
    safety: list[str] = []
    meds = session.value("safety_profile.current_medications")
    if meds is not None:
        names = [m["name"] for m in meds]
        safety.append(f"{v.Subj} take {_join(names)}." if names else f"{v.are} not taking any medicines.")
    allergies = session.value("safety_profile.allergies")
    conditions = session.value("safety_profile.chronic_conditions")
    parts = []
    if allergies is not None:
        parts.append(f"allergic to {_join(allergies)}" if allergies else "no allergies")
    if conditions is not None:
        parts.append(
            f"{_join(conditions)} as a long-term condition" if conditions else "no long-term conditions"
        )
    if parts:
        if allergies:
            safety.append(f"{v.are} {parts[0]}" + (f", and {v.has} {parts[1]}." if len(parts) > 1 else "."))
        else:
            safety.append(f"{v.Has} {_join(parts)}.")
    pregnancy = session.value("subject.pregnancy_status")
    pregnancy_text = {
        "pregnant": f"{v.are} pregnant.",
        "breastfeeding": f"{v.are} breastfeeding.",
        "trying_to_conceive": f"{v.are} trying to conceive.",
        "none": f"{v.are} not pregnant or breastfeeding.",
        "unknown": f"{v.are} not sure about pregnancy.",
    }.get(pregnancy)
    if pregnancy_text:
        safety.append(pregnancy_text)
    if safety:
        chunks.append(" ".join(safety))

    # Chunk 3: declined slots, then the confirmation question.
    declined = [label for key, label in _DECLINED_LABELS.items() if _declined(session, key)]
    closing = "Is all of that correct?"
    if declined:
        closing = f"{v.Subj} preferred not to say {v.poss} {_join(declined)}. {closing}"
    chunks.append(closing)
    return chunks
