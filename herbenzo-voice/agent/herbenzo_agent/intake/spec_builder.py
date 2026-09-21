"""Deterministic mapping from an IntakeSession to the immutable SymptomSpec contract."""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from herbenzo_agent.contracts.symptom_spec import (
    CaptureMode,
    Provenance,
    SexAtBirth,
    SpecStatus,
    SymptomSpec,
    pregnancy_applicable,
)
from herbenzo_agent.intake.models import CONSENT_TEXT_VERSION, IntakePolicy, IntakeSession

logger = logging.getLogger(__name__)

_SYMPTOM_FIELDS = (
    "severity_0_10",
    "frequency_pattern",
    "location",
    "character",
    "aggravating_factors",
    "relieving_factors",
    "associated_symptoms",
)
_AYURVEDIC_FIELDS = ("appetite", "digestion", "bowel_pattern", "sleep_quality", "stress_level", "diet_type")

# Required slots that must be resolved in the session before a spec may be `complete`.
_REQUIRED_KEYS = ("reporter_role",)


class SpecBuildError(RuntimeError):
    pass


def _cap(session: IntakeSession, key: str) -> dict[str, Any] | None:
    slot = session.slots.get(key)
    if slot is None:
        return None
    return {
        "value": slot.value,
        "confidence": slot.confidence,
        "capture": slot.capture.value,
        "turn_ids": list(slot.turn_ids),
    }


def _subject(session: IntakeSession) -> dict | None:
    age, sex = _cap(session, "subject.age_years"), _cap(session, "subject.sex_at_birth")
    if age is None or sex is None:
        return None
    subject = {"age_years": age, "sex_at_birth": sex, "pregnancy_status": None}
    sex_value = SexAtBirth(sex["value"]) if sex["value"] else None
    if pregnancy_applicable(sex_value, age["value"]):
        subject["pregnancy_status"] = _cap(session, "subject.pregnancy_status")
    return subject


def _chief_complaint(session: IntakeSession) -> dict | None:
    verbatim = session.value("chief_complaint.verbatim")
    summary, system = _cap(session, "chief_complaint.summary"), _cap(session, "chief_complaint.body_system")
    if not verbatim or summary is None or system is None:
        return None
    return {"verbatim": verbatim, "summary": summary, "body_system": system}


def _symptoms(session: IntakeSession) -> list[dict]:
    name, duration = _cap(session, "symptoms[0].name"), _cap(session, "symptoms[0].duration")
    if name is None or duration is None:
        return []
    symptom = {"name": name, "duration": duration}
    symptom.update({f: _cap(session, f"symptoms[0].{f}") for f in _SYMPTOM_FIELDS})
    return [symptom]


def _safety(session: IntakeSession) -> dict | None:
    keys = ("current_medications", "allergies", "chronic_conditions")
    values = {k: _cap(session, f"safety_profile.{k}") for k in keys}
    if any(v is None for v in values.values()):
        return None
    values["recent_or_planned_surgery"] = _cap(session, "safety_profile.recent_or_planned_surgery")
    return values


def _ayurvedic(session: IntakeSession) -> dict | None:
    values = {f: _cap(session, f"ayurvedic_context.{f}") for f in _AYURVEDIC_FIELDS}
    return values if any(v is not None for v in values.values()) else None


def _floor(sections: dict[str, Any]) -> float:
    confidences: list[float] = []
    if subject := sections["subject"]:
        confidences += [subject["age_years"]["confidence"], subject["sex_at_birth"]["confidence"]]
        if subject["pregnancy_status"]:
            confidences.append(subject["pregnancy_status"]["confidence"])
    if cc := sections["chief_complaint"]:
        confidences += [cc["summary"]["confidence"], cc["body_system"]["confidence"]]
    if symptoms := sections["symptoms"]:
        confidences += [symptoms[0]["name"]["confidence"], symptoms[0]["duration"]["confidence"]]
    if safety := sections["safety_profile"]:
        confidences += [
            safety[k]["confidence"] for k in ("current_medications", "allergies", "chronic_conditions")
        ]
    return min(confidences, default=0.0)


def build_spec(session: IntakeSession, provenance: Provenance, policy: IntakePolicy) -> SymptomSpec:
    status = session.final_status or SpecStatus.incomplete
    reason = session.status_reason
    if status is SpecStatus.complete:
        unresolved = [
            k
            for k in _REQUIRED_KEYS
            if not session.has(k) or session.slots[k].capture is CaptureMode.declined
        ]
        if unresolved:
            status, reason = SpecStatus.incomplete, "unresolved:" + ",".join(unresolved)

    sections: dict[str, Any] = {
        "subject": _subject(session),
        "chief_complaint": _chief_complaint(session),
        "symptoms": _symptoms(session),
        "safety_profile": _safety(session),
        "ayurvedic_context": _ayurvedic(session),
    }
    data: dict[str, Any] = {
        "session_id": session.session_id,
        "channel": session.channel,
        "language": session.language,
        "jurisdiction": session.jurisdiction,
        "reporter_role": session.value("reporter_role") or "self",
        "consent": {
            "granted": session.consent_granted,
            "consent_text_version": CONSENT_TEXT_VERSION,
            "granted_at": session.consent_at,
            "turn_id": session.consent_turn_id,
        },
        "red_flag_screen": {
            "screened_every_turn": True,
            "flags": [f.model_dump() for f in session.red_flags],
            "outcome": "escalated" if session.red_flags else "clear",
        },
        **sections,
        "confidence_floor": _floor(sections),
        "provenance": provenance.model_dump(),
    }
    if session.spec_id:
        data["spec_id"] = session.spec_id

    context = {"min_adult_age": policy.min_adult_age}
    try:
        return SymptomSpec.model_validate(
            {**data, "status": status, "status_reason": reason}, context=context
        )
    except ValidationError as exc:
        if status is not SpecStatus.complete:
            raise SpecBuildError(f"could not build {status} spec: {exc}") from exc
        codes = sorted(set(re.findall(r"\bE\d{3}\b", str(exc))))
        logger.warning("spec downgraded to incomplete: %s", codes)
        reason = "validation:" + ",".join(codes)
        try:
            return SymptomSpec.model_validate(
                {**data, "status": SpecStatus.incomplete, "status_reason": reason}, context=context
            )
        except ValidationError as exc2:
            raise SpecBuildError(f"could not build incomplete spec: {exc2}") from exc2
