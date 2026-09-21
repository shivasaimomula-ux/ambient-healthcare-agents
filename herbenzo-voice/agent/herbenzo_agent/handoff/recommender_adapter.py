"""Map a SymptomSpec onto the current Stage A Recommender API: POST /predict {"query", "context"}.

The full contract travels in `context.symptom_spec`, so Stage A can move to consuming it natively
without another change here. `confidence_floor` must be carried by Stage A as min(own, this).
"""

from __future__ import annotations

from typing import Any

from herbenzo_agent.contracts.symptom_spec import Captured, CaptureMode, SymptomSpec


def _value(captured: Captured | None) -> Any:
    if captured is None or captured.capture is CaptureMode.declined:
        return None
    value = captured.value
    return value.value if hasattr(value, "value") and not isinstance(value, (str, int, float)) else value


def build_query(spec: SymptomSpec) -> str:
    if spec.chief_complaint is None or not spec.symptoms:
        raise ValueError("spec has no chief complaint/symptom to build a query from")
    symptom = spec.symptoms[0]
    parts = [str(spec.chief_complaint.summary.value or spec.chief_complaint.verbatim)]
    duration = symptom.duration.value
    if symptom.duration.capture is not CaptureMode.declined and duration is not None:
        parts.append(f"for {duration.value:g} {duration.unit.value}")
    for field in (symptom.character, symptom.frequency_pattern):
        text = _value(field)
        if text and text.lower() not in parts[0].lower():
            parts.append(str(text))
    worse = _value(symptom.aggravating_factors)
    if worse:
        parts.append("worse with " + ", ".join(worse))
    return ", ".join(parts)


def to_predict_request(spec: SymptomSpec) -> dict[str, Any]:
    subject, safety = spec.subject, spec.safety_profile
    medications = _value(safety.current_medications) if safety else None
    spec_id = str(spec.spec_id)
    # Task T22 — start the run-thread with F's spec_id (operators see it on A/F meta).
    provenance_thread = {
        "schema_version": "1.0.0",
        "spec_id": spec_id,
        "stages": ["F"],
    }
    return {
        "query": build_query(spec),
        "context": {
            "source": "herbenzo-stage-f",
            "spec_id": spec_id,
            "spec_version": spec.spec_version,
            "confidence_floor": spec.confidence_floor,
            "jurisdiction": spec.jurisdiction,
            "provenance_thread": provenance_thread,
            "safety": {
                "age_years": _value(subject.age_years) if subject else None,
                "sex_at_birth": _value(subject.sex_at_birth) if subject else None,
                "pregnancy_status": _value(subject.pregnancy_status) if subject else None,
                "current_medications": [m.model_dump(mode="json") for m in medications]
                if medications
                else medications,
                "allergies": _value(safety.allergies) if safety else None,
                "chronic_conditions": _value(safety.chronic_conditions) if safety else None,
                "recent_or_planned_surgery": _value(safety.recent_or_planned_surgery) if safety else None,
            },
            "symptom_spec": spec.model_dump(mode="json"),
        },
    }
