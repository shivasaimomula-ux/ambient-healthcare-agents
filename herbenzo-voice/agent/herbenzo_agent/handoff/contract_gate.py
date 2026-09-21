"""Outbound F→A contract gate using shared herbenzo-contracts SymptomSpec.

Local ``herbenzo_agent.contracts.symptom_spec`` remains the in-process type;
this module re-validates dumps against the federated package before delivery
so unknown fields / floor violations never leave Stage F.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from herbenzo_contracts import SymptomSpec as SharedSymptomSpec
from herbenzo_contracts import validation_error_body


def validate_outbound_symptom_spec(payload: dict[str, Any]) -> SharedSymptomSpec:
    """Validate a SymptomSpec JSON dump; raise ValidationError on failure."""
    return SharedSymptomSpec.model_validate(payload)


def validate_predict_request(request: dict[str, Any]) -> SharedSymptomSpec:
    """Validate ``context.symptom_spec`` inside a POST /predict body."""
    ctx = request.get("context")
    if not isinstance(ctx, dict):
        raise ValueError("context must be an object")
    raw = ctx.get("symptom_spec")
    if not isinstance(raw, dict):
        raise ValueError("context.symptom_spec must be an object")
    spec = validate_outbound_symptom_spec(raw)
    env_floor = ctx.get("confidence_floor")
    if env_floor is not None and float(env_floor) > float(spec.confidence_floor) + 1e-9:
        raise ValueError(
            f"context.confidence_floor {env_floor} exceeds "
            f"symptom_spec.confidence_floor {spec.confidence_floor}"
        )
    return spec


__all__ = [
    "validate_outbound_symptom_spec",
    "validate_predict_request",
    "validation_error_body",
    "ValidationError",
]
