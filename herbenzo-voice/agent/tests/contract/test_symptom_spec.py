import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from herbenzo_agent.contracts import export_schema
from herbenzo_agent.contracts.symptom_spec import Captured, CaptureMode, SpecStatus, SymptomSpec

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def validate(data: dict, **ctx) -> SymptomSpec:
    return SymptomSpec.model_validate(data, context=ctx or None)


def cap(value, confidence=0.95, capture="confirmed", turn="t:1") -> dict:
    return {"value": value, "confidence": confidence, "capture": capture, "turn_ids": [turn]}


def assert_error(data: dict, code: str, **ctx):
    with pytest.raises(ValidationError) as exc:
        validate(data, **ctx)
    assert code in str(exc.value), str(exc.value)


# --- fixtures round-trip -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["symptom_spec_complete.json", "symptom_spec_escalated.json"])
def test_fixture_round_trip(name):
    spec = validate(load(name))
    again = validate(json.loads(spec.model_dump_json()))
    assert again == spec


def test_complete_fixture_is_handoff_eligible():
    spec = validate(load("symptom_spec_complete.json"))
    assert spec.status is SpecStatus.complete
    assert spec.is_handoff_eligible(0.5)
    assert spec.confidence_floor == 0.95


def test_escalated_fixture_is_never_eligible():
    assert not validate(load("symptom_spec_escalated.json")).is_handoff_eligible(0.0)


def test_unknown_fields_rejected():
    data = load("symptom_spec_complete.json")
    data["diagnosis"] = "GERD"
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate(data)


def test_spec_is_immutable():
    spec = validate(load("symptom_spec_complete.json"))
    with pytest.raises(ValidationError):
        spec.status = SpecStatus.incomplete


# --- Captured -----------------------------------------------------------------------------------


def test_declined_requires_empty_value_and_zero_confidence():
    Captured[str].model_validate({"value": None, "confidence": 0.0, "capture": "declined", "turn_ids": ["t"]})
    with pytest.raises(ValidationError, match="E010"):
        Captured[str].model_validate(
            {"value": "x", "confidence": 0.0, "capture": "declined", "turn_ids": ["t"]}
        )
    with pytest.raises(ValidationError, match="E010"):
        Captured[str].model_validate(
            {"value": None, "confidence": 0.9, "capture": "stated", "turn_ids": ["t"]}
        )


def test_captured_requires_provenance():
    with pytest.raises(ValidationError):
        Captured[str].model_validate({"value": "x", "confidence": 0.9, "capture": "stated", "turn_ids": []})


# --- validator rules ----------------------------------------------------------------------------


def test_e001_complete_requires_consent():
    data = load("symptom_spec_complete.json")
    data["consent"] = {"granted": False, "consent_text_version": "v1"}
    assert_error(data, "E001")


def test_e002_complete_requires_clear_screen():
    data = load("symptom_spec_complete.json")
    data["red_flag_screen"] = load("symptom_spec_escalated.json")["red_flag_screen"]
    assert_error(data, "E002")


def test_e003_escalated_requires_flags():
    data = load("symptom_spec_escalated.json")
    data["red_flag_screen"] = {"screened_every_turn": True, "flags": [], "outcome": "clear"}
    assert_error(data, "E003")


def test_e004_complete_requires_core_sections():
    data = load("symptom_spec_complete.json")
    data["safety_profile"] = None
    data["confidence_floor"] = 0.95
    assert_error(data, "E004")


def test_e005_required_slot_cannot_be_inferred():
    data = load("symptom_spec_complete.json")
    data["subject"]["age_years"] = cap(34, confidence=0.4, capture="inferred")
    data["confidence_floor"] = 0.4
    assert_error(data, "E005")


def test_e005_optional_slot_may_be_inferred():
    data = load("symptom_spec_complete.json")
    data["symptoms"][0]["character"] = cap("burning", confidence=0.4, capture="inferred")
    validate(data)


def test_e006_pregnancy_required_for_female_of_reproductive_age():
    data = load("symptom_spec_complete.json")
    data["subject"]["sex_at_birth"] = cap("female")
    assert_error(data, "E006")
    data["subject"]["pregnancy_status"] = cap("none")
    validate(data)


def test_e006_pregnancy_forbidden_when_not_applicable():
    data = load("symptom_spec_complete.json")
    data["subject"]["pregnancy_status"] = cap("none")
    assert_error(data, "E006")


def test_e007_severity_range():
    data = load("symptom_spec_complete.json")
    data["symptoms"][0]["severity_0_10"] = cap(11, confidence=0.8, capture="stated")
    assert_error(data, "E007")


def test_e007_age_range():
    data = load("symptom_spec_complete.json")
    data["subject"]["age_years"] = cap(130)
    assert_error(data, "E007")


def test_e008_minor_cannot_be_complete():
    data = load("symptom_spec_complete.json")
    data["subject"]["age_years"] = cap(15)
    assert_error(data, "E008")
    data["status"] = "out_of_scope"
    validate(data)


def test_e008_adult_age_comes_from_context():
    data = load("symptom_spec_complete.json")
    data["subject"]["age_years"] = cap(19)
    assert_error(data, "E008", min_adult_age=21)
    validate(data, min_adult_age=18)


def test_e009_confidence_floor_must_match():
    data = load("symptom_spec_complete.json")
    data["confidence_floor"] = 0.99
    assert_error(data, "E009")


def test_declined_safety_slot_drops_floor_and_blocks_handoff():
    data = load("symptom_spec_complete.json")
    data["safety_profile"]["current_medications"] = {
        "value": None,
        "confidence": 0.0,
        "capture": CaptureMode.declined,
        "turn_ids": ["t:7"],
    }
    data["confidence_floor"] = 0.0
    spec = validate(data)
    assert spec.status is SpecStatus.complete
    assert not spec.is_handoff_eligible(0.5)


def test_errors_are_reported_together():
    data = copy.deepcopy(load("symptom_spec_complete.json"))
    data["consent"] = {"granted": False, "consent_text_version": "v1"}
    data["confidence_floor"] = 0.1
    with pytest.raises(ValidationError) as exc:
        validate(data)
    assert "E001" in str(exc.value) and "E009" in str(exc.value)


# --- schema -------------------------------------------------------------------------------------


def test_committed_schema_matches_models():
    committed = export_schema.SCHEMA_PATH.read_text()
    assert committed == export_schema.render_schema(), (
        "SymptomSpec changed without regenerating the schema. Run "
        "`uv run python -m herbenzo_agent.contracts.export_schema`, bump SPEC_VERSION if the "
        "change is not backwards compatible, and add a contracts/CHANGELOG.md entry."
    )
