from herbenzo_agent.contracts.symptom_spec import SpecStatus, SymptomSpec
from herbenzo_agent.intake.models import IntakePolicy, Intent
from herbenzo_agent.intake.spec_builder import build_spec
from tests.helpers import Sim, provenance, run_happy_path, skip_ayurvedic_policy, up


def build(sim: Sim) -> SymptomSpec:
    return build_spec(sim.session, provenance(), sim.policy)


def test_happy_path_builds_complete_eligible_spec():
    sim = run_happy_path(Sim(skip_ayurvedic_policy()))
    sim.user("yes", Intent.confirm_yes)
    spec = build(sim)
    assert spec.status is SpecStatus.complete
    assert spec.confidence_floor == 0.95
    assert spec.is_handoff_eligible(0.5)
    assert spec.consent.granted and spec.consent.turn_id == "s1:1"
    assert spec.symptoms[0].name.value == "burning acidity"
    assert spec.symptoms[0].duration.value.unit == "weeks"
    assert spec.safety_profile.current_medications.value[0].name == "pantoprazole"
    assert spec.safety_profile.allergies.value == []
    assert spec.subject.pregnancy_status is None
    # provenance: every value points at the user turn it came from
    assert spec.subject.age_years.turn_ids[0] == "s1:9"
    # round trip through the contract
    assert SymptomSpec.model_validate_json(spec.model_dump_json()) == spec


def test_unconfirmed_submission_keeps_stated_confidence():
    sim = run_happy_path(Sim(IntakePolicy(ayurvedic_context_enabled=False, max_readback_cycles=1)))
    sim.user("no", Intent.confirm_no)
    sim.user("hmm", Intent.unclear)
    spec = build(sim)
    assert spec.status is SpecStatus.complete
    assert spec.confidence_floor <= 0.85


def test_escalated_spec_never_eligible():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user("I'm 40", Intent.answer, up("subject.age_years", 40, "40"))
    sim.user("I'm vomiting blood", Intent.answer)
    spec = build(sim)
    assert spec.status is SpecStatus.escalated_red_flag
    assert spec.red_flag_screen.flags[0].code == "RF_SEVERE_BLEEDING"
    assert not spec.is_handoff_eligible(0.0)


def test_declined_medications_block_handoff():
    sim = Sim(skip_ayurvedic_policy())
    run_happy_path(sim)
    sim.user("no", Intent.confirm_no)
    sim.user(
        "actually I'd rather not say about medicines",
        Intent.correction,
        up("safety_profile.current_medications", None, "", declined=True),
    )
    sim.user("yes", Intent.confirm_yes)
    spec = build(sim)
    assert spec.status is SpecStatus.complete
    assert spec.confidence_floor == 0.0
    assert not spec.is_handoff_eligible(0.5)


def test_missing_required_downgrades_to_incomplete_with_codes():
    sim = run_happy_path(Sim(skip_ayurvedic_policy()))
    del sim.session.slots["safety_profile.allergies"]
    sim.session.final_status = SpecStatus.complete
    spec = build(sim)
    assert spec.status is SpecStatus.incomplete
    assert spec.status_reason == "validation:E004"


def test_minor_builds_out_of_scope_spec():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user(
        "I'm 16, male",
        Intent.answer,
        up("subject.age_years", 16, "16"),
        up("subject.sex_at_birth", "male", "male"),
    )
    spec = build(sim)
    assert spec.status is SpecStatus.out_of_scope and spec.status_reason == "minor"


def test_stopped_session_builds_incomplete_spec():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user("me", Intent.answer, up("reporter_role", "caregiver", "me"))
    sim.user("stop", Intent.stop)
    spec = build(sim)
    assert spec.status is SpecStatus.incomplete and spec.reporter_role == "caregiver"
