from herbenzo_agent.contracts.symptom_spec import CaptureMode, SpecStatus
from herbenzo_agent.intake.models import IntakePolicy, Intent, Phase
from herbenzo_agent.intake.planner import CLOSING_PLACEHOLDER
from tests.helpers import Sim, run_happy_path, skip_ayurvedic_policy, up


def test_session_starts_with_scripted_greeting():
    sim = Sim()
    assert sim.keys() == ["greeting_consent"]
    assert sim.session.phase is Phase.GREETING_CONSENT


def test_happy_path_asks_in_order_then_reads_back_and_submits():
    sim = run_happy_path(Sim(skip_ayurvedic_policy()))
    asked = [d.target_slot for d in sim.decisions if d.target_slot]
    assert asked == [
        "reporter_role",
        "chief_complaint.verbatim",
        "symptoms[0].duration",
        "symptoms[0].severity_0_10",
        "symptoms[0].frequency_pattern",
        "symptoms[0].relieving_factors",  # aggravating was volunteered in the complaint
        "symptoms[0].associated_symptoms",
        "subject.age_years",
        "subject.sex_at_birth",  # pregnancy not applicable for male
        "safety_profile.current_medications",
        "safety_profile.allergies",
        "safety_profile.chronic_conditions",
        "safety_profile.recent_or_planned_surgery",
    ]
    assert sim.keys() == ["readback"]
    assert sim.session.phase is Phase.READBACK

    done = sim.user("Yes, that's right", Intent.confirm_yes)
    assert done.submit and done.final_status is SpecStatus.complete
    assert [s.key for s in done.says] == [CLOSING_PLACEHOLDER]
    assert all(s.capture is CaptureMode.confirmed for s in sim.session.slots.values())


def test_volunteered_information_is_not_asked_again():
    sim = Sim(skip_ayurvedic_policy())
    sim.user("yes", Intent.confirm_yes)
    d = sim.user(
        "It's me, I'm 34, male, and I take metformin",
        Intent.answer,
        up("reporter_role", "self", "it's me"),
        up("subject.age_years", 34, "34"),
        up("subject.sex_at_birth", "male", "male"),
        up("safety_profile.current_medications", ["metformin"], "metformin"),
    )
    assert d.target_slot == "chief_complaint.verbatim"
    later = [dd.target_slot for dd in sim.decisions]
    assert "subject.age_years" not in later


def test_consent_refused_closes_without_submitting():
    sim = Sim()
    d = sim.user("No thanks", Intent.confirm_no)
    assert d.phase is Phase.CLOSED and d.final_status is SpecStatus.out_of_scope and not d.submit
    assert sim.keys() == ["consent_refused"]


def test_unclear_consent_reasked_then_treated_as_refused():
    sim = Sim()
    assert sim.user("hmm what", Intent.unclear).says[0].key == "consent_reask"
    d = sim.user("I don't know", Intent.unclear)
    assert d.final_status is SpecStatus.out_of_scope and d.status_reason == "consent_refused"


def test_no_health_data_stored_before_consent():
    sim = Sim()
    sim.user("I have acidity", Intent.unclear, up("chief_complaint.summary", "acidity", "acidity"))
    assert sim.session.slots == {}


def test_red_flag_before_consent_still_escalates():
    sim = Sim()
    d = sim.user("I have chest pain going to my left arm", Intent.answer)
    assert d.phase is Phase.ESCALATED and d.final_status is SpecStatus.escalated_red_flag and d.submit
    assert sim.keys() == ["escalation"]


def test_self_harm_uses_dedicated_script():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    assert sim.user("honestly I want to end my life", Intent.answer).says[0].key == "escalation_self_harm"


def test_terminal_session_rejects_further_turns():
    sim = Sim()
    sim.user("chest pain", Intent.answer)
    assert sim.user("hello?", Intent.answer).says[0].key == "session_closed"


def test_restart_word_inside_symptom_does_not_reset():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user("me", Intent.answer, up("reporter_role", "self", "me"))
    d = sim.user(
        "my pain restarts every morning",
        Intent.answer,
        up("symptoms[0].frequency_pattern", "every morning", "every morning"),
    )
    assert d.phase is Phase.COLLECTING and not d.restart
    assert sim.session.value("symptoms[0].frequency_pattern") == "every morning"


def test_restart_requires_confirmation():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user("me", Intent.answer, up("reporter_role", "self", "me"))
    assert sim.user("can we start over", Intent.restart).says[0].key == "confirm_restart"
    d = sim.user("no, carry on", Intent.confirm_no)
    assert [s.key for s in d.says] == ["restart_resume", "chief_complaint.verbatim"]
    assert sim.session.value("reporter_role") == "self"

    sim.user("start over please", Intent.restart)
    d = sim.user("yes", Intent.confirm_yes)
    assert d.restart and d.final_status is SpecStatus.incomplete and d.submit
    assert sim.session.phase is Phase.CLOSED


def test_stop_closes_incomplete():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    d = sim.user("stop", Intent.stop)
    assert d.final_status is SpecStatus.incomplete and d.status_reason == "user_stopped" and d.submit


def test_advice_request_is_deflected_and_question_repeated():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    d = sim.user("which herb should I take?", Intent.asks_advice)
    assert [s.key for s in d.says] == ["deflect_advice", "reporter_role"]


def test_off_topic_deflected_only_when_nothing_captured():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    assert sim.user("who won the cricket", Intent.off_topic).says[0].key == "deflect_off_topic"
    d = sim.user("haha anyway it's me", Intent.off_topic, up("reporter_role", "self", "it's me"))
    assert d.says[0].key == "chief_complaint.verbatim"


def test_required_slot_declined_after_max_asks():
    sim = Sim(skip_ayurvedic_policy())
    sim.user("yes", Intent.confirm_yes)
    sim.user("hmm", Intent.unclear)  # reporter_role asked twice now
    d = sim.user("not sure", Intent.unclear)
    assert d.target_slot == "chief_complaint.verbatim"
    slot = sim.session.slots["reporter_role"]
    assert slot.capture is CaptureMode.declined and slot.confidence == 0.0


def test_explicit_decline_is_immediate():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    d = sim.user("I'd rather not say", Intent.answer, up("reporter_role", None, "", declined=True))
    assert d.target_slot == "chief_complaint.verbatim"


def _through_symptoms(sim: Sim) -> None:
    sim.user("yes", Intent.confirm_yes)
    sim.user("me", Intent.answer, up("reporter_role", "self", "me"))
    sim.user(
        "nausea for two days, nothing makes it better or worse, maybe a four, it comes and goes, nothing else",
        Intent.answer,
        up("chief_complaint.verbatim", "nausea for two days", "nausea"),
        up("chief_complaint.summary", "nausea", "nausea"),
        up("chief_complaint.body_system", "digestive", "nausea"),
        up("symptoms[0].duration", {"value": 2, "unit": "days", "raw_text": "two days"}, "two days"),
        up("symptoms[0].severity_0_10", 4, "a four"),
        up("symptoms[0].frequency_pattern", "comes and goes", "comes and goes"),
        up("symptoms[0].aggravating_factors", [], "nothing makes it better or worse"),
        up("symptoms[0].relieving_factors", [], "nothing makes it better or worse"),
        up("symptoms[0].associated_symptoms", [], "nothing else"),
    )


def test_pregnancy_asked_for_female_of_reproductive_age():
    sim = Sim(skip_ayurvedic_policy())
    _through_symptoms(sim)
    assert sim.last.target_slot == "subject.age_years"
    d = sim.user(
        "I'm 29 and female",
        Intent.answer,
        up("subject.age_years", 29, "29"),
        up("subject.sex_at_birth", "female", "female"),
    )
    assert d.target_slot == "subject.pregnancy_status"


def test_pregnancy_not_asked_when_not_applicable():
    sim = Sim(skip_ayurvedic_policy())
    _through_symptoms(sim)
    d = sim.user(
        "I'm 61 and female",
        Intent.answer,
        up("subject.age_years", 61, "61"),
        up("subject.sex_at_birth", "female", "female"),
    )
    assert d.target_slot == "safety_profile.current_medications"


def test_minor_is_out_of_scope():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    d = sim.user("I'm 15", Intent.answer, up("subject.age_years", 15, "15"))
    assert d.final_status is SpecStatus.out_of_scope and d.status_reason == "minor"
    assert d.says[0].params == {"min_age": 18}


def test_evidence_quote_gate_drops_fabricated_values():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user(
        "it's me", Intent.answer, up("reporter_role", "self", "it's me"), up("subject.age_years", 40, "forty")
    )
    assert "subject.age_years" not in sim.session.slots
    assert sim.session.value("reporter_role") == "self"


def test_invalid_and_unknown_updates_are_dropped():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user(
        "I'm 200, severity eleven",
        Intent.answer,
        up("subject.age_years", 200, "200"),
        up("symptoms[0].severity_0_10", 11, "eleven"),
        up("diagnosis", "GERD", "severity"),
    )
    assert sim.session.slots == {}


def test_inferred_value_does_not_resolve_required_slot():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    d = sim.user("hello", Intent.answer, up("reporter_role", "self", "", inferred=True))
    assert d.target_slot == "reporter_role"
    assert sim.session.slots["reporter_role"].confidence <= 0.5


def test_stated_confidence_is_clipped():
    sim = Sim()
    sim.user("yes", Intent.confirm_yes)
    sim.user("me", Intent.answer, up("reporter_role", "self", "me", confidence=1.0))
    assert sim.session.slots["reporter_role"].confidence == 0.85


def test_correction_at_readback_updates_and_reads_back_again():
    sim = run_happy_path(Sim(skip_ayurvedic_policy()))
    assert sim.keys() == ["readback"]
    d = sim.user("No", Intent.confirm_no)
    assert d.says[0].key == "correction_prompt" and d.phase is Phase.CORRECTION
    d = sim.user(
        "it's been four weeks",
        Intent.correction,
        up("symptoms[0].duration", {"value": 4, "unit": "weeks", "raw_text": "four weeks"}, "four weeks"),
    )
    assert d.says[-1].key == "readback"
    assert sim.session.value("symptoms[0].duration")["value"] == 4


def test_yes_with_restated_values_still_confirms():
    sim = run_happy_path(Sim(skip_ayurvedic_policy()))
    d = sim.user("Yes I'm 34", Intent.confirm_yes, up("subject.age_years", 34, "34"))
    assert d.submit and d.final_status is SpecStatus.complete


def test_readback_cycles_are_bounded():
    sim = run_happy_path(Sim(IntakePolicy(ayurvedic_context_enabled=False, max_readback_cycles=2)))
    sim.user("no", Intent.confirm_no)
    sim.user("hmm", Intent.unclear)  # correction with nothing -> readback cycle 2
    assert sim.keys() == ["readback"]
    sim.user("no", Intent.confirm_no)
    d = sim.user("hmm", Intent.unclear)
    assert d.submit and d.final_status is SpecStatus.complete


def test_turn_budget_skips_optional_slots():
    policy = IntakePolicy(max_user_turns=10, ayurvedic_context_enabled=True)
    sim = run_happy_path(Sim(policy))
    asked = [d.target_slot for d in sim.decisions if d.target_slot]
    assert not any(a.startswith("ayurvedic_context") for a in asked)
    assert sim.session.phase in (Phase.READBACK, Phase.CLOSED)


def test_ayurvedic_context_asked_when_enabled_and_budget_allows():
    sim = run_happy_path(Sim(IntakePolicy(ayurvedic_context_enabled=True)))
    assert sim.last.target_slot == "ayurvedic_context.appetite"
