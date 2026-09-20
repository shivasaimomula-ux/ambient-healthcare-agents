"""Regressions for defects found in live runs against the real model (2026-09-16)."""

import pytest

from herbenzo_agent.intake.lint import lint_response
from herbenzo_agent.intake.models import Intent
from herbenzo_agent.intake.quick_intent import quick_confirmation
from tests.helpers import Sim, skip_ayurvedic_policy, up


@pytest.mark.parametrize(
    "text,intent",
    [
        ("yes, go ahead", Intent.confirm_yes),
        ("Yes that's correct", Intent.confirm_yes),
        ("okay sure", Intent.confirm_yes),
        ("haan ji", Intent.confirm_yes),
        ("yes, that's all correct thank you", Intent.confirm_yes),
        ("no", Intent.confirm_no),
        ("no that's not right", Intent.confirm_no),
        ("nope", Intent.confirm_no),
    ],
)
def test_quick_confirmation_detects_plain_yes_no(text, intent):
    assert quick_confirmation(text) is intent


@pytest.mark.parametrize(
    "text",
    [
        "no, it's been four weeks",
        "yes I'm 34",
        "I have no allergies",
        "yes but my pain is worse",
        "not sure",
        "",
    ],
)
def test_quick_confirmation_leaves_substantive_answers_to_llm(text):
    assert quick_confirmation(text) is None


def test_lint_rejects_observed_prompt_leak():
    leaked = "Thanks. I'll output only the assistant says: \"You can call me Nemotron. How old are you?"
    problems = lint_response(leaked, base_question="How old are you?")
    assert any(p.startswith("blocked_term") for p in problems)
    assert "quote_or_colon" in problems


def test_lint_rejects_changed_question_meaning():
    assert "question_meaning_changed" in lint_response(
        "Thanks. Do you have any allergies?", base_question="How long has this been going on?"
    )
    assert (
        lint_response(
            "Thanks. How long have you had this going on?", base_question="How long has this been going on?"
        )
        == []
    )


def test_off_target_none_for_safety_slot_needs_topic_word():
    sim = Sim(skip_ayurvedic_policy())
    sim.user("yes", Intent.confirm_yes)
    sim.session.target_slot = "safety_profile.current_medications"
    sim.user(
        "just pantoprazole 40 mg",
        Intent.answer,
        up("safety_profile.current_medications", ["pantoprazole"], "pantoprazole"),
        up("safety_profile.allergies", [], "just pantoprazole 40 mg"),  # the observed hallucination
    )
    assert "safety_profile.allergies" not in sim.session.slots
    assert sim.session.value("safety_profile.current_medications")[0]["name"] == "pantoprazole"


def test_off_target_none_accepted_when_user_names_topic():
    sim = Sim(skip_ayurvedic_policy())
    sim.user("yes", Intent.confirm_yes)
    sim.session.target_slot = "safety_profile.current_medications"
    sim.user(
        "just pantoprazole, and no allergies",
        Intent.answer,
        up("safety_profile.current_medications", ["pantoprazole"], "pantoprazole"),
        up("safety_profile.allergies", [], "no allergies"),
    )
    assert sim.session.value("safety_profile.allergies") == []


def test_extractor_failure_does_not_spend_an_attempt():
    from herbenzo_agent.intake.models import ExtractionResult
    from herbenzo_agent.intake.planner import decide

    sim = Sim(skip_ayurvedic_policy())
    sim.user("yes", Intent.confirm_yes)
    assert sim.session.ask_counts["reporter_role"] == 1
    for _ in range(3):
        turn = sim.session.add_turn("user", "it's for me")
        d = decide(sim.session, turn, ExtractionResult(intent=Intent.unclear, failed=True), [], sim.policy)
        assert [s.key for s in d.says] == ["system_retry"]
    assert sim.session.ask_counts["reporter_role"] == 1
    assert "reporter_role" not in sim.session.slots


def test_deflection_does_not_spend_an_attempt():
    sim = Sim(skip_ayurvedic_policy())
    sim.user("yes", Intent.confirm_yes)
    sim.user("what herb should I take?", Intent.asks_advice)
    sim.user("and what dose?", Intent.asks_advice)
    assert sim.session.ask_counts["reporter_role"] == 1
    assert "reporter_role" not in sim.session.slots
    assert sim.keys() == ["deflect_advice", "reporter_role"]


def test_readback_with_unchanged_answers_terminates():
    """Live E2E run: repeating an already-captured value at readback looped the readback forever."""
    from herbenzo_agent.contracts.symptom_spec import SpecStatus
    from tests.helpers import run_happy_path

    sim = run_happy_path(Sim(skip_ayurvedic_policy()))
    assert sim.keys() == ["readback"]
    decisions = [
        sim.user("no allergies", Intent.answer, up("safety_profile.allergies", [], "no allergies"))
        for _ in range(5)
    ]
    assert any(d.submit for d in decisions), [d.says for d in decisions]
    assert sim.session.final_status is SpecStatus.complete
    assert (
        sum(1 for d in decisions if d.says and d.says[-1].key == "readback") <= sim.policy.max_readback_cycles
    )
