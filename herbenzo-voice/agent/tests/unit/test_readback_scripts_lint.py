import re

import pytest

from herbenzo_agent.contracts.symptom_spec import CaptureMode
from herbenzo_agent.intake import scripts
from herbenzo_agent.intake.lint import lint_response
from herbenzo_agent.intake.models import IntakeSession
from herbenzo_agent.intake.readback import build_readback
from herbenzo_agent.intake.slots import SLOTS
from tests.helpers import Sim, run_happy_path, skip_ayurvedic_policy

# --- readback ------------------------------------------------------------------------------------


def test_readback_for_self():
    chunks = build_readback(run_happy_path(Sim(skip_ayurvedic_policy())).session)
    assert chunks == [
        "Here's what I have. You're 34, and you've had burning acidity for about three weeks, worse with spicy food, "
        "better with cold milk, about six out of ten at its worst.",
        "You take pantoprazole. You have no allergies and no long-term conditions.",
        "Is all of that correct?",
    ]


def test_readback_for_caregiver_with_declined_and_pregnancy():
    s = IntakeSession(session_id="c1")
    s.set_slot("reporter_role", "caregiver", 0.8, CaptureMode.stated, "c1:1")
    s.set_slot("chief_complaint.summary", "joint pain", 0.8, CaptureMode.stated, "c1:2")
    s.set_slot(
        "symptoms[0].duration",
        {"value": 2, "unit": "months", "raw_text": "since last monsoon"},
        0.8,
        CaptureMode.stated,
        "c1:3",
    )
    s.set_slot("subject.age_years", 52, 0.8, CaptureMode.stated, "c1:4")
    s.set_slot("subject.pregnancy_status", "none", 0.8, CaptureMode.stated, "c1:5")
    s.set_slot("safety_profile.current_medications", None, 0.0, CaptureMode.declined, "c1:6")
    s.set_slot("safety_profile.allergies", ["penicillin", "sulfa drugs"], 0.8, CaptureMode.stated, "c1:7")
    s.set_slot("safety_profile.chronic_conditions", ["diabetes"], 0.8, CaptureMode.stated, "c1:8")
    chunks = build_readback(s)
    assert chunks[0] == "Here's what I have. They're 52, and they've had joint pain since last monsoon."
    assert chunks[1] == (
        "They're allergic to penicillin and sulfa drugs, and they have diabetes as a long-term condition. "
        "They're not pregnant or breastfeeding."
    )
    assert chunks[2] == "They preferred not to say their medicines. Is all of that correct?"


def test_readback_chunks_are_short():
    for chunk in build_readback(run_happy_path(Sim(skip_ayurvedic_policy())).session):
        assert len(re.findall(r"[.?!](\s|$)", chunk)) <= 2


# --- scripts -------------------------------------------------------------------------------------


@pytest.mark.parametrize("channel", ["voice", "chat"])
@pytest.mark.parametrize("key", sorted(scripts.SCRIPTS))
def test_every_script_renders(key, channel):
    text = scripts.render(key, IntakeSession(session_id="x", channel=channel), min_age=18)
    assert "{" not in text and "}" not in text


def test_voice_reads_numbers_as_words_and_chat_as_digits():
    voice = scripts.render("escalation", IntakeSession(session_id="x", channel="voice"))
    chat = scripts.render("escalation", IntakeSession(session_id="x", channel="chat"))
    assert "one one two" in voice and "112" not in voice
    assert "112" in chat and "108" in chat


def test_scripts_hash():
    assert len(scripts.scripts_sha256()) == 64


# --- lint ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("slot", SLOTS, ids=lambda s: s.key)
def test_base_questions_pass_lint(slot):
    assert lint_response(slot.question) == []
    if slot.about_other:
        assert lint_response(slot.about_other) == []


@pytest.mark.parametrize(
    "text,problem",
    [
        ("Thanks. You should try ashwagandha. How old are you?", "blocked_term"),
        ("It sounds like an ulcer. How long has it lasted?", "blocked_term"),
        ("Take 500 mg daily. Is that okay?", "blocked_term"),
        ("How old are you? And your sex?", "must_contain_exactly_one_question"),
        ("**How old are you?**", "markdown"),
        ("How old are you? 🙂", "emoji"),
        ("Is it about 7 out of 10?", "number_not_from_user"),
        ("", "empty"),
    ],
)
def test_lint_rejects(text, problem):
    assert any(p.startswith(problem) for p in lint_response(text))


def test_lint_allows_numbers_the_user_said():
    assert lint_response("Thanks. You mentioned 3 weeks, is it constant?", user_text="about 3 weeks") == []
