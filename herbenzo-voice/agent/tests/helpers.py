"""Test helpers: drive the deterministic core turn by turn without any LLM."""

from __future__ import annotations

from typing import Any

from herbenzo_agent.contracts.symptom_spec import Provenance
from herbenzo_agent.intake.models import (
    Decision,
    ExtractionResult,
    IntakePolicy,
    IntakeSession,
    Intent,
    SlotUpdate,
)
from herbenzo_agent.intake.planner import decide, start_session
from herbenzo_agent.intake.red_flags import screen_text


def provenance() -> Provenance:
    return Provenance(
        agent_version="test",
        graph_version="test",
        llm_models={"extractor": "fake", "responder": "fake"},
        prompt_sha256={"scripts": "test"},
        guardrails_config_sha256="disabled",
        transcript_ref="turns:test",
    )


def up(slot_key: str, value: Any, quote: str, confidence: float = 0.8, **kw: Any) -> SlotUpdate:
    return SlotUpdate(slot_key=slot_key, value=value, evidence_quote=quote, confidence=confidence, **kw)


class Sim:
    def __init__(self, policy: IntakePolicy | None = None, **session_kw: Any):
        self.policy = policy or IntakePolicy()
        self.session = IntakeSession(session_id=session_kw.pop("session_id", "s1"), **session_kw)
        self.decisions: list[Decision] = [start_session(self.session)]

    @property
    def last(self) -> Decision:
        return self.decisions[-1]

    def user(
        self, text: str, intent: Intent = Intent.answer, *updates: SlotUpdate, model_codes=()
    ) -> Decision:
        turn = self.session.add_turn("user", text)
        flags = screen_text(text, turn.turn_id)
        extraction = ExtractionResult(
            updates=list(updates), intent=intent, red_flag_suspected=list(model_codes)
        )
        decision = decide(self.session, turn, extraction, flags, self.policy)
        self.decisions.append(decision)
        return decision

    def keys(self) -> list[str]:
        return [s.key for s in self.last.says]


def run_happy_path(sim: Sim) -> Sim:
    """Adult male, burning acidity for three weeks, one medicine, no allergies or conditions."""
    sim.user("Yes, that's fine", Intent.confirm_yes)
    sim.user("It's for myself", Intent.answer, up("reporter_role", "self", "myself"))
    sim.user(
        "I've got burning acidity, worse after spicy food",
        Intent.answer,
        up("chief_complaint.verbatim", "I've got burning acidity, worse after spicy food", "burning acidity"),
        up("chief_complaint.summary", "burning acidity", "burning acidity"),
        up("chief_complaint.body_system", "digestive", "acidity"),
        up("symptoms[0].character", "burning", "burning"),
        up("symptoms[0].aggravating_factors", ["spicy food"], "spicy food"),
    )
    sim.user(
        "About three weeks",
        Intent.answer,
        up(
            "symptoms[0].duration",
            {"value": 3, "unit": "weeks", "raw_text": "about three weeks"},
            "three weeks",
        ),
    )
    sim.user("Maybe a six", Intent.answer, up("symptoms[0].severity_0_10", 6, "six"))
    sim.user(
        "After meals mostly", Intent.answer, up("symptoms[0].frequency_pattern", "after meals", "after meals")
    )
    sim.user(
        "Cold milk helps", Intent.answer, up("symptoms[0].relieving_factors", ["cold milk"], "cold milk")
    )
    sim.user("Nothing else", Intent.answer, up("symptoms[0].associated_symptoms", [], "nothing else"))
    sim.user("I'm 34", Intent.answer, up("subject.age_years", 34, "34"))
    sim.user("Male", Intent.answer, up("subject.sex_at_birth", "male", "male"))
    sim.user(
        "Just pantoprazole",
        Intent.answer,
        up(
            "safety_profile.current_medications",
            [{"name": "pantoprazole", "kind": "prescription"}],
            "pantoprazole",
        ),
    )
    sim.user("No allergies", Intent.answer, up("safety_profile.allergies", [], "no allergies"))
    sim.user("None", Intent.answer, up("safety_profile.chronic_conditions", [], "none"))
    sim.user("No", Intent.answer, up("safety_profile.recent_or_planned_surgery", False, "no"))
    return sim


def skip_ayurvedic_policy() -> IntakePolicy:
    return IntakePolicy(ayurvedic_context_enabled=False)
