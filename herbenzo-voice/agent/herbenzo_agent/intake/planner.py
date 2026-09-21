"""The deterministic conversation planner.

Given the session, the user's latest turn, what the extractor understood and any red flags, decide
what happens next. No I/O and no LLM calls here: every branch is unit-testable.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import ValidationError

from herbenzo_agent.contracts.symptom_spec import CaptureMode, RedFlag, SpecStatus
from herbenzo_agent.intake.models import (
    TERMINAL_PHASES,
    Decision,
    ExtractionResult,
    IntakePolicy,
    IntakeSession,
    Intent,
    Phase,
    Say,
    Turn,
)
from herbenzo_agent.intake.slots import SLOTS, get_slot

CLOSING_PLACEHOLDER = "__closing__"
_SPACES = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text.lower().replace("’", "'")).strip()


def start_session(session: IntakeSession) -> Decision:
    session.phase = Phase.GREETING_CONSENT
    return Decision(says=[Say(kind="script", key="greeting_consent")], phase=session.phase)


# --- applying extraction --------------------------------------------------------------------------


def apply_extraction(session: IntakeSession, extraction: ExtractionResult, turn: Turn) -> list[str]:
    """Write validated updates into slots. Returns the keys whose value actually changed.

    An update is dropped when its slot is unknown, its value fails the slot type, or (unless it is a
    decline or explicitly inferred) its evidence quote is not a verbatim part of the user's turn.
    """
    applied: list[str] = []
    user_text = _norm(turn.text)
    for update in extraction.updates:
        slot = get_slot(update.slot_key)
        if slot is None:
            continue
        previous = session.slots.get(slot.key)
        if _unsupported_negative(session, slot, update, user_text):
            continue
        if update.declined:
            session.set_slot(slot.key, None, 0.0, CaptureMode.declined, turn.turn_id)
            if previous is None or previous.capture is not CaptureMode.declined:
                applied.append(slot.key)
            continue
        quote = _norm(update.evidence_quote)
        if not update.inferred and (not quote or quote not in user_text):
            continue
        try:
            value = slot.parse(update.value)
        except (ValidationError, ValueError, TypeError, KeyError):
            continue
        capture = CaptureMode.inferred if update.inferred else CaptureMode.stated
        if (
            previous is not None
            and previous.value == value
            and previous.capture in (CaptureMode.stated, CaptureMode.confirmed)
            and capture is CaptureMode.stated
        ):
            continue  # restating a known value is not a change (keeps "yes, that's right" a confirmation)
        session.set_slot(slot.key, value, update.confidence, capture, turn.turn_id)
        applied.append(slot.key)

    # The symptom name is the chief complaint summary unless the user named it separately.
    summary = session.slots.get("chief_complaint.summary")
    if summary and summary.value is not None and "symptoms[0].name" not in session.slots:
        session.slots["symptoms[0].name"] = summary.model_copy(deep=True)
    return applied


def _unsupported_negative(session: IntakeSession, slot, update, user_text: str) -> bool:
    """Reject "none"/declined for a safety slot that was not being asked about, unless the user named the topic.

    Stops e.g. "just pantoprazole" (answer about medicines) being recorded as "no allergies".
    """
    if not slot.topic_keywords or slot.key == session.target_slot:
        return False
    negative = update.declined or update.value in ([], False, None)
    return negative and not any(k in user_text for k in slot.topic_keywords)


# --- choosing the next question ---------------------------------------------------------------------


def _last_user_turn_id(session: IntakeSession) -> str:
    for turn in reversed(session.turns):
        if turn.role == "user":
            return turn.turn_id
    return session.turns[-1].turn_id if session.turns else f"{session.session_id}:0"


def next_target(session: IntakeSession, policy: IntakePolicy) -> str | None:
    if session.user_turn_count >= policy.max_user_turns:
        return None
    optional_allowed = session.user_turn_count < policy.max_user_turns * policy.optional_budget_fraction
    for slot in SLOTS:
        if not slot.askable or not slot.applies(session, policy):
            continue
        if session.has(slot.key) or slot.key in session.skipped:
            continue
        if not slot.required and not optional_allowed:
            continue
        if session.ask_counts.get(slot.key, 0) >= slot.max_asks:
            if slot.required:
                session.set_slot(slot.key, None, 0.0, CaptureMode.declined, _last_user_turn_id(session))
            else:
                session.skipped.append(slot.key)
            continue
        return slot.key
    return None


def _ask_next(session: IntakeSession, policy: IntakePolicy, prefix: list[Say] | None = None) -> Decision:
    says = list(prefix or [])
    target = next_target(session, policy)
    if target is None:
        if session.readback_cycles >= policy.max_readback_cycles:
            return _submit(session, says)
        session.phase = Phase.READBACK
        session.target_slot = None
        session.readback_cycles += 1
        return Decision(says=[*says, Say(kind="readback", key="readback")], phase=session.phase)
    session.phase = Phase.COLLECTING
    session.target_slot = target
    session.ask_counts[target] = session.ask_counts.get(target, 0) + 1
    return Decision(says=[*says, Say(kind="ask", key=target)], phase=session.phase, target_slot=target)


# --- terminal transitions ---------------------------------------------------------------------------


def _close(
    session: IntakeSession, status: SpecStatus, reason: str | None, says: list[Say], submit: bool
) -> Decision:
    session.phase = Phase.CLOSED
    session.target_slot = None
    session.final_status = status
    session.status_reason = reason
    return Decision(says=says, phase=session.phase, final_status=status, status_reason=reason, submit=submit)


def _submit(session: IntakeSession, prefix: list[Say] | None = None) -> Decision:
    says = [*(prefix or []), Say(kind="script", key=CLOSING_PLACEHOLDER)]
    return _close(session, SpecStatus.complete, None, says, submit=True)


def _escalate(session: IntakeSession, flags: list[RedFlag]) -> Decision:
    session.red_flags.extend(flags)
    session.phase = Phase.ESCALATED
    session.target_slot = None
    session.final_status = SpecStatus.escalated_red_flag
    session.status_reason = ",".join(f.code for f in flags)
    key = "escalation_self_harm" if any(f.code == "RF_SELF_HARM" for f in flags) else "escalation"
    return Decision(
        says=[Say(kind="script", key=key)],
        phase=session.phase,
        final_status=session.final_status,
        status_reason=session.status_reason,
        submit=True,
    )


def _minor_check(session: IntakeSession, policy: IntakePolicy) -> Decision | None:
    age = session.value("subject.age_years")
    if age is not None and age < policy.min_adult_age:
        says = [Say(kind="script", key="minor_out_of_scope", params={"min_age": policy.min_adult_age})]
        return _close(session, SpecStatus.out_of_scope, "minor", says, submit=True)
    return None


def decide_blocked(session: IntakeSession) -> Decision:
    """The input rail blocked the message: say so and repeat the pending request, changing nothing else."""
    prefix = [Say(kind="script", key="deflect_unsafe")]
    if session.phase in TERMINAL_PHASES:
        return Decision(says=[Say(kind="script", key="session_closed")], phase=session.phase)
    repeat = {
        Phase.GREETING_CONSENT: Say(kind="script", key="consent_reask"),
        Phase.READBACK: Say(kind="readback", key="readback"),
        Phase.CORRECTION: Say(kind="script", key="correction_prompt"),
        Phase.CONFIRM_RESTART: Say(kind="script", key="confirm_restart"),
    }.get(session.phase)
    if repeat is None and session.target_slot:
        repeat = Say(kind="ask", key=session.target_slot)
    return Decision(
        says=[*prefix, *([repeat] if repeat else [])], phase=session.phase, target_slot=session.target_slot
    )


# --- main entry point -------------------------------------------------------------------------------


def decide(
    session: IntakeSession,
    turn: Turn,
    extraction: ExtractionResult,
    red_flags: list[RedFlag],
    policy: IntakePolicy,
) -> Decision:
    if session.phase in TERMINAL_PHASES:
        return Decision(says=[Say(kind="script", key="session_closed")], phase=session.phase)

    # Emergencies override everything, including consent.
    if red_flags:
        return _escalate(session, red_flags)

    intent = extraction.intent

    if extraction.failed:
        # Our fault, not the user's: repeat the request without touching phase or attempt counts.
        return Decision(
            says=[Say(kind="script", key="system_retry")],
            phase=session.phase,
            target_slot=session.target_slot,
        )

    if session.phase is Phase.CONFIRM_RESTART:
        if intent is Intent.confirm_yes:
            session.phase = Phase.CLOSED
            session.final_status = SpecStatus.incomplete
            session.status_reason = "restarted"
            return Decision(
                says=[Say(kind="script", key="greeting_consent")],
                phase=Phase.GREETING_CONSENT,
                final_status=SpecStatus.incomplete,
                status_reason="restarted",
                restart=True,
                submit=session.consent_granted,
            )
        session.phase = session.resume_phase or Phase.COLLECTING
        session.resume_phase = None
        resume = [Say(kind="script", key="restart_resume")]
        if session.phase is Phase.GREETING_CONSENT:
            return Decision(says=[*resume, Say(kind="script", key="consent_reask")], phase=session.phase)
        if session.phase in (Phase.READBACK, Phase.CORRECTION):
            session.phase = Phase.READBACK
            return Decision(says=[*resume, Say(kind="readback", key="readback")], phase=session.phase)
        if session.target_slot:
            return Decision(
                says=[*resume, Say(kind="ask", key=session.target_slot)],
                phase=session.phase,
                target_slot=session.target_slot,
            )
        return _ask_next(session, policy, resume)

    if intent is Intent.restart:
        session.resume_phase = session.phase
        session.phase = Phase.CONFIRM_RESTART
        return Decision(says=[Say(kind="script", key="confirm_restart")], phase=session.phase)

    if intent is Intent.stop:
        return _close(
            session,
            SpecStatus.incomplete,
            "user_stopped",
            [Say(kind="script", key="stopped")],
            submit=session.consent_granted,
        )

    if session.phase is Phase.GREETING_CONSENT:
        if intent is Intent.confirm_yes:
            session.consent_granted = True
            session.consent_turn_id = turn.turn_id
            session.consent_at = datetime.now(UTC)
            apply_extraction(session, extraction, turn)
            return _minor_check(session, policy) or _ask_next(session, policy)
        session.consent_asks += 1
        if intent is Intent.confirm_no or session.consent_asks >= policy.max_consent_asks:
            return _close(
                session,
                SpecStatus.out_of_scope,
                "consent_refused",
                [Say(kind="script", key="consent_refused")],
                False,
            )
        return Decision(says=[Say(kind="script", key="consent_reask")], phase=session.phase)

    applied = apply_extraction(session, extraction, turn)
    if minor := _minor_check(session, policy):
        return minor

    prefix: list[Say] = []
    if intent is Intent.asks_advice:
        prefix = [Say(kind="script", key="deflect_advice")]
    elif intent is Intent.off_topic and not applied:
        prefix = [Say(kind="script", key="deflect_off_topic")]

    # A deflected question is repeated without spending one of the user's attempts at it.
    if prefix and not applied and session.phase is Phase.COLLECTING and session.target_slot:
        return Decision(
            says=[*prefix, Say(kind="ask", key=session.target_slot)],
            phase=session.phase,
            target_slot=session.target_slot,
        )

    if session.phase is Phase.READBACK:
        if intent is Intent.confirm_yes and not applied:
            session.confirm_all(turn.turn_id)
            return _submit(session)
        if applied:
            return _ask_next(session, policy, prefix)
        if intent in (Intent.confirm_no, Intent.correction):
            session.phase = Phase.CORRECTION
            return Decision(says=[*prefix, Say(kind="script", key="correction_prompt")], phase=session.phase)
        # Anything else (unclear, a restated value, an off-topic aside) re-reads, but every re-read counts
        # toward the cycle limit so the conversation always terminates.
        return _ask_next(session, policy, prefix)

    if session.phase is Phase.CORRECTION:
        return _ask_next(session, policy, prefix)

    return _ask_next(session, policy, prefix)
