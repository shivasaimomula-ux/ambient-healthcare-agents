"""LangGraph wiring for one intake turn.

    START -> ingest -> screen -> extract -> plan -> render -> END
                  \\-> render (session start / empty input)
                          screen -> plan (red flag: skip extraction)

Conversation memory is the `session` dict in graph state, persisted by the checkpointer under
thread_id = the caller's session id (voice: WebRTC pc_id; chat: generated id).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from herbenzo_agent.audit.reproducibility import build_provenance
from herbenzo_agent.contracts.symptom_spec import RedFlag, SpecStatus, SymptomSpec
from herbenzo_agent.guardrails.nemoguard import Guardrails, NoopGuardrails
from herbenzo_agent.handoff.worker import Handoff, NoopHandoff
from herbenzo_agent.intake import scripts
from herbenzo_agent.intake.extractor import Extractor
from herbenzo_agent.intake.models import (
    TERMINAL_PHASES,
    Decision,
    ExtractionResult,
    IntakePolicy,
    IntakeSession,
    Phase,
    Say,
)
from herbenzo_agent.intake.planner import CLOSING_PLACEHOLDER, decide, decide_blocked, start_session
from herbenzo_agent.intake.quick_intent import quick_confirmation
from herbenzo_agent.intake.readback import build_readback
from herbenzo_agent.intake.red_flag_model import RedFlagClassifier
from herbenzo_agent.intake.red_flags import merge_model_codes, needs_model_check, screen_text
from herbenzo_agent.intake.responder import Responder
from herbenzo_agent.intake.slots import get_slot
from herbenzo_agent.intake.spec_builder import build_spec
from herbenzo_agent.observability import TURN_COUNTERS, TURN_TIMINGS, count, timed, timed_await
from herbenzo_agent.persistence.spec_sink import SpecSink

logger = logging.getLogger(__name__)

SESSION_START = "__SESSION_START__"
CONFIRMATION_PHASES = frozenset({Phase.GREETING_CONSENT, Phase.READBACK, Phase.CONFIRM_RESTART})


@dataclass
class IntakeDeps:
    extractor: Extractor
    responder: Responder
    red_flag_classifier: RedFlagClassifier
    spec_sink: SpecSink
    guardrails: Guardrails = field(default_factory=NoopGuardrails)
    handoff: Handoff = field(default_factory=NoopHandoff)
    policy: IntakePolicy = field(default_factory=IntakePolicy)
    llm_models: dict[str, str] = field(default_factory=dict)
    handoff_enabled: bool = True
    handoff_min_confidence: float = 0.5
    default_language: str = "en-IN"
    default_jurisdiction: str = "IN"
    asr_model: str | None = None
    tts_model: str | None = None


class TurnState(TypedDict, total=False):
    # persisted conversation
    session: dict[str, Any]
    # input for this turn
    user_text: str
    channel: Literal["voice", "chat"]
    asr_confidence: float | None
    speech_models: dict[str, str]
    # per-turn scratch (overwritten every turn)
    turn_id: str | None
    red_flags: list[dict[str, Any]]
    extraction: dict[str, Any]
    input_blocked: bool
    decision: dict[str, Any]
    reply_chunks: list[str]
    spec_ids: list[str]


def _load(state: TurnState, thread_id: str, deps: IntakeDeps) -> IntakeSession:
    if state.get("session"):
        return IntakeSession.model_validate(state["session"])
    return IntakeSession(
        session_id=thread_id,
        channel=state.get("channel", "chat"),
        language=deps.default_language,
        jurisdiction=deps.default_jurisdiction,
    )


async def submit_spec(deps: IntakeDeps, session: IntakeSession) -> tuple[SymptomSpec, bool]:
    """Build, persist and hand off the spec for a finished session. Returns (spec, handoff_eligible)."""
    provenance = build_provenance(
        session_id=session.session_id,
        llm_models=deps.llm_models,
        guardrails_config_sha256=deps.guardrails.config_sha256,
        asr_model=(session.asr_model or deps.asr_model) if session.channel == "voice" else None,
        tts_model=(session.tts_model or deps.tts_model) if session.channel == "voice" else None,
    )
    spec = build_spec(session, provenance, deps.policy)
    session.spec_id = str(spec.spec_id)
    await deps.spec_sink.save(spec, session)
    logger.info("spec %s saved: status=%s floor=%.2f", spec.spec_id, spec.status.value, spec.confidence_floor)
    eligible = deps.handoff_enabled and spec.is_handoff_eligible(deps.handoff_min_confidence)
    await deps.handoff.submit(spec, eligible, None if eligible else _ineligible_reason(deps, spec))
    return spec, eligible


def _ineligible_reason(deps: IntakeDeps, spec: SymptomSpec) -> str:
    if not deps.handoff_enabled:
        return "handoff_disabled"
    if spec.status is not SpecStatus.complete:
        return f"status:{spec.status.value}"
    return f"confidence_floor:{spec.confidence_floor:.2f}<{deps.handoff_min_confidence:.2f}"


def build_intake_graph(deps: IntakeDeps, checkpointer: Any = None):
    async def ingest(state: TurnState, config) -> TurnState:
        thread_id = config["configurable"]["thread_id"]
        session = _load(state, thread_id, deps)
        speech = state.get("speech_models") or {}
        session.asr_model = speech.get("asr") or session.asr_model
        session.tts_model = speech.get("tts") or session.tts_model
        text = (state.get("user_text") or "").strip()
        scratch: TurnState = {
            "turn_id": None,
            "red_flags": [],
            "extraction": {},
            "input_blocked": False,
            "reply_chunks": [],
            "spec_ids": [],
        }

        if text == SESSION_START:
            if not session.turns:
                decision = start_session(session)
            else:  # reconnect: repeat what the user last heard
                last = next((t.text for t in reversed(session.turns) if t.role == "assistant"), "")
                decision = Decision(says=[], phase=session.phase)
                scratch["reply_chunks"] = [last] if last else []
            return {
                **scratch,
                "session": session.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }

        if not text:
            decision = Decision(says=[Say(kind="script", key="empty_input")], phase=session.phase)
            return {
                **scratch,
                "session": session.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
            }

        turn = session.add_turn("user", text[:2000], asr_confidence=state.get("asr_confidence"))
        return {
            **scratch,
            "session": session.model_dump(mode="json"),
            "turn_id": turn.turn_id,
            "decision": {},
        }

    def after_ingest(state: TurnState) -> str:
        return "screen" if state.get("turn_id") else "render"

    async def screen(state: TurnState) -> TurnState:
        session = IntakeSession.model_validate(state["session"])
        turn = next(t for t in session.turns if t.turn_id == state["turn_id"])
        with timed("red_flag_rules"):
            flags = screen_text(turn.text, turn.turn_id)
        if not flags and needs_model_check(turn.text):
            codes = await timed_await("red_flag_model", deps.red_flag_classifier.classify(turn.text))
            flags = merge_model_codes(flags, codes, turn.text, turn.turn_id)
        return {"red_flags": [f.model_dump() for f in flags]}

    def after_screen(state: TurnState) -> str:
        return "plan" if state.get("red_flags") else "extract"

    async def extract(state: TurnState) -> TurnState:
        """Input safety rail and extraction run concurrently; a blocked message discards the extraction."""
        session = IntakeSession.model_validate(state["session"])
        turn = next(t for t in session.turns if t.turn_id == state["turn_id"])
        if session.phase in CONFIRMATION_PHASES and (intent := quick_confirmation(turn.text)):
            return {
                "extraction": ExtractionResult(intent=intent).model_dump(mode="json"),
                "input_blocked": False,
            }
        with timed("understand"):
            guard, extraction = await asyncio.gather(
                timed_await("guardrail_input", deps.guardrails.check_input(turn.text)),
                timed_await("extractor", deps.extractor.extract(session, turn)),
            )
        count(f"guardrail_input.{guard.status.value}")
        if guard.blocked:
            logger.info("input blocked by %s", guard.rail)
            return {"extraction": {}, "input_blocked": True}
        return {"extraction": extraction.model_dump(mode="json"), "input_blocked": False}

    async def plan(state: TurnState) -> TurnState:
        session = IntakeSession.model_validate(state["session"])
        turn = next(t for t in session.turns if t.turn_id == state["turn_id"])
        extraction = ExtractionResult.model_validate(state.get("extraction") or {})
        flags = [RedFlag.model_validate(f) for f in state.get("red_flags") or []]
        # Model-suggested codes from the extractor also escalate (restricted to the lexicon).
        if not flags and extraction.red_flag_suspected:
            flags = merge_model_codes([], extraction.red_flag_suspected, turn.text, turn.turn_id)
        if not flags and state.get("input_blocked"):
            decision = decide_blocked(session)
        else:
            decision = decide(session, turn, extraction, flags, deps.policy)
        return {"session": session.model_dump(mode="json"), "decision": decision.model_dump(mode="json")}

    async def render(state: TurnState) -> TurnState:
        session = IntakeSession.model_validate(state["session"])
        decision = Decision.model_validate(state["decision"]) if state.get("decision") else None
        chunks: list[str] = list(state.get("reply_chunks") or [])
        spec_ids: list[str] = []
        if decision is None:
            return {"reply_chunks": chunks}

        eligible = False
        if decision.submit:
            spec, eligible = await timed_await("submit_spec", submit_spec(deps, session))
            spec_ids.append(str(spec.spec_id))

        if decision.restart:
            restarted = IntakeSession(
                session_id=f"{session.session_id.split('#')[0]}#r{session.restart_count + 1}",
                channel=session.channel,
                language=session.language,
                jurisdiction=session.jurisdiction,
                restart_count=session.restart_count + 1,
            )
            session = restarted
            decision = start_session(session)

        acknowledge = bool(state.get("turn_id")) and not any(s.kind == "script" for s in decision.says)
        for say in decision.says:
            chunks.extend(await _render_say(session, say, eligible, acknowledge))
        if chunks:
            session.add_turn("assistant", " ".join(chunks))
        return {"session": session.model_dump(mode="json"), "reply_chunks": chunks, "spec_ids": spec_ids}

    async def _render_say(session: IntakeSession, say: Say, eligible: bool, acknowledge: bool) -> list[str]:
        if say.kind == "readback":
            return build_readback(session)
        if say.kind == "ask":
            slot = get_slot(say.key)
            question = slot.question_for(session) if slot else say.key
            return [await timed_await("responder", deps.responder.phrase(session, question, acknowledge))]
        key = say.key
        if key == CLOSING_PLACEHOLDER:
            key = "closing_eligible" if eligible else "closing_not_eligible"
        return [scripts.render(key, session, **say.params)]

    builder = StateGraph(TurnState)
    builder.add_node("ingest", ingest)
    builder.add_node("screen", screen)
    builder.add_node("extract", extract)
    builder.add_node("plan", plan)
    builder.add_node("render", render)
    builder.add_edge(START, "ingest")
    builder.add_conditional_edges("ingest", after_ingest, {"screen": "screen", "render": "render"})
    builder.add_conditional_edges("screen", after_screen, {"plan": "plan", "extract": "extract"})
    builder.add_edge("extract", "plan")
    builder.add_edge("plan", "render")
    builder.add_edge("render", END)
    return builder.compile(checkpointer=checkpointer)


@dataclass
class TurnResult:
    session_id: str
    reply_chunks: list[str]
    phase: str
    final_status: SpecStatus | None
    spec_ids: list[str]
    timings: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)

    @property
    def reply(self) -> str:
        return " ".join(self.reply_chunks)


class IntakeService:
    """Thin async facade over the compiled graph, keyed by thread id."""

    def __init__(self, graph, deps: IntakeDeps | None = None):
        self.graph = graph
        self.deps = deps

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    async def turn(
        self,
        thread_id: str,
        text: str,
        channel: Literal["voice", "chat"] = "chat",
        asr_confidence: float | None = None,
        speech_models: dict[str, str] | None = None,
    ) -> TurnResult:
        timings: dict[str, float] = {}
        counters: dict[str, int] = {}
        token = TURN_TIMINGS.set(timings)
        counter_token = TURN_COUNTERS.set(counters)
        try:
            with timed("turn_total"):
                out = await self.graph.ainvoke(
                    {
                        "user_text": text,
                        "channel": channel,
                        "asr_confidence": asr_confidence,
                        "speech_models": speech_models or {},
                    },
                    self._config(thread_id),
                )
        finally:
            TURN_TIMINGS.reset(token)
            TURN_COUNTERS.reset(counter_token)
        touch = getattr(self.deps.spec_sink, "touch_session", None) if self.deps else None
        if touch is not None:
            await touch(thread_id, channel)  # activity drives retention purging
        session = IntakeSession.model_validate(out["session"])
        return TurnResult(
            session_id=session.session_id,
            reply_chunks=out.get("reply_chunks") or [],
            phase=session.phase.value,
            final_status=session.final_status,
            spec_ids=out.get("spec_ids") or [],
            timings=timings,
            counters=counters,
        )

    async def get_session(self, thread_id: str) -> IntakeSession | None:
        snapshot = await self.graph.aget_state(self._config(thread_id))
        data = snapshot.values.get("session") if snapshot and snapshot.values else None
        return IntakeSession.model_validate(data) if data else None

    async def close(self, thread_id: str, reason: str = "disconnected") -> str | None:
        """End a session that stopped mid-intake (e.g. the caller hung up). Returns the spec id if one was saved.

        Nothing is stored for a session that never consented; a finished or escalated session is left as is.
        """
        if self.deps is None:
            raise RuntimeError("IntakeService.close needs deps")
        session = await self.get_session(thread_id)
        if session is None or session.phase in TERMINAL_PHASES:
            return None
        session.phase = Phase.CLOSED
        session.target_slot = None
        session.final_status = SpecStatus.incomplete
        session.status_reason = reason
        spec_id = None
        if session.consent_granted:
            spec, _ = await submit_spec(self.deps, session)
            spec_id = str(spec.spec_id)
        await self.graph.aupdate_state(self._config(thread_id), {"session": session.model_dump(mode="json")})
        return spec_id
