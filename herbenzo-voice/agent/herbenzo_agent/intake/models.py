"""In-memory models for one intake conversation.

These are working types for the deterministic core (planner, readback, spec builder). They are
serialised to plain dicts inside the LangGraph state, and converted to the immutable
`SymptomSpec` contract only at submission time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from herbenzo_agent.contracts.symptom_spec import CaptureMode, RedFlag, SpecStatus

CONSENT_TEXT_VERSION = "consent-en-2026-09-v1"

STATED_CONFIDENCE_RANGE = (0.5, 0.85)
INFERRED_CONFIDENCE_MAX = 0.5
CONFIRMED_CONFIDENCE_MIN = 0.95


class Phase(StrEnum):
    GREETING_CONSENT = "greeting_consent"
    COLLECTING = "collecting"
    READBACK = "readback"
    CORRECTION = "correction"
    CONFIRM_RESTART = "confirm_restart"
    CLOSED = "closed"
    ESCALATED = "escalated"


TERMINAL_PHASES = frozenset({Phase.CLOSED, Phase.ESCALATED})


class Intent(StrEnum):
    answer = "answer"
    correction = "correction"
    confirm_yes = "confirm_yes"
    confirm_no = "confirm_no"
    restart = "restart"
    stop = "stop"
    asks_advice = "asks_advice"
    off_topic = "off_topic"
    unclear = "unclear"


class SlotUpdate(BaseModel):
    slot_key: str
    value: Any = None
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence_quote: str = ""
    inferred: bool = False
    declined: bool = False


class ExtractionResult(BaseModel):
    updates: list[SlotUpdate] = Field(default_factory=list)
    intent: Intent = Intent.answer
    red_flag_suspected: list[str] = Field(default_factory=list)
    # True when the extractor itself failed (timeout, provider error, unparseable output). The user did
    # answer, so the planner must ask them to repeat without spending one of their attempts.
    failed: bool = False


class Turn(BaseModel):
    turn_id: str
    role: Literal["user", "assistant"]
    text: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    asr_confidence: float | None = None
    delivered: bool = True


class SlotValue(BaseModel):
    value: Any = None
    confidence: float
    capture: CaptureMode
    turn_ids: list[str]


class IntakePolicy(BaseModel):
    min_adult_age: int = 18
    max_user_turns: int = 30
    ayurvedic_context_enabled: bool = True
    max_readback_cycles: int = 3
    max_consent_asks: int = 2
    optional_budget_fraction: float = 0.8


class Say(BaseModel):
    """One thing the agent must say. `ask` is phrased by the responder; the rest are scripted."""

    kind: Literal["script", "ask", "readback"]
    key: str
    params: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    says: list[Say]
    phase: Phase
    target_slot: str | None = None
    final_status: SpecStatus | None = None
    status_reason: str | None = None
    restart: bool = False
    submit: bool = False


class IntakeSession(BaseModel):
    session_id: str
    channel: Literal["voice", "chat"] = "chat"
    language: str = "en-IN"
    jurisdiction: str = "IN"
    phase: Phase = Phase.GREETING_CONSENT
    resume_phase: Phase | None = None
    turns: list[Turn] = Field(default_factory=list)
    slots: dict[str, SlotValue] = Field(default_factory=dict)
    ask_counts: dict[str, int] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    target_slot: str | None = None
    consent_granted: bool = False
    consent_turn_id: str | None = None
    consent_at: datetime | None = None
    consent_asks: int = 0
    readback_cycles: int = 0
    red_flags: list[RedFlag] = Field(default_factory=list)
    restart_count: int = 0
    final_status: SpecStatus | None = None
    status_reason: str | None = None
    spec_id: str | None = None
    asr_model: str | None = None
    tts_model: str | None = None

    # --- turns ----------------------------------------------------------------------------------

    def next_turn_id(self) -> str:
        return f"{self.session_id}:{len(self.turns) + 1}"

    def add_turn(self, role: Literal["user", "assistant"], text: str, **kw: Any) -> Turn:
        turn = Turn(turn_id=self.next_turn_id(), role=role, text=text, **kw)
        self.turns.append(turn)
        return turn

    @property
    def user_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def user_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.role == "user")

    # --- slots ----------------------------------------------------------------------------------

    def value(self, key: str) -> Any:
        slot = self.slots.get(key)
        return slot.value if slot else None

    def has(self, key: str) -> bool:
        """True if the slot is resolved: captured (not inferred) or declined."""
        slot = self.slots.get(key)
        return slot is not None and slot.capture is not CaptureMode.inferred

    def set_slot(self, key: str, value: Any, confidence: float, capture: CaptureMode, turn_id: str) -> None:
        if capture is CaptureMode.declined:
            value, confidence = None, 0.0
        elif capture is CaptureMode.stated:
            lo, hi = STATED_CONFIDENCE_RANGE
            confidence = min(max(confidence, lo), hi)
        elif capture is CaptureMode.inferred:
            confidence = min(confidence, INFERRED_CONFIDENCE_MAX)
        existing = self.slots.get(key)
        turn_ids = [turn_id]
        if existing and existing.value == value and turn_id not in existing.turn_ids:
            turn_ids = [*existing.turn_ids, turn_id]
        self.slots[key] = SlotValue(value=value, confidence=confidence, capture=capture, turn_ids=turn_ids)

    def confirm_all(self, turn_id: str) -> None:
        for slot in self.slots.values():
            if slot.capture is CaptureMode.stated:
                slot.capture = CaptureMode.confirmed
                slot.confidence = max(slot.confidence, CONFIRMED_CONFIDENCE_MIN)
                if turn_id not in slot.turn_ids:
                    slot.turn_ids.append(turn_id)
