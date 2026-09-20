"""Turn a user message into slot updates + intent, via an LLM or a scripted fake."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from herbenzo_agent.intake.models import ExtractionResult, IntakeSession, Intent, Phase, Turn
from herbenzo_agent.intake.red_flags import load_lexicon
from herbenzo_agent.intake.slots import SLOTS, Slot

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "extractor_system.md"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class Extractor(Protocol):
    async def extract(self, session: IntakeSession, turn: Turn) -> ExtractionResult: ...


# --- slot catalogue for the prompt -------------------------------------------------------------------

_OVERRIDES = {
    "symptoms[0].duration": 'duration object {"value": number > 0, "unit": hours|days|weeks|months|years, "raw_text": text}',
    "safety_profile.current_medications": 'list of {"name": text, "dose_text": text|null, "kind": prescription|otc|herbal_or_ayurvedic|supplement|unknown}',
}


def _describe(schema: dict[str, Any]) -> str:
    if "enum" in schema:
        return "one of " + "|".join(str(e) for e in schema["enum"])
    if "const" in schema:
        return f"exactly {schema['const']}"
    t = schema.get("type")
    if t == "string":
        return "text"
    if t == "integer":
        lo, hi = schema.get("minimum"), schema.get("maximum")
        return f"integer {lo}-{hi}" if lo is not None else "integer"
    if t == "boolean":
        return "true|false"
    if t == "array":
        return "list of " + _describe(schema.get("items", {}))
    if "$defs" in schema and "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return _describe(schema["$defs"][ref])
    return "value"


def _slot_line(slot: Slot) -> str:
    type_text = _OVERRIDES.get(slot.key) or (
        _describe(slot.adapter.json_schema()) if slot.adapter else "value"
    )
    need = "required" if slot.required else "optional"
    return f"- {slot.key} ({need}): {type_text}"


@lru_cache
def slot_catalogue() -> str:
    return "\n".join(_slot_line(s) for s in SLOTS)


@lru_cache
def system_prompt() -> str:
    return PROMPT_PATH.read_text()


def build_user_prompt(session: IntakeSession, turn: Turn) -> str:
    # Earlier turns are deliberately NOT included: the model must only extract from the latest message,
    # and history makes it re-emit old facts (slower, and rejected by the evidence gate anyway).
    previous = [t for t in session.turns if t.turn_id != turn.turn_id]
    last_question = next((t.text for t in reversed(previous) if t.role == "assistant"), "")
    filled = ", ".join(sorted(k for k in session.slots)) or "(none)"
    phase_hint = {
        Phase.GREETING_CONSENT: "The assistant asked for consent to continue (a yes/no confirmation).",
        Phase.READBACK: "The assistant read back the details and asked if they are correct (a yes/no confirmation).",
        Phase.CORRECTION: "The assistant asked what should be changed.",
        Phase.CONFIRM_RESTART: "The assistant asked whether to start over (a yes/no confirmation).",
    }.get(session.phase, f"The question was about slot: {session.target_slot}.")
    return (
        f"PHASE: {session.phase.value}. {phase_hint}\n"
        f"LAST ASSISTANT MESSAGE: {last_question}\n"
        f"ALREADY FILLED SLOTS (only re-emit one if the user changes it): {filled}\n\n"
        f"SLOT CATALOGUE:\n{slot_catalogue()}\n\n"
        f"RED FLAG CODES: {', '.join(sorted(load_lexicon().codes))}\n\n"
        f"LATEST USER MESSAGE: {turn.text}\n"
    )


def parse_extraction(raw: str) -> ExtractionResult:
    text = _THINK.sub("", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in extractor output")
    data = json.loads(text[start : end + 1])
    if isinstance(data.get("intent"), str) and data["intent"] not in Intent._value2member_map_:
        data["intent"] = Intent.unclear.value
    updates = []
    for u in data.get("updates") or []:
        if isinstance(u, dict) and isinstance(u.get("slot_key"), str):
            u.setdefault("evidence_quote", "")
            updates.append(u)
    data["updates"] = updates
    data["red_flag_suspected"] = [c for c in data.get("red_flag_suspected") or [] if isinstance(c, str)]
    return ExtractionResult.model_validate(data)


class LLMExtractor:
    def __init__(self, llm: BaseChatModel, timeout_s: float = 20.0, retries: int = 1):
        self.llm = llm
        self.timeout_s = timeout_s
        self.retries = retries

    async def extract(self, session: IntakeSession, turn: Turn) -> ExtractionResult:
        messages = [SystemMessage(system_prompt()), HumanMessage(build_user_prompt(session, turn))]
        for attempt in range(self.retries + 1):
            try:
                response = await asyncio.wait_for(self.llm.ainvoke(messages), timeout=self.timeout_s)
                return parse_extraction(str(response.content))
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                logger.warning("extractor output invalid (attempt %s): %s", attempt + 1, type(exc).__name__)
                messages = [
                    *messages[:2],
                    HumanMessage(
                        "Your previous output was not valid. Return only the JSON object described."
                    ),
                ]
            except TimeoutError:
                logger.warning("extractor timed out after %ss", self.timeout_s)
                break
            except Exception:  # provider errors must not kill the conversation
                logger.exception("extractor call failed")
                break
        return ExtractionResult(intent=Intent.unclear, failed=True)


class ScriptedExtractor:
    """Test double: returns a pre-recorded extraction for each exact user message."""

    def __init__(self, recordings: dict[str, ExtractionResult] | None = None):
        self.recordings = dict(recordings or {})

    def record(self, text: str, extraction: ExtractionResult) -> None:
        self.recordings[text] = extraction

    async def extract(self, session: IntakeSession, turn: Turn) -> ExtractionResult:
        return self.recordings.get(turn.text, ExtractionResult(intent=Intent.unclear))
