"""Model layer of the red-flag screen: a small classifier used only when rules are clear but the
message mentions symptom words the rules might have missed."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from herbenzo_agent.intake.red_flags import load_lexicon

logger = logging.getLogger(__name__)

SYSTEM = (
    "You screen one patient message for medical emergencies. Answer with a JSON list of codes from the allowed list "
    "that the message clearly describes as happening now or very recently, or [] if none. Do not flag past history "
    "that is resolved, negated statements, or mild everyday symptoms. Output only the JSON list."
)


class RedFlagClassifier(Protocol):
    async def classify(self, text: str) -> list[str]: ...


class LLMRedFlagClassifier:
    def __init__(self, llm: BaseChatModel, timeout_s: float = 6.0):
        self.llm = llm
        self.timeout_s = timeout_s

    async def classify(self, text: str) -> list[str]:
        codes = sorted(load_lexicon().codes)
        prompt = f"ALLOWED CODES: {', '.join(codes)}\nMESSAGE: {text}"
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke([SystemMessage(SYSTEM), HumanMessage(prompt)]), timeout=self.timeout_s
            )
            raw = re.sub(r"<think>.*?</think>", "", str(response.content), flags=re.DOTALL)
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            found = json.loads(match.group(0)) if match else []
            return [c for c in found if isinstance(c, str) and c in codes]
        except Exception:
            # Fail open to the rule layer, but make the failure visible.
            logger.exception("red-flag classifier failed; relying on rules only")
            return []


class NullRedFlagClassifier:
    async def classify(self, text: str) -> list[str]:
        return []
