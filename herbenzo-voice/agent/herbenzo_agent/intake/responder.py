"""Phrase the next question. Any failure (timeout, error, lint) falls back to the slot's base question."""

from __future__ import annotations

import asyncio
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from herbenzo_agent.intake.lint import lint_response
from herbenzo_agent.intake.models import IntakeSession

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "responder_system.md"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class Responder(Protocol):
    async def phrase(self, session: IntakeSession, question: str, acknowledge: bool) -> str: ...


@lru_cache
def system_prompt() -> str:
    return PROMPT_PATH.read_text()


def clean(text: str) -> str:
    text = _THINK.sub("", text).strip().strip('"').strip()
    return re.sub(r"\s+", " ", text)


class LLMResponder:
    def __init__(self, llm: BaseChatModel, timeout_s: float = 6.0):
        self.llm = llm
        self.timeout_s = timeout_s

    async def phrase(self, session: IntakeSession, question: str, acknowledge: bool) -> str:
        last_user = next((t.text for t in reversed(session.turns) if t.role == "user"), "")
        prompt = (
            f"TARGET_QUESTION: {question}\n"
            f"USER_LAST_MESSAGE: {last_user or '(none)'}\n"
            f"ACKNOWLEDGEMENT_ALLOWED: {'yes' if acknowledge else 'no'}\n"
            f"SPEAKING_ABOUT: {'the user' if session.value('reporter_role') in (None, 'self') else 'someone else'}"
        )
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke([SystemMessage(system_prompt()), HumanMessage(prompt)]),
                timeout=self.timeout_s,
            )
        except TimeoutError:
            logger.warning("responder timed out; using base question")
            return question
        except Exception:
            logger.exception("responder call failed; using base question")
            return question
        text = clean(str(response.content))
        problems = lint_response(text, user_text=session.user_text(), base_question=question)
        if problems:
            logger.info("responder output rejected by lint: %s", problems)
            return question
        return text


class GuardedResponder:
    """Runs the output safety rail on phrased text; anything not clearly safe becomes the base question."""

    def __init__(self, inner: Responder, guardrails):
        self.inner = inner
        self.guardrails = guardrails

    async def phrase(self, session: IntakeSession, question: str, acknowledge: bool) -> str:
        text = await self.inner.phrase(session, question, acknowledge)
        if text == question:
            return text
        last_user = next((t.text for t in reversed(session.turns) if t.role == "user"), "")
        result = await self.guardrails.check_output(last_user, text)
        if result.status.value != "passed":
            logger.info("responder output not released by output rail (%s)", result.status.value)
            return question
        return text


class BaseQuestionResponder:
    """Deterministic responder: always the canonical question. Used in tests and as a no-LLM mode."""

    async def phrase(self, session: IntakeSession, question: str, acknowledge: bool) -> str:
        return question
