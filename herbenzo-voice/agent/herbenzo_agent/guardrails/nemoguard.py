"""NVIDIA NemoGuard safety rails, called directly.

The rail configuration lives in NeMo Guardrails format (`config.yml` + `prompts.yml`) so it stays
portable to the `nemoguardrails` runtime. We call the NemoGuard NIMs ourselves because the library
(0.24) treats an empty model response as "blocked" (observed ~1 in 3 calls on the hosted endpoint),
and its categories are not reliable enough to route on. Here the failure modes are explicit:

- input rail unavailable  -> status `unavailable` (caller decides: pipeline/prod fail closed;
  intake-only demo may fail open — see Settings.guardrails_fail_closed / IntakeDeps)
- output rail unavailable -> the caller uses the canonical question (never speaks unchecked text)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import httpx
import yaml

from herbenzo_agent.audit.reproducibility import directory_sha256

logger = logging.getLogger(__name__)

CONTENT_INPUT_TASK = "content_safety_check_input $model=content_safety"
CONTENT_OUTPUT_TASK = "content_safety_check_output $model=content_safety"
TOPIC_INPUT_TASK = "topic_safety_check_input $model=topic_control"
TOPIC_SUFFIX = '\n\nIf any of the above conditions are violated, please respond with "off-topic". Otherwise, respond with "on-topic". You must respond with "on-topic" or "off-topic".'


class GuardStatus(StrEnum):
    passed = "passed"
    blocked = "blocked"
    unavailable = "unavailable"


@dataclass
class GuardResult:
    status: GuardStatus
    rail: str | None = None
    categories: list[str] = field(default_factory=list)
    latency_s: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.status is GuardStatus.blocked


class Guardrails(Protocol):
    config_sha256: str

    async def check_input(self, user_text: str) -> GuardResult: ...

    async def check_output(self, user_text: str, bot_text: str) -> GuardResult: ...


class NoopGuardrails:
    config_sha256 = "disabled"

    async def check_input(self, user_text: str) -> GuardResult:
        return GuardResult(GuardStatus.passed)

    async def check_output(self, user_text: str, bot_text: str) -> GuardResult:
        return GuardResult(GuardStatus.passed)


# --- parsing (same semantics as nemoguardrails nemoguard_parse_*_safety) -----------------------------


def parse_safety(raw: str, field_name: str) -> tuple[bool, list[str]] | None:
    """Return (is_safe, categories) or None when the response is empty/unparseable."""
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    verdict = str(data.get(field_name, "")).strip().lower()
    if verdict not in ("safe", "unsafe"):
        return None
    categories = [c.strip() for c in str(data.get("Safety Categories", "")).split(",") if c.strip()]
    return verdict == "safe", categories


def parse_topic(raw: str) -> bool | None:
    text = (raw or "").strip().lower()
    if "off-topic" in text:
        return False
    if "on-topic" in text:
        return True
    return None


# --- client -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RailModels:
    content_safety: str
    topic_control: str | None
    base_url: str


def load_rails_config(path: Path) -> tuple[RailModels, dict[str, dict]]:
    config = yaml.safe_load((path / "config.yml").read_text())
    prompts = {p["task"]: p for p in yaml.safe_load((path / "prompts.yml").read_text())["prompts"]}
    models = {m["type"]: m for m in config["models"]}
    content = models["content_safety"]
    topic = models.get("topic_control")
    return (
        RailModels(
            content_safety=content["model"],
            topic_control=topic["model"] if topic else None,
            base_url=content.get("parameters", {}).get("base_url", "https://integrate.api.nvidia.com/v1"),
        ),
        prompts,
    )


class NemoGuardRails:
    def __init__(
        self,
        config_path: Path,
        api_key: str | None,
        base_url: str | None = None,
        timeout_s: float = 4.0,
        topic_control_enabled: bool = False,
        client: httpx.AsyncClient | None = None,
        attempts: int = 1,
    ):
        self.models, self.prompts = load_rails_config(config_path)
        self.base_url = (base_url or self.models.base_url).rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        # One attempt by default: the hosted NIM is bimodal (~0.3 s or 20-40 s), so retrying a slow call only
        # stalls the conversation. Unavailable input is policy-decided by the intake graph; output falls
        # back to the base question.
        self.attempts = max(1, attempts)
        self.topic_control_enabled = topic_control_enabled and self.models.topic_control is not None
        self.config_sha256 = directory_sha256(config_path)
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def _complete(self, model: str, messages: list[dict], max_tokens: int) -> str | None:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
        for attempt in range(self.attempts):
            try:
                response = await self._client.post(
                    f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout_s
                )
                if response.status_code >= 500:
                    logger.warning(
                        "guardrail model %s returned %s (attempt %s)",
                        model,
                        response.status_code,
                        attempt + 1,
                    )
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content") or ""
                if content.strip():
                    return content
                logger.warning("guardrail model %s returned empty content (attempt %s)", model, attempt + 1)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                logger.warning(
                    "guardrail model %s call failed (attempt %s): %s", model, attempt + 1, type(exc).__name__
                )
        return None

    def _render(self, task: str, **values: str) -> tuple[str, int]:
        prompt = self.prompts[task]
        content = prompt["content"]
        for key, value in values.items():
            content = content.replace("{{ " + key + " }}", value)
        return content, int(prompt.get("max_tokens", 64))

    async def _content_input(self, user_text: str) -> GuardResult:
        content, max_tokens = self._render(CONTENT_INPUT_TASK, user_input=user_text)
        raw = await self._complete(
            self.models.content_safety, [{"role": "user", "content": content}], max_tokens
        )
        parsed = parse_safety(raw or "", "User Safety")
        if parsed is None:
            return GuardResult(GuardStatus.unavailable, rail="content_safety_input")
        safe, categories = parsed
        return GuardResult(
            GuardStatus.passed if safe else GuardStatus.blocked, "content_safety_input", categories
        )

    async def _topic_input(self, user_text: str) -> GuardResult:
        system = self.prompts[TOPIC_INPUT_TASK]["content"] + TOPIC_SUFFIX
        raw = await self._complete(
            self.models.topic_control,
            [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
            max_tokens=8,
        )
        on_topic = parse_topic(raw or "")
        if on_topic is None:
            return GuardResult(GuardStatus.unavailable, rail="topic_control_input")
        return GuardResult(GuardStatus.passed if on_topic else GuardStatus.blocked, "topic_control_input")

    async def check_input(self, user_text: str) -> GuardResult:
        started = time.perf_counter()
        checks = [self._content_input(user_text)]
        if self.topic_control_enabled:
            checks.append(self._topic_input(user_text))
        results = await asyncio.gather(*checks)
        blocked = next((r for r in results if r.blocked), None)
        result = (
            blocked or next((r for r in results if r.status is GuardStatus.unavailable), None) or results[0]
        )
        result.latency_s = time.perf_counter() - started
        if result.status is GuardStatus.unavailable:
            logger.warning("input guardrail unavailable (%s); status=unavailable", result.rail)
        return result

    async def check_output(self, user_text: str, bot_text: str) -> GuardResult:
        started = time.perf_counter()
        content, max_tokens = self._render(CONTENT_OUTPUT_TASK, user_input=user_text, bot_response=bot_text)
        raw = await self._complete(
            self.models.content_safety, [{"role": "user", "content": content}], max_tokens
        )
        parsed = parse_safety(raw or "", "Response Safety")
        if parsed is None:
            return GuardResult(
                GuardStatus.unavailable, "content_safety_output", latency_s=time.perf_counter() - started
            )
        safe, categories = parsed
        status = GuardStatus.passed if safe else GuardStatus.blocked
        return GuardResult(status, "content_safety_output", categories, time.perf_counter() - started)

    async def aclose(self) -> None:
        await self._client.aclose()
