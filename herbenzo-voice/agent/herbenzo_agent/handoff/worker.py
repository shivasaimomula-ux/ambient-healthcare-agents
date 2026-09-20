"""Background delivery of eligible SymptomSpecs to the Stage A Recommender.

Never on the conversation path: the Recommender's evidence loop can take minutes. Every attempt is
recorded; retries only on timeouts, connection errors and 5xx; the spec id is the idempotency key.
Queued/in-progress handoffs are resumed when the service restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Protocol

import httpx

from herbenzo_agent.contracts.symptom_spec import SymptomSpec
from herbenzo_agent.handoff.recommender_adapter import to_predict_request
from herbenzo_agent.persistence.store import SqliteStore

logger = logging.getLogger(__name__)

TARGET = "stage_a_recommender"
MAX_RESPONSE_BYTES = 2_000_000


class Handoff(Protocol):
    async def submit(self, spec: SymptomSpec, eligible: bool, reason: str | None = None) -> str: ...


class NoopHandoff:
    """Handoff disabled: nothing is recorded or sent."""

    async def submit(self, spec: SymptomSpec, eligible: bool, reason: str | None = None) -> str:
        return "disabled"


class RecommenderHandoff:
    def __init__(
        self,
        store: SqliteStore,
        base_url: str,
        timeout_s: float = 600.0,
        max_attempts: int = 3,
        backoff_s: float = 2.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.store = store
        self.url = base_url.rstrip("/") + "/predict"
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._tasks: set[asyncio.Task] = set()

    async def submit(self, spec: SymptomSpec, eligible: bool, reason: str | None = None) -> str:
        if not eligible:
            await self.store.create_handoff(spec, TARGET, "not_eligible", reason=reason)
            return "not_eligible"
        await self.store.create_handoff(spec, TARGET, "queued", request=to_predict_request(spec))
        self._spawn(str(spec.spec_id))
        return "queued"

    def _spawn(self, spec_id: str) -> None:
        task = asyncio.create_task(self._deliver(spec_id), name=f"handoff-{spec_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def resume_pending(self) -> int:
        pending = await self.store.handoffs_with_status("queued", "in_progress")
        for record in pending:
            self._spawn(record.spec_id)
        if pending:
            logger.info("resumed %d pending handoff(s)", len(pending))
        return len(pending)

    async def wait_idle(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        await self._client.aclose()

    async def _deliver(self, spec_id: str) -> None:
        record = await self.store.get_handoff(spec_id)
        if record is None or record.status not in ("queued", "in_progress") or record.request is None:
            return
        attempts = record.attempts
        while attempts < self.max_attempts:
            attempts += 1
            await self.store.update_handoff(spec_id, status="in_progress", attempts=attempts)
            try:
                response = await self._client.post(
                    self.url,
                    json=record.request,
                    headers={"Idempotency-Key": record.idempotency_key, "X-Herbenzo-Spec-Id": spec_id},
                    timeout=self.timeout_s,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                error = f"{type(exc).__name__}"
                logger.warning("handoff %s attempt %d failed: %s", spec_id, attempts, error)
                await self.store.update_handoff(spec_id, last_error=error)
            else:
                body = response.text[:MAX_RESPONSE_BYTES]
                try:
                    json.loads(body)
                    response_json = body
                except ValueError:
                    response_json = json.dumps({"raw": body})
                if response.is_success:
                    await self.store.update_handoff(
                        spec_id,
                        status="sent",
                        response_status=response.status_code,
                        response_json=response_json,
                        last_error=None,
                    )
                    logger.info("handoff %s delivered (%d)", spec_id, response.status_code)
                    return
                await self.store.update_handoff(
                    spec_id,
                    response_status=response.status_code,
                    response_json=response_json,
                    last_error=f"HTTP {response.status_code}",
                )
                if response.status_code < 500:
                    await self.store.update_handoff(spec_id, status="failed")
                    logger.warning("handoff %s rejected by recommender (%d)", spec_id, response.status_code)
                    return
            if attempts < self.max_attempts:
                await asyncio.sleep(self.backoff_s * 2 ** (attempts - 1))
        await self.store.update_handoff(spec_id, status="failed")
        logger.warning("handoff %s failed after %d attempts", spec_id, attempts)
