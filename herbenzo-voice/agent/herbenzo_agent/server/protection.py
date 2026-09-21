"""Rate limiting and retention housekeeping for the agent server."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from herbenzo_agent.persistence.store import SqliteStore

logger = logging.getLogger(__name__)


class SlidingWindowLimiter:
    """At most `limit` events per `window_s` seconds per key. In-process: one limiter per server instance."""

    def __init__(self, limit: int, window_s: float = 60.0, max_keys: int = 50_000):
        self.limit = limit
        self.window_s = window_s
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        events = self._events[key]
        while events and now - events[0] >= self.window_s:
            events.popleft()
        if len(events) >= self.limit:
            return False
        events.append(now)
        if len(self._events) > self.max_keys:  # drop idle keys to bound memory
            for idle in [k for k, v in self._events.items() if not v or now - v[-1] >= self.window_s]:
                del self._events[idle]
        return True


def client_key(request: Request, trust_forwarded_for: bool) -> str:
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce(limiter: SlidingWindowLimiter, key: str) -> None:
    if not limiter.allow(key):
        raise HTTPException(
            status_code=429, detail="too many requests, please slow down", headers={"Retry-After": "60"}
        )


async def purge_expired(store: SqliteStore, checkpointer, retention_days: int) -> dict[str, int]:
    """Delete conversation state and transcripts older than the retention period. Specs are kept (immutable evidence)."""
    threads = await store.inactive_threads(retention_days)
    for thread_id in threads:
        await checkpointer.adelete_thread(thread_id)
    await store.mark_purged(threads)
    transcripts = await store.purge_transcripts_older_than(retention_days)
    if threads or transcripts:
        logger.info("retention purge: %d conversation(s), %d transcript(s)", len(threads), transcripts)
    return {"conversations": len(threads), "transcripts": transcripts}


async def retention_loop(store: SqliteStore, checkpointer, retention_days: int, interval_s: float) -> None:
    while True:
        try:
            await purge_expired(store, checkpointer, retention_days)
        except Exception:
            logger.exception("retention purge failed")
        await asyncio.sleep(interval_s)
