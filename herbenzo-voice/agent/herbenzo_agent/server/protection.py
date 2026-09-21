"""Rate limiting and retention housekeeping for the agent server."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict, defaultdict, deque

from fastapi import HTTPException, Request

from herbenzo_agent.persistence.store import SqliteStore

logger = logging.getLogger(__name__)


class SessionLockMap:
    """Per-session asyncio locks with idle TTL + max-size eviction (Finding #24).

    Unbounded ``defaultdict(asyncio.Lock)`` retained every session_id for process
    lifetime. This map drops idle, unlocked entries so long-lived intake services
    do not accumulate lock objects without bound.
    """

    def __init__(self, max_size: int = 10_000, idle_ttl_s: float = 3600.0):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.idle_ttl_s = idle_ttl_s
        self._entries: OrderedDict[str, tuple[asyncio.Lock, float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._entries

    def __getitem__(self, key: str) -> asyncio.Lock:
        now = time.monotonic()
        if key in self._entries:
            lock, _ = self._entries.pop(key)
            self._entries[key] = (lock, now)
            return lock
        self._evict(now)
        lock = asyncio.Lock()
        self._entries[key] = (lock, now)
        return lock

    def pop(self, key: str, default: asyncio.Lock | None = None) -> asyncio.Lock | None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return default
        return entry[0]

    def _evict(self, now: float) -> None:
        stale = [
            k
            for k, (lock, ts) in self._entries.items()
            if not lock.locked() and now - ts >= self.idle_ttl_s
        ]
        for k in stale:
            del self._entries[k]
        while len(self._entries) >= self.max_size:
            victim = next(
                (k for k, (lock, _) in self._entries.items() if not lock.locked()),
                None,
            )
            if victim is None:
                break
            del self._entries[victim]


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
