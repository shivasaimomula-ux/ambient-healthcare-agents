"""Finding #24: SessionLockMap idle TTL / max-size eviction."""

from __future__ import annotations

import time

import pytest

from herbenzo_agent.server.protection import SessionLockMap


@pytest.mark.asyncio
async def test_session_lock_map_evicts_idle_unlocked_entries():
    locks = SessionLockMap(max_size=10, idle_ttl_s=0.05)
    a = locks["a"]
    b = locks["b"]
    assert len(locks) == 2
    assert a is locks["a"]

    time.sleep(0.06)
    # Touching a new key triggers TTL eviction of idle unlocked locks.
    c = locks["c"]
    assert c is locks["c"]
    assert "a" not in locks and "b" not in locks
    assert len(locks) == 1


@pytest.mark.asyncio
async def test_session_lock_map_respects_max_size_lru():
    locks = SessionLockMap(max_size=2, idle_ttl_s=3600.0)
    locks["s1"]
    locks["s2"]
    assert len(locks) == 2
    locks["s3"]  # must evict oldest unlocked (s1)
    assert "s1" not in locks
    assert "s2" in locks and "s3" in locks
    assert len(locks) == 2


@pytest.mark.asyncio
async def test_session_lock_map_does_not_evict_held_lock():
    locks = SessionLockMap(max_size=1, idle_ttl_s=0.01)
    held = locks["held"]
    async with held:
        time.sleep(0.02)
        locks["other"]
        # Held lock cannot be dropped even when over max_size / past TTL.
        assert "held" in locks
        assert held.locked()
    locks.pop("held")
    assert "held" not in locks


@pytest.mark.asyncio
async def test_session_lock_map_pop_removes_entry():
    locks = SessionLockMap()
    lock = locks["sess"]
    assert locks.pop("sess") is lock
    assert "sess" not in locks
    assert locks.pop("missing", None) is None
