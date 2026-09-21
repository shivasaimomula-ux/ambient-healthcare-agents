"""Per-turn stage timings (no content) and a rolling in-process summary served at /metrics."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Set by IntakeService.turn for the duration of one graph run; graph nodes add their stage durations.
TURN_TIMINGS: ContextVar[dict[str, float] | None] = ContextVar("turn_timings", default=None)
# Per-turn event counters (e.g. guardrail outcomes), reported next to the timings.
TURN_COUNTERS: ContextVar[dict[str, int] | None] = ContextVar("turn_counters", default=None)


def count(event: str) -> None:
    counters = TURN_COUNTERS.get()
    if counters is not None:
        counters[event] = counters.get(event, 0) + 1


@contextmanager
def timed(stage: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        timings = TURN_TIMINGS.get()
        if timings is not None:
            timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - started


async def timed_await[T](stage: str, awaitable: Awaitable[T]) -> T:
    with timed(stage):
        return await awaitable


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


class StageMetrics:
    """Rolling window of recent turn timings per stage and channel."""

    def __init__(self, window: int = 500):
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.counters: dict[str, int] = defaultdict(int)
        self.turns = 0

    def record(self, channel: str, timings: dict[str, float], counters: dict[str, int] | None = None) -> None:
        self.turns += 1
        for stage, seconds in timings.items():
            self._samples[f"{channel}.{stage}"].append(seconds)
        for event, value in (counters or {}).items():
            self.counters[f"{channel}.{event}"] += value

    def snapshot(self) -> dict[str, dict[str, float]]:
        summary = {}
        for key, values in sorted(self._samples.items()):
            ordered = sorted(values)
            summary[key] = {
                "count": len(ordered),
                "p50_s": round(_percentile(ordered, 0.5), 3),
                "p95_s": round(_percentile(ordered, 0.95), 3),
                "max_s": round(ordered[-1], 3),
            }
        return summary
