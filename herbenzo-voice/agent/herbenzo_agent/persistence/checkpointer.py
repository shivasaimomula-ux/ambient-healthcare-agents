"""LangGraph checkpointer from DATABASE_URL (sqlite:///path or sqlite:///:memory:)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver


@asynccontextmanager
async def open_checkpointer(database_url: str) -> AsyncIterator[object]:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("Only sqlite:/// DATABASE_URL is supported until M5 adds Postgres.")
    path = database_url.removeprefix("sqlite:///")
    if path in (":memory:", ""):
        yield InMemorySaver()
        return
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        yield saver
