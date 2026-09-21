"""Durable audit store: SymptomSpecs (insert-only), transcripts, and handoffs to downstream stages.

SQLite via aiosqlite. The SQL is plain and portable; a Postgres implementation of the same interface is
a M7 item. Conversation state itself lives in the LangGraph checkpointer; this store holds what must be
queryable and auditable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite

from herbenzo_agent.contracts.symptom_spec import SymptomSpec
from herbenzo_agent.intake.models import IntakeSession

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS specs (
    spec_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    status TEXT NOT NULL,
    status_reason TEXT,
    confidence_floor REAL NOT NULL,
    channel TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS specs_thread ON specs(thread_id);
CREATE INDEX IF NOT EXISTS specs_status ON specs(status);

-- Specs are evidence: they can never be changed or removed through the application.
CREATE TRIGGER IF NOT EXISTS specs_no_update BEFORE UPDATE ON specs
BEGIN SELECT RAISE(ABORT, 'specs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS specs_no_delete BEFORE DELETE ON specs
BEGIN SELECT RAISE(ABORT, 'specs are immutable'); END;

CREATE TABLE IF NOT EXISTS transcripts (
    transcript_ref TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    turns_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS handoffs (
    spec_id TEXT PRIMARY KEY REFERENCES specs(spec_id),
    thread_id TEXT NOT NULL,
    target TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    idempotency_key TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    request_json TEXT,
    response_status INTEGER,
    response_sha256 TEXT,
    response_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS handoffs_status ON handoffs(status);

-- v2: conversation activity, so checkpointer threads (which hold full conversations) can be purged too.
CREATE TABLE IF NOT EXISTS sessions (
    thread_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    purged_at TEXT
);
CREATE INDEX IF NOT EXISTS sessions_activity ON sessions(last_activity_at);
"""

HANDOFF_STATUSES = ("not_eligible", "queued", "in_progress", "sent", "failed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def thread_of(session_id: str) -> str:
    return session_id.split("#")[0]


@dataclass
class HandoffRecord:
    spec_id: str
    thread_id: str
    target: str
    status: str
    reason: str | None
    idempotency_key: str
    attempts: int
    last_error: str | None
    request: dict[str, Any] | None
    response_status: int | None
    response_sha256: str | None
    response: Any
    created_at: str
    updated_at: str

    def public(self, include_response: bool = True) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k not in ("request", "response")}
        if include_response:
            data["response"] = self.response
        return data


class SqliteStore:
    def __init__(self, path: str):
        self.path = path
        self._db: aiosqlite.Connection | None = None

    @classmethod
    def from_database_url(cls, database_url: str) -> SqliteStore:
        if not database_url.startswith("sqlite:///"):
            raise ValueError("Only sqlite:/// DATABASE_URL is supported (Postgres store is a M7 item).")
        path = database_url.removeprefix("sqlite:///") or ":memory:"
        return cls(path)

    async def open(self) -> SqliteStore:
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA)
        async with self._db.execute("SELECT version FROM schema_version") as cur:
            row = await cur.fetchone()
        if row is None:
            await self._db.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
        elif row["version"] > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema v{row['version']} is newer than this code (v{SCHEMA_VERSION})"
            )
        elif row["version"] < SCHEMA_VERSION:
            # Every migration so far is additive (CREATE ... IF NOT EXISTS above), so recording the version is enough.
            await self._db.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("store is not open")
        return self._db

    async def _write(self, *statements: tuple[str, tuple]) -> aiosqlite.Cursor:
        """Run statements in one transaction. Roll back on any error so a failed write never leaves the file
        locked (the LangGraph checkpointer writes to the same database)."""
        cursor = None
        try:
            for sql, params in statements:
                cursor = await self.db.execute(sql, params)
        except BaseException:
            await self.db.rollback()
            raise
        await self.db.commit()
        return cursor

    # --- specs + transcripts (SpecSink) -------------------------------------------------------------

    async def save(self, spec: SymptomSpec, session: IntakeSession) -> None:
        payload = spec.model_dump_json()
        created = spec.created_at.isoformat()
        await self._write(
            (
                "INSERT INTO specs(spec_id, session_id, thread_id, spec_version, status, status_reason, confidence_floor,"
                " channel, sha256, json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(spec.spec_id),
                    spec.session_id,
                    thread_of(spec.session_id),
                    spec.spec_version,
                    spec.status.value,
                    spec.status_reason,
                    spec.confidence_floor,
                    spec.channel,
                    _sha(payload),
                    payload,
                    created,
                ),
            ),
            (
                "INSERT OR REPLACE INTO transcripts(transcript_ref, session_id, thread_id, turns_json, created_at)"
                " VALUES (?,?,?,?,?)",
                (
                    spec.provenance.transcript_ref,
                    session.session_id,
                    thread_of(session.session_id),
                    json.dumps([t.model_dump(mode="json") for t in session.turns]),
                    _now(),
                ),
            ),
        )

    async def get_for_session(self, session_id: str) -> list[SymptomSpec]:
        async with self.db.execute(
            "SELECT json FROM specs WHERE thread_id = ? ORDER BY created_at", (thread_of(session_id),)
        ) as cur:
            return [SymptomSpec.model_validate_json(row["json"]) for row in await cur.fetchall()]

    async def get_spec(self, spec_id: str) -> SymptomSpec | None:
        async with self.db.execute("SELECT json FROM specs WHERE spec_id = ?", (spec_id,)) as cur:
            row = await cur.fetchone()
        return SymptomSpec.model_validate_json(row["json"]) if row else None

    async def get_transcript(self, transcript_ref: str) -> list[dict] | None:
        async with self.db.execute(
            "SELECT turns_json FROM transcripts WHERE transcript_ref = ?", (transcript_ref,)
        ) as cur:
            row = await cur.fetchone()
        return json.loads(row["turns_json"]) if row else None

    async def touch_session(self, thread_id: str, channel: str) -> None:
        now = _now()
        await self._write(
            (
                "INSERT INTO sessions(thread_id, channel, created_at, last_activity_at) VALUES (?,?,?,?)"
                " ON CONFLICT(thread_id) DO UPDATE SET last_activity_at = excluded.last_activity_at",
                (thread_of(thread_id), channel, now, now),
            )
        )

    async def inactive_threads(self, days: int, limit: int = 500) -> list[str]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        async with self.db.execute(
            "SELECT thread_id FROM sessions WHERE purged_at IS NULL AND last_activity_at < ?"
            " ORDER BY last_activity_at LIMIT ?",
            (cutoff, limit),
        ) as cur:
            return [row["thread_id"] for row in await cur.fetchall()]

    async def mark_purged(self, thread_ids: list[str]) -> None:
        if thread_ids:
            now = _now()
            await self._write(
                *[("UPDATE sessions SET purged_at = ? WHERE thread_id = ?", (now, t)) for t in thread_ids]
            )

    async def ping(self) -> bool:
        async with self.db.execute("SELECT 1") as cur:
            return (await cur.fetchone()) is not None

    async def purge_transcripts_older_than(self, days: int) -> int:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        cur = await self._write(("DELETE FROM transcripts WHERE created_at < ?", (cutoff,)))
        return cur.rowcount

    # --- handoffs ----------------------------------------------------------------------------------------

    async def create_handoff(
        self,
        spec: SymptomSpec,
        target: str,
        status: str,
        reason: str | None = None,
        request: dict | None = None,
    ) -> HandoffRecord:
        assert status in HANDOFF_STATUSES
        now = _now()
        await self._write(
            (
                "INSERT OR IGNORE INTO handoffs(spec_id, thread_id, target, status, reason, idempotency_key,"
                " request_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    str(spec.spec_id),
                    thread_of(spec.session_id),
                    target,
                    status,
                    reason,
                    str(spec.spec_id),
                    json.dumps(request) if request is not None else None,
                    now,
                    now,
                ),
            )
        )
        record = await self.get_handoff(str(spec.spec_id))
        assert record is not None
        return record

    async def update_handoff(self, spec_id: str, **fields: Any) -> None:
        allowed = {"status", "reason", "attempts", "last_error", "response_status", "response_json"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown handoff fields: {unknown}")
        if "status" in fields:
            assert fields["status"] in HANDOFF_STATUSES
        if "response_json" in fields and fields["response_json"] is not None:
            fields["response_sha256"] = _sha(fields["response_json"])
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        await self._write(
            (f"UPDATE handoffs SET {assignments} WHERE spec_id = ?", (*fields.values(), spec_id))
        )

    def _record(self, row: aiosqlite.Row) -> HandoffRecord:
        return HandoffRecord(
            spec_id=row["spec_id"],
            thread_id=row["thread_id"],
            target=row["target"],
            status=row["status"],
            reason=row["reason"],
            idempotency_key=row["idempotency_key"],
            attempts=row["attempts"],
            last_error=row["last_error"],
            request=json.loads(row["request_json"]) if row["request_json"] else None,
            response_status=row["response_status"],
            response_sha256=row["response_sha256"],
            response=json.loads(row["response_json"]) if row["response_json"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def get_handoff(self, spec_id: str) -> HandoffRecord | None:
        async with self.db.execute("SELECT * FROM handoffs WHERE spec_id = ?", (spec_id,)) as cur:
            row = await cur.fetchone()
        return self._record(row) if row else None

    async def handoffs_for_thread(self, thread_id: str) -> list[HandoffRecord]:
        async with self.db.execute(
            "SELECT * FROM handoffs WHERE thread_id = ? ORDER BY created_at", (thread_of(thread_id),)
        ) as cur:
            return [self._record(r) for r in await cur.fetchall()]

    async def handoffs_with_status(self, *statuses: str, limit: int = 100) -> list[HandoffRecord]:
        marks = ",".join("?" for _ in statuses)
        async with self.db.execute(
            f"SELECT * FROM handoffs WHERE status IN ({marks}) ORDER BY created_at LIMIT ?",
            (*statuses, limit),
        ) as cur:
            return [self._record(r) for r in await cur.fetchall()]
