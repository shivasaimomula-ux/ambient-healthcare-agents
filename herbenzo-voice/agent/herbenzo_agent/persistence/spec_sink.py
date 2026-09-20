"""Where finished SymptomSpecs go. `SqliteStore` is the durable implementation; this in-memory one is for tests."""

from __future__ import annotations

from typing import Protocol

from herbenzo_agent.contracts.symptom_spec import SymptomSpec
from herbenzo_agent.intake.models import IntakeSession


class SpecSink(Protocol):
    async def save(self, spec: SymptomSpec, session: IntakeSession) -> None: ...

    async def get_for_session(self, session_id: str) -> list[SymptomSpec]: ...


class InMemorySpecSink:
    def __init__(self) -> None:
        self.specs: list[SymptomSpec] = []
        self.transcripts: dict[str, list] = {}

    async def save(self, spec: SymptomSpec, session: IntakeSession) -> None:
        self.specs.append(spec)
        self.transcripts[spec.provenance.transcript_ref] = [t.model_dump(mode="json") for t in session.turns]

    async def get_for_session(self, session_id: str) -> list[SymptomSpec]:
        base = session_id.split("#")[0]
        return [s for s in self.specs if s.session_id.split("#")[0] == base]
