"""Run conversation goldens end-to-end through the LangGraph intake service with a recorded extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from langgraph.checkpoint.memory import InMemorySaver

from herbenzo_agent.intake.graph import (
    SESSION_START,
    IntakeDeps,
    IntakeService,
    TurnResult,
    build_intake_graph,
)
from herbenzo_agent.intake.lint import blocked_patterns
from herbenzo_agent.intake.models import (
    ExtractionResult,
    IntakePolicy,
    IntakeSession,
    Intent,
    SlotUpdate,
    Turn,
)
from herbenzo_agent.intake.red_flag_model import NullRedFlagClassifier
from herbenzo_agent.intake.responder import BaseQuestionResponder
from herbenzo_agent.persistence.spec_sink import InMemorySpecSink

GOLDENS = yaml.safe_load((Path(__file__).parent / "scripts" / "goldens.yaml").read_text())


class RecordingExtractor:
    def __init__(self) -> None:
        self.recordings: dict[str, ExtractionResult] = {}
        self.calls: list[str] = []

    async def extract(self, session: IntakeSession, turn: Turn) -> ExtractionResult:
        self.calls.append(turn.text)
        return self.recordings.get(turn.text, ExtractionResult(intent=Intent.unclear))


def _update(raw: Any) -> SlotUpdate:
    if isinstance(raw, dict):
        return SlotUpdate(
            slot_key=raw["slot"], declined=raw.get("declined", False), evidence_quote=raw.get("quote", "")
        )
    key, value, quote = raw
    return SlotUpdate(slot_key=key, value=value, evidence_quote=str(quote), confidence=0.8)


def _expand(turns: list[Any]) -> list[dict]:
    out: list[dict] = []
    for item in turns:
        if isinstance(item, str):
            out.extend(GOLDENS["fragments"][item])
        else:
            out.append(item)
    return out


async def run_script(script: dict) -> dict:
    extractor = RecordingExtractor()
    turns = _expand(script["turns"])
    for t in turns:
        extractor.recordings[t["user"]] = ExtractionResult(
            intent=Intent(t.get("intent", "answer")), updates=[_update(u) for u in t.get("updates", [])]
        )
    sink = InMemorySpecSink()
    policy = IntakePolicy(**{"ayurvedic_context_enabled": False, **script.get("policy", {})})
    deps = IntakeDeps(
        extractor=extractor,
        responder=BaseQuestionResponder(),
        red_flag_classifier=NullRedFlagClassifier(),
        spec_sink=sink,
        policy=policy,
        llm_models={"extractor": "recorded", "responder": "base-question"},
    )
    service = IntakeService(build_intake_graph(deps, InMemorySaver()))
    thread = f"golden-{script['name']}"
    results: list[TurnResult] = [await service.turn(thread, SESSION_START)]
    for t in turns:
        results.append(await service.turn(thread, t["user"]))
    return {"results": results, "sink": sink, "extractor": extractor, "service": service, "thread": thread}


@pytest.mark.parametrize("script", GOLDENS["scripts"], ids=lambda s: s["name"])
async def test_golden(script):
    run = await run_script(script)
    results: list[TurnResult] = run["results"]
    expect = script["expect"]
    session = await run["service"].get_session(run["thread"])
    replies = [r.reply for r in results]
    all_text = "\n".join(replies)
    specs = run["sink"].specs

    assert all(replies), f"every turn must produce a reply: {replies}"

    if "final_status" in expect:
        assert specs or expect["final_status"] == "out_of_scope", "expected a saved spec"
        status = results[-1].final_status or session.final_status
        assert status is not None and status.value == expect["final_status"], all_text
    if "phase" in expect:
        assert session.phase.value == expect["phase"], all_text
    if "spec_count" in expect:
        assert len(specs) == expect["spec_count"]
    if "spec_status" in expect:
        assert specs[-1].status.value == expect["spec_status"]
    if "eligible" in expect:
        assert specs and specs[-1].is_handoff_eligible(0.5) is expect["eligible"]
    if "floor" in expect:
        lo, hi = expect["floor"]
        assert lo <= specs[-1].confidence_floor <= hi
    for key, value in expect.get("slots", {}).items():
        assert session.value(key) == value, key
    for phrase in expect.get("reply_contains", []):
        assert phrase.lower() in all_text.lower(), f"{phrase!r} not in replies:\n{all_text}"
    for phrase in expect.get("last_reply_contains", []):
        assert phrase.lower() in replies[-1].lower()
    for phrase in expect.get("asked", []):
        assert phrase.lower() in all_text.lower()
    for phrase in expect.get("never_asked", []):
        assert phrase.lower() not in all_text.lower(), f"{phrase!r} was asked"
    for text in expect.get("extractor_not_called_for", []):
        assert text not in run["extractor"].calls
    if "session_id_suffix" in expect:
        assert session.session_id.endswith(expect["session_id_suffix"])
    if "med_kinds" in expect:
        assert [m.kind for m in specs[-1].safety_profile.current_medications.value] == expect["med_kinds"]
    if "duration_value" in expect:
        assert specs[-1].symptoms[0].duration.value.value == expect["duration_value"]
    if expect.get("no_blocked_terms"):
        for reply in replies[1:]:  # the scripted greeting legitimately says "I can't diagnose"
            assert not [p.pattern for p in blocked_patterns() if p.search(reply)], reply

    for spec in specs:  # every saved spec is a valid contract with provenance
        assert spec.provenance.graph_version and spec.provenance.prompt_sha256["scripts"]
