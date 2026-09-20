"""M5: audit store, Recommender adapter, handoff worker, and the end-to-end chat -> spec -> handoff path."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from herbenzo_agent.contracts.symptom_spec import CaptureMode, SymptomSpec
from herbenzo_agent.handoff.recommender_adapter import to_predict_request
from herbenzo_agent.handoff.worker import RecommenderHandoff
from herbenzo_agent.intake.graph import IntakeDeps
from herbenzo_agent.intake.models import ExtractionResult, IntakePolicy, IntakeSession, Intent
from herbenzo_agent.intake.red_flag_model import NullRedFlagClassifier
from herbenzo_agent.intake.responder import BaseQuestionResponder
from herbenzo_agent.persistence.store import SqliteStore
from herbenzo_agent.server.app import create_app
from herbenzo_agent.settings import Settings
from tests.conversation.test_goldens import RecordingExtractor, _expand, _update

FIXTURES = Path(__file__).parents[1] / "contract" / "fixtures"
REC_URL = "http://rec.test"


def fixture_spec(name: str = "symptom_spec_complete.json", **changes) -> SymptomSpec:
    data = json.loads((FIXTURES / name).read_text())
    data.update(changes)
    return SymptomSpec.model_validate(data)


def session_for(spec: SymptomSpec) -> IntakeSession:
    s = IntakeSession(session_id=spec.session_id)
    s.add_turn("user", "I've got burning acidity")
    return s


@pytest.fixture
async def store(tmp_path):
    s = await SqliteStore(str(tmp_path / "audit.db")).open()
    yield s
    await s.close()


# --- store -------------------------------------------------------------------------------------------


async def test_spec_and_transcript_round_trip(store):
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    assert await store.get_spec(str(spec.spec_id)) == spec
    assert await store.get_for_session(spec.session_id) == [spec]
    transcript = await store.get_transcript(spec.provenance.transcript_ref)
    assert transcript[0]["text"] == "I've got burning acidity"


async def test_specs_are_immutable(store, tmp_path):
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    with pytest.raises(sqlite3.IntegrityError):
        await store.save(spec, session_for(spec))  # same spec id twice
    # the failed insert must not leave the database locked for other writers (e.g. the checkpointer)
    raw = sqlite3.connect(tmp_path / "audit.db", timeout=1)
    raw.execute("CREATE TABLE IF NOT EXISTS lock_probe (x)")
    raw.commit()
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        raw.execute("UPDATE specs SET status = 'complete'")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        raw.execute("DELETE FROM specs")


async def test_transcript_retention_purge(store):
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    assert await store.purge_transcripts_older_than(0) == 1
    assert await store.get_transcript(spec.provenance.transcript_ref) is None
    assert await store.get_spec(str(spec.spec_id)) is not None  # the spec itself is retained


async def test_store_rejects_newer_schema(tmp_path):
    path = tmp_path / "future.db"
    s = await SqliteStore(str(path)).open()
    await s.db.execute("UPDATE schema_version SET version = 99")
    await s.db.commit()
    await s.close()
    with pytest.raises(RuntimeError, match="newer"):
        await SqliteStore(str(path)).open()


# --- adapter -----------------------------------------------------------------------------------------


def test_predict_request_from_complete_spec():
    spec = fixture_spec()
    request = to_predict_request(spec)
    assert request["query"] == "burning acidity, for 3 weeks, after meals, worse with spicy food"
    ctx = request["context"]
    assert ctx["confidence_floor"] == 0.95 and ctx["spec_id"] == str(spec.spec_id)
    assert ctx["safety"]["age_years"] == 34 and ctx["safety"]["pregnancy_status"] is None
    assert ctx["safety"]["current_medications"][0]["name"] == "pantoprazole"
    assert ctx["safety"]["allergies"] == []
    assert SymptomSpec.model_validate(ctx["symptom_spec"]) == spec
    json.dumps(request)  # fully serialisable


def test_declined_safety_values_are_none_not_empty():
    data = json.loads((FIXTURES / "symptom_spec_complete.json").read_text())
    data["safety_profile"]["current_medications"] = {
        "value": None,
        "confidence": 0.0,
        "capture": CaptureMode.declined,
        "turn_ids": ["x:1"],
    }
    data["confidence_floor"] = 0.0
    ctx = to_predict_request(SymptomSpec.model_validate(data))["context"]
    assert ctx["safety"]["current_medications"] is None  # "unknown", never "takes nothing"


def test_escalated_spec_cannot_be_adapted():
    with pytest.raises(ValueError):
        to_predict_request(fixture_spec("symptom_spec_escalated.json"))


# --- worker ------------------------------------------------------------------------------------------


def worker(store, **kw) -> RecommenderHandoff:
    return RecommenderHandoff(store, REC_URL, timeout_s=1.0, backoff_s=0.0, **kw)


@respx.mock
async def test_eligible_spec_is_delivered(store):
    route = respx.post(f"{REC_URL}/predict").mock(
        return_value=httpx.Response(200, json={"status": "recommendation", "outcome": {"formula": "X"}})
    )
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    w = worker(store)
    assert await w.submit(spec, eligible=True) == "queued"
    await w.wait_idle()
    record = await store.get_handoff(str(spec.spec_id))
    assert record.status == "sent" and record.attempts == 1 and record.response["status"] == "recommendation"
    request = route.calls.last.request
    assert request.headers["Idempotency-Key"] == str(spec.spec_id)
    assert json.loads(request.content)["query"].startswith("burning acidity")


@respx.mock
async def test_server_errors_are_retried(store):
    respx.post(f"{REC_URL}/predict").mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectError("down"),
            httpx.Response(200, json={"status": "ok"}),
        ]
    )
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    w = worker(store)
    await w.submit(spec, eligible=True)
    await w.wait_idle()
    record = await store.get_handoff(str(spec.spec_id))
    assert record.status == "sent" and record.attempts == 3


@respx.mock
async def test_retries_exhausted_marks_failed(store):
    respx.post(f"{REC_URL}/predict").mock(side_effect=httpx.ReadTimeout("slow"))
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    w = worker(store)
    await w.submit(spec, eligible=True)
    await w.wait_idle()
    record = await store.get_handoff(str(spec.spec_id))
    assert record.status == "failed" and record.attempts == 3 and record.last_error == "ReadTimeout"


@respx.mock
async def test_client_errors_are_not_retried(store):
    route = respx.post(f"{REC_URL}/predict").mock(return_value=httpx.Response(422, json={"detail": "bad"}))
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    w = worker(store)
    await w.submit(spec, eligible=True)
    await w.wait_idle()
    record = await store.get_handoff(str(spec.spec_id))
    assert record.status == "failed" and record.attempts == 1 and route.call_count == 1


@respx.mock
async def test_ineligible_spec_is_recorded_for_review_and_never_sent(store):
    route = respx.post(f"{REC_URL}/predict")
    spec = fixture_spec("symptom_spec_escalated.json")
    await store.save(spec, session_for(spec))
    assert (
        await worker(store).submit(spec, eligible=False, reason="status:escalated_red_flag") == "not_eligible"
    )
    record = await store.get_handoff(str(spec.spec_id))
    assert record.status == "not_eligible" and record.reason == "status:escalated_red_flag"
    assert route.call_count == 0
    assert [r.spec_id for r in await store.handoffs_with_status("not_eligible")] == [str(spec.spec_id)]


@respx.mock
async def test_pending_handoffs_resume_after_restart(store):
    respx.post(f"{REC_URL}/predict").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    spec = fixture_spec()
    await store.save(spec, session_for(spec))
    await store.create_handoff(spec, "stage_a_recommender", "queued", request=to_predict_request(spec))
    w = worker(store)
    assert await w.resume_pending() == 1
    await w.wait_idle()
    assert (await store.get_handoff(str(spec.spec_id))).status == "sent"


# --- end to end through the API ----------------------------------------------------------------------


def happy_path_extractor() -> tuple[RecordingExtractor, list[str]]:
    extractor = RecordingExtractor()
    names = ["consent_self", "acidity_core", "adult_male", "safety_basic", "confirm"]
    turns = _expand(names)
    for t in turns:
        extractor.recordings[t["user"]] = ExtractionResult(
            intent=Intent(t.get("intent", "answer")), updates=[_update(u) for u in t.get("updates", [])]
        )
    return extractor, [t["user"] for t in turns]


@respx.mock(assert_all_called=False)
def test_chat_intake_persists_spec_and_delivers_handoff(tmp_path, respx_mock):
    respx_mock.route(host="testserver").pass_through()
    respx_mock.post(f"{REC_URL}/predict").mock(
        return_value=httpx.Response(200, json={"status": "recommendation", "n_citations": 4})
    )
    extractor, messages = happy_path_extractor()

    def deps(settings: Settings, store: SqliteStore) -> IntakeDeps:
        return IntakeDeps(
            extractor=extractor,
            responder=BaseQuestionResponder(),
            red_flag_classifier=NullRedFlagClassifier(),
            spec_sink=store,
            handoff=RecommenderHandoff(store, REC_URL, timeout_s=1.0, backoff_s=0.0),
            policy=IntakePolicy(ayurvedic_context_enabled=False),
        )

    settings = Settings(_env_file=None, database_url=f"sqlite:///{tmp_path}/e2e.db", admin_api_token="admin")
    admin = {"X-Admin-Token": "admin"}
    with TestClient(create_app(settings, deps)) as client:
        sid = client.post("/v1/chat", json={}).json()["session_id"]
        for message in messages:
            last = client.post("/v1/chat", json={"session_id": sid, "message": message}).json()
        assert last["final_status"] == "complete" and "results page" in last["reply"]

        deadline = time.time() + 5
        while time.time() < deadline:
            records = client.get(f"/v1/sessions/{sid}/recommendation", headers=admin).json()
            if records and records[0]["status"] == "sent":
                break
            time.sleep(0.05)
        assert records[0]["status"] == "sent" and records[0]["response"]["n_citations"] == 4

        spec_id = last["spec_ids"][0]
        detail = client.get(f"/v1/specs/{spec_id}?include_transcript=true", headers=admin).json()
        assert detail["spec"]["status"] == "complete"
        assert detail["handoff"]["status"] == "sent"
        assert any(t["text"] == "Just pantoprazole" for t in detail["transcript"])
        assert client.get("/v1/review-queue", headers=admin).json() == []
