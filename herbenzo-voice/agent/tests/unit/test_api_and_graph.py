"""Chat API, per-session isolation, persistence across restarts, and extractor parsing."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from herbenzo_agent.intake.extractor import parse_extraction, slot_catalogue
from herbenzo_agent.intake.graph import IntakeDeps
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
from herbenzo_agent.server.app import create_app
from herbenzo_agent.settings import Settings


class KeywordExtractor:
    """Tiny fake: 'yes' confirms, 'me' sets reporter_role, 'caregiver' sets caregiver."""

    async def extract(self, session: IntakeSession, turn: Turn) -> ExtractionResult:
        text = turn.text.lower()
        if text in ("yes", "yes please"):
            return ExtractionResult(intent=Intent.confirm_yes)
        if "caregiver" in text:
            return ExtractionResult(
                updates=[SlotUpdate(slot_key="reporter_role", value="caregiver", evidence_quote="caregiver")]
            )
        if text == "me":
            return ExtractionResult(
                updates=[SlotUpdate(slot_key="reporter_role", value="self", evidence_quote="me")]
            )
        return ExtractionResult(intent=Intent.unclear)


def fake_deps(settings: Settings, store) -> IntakeDeps:
    return IntakeDeps(
        extractor=KeywordExtractor(),
        responder=BaseQuestionResponder(),
        red_flag_classifier=NullRedFlagClassifier(),
        spec_sink=store,
        policy=IntakePolicy(ayurvedic_context_enabled=False),
    )


def settings(tmp_path, **kw) -> Settings:
    return Settings(
        _env_file=None, database_url=f"sqlite:///{tmp_path}/test.db", admin_api_token="admin", **kw
    )


def test_chat_flow_and_session_isolation(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        a = client.post("/v1/chat", json={}).json()
        b = client.post("/v1/chat", json={}).json()
        assert a["session_id"] != b["session_id"]
        assert "welcome to Herbenzo" in a["reply"] and a["phase"] == "greeting_consent"

        client.post("/v1/chat", json={"session_id": a["session_id"], "message": "yes"})
        client.post("/v1/chat", json={"session_id": b["session_id"], "message": "yes"})
        ra = client.post("/v1/chat", json={"session_id": a["session_id"], "message": "me"}).json()
        rb = client.post(
            "/v1/chat", json={"session_id": b["session_id"], "message": "I'm a caregiver"}
        ).json()
        assert "What's been bothering you?" in ra["reply"]
        assert (
            "What's been bothering them?" in rb["reply"]
        )  # B's answer never leaked into A (blueprint defect D1)

        sa = client.get(
            f"/v1/sessions/{a['session_id']}?include_values=true", headers={"X-Admin-Token": "admin"}
        ).json()
        assert sa["slots"]["reporter_role"]["value"] == "self"


def test_state_survives_app_restart(tmp_path):
    s = settings(tmp_path)
    with TestClient(create_app(s, fake_deps)) as client:
        sid = client.post("/v1/chat", json={}).json()["session_id"]
        client.post("/v1/chat", json={"session_id": sid, "message": "yes"})
    with TestClient(create_app(s, fake_deps)) as client:
        reply = client.post("/v1/chat", json={"session_id": sid, "message": "me"}).json()
        assert reply["phase"] == "collecting" and "bothering you" in reply["reply"]


def test_admin_endpoints_require_token(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        sid = client.post("/v1/chat", json={}).json()["session_id"]
        assert client.get(f"/v1/sessions/{sid}").status_code == 403
        assert client.get(f"/v1/sessions/{sid}", headers={"X-Admin-Token": "wrong"}).status_code == 403
        assert client.get(f"/v1/sessions/{sid}/spec", headers={"X-Admin-Token": "admin"}).status_code == 404


def test_invalid_session_id_rejected(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        assert client.post("/v1/chat", json={"session_id": "../etc", "message": "hi"}).status_code == 400


def test_empty_message_asks_to_repeat(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        sid = client.post("/v1/chat", json={}).json()["session_id"]
        reply = client.post("/v1/chat", json={"session_id": sid, "message": "   "}).json()
        assert "didn't catch that" in reply["reply"] and reply["phase"] == "greeting_consent"


def test_contract_endpoint(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        assert client.get("/v1/contract/symptom-spec").json()["x-spec-version"] == "1.0.0"


# --- extractor parsing -------------------------------------------------------------------------


def test_parse_extraction_handles_think_tags_and_noise():
    raw = '<think>reasoning...</think>\nHere you go: {"intent":"confirm_yes","updates":[],"red_flag_suspected":[]}'
    assert parse_extraction(raw).intent is Intent.confirm_yes


def test_parse_extraction_sanitises_bad_fields():
    raw = '{"intent":"maybe","updates":[{"slot_key":"subject.age_years","value":34},"junk",{"value":1}],"red_flag_suspected":[1,"RF_STROKE"]}'
    result = parse_extraction(raw)
    assert result.intent is Intent.unclear
    assert [u.slot_key for u in result.updates] == ["subject.age_years"]
    assert result.red_flag_suspected == ["RF_STROKE"]


def test_parse_extraction_rejects_non_json():
    with pytest.raises(ValueError):
        parse_extraction("I think the user said yes")


def test_slot_catalogue_lists_every_slot_with_types():
    catalogue = slot_catalogue()
    assert "- subject.sex_at_birth (required): one of female|male|intersex|prefer_not_to_say" in catalogue
    assert "- symptoms[0].duration (required): duration object" in catalogue
