"""Voice /generate endpoint: ACE-compatible SSE, per-session isolation (blueprint defect D1), auth, close."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from herbenzo_agent.intake.graph import IntakeDeps
from herbenzo_agent.intake.models import IntakePolicy
from herbenzo_agent.intake.red_flag_model import NullRedFlagClassifier
from herbenzo_agent.intake.responder import BaseQuestionResponder
from herbenzo_agent.server.app import create_app
from herbenzo_agent.server.voice_routes import speakable_sentences, sse_chunk
from herbenzo_agent.settings import Settings
from tests.unit.test_api_and_graph import KeywordExtractor

TOKEN = {"X-Internal-Token": "internal"}


def deps(settings: Settings, store) -> IntakeDeps:
    return IntakeDeps(
        extractor=KeywordExtractor(),
        responder=BaseQuestionResponder(),
        red_flag_classifier=NullRedFlagClassifier(),
        spec_sink=store,
        policy=IntakePolicy(ayurvedic_context_enabled=False),
    )


def app(tmp_path, **kw):
    settings = Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/voice.db",
        internal_api_token=kw.pop("internal_api_token", "internal"),
        admin_api_token="admin",
        **kw,
    )
    return create_app(settings, deps)


def parse_like_nvidia_rag_service(body: str) -> tuple[str, list[str]]:
    """Exactly how NvidiaRAGService._process_context reads the stream: json.loads(line[6:]) per line."""
    text, finishes = "", []
    for line in body.splitlines():
        line = line.strip("\n")
        if len(line) > 6:
            parsed = json.loads(line[6:])
            text += parsed["choices"][0]["message"]["content"]
            finishes.append(parsed["choices"][0]["finish_reason"])
    return text, finishes


def say(client, session_id, text):
    payload = {
        "messages": [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": text},
        ],
        "session_id": session_id,
        "use_knowledge_base": False,
        "collection_name": "herbenzo-intake",
        "vdb_top_k": 20,
    }
    response = client.post("/generate", json=payload, headers=TOKEN)
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    return parse_like_nvidia_rag_service(response.text)


def test_sse_chunk_format_matches_blueprint():
    chunk = sse_chunk("r1", "Hello.")
    assert chunk.startswith("data: ") and chunk.endswith("\n\n")
    assert json.loads(chunk[6:]) == {
        "id": "r1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": ""}],
    }


def test_sentences_are_split_and_spaced_for_concatenation():
    assert speakable_sentences(["Thanks. How old are you?", "**Bold** text!"]) == [
        "Thanks. ",
        "How old are you? ",
        "Bold text! ",
    ]


def test_voice_session_start_speaks_numbers_as_words(tmp_path):
    with TestClient(app(tmp_path)) as client:
        text, finishes = say(client, "pc-1", "__SESSION_START__")
        assert "welcome to Herbenzo" in text and "one one two" in text and "112" not in text
        assert finishes[-1] == "[DONE]" and all(f == "" for f in finishes[:-1])


def test_concurrent_voice_sessions_are_isolated(tmp_path):
    with TestClient(app(tmp_path)) as client:
        for pc in ("pc-a", "pc-b"):
            say(client, pc, "__SESSION_START__")
            say(client, pc, "yes")
        a, _ = say(client, "pc-a", "me")
        b, _ = say(client, "pc-b", "I'm a caregiver")
        assert "bothering you" in a and "bothering them" in b


def test_session_id_header_is_accepted(tmp_path):
    with TestClient(app(tmp_path)) as client:
        response = client.post(
            "/generate",
            json={"messages": [{"role": "user", "content": "__SESSION_START__"}]},
            headers={**TOKEN, "X-Session-Id": "pc-h"},
        )
        assert "welcome to Herbenzo" in parse_like_nvidia_rag_service(response.text)[0]


def test_generate_requires_session_and_token(tmp_path):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    with TestClient(app(tmp_path)) as client:
        assert client.post("/generate", json=body, headers=TOKEN).status_code == 400
        assert client.post("/generate", json={**body, "session_id": "s"}).status_code == 401
        assert (
            client.post(
                "/generate", json={**body, "session_id": "s"}, headers={"X-Internal-Token": "no"}
            ).status_code
            == 401
        )
        assert (
            client.post("/generate", json={"messages": [], "session_id": "s"}, headers=TOKEN).status_code
            == 400
        )
    with TestClient(app(tmp_path, internal_api_token=None)) as client:
        assert client.post("/generate", json={**body, "session_id": "s"}, headers=TOKEN).status_code == 503


def test_errors_are_spoken_not_raised(tmp_path):
    with TestClient(app(tmp_path)) as client:

        async def boom(*a, **kw):
            raise RuntimeError("model down")

        client.app.state.service.turn = boom
        text, finishes = say(client, "pc-err", "hello")
        assert "something went wrong" in text and finishes[-1] == "[DONE]"


def test_close_after_consent_saves_incomplete_spec(tmp_path):
    with TestClient(app(tmp_path)) as client:
        say(client, "pc-c", "__SESSION_START__")
        say(client, "pc-c", "yes")
        closed = client.post("/v1/sessions/pc-c/close", headers=TOKEN).json()
        assert closed["spec_id"]
        spec = client.get("/v1/sessions/pc-c/spec", headers={"X-Admin-Token": "admin"}).json()[0]
        assert spec["status"] == "incomplete" and spec["status_reason"] == "disconnected"
        text, _ = say(client, "pc-c", "hello again")
        assert "session has ended" in text


def test_voice_spec_records_speech_models(tmp_path):
    with TestClient(app(tmp_path)) as client:
        headers = {**TOKEN, "X-Voice-ASR-Model": "parakeet-test", "X-Voice-TTS-Model": "magpie-test"}
        for text in ("__SESSION_START__", "yes"):
            client.post(
                "/generate",
                json={"messages": [{"role": "user", "content": text}], "session_id": "pc-m"},
                headers=headers,
            )
        client.post("/v1/sessions/pc-m/close", headers=TOKEN)
        spec = client.get("/v1/sessions/pc-m/spec", headers={"X-Admin-Token": "admin"}).json()[0]
        assert (
            spec["provenance"]["asr_model"] == "parakeet-test"
            and spec["provenance"]["tts_model"] == "magpie-test"
        )


def test_close_without_consent_saves_nothing(tmp_path):
    with TestClient(app(tmp_path)) as client:
        say(client, "pc-n", "__SESSION_START__")
        assert client.post("/v1/sessions/pc-n/close", headers=TOKEN).json()["spec_id"] is None
        assert client.post("/v1/sessions/unknown/close", headers=TOKEN).json()["spec_id"] is None
        assert client.post("/v1/sessions/pc-n/close").status_code == 401
