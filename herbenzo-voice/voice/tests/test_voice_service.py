"""Offline tests for the voice controller: agent client identity, config, and app wiring."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from config import VOICE_DIR, Env, load_config
from herbenzo_agent_service import COLLECTION_NAME, HerbenzoAgentService
from pipeline_herbenzo import create_app, load_ipa_dict, new_session_id


def env(**kw) -> Env:
    base = dict(
        nvidia_api_key="k",
        agent_url="http://agent.test",
        internal_api_token="internal",
        config_path=VOICE_DIR / "configs/riva_public.yaml",
        enable_speculative_speech=False,
        dump_audio_files=False,
        turn_server_url=None,
        turn_username=None,
        turn_password=None,
        allowed_origins=["http://localhost:4400"],
    )
    base.update(kw)
    return Env(**base)


@respx.mock
async def test_agent_requests_carry_session_and_token():
    route = respx.post("http://agent.test/generate").mock(
        return_value=httpx.Response(200, text="data: {}\n\n")
    )
    service = HerbenzoAgentService(
        session_id="voice-abc",
        internal_token="internal",
        rag_server_url="http://agent.test",
        session=httpx.AsyncClient(),
        asr_model="parakeet",
        tts_model="magpie",
    )
    await service._get_rag_response(
        {"messages": [{"role": "user", "content": "hi"}], "collection_name": COLLECTION_NAME}
    )
    request = route.calls.last.request
    assert request.headers["X-Session-Id"] == "voice-abc"
    assert request.headers["X-Internal-Token"] == "internal"
    assert json.loads(request.content)["session_id"] == "voice-abc"
    assert (
        request.headers["X-Voice-ASR-Model"] == "parakeet"
        and request.headers["X-Voice-TTS-Model"] == "magpie"
    )


def test_session_ids_are_unique_and_safe():
    ids = {new_session_id() for _ in range(100)}
    assert len(ids) == 100 and all("#" not in i and i.startswith("voice-") for i in ids)
    with pytest.raises(ValueError):
        HerbenzoAgentService(
            session_id="SmallWebRTCConnection#0", internal_token="t", rag_server_url="http://x"
        )


@pytest.mark.parametrize("name", ["riva_public.yaml", "riva_self_hosted.yaml"])
def test_configs_load(name):
    config = load_config(VOICE_DIR / "configs" / name)
    assert config.RivaASRService.sample_rate == 16000 and config.RivaTTSService.voice_id


def test_ipa_dictionary_covers_brand_terms():
    ipa = load_ipa_dict()
    assert {"Herbenzo", "Namaste", "Tele-MANAS"} <= set(ipa)


def test_app_routes():
    app = create_app(env(), load_config(VOICE_DIR / "configs/riva_public.yaml"))
    paths = {getattr(r, "path", None) for r in app.routes}
    assert {"/ws", "/get_prompt", "/health"} <= paths
