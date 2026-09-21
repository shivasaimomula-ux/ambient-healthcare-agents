"""Voice-facing endpoints used by the ACE Controller (Pipecat) voice service.

`POST /generate` reproduces the SSE contract that `nvidia_pipecat.services.nvidia_rag.NvidiaRAGService`
parses (see ambient-patient `chain_server.py`), with one addition the blueprint lacked: every request must
identify its conversation (`session_id`), so concurrent callers never share memory.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable, Iterator

import bleach
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from herbenzo_agent.intake.graph import SESSION_START, IntakeService
from herbenzo_agent.observability import StageMetrics
from herbenzo_agent.server.protection import SessionLockMap
from herbenzo_agent.settings import Settings

logger = logging.getLogger(__name__)

MAX_UTTERANCE_CHARS = 2000
VOICE_ERROR_REPLY = "Sorry, something went wrong on my side. Could you say that again?"
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN = re.compile(r"[*_`#>]+")


class VoiceMessage(BaseModel):
    role: str
    content: str = ""


class GenerateRequest(BaseModel):
    """Superset of what NvidiaRAGService sends; its extra RAG fields are accepted and ignored."""

    model_config = ConfigDict(extra="ignore")
    messages: list[VoiceMessage] = Field(default_factory=list, max_length=500)
    session_id: str | None = None


def sse_chunk(response_id: str, content: str, finish_reason: str = "") -> str:
    payload = {
        "id": response_id,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}
        ],
    }
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def speakable_sentences(chunks: list[str]) -> list[str]:
    """Plain sentences for TTS, one SSE chunk each, so speech can start before the whole reply is sent.

    Each carries a trailing space because the client concatenates chunks without separators.
    """
    sentences: list[str] = []
    for chunk in chunks:
        text = _MARKDOWN.sub("", chunk).strip()
        sentences.extend(s.strip() + " " for s in _SENTENCE.split(text) if s.strip())
    return sentences


def stream(sentences: list[str]) -> Iterator[str]:
    response_id = str(uuid.uuid4())
    for sentence in sentences:
        yield sse_chunk(response_id, sentence)
    yield sse_chunk(response_id, "", finish_reason="[DONE]")


def register_voice_routes(
    app: FastAPI,
    settings: Settings,
    locks: SessionLockMap,
    valid_thread: Callable[[str], str],
    metrics: StageMetrics,
) -> None:
    def require_internal(token: str | None) -> None:
        if not settings.internal_api_token:
            raise HTTPException(
                status_code=503, detail="INTERNAL_API_TOKEN is not configured on the agent server"
            )
        if token != settings.internal_api_token:
            raise HTTPException(status_code=401, detail="invalid internal token")

    @app.post("/generate")
    async def generate(
        body: GenerateRequest,
        request: Request,
        x_session_id: str | None = Header(default=None),
        x_internal_token: str | None = Header(default=None),
        x_voice_asr_model: str | None = Header(default=None),
        x_voice_tts_model: str | None = Header(default=None),
    ) -> StreamingResponse:
        require_internal(x_internal_token)
        session_id = body.session_id or x_session_id
        if not session_id:
            raise HTTPException(
                status_code=400, detail="session_id is required (body or X-Session-Id header)"
            )
        thread_id = valid_thread(session_id)
        utterance = next((m.content for m in reversed(body.messages) if m.role == "user"), None)
        if utterance is None:
            raise HTTPException(status_code=400, detail="no user message")
        text = utterance.strip()
        if text != SESSION_START:
            text = bleach.clean(text, strip=True)[:MAX_UTTERANCE_CHARS]

        service: IntakeService = request.app.state.service
        try:
            async with locks[thread_id]:
                speech = {"asr": (x_voice_asr_model or "")[:128], "tts": (x_voice_tts_model or "")[:128]}
                result = await service.turn(thread_id, text, channel="voice", speech_models=speech)
            metrics.record("voice", result.timings, result.counters)
            logger.info(
                "voice turn session=%s phase=%s total=%.2fs",
                thread_id,
                result.phase,
                result.timings.get("turn_total", 0.0),
            )
            sentences = speakable_sentences(result.reply_chunks)
        except Exception:
            logger.exception("voice turn failed for session %s", thread_id)
            sentences = [VOICE_ERROR_REPLY]
        return StreamingResponse(stream(sentences), media_type="text/event-stream")

    @app.post("/v1/sessions/{thread_id}/close")
    async def close_session(
        thread_id: str, request: Request, x_internal_token: str | None = Header(default=None)
    ) -> dict:
        require_internal(x_internal_token)
        thread_id = valid_thread(thread_id)
        service: IntakeService = request.app.state.service
        async with locks[thread_id]:
            spec_id = await service.close(thread_id)
        locks.pop(thread_id, None)
        return {"session_id": thread_id, "spec_id": spec_id}
