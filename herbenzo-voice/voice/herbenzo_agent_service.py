"""NvidiaRAGService that identifies each conversation to the Herbenzo intake agent.

The ambient-patient blueprint used NvidiaRAGService as-is, which sends no session identity; its chain server
then kept one global conversation for every caller. This subclass adds the session id and the shared
internal token to every `/generate` request.
"""

from __future__ import annotations

import httpx
from nvidia_pipecat.services.nvidia_rag import NvidiaRAGService

COLLECTION_NAME = "herbenzo-intake"  # must be non-empty or the base class refuses to send


class HerbenzoAgentService(NvidiaRAGService):
    def __init__(
        self,
        *,
        session_id: str,
        internal_token: str,
        asr_model: str | None = None,
        tts_model: str | None = None,
        **kwargs,
    ):
        kwargs.setdefault("collection_name", COLLECTION_NAME)
        super().__init__(use_knowledge_base=False, enable_citations=False, **kwargs)
        if not session_id or "#" in session_id:
            raise ValueError("session_id must be non-empty and must not contain '#'")
        self.session_id = session_id
        self._internal_token = internal_token
        # Reported to the agent so each SymptomSpec records which speech models produced it.
        self._model_headers = {
            k: v for k, v in {"X-Voice-ASR-Model": asr_model, "X-Voice-TTS-Model": tts_model}.items() if v
        }

    async def _get_rag_response(self, request_json: dict) -> httpx.Response:
        request_json = {**request_json, "session_id": self.session_id}
        return await self.shared_session.post(
            f"{self.rag_server_url}/generate",
            json=request_json,
            headers={
                "X-Session-Id": self.session_id,
                "X-Internal-Token": self._internal_token,
                **self._model_headers,
            },
        )
