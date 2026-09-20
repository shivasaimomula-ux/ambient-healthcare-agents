# Adapted from NVIDIA ambient-patient ace-controller-voice-interface/pipeline-patient.py
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Herbenzo voice controller: browser WebRTC audio <-> Riva ASR/TTS <-> Herbenzo intake agent.

Changes from the blueprint:
- Each call gets its own random session id, sent to the agent on every request (blueprint had none).
- The greeting is requested with a `__SESSION_START__` turn (blueprint sent an empty message and errored).
- A disconnect closes the intake on the agent (saves an `incomplete` spec if the caller had consented).
- The UI "dynamic prompt"/context-reset path is removed: the agent owns all wording.
- CORS is restricted to configured origins; audio recording stays off unless DUMP_AUDIO_FILES=true.

Run: uv run python pipeline_herbenzo.py --port 7860
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from nvidia_pipecat.processors.acknowledgment import AcknowledgmentProcessor
from nvidia_pipecat.processors.audio_util import AudioRecorder
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.processors.transcript_synchronization import (
    BotTranscriptSynchronization,
    UserTranscriptSynchronization,
)
from nvidia_pipecat.services.riva_speech import RivaASRService, RivaTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import InputAudioRawFrame, LLMMessagesFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

from config import VOICE_DIR, Env, VoiceConfig, load_config, load_env
from herbenzo_agent_service import HerbenzoAgentService
from websocket_transcript_output import WebsocketTranscriptOutput

SESSION_START = "__SESSION_START__"


def new_session_id() -> str:
    # Never use the WebRTC pc_id: it looks like "SmallWebRTCConnection#0" and repeats after a restart.
    return f"voice-{uuid.uuid4().hex}"


def load_ipa_dict() -> dict[str, str]:
    return json.loads((VOICE_DIR / "ipa.json").read_text(encoding="utf-8"))


def build_services(env: Env, config: VoiceConfig, session_id: str) -> dict:
    agent = HerbenzoAgentService(
        session_id=session_id,
        internal_token=env.internal_api_token,
        rag_server_url=env.agent_url,
        asr_model=config.RivaASRService.model_label,
        tts_model=config.RivaTTSService.model_label,
        max_tokens=config.Agent.max_tokens,
        session=httpx.AsyncClient(timeout=config.Agent.request_timeout_s),
    )
    # Only forward the streaming / Two-Pass EOU knobs that the yaml actually sets, so an
    # unset field keeps the RivaASRService library default instead of being clobbered to None.
    asr_tuning = {
        k: v
        for k, v in {
            "stop_history": config.RivaASRService.stop_history,
            "stop_threshold": config.RivaASRService.stop_threshold,
            "stop_history_eou": config.RivaASRService.stop_history_eou,
            "stop_threshold_eou": config.RivaASRService.stop_threshold_eou,
            "custom_configuration": config.RivaASRService.custom_configuration,
        }.items()
        if v is not None
    }
    stt = RivaASRService(
        server=config.RivaASRService.server,
        api_key=env.nvidia_api_key,
        language=config.RivaASRService.language,
        sample_rate=config.RivaASRService.sample_rate,
        automatic_punctuation=True,
        model=config.RivaASRService.model,
        function_id=config.RivaASRService.function_id,
        **asr_tuning,
    )
    tts = RivaTTSService(
        server=config.RivaTTSService.server,
        api_key=env.nvidia_api_key,
        voice_id=config.RivaTTSService.voice_id,
        model=config.RivaTTSService.model,
        function_id=config.RivaTTSService.function_id,
        language=config.RivaTTSService.language,
        ipa_dict=load_ipa_dict(),
    )
    return {"agent": agent, "stt": stt, "tts": tts}


async def close_agent_session(env: Env, session_id: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{env.agent_url}/v1/sessions/{session_id}/close",
                headers={"X-Internal-Token": env.internal_api_token},
            )
            logger.info(f"closed agent session {session_id}: {response.status_code}")
    except httpx.HTTPError as exc:
        logger.warning(f"could not close agent session {session_id}: {exc}")


def create_app(env: Env, config: VoiceConfig) -> FastAPI:
    app = FastAPI(title="Herbenzo Voice Controller")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=env.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    pcs_map: dict[str, SmallWebRTCConnection] = {}
    sessions: dict[str, str] = {}
    ice_servers = (
        [
            IceServer(
                urls=env.turn_server_url, username=env.turn_username or "", credential=env.turn_password or ""
            )
        ]
        if env.turn_server_url
        else []
    )

    async def run_bot(webrtc_connection: SmallWebRTCConnection, ws: WebSocket, session_id: str) -> None:
        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_10ms_chunks=5,
        )
        transport = SmallWebRTCTransport(webrtc_connection=webrtc_connection, params=transport_params)
        services = build_services(env, config, session_id)

        recorders = []
        if env.dump_audio_files:
            logger.warning("DUMP_AUDIO_FILES=true: caller audio is being written to disk")
            dumps = VOICE_DIR / "audio_dumps"
            dumps.mkdir(exist_ok=True)
            recorders = [
                AudioRecorder(
                    output_file=str(dumps / f"asr_{session_id}.wav"),
                    params=transport_params,
                    frame_type=InputAudioRawFrame,
                ),
                AudioRecorder(
                    output_file=str(dumps / f"tts_{session_id}.wav"),
                    params=transport_params,
                    frame_type=TTSAudioRawFrame,
                ),
            ]

        context = OpenAILLMContext([])
        if env.enable_speculative_speech:
            # Only safe for stateless backends; the Herbenzo agent keeps memory, so keep this off.
            context_aggregator = create_nvidia_context_aggregator(context, send_interims=True)
            tts_cacher = NvidiaTTSResponseCacher()
        else:
            context_aggregator = services["agent"].create_context_aggregator(context)
            tts_cacher = None

        filler = (
            [
                AcknowledgmentProcessor(
                    filler_words=config.Pipeline.filler_words,
                    filler_probability=config.Pipeline.filler_probability,
                )
            ]
            if config.Pipeline.filler_probability > 0
            else []
        )

        pipeline = Pipeline(
            [
                transport.input(),
                *recorders[:1],
                services["stt"],
                UserTranscriptSynchronization(),
                *filler,
                context_aggregator.user(),
                services["agent"],
                services["tts"],
                *recorders[1:],
                *([tts_cacher] if tts_cacher else []),
                BotTranscriptSynchronization(),
                WebsocketTranscriptOutput(ws),
                transport.output(),
                context_aggregator.assistant(),
            ]
        )
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                send_initial_empty_metrics=True,
                start_metadata={"stream_id": session_id},
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            await asyncio.sleep(0.05)
            await task.queue_frames([LLMMessagesFrame([{"role": "user", "content": SESSION_START}])])

        await PipelineRunner(handle_sigint=False).run(task)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            request = await websocket.receive_json()
            pc_id = request.get("pc_id")
            if pc_id and pc_id in pcs_map:
                connection = pcs_map[pc_id]
                logger.info(f"renegotiating {pc_id}")
                await connection.renegotiate(sdp=request["sdp"], type=request["type"])
            else:
                connection = SmallWebRTCConnection(ice_servers)
                await connection.initialize(sdp=request["sdp"], type=request["type"])
                session_id = new_session_id()

                @connection.event_handler("closed")
                async def handle_closed(conn: SmallWebRTCConnection):
                    pcs_map.pop(conn.pc_id, None)
                    closed_session = sessions.pop(conn.pc_id, None)
                    if closed_session:
                        await close_agent_session(env, closed_session)

                asyncio.create_task(run_bot(connection, websocket, session_id))

            answer = connection.get_answer()
            pcs_map[answer["pc_id"]] = connection
            if not (pc_id and pc_id in sessions):
                sessions[answer["pc_id"]] = session_id
            await websocket.send_json(answer)

            # Keep the socket open for transcripts. Client messages are ignored: prompts are owned by the agent.
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("client disconnected from websocket")

    @app.get("/get_prompt")
    async def get_prompt():
        # The web UI polls this; the prompt lives in the agent, not here.
        return {
            "prompt": "Managed by the Herbenzo intake agent",
            "name": "Herbenzo intake",
            "description": "",
        }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "agent_url": env.agent_url,
            "asr": config.RivaASRService.model_label,
            "tts": config.RivaTTSService.model_label,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Herbenzo voice controller")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="TRACE" if args.verbose else "INFO")
    env = load_env()
    config = load_config(env.config_path)
    uvicorn.run(create_app(env, config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
