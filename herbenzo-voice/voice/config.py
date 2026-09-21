# Adapted from NVIDIA ambient-patient ace-controller-voice-interface/config.py
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Voice pipeline configuration (YAML) plus environment settings."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

VOICE_DIR = Path(__file__).resolve().parent


class PipelineConfig(BaseModel):
    filler_words: list[str] = Field(default_factory=lambda: ["One moment."])
    filler_probability: float = Field(default=0.0, ge=0.0, le=1.0)


class AgentConfig(BaseModel):
    max_tokens: int = 1024
    request_timeout_s: float = 45.0


class RivaASRConfig(BaseModel):
    server: str
    language: str = "en-US"
    sample_rate: int = 16000
    model: str | None = None
    function_id: str | None = None
    model_label: str = "parakeet-ctc-1.1b-asr"
    # Streaming / Two-Pass end-of-utterance tuning. Left unset -> the RivaASRService
    # library defaults apply (stop_history=500, stop_history_eou=240). Set these in the
    # yaml to trade endpointing latency against the risk of clipping slow talkers, or to
    # pass an operating-point / VAD string to a cache-aware streaming NIM.
    stop_history: int | None = None
    stop_threshold: float | None = None
    stop_history_eou: int | None = None
    stop_threshold_eou: float | None = None
    custom_configuration: str | None = None


class RivaTTSConfig(BaseModel):
    server: str
    language: str = "en-US"
    voice_id: str
    model: str | None = None
    function_id: str | None = None
    model_label: str = "magpie-tts-multilingual"


class VoiceConfig(BaseModel):
    Pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    Agent: AgentConfig = Field(default_factory=AgentConfig)
    RivaASRService: RivaASRConfig
    RivaTTSService: RivaTTSConfig


class Env(BaseModel):
    nvidia_api_key: str | None
    agent_url: str
    internal_api_token: str
    config_path: Path
    enable_speculative_speech: bool
    dump_audio_files: bool
    turn_server_url: str | None
    turn_username: str | None
    turn_password: str | None
    allowed_origins: list[str]


def load_env() -> Env:
    load_dotenv(find_dotenv(usecwd=True), override=False)
    token = os.getenv("INTERNAL_API_TOKEN")
    if not token:
        raise RuntimeError("INTERNAL_API_TOKEN must be set (same value as the agent server).")
    config_path = Path(os.getenv("VOICE_CONFIG_PATH", "configs/riva_public.yaml"))
    if not config_path.is_absolute():
        config_path = VOICE_DIR / config_path
    return Env(
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        agent_url=os.getenv("AGENT_URL", "http://localhost:8081").rstrip("/"),
        internal_api_token=token,
        config_path=config_path,
        enable_speculative_speech=os.getenv("ENABLE_SPECULATIVE_SPEECH", "false").lower() == "true",
        dump_audio_files=os.getenv("DUMP_AUDIO_FILES", "false").lower() == "true",
        turn_server_url=os.getenv("TURN_SERVER_URL") or None,
        turn_username=os.getenv("TURN_USERNAME") or None,
        turn_password=os.getenv("TURN_PASSWORD") or None,
        allowed_origins=[
            o.strip() for o in os.getenv("VOICE_ALLOWED_ORIGINS", "http://localhost:4400").split(",")
        ],
    )


def load_config(path: Path) -> VoiceConfig:
    return VoiceConfig.model_validate(yaml.safe_load(path.read_text()))
