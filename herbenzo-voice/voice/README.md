# Herbenzo voice controller

Browser microphone ⇄ WebRTC ⇄ Silero VAD ⇄ Riva ASR (Parakeet) ⇄ **Herbenzo intake agent** ⇄ Riva TTS (Magpie) ⇄ browser speaker.
Built on NVIDIA ACE Controller / Pipecat, adapted from the ambient-patient blueprint (see `pipeline_herbenzo.py` header for what changed).

## Run locally on a Mac (recommended for development)

Everything reads the repo-root `.env` (needs `NVIDIA_API_KEY`, `INTERNAL_API_TOKEN`, `ADMIN_API_TOKEN`).

Terminal 1, the intake agent:

```bash
cd ~/Desktop/ambient-healthcare-agents/herbenzo-voice/agent && uv run uvicorn herbenzo_agent.server.app:create_app --factory --port 8081
```

Terminal 2, the voice controller:

```bash
cd ~/Desktop/ambient-healthcare-agents/herbenzo-voice/voice && uv run python pipeline_herbenzo.py --port 7860
```

Terminal 3, the web page (first time: `./ui/setup_ui.sh`):

```bash
cd ~/Desktop/ambient-healthcare-agents/herbenzo-voice/voice/ui/.build/webrtc_ui && npm run dev -- --host localhost --port 4400
```

Open **http://localhost:4400 in Chrome**, press **Start**, allow the microphone, and talk. `localhost` counts as a
secure origin, so no certificate is needed. Optional Terminal 4: the Stage A Recommender for handoffs
(`cd ~/Desktop/main.py && python3 -m uvicorn predictor.api:app --port 8000`).

## Tuning

| Setting | Where | Notes |
|---|---|---|
| ASR/TTS endpoints, voice, language | `configs/riva_public.yaml` / `riva_self_hosted.yaml`, chosen by `VOICE_CONFIG_PATH` | Hosted NVCF by default |
| Filler phrase while the agent thinks | `Pipeline.filler_probability` in the config | 0 = off; e.g. 0.5 says "One moment." on half the turns |
| Pronunciations | `ipa.json` | Listen and adjust; ASR can't verify these |
| TURN server (callers outside your network) | `TURN_SERVER_URL`, `TURN_USERNAME`, `TURN_PASSWORD` | Required for most cloud deployments |

## Tests

```bash
uv run pytest -q tests
```

The live end-to-end check (a headless WebRTC caller speaking synthesized audio) is described in `../DECISIONS.md` (M6 entries).
