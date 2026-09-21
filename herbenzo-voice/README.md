# Herbenzo Voice — Stage F intake agent

Voice/chat intake that screens for emergencies, captures symptoms and safety history, reads them back, and emits a
validated **`SymptomSpec`** to the Stage A Recommender. Built from the NVIDIA ambient-patient blueprint; full design in
[`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md), deviations in [`DECISIONS.md`](DECISIONS.md).

## Build status

| Milestone | Status |
|---|---|
| M0 Scaffold | ✅ |
| M1 SymptomSpec contract (`contracts/`, `agent/herbenzo_agent/contracts/`) | ✅ |
| M2 Deterministic core (slots, planner, red flags, scripts, readback, lint, spec builder) | ✅ |
| M3 LLM nodes + LangGraph + chat API | ✅ |
| M4 Guardrails (NemoGuard content safety, input + output) | ✅ |
| M5 Audit store + handoff to Stage A Recommender | ✅ |
| M6 Voice (ACE Controller / Riva, WebRTC UI, Docker/Caddy) | ✅ (browser mic test by Shiva pending; Docker files not yet run) |
| M7 Hardening (redaction, retention, metrics, readiness, rate limits) | ✅ (Postgres deferred) |

## Layout

```
herbenzo-voice/
├── contracts/                 generated JSON Schema + changelog
├── docs/BUILD_SPEC.md         the architecture/build spec
├── agent/                     FastAPI + LangGraph intake service (Python 3.12, uv)
│   ├── herbenzo_agent/
│   │   ├── contracts/         SymptomSpec models (source of truth)
│   │   ├── intake/            models, slots, planner, red_flags, scripts, readback, lint, spec_builder
│   │   └── data_files/        red_flags.yaml, jurisdictions.yaml, blocked_terms.txt
│   └── tests/
└── voice/                     ACE Controller voice pipeline, WebRTC UI, smoke test (see voice/README.md)
```

## Develop

Secrets live in the repo-root `.env` (git-ignored); see `.env.example` for every variable.

```bash
cd herbenzo-voice/agent
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python -m herbenzo_agent.contracts.export_schema   # after any contract change
```

## Try the chat intake

```bash
cd herbenzo-voice/agent
uv run uvicorn herbenzo_agent.server.app:create_app --factory --port 8081
```

```bash
curl -s -X POST localhost:8081/v1/chat -H 'Content-Type: application/json' -d '{}'
```

That returns the greeting and a `session_id`. Send each answer with it:

```bash
curl -s -X POST localhost:8081/v1/chat -H 'Content-Type: application/json' -d '{"session_id":"<id>","message":"yes"}'
```

**Profiles:** `ENV=dev` (default) is the **localhost demo** — `/v1/chat` stays open behind rate
limits (no token). **Prod profile** (`ENV=prod`, typically with `PIPELINE_MODE=true`) requires
`CHAT_BOOTSTRAP_TOKEN` and header `X-Chat-Token` on every chat turn; startup refuses a weak/missing
token. Override with `CHAT_AUTH_REQUIRED=true|false`.

```bash
# Prod / pipeline caller
curl -s -X POST localhost:8081/v1/chat \
  -H 'Content-Type: application/json' \
  -H "X-Chat-Token: $CHAT_BOOTSTRAP_TOKEN" \
  -d '{}'
```

With `RECOMMENDER_URL` pointing at a running Stage A (`uvicorn predictor.api:app --port 8000`),
every eligible intake is delivered in the background as
`POST {RECOMMENDER_URL}/predict` with `query` plus rich `context` (`symptom_spec`, `safety`,
`confidence_floor`). **`PIPELINE_MODE=true` (default):** `GET /ready` returns **503** when Stage A
is unreachable — the agent will not look “ready” for the F→A path. Input guardrails also **fail
closed**: if NemoGuard / content-safety is unavailable (timeout, empty/unparseable response), the
turn is rejected (same deflection as an explicit block) instead of proceeding. Set
`PIPELINE_MODE=false` for **intake-only demo** (handoffs still queue/retry; readiness ignores A;
unavailable NemoGuard fails open so local chat still works when the hosted NIM is flaky).
`ENV=prod` keeps fail-closed even if `PIPELINE_MODE=false`. Review endpoints (header
`X-Admin-Token: $ADMIN_API_TOKEN`):

| Endpoint | Returns |
|---|---|
| `GET /v1/sessions/{id}/spec` | SymptomSpec(s) for a session |
| `GET /v1/sessions/{id}/recommendation` | handoff status and the Recommender's response |
| `GET /v1/specs/{spec_id}?include_transcript=true` | one spec, its handoff, and the transcript it came from |
| `GET /v1/review-queue` | specs needing a human: not eligible for automated review, or delivery failed |

Live tests against the real model (about 3 minutes): `HERBENZO_LIVE_TESTS=1 uv run pytest -s tests/conversation/test_live.py`

## Operations

| Endpoint | Purpose |
|---|---|
| `GET /health` | process alive (no auth) |
| `GET /ready` | database, configured LLMs, guardrail config; **in pipeline mode** also Stage A `/health`; 503 when not ready |
| `GET /metrics` | rolling p50/p95/max per stage and per channel, plus guardrail outcome counts (admin token) |
| `POST /v1/admin/purge` | run the retention purge now; it also runs every `RETENTION_PURGE_INTERVAL_S` (admin token) |

**Measured on 2026-09-16** (16-turn chat intake, hosted NVIDIA endpoints, guardrails on):
turn p50 **2.8 s**, p95 11.9 s; extractor p50 1.6 s; responder p50 1.4 s; input guardrail capped at its 1.5 s
timeout and unavailable on 8 of 13 turns (pipeline mode now fails closed on unavailable; intake-only
demo still fails open). Voice adds ASR + TTS on top: first audio
3–6 s after short answers, ~13 s after a long description.

### Data retention

Conversations (LangGraph checkpoints) and transcripts older than `TRANSCRIPT_RETENTION_DAYS` (default 90) are
deleted automatically. **SymptomSpecs are never deleted** — they are the evidence record, and the database
refuses updates and deletes on them.

### Security checklist before a pilot

- [ ] `ENV=prod` (refuses to start with weak tokens, `*`/http CORS, guardrails off, or `LOG_CONTENT=true`)
- [ ] `INTERNAL_API_TOKEN`, `ADMIN_API_TOKEN`, and `CHAT_BOOTSTRAP_TOKEN` are fresh 32-character random values, not in git
- [ ] Callers send `X-Chat-Token` on `POST /v1/chat` (prod profile); localhost demo keeps `ENV=dev` without a token
- [ ] Only `/`, `/api/v1/chat` and `/api/health` are exposed publicly (see `voice/ui/Caddyfile`); `/generate`,
      `/v1/sessions/*` and `/metrics` stay on the internal network
- [ ] HTTPS in front (Caddy) — browsers need a secure origin for the microphone
- [ ] `TRUST_FORWARDED_FOR=true` only when running behind that proxy (rate limits key on client IP)
- [ ] `DUMP_AUDIO_FILES=false`; database volume encrypted at rest; backups cover `specs` and `handoffs`
- [ ] Clinician sign-off on `red_flags.yaml`, consent and escalation scripts, and emergency numbers
- [ ] Counsel sign-off on the consent text, retention period and storing escalation records before consent

## Safety data that needs clinician review before any pilot

- `agent/herbenzo_agent/data_files/red_flags.yaml` — emergency lexicon
- `agent/herbenzo_agent/intake/scripts.py` — consent, escalation and closing wording
- `agent/herbenzo_agent/data_files/jurisdictions.yaml` — emergency numbers
