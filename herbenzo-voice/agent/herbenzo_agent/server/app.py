"""FastAPI app: chat intake API, audit/review endpoints (voice /generate is added in M6).

Run: uv run uvicorn herbenzo_agent.server.app:create_app --factory --port 8081
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import bleach
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from herbenzo_agent import __version__
from herbenzo_agent.contracts.export_schema import build_schema
from herbenzo_agent.guardrails.nemoguard import Guardrails, NemoGuardRails, NoopGuardrails
from herbenzo_agent.handoff.worker import NoopHandoff, RecommenderHandoff
from herbenzo_agent.intake.extractor import LLMExtractor
from herbenzo_agent.intake.graph import SESSION_START, IntakeDeps, IntakeService, build_intake_graph
from herbenzo_agent.intake.models import IntakePolicy
from herbenzo_agent.intake.red_flag_model import LLMRedFlagClassifier
from herbenzo_agent.intake.responder import GuardedResponder, LLMResponder
from herbenzo_agent.llm_factory import get_llm, role_model_id
from herbenzo_agent.logging_setup import configure_logging
from herbenzo_agent.observability import StageMetrics
from herbenzo_agent.persistence.checkpointer import open_checkpointer
from herbenzo_agent.persistence.store import SqliteStore
from herbenzo_agent.server.protection import (
    SlidingWindowLimiter,
    client_key,
    enforce,
    purge_expired,
    retention_loop,
)
from herbenzo_agent.server.readiness import ReadinessChecker
from herbenzo_agent.server.voice_routes import register_voice_routes
from herbenzo_agent.settings import Settings, get_settings

logger = logging.getLogger(__name__)

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-:.]{1,128}$")  # "#" is reserved for restart suffixes
MAX_MESSAGE_CHARS = 2000


AGENT_DIR = Path(__file__).resolve().parents[2]


def build_guardrails(settings: Settings) -> Guardrails:
    if not settings.guardrails_enabled:
        return NoopGuardrails()
    path = Path(settings.guardrails_config_path)
    if not path.is_absolute():
        path = AGENT_DIR / path
    return NemoGuardRails(
        path,
        api_key=settings.nvidia_api_key,
        base_url=settings.guardrails_base_url,
        timeout_s=settings.guardrails_timeout_s,
        topic_control_enabled=settings.guardrails_topic_control,
    )


def default_deps(settings: Settings, store: SqliteStore) -> IntakeDeps:
    guardrails = build_guardrails(settings)
    responder = LLMResponder(get_llm(settings, "responder"), timeout_s=settings.responder_timeout_s)
    return IntakeDeps(
        extractor=LLMExtractor(get_llm(settings, "extractor"), timeout_s=settings.llm_extractor_timeout_s),
        responder=(
            GuardedResponder(responder, guardrails)
            if settings.guardrails_enabled and settings.guardrails_output_enabled
            else responder
        ),
        red_flag_classifier=LLMRedFlagClassifier(get_llm(settings, "red_flag")),
        guardrails=guardrails,
        spec_sink=store,
        handoff=(
            RecommenderHandoff(store, settings.recommender_url, timeout_s=settings.recommender_timeout_s)
            if settings.handoff_enabled
            else NoopHandoff()
        ),
        policy=policy_from(settings),
        llm_models={role: role_model_id(settings, role) for role in ("extractor", "responder", "red_flag")},
        handoff_enabled=settings.handoff_enabled,
        handoff_min_confidence=settings.handoff_min_confidence,
        default_language=settings.default_language,
        default_jurisdiction=settings.default_jurisdiction,
    )


def policy_from(settings: Settings) -> IntakePolicy:
    return IntakePolicy(
        min_adult_age=settings.min_adult_age,
        max_user_turns=settings.max_user_turns,
        ayurvedic_context_enabled=settings.ayurvedic_context_enabled,
    )


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None, description="Omit to start a new session (returns the greeting)."
    )
    message: str = Field(default="", max_length=MAX_MESSAGE_CHARS)


class ChatResponse(BaseModel):
    session_id: str
    thread_id: str
    reply: str
    reply_chunks: list[str]
    phase: str
    final_status: str | None
    spec_ids: list[str]


def create_app(
    settings: Settings | None = None,
    deps_factory: Callable[[Settings, SqliteStore], IntakeDeps] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    deps_factory = deps_factory or default_deps
    locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    metrics = StageMetrics()
    chat_limiter = SlidingWindowLimiter(settings.rate_limit_chat_per_minute)
    new_session_limiter = SlidingWindowLimiter(settings.rate_limit_new_sessions_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level, settings.log_content)
        for warning in settings.validate_for_startup():
            logger.warning("startup: %s", warning)
        async with AsyncExitStack() as stack:
            checkpointer = await stack.enter_async_context(open_checkpointer(settings.database_url))
            store = await SqliteStore.from_database_url(settings.database_url).open()
            stack.push_async_callback(store.close)
            deps = deps_factory(settings, store)
            for component in (deps.guardrails, deps.handoff):
                if hasattr(component, "aclose"):
                    stack.push_async_callback(component.aclose)
            if hasattr(deps.handoff, "resume_pending"):
                await deps.handoff.resume_pending()
            app.state.store = store
            app.state.deps = deps
            app.state.metrics = metrics
            app.state.checkpointer = checkpointer
            app.state.readiness = ReadinessChecker(settings, store, deps)
            app.state.service = IntakeService(build_intake_graph(deps, checkpointer), deps)
            purge = asyncio.create_task(
                retention_loop(
                    store,
                    checkpointer,
                    settings.transcript_retention_days,
                    settings.retention_purge_interval_s,
                )
            )

            async def stop_purge() -> None:
                purge.cancel()
                await asyncio.gather(purge, return_exceptions=True)

            stack.push_async_callback(stop_purge)
            logger.info("herbenzo agent %s ready (db=%s)", __version__, settings.database_url.split("///")[0])
            yield

    app = FastAPI(title="Herbenzo Intake Agent", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Content-Type",
            "X-Admin-Token",
            "X-Internal-Token",
            "X-Chat-Token",
            "X-Session-Id",
        ],
    )

    def service(request: Request) -> IntakeService:
        return request.app.state.service

    def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if not settings.admin_api_token or x_admin_token != settings.admin_api_token:
            raise HTTPException(status_code=403, detail="admin token required")

    def require_chat_bootstrap(x_chat_token: str | None = Header(default=None)) -> None:
        """Prod profile: reject unauthenticated public chat. Localhost demo (ENV=dev) skips this."""
        if not settings.requires_chat_auth():
            return
        expected = settings.chat_bootstrap_token
        if not expected or x_chat_token != expected:
            raise HTTPException(status_code=401, detail="chat bootstrap token required")

    def valid_thread(thread_id: str) -> str:
        if not SESSION_ID_RE.match(thread_id):
            raise HTTPException(status_code=400, detail="invalid session_id")
        return thread_id

    register_voice_routes(app, settings, locks, valid_thread, metrics)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    async def ready(request: Request, response: Response) -> dict:
        report = await request.app.state.readiness.check()
        if not report["ready"]:
            response.status_code = 503
        return report

    @app.get("/metrics", dependencies=[Depends(require_admin)])
    async def turn_metrics(request: Request) -> dict:
        registry = request.app.state.metrics
        return {"turns": registry.turns, "stages": registry.snapshot(), "events": dict(registry.counters)}

    @app.post("/v1/admin/purge", dependencies=[Depends(require_admin)])
    async def purge_now(request: Request) -> dict:
        """Run the retention purge immediately (the same job runs on a timer)."""
        return await purge_expired(
            request.app.state.store, request.app.state.checkpointer, settings.transcript_retention_days
        )

    @app.get("/v1/contract/symptom-spec")
    async def contract() -> dict:
        return build_schema()

    @app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_chat_bootstrap)])
    async def chat(
        body: ChatRequest, request: Request, svc: IntakeService = Depends(service)
    ) -> ChatResponse:
        caller = client_key(request, settings.trust_forwarded_for)
        enforce(chat_limiter, caller)
        if body.session_id is None:
            enforce(new_session_limiter, caller)
            thread_id, text = uuid.uuid4().hex, SESSION_START
        else:
            thread_id = valid_thread(body.session_id.split("#")[0])
            text = bleach.clean(body.message, strip=True)
        async with locks[thread_id]:
            result = await svc.turn(thread_id, text, channel="chat")
        metrics.record("chat", result.timings, result.counters)
        logger.info(
            "turn session=%s phase=%s total=%.2fs",
            thread_id,
            result.phase,
            result.timings.get("turn_total", 0.0),
        )
        return ChatResponse(
            session_id=thread_id,
            thread_id=thread_id,
            reply=result.reply,
            reply_chunks=result.reply_chunks,
            phase=result.phase,
            final_status=result.final_status.value if result.final_status else None,
            spec_ids=result.spec_ids,
        )

    @app.get("/v1/sessions/{thread_id}", dependencies=[Depends(require_admin)])
    async def get_session(
        thread_id: str, include_values: bool = False, svc: IntakeService = Depends(service)
    ):
        session = await svc.get_session(valid_thread(thread_id))
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        data = {
            "session_id": session.session_id,
            "phase": session.phase.value,
            "final_status": session.final_status,
            "user_turns": session.user_turn_count,
            "filled_slots": sorted(session.slots),
        }
        if include_values:
            data["slots"] = {k: v.model_dump(mode="json") for k, v in session.slots.items()}
            data["turns"] = [t.model_dump(mode="json") for t in session.turns]
        return data

    @app.get("/v1/sessions/{thread_id}/spec", dependencies=[Depends(require_admin)])
    async def get_specs(thread_id: str, request: Request):
        specs = await request.app.state.deps.spec_sink.get_for_session(valid_thread(thread_id))
        if not specs:
            raise HTTPException(status_code=404, detail="no spec for session")
        return [s.model_dump(mode="json") for s in specs]

    @app.get("/v1/sessions/{thread_id}/recommendation", dependencies=[Depends(require_admin)])
    async def get_recommendation(thread_id: str, request: Request):
        records = await request.app.state.store.handoffs_for_thread(valid_thread(thread_id))
        if not records:
            raise HTTPException(status_code=404, detail="no handoff for session")
        return [r.public() for r in records]

    @app.get("/v1/specs/{spec_id}", dependencies=[Depends(require_admin)])
    async def get_spec(spec_id: str, request: Request, include_transcript: bool = False):
        store: SqliteStore = request.app.state.store
        spec = await store.get_spec(valid_thread(spec_id))
        if spec is None:
            raise HTTPException(status_code=404, detail="spec not found")
        data: dict = {"spec": spec.model_dump(mode="json")}
        handoff = await store.get_handoff(str(spec.spec_id))
        data["handoff"] = handoff.public(include_response=False) if handoff else None
        if include_transcript:
            data["transcript"] = await store.get_transcript(spec.provenance.transcript_ref)
        return data

    @app.get("/v1/review-queue", dependencies=[Depends(require_admin)])
    async def review_queue(request: Request, status: str = "not_eligible,failed", limit: int = 100):
        """Specs that need a human: not eligible for automated review, or delivery failed."""
        statuses = [
            s for s in status.split(",") if s in ("not_eligible", "failed", "queued", "in_progress", "sent")
        ]
        if not statuses:
            raise HTTPException(status_code=400, detail="invalid status filter")
        records = await request.app.state.store.handoffs_with_status(*statuses, limit=min(limit, 500))
        return [r.public(include_response=False) for r in records]

    return app
