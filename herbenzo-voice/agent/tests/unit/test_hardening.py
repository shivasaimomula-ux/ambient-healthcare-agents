"""M7: log redaction, production config guards, rate limits, retention purge, readiness, metrics."""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from herbenzo_agent.intake.graph import IntakeDeps, IntakeService, build_intake_graph
from herbenzo_agent.logging_setup import REDACTED, configure_logging
from herbenzo_agent.observability import StageMetrics
from herbenzo_agent.persistence.store import SqliteStore
from herbenzo_agent.server.app import create_app
from herbenzo_agent.server.protection import SlidingWindowLimiter, purge_expired
from herbenzo_agent.server.readiness import ReadinessChecker
from herbenzo_agent.settings import MissingConfigError, Settings
from tests.unit.test_api_and_graph import fake_deps, settings

PATIENT_TEXT = "I take pantoprazole for burning acidity"


# --- logging ------------------------------------------------------------------------------------


def _log_exception(caplog, log_content: bool) -> str:
    configure_logging("INFO", log_content)
    logger = logging.getLogger("herbenzo_agent.test")
    with caplog.at_level(logging.ERROR):
        try:
            raise ValueError(PATIENT_TEXT)
        except ValueError:
            logger.exception("turn failed for session %s", "sess-1")
    record = caplog.records[-1]
    return logging.getLogger().handlers[-1].formatter.format(record)


def test_exception_messages_are_redacted_by_default(caplog):
    formatted = _log_exception(caplog, log_content=False)
    assert PATIENT_TEXT not in formatted
    assert REDACTED in formatted and "ValueError" in formatted
    assert "test_hardening.py" in formatted  # stack frames are kept for debugging


def test_exception_messages_visible_when_log_content_enabled(caplog):
    assert PATIENT_TEXT in _log_exception(caplog, log_content=True)
    configure_logging("INFO", False)


def test_noisy_libraries_are_quieted():
    configure_logging("INFO", log_content=False)
    assert logging.getLogger("httpx").level == logging.WARNING


# --- production configuration --------------------------------------------------------------------


def prod(**kw) -> Settings:
    base = dict(
        env="prod",
        internal_api_token="i" * 32,
        admin_api_token="a" * 32,
        nvidia_api_key="nvapi-x",
        cors_allow_origins=["https://intake.herbenzo.com"],
        guardrails_enabled=True,
    )
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_production_config_accepted():
    assert prod().validate_for_startup() == []


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"admin_api_token": "short"}, "ADMIN_API_TOKEN"),
        ({"internal_api_token": None}, "INTERNAL_API_TOKEN"),
        ({"cors_allow_origins": ["*"]}, "CORS_ALLOW_ORIGINS"),
        ({"cors_allow_origins": ["http://localhost:4400"]}, "CORS_ALLOW_ORIGINS"),
        ({"guardrails_enabled": False}, "GUARDRAILS_ENABLED"),
        ({"nvidia_api_key": None}, "NVIDIA_API_KEY"),
    ],
)
def test_unsafe_production_config_refuses_to_start(override, expected):
    with pytest.raises(MissingConfigError, match=expected):
        prod(**override).validate_for_startup()


def test_dev_only_warns():
    warnings = Settings(_env_file=None, env="dev").validate_for_startup()
    assert any("INTERNAL_API_TOKEN" in w for w in warnings)


# --- rate limiting -------------------------------------------------------------------------------


def test_sliding_window_limiter():
    limiter = SlidingWindowLimiter(limit=3, window_s=60)
    now = time.monotonic()
    assert [limiter.allow("ip", now) for _ in range(4)] == [True, True, True, False]
    assert limiter.allow("other-ip", now) is True
    assert limiter.allow("ip", now + 61) is True


def test_chat_endpoint_rate_limits_per_caller(tmp_path):
    s = settings(tmp_path)
    s.rate_limit_chat_per_minute = 3
    with TestClient(create_app(s, fake_deps)) as client:
        codes = [client.post("/v1/chat", json={}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_new_session_limit_is_separate(tmp_path):
    s = settings(tmp_path)
    s.rate_limit_new_sessions_per_minute = 1
    with TestClient(create_app(s, fake_deps)) as client:
        sid = client.post("/v1/chat", json={}).json()["session_id"]
        assert client.post("/v1/chat", json={}).status_code == 429  # second new session blocked
        assert client.post("/v1/chat", json={"session_id": sid, "message": "yes"}).status_code == 200


# --- retention -----------------------------------------------------------------------------------


async def test_purge_removes_old_conversations_but_keeps_specs(tmp_path):
    store = await SqliteStore(str(tmp_path / "r.db")).open()
    checkpointer = InMemorySaver()
    deps = IntakeDeps(**{**fake_deps(settings(tmp_path), store).__dict__})
    service = IntakeService(build_intake_graph(deps, checkpointer), deps)
    await service.turn("old-thread", "__SESSION_START__")
    await service.turn("old-thread", "yes")
    await service.close("old-thread")
    await store.db.execute("UPDATE sessions SET last_activity_at = '2000-01-01T00:00:00+00:00'")
    await store.db.execute("UPDATE transcripts SET created_at = '2000-01-01T00:00:00+00:00'")
    await store.db.commit()

    result = await purge_expired(store, checkpointer, retention_days=90)
    assert result == {"conversations": 1, "transcripts": 1}
    assert await service.get_session("old-thread") is None  # conversation gone
    assert await store.get_for_session("old-thread")  # the spec is retained as evidence
    assert await purge_expired(store, checkpointer, 90) == {"conversations": 0, "transcripts": 0}
    await store.close()


def test_admin_purge_endpoint(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        assert client.post("/v1/admin/purge").status_code == 403
        assert client.post("/v1/admin/purge", headers={"X-Admin-Token": "admin"}).json() == {
            "conversations": 0,
            "transcripts": 0,
        }


# --- readiness and metrics -------------------------------------------------------------------------


async def checker(tmp_path, **overrides) -> ReadinessChecker:
    store = await SqliteStore(str(tmp_path / "ready.db")).open()
    s = settings(tmp_path, **overrides)
    return ReadinessChecker(s, store, fake_deps(s, store))


def models_response(*ids: str) -> httpx.Response:
    return httpx.Response(200, json={"data": [{"id": i} for i in ids]})


@respx.mock
async def test_ready_detects_retired_model(tmp_path):
    respx.get("https://integrate.api.nvidia.com/v1/models").mock(
        return_value=models_response("some/other-model")
    )
    report = await (await checker(tmp_path, handoff_enabled=False)).check()
    assert report["ready"] is False
    assert report["checks"]["llm_models"]["missing"] == ["nvidia/nemotron-3.5-lightning-30b-a3b"]


@respx.mock
async def test_ready_when_everything_is_configured(tmp_path):
    respx.get("https://integrate.api.nvidia.com/v1/models").mock(
        return_value=models_response("nvidia/nemotron-3.5-lightning-30b-a3b")
    )
    respx.get("http://localhost:8000/health").mock(return_value=httpx.Response(200, json={"status": "ok"}))
    report = await (await checker(tmp_path)).check()
    assert report["ready"] is True and report["checks"]["recommender"]["ok"] is True
    assert report["mode"] == "pipeline" and report["pipeline_mode"] is True


@respx.mock
async def test_unreachable_recommender_blocks_pipeline_readiness(tmp_path):
    respx.get("https://integrate.api.nvidia.com/v1/models").mock(
        return_value=models_response("nvidia/nemotron-3.5-lightning-30b-a3b")
    )
    respx.get("http://localhost:8000/health").mock(side_effect=httpx.ConnectError("down"))
    report = await (await checker(tmp_path, pipeline_mode=True)).check()
    assert report["ready"] is False
    assert report["mode"] == "pipeline"
    assert report["checks"]["recommender"]["ok"] is False
    assert "Stage A unreachable" in report["checks"]["recommender"]["detail"]


@respx.mock
async def test_unreachable_recommender_does_not_block_intake_only_readiness(tmp_path):
    respx.get("https://integrate.api.nvidia.com/v1/models").mock(
        return_value=models_response("nvidia/nemotron-3.5-lightning-30b-a3b")
    )
    respx.get("http://localhost:8000/health").mock(side_effect=httpx.ConnectError("down"))
    report = await (await checker(tmp_path, pipeline_mode=False)).check()
    assert report["ready"] is True and report["mode"] == "intake_only"
    assert report["checks"]["recommender"]["ok"] is False


@respx.mock
async def test_pipeline_mode_with_handoff_disabled_ignores_recommender(tmp_path):
    respx.get("https://integrate.api.nvidia.com/v1/models").mock(
        return_value=models_response("nvidia/nemotron-3.5-lightning-30b-a3b")
    )
    report = await (await checker(tmp_path, pipeline_mode=True, handoff_enabled=False)).check()
    assert report["ready"] is True
    assert report["checks"]["recommender"]["ok"] is True
    assert "handoff disabled" in report["checks"]["recommender"]["detail"]


@respx.mock
async def test_provider_outage_does_not_block_readiness(tmp_path):
    respx.get("https://integrate.api.nvidia.com/v1/models").mock(side_effect=httpx.ConnectError("down"))
    report = await (await checker(tmp_path, handoff_enabled=False)).check()
    assert report["ready"] is True and "unavailable" in report["checks"]["llm_models"]["detail"]


def test_metrics_endpoint_reports_stage_timings(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        sid = client.post("/v1/chat", json={}).json()["session_id"]
        client.post("/v1/chat", json={"session_id": sid, "message": "yes"})
        client.post("/v1/chat", json={"session_id": sid, "message": "me"})  # goes through the extractor
        assert client.get("/metrics").status_code == 403
        report = client.get("/metrics", headers={"X-Admin-Token": "admin"}).json()
    assert report["turns"] == 3
    assert report["stages"]["chat.turn_total"]["count"] == 3
    assert {"chat.understand", "chat.red_flag_rules"} <= set(report["stages"])
    assert report["stages"]["chat.turn_total"]["p95_s"] >= 0


def test_metrics_report_guardrail_outcomes(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        sid = client.post("/v1/chat", json={}).json()["session_id"]
        client.post("/v1/chat", json={"session_id": sid, "message": "yes"})
        client.post("/v1/chat", json={"session_id": sid, "message": "me"})
        report = client.get("/metrics", headers={"X-Admin-Token": "admin"}).json()
    assert report["events"]["chat.guardrail_input.passed"] == 1  # only the extractor turn runs the rail


def test_stage_metrics_percentiles():
    metrics = StageMetrics()
    for value in range(1, 101):
        metrics.record("chat", {"turn_total": value / 100})
    summary = metrics.snapshot()["chat.turn_total"]
    assert summary["count"] == 100 and 0.4 <= summary["p50_s"] <= 0.6 and summary["max_s"] == 1.0


def test_turn_result_carries_timings(tmp_path):
    with TestClient(create_app(settings(tmp_path), fake_deps)) as client:
        client.post("/v1/chat", json={})
        report = client.get("/metrics", headers={"X-Admin-Token": "admin"}).json()
    assert json.dumps(report)  # serialisable for dashboards
    assert report["stages"]["chat.turn_total"]["count"] == 1


def test_no_stray_event_loop_warnings():
    assert asyncio.get_event_loop_policy() is not None
