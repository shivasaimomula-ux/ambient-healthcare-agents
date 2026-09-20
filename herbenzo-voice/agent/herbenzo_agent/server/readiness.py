"""Readiness: is this instance actually able to serve intakes right now?

`/health` says the process is alive; `/ready` checks the things that silently break a deployment:
the database, whether the configured LLMs still exist on the provider (NVIDIA retires hosted models on
fixed dates), whether the guardrail config loaded, and whether Stage A is reachable.

In `PIPELINE_MODE=true` (default), Stage A reachability is required for ready — a down Recommender
must not look “ready” for the F→A pipeline. Set `PIPELINE_MODE=false` for intake-only soft-fail.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from herbenzo_agent.intake.graph import IntakeDeps
from herbenzo_agent.persistence.store import SqliteStore
from herbenzo_agent.settings import NVIDIA_BASE_URL, Settings

logger = logging.getLogger(__name__)

MODEL_CACHE_S = 600.0


class ReadinessChecker:
    def __init__(
        self,
        settings: Settings,
        store: SqliteStore,
        deps: IntakeDeps,
        client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings
        self.store = store
        self.deps = deps
        self._client = client or httpx.AsyncClient(timeout=10)
        self._models: tuple[float, set[str]] | None = None

    async def available_models(self) -> set[str]:
        now = time.monotonic()
        if self._models and now - self._models[0] < MODEL_CACHE_S:
            return self._models[1]
        base = self.settings.llm_extractor_base_url or NVIDIA_BASE_URL
        headers = (
            {"Authorization": f"Bearer {self.settings.nvidia_api_key}"}
            if self.settings.nvidia_api_key
            else {}
        )
        response = await self._client.get(f"{base.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        models = {entry["id"] for entry in response.json().get("data", [])}
        self._models = (now, models)
        return models

    async def _check_models(self) -> dict[str, Any]:
        configured = {
            role: getattr(self.settings, f"llm_{role}_model")
            for role in ("extractor", "responder", "red_flag")
            if getattr(self.settings, f"llm_{role}_provider") == "nvidia"
        }
        if not configured:
            return {"ok": True, "detail": "no hosted NVIDIA models configured"}
        try:
            available = await self.available_models()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            # Can't prove they are gone; don't fail readiness on a transient provider problem.
            return {"ok": True, "detail": f"model list unavailable ({type(exc).__name__})"}
        missing = sorted({model for model in configured.values() if model not in available})
        return {
            "ok": not missing,
            "missing": missing,
            "detail": "retired or unknown model" if missing else "ok",
        }

    async def _check_recommender(self) -> dict[str, Any]:
        url = self.settings.recommender_url.rstrip("/")
        if not self.settings.handoff_enabled:
            return {"ok": True, "detail": "handoff disabled", "url": url}
        try:
            response = await self._client.get(f"{url}/health", timeout=5)
            if response.is_success:
                return {"ok": True, "status": response.status_code, "url": url}
            return {
                "ok": False,
                "status": response.status_code,
                "url": url,
                "detail": f"Stage A health returned HTTP {response.status_code} at {url}",
            }
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "url": url,
                "detail": f"Stage A unreachable at {url} ({type(exc).__name__})",
            }

    def _required_checks(self) -> tuple[str, ...]:
        required = ["database", "llm_models", "guardrails"]
        # Pipeline mode + handoff on: Stage A must be up or /ready is 503.
        if self.settings.pipeline_mode and self.settings.handoff_enabled:
            required.append("recommender")
        return tuple(required)

    async def check(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        try:
            checks["database"] = {"ok": await self.store.ping()}
        except Exception as exc:
            logger.exception("readiness: database check failed")
            checks["database"] = {"ok": False, "detail": type(exc).__name__}
        checks["llm_models"] = await self._check_models()
        checks["guardrails"] = {
            "ok": self.deps.guardrails.config_sha256 != "missing",
            "config_sha256": self.deps.guardrails.config_sha256,
        }
        checks["recommender"] = await self._check_recommender()
        required = self._required_checks()
        ready = all(checks[name]["ok"] for name in required)
        mode = "pipeline" if self.settings.pipeline_mode else "intake_only"
        if not ready and "recommender" in required and not checks["recommender"]["ok"]:
            logger.warning(
                "readiness failed in pipeline mode: %s",
                checks["recommender"].get("detail", "Stage A not ready"),
            )
        return {
            "ready": ready,
            "mode": mode,
            "pipeline_mode": self.settings.pipeline_mode,
            "llm_models": self.deps.llm_models,
            "checks": checks,
        }

    async def aclose(self) -> None:
        await self._client.aclose()
