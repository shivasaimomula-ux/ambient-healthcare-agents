"""Chat model per role (extractor, responder, red_flag), chosen by environment."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from herbenzo_agent.settings import NVIDIA_BASE_URL, Settings

Role = Literal["extractor", "responder", "red_flag"]

_TEMPERATURE: dict[str, float] = {"extractor": 0.0, "red_flag": 0.0, "responder": 0.3}
_MAX_TOKENS: dict[str, int] = {"extractor": 900, "red_flag": 120, "responder": 120}


class LLMConfigError(RuntimeError):
    pass


def role_model_id(settings: Settings, role: Role) -> str:
    return f"{getattr(settings, f'llm_{role}_provider')}:{getattr(settings, f'llm_{role}_model')}"


def get_llm(settings: Settings, role: Role) -> BaseChatModel:
    provider = getattr(settings, f"llm_{role}_provider")
    model = getattr(settings, f"llm_{role}_model")
    base_url = getattr(settings, f"llm_{role}_base_url")
    temperature, max_tokens = _TEMPERATURE[role], _MAX_TOKENS[role]

    if provider == "nvidia":
        if not settings.nvidia_api_key and not base_url:
            raise LLMConfigError(
                "NVIDIA_API_KEY is required for the nvidia provider (or set a self-hosted BASE_URL)."
            )
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        return ChatNVIDIA(
            model=model,
            base_url=base_url or NVIDIA_BASE_URL,
            api_key=settings.nvidia_api_key,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            # Reasoning models think for ~30s by default; intake turns need answers in seconds.
            model_kwargs={"chat_template_kwargs": {"enable_thinking": False}}
            if settings.llm_disable_thinking
            else {},
        )
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise LLMConfigError("Install langchain-anthropic to use the anthropic provider.") from exc
        return ChatAnthropic(
            model=model, api_key=settings.anthropic_api_key, temperature=temperature, max_tokens=max_tokens
        )
    if provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise LLMConfigError("Install langchain-google-genai to use the google provider.") from exc
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    raise LLMConfigError(f"Provider {provider!r} has no chat model; use the fake components in tests.")
