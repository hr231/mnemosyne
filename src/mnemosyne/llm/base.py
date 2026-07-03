from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from mnemosyne.db.models.memory import ExtractionResult
from mnemosyne.monitoring.metrics import global_registry

logger = logging.getLogger(__name__)

DEFAULT_LLM_MAX_CONCURRENCY = 4

_llm_concurrency_limit = DEFAULT_LLM_MAX_CONCURRENCY
_llm_semaphore: asyncio.Semaphore | None = None


def configure_llm_concurrency(limit: int) -> None:
    """Set the process-wide ceiling on concurrent outbound LLM calls.

    Call once during bootstrap. A limit below 1 is coerced to 1. Resets the
    shared semaphore so the new limit takes effect immediately.
    """
    global _llm_concurrency_limit, _llm_semaphore
    _llm_concurrency_limit = max(1, int(limit))
    _llm_semaphore = asyncio.Semaphore(_llm_concurrency_limit)


def llm_semaphore() -> asyncio.Semaphore:
    """Return the shared semaphore bounding concurrent outbound LLM calls."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(_llm_concurrency_limit)
    return _llm_semaphore


def record_llm_usage(tokens: int | None = None) -> None:
    """Record one outbound LLM call and, when known, its token consumption.

    Metrics objects are duck-typed; missing hooks are tolerated so a slim
    registry never breaks the call path.
    """
    registry = global_registry()
    record_call = getattr(registry, "record_llm_call", None)
    if callable(record_call):
        try:
            record_call(1)
        except Exception:  # noqa: BLE001
            logger.debug("record_llm_call failed", exc_info=True)
    if tokens is not None:
        record_tokens = getattr(registry, "record_llm_tokens", None)
        if callable(record_tokens):
            try:
                record_tokens(int(tokens))
            except Exception:  # noqa: BLE001
                logger.debug("record_llm_tokens failed", exc_info=True)


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Raw text completion."""
        ...

    @abstractmethod
    async def extract_memories(self, text: str) -> list[ExtractionResult]:
        """Extract structured memories from text."""
        ...

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        """Create an LLMClient from a config dict.

        Supported providers:
            "openai"              — OpenAI SDK (direct)
            "azure"               — Azure OpenAI (same SDK)
            "anthropic"           — Anthropic SDK
            "google"              — Google GenAI SDK
            "openai_compatible"   — any /v1/chat/completions endpoint
        """
        provider = config.get("provider")
        if not provider:
            raise ValueError("LLM provider not specified in config")

        timeout = config.get("timeout", 60.0)
        max_retries = config.get("max_retries", 3)

        if provider == "openai":
            from mnemosyne.llm.openai_sdk import OpenAILLMClient
            return OpenAILLMClient(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config.get("api_key"),
                timeout=timeout,
                max_retries=max_retries,
            )

        if provider == "azure":
            from mnemosyne.llm.openai_sdk import OpenAILLMClient
            return OpenAILLMClient(
                model=config.get("model", "gpt-4o-mini"),
                api_key=config.get("api_key"),
                azure_endpoint=config["azure_endpoint"],
                api_version=config.get("api_version"),
                timeout=timeout,
                max_retries=max_retries,
            )

        if provider == "anthropic":
            from mnemosyne.llm.anthropic_sdk import AnthropicLLMClient
            return AnthropicLLMClient(
                model=config.get("model", "claude-sonnet-4-20250514"),
                api_key=config.get("api_key"),
                timeout=timeout,
                max_retries=max_retries,
            )

        if provider == "google":
            from mnemosyne.llm.google_sdk import GoogleLLMClient
            return GoogleLLMClient(
                model=config.get("model", "gemini-2.0-flash"),
                api_key=config.get("api_key"),
                timeout=timeout,
                max_retries=max_retries,
            )

        if provider == "openai_compatible":
            from mnemosyne.llm.openai_compatible import OpenAICompatibleClient
            return OpenAICompatibleClient(
                base_url=config["base_url"],
                model=config["model"],
                api_key=config.get("api_key"),
                timeout=timeout,
                max_retries=max_retries,
            )

        raise ValueError(f"Unknown LLM provider: {provider!r}")
