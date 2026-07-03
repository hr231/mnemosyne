from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def content_hash(content: str) -> str:
    """Canonical sha256 hash of normalised content (strip + lower)."""
    return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract a Retry-After delay from an httpx.HTTPStatusError, if present."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    value = response.headers.get("retry-after") if hasattr(response, "headers") else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _is_retryable_http(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is None:
        return True
    status = getattr(response, "status_code", None)
    return status is None or status in _RETRYABLE_STATUS


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    honor_retry_after: bool = True,
) -> T:
    """Run ``fn`` with exponential backoff and jitter.

    Retries on exceptions in ``retry_on``. HTTP errors carrying a response are
    only retried for 429/5xx status codes; a Retry-After header is honored when
    ``honor_retry_after`` is set. The final attempt re-raises the last error.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retry_on as exc:
            if not _is_retryable_http(exc) or attempt >= max_retries:
                raise
            last_exc = exc
            delay = min(max_delay, base_delay * (2**attempt))
            delay += random.uniform(0, delay * 0.25)
            if honor_retry_after:
                hinted = _retry_after_seconds(exc)
                if hinted is not None:
                    delay = min(max_delay, hinted)
            logger.warning(
                "Retryable error (attempt %d/%d), sleeping %.2fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    raise last_exc if last_exc else RuntimeError("retry_async exhausted without exception")
