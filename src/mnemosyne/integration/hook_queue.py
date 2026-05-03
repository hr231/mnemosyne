from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from mnemosyne.monitoring.metrics import MetricsRegistry

logger = logging.getLogger(__name__)


class HookQueueFull(Exception):
    """Raised when the hook queue is at capacity."""


@dataclass
class HookItem:
    """A pending session-close event waiting to be persisted to processing_log."""

    session_id: uuid.UUID
    user_id: uuid.UUID
    enqueued_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    attempts: int = 0
    last_error: Optional[str] = None


@dataclass
class _DeadItem:
    session_id: uuid.UUID
    user_id: uuid.UUID
    attempts: int
    last_error: str
    died_at: datetime


class HookQueue:
    """Bounded in-process queue between session-close hooks and the pipeline.

    Producers (the agent server's session-close path) call ``enqueue``.
    A consumer (the pipeline runner or a dedicated drain task) calls
    ``dequeue`` and is responsible for persisting the event into the
    ``memory.processing_log`` table. On persistence failure the consumer
    calls ``requeue_after_failure`` which applies exponential backoff
    and routes to the DLQ once ``max_retries`` is exceeded.
    """

    def __init__(
        self,
        maxsize: int = 1024,
        max_retries: int = 5,
        base_backoff_seconds: float = 0.5,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._queue: asyncio.Queue[HookItem] = asyncio.Queue(maxsize=maxsize)
        self._dlq: list[_DeadItem] = []
        self._max_retries = max_retries
        self._base_backoff = base_backoff_seconds
        self._metrics = metrics

    async def enqueue(self, session_id: uuid.UUID, user_id: uuid.UUID) -> HookItem:
        """Add a session-close event to the queue.

        Raises ``HookQueueFull`` when the queue is at capacity (the caller
        decides whether to drop or synchronously persist).
        """
        item = HookItem(session_id=session_id, user_id=user_id)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise HookQueueFull(
                f"hook queue full (maxsize={self._queue.maxsize})"
            ) from exc
        self._publish_depth()
        return item

    async def dequeue(self) -> HookItem:
        """Block until an item is available, then return it."""
        item = await self._queue.get()
        self._publish_depth()
        return item

    async def requeue_after_failure(self, item: HookItem, error: str) -> None:
        """Increment retry counter, sleep with exponential backoff, then
        either re-enqueue or move to the DLQ.
        """
        item.attempts += 1
        item.last_error = error
        backoff = self._base_backoff * (2 ** (item.attempts - 1))
        logger.warning(
            "hook item %s failed (attempt %d/%d): %s — backing off %.2fs",
            item.session_id,
            item.attempts,
            self._max_retries,
            error,
            backoff,
        )
        await asyncio.sleep(backoff)

        if item.attempts >= self._max_retries:
            self._dlq.append(
                _DeadItem(
                    session_id=item.session_id,
                    user_id=item.user_id,
                    attempts=item.attempts,
                    last_error=error,
                    died_at=datetime.now(timezone.utc),
                )
            )
            logger.error(
                "hook item %s exceeded max_retries=%d — sent to DLQ",
                item.session_id,
                self._max_retries,
            )
            self._publish_dlq()
            self._publish_depth()
            return

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._dlq.append(
                _DeadItem(
                    session_id=item.session_id,
                    user_id=item.user_id,
                    attempts=item.attempts,
                    last_error="queue full on requeue",
                    died_at=datetime.now(timezone.utc),
                )
            )
            self._publish_dlq()
        self._publish_depth()

    def depth(self) -> int:
        """Current number of pending items."""
        return self._queue.qsize()

    def dlq_size(self) -> int:
        """Number of permanently failed items."""
        return len(self._dlq)

    def dlq_items(self) -> list[_DeadItem]:
        """Snapshot copy of the DLQ contents for inspection."""
        return list(self._dlq)

    def _publish_depth(self) -> None:
        if self._metrics is not None:
            self._metrics.set_session_queue_depth(self._queue.qsize())

    def _publish_dlq(self) -> None:
        if self._metrics is not None:
            self._metrics.record_session_dlq()
