from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from mnemosyne.db.models.processing import ProcessingLog

if TYPE_CHECKING:
    from mnemosyne.integration.hook_queue import HookQueue
    from mnemosyne.integration.session_assembler import (
        PersistentSessionAssembler,
        SessionAssembler,
    )
    from mnemosyne.pipeline.session_pipeline import SessionPipeline

logger = logging.getLogger(__name__)


async def on_session_close(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    provider=None,
    queue: "HookQueue | None" = None,
    *,
    assembler: "SessionAssembler | PersistentSessionAssembler | None" = None,
    session_pipeline: "SessionPipeline | None" = None,
) -> ProcessingLog:
    """Record a session for processing.

    Three modes:

    - Legacy / queued: only ``session_id``, ``user_id`` (and optional
      ``queue``) are passed. Returns a pending ``ProcessingLog`` for the
      background runner to pick up. When ``queue`` is provided, the event
      is also enqueued; on ``HookQueueFull`` the queue path falls back to
      direct return.

    - Streamed: ``assembler`` and ``session_pipeline`` are passed; the
      assembler is flushed and the resulting batch is processed inline.
      Returns a ``ProcessingLog`` with ``status="completed"`` (or
      ``"empty"`` if the buffer was empty).

    The streamed path is non-blocking from the agent server's perspective
    only insofar as the caller awaits it; this function is intended for
    short batches.
    """
    if assembler is not None and session_pipeline is not None:
        batch = await assembler.flush(session_id)
        if batch is None:
            logger.info("Session %s closed with no buffered messages", session_id)
            return ProcessingLog(
                session_id=session_id,
                pipeline_step="extraction",
                status="empty",
            )
        await session_pipeline.process_session_batch(batch)
        logger.info(
            "Session %s processed inline (%d messages)",
            session_id,
            len(batch.messages),
        )
        return ProcessingLog(
            session_id=session_id,
            pipeline_step="extraction",
            status="completed",
        )

    if queue is not None:
        from mnemosyne.integration.hook_queue import HookQueueFull

        try:
            await queue.enqueue(session_id=session_id, user_id=user_id)
        except HookQueueFull:
            logger.error(
                "hook queue full — session %s falling back to direct return",
                session_id,
            )

    entry = ProcessingLog(
        session_id=session_id,
        pipeline_step="extraction",
        status="pending",
    )
    logger.info("Session %s queued for processing", session_id)
    return entry
