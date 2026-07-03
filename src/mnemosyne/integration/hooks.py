from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from mnemosyne.db.models.processing import ProcessingLog
from mnemosyne.db.repositories.processing_log import insert_pending

if TYPE_CHECKING:
    from mnemosyne.integration.hook_queue import HookQueue
    from mnemosyne.integration.session_assembler import (
        PersistentSessionAssembler,
        SessionAssembler,
    )
    from mnemosyne.pipeline.session_pipeline import SessionPipeline

logger = logging.getLogger(__name__)


def _record_session_failure() -> None:
    """Best-effort metric for an isolated session-close hook failure.

    Kept distinct from the dead-letter counter: an isolated hook failure is
    not a dead letter, so it must not inflate ``session_dlq``. Uses the
    registry's ``record_hook_failure`` when available and otherwise stays
    silent — only the queue's own retry-exhaustion path counts as a DLQ event.
    """
    try:
        from mnemosyne.monitoring.metrics import global_registry

        record = getattr(global_registry(), "record_hook_failure", None)
        if record is not None:
            record()
    except Exception:  # pragma: no cover - metrics never break the hook
        pass


async def _extraction_allowed(
    pool, user_id: uuid.UUID, session_id: uuid.UUID, fail_open: bool
) -> bool:
    """Return whether the durable opt-out permits extraction for ``user_id``.

    On gate-lookup error the call defaults to allowing extraction (so a
    transient settings-read failure never silently drops a user's session)
    unless ``fail_open`` is False, in which case the error propagates.
    """
    from mnemosyne.integration.extraction_gate import is_extraction_enabled_for

    try:
        return await is_extraction_enabled_for(pool, user_id)
    except Exception as exc:
        if not fail_open:
            raise
        logger.warning(
            "Session %s extraction-gate lookup failed (defaulting to allow): %s",
            session_id,
            exc,
            exc_info=True,
        )
        return True


async def on_session_close(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    provider=None,
    queue: "HookQueue | None" = None,
    *,
    pool=None,
    assembler: "SessionAssembler | PersistentSessionAssembler | None" = None,
    session_pipeline: "SessionPipeline | None" = None,
    fail_open: bool = True,
) -> ProcessingLog:
    """Record a session for processing.

    Modes:

    - Streamed: ``assembler`` and ``session_pipeline`` are passed; the batch
      is read and processed inline. When the assembler supports a non-deleting
      ``peek`` plus ``delete_session`` (the Postgres-backed assembler), the
      buffered turns are removed only AFTER processing succeeds, so a failure
      leaves them durable for retry. Returns a ``ProcessingLog`` with
      ``status="completed"`` (or ``"empty"`` when the buffer was empty).

    - Durable / queued: when ``pool`` is provided, a durable pending row is
      written to ``memory.processing_log`` and its id is reflected on the
      returned ``ProcessingLog``; the background runner's poll loop picks it
      up. Otherwise, when ``queue`` is provided, the event is enqueued as a
      non-blocking buffer. ``pool`` and ``queue`` are mutually exclusive — a
      present pool takes the durable path and the queue is not also enqueued
      (that would persist a second row and double-process the session).

    Extraction opt-out: when a ``pool`` is available the per-user extraction
    flag is consulted first. A disabled user's session is skipped entirely —
    no pending row is written and no streamed batch is processed — and a
    ``ProcessingLog`` with ``status="skipped"`` is returned.

    Failure isolation: when ``fail_open`` is True (the default) any error in
    the streamed inline processing or the durable insert is swallowed — the
    hook logs, records a hook-failure metric and returns a ``ProcessingLog``
    rather than propagating into the host agent. Set ``fail_open=False`` to
    surface errors (used by tests).
    """
    if pool is not None and not await _extraction_allowed(
        pool, user_id, session_id, fail_open
    ):
        logger.info(
            "Session %s skipped — extraction disabled for user %s",
            session_id,
            user_id,
        )
        return ProcessingLog(
            session_id=session_id,
            user_id=user_id,
            pipeline_step="extraction",
            status="skipped",
        )

    if assembler is not None and session_pipeline is not None:
        return await _process_streamed(
            session_id, user_id, assembler, session_pipeline, fail_open
        )

    log_id = uuid.uuid4()
    if pool is not None:
        try:
            log_id = await insert_pending(pool, session_id, user_id)
        except Exception as exc:
            if not fail_open:
                raise
            logger.warning(
                "Session %s durable enqueue failed (isolated): %s",
                session_id,
                exc,
                exc_info=True,
            )
            _record_session_failure()
    elif queue is not None:
        from mnemosyne.integration.hook_queue import HookQueueFull

        try:
            await queue.enqueue(session_id=session_id, user_id=user_id)
        except HookQueueFull:
            logger.error(
                "hook queue full — session %s falling back to direct return",
                session_id,
            )

    entry = ProcessingLog(
        id=log_id,
        session_id=session_id,
        user_id=user_id,
        pipeline_step="extraction",
        status="pending",
    )
    logger.info("Session %s queued for processing", session_id)
    return entry


async def _process_streamed(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    assembler: "SessionAssembler | PersistentSessionAssembler",
    session_pipeline: "SessionPipeline",
    fail_open: bool,
) -> ProcessingLog:
    """Process a session batch inline, deleting buffered turns only on success.

    A durable assembler (one exposing ``peek`` + ``delete_session``) is read
    without deleting; its rows are removed only after the pipeline succeeds so
    a mid-processing failure leaves them for retry. An in-process assembler
    with only ``flush`` keeps its prior read-and-clear behaviour.
    """
    durable = hasattr(assembler, "peek") and hasattr(assembler, "delete_session")
    try:
        if durable:
            batch = await assembler.peek(session_id)
        else:
            batch = await assembler.flush(session_id)
        if batch is None:
            logger.info("Session %s closed with no buffered messages", session_id)
            return ProcessingLog(
                session_id=session_id,
                user_id=user_id,
                pipeline_step="extraction",
                status="empty",
            )
        await session_pipeline.process_session_batch(batch)
        if durable:
            await assembler.delete_session(session_id)
        logger.info(
            "Session %s processed inline (%d messages)",
            session_id,
            len(batch.messages),
        )
        return ProcessingLog(
            session_id=session_id,
            user_id=user_id,
            pipeline_step="extraction",
            status="completed",
        )
    except Exception as exc:
        if not fail_open:
            raise
        logger.warning(
            "Session %s streamed processing failed (isolated, buffer preserved): %s",
            session_id,
            exc,
            exc_info=True,
        )
        _record_session_failure()
        return ProcessingLog(
            session_id=session_id,
            user_id=user_id,
            pipeline_step="extraction",
            status="failed",
            error_message=str(exc),
        )
