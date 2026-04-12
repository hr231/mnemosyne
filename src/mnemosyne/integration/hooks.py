from __future__ import annotations

import logging
import uuid

from mnemosyne.db.models.processing import ProcessingLog

logger = logging.getLogger(__name__)


async def on_session_close(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    provider=None,
) -> ProcessingLog:
    """Record a session for background pipeline processing.

    The agent server calls this when a session ends. It creates a
    ProcessingLog entry that the pipeline runner picks up.

    Returns the ProcessingLog entry (the caller can persist it or
    the pipeline runner can manage the processing_log table directly).

    This function is intentionally non-blocking — it does no I/O and
    returns immediately so that session close latency is unaffected.
    """
    entry = ProcessingLog(
        session_id=session_id,
        pipeline_step="extraction",
        status="pending",
    )
    logger.info("Session %s queued for processing", session_id)
    return entry
