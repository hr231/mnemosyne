from __future__ import annotations

import logging
from dataclasses import dataclass

from mnemosyne.integration.session_models import (
    ConversationMessage,
    SessionBatch,
)
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline

logger = logging.getLogger(__name__)


@dataclass
class SessionExtractionStats:
    """Stats from running extraction over a session batch."""

    session_id: str
    extracted_count: int
    message_count: int


def format_messages_for_extraction(messages: list[ConversationMessage]) -> str:
    """Format role-tagged messages into a single text input for extraction."""
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


class SessionPipeline:
    """Wraps ExtractionPipeline to consume SessionBatch as the input unit."""

    def __init__(self, extraction: ExtractionPipeline) -> None:
        self._extraction = extraction

    async def process_session_batch(
        self, batch: SessionBatch
    ) -> SessionExtractionStats:
        """Run extraction over the batch and stamp per-message provenance."""
        text = format_messages_for_extraction(batch.messages)
        message_ids = [str(m.message_id) for m in batch.messages]

        results = await self._extraction.process(batch.user_id, text)

        provider = self._extraction.provider
        for r in results:
            if r.memory_id is None:
                continue
            mem = await provider.get_by_id(r.memory_id)
            if mem is None:
                continue
            md = dict(mem.metadata or {})
            md["source_session_id"] = str(batch.session_id)
            md["source_message_ids"] = message_ids
            await provider.update(r.memory_id, metadata=md)

        logger.debug(
            "session_pipeline: %d extraction(s) from %d message(s) (session=%s)",
            len(results),
            len(batch.messages),
            batch.session_id,
        )
        return SessionExtractionStats(
            session_id=str(batch.session_id),
            extracted_count=len(results),
            message_count=len(batch.messages),
        )
