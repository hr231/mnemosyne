from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mnemosyne.config.settings import Settings
from mnemosyne.db.models.memory import ExtractionResult, Memory
from mnemosyne.db.models.processing import ProcessingLog
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.pipeline.consolidation import run_dedup
from mnemosyne.pipeline.decay import apply_decay
from mnemosyne.pipeline.embedding import embed_pending_memories
from mnemosyne.pipeline.episodes import create_episode
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.rules.base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


@dataclass
class SessionProcessingResult:
    """Summary of what the pipeline did for a single session."""

    session_id: uuid.UUID
    memories_created: int = 0
    embedded: int = 0
    episode_created: bool = False
    deduped: int = 0
    decay_stats: dict = field(default_factory=dict)
    used_llm: bool = False
    error: str | None = None


async def process_session(
    session_id: uuid.UUID,
    user_id: uuid.UUID,
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    settings: Settings,
    text: str | None = None,
    extraction_results: list[ExtractionResult] | None = None,
    extractors: list[BaseExtractor] | None = None,
    llm_client: LLMClient | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
) -> SessionProcessingResult:
    """Process a single session through all pipeline stages.

    Stages run in order:

    1. **Extraction** — extract memories from *text* if provided. If
       *extraction_results* are provided directly, persist them without
       re-extracting.  If neither is given, the embedding and episode
       stages still run (idempotent).

    2. **Embedding** — embed any memories that are missing embeddings.

    3. **Episodes** — create an episode record for the session.

    4. **Consolidation** — three-tier dedup for the user.

    5. **Decay** — apply importance decay for the user.

    The function is idempotent: re-running it for the same session is safe.
    Extraction is skipped if *extraction_results* are already provided.

    Returns a ``SessionProcessingResult`` with per-stage counts.
    """
    result = SessionProcessingResult(session_id=session_id)
    memory_ids: list[uuid.UUID] = []

    try:
        # ----------------------------------------------------------------
        # Stage 1: Extraction
        # ----------------------------------------------------------------
        if extraction_results is not None:
            # Caller pre-extracted (e.g. hot-path at session close)
            for er in extraction_results:
                embedding = await embedder.embed(er.content)
                memory = Memory(
                    user_id=user_id,
                    content=er.content,
                    memory_type=er.memory_type,
                    importance=er.importance,
                    embedding=embedding,
                    extraction_version=er.extraction_version,
                    rule_id=er.rule_id,
                    source_session_id=session_id,
                    metadata=er.metadata,
                )
                mid = await provider.add(memory)
                memory_ids.append(mid)
            result.memories_created = len(memory_ids)

        elif text is not None:
            # Run the full extraction pipeline
            pipeline = ExtractionPipeline(
                settings=settings,
                provider=provider,
                embedder=embedder,
                extractors=extractors,
                llm_client=llm_client,
            )
            extracted = await pipeline.process(user_id=user_id, text=text)
            result.used_llm = llm_client is not None
            for er in extracted:
                if er.memory_id is not None:
                    memory_ids.append(er.memory_id)
            result.memories_created = len(memory_ids)

        # ----------------------------------------------------------------
        # Stage 2: Embedding (catch any memories without embeddings)
        # ----------------------------------------------------------------
        result.embedded = await embed_pending_memories(provider, embedder)

        # ----------------------------------------------------------------
        # Stage 3: Episode creation
        # ----------------------------------------------------------------
        episode = await create_episode(
            provider=provider,
            session_id=session_id,
            user_id=user_id,
            memory_ids=memory_ids,
            llm_client=llm_client,
            embedder=embedder,
            started_at=started_at or datetime.now(timezone.utc),
            ended_at=ended_at,
        )
        result.episode_created = True
        logger.debug(
            "process_session: created episode %s for session %s",
            episode.episode_id,
            session_id,
        )

        # ----------------------------------------------------------------
        # Stage 4: Consolidation (dedup)
        # ----------------------------------------------------------------
        result.deduped = await run_dedup(provider, user_id)

        # ----------------------------------------------------------------
        # Stage 5: Decay
        # ----------------------------------------------------------------
        result.decay_stats = await apply_decay(provider, user_id)

    except Exception as exc:
        result.error = str(exc)
        logger.error(
            "process_session failed at session %s: %s",
            session_id,
            exc,
            exc_info=True,
        )
        raise

    logger.info(
        "process_session complete: session=%s memories=%d embedded=%d deduped=%d decay=%s",
        session_id,
        result.memories_created,
        result.embedded,
        result.deduped,
        result.decay_stats,
    )
    return result


async def process_pending(
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    settings: Settings,
    llm_client: LLMClient | None = None,
) -> list[SessionProcessingResult]:
    """Process all sessions recorded in the processing_log with status='pending'.

    This is the catch-all background runner called on a periodic schedule.
    For each pending record it calls ``process_session``, then marks the
    processing_log entry as completed (or failed on error).

    For ``InMemoryProvider`` there is no persistent ``processing_log`` table.
    In that case the function is a no-op and returns an empty list (the
    hot-path ``process_session`` calls handle in-memory work directly).

    Returns a list of ``SessionProcessingResult`` objects, one per session
    processed.
    """
    if not hasattr(provider, "_pool"):
        # InMemoryProvider — no persistent log to poll
        logger.debug("process_pending: InMemoryProvider has no processing_log — skipping")
        return []

    pool = provider._pool  # type: ignore[attr-defined]
    results: list[SessionProcessingResult] = []

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, user_id
            FROM memory.processing_log
            WHERE status = 'pending'
            ORDER BY processed_at
            FOR UPDATE SKIP LOCKED
            """,
        )

    for row in rows:
        log_id = row["id"]
        session_id = row["session_id"]
        user_id = row["user_id"]

        # Mark as processing
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory.processing_log SET status = 'processing' WHERE id = $1",
                log_id,
            )

        try:
            result = await process_session(
                session_id=session_id,
                user_id=user_id,
                provider=provider,
                embedder=embedder,
                settings=settings,
                llm_client=llm_client,
            )
            # Mark as completed
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE memory.processing_log SET status = 'completed', processed_at = now() WHERE id = $1",
                    log_id,
                )
            results.append(result)
        except Exception as exc:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE memory.processing_log
                    SET status = 'failed', error_message = $1, processed_at = now()
                    WHERE id = $2
                    """,
                    str(exc),
                    log_id,
                )
            logger.error(
                "process_pending: session %s failed: %s", session_id, exc, exc_info=True
            )

    return results
