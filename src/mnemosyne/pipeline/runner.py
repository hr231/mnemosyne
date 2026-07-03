from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from mnemosyne.config.settings import Settings
from mnemosyne.db.models.memory import ExtractionResult, Memory
from mnemosyne.db.repositories import processing_log
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.pipeline.consolidation import run_dedup
from mnemosyne.pipeline.contradiction import run_contradiction_check
from mnemosyne.pipeline.decay import apply_decay
from mnemosyne.pipeline.embedding import embed_pending_memories
from mnemosyne.pipeline.episodes import create_episode
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.pipeline.extraction.reextraction_driver import ReextractionDriver
from mnemosyne.pipeline.reflection import maybe_run_reflection
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.rules.base_extractor import BaseExtractor
from mnemosyne.rules.stub import StubRegexExtractor

logger = logging.getLogger(__name__)


@dataclass
class SessionProcessingResult:
    """Summary of what the pipeline did for a single session."""

    session_id: uuid.UUID
    memories_created: int = 0
    embedded: int = 0
    episode_created: bool = False
    contradictions_resolved: int = 0
    used_llm: bool = False
    error: str | None = None


async def _persist_extraction_results(
    extraction_results: list[ExtractionResult],
    *,
    user_id: uuid.UUID,
    session_id: uuid.UUID,
    provider: MemoryProvider,
    embedder: EmbeddingClient,
) -> list[uuid.UUID]:
    memory_ids: list[uuid.UUID] = []
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
    return memory_ids


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
    audit_store: object | None = None,
    metrics: object | None = None,
) -> SessionProcessingResult:
    """Process a single session through the per-session pipeline stages.

    Stages run in order:

    1. **Extraction** — extract memories from *text* (or persist
       *extraction_results* directly). When neither is given, only the
       remaining stages run.
    2. **Embedding** — embed any of this user's memories missing embeddings.
    3. **Episode** — persist an episode for the session via
       ``provider.add_episode`` (upsert on session_id).
    4. **Contradiction check** — when an ``llm_client`` is available, check the
       memories created in this session against existing ones, bounded by
       ``created_after``.

    Per-session dedup, decay, and reflection are intentionally NOT run here —
    those are maintenance-loop concerns.

    The function is idempotent: re-running it for the same session is safe.
    """
    result = SessionProcessingResult(session_id=session_id)
    started = started_at or datetime.now(timezone.utc)
    memory_ids: list[uuid.UUID] = []
    extract_start = time.monotonic()
    extraction_ran = extraction_results is not None or text is not None

    try:
        # Stage 1: Extraction
        try:
            if extraction_results is not None:
                memory_ids = await _persist_extraction_results(
                    extraction_results,
                    user_id=user_id,
                    session_id=session_id,
                    provider=provider,
                    embedder=embedder,
                )
                result.memories_created = len(memory_ids)
            elif text is not None:
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
            _record_extraction(
                metrics, result.memories_created, extract_start, success=True
            )
        except Exception:
            _record_extraction(
                metrics, result.memories_created, extract_start, success=False
            )
            raise

        # Stage 2: Embedding catch-up for NULL embeddings. Extraction embeds
        # inline, so the user-wide scan is only needed when this invocation did
        # not run extraction (e.g. a maintenance-style re-run).
        if not extraction_ran:
            try:
                result.embedded = await embed_pending_memories(
                    provider,
                    embedder,
                    user_id=user_id,
                    expected_dim=settings.embedding_dim,
                )
                _metrics_call(metrics, "record_stage", "embedding", True)
            except Exception:
                _metrics_call(metrics, "record_stage", "embedding", False)
                raise

        # Stage 3: Episode persistence (upsert on session_id)
        try:
            episode = await create_episode(
                provider=provider,
                session_id=session_id,
                user_id=user_id,
                memory_ids=memory_ids,
                llm_client=llm_client,
                embedder=embedder,
                started_at=started,
                ended_at=ended_at,
            )
            await provider.add_episode(episode)
            result.episode_created = True
            _metrics_call(metrics, "record_stage", "episode", True)
        except Exception:
            _metrics_call(metrics, "record_stage", "episode", False)
            raise

        # Stage 4: Contradiction check (LLM-gated, bounded to this session)
        if llm_client is not None and memory_ids:
            try:
                result.contradictions_resolved = await run_contradiction_check(
                    provider=provider,
                    user_id=user_id,
                    llm_client=llm_client,
                    embedder=embedder,
                    created_after=started,
                    audit_store=audit_store,
                )
                _metrics_call(metrics, "record_stage", "contradiction", True)
            except Exception as exc:  # noqa: BLE001
                _metrics_call(metrics, "record_stage", "contradiction", False)
                logger.warning(
                    "contradiction check failed for session %s: %s",
                    session_id,
                    exc,
                )

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
        "process_session complete: session=%s memories=%d embedded=%d contradictions=%d",
        session_id,
        result.memories_created,
        result.embedded,
        result.contradictions_resolved,
    )
    return result


def _metrics_call(metrics: object | None, name: str, *args: object) -> None:
    """Invoke a duck-typed metrics hook if present, swallowing any failure."""
    if metrics is None:
        return
    fn = getattr(metrics, name, None)
    if not callable(fn):
        return
    try:
        fn(*args)
    except Exception:  # noqa: BLE001
        logger.debug("metrics.%s failed", name, exc_info=True)


def _record_extraction(
    metrics: object | None, count: int, start: float, *, success: bool = True
) -> None:
    if metrics is None:
        return
    fn = getattr(metrics, "record_extraction", None)
    if not callable(fn):
        return
    latency_ms = (time.monotonic() - start) * 1000.0
    try:
        fn(success=success, latency_ms=latency_ms)
    except Exception:  # noqa: BLE001
        logger.debug("metrics.record_extraction failed", exc_info=True)


async def _extraction_enabled(pool: object, user_id: uuid.UUID) -> bool:
    """Return whether extraction is enabled for *user_id* (default enabled).

    Defense-in-depth mirror of the integration-layer opt-out gate. A missing
    row, NULL column, or lookup error all resolve to enabled so the worker
    never silently drops extraction on a transient read failure.
    """
    from mnemosyne.db.repositories.user_settings import get_extraction_enabled

    try:
        value = await get_extraction_enabled(pool, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "extraction_enabled lookup failed for user %s: %s — defaulting enabled",
            user_id,
            exc,
        )
        return True
    if value is None:
        return True
    return bool(value)


async def _oldest_pending_age_s(pool: object) -> float:
    """Return the age in seconds of the oldest pending processing_log row (0 if none)."""
    from mnemosyne.db.repositories.processing_log import oldest_pending_age_seconds

    return await oldest_pending_age_seconds(pool)


def _provider_pool(provider: MemoryProvider) -> object | None:
    return getattr(provider, "_pool", None)


def _default_session_loader(pool: object) -> object | None:
    """Build a Postgres session loader exposing peek/delete_session.

    Returns ``None`` when the integration-owned assembler does not yet expose
    the cold-path methods this worker depends on.
    """
    try:
        from mnemosyne.integration.session_assembler import PersistentSessionAssembler
    except Exception:  # noqa: BLE001
        return None
    loader = PersistentSessionAssembler(pool)
    if hasattr(loader, "peek") and hasattr(loader, "delete_session"):
        return loader
    return None


async def process_pending(
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    settings: Settings,
    llm_client: LLMClient | None = None,
    batch: int = 10,
    session_loader: object | None = None,
    audit_store: object | None = None,
    metrics: object | None = None,
) -> list[SessionProcessingResult]:
    """Process pending sessions claimed from the processing_log.

    Claims up to *batch* pending rows atomically, loads each session's
    buffered messages, runs ``process_session``, and marks the row completed
    or failed. Session-buffer rows are deleted only after the session
    completes successfully.

    For providers without a persistent pool (``InMemoryProvider``) this is a
    no-op returning an empty list.
    """
    pool = _provider_pool(provider)
    if pool is None:
        logger.debug("process_pending: provider has no pool — skipping")
        return []

    rows = await processing_log.claim_pending(pool, batch)
    results: list[SessionProcessingResult] = []

    for row in rows:
        log_id = row["id"]
        session_id = row["session_id"]
        user_id = row["user_id"]

        try:
            # Defense-in-depth: honour the per-user extraction opt-out even
            # though the hook boundary already gates it. A disabled user's row
            # completes as a skip without extraction.
            if user_id is not None and not await _extraction_enabled(pool, user_id):
                logger.info(
                    "process_pending: extraction disabled for user %s — "
                    "skipping session %s",
                    user_id,
                    session_id,
                )
                await processing_log.mark_completed(pool, log_id)
                if session_loader is not None:
                    await session_loader.delete_session(session_id)
                continue

            text: str | None = None
            if session_loader is not None and user_id is not None:
                batch_obj = await session_loader.peek(session_id)
                if batch_obj is None:
                    # A claimed row whose buffered session has vanished must not
                    # be silently completed with an empty episode — fail it so
                    # the loss is visible and it can be requeued.
                    logger.warning(
                        "process_pending: no buffered session for claimed row %s "
                        "(session %s) — marking failed",
                        log_id,
                        session_id,
                    )
                    await processing_log.mark_failed(
                        pool, log_id, error="session buffer missing"
                    )
                    _metrics_call(metrics, "record_processing_failed")
                    continue
                from mnemosyne.pipeline.session_pipeline import (
                    format_messages_for_extraction,
                )

                text = format_messages_for_extraction(batch_obj.messages)

            result = await process_session(
                session_id=session_id,
                user_id=user_id,
                provider=provider,
                embedder=embedder,
                settings=settings,
                text=text,
                llm_client=llm_client,
                audit_store=audit_store,
                metrics=metrics,
            )
            await processing_log.mark_completed(pool, log_id)
            if session_loader is not None:
                await session_loader.delete_session(session_id)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            # Keep the session buffer intact on failure so a requeue can retry.
            await processing_log.mark_failed(pool, log_id, error=str(exc))
            _metrics_call(metrics, "record_processing_failed")
            logger.error(
                "process_pending: session %s failed: %s",
                session_id,
                exc,
                exc_info=True,
            )

    return results


class PipelineWorker:
    """Long-running background worker for durable session processing.

    Three cooperating tasks are spawned by :meth:`start`:

    - **drain**: pulls items off the in-process ``HookQueue`` and persists a
      durable pending row via ``processing_log.insert_pending``.
    - **poll**: every ``worker_poll_interval_s`` claims and processes pending
      sessions via :func:`process_pending`.
    - **maintenance**: every ``maintenance_interval_s`` requeues stale rows and
      runs dedup, decay, and reflection per active user.

    :meth:`stop` cancels the loops, drains any remaining queued items, and
    waits for in-flight work up to a timeout.
    """

    def __init__(
        self,
        provider: MemoryProvider,
        embedder: EmbeddingClient,
        settings: Settings,
        hook_queue: object | None = None,
        llm_client: LLMClient | None = None,
        pool: object | None = None,
        session_loader: object | None = None,
        audit_store: object | None = None,
        metrics: object | None = None,
        active_user_provider: object | None = None,
        requeue_max_retries: int = 3,
    ) -> None:
        self._provider = provider
        self._embedder = embedder
        self._settings = settings
        self._queue = hook_queue
        self._llm_client = llm_client
        self._pool = pool if pool is not None else _provider_pool(provider)
        self._session_loader = session_loader
        if self._session_loader is None and self._pool is not None:
            self._session_loader = _default_session_loader(self._pool)
        self._audit_store = audit_store
        self._metrics = metrics
        self._active_user_provider = active_user_provider
        self._requeue_max_retries = requeue_max_retries

        self._tasks: list[asyncio.Task] = []
        self._stopping = asyncio.Event()
        self._processing_lock = asyncio.Lock()
        self._warned_no_active_users = False

    async def start(self) -> None:
        """Spawn the background loops. Safe to call once per worker."""
        if self._tasks:
            return
        self._stopping.clear()
        if self._queue is not None:
            self._tasks.append(asyncio.create_task(self._drain_loop()))
        self._tasks.append(asyncio.create_task(self._poll_loop()))
        self._tasks.append(asyncio.create_task(self._maintenance_loop()))

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop all loops, drain remaining queued items, finish in-flight work.

        Loops are asked to exit cooperatively first: setting ``_stopping`` wakes
        their interruptible sleeps so a running poll/maintenance tick finishes
        (and its in-flight row reaches mark_completed/mark_failed) before the
        loop returns. Only tasks that do not settle within *timeout* — typically
        the drain loop blocked on ``dequeue`` — are cancelled.
        """
        if not self._tasks:
            return
        self._stopping.set()

        done, pending = await asyncio.wait(self._tasks, timeout=timeout)
        if pending:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            logger.warning(
                "PipelineWorker.stop cancelled %d task(s) that did not drain "
                "within %.1fs",
                len(pending),
                timeout,
            )
        self._tasks = []

        # Drain any items still buffered in the queue so nothing is lost.
        if self._queue is not None and self._pool is not None:
            await self._drain_remaining()

        # Let in-flight processing settle.
        async with self._processing_lock:
            pass

    async def _drain_loop(self) -> None:
        assert self._queue is not None
        while not self._stopping.is_set():
            dequeue_task = asyncio.ensure_future(self._queue.dequeue())
            stop_task = asyncio.ensure_future(self._stopping.wait())
            try:
                done, _pending = await asyncio.wait(
                    {dequeue_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                dequeue_task.cancel()
                stop_task.cancel()
                raise
            if dequeue_task in done:
                # An item won the race — persist it even if a stop also fired,
                # so nothing dequeued is dropped.
                stop_task.cancel()
                await self._persist_item(dequeue_task.result())
            else:
                # Stop requested with no item in hand. Abandon the pending
                # dequeue (cancellation-safe: no item is consumed) — any items
                # still buffered are flushed by _drain_remaining during stop().
                dequeue_task.cancel()
                break

    async def _drain_remaining(self) -> None:
        assert self._queue is not None
        while self._queue.depth() > 0:
            try:
                item = await asyncio.wait_for(self._queue.dequeue(), timeout=0.1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                break
            await self._persist_item(item)

    async def _persist_item(self, item: object) -> None:
        if self._pool is None:
            return
        try:
            await processing_log.insert_pending(
                self._pool, item.session_id, item.user_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "failed to persist hook item %s: %s",
                getattr(item, "session_id", "?"),
                exc,
            )
            if hasattr(self._queue, "requeue_after_failure"):
                await self._queue.requeue_after_failure(item, str(exc))

    async def _interruptible_sleep(self, interval: float) -> None:
        """Sleep for *interval* seconds, returning early when a stop is signalled."""
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

    async def _poll_loop(self) -> None:
        interval = self._settings.worker_poll_interval_s
        while not self._stopping.is_set():
            try:
                await self.run_poll_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("poll tick failed: %s", exc, exc_info=True)
            if self._stopping.is_set():
                break
            await self._interruptible_sleep(interval)

    async def run_poll_once(self) -> list[SessionProcessingResult]:
        async with self._processing_lock:
            results = await process_pending(
                provider=self._provider,
                embedder=self._embedder,
                settings=self._settings,
                llm_client=self._llm_client,
                session_loader=self._session_loader,
                audit_store=self._audit_store,
                metrics=self._metrics,
            )
        await self._publish_poll_metrics()
        return results

    async def _maintenance_loop(self) -> None:
        interval = self._settings.maintenance_interval_s
        while not self._stopping.is_set():
            try:
                await self.run_maintenance_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("maintenance tick failed: %s", exc, exc_info=True)
            if self._stopping.is_set():
                break
            await self._interruptible_sleep(interval)

    async def run_maintenance_once(
        self, active_user_ids: list[uuid.UUID] | None = None
    ) -> None:
        _metrics_call(self._metrics, "set_last_maintenance_timestamp")

        if self._pool is not None:
            # Requeue rows abandoned by a dead worker. Hold the processing lock
            # so a stale requeue cannot race a poll tick that is mid-flight on
            # the same row.
            async with self._processing_lock:
                try:
                    requeued = await processing_log.requeue_stale(
                        self._pool, self._settings.processing_visibility_timeout_s
                    )
                    if requeued:
                        logger.info("maintenance: requeued %d stale rows", requeued)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("requeue_stale failed: %s", exc)

            # Return retryable failed rows to the queue so a transient error is
            # not terminal, then publish the remaining failed backlog.
            try:
                requeued_failed = await processing_log.requeue_failed(
                    self._pool, self._requeue_max_retries
                )
                if requeued_failed:
                    logger.info(
                        "maintenance: requeued %d failed rows", requeued_failed
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("requeue_failed failed: %s", exc)

            try:
                backlog = await processing_log.count_failed(self._pool)
                _metrics_call(
                    self._metrics, "set_processing_failed_backlog", backlog
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("count_failed failed: %s", exc)

        users = active_user_ids
        if users is None:
            users = await self._resolve_active_users()

        for uid in users:
            try:
                merged = await run_dedup(self._provider, uid)
                for _ in range(merged or 0):
                    _metrics_call(self._metrics, "record_dedup_merge")
            except Exception as exc:  # noqa: BLE001
                logger.warning("dedup failed for user %s: %s", uid, exc)

            try:
                await apply_decay(self._provider, uid, metrics=self._metrics)
            except Exception as exc:  # noqa: BLE001
                logger.warning("decay failed for user %s: %s", uid, exc)

            if self._llm_client is not None:
                try:
                    await maybe_run_reflection(
                        self._provider,
                        self._llm_client,
                        self._embedder,
                        uid,
                        pool=self._pool,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("reflection failed for user %s: %s", uid, exc)

    async def _resolve_active_users(self) -> list[uuid.UUID]:
        """Resolve the users maintenance should sweep.

        An explicit ``active_user_provider`` always wins. Otherwise, when a pool
        is available, every user with at least one active memory is swept. With
        neither, maintenance has nothing to act on — warn once so the silent
        no-op is visible.
        """
        if self._active_user_provider is not None:
            return await self._active_user_provider()

        if self._pool is not None:
            from mnemosyne.db.repositories.memories import active_user_ids

            try:
                return await active_user_ids(self._pool)
            except Exception as exc:  # noqa: BLE001
                logger.warning("active-user resolution failed: %s", exc)
                return []

        if not self._warned_no_active_users:
            logger.warning(
                "maintenance has no active_user_provider and no pool — "
                "dedup, decay, and reflection will not run"
            )
            self._warned_no_active_users = True
        return []

    async def _publish_poll_metrics(self) -> None:
        """Publish liveness, queue depth, and real pipeline lag after a poll."""
        _metrics_call(self._metrics, "set_last_poll_timestamp")
        if self._pool is None or self._metrics is None:
            return
        try:
            depth = await processing_log.count_pending(self._pool)
            _metrics_call(self._metrics, "set_queue_depth", depth)
            lag = await _oldest_pending_age_s(self._pool)
            _metrics_call(self._metrics, "record_pipeline_lag_seconds", lag)
        except Exception:  # noqa: BLE001
            logger.debug("poll metric update failed", exc_info=True)


@dataclass
class PipelineStats:
    """Aggregate counters for a single ``PipelineRunner.run_once`` invocation."""

    reextraction_processed: int = 0
    reextraction_changed: int = 0
    reextraction_kept: int = 0
    reextraction_superseded: int = 0
    reextraction_new: int = 0
    reflections_created: int = 0


class PipelineRunner:
    """Opt-in periodic runner for background pipeline work.

    The runner is a thin orchestration layer on top of ``ReextractionDriver``
    and :func:`maybe_run_reflection`. A single tick processes re-extraction for
    a given list of users when ``reextraction_target_version`` is configured.
    """

    def __init__(
        self,
        provider: MemoryProvider,
        embedder: EmbeddingClient,
        llm_client: LLMClient | None = None,
        settings: Settings | None = None,
        extractors: list[BaseExtractor] | None = None,
        reextraction_target_version: str | None = None,
    ) -> None:
        self._provider = provider
        self._embedder = embedder
        self._llm_client = llm_client
        self._settings = settings
        self._reextraction_target = reextraction_target_version

        self._reextraction_driver: ReextractionDriver | None = None
        if reextraction_target_version:
            if settings is None:
                raise ValueError(
                    "settings required when reextraction_target_version is set"
                )
            extractor_list = extractors or [
                StubRegexExtractor(extraction_version=settings.extraction_version)
            ]
            pipeline = ExtractionPipeline(
                settings=settings,
                provider=provider,
                embedder=embedder,
                extractors=extractor_list,
                llm_client=llm_client,
            )
            pg_pool = getattr(provider, "_pool", None)
            self._reextraction_driver = ReextractionDriver(
                provider=provider,
                pipeline=pipeline,
                embedder=embedder,
                pg_pool=pg_pool,
            )

    async def run_once(
        self, user_ids: list[uuid.UUID] | None = None
    ) -> PipelineStats:
        """Execute one runner tick.

        When ``user_ids`` is provided and a re-extraction target is set, the
        runner drives ``ReextractionDriver.reextract_user`` for each user and
        accumulates the totals onto ``PipelineStats``. For every user it also
        calls :func:`maybe_run_reflection` which fires only when the
        importance-sum threshold has been crossed.
        """
        stats = PipelineStats()
        if not user_ids:
            return stats

        if self._reextraction_driver is not None:
            assert self._reextraction_target is not None
            for uid in user_ids:
                r = await self._reextraction_driver.reextract_user(
                    user_id=uid, target_version=self._reextraction_target
                )
                stats.reextraction_processed += r.count_processed
                stats.reextraction_changed += r.count_changed
                stats.reextraction_kept += r.count_kept
                stats.reextraction_superseded += r.count_superseded
                stats.reextraction_new += r.count_new

        if self._llm_client is not None:
            pool = getattr(self._provider, "_pool", None)
            for uid in user_ids:
                try:
                    stats.reflections_created += await maybe_run_reflection(
                        provider=self._provider,
                        llm=self._llm_client,
                        embedder=self._embedder,
                        user_id=uid,
                        pool=pool,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "reflection stage failed for user %s: %s", uid, exc
                    )
        return stats
