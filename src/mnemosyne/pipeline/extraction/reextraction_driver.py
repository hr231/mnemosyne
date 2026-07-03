"""Re-extraction driver.

Walks every active memory for a user whose ``extraction_version`` is below
the target, runs the current extractor against the memory's own content,
and emits KEEP / SUPERSEDE / NEW decisions. Optionally records progress in
``memory.reextraction_jobs`` when a pool is supplied.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mnemosyne.db.models.memory import ExtractionResult, Memory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.pipeline.extraction.orchestrator import ExtractionPipeline
from mnemosyne.pipeline.extraction.reextraction_models import (
    DecisionKind,
    ReextractionDecision,
    ReextractionResult,
)
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)

_SUPERSEDE_IMPORTANCE_MARGIN = 0.2
_SUPERSEDE_OVERLAP_THRESHOLD = 0.5


def decide_for_pair(old: Memory, new: Memory | ExtractionResult) -> ReextractionDecision:
    """Decide how to reconcile an old memory with a freshly-extracted one.

    Rules:
    - identical (case-insensitive, whitespace-stripped) content → KEEP
    - sufficient token overlap AND the new importance is not much lower → SUPERSEDE
    - otherwise → NEW
    """
    new_content = new.content
    new_importance = getattr(new, "importance", 0.0)

    if old.content.strip().lower() == new_content.strip().lower():
        return ReextractionDecision(
            kind=DecisionKind.KEEP,
            old_memory_id=old.memory_id,
            reason="content identical",
        )

    overlap = _token_overlap(old.content, new_content)
    if (
        overlap > _SUPERSEDE_OVERLAP_THRESHOLD
        and new_importance >= old.importance - _SUPERSEDE_IMPORTANCE_MARGIN
    ):
        return ReextractionDecision(
            kind=DecisionKind.SUPERSEDE,
            old_memory_id=old.memory_id,
            new_content=new_content,
            reason=f"overlap={overlap:.2f} supersede",
        )
    return ReextractionDecision(
        kind=DecisionKind.NEW,
        new_content=new_content,
        reason=f"overlap={overlap:.2f} insufficient for supersede",
    )


def _token_overlap(a: str, b: str) -> float:
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / max(len(aw), len(bw))


@dataclass
class ReextractionDriver:
    """Drives background re-extraction of stale memories for a single user."""

    provider: MemoryProvider
    pipeline: ExtractionPipeline
    embedder: EmbeddingClient
    batch_size: int = 25
    pg_pool: object | None = None
    inter_batch_pause_s: float = 0.0
    _rng: object = field(default=None, repr=False)

    async def reextract_user(
        self,
        user_id: uuid.UUID,
        target_version: str,
        dry_run: bool = False,
    ) -> ReextractionResult:
        """Re-extract every memory for *user_id* below *target_version*.

        The operation is idempotent: once every memory has been advanced
        to *target_version*, subsequent calls are no-ops (``count_processed=0``).

        When *dry_run* is set no writes are performed — no version stamping,
        invalidation, persistence, or job rows — but the decision counts are
        still computed so the caller can preview the effect. Between batches the
        driver pauses for ``inter_batch_pause_s`` seconds (0 disables) to keep a
        large backfill from monopolising the LLM/embedding budget.
        """
        started = datetime.now(timezone.utc)
        job_id = uuid.uuid4()
        if not dry_run:
            await self._insert_job_row(job_id, user_id, target_version, started)
        error: str | None = None
        result: ReextractionResult | None = None

        try:
            stale = await self.provider.select_by_extraction_version_below(
                user_id=user_id, target_version=target_version
            )
            count_kept = count_superseded = count_new = 0

            for batch_start in range(0, len(stale), self.batch_size):
                if batch_start and self.inter_batch_pause_s > 0:
                    await asyncio.sleep(self.inter_batch_pause_s)
                batch = stale[batch_start : batch_start + self.batch_size]
                for old in batch:
                    candidates = await self.pipeline.extract_only(text=old.content)
                    if not candidates:
                        # Nothing new — the old memory still represents the
                        # current best distillation. Advance its version.
                        if not dry_run:
                            await self._stamp_version(old, target_version)
                        count_kept += 1
                        continue

                    new_candidate = candidates[0]
                    new_candidate = new_candidate.model_copy(
                        update={"extraction_version": target_version}
                    )
                    decision = decide_for_pair(old, new_candidate)

                    if decision.kind is DecisionKind.KEEP:
                        if not dry_run:
                            await self._stamp_version(old, target_version)
                        count_kept += 1
                    elif decision.kind is DecisionKind.SUPERSEDE:
                        if not dry_run:
                            await self.provider.invalidate(
                                memory_id=old.memory_id,
                                reason=f"superseded by re-extraction to {target_version}",
                            )
                            await self._persist_new(
                                user_id=user_id,
                                candidate=new_candidate,
                                target_version=target_version,
                            )
                        count_superseded += 1
                    else:
                        if not dry_run:
                            await self._persist_new(
                                user_id=user_id,
                                candidate=new_candidate,
                                target_version=target_version,
                            )
                            # The old memory still stands — advance its version
                            # too so the loop is idempotent.
                            await self._stamp_version(old, target_version)
                        count_new += 1

            finished = datetime.now(timezone.utc)
            result = ReextractionResult(
                user_id=user_id,
                target_version=target_version,
                count_processed=len(stale),
                count_changed=count_superseded + count_new,
                count_superseded=count_superseded,
                count_new=count_new,
                count_kept=count_kept,
                started_at=started,
                finished_at=finished,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            logger.exception(
                "re-extraction failed for user=%s target=%s", user_id, target_version
            )
            raise
        finally:
            if not dry_run:
                finished_at = datetime.now(timezone.utc)
                await self._complete_job_row(
                    job_id=job_id,
                    count_processed=result.count_processed if result else 0,
                    count_changed=result.count_changed if result else 0,
                    error=error,
                    finished=finished_at,
                )

    async def _stamp_version(self, old: Memory, target_version: str) -> None:
        """Advance the extraction_version of *old* to *target_version*.

        Goes around ``provider.update`` because ``extraction_version`` is a
        read-only field on the update contract: stamping the canonical version
        is an internal bookkeeping operation, not a caller-initiated mutation.
        """
        if hasattr(self.provider, "_memories"):
            mem = self.provider._memories.get(old.memory_id)  # type: ignore[attr-defined]
            if mem is not None:
                mem.extraction_version = target_version
                mem.updated_at = datetime.now(timezone.utc)
            return

        pool = getattr(self.provider, "_pool", None)
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE memory.memories
                SET extraction_version = $2, updated_at = now()
                WHERE memory_id = $1
                """,
                old.memory_id,
                target_version,
            )

    async def _persist_new(
        self,
        user_id: uuid.UUID,
        candidate: ExtractionResult,
        target_version: str,
    ) -> uuid.UUID:
        emb = await self.embedder.embed(candidate.content)
        memory = Memory(
            user_id=user_id,
            content=candidate.content,
            memory_type=candidate.memory_type,
            importance=candidate.importance,
            embedding=emb,
            extraction_version=target_version,
            rule_id=candidate.rule_id or "pipeline_reextraction",
            metadata={**candidate.metadata, "reextracted": True},
        )
        return await self.provider.add(memory)

    async def _insert_job_row(
        self,
        job_id: uuid.UUID,
        user_id: uuid.UUID,
        target_version: str,
        started: datetime,
    ) -> None:
        if self.pg_pool is None:
            return
        async with self.pg_pool.acquire() as conn:  # type: ignore[attr-defined]
            await conn.execute(
                """
                INSERT INTO memory.reextraction_jobs
                    (id, user_id, target_version, status, count_processed,
                     count_changed, started_at)
                VALUES ($1, $2, $3, 'running', 0, 0, $4)
                """,
                job_id,
                user_id,
                target_version,
                started,
            )

    async def _complete_job_row(
        self,
        job_id: uuid.UUID,
        count_processed: int,
        count_changed: int,
        error: str | None,
        finished: datetime,
    ) -> None:
        if self.pg_pool is None:
            return
        status = "failed" if error else "completed"
        async with self.pg_pool.acquire() as conn:  # type: ignore[attr-defined]
            await conn.execute(
                """
                UPDATE memory.reextraction_jobs
                SET status=$2, count_processed=$3, count_changed=$4,
                    finished_at=$5, error=$6
                WHERE id=$1
                """,
                job_id,
                status,
                count_processed,
                count_changed,
                finished,
                error,
            )
