from __future__ import annotations

import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.history import MemoryHistoryEntry
from mnemosyne.db.models.memory import Memory, MemoryType, ScoredMemory
from mnemosyne.errors import MemoryNotFound
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.scoring import MultiSignalScorer, ScoringWeights
from mnemosyne.utils import content_hash

# Fields that must never be overwritten via update()
READ_ONLY_FIELDS = frozenset({"memory_id", "content_hash", "extraction_version"})


class InMemoryProvider(MemoryProvider):
    """In-process memory provider for development and testing.

    Implements all provider invariants:

    - Content-hash dedup: same (user_id, content_hash) with valid_until IS
      NULL is a no-op; the existing memory_id is returned.
    - Bi-temporal filter: search drops memories where
      ``valid_until IS NOT NULL AND valid_until <= now()``.
    - invalidate: sets valid_until = now(UTC) and stores the reason in
      ``metadata['invalidation_reason']``; never deletes the row.
    - extraction_version and rule_id are preserved across update calls.
    - access_count++ and last_accessed are applied AFTER scoring, only to
      memories actually returned to the caller.
    - Deterministic tie-breaking: (-score, -created_at timestamp, memory_id)
      so tests that produce equal scores are stable.
    """

    def __init__(self) -> None:
        self._memories: dict[uuid.UUID, Memory] = {}
        self._history: list[MemoryHistoryEntry] = []
        self._gdpr_audit: list[dict] = []

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    async def add(self, memory: Memory) -> uuid.UUID:
        """Persist *memory* and return its UUID.

        Raises ``ValueError`` if ``memory.embedding`` is ``None``.
        Returns the existing ``memory_id`` without writing a duplicate if
        a non-invalidated memory with the same ``(user_id, content_hash)``
        already exists.
        """
        if memory.embedding is None:
            raise ValueError("caller must set embedding before add")

        ch = content_hash(memory.content)

        # Dedup: same (user_id, content_hash) that is still active
        for existing in self._memories.values():
            if (
                existing.user_id == memory.user_id
                and existing.content_hash == ch
                and existing.valid_until is None
            ):
                return existing.memory_id

        # Stamp the canonical content_hash onto the memory
        memory = memory.model_copy(update={"content_hash": ch})
        self._memories[memory.memory_id] = memory
        self._history.append(MemoryHistoryEntry(
            memory_id=memory.memory_id,
            operation="create",
            new_content=memory.content,
            new_importance=memory.importance,
            actor="agent_tool",
        ))
        return memory.memory_id

    async def get_by_id(self, memory_id: uuid.UUID) -> Memory | None:
        return self._memories.get(memory_id)

    async def search(
        self,
        query_embedding: list[float],
        user_id: uuid.UUID,
        limit: int = 10,
        weights: ScoringWeights | None = None,
        include_invalidated: bool = False,
    ) -> list[ScoredMemory]:
        """Return up to *limit* scored memories for *user_id*.

        Scoring order:
        1. Collect candidates (bi-temporal filter applied unless
           include_invalidated=True).
        2. Score each candidate with MultiSignalScorer (relevance, recency,
           importance, frequency).
        3. Sort by (-score, -created_at, memory_id) for determinism.
        4. Slice to limit.
        5. Bump access_count / last_accessed on the returned slice ONLY.
        """
        now = datetime.now(timezone.utc)

        candidates: list[Memory] = []
        for m in self._memories.values():
            if m.user_id != user_id:
                continue
            if m.embedding is None:
                continue
            if not include_invalidated:
                # Drop hard-invalidated memories (valid_until set and in the past)
                if m.valid_until is not None and m.valid_until <= now:
                    continue
            candidates.append(m)

        if not candidates:
            return []

        # Score — multi-signal: relevance, recency, importance, frequency
        scorer = MultiSignalScorer(weights or ScoringWeights())
        scored: list[ScoredMemory] = []
        for m in candidates:
            total, breakdown = scorer.score(m, query_embedding, now)
            scored.append(ScoredMemory(memory=m, score=total, score_breakdown=breakdown))

        # Deterministic sort: highest score first; among ties newest first,
        # then ascending memory_id for final stability.
        scored.sort(
            key=lambda s: (
                -s.score,
                -s.memory.created_at.timestamp(),
                str(s.memory.memory_id),
            )
        )
        result = scored[:limit]

        # Side-effect: bump access bookkeeping AFTER scoring and slicing
        for sm in result:
            mem = self._memories[sm.memory.memory_id]
            mem.access_count += 1
            mem.last_accessed = now
            # Keep the ScoredMemory's reference consistent with the store
            sm.memory = mem

        return result

    async def invalidate(self, memory_id: uuid.UUID, reason: str) -> None:
        """Soft-delete *memory_id* by setting valid_until = now(UTC).

        Raises ``MemoryNotFound`` if the id is unknown.
        Records *reason* in ``metadata['invalidation_reason']``.
        """
        mem = self._memories.get(memory_id)
        if mem is None:
            raise MemoryNotFound(memory_id)
        mem.valid_until = datetime.now(timezone.utc)
        mem.metadata["invalidation_reason"] = reason
        self._history.append(MemoryHistoryEntry(
            memory_id=memory_id,
            operation="invalidate",
            old_content=mem.content,
            actor="agent_tool",
            actor_details={"reason": reason},
        ))

    async def update(self, memory_id: uuid.UUID, **fields) -> Memory:
        """Update mutable fields on *memory_id* and return the updated memory.

        Raises ``MemoryNotFound`` if the id is unknown.
        Raises ``ValueError`` if any read-only field is present in *fields*.
        """
        mem = self._memories.get(memory_id)
        if mem is None:
            raise MemoryNotFound(memory_id)

        bad = READ_ONLY_FIELDS & set(fields.keys())
        if bad:
            raise ValueError(f"Cannot update read-only fields: {bad}")

        old_content = mem.content
        old_importance = mem.importance
        for k, v in fields.items():
            setattr(mem, k, v)
        mem.updated_at = datetime.now(timezone.utc)
        self._history.append(MemoryHistoryEntry(
            memory_id=memory_id,
            operation="update",
            old_content=old_content,
            new_content=mem.content,
            old_importance=old_importance,
            new_importance=mem.importance,
            actor="agent_tool",
        ))
        return mem

    async def get_history(self, memory_id: uuid.UUID) -> list[MemoryHistoryEntry]:
        """Return mutation history for *memory_id*, newest first."""
        entries = [h for h in self._history if h.memory_id == memory_id]
        entries.sort(key=lambda h: h.occurred_at, reverse=True)
        return entries

    async def list_for_user(
        self, user_id: uuid.UUID, include_invalidated: bool = False
    ) -> list[Memory]:
        """Return every memory for *user_id*, newest first."""
        out = [
            m
            for m in self._memories.values()
            if m.user_id == user_id and (include_invalidated or m.valid_until is None)
        ]
        out.sort(key=lambda m: m.created_at, reverse=True)
        return out

    async def physical_delete_user(
        self, user_id: uuid.UUID, requestor: str, dry_run: bool = False
    ) -> int:
        """Physically drop every memory + history row for *user_id*.

        Appends an entry to ``self._gdpr_audit`` BEFORE deletion so the
        audit intent is observable even on dry_run.
        """
        if not requestor:
            raise ValueError("requestor is required")

        to_delete = [mid for mid, m in self._memories.items() if m.user_id == user_id]
        history_count = sum(1 for h in self._history if h.memory_id in set(to_delete))

        self._gdpr_audit.append(
            {
                "id": uuid.uuid4(),
                "user_id": user_id,
                "requestor": requestor,
                "reason": "user_request",
                "rows_memories": len(to_delete),
                "rows_history": history_count,
                "dry_run": dry_run,
                "occurred_at": datetime.now(timezone.utc),
            }
        )
        if dry_run:
            return len(to_delete)

        deleted_set = set(to_delete)

        # Soft-invalidate cross-user reflections that source any deleted memory.
        # The rows are retained (audit trail); only valid_until + reason are stamped.
        now = datetime.now(timezone.utc)
        for mem in self._memories.values():
            if mem.user_id == user_id:
                continue
            if mem.memory_type != MemoryType.REFLECTION:
                continue
            if mem.valid_until is not None:
                continue
            if not any(sid in deleted_set for sid in mem.source_memory_ids):
                continue
            mem.valid_until = now
            mem.metadata["invalidation_reason"] = "gdpr_source_deleted"

        for mid in to_delete:
            del self._memories[mid]
        self._history = [h for h in self._history if h.memory_id not in deleted_set]
        return len(to_delete)

    async def select_by_extraction_version_below(
        self, user_id: uuid.UUID, target_version: str
    ) -> list[Memory]:
        """Return active memories for *user_id* whose extraction_version is
        strictly less than *target_version* (semver tuple order)."""
        target = _version_key(target_version)
        if target is None:
            return []
        out: list[Memory] = []
        for mem in self._memories.values():
            if mem.user_id != user_id:
                continue
            if mem.valid_until is not None:
                continue
            if mem.extraction_version is None:
                continue
            key = _version_key(mem.extraction_version)
            if key is None:
                continue
            if key < target:
                out.append(mem)
        out.sort(key=lambda m: m.created_at)
        return out


def _version_key(v: str) -> tuple[int, int, int] | None:
    """Parse a semver string into a comparable tuple. Returns None on failure."""
    if not v:
        return None
    parts = v.split(".")
    if len(parts) != 3:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
