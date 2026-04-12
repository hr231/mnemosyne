from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.memory import Memory, ScoredMemory
from mnemosyne.errors import MemoryNotFound
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.scoring import ScoringWeights

# Fields that must never be overwritten via update()
READ_ONLY_FIELDS = frozenset({"memory_id", "content_hash", "extraction_version"})


def _content_hash(content: str) -> str:
    """Canonical sha256 hash of normalised content (strip + lower)."""
    return hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryProvider(MemoryProvider):
    """In-process memory provider for development and testing.

    Implements all invariants of the Honest InMemoryProvider spec:

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

    Day 1 morning scorer: naive cosine pass only.
    score_breakdown = {"relevance": cosine, "recency": 0, "importance": 0,
                       "frequency": 0}
    Track B replaces the scorer body on Day 1 afternoon; the signature is
    unchanged.
    """

    def __init__(self) -> None:
        self._memories: dict[uuid.UUID, Memory] = {}

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

        ch = _content_hash(memory.content)

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
        2. Score each candidate with cosine similarity (Day 1 naive pass).
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

        # Score — Day 1 morning: cosine only; other signals stub at 0
        scored: list[ScoredMemory] = []
        for m in candidates:
            cosine = _cosine_sim(query_embedding, m.embedding)  # type: ignore[arg-type]
            scored.append(
                ScoredMemory(
                    memory=m,
                    score=cosine,
                    score_breakdown={
                        "relevance": cosine,
                        "recency": 0.0,
                        "importance": 0.0,
                        "frequency": 0.0,
                    },
                )
            )

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

        for k, v in fields.items():
            setattr(mem, k, v)
        mem.updated_at = datetime.now(timezone.utc)
        return mem
