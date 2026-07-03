from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.episode import Episode
from mnemosyne.db.models.history import MemoryHistoryEntry
from mnemosyne.db.models.memory import Memory, MemoryType, ScoredMemory
from mnemosyne.errors import MemoryNotFound
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.fusion import fuse_rrf
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
        self._episodes: dict[uuid.UUID, Episode] = {}

    # ------------------------------------------------------------------
    # MemoryProvider interface
    # ------------------------------------------------------------------

    async def add(self, memory: Memory, actor: str = "agent_tool") -> uuid.UUID:
        """Persist *memory* and return its UUID.

        Raises ``ValueError`` if ``memory.embedding`` is ``None``.
        Returns the existing ``memory_id`` without writing a duplicate if
        a non-invalidated memory with the same ``(user_id, content_hash)``
        already exists. ``actor`` is recorded on the ``create`` history row.
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
            actor=actor,
        ))
        return memory.memory_id

    async def get_by_id(
        self, memory_id: uuid.UUID, user_id: uuid.UUID | None = None
    ) -> Memory | None:
        mem = self._memories.get(memory_id)
        if mem is None:
            return None
        if user_id is not None and mem.user_id != user_id:
            return None
        return mem

    async def get_by_ids(self, ids: list[uuid.UUID]) -> list[Memory]:
        """Batch-fetch active memories for *ids* (parity with Postgres).

        Invalidated and unknown ids are dropped. Order follows first
        appearance in *ids* for determinism.
        """
        if not ids:
            return []
        now = datetime.now(timezone.utc)
        out: list[Memory] = []
        seen: set[uuid.UUID] = set()
        for mid in ids:
            if mid in seen:
                continue
            seen.add(mid)
            mem = self._memories.get(mid)
            if mem is None:
                continue
            if mem.valid_until is not None and mem.valid_until <= now:
                continue
            out.append(mem)
        return out

    async def bump_access(self, ids: list[uuid.UUID]) -> None:
        """Increment access_count and advance last_accessed for *ids*.

        No history row is written. Unknown ids are ignored.
        """
        if not ids:
            return
        now = datetime.now(timezone.utc)
        for mid in set(ids):
            mem = self._memories.get(mid)
            if mem is None:
                continue
            mem.access_count += 1
            mem.last_accessed = max(now, mem.last_accessed)

    async def search(
        self,
        query_embedding: list[float],
        user_id: uuid.UUID,
        limit: int = 10,
        weights: ScoringWeights | None = None,
        include_invalidated: bool = False,
        explain: bool = False,
        source_message_id: uuid.UUID | None = None,
        query_text: str | None = None,
        *,
        bump_access: bool = True,
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

        When ``explain=True`` every returned :class:`ScoredMemory` carries a
        populated ``score_breakdown_explain`` :class:`ScoreBreakdown`; the
        legacy ``score_breakdown`` dict remains empty in that mode.

        When ``query_text`` is supplied a case-insensitive token-overlap list is
        built as the full-text parity leg and fused with the vector-scored list
        via Reciprocal Rank Fusion. ``bump_access=False`` suppresses the
        access-bookkeeping side effect.
        """
        now = datetime.now(timezone.utc)

        smid = str(source_message_id) if source_message_id is not None else None

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
            if smid is not None:
                ids = (m.metadata or {}).get("source_message_ids") or []
                if smid not in ids:
                    continue
            candidates.append(m)

        if not candidates:
            return []

        # Score — multi-signal: relevance, recency, importance, frequency
        scorer = MultiSignalScorer(weights or ScoringWeights())
        query_norm = math.sqrt(sum(x * x for x in query_embedding))
        scored: list[ScoredMemory] = []
        for m in candidates:
            if explain:
                total, bd = scorer.score(
                    m, query_embedding, now, explain=True, query_norm=query_norm
                )
                scored.append(
                    ScoredMemory(memory=m, score=total, score_breakdown_explain=bd)
                )
            else:
                total, breakdown = scorer.score(
                    m, query_embedding, now, query_norm=query_norm
                )
                scored.append(
                    ScoredMemory(memory=m, score=total, score_breakdown=breakdown)
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

        # Fusion runs in explain mode too, so explain reflects the real ranking.
        if query_text:
            fts_ranked = _fts_rank(candidates, query_text)
            if fts_ranked:
                fused = fuse_rrf(scored, fts_ranked, max(limit, len(scored)))
                result = fused[:limit]
            else:
                result = scored[:limit]
        else:
            result = scored[:limit]

        # Side-effect: bump access bookkeeping AFTER scoring and slicing
        if bump_access:
            await self.bump_access([sm.memory.memory_id for sm in result])
            for sm in result:
                # Keep the ScoredMemory's reference consistent with the store
                sm.memory = self._memories[sm.memory.memory_id]

        return result

    async def invalidate(
        self,
        memory_id: uuid.UUID,
        reason: str,
        user_id: uuid.UUID | None = None,
        actor: str = "agent_tool",
    ) -> Memory:
        """Soft-delete *memory_id* by setting valid_until = now(UTC).

        Returns the invalidated memory. When *user_id* is supplied a memory
        owned by another user raises ``MemoryNotFound``.

        Raises ``MemoryNotFound`` if the id is unknown.
        Records *reason* in ``metadata['invalidation_reason']`` and stamps
        ``actor`` on the ``invalidate`` history row.
        """
        mem = self._memories.get(memory_id)
        if mem is None or (user_id is not None and mem.user_id != user_id):
            raise MemoryNotFound(memory_id)
        mem.valid_until = datetime.now(timezone.utc)
        mem.metadata["invalidation_reason"] = reason
        self._history.append(MemoryHistoryEntry(
            memory_id=memory_id,
            operation="invalidate",
            old_content=mem.content,
            actor=actor,
            actor_details={"reason": reason},
        ))
        return mem

    async def add_episode(self, episode: Episode) -> Episode:
        """Persist *episode*, upserting on ``session_id``."""
        for existing_id, existing in list(self._episodes.items()):
            if (
                existing.user_id == episode.user_id
                and existing.session_id == episode.session_id
            ):
                stored = episode.model_copy(
                    update={"episode_id": existing.episode_id}
                )
                self._episodes[existing_id] = stored
                return stored
        self._episodes[episode.episode_id] = episode
        return episode

    async def list_episodes(
        self, user_id: uuid.UUID, limit: int = 20, offset: int = 0
    ) -> list[Episode]:
        """Return episodes for *user_id* ordered by created_at DESC."""
        out = [e for e in self._episodes.values() if e.user_id == user_id]
        out.sort(key=lambda e: e.created_at, reverse=True)
        return out[offset : offset + limit]

    async def update(
        self, memory_id: uuid.UUID, *, actor: str = "agent_tool", **fields
    ) -> Memory:
        """Update mutable fields on *memory_id* and return the updated memory.

        When ``content`` changes the ``content_hash`` is recomputed to match, so
        dedup stays consistent. ``actor`` is recorded on the history row.

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
        if "content" in fields:
            mem.content_hash = content_hash(mem.content)
        mem.updated_at = datetime.now(timezone.utc)
        self._history.append(MemoryHistoryEntry(
            memory_id=memory_id,
            operation="update",
            old_content=old_content,
            new_content=mem.content,
            old_importance=old_importance,
            new_importance=mem.importance,
            actor=actor,
        ))
        return mem

    async def get_history(self, memory_id: uuid.UUID) -> list[MemoryHistoryEntry]:
        """Return mutation history for *memory_id*, newest first."""
        entries = [h for h in self._history if h.memory_id == memory_id]
        entries.sort(key=lambda h: h.occurred_at, reverse=True)
        return entries

    async def list_for_user(
        self,
        user_id: uuid.UUID,
        include_invalidated: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Return memories for *user_id*, newest first.

        ``limit=None`` returns every row; supply ``limit``/``offset`` to page.
        """
        out = [
            m
            for m in self._memories.values()
            if m.user_id == user_id and (include_invalidated or m.valid_until is None)
        ]
        out.sort(key=lambda m: (m.created_at, m.memory_id), reverse=True)
        if limit is None:
            return out[offset:]
        return out[offset : offset + limit]

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
        # The rows are retained (audit trail) but scrubbed of personal data:
        # content/hash/embedding are cleared so no erased content is retained.
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
            mem.content = ""
            mem.content_hash = None
            mem.embedding = None
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


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric token set used by the FTS parity leg."""
    out: set[str] = set()
    token: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            token.append(ch)
        elif token:
            out.add("".join(token))
            token = []
    if token:
        out.add("".join(token))
    return out


def _fts_rank(candidates: list[Memory], query_text: str) -> list[ScoredMemory]:
    """Rank candidates by case-insensitive token overlap with *query_text*.

    Mirrors the Postgres full-text leg: only memories sharing at least one
    token rank, ordered by overlap count (newest first to break ties).
    """
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []
    matches: list[tuple[int, Memory]] = []
    for m in candidates:
        overlap = len(query_tokens & _tokenize(m.content))
        if overlap > 0:
            matches.append((overlap, m))
    matches.sort(key=lambda t: (-t[0], -t[1].created_at.timestamp(), str(t[1].memory_id)))
    return [
        ScoredMemory(memory=m, score=0.0, score_breakdown={"fts_overlap": float(o)})
        for o, m in matches
    ]


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
