from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from mnemosyne.db.models.episode import Episode
from mnemosyne.db.models.history import MemoryHistoryEntry
from mnemosyne.db.models.memory import Memory, ScoredMemory
from mnemosyne.retrieval.scoring import ScoringWeights


class MemoryProvider(ABC):
    """Abstract base class for all memory provider implementations.

    This is the stable interface contract between the memory system and
    the agent server.  Changes require lead approval.

    Error contracts
    ---------------
    - ``get_by_id(bad_id)``                            → returns ``None``
    - ``invalidate(bad_id, ...)``                      → raises ``MemoryNotFound``
    - ``update(bad_id, ...)``                          → raises ``MemoryNotFound``
    - ``update()`` on read-only fields                 → raises ``ValueError``
    - ``add(memory)`` with ``memory.embedding is None``→ raises ``ValueError``

    There is intentionally no ``delete()`` method.  The bi-temporal model
    retires memories via ``invalidate()`` only, preserving the audit trail.

    Capabilities
    ------------
    ``cascades_entities``: True when ``physical_delete_user`` also removes
    entities + entity_mentions owned by the user in the same transaction.
    Callers that compose a provider with a separate EntityStore should skip
    the entity store's own ``physical_delete_user`` when this is True.
    """

    cascades_entities: bool = False

    @abstractmethod
    async def add(
        self, memory: Memory, actor: str = "pipeline_extraction"
    ) -> uuid.UUID:
        """Persist a memory and return its UUID.

        The caller is responsible for populating ``memory.embedding``
        before calling this method.  Raises ``ValueError`` if embedding
        is ``None``.

        ``actor`` is recorded on the ``create`` history row; it defaults to the
        pipeline writer so existing callers are unaffected. Pass the real actor
        (e.g. ``"agent_tool"``) when the write originates elsewhere.
        """
        ...

    @abstractmethod
    async def get_by_id(
        self, memory_id: uuid.UUID, user_id: uuid.UUID | None = None
    ) -> Memory | None:
        """Return the memory or ``None`` if it does not exist.

        When ``user_id`` is supplied the lookup is scoped: a memory owned by a
        different user is treated as not found (``None``).
        """
        ...

    @abstractmethod
    async def get_by_ids(self, ids: list[uuid.UUID]) -> list[Memory]:
        """Batch-fetch ACTIVE memories for the supplied ids in one round-trip.

        Only memories that are currently valid are returned
        (``valid_until IS NULL OR valid_until > now()``); invalidated rows and
        unknown ids are silently dropped. Order is stable but not tied to the
        input order. Passing an empty list returns ``[]`` without a query.

        This is the batched counterpart to ``get_by_id`` used by retrieval
        expansion paths to avoid an N+1 fetch loop.
        """
        ...

    @abstractmethod
    async def bump_access(self, ids: list[uuid.UUID]) -> None:
        """Record a single access on each of *ids* in one batched write.

        Increments ``access_count`` and advances ``last_accessed`` to
        ``GREATEST(now(), last_accessed)`` for every id. Unknown or invalidated
        ids are no-ops. Writes NO ``memory_history`` row — access bookkeeping is
        deliberately kept off the mutation audit trail and out of ``update()``.
        Passing an empty list is a no-op.
        """
        ...

    @abstractmethod
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

        Default behaviour filters out invalidated memories
        (``valid_until IS NOT NULL AND valid_until <= now()``).
        Pass ``include_invalidated=True`` for historical queries.

        When ``query_text`` is supplied, Postgres also runs a full-text
        candidate query (``websearch_to_tsquery``) and fuses it with the vector
        candidates via Reciprocal Rank Fusion before re-ranking. InMemory
        applies a case-insensitive token match as a parity leg. When omitted,
        retrieval is vector-only.

        ``bump_access`` defaults to ``True``; pass ``False`` to read without the
        access-bookkeeping side effect (used by over-fetch callers that bump a
        smaller returned slice themselves).

        When ``explain=True``, every returned :class:`ScoredMemory` MUST carry
        a populated ``score_breakdown_explain`` field. When ``False`` (the
        default), the field MUST be ``None``. Backends that cannot honour the
        contract should raise ``NotImplementedError``; both first-party
        backends honour it.

        When ``source_message_id`` is set, results are restricted to memories
        whose ``metadata['source_message_ids']`` contains that UUID. Backends
        without provenance metadata may ignore the filter.

        Side-effects: when ``bump_access`` is ``True`` (the default),
        increments ``access_count`` and sets ``last_accessed`` on every memory
        in the returned list. When ``False`` the read is side-effect free.
        """
        ...

    @abstractmethod
    async def invalidate(
        self,
        memory_id: uuid.UUID,
        reason: str,
        user_id: uuid.UUID | None = None,
        actor: str = "pipeline_decay",
    ) -> Memory:
        """Soft-delete a memory by setting ``valid_until`` to now(UTC).

        Returns the invalidated :class:`Memory`. When ``user_id`` is supplied
        the operation is scoped: a memory owned by a different user raises
        ``MemoryNotFound``.

        ``actor`` is recorded on the ``invalidate`` history row; it defaults to
        the decay writer so existing callers are unaffected.

        Raises ``MemoryNotFound`` if *memory_id* does not exist (or is already
        invalidated). Never physically removes the row.
        """
        ...

    @abstractmethod
    async def add_episode(self, episode: Episode) -> Episode:
        """Persist an episode, upserting on ``session_id``.

        A second call with the same ``session_id`` updates the existing row in
        place rather than creating a duplicate. Returns the stored episode.
        """
        ...

    @abstractmethod
    async def list_episodes(
        self, user_id: uuid.UUID, limit: int = 20, offset: int = 0
    ) -> list[Episode]:
        """Return episodes for ``user_id`` ordered by ``created_at`` DESC."""
        ...

    @abstractmethod
    async def update(
        self, memory_id: uuid.UUID, *, actor: str = "manual", **fields
    ) -> Memory:
        """Update mutable fields on an existing memory and return it.

        ``actor`` is recorded on the ``update`` history row and defaults to
        ``"manual"`` so existing callers are unaffected.

        When ``content`` is updated the ``content_hash`` is recomputed so the
        dedup unique index stays consistent; callers may not set
        ``content_hash`` directly (it is read-only).

        Raises ``MemoryNotFound`` if *memory_id* does not exist.
        Raises ``ValueError`` if any of the read-only fields
        (``memory_id``, ``content_hash``, ``extraction_version``) are
        included in *fields*.
        """
        ...

    @abstractmethod
    async def get_history(self, memory_id: uuid.UUID) -> list[MemoryHistoryEntry]:
        """Return the mutation history for a memory, newest first."""
        ...

    @abstractmethod
    async def select_by_extraction_version_below(
        self, user_id: uuid.UUID, target_version: str
    ) -> list[Memory]:
        """Return all memories for ``user_id`` whose ``extraction_version``
        is strictly less than ``target_version`` (semver tuple order).

        Memories with a ``NULL`` or malformed extraction_version are skipped.
        Invalidated memories (``valid_until`` set) are excluded.
        """
        ...

    @abstractmethod
    async def list_for_user(
        self,
        user_id: uuid.UUID,
        include_invalidated: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Memory]:
        """Return memories owned by ``user_id`` ordered by created_at DESC.

        Unlike ``search``, no scoring or query embedding is applied.
        Defaults to excluding rows where ``valid_until`` is set.

        ``limit``/``offset`` page the result; ``limit=None`` (the default)
        returns every matching row, preserving the historical unbounded
        behaviour. Latency-sensitive callers should page.
        """
        ...

    @abstractmethod
    async def physical_delete_user(
        self, user_id: uuid.UUID, requestor: str, dry_run: bool = False
    ) -> int:
        """Physically delete every row owned by ``user_id``.

        Writes an immutable audit row into ``memory.gdpr_deletions`` BEFORE
        performing any destructive delete, inside the same transaction.

        Returns the number of memory rows deleted (or counted, on dry_run).
        Raises ``ValueError`` if ``requestor`` is empty.
        """
        ...
