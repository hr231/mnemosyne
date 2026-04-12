from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

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
    """

    @abstractmethod
    async def add(self, memory: Memory) -> uuid.UUID:
        """Persist a memory and return its UUID.

        The caller is responsible for populating ``memory.embedding``
        before calling this method.  Raises ``ValueError`` if embedding
        is ``None``.
        """
        ...

    @abstractmethod
    async def get_by_id(self, memory_id: uuid.UUID) -> Memory | None:
        """Return the memory or ``None`` if it does not exist."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        user_id: uuid.UUID,
        limit: int = 10,
        weights: ScoringWeights | None = None,
        include_invalidated: bool = False,
    ) -> list[ScoredMemory]:
        """Return up to *limit* scored memories for *user_id*.

        Default behaviour filters out invalidated memories
        (``valid_until IS NOT NULL AND valid_until <= now()``).
        Pass ``include_invalidated=True`` for historical queries.

        Side-effects: increments ``access_count`` and sets
        ``last_accessed`` on every memory in the returned list.
        """
        ...

    @abstractmethod
    async def invalidate(self, memory_id: uuid.UUID, reason: str) -> None:
        """Soft-delete a memory by setting ``valid_until`` to now(UTC).

        Raises ``MemoryNotFound`` if *memory_id* does not exist.
        Never physically removes the row.
        """
        ...

    @abstractmethod
    async def update(self, memory_id: uuid.UUID, **fields) -> Memory:
        """Update mutable fields on an existing memory and return it.

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
