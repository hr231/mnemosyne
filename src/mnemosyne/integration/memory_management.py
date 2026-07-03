from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.memory import Memory
from mnemosyne.db.repositories.contradiction_audit import ContradictionAuditStore
from mnemosyne.db.repositories.entity import EntityStore
from mnemosyne.integration.memory_management_models import (
    DeleteMemoryRequest,
    DeleteUserRequest,
    DeleteUserResponse,
    ExportUserResponse,
    GetMemoryRequest,
    ListContradictionsRequest,
    ListContradictionsResponse,
    ListMemoriesRequest,
    ListMemoriesResponse,
    ToggleExtractionRequest,
    ToggleExtractionResponse,
)
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)


class MemoryManagementService:
    """Typed API over MemoryProvider + EntityStore for user-facing memory ops.

    Methods:
        list_memories      — paginated listing with invalidated filter.
        get_memory         — single fetch by id, ``None`` if missing.
        delete_memory      — soft invalidation (sets valid_until).
        delete_user        — GDPR physical delete (audits then cascades).
        export_user        — snapshot of every memory + entity for a user.
        toggle_extraction  — per-user switch for background extraction.
    """

    def __init__(
        self,
        provider: MemoryProvider,
        entity_store: EntityStore,
        audit_store: ContradictionAuditStore | None = None,
        pool=None,
    ) -> None:
        self._provider = provider
        self._entity_store = entity_store
        self._audit_store = audit_store
        self._pool = pool
        self._extraction_disabled: set[uuid.UUID] = set()

    async def list_memories(self, req: ListMemoriesRequest) -> ListMemoriesResponse:
        all_items = await self._provider.list_for_user(
            user_id=req.user_id, include_invalidated=req.include_invalidated
        )
        total = len(all_items)
        page = all_items[req.offset : req.offset + req.limit]
        return ListMemoriesResponse(user_id=req.user_id, total=total, items=page)

    async def get_memory(self, req: GetMemoryRequest) -> Memory | None:
        return await self._provider.get_by_id(req.memory_id, user_id=req.user_id)

    async def delete_memory(self, req: DeleteMemoryRequest) -> None:
        await self._provider.invalidate(
            req.memory_id, reason=f"manual:{req.requestor}", user_id=req.user_id
        )

    async def delete_user(self, req: DeleteUserRequest) -> DeleteUserResponse:
        """Physically delete everything owned by ``req.user_id``.

        The provider writes its audit row (with pre-delete counts) before
        destructive work. When the provider advertises
        ``cascades_entities=True`` (Postgres), it also removes entities and
        entity_mentions in the same transaction and the service must NOT
        call ``entity_store.physical_delete_user`` again. Otherwise the
        service deletes entities via the store.
        """
        entity_count_pre = len(await self._entity_store.list_for_user(req.user_id))
        memory_count = await self._provider.physical_delete_user(
            user_id=req.user_id, requestor=req.requestor, dry_run=req.dry_run
        )

        entities_deleted = 0
        if not req.dry_run:
            if getattr(self._provider, "cascades_entities", False):
                entities_deleted = entity_count_pre
            else:
                entities_deleted = await self._entity_store.physical_delete_user(
                    req.user_id
                )
            self._extraction_disabled.discard(req.user_id)

        breakdown = {"memories": memory_count, "entities": entities_deleted}
        return DeleteUserResponse(
            user_id=req.user_id,
            rows_deleted=memory_count + entities_deleted,
            dry_run=req.dry_run,
            breakdown=breakdown,
        )

    async def export_user(
        self, user_id: uuid.UUID, requestor: str = "system"
    ) -> ExportUserResponse:
        memories = await self._provider.list_for_user(
            user_id=user_id, include_invalidated=True
        )
        entities = await self._entity_store.list_for_user(user_id)
        logger.info(
            "gdpr export for user %s requested by %s (%d memories, %d entities)",
            user_id,
            requestor,
            len(memories),
            len(entities),
        )
        return ExportUserResponse(
            user_id=user_id,
            exported_at=datetime.now(timezone.utc),
            requestor=requestor,
            memory_count=len(memories),
            entity_count=len(entities),
            memories=[m.model_dump(mode="json") for m in memories],
            entities=[e.model_dump(mode="json") for e in entities],
        )

    async def toggle_extraction(
        self, req: ToggleExtractionRequest
    ) -> ToggleExtractionResponse:
        if req.enabled:
            self._extraction_disabled.discard(req.user_id)
        else:
            self._extraction_disabled.add(req.user_id)

        if self._pool is not None:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO memory.user_settings
                        (user_id, extraction_enabled, updated_at)
                    VALUES ($1, $2, now())
                    ON CONFLICT (user_id) DO UPDATE
                    SET extraction_enabled = EXCLUDED.extraction_enabled,
                        updated_at = now()
                    """,
                    req.user_id,
                    req.enabled,
                )
        return ToggleExtractionResponse(user_id=req.user_id, enabled=req.enabled)

    def is_extraction_enabled(self, user_id: uuid.UUID) -> bool:
        """Synchronous in-memory view of the extraction flag.

        Reflects toggles applied in this process. For the durable, source-of-
        truth answer use the async :meth:`extraction_enabled`.
        """
        return user_id not in self._extraction_disabled

    async def extraction_enabled(self, user_id: uuid.UUID) -> bool:
        """Return whether background extraction is enabled for ``user_id``.

        Consults ``memory.user_settings`` when a pool is wired in (a missing
        row defaults to enabled); otherwise falls back to the in-process set.
        """
        if self._pool is not None:
            from mnemosyne.integration.extraction_gate import (
                is_extraction_enabled_for,
            )

            return await is_extraction_enabled_for(self._pool, user_id)
        return self.is_extraction_enabled(user_id)

    async def list_contradictions(
        self, req: ListContradictionsRequest
    ) -> ListContradictionsResponse:
        """Return resolved contradictions for ``req.user_id``, newest first.

        When no audit store was wired in at construction time, returns an
        empty response so callers can introspect the API without crashing.
        """
        if self._audit_store is None:
            return ListContradictionsResponse(
                user_id=req.user_id, total=0, items=[]
            )
        items = await self._audit_store.list_for_user(
            user_id=req.user_id,
            limit=req.limit,
            offset=req.offset,
            since=req.since,
        )
        total = await self._audit_store.count_for_user(
            req.user_id, since=req.since
        )
        return ListContradictionsResponse(
            user_id=req.user_id, total=total, items=items
        )
