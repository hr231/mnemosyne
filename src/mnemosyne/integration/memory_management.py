from __future__ import annotations

import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.memory import Memory
from mnemosyne.db.repositories.entity import EntityStore
from mnemosyne.integration.memory_management_models import (
    DeleteMemoryRequest,
    DeleteUserRequest,
    DeleteUserResponse,
    ExportUserResponse,
    GetMemoryRequest,
    ListMemoriesRequest,
    ListMemoriesResponse,
    ToggleExtractionRequest,
    ToggleExtractionResponse,
)
from mnemosyne.providers.base import MemoryProvider


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

    def __init__(self, provider: MemoryProvider, entity_store: EntityStore) -> None:
        self._provider = provider
        self._entity_store = entity_store
        self._extraction_disabled: set[uuid.UUID] = set()

    async def list_memories(self, req: ListMemoriesRequest) -> ListMemoriesResponse:
        all_items = await self._provider.list_for_user(
            user_id=req.user_id, include_invalidated=req.include_invalidated
        )
        total = len(all_items)
        page = all_items[req.offset : req.offset + req.limit]
        return ListMemoriesResponse(user_id=req.user_id, total=total, items=page)

    async def get_memory(self, req: GetMemoryRequest) -> Memory | None:
        return await self._provider.get_by_id(req.memory_id)

    async def delete_memory(self, req: DeleteMemoryRequest) -> None:
        await self._provider.invalidate(
            req.memory_id, reason=f"manual:{req.requestor}"
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

        return DeleteUserResponse(
            user_id=req.user_id,
            rows_deleted=memory_count + entities_deleted,
            dry_run=req.dry_run,
        )

    async def export_user(self, user_id: uuid.UUID) -> ExportUserResponse:
        memories = await self._provider.list_for_user(
            user_id=user_id, include_invalidated=True
        )
        entities = await self._entity_store.list_for_user(user_id)
        return ExportUserResponse(
            user_id=user_id,
            exported_at=datetime.now(timezone.utc),
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
        return ToggleExtractionResponse(user_id=req.user_id, enabled=req.enabled)

    def is_extraction_enabled(self, user_id: uuid.UUID) -> bool:
        return user_id not in self._extraction_disabled
