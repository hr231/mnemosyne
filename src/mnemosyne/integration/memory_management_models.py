from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from mnemosyne.db.models.memory import Memory


class ListMemoriesRequest(BaseModel):
    """Page a user's memories."""

    user_id: uuid.UUID
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    include_invalidated: bool = False


class ListMemoriesResponse(BaseModel):
    user_id: uuid.UUID
    total: int
    items: list[Memory]


class GetMemoryRequest(BaseModel):
    memory_id: uuid.UUID


class DeleteMemoryRequest(BaseModel):
    """Soft-invalidate a single memory (does not physically delete)."""

    memory_id: uuid.UUID
    requestor: str = Field(..., min_length=1)


class DeleteUserRequest(BaseModel):
    """Physically delete all rows owned by a user (GDPR)."""

    user_id: uuid.UUID
    requestor: str = Field(..., min_length=1)
    dry_run: bool = False


class DeleteUserResponse(BaseModel):
    user_id: uuid.UUID
    rows_deleted: int
    dry_run: bool


class ExportUserResponse(BaseModel):
    """Snapshot of everything the system remembers about a user."""

    user_id: uuid.UUID
    exported_at: datetime
    memory_count: int
    entity_count: int
    memories: list[dict[str, Any]]
    entities: list[dict[str, Any]]


class ToggleExtractionRequest(BaseModel):
    user_id: uuid.UUID
    enabled: bool
    requestor: str = Field(..., min_length=1)


class ToggleExtractionResponse(BaseModel):
    user_id: uuid.UUID
    enabled: bool
