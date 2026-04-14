from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    entity_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    agent_id: uuid.UUID = Field(default=uuid.UUID("00000000-0000-0000-0000-000000000000"))
    entity_name: str
    entity_type: str  # "person", "organization", "product", "brand", "location", "concept"
    normalized_name: str = ""
    embedding: list[float] | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    mention_count: int = 0
    source_memory_ids: list[uuid.UUID] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if not self.normalized_name:
            self.normalized_name = self.entity_name.strip().lower()


class EntityMention(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    entity_id: uuid.UUID
    memory_id: uuid.UUID
    mention_text: str = ""
    context: str = ""
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
