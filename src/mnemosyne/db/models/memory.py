from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MemoryType(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    ENTITY = "entity"
    PROCEDURAL = "procedural"
    REFLECTION = "reflection"


class Memory(BaseModel):
    memory_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    agent_id: uuid.UUID = Field(default=uuid.UUID("00000000-0000-0000-0000-000000000000"))
    org_id: uuid.UUID | None = None

    memory_type: MemoryType = MemoryType.FACT
    content: str
    content_hash: str | None = None

    embedding: list[float] | None = None

    importance: float = 0.5
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decay_rate: float = 0.01

    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: datetime | None = None

    extraction_version: str = "0.1.0"
    extraction_model: str | None = None
    prompt_hash: str | None = None
    rule_id: str | None = None

    source_session_id: uuid.UUID | None = None
    source_memory_ids: list[uuid.UUID] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("importance")
    @classmethod
    def clamp_importance(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class ScoredMemory(BaseModel):
    memory: Memory
    score: float
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.FACT
    importance: float = 0.5
    matched_chars: int = 0
    rule_id: str = ""
    confidence: float = 1.0
    extraction_version: str = "0.1.0"
    metadata: dict[str, Any] = Field(default_factory=dict)

    # set by the pipeline after provider.add
    memory_id: uuid.UUID | None = None
