from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class Episode(BaseModel):
    episode_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: uuid.UUID
    agent_id: uuid.UUID = Field(default=uuid.UUID("00000000-0000-0000-0000-000000000000"))
    session_id: uuid.UUID

    summary: str
    summary_embedding: list[float] | None = None
    key_topics: list[str] = Field(default_factory=list)
    memory_ids: list[uuid.UUID] = Field(default_factory=list)
    outcome: str | None = None

    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
