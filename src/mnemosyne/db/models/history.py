from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class MemoryHistoryEntry(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    memory_id: uuid.UUID
    operation: Literal["create", "update", "invalidate", "merge"]
    old_content: str | None = None
    new_content: str | None = None
    old_importance: float | None = None
    new_importance: float | None = None
    actor: Literal[
        "agent_tool", "pipeline_extraction", "pipeline_consolidation",
        "pipeline_decay", "manual"
    ]
    actor_details: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
