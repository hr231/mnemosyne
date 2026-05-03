from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class GdprDeletion(BaseModel):
    """Audit row written BEFORE a physical user-delete."""

    id: uuid.UUID
    user_id: uuid.UUID
    requestor: str = Field(..., min_length=1)
    reason: str = "user_request"
    rows_memories: int = 0
    rows_entities: int = 0
    rows_mentions: int = 0
    rows_episodes: int = 0
    rows_history: int = 0
    occurred_at: datetime
    dry_run: bool = False
