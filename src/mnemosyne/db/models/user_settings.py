from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class UserSettings(BaseModel):
    user_id: uuid.UUID
    extraction_enabled: bool = True
    last_reflected_at: datetime | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
