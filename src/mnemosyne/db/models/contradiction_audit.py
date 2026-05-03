from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ResolutionLiteral = Literal["supersede", "keep_both", "merge", "keep_old"]


class ContradictionAudit(BaseModel):
    """One row per resolved contradiction pair."""

    id: uuid.UUID
    user_id: uuid.UUID
    detected_at: datetime
    new_memory_id: uuid.UUID
    existing_memory_id: uuid.UUID
    nli_scores: dict[str, float] = Field(default_factory=dict)
    llm_adjudication: str | None = None
    resolution: ResolutionLiteral
    reasoning: str | None = None
    applied: bool = True
