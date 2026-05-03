"""Pydantic models for the re-extraction pipeline."""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class DecisionKind(str, Enum):
    KEEP = "keep"
    SUPERSEDE = "supersede"
    NEW = "new"


class ReextractionDecision(BaseModel):
    """Decision made by the driver for a single old/new memory pair."""

    kind: DecisionKind
    old_memory_id: uuid.UUID | None = None
    new_content: str | None = None
    reason: str = ""


class ReextractionResult(BaseModel):
    """Aggregate result of ``ReextractionDriver.reextract_user``."""

    user_id: uuid.UUID
    target_version: str
    count_processed: int
    count_changed: int
    count_superseded: int
    count_new: int
    count_kept: int
    started_at: datetime
    finished_at: datetime


class ReextractionJobRow(BaseModel):
    """One row in ``memory.reextraction_jobs``."""

    id: uuid.UUID
    user_id: uuid.UUID
    target_version: str
    status: str
    count_processed: int
    count_changed: int
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None
