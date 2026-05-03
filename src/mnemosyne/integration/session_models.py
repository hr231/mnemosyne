from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


MessageRole = Literal["user", "assistant", "system"]


class ConversationMessage(BaseModel):
    """A single role-tagged message in a session stream."""

    message_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    role: MessageRole
    content: str
    sent_at: datetime

    @field_validator("content")
    @classmethod
    def _content_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content must be non-empty")
        return v


class SessionBatch(BaseModel):
    """An assembled, ordered batch of messages from one session."""

    session_id: uuid.UUID
    user_id: uuid.UUID
    messages: list[ConversationMessage]
    started_at: datetime
    closed_at: datetime

    @model_validator(mode="after")
    def _non_empty_messages(self) -> SessionBatch:
        if not self.messages:
            raise ValueError("SessionBatch must contain at least one message")
        return self
