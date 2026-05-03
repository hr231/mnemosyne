from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from mnemosyne.integration.session_models import ConversationMessage, SessionBatch


class SessionAssembler:
    """In-process buffer that accumulates ConversationMessages into SessionBatches."""

    def __init__(self) -> None:
        self._buffers: dict[uuid.UUID, list[ConversationMessage]] = {}
        self._users: dict[uuid.UUID, uuid.UUID] = {}
        self._started_at: dict[uuid.UUID, datetime] = {}
        self._lock = asyncio.Lock()

    async def add_message(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        message: ConversationMessage,
    ) -> None:
        """Append a message to the session's buffer."""
        async with self._lock:
            buf = self._buffers.setdefault(session_id, [])
            buf.append(message)
            self._users[session_id] = user_id
            self._started_at.setdefault(session_id, message.sent_at)

    async def flush(self, session_id: uuid.UUID) -> SessionBatch | None:
        """Return the assembled batch and clear the buffer. None if unknown."""
        async with self._lock:
            messages = self._buffers.pop(session_id, None)
            user_id = self._users.pop(session_id, None)
            started_at = self._started_at.pop(session_id, None)
            if not messages or user_id is None or started_at is None:
                return None
            return SessionBatch(
                session_id=session_id,
                user_id=user_id,
                messages=messages,
                started_at=started_at,
                closed_at=datetime.now(timezone.utc),
            )


class PersistentSessionAssembler:
    """Postgres-backed assembler. Use when MNEMOSYNE_SESSION_PERSIST=1."""

    def __init__(self, pool) -> None:
        self._pool = pool

    async def add_message(
        self,
        session_id: uuid.UUID,
        user_id: uuid.UUID,
        message: ConversationMessage,
    ) -> None:
        async with self._pool.acquire() as conn:
            ordinal = await conn.fetchval(
                """
                SELECT COALESCE(MAX(ordinal), -1) + 1
                FROM memory.session_buffer
                WHERE session_id = $1
                """,
                session_id,
            )
            await conn.execute(
                """
                INSERT INTO memory.session_buffer
                  (session_id, ordinal, message_id, user_id, role, content, sent_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                session_id,
                ordinal,
                message.message_id,
                user_id,
                message.role,
                message.content,
                message.sent_at,
            )

    async def flush(self, session_id: uuid.UUID) -> SessionBatch | None:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT message_id, user_id, role, content, sent_at
                FROM memory.session_buffer
                WHERE session_id = $1
                ORDER BY ordinal ASC
                """,
                session_id,
            )
            if not rows:
                return None
            await conn.execute(
                "DELETE FROM memory.session_buffer WHERE session_id = $1",
                session_id,
            )
        user_id = rows[0]["user_id"]
        messages = [
            ConversationMessage(
                message_id=r["message_id"],
                role=r["role"],
                content=r["content"],
                sent_at=r["sent_at"],
            )
            for r in rows
        ]
        return SessionBatch(
            session_id=session_id,
            user_id=user_id,
            messages=messages,
            started_at=messages[0].sent_at,
            closed_at=datetime.now(timezone.utc),
        )
