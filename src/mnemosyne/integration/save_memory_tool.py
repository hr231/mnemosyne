from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider

if TYPE_CHECKING:
    from mnemosyne.config.settings import Settings

logger = logging.getLogger(__name__)

_VALID_TYPES = {t.value for t in MemoryType}
_DEFAULT_CONTENT_CAP = 10_000


def save_memory_tool_spec() -> dict:
    return {
        "name": "save_memory",
        "description": "Save an important fact, preference, or observation about the user to long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to save.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": [t.value for t in MemoryType],
                    "description": "The type of memory.",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score from 0.0 to 1.0.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "source_session_id": {
                    "type": "string",
                    "description": "UUID of the session this memory originated from.",
                },
            },
            "required": ["content"],
        },
    }


async def handle_save_memory(
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    user_id: UUID,
    args: dict,
    *,
    settings: "Settings | None" = None,
    content_cap: int | None = None,
    agent_id: UUID | None = None,
    org_id: UUID | None = None,
    fail_open: bool = True,
) -> dict:
    """Validate ``args`` and persist a single memory for ``user_id``.

    ``content_cap`` takes precedence when supplied; otherwise the cap is read
    from ``settings.save_memory_content_cap`` and falls back to the module
    default. ``agent_id`` / ``org_id`` are host-supplied provenance and are
    stamped on the memory only when provided — they are deliberately NOT part
    of the tool schema so the calling model cannot spoof another agent's or
    org's scope.
    """
    if content_cap is None:
        content_cap = (
            settings.save_memory_content_cap
            if settings is not None
            else _DEFAULT_CONTENT_CAP
        )
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        return {"status": "error", "error": "content is required and must be a non-empty string"}
    if len(content) > content_cap:
        return {
            "status": "error",
            "error": f"content exceeds maximum length of {content_cap} characters",
        }

    raw_type = args.get("memory_type", "fact")
    if raw_type not in _VALID_TYPES:
        return {"status": "error", "error": f"invalid memory_type: {raw_type!r}"}

    raw_importance = args.get("importance", 0.5)
    if raw_importance is None:
        raw_importance = 0.5
    try:
        importance = float(raw_importance)
    except (TypeError, ValueError):
        return {"status": "error", "error": f"importance must be numeric, got {raw_importance!r}"}
    if importance != importance:  # NaN guard
        return {"status": "error", "error": "importance must be a real number"}
    if not 0.0 <= importance <= 1.0:
        return {"status": "error", "error": f"importance must be 0.0-1.0, got {importance}"}

    session_id = None
    if "source_session_id" in args:
        try:
            session_id = UUID(args["source_session_id"])
        except (ValueError, AttributeError):
            return {"status": "error", "error": f"invalid source_session_id: {args['source_session_id']!r}"}

    memory_type = MemoryType(raw_type)
    scope: dict = {}
    if agent_id is not None:
        scope["agent_id"] = agent_id
    if org_id is not None:
        scope["org_id"] = org_id
    try:
        embedding = await embedder.embed(content)
        memory = Memory(
            user_id=user_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            embedding=embedding,
            source_session_id=session_id,
            **scope,
        )
        mem_id = await provider.add(memory)
    except Exception as exc:
        if not fail_open:
            raise
        logger.warning(
            "save_memory failed for user %s (isolated): %s",
            user_id,
            exc,
            exc_info=True,
        )
        return {"status": "error", "error": f"failed to save memory: {exc}"}
    return {"status": "saved", "memory_id": str(mem_id)}
