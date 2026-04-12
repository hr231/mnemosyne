from __future__ import annotations

from uuid import UUID

from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider

_VALID_TYPES = {t.value for t in MemoryType}


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
) -> dict:
    content = args.get("content")
    if not content or not content.strip():
        return {"status": "error", "error": "content is required and must be non-empty"}

    raw_type = args.get("memory_type", "fact")
    if raw_type not in _VALID_TYPES:
        return {"status": "error", "error": f"invalid memory_type: {raw_type!r}"}

    importance = float(args.get("importance", 0.5))
    if not 0.0 <= importance <= 1.0:
        return {"status": "error", "error": f"importance must be 0.0-1.0, got {importance}"}

    session_id = None
    if "source_session_id" in args:
        try:
            session_id = UUID(args["source_session_id"])
        except (ValueError, AttributeError):
            return {"status": "error", "error": f"invalid source_session_id: {args['source_session_id']!r}"}

    memory_type = MemoryType(raw_type)
    embedding = await embedder.embed(content)
    memory = Memory(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        importance=importance,
        embedding=embedding,
        source_session_id=session_id,
    )
    mem_id = await provider.add(memory)
    return {"status": "saved", "memory_id": str(mem_id)}
