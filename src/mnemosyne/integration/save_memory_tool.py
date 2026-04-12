from __future__ import annotations

from uuid import UUID

from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider


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
                    "enum": ["fact", "preference", "entity", "procedural", "reflection"],
                    "description": "The type of memory.",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score from 0.0 to 1.0.",
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
    content = args["content"]
    memory_type = MemoryType(args.get("memory_type", "fact"))
    importance = float(args.get("importance", 0.5))

    embedding = await embedder.embed(content)
    memory = Memory(
        user_id=user_id,
        content=content,
        memory_type=memory_type,
        importance=importance,
        embedding=embedding,
    )
    mem_id = await provider.add(memory)
    return {"status": "saved", "memory_id": str(mem_id)}
