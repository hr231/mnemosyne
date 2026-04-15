from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.pipeline.consolidation import accumulated_importance_sum, REFLECTION_IMPORTANCE_SUM_THRESHOLD
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = """Given these memories about a user, generate 2-3 high-level insights about their preferences, patterns, or needs. Each insight should be a single sentence.

Memories:
{memories}

Return a JSON array of strings, each being one insight. Example:
["User prefers comfort over style in footwear", "User has a budget range of $100-200"]

Respond with ONLY valid JSON array."""

MAX_REFLECTION_DEPTH = 2


async def should_generate_reflection(
    provider: MemoryProvider,
    user_id: uuid.UUID,
    since: datetime | None = None,
) -> bool:
    """Check if enough importance has accumulated to trigger reflection."""
    dummy_vec = [0.0] * 768  # dummy vector for trigger check scan

    hits = await provider.search(dummy_vec, user_id=user_id, limit=200)

    recent_memories = []
    for h in hits:
        mem = h.memory
        depth = mem.metadata.get("reflection_depth", 0)
        if depth >= MAX_REFLECTION_DEPTH:
            continue
        if since and mem.created_at < since:
            continue
        recent_memories.append(mem)

    return accumulated_importance_sum(recent_memories) >= REFLECTION_IMPORTANCE_SUM_THRESHOLD


async def generate_reflections(
    provider: MemoryProvider,
    user_id: uuid.UUID,
    llm_client: LLMClient,
    embedder: EmbeddingClient,
    since: datetime | None = None,
    max_input_memories: int = 100,
) -> list[Memory]:
    """Generate reflection memories from accumulated knowledge.

    Returns the list of newly created reflection Memory objects.
    """
    dummy_vec = await embedder.embed("reflection generation")
    hits = await provider.search(dummy_vec, user_id=user_id, limit=max_input_memories * 2)

    input_memories = []
    max_depth = 0
    for h in hits[:max_input_memories]:
        mem = h.memory
        depth = mem.metadata.get("reflection_depth", 0)
        if depth >= MAX_REFLECTION_DEPTH:
            continue
        input_memories.append(mem)
        max_depth = max(max_depth, depth)

    if not input_memories:
        return []

    memory_texts = "\n".join(
        f"- [{m.memory_type.value}] (importance: {m.importance:.1f}) {m.content}"
        for m in input_memories
    )
    prompt = REFLECTION_PROMPT.format(memories=memory_texts)

    try:
        raw = await llm_client.complete(prompt)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.split("\n") if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        insights = json.loads(cleaned)
        if not isinstance(insights, list):
            logger.warning("Reflection LLM returned non-array: %s", type(insights).__name__)
            return []
    except Exception as exc:
        logger.warning("Reflection generation failed: %s", exc)
        return []

    source_ids = [m.memory_id for m in input_memories]
    new_depth = max_depth + 1

    reflections = []
    for insight in insights:
        if not isinstance(insight, str) or not insight.strip():
            continue

        embedding = await embedder.embed(insight)
        reflection = Memory(
            user_id=user_id,
            content=insight.strip(),
            memory_type=MemoryType.REFLECTION,
            importance=0.9,
            embedding=embedding,
            source_memory_ids=source_ids,
            metadata={"reflection_depth": new_depth},
        )
        mem_id = await provider.add(reflection)
        reflection = reflection.model_copy(update={"memory_id": mem_id})
        reflections.append(reflection)

    logger.info(
        "Generated %d reflections for user %s (depth %d)",
        len(reflections), user_id, new_depth,
    )
    return reflections
