"""Reflection generation.

Reflections are persisted as first-class :class:`Memory` rows with
``memory_type=MemoryType.REFLECTION``. Retrieval does NOT branch on reflection
type — reflections rank via the same vector + RRF paths as any other memory.
Recursive depth is enforced via ``metadata["reflection_depth"]`` capped at
``MAX_REFLECTION_DEPTH`` (=2): memories at that depth are excluded from the
importance-sum trigger so the pipeline cannot recurse indefinitely.

Runner entry point: :func:`maybe_run_reflection` — fires only when the
importance sum crosses the Generative-Agents-style threshold; otherwise a
no-op that returns 0.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.db.repositories.user_settings import (
    get_last_reflected_at,
    set_last_reflected_at,
)
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.llm.hardening import render_with_untrusted
from mnemosyne.pipeline.consolidation import accumulated_importance_sum, REFLECTION_IMPORTANCE_SUM_THRESHOLD
from mnemosyne.providers.base import MemoryProvider

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = """Given these memories about a user, generate 2-3 high-level insights about their preferences, patterns, or needs. Each insight should be a single sentence.

Memories:
$input

Return a JSON array of strings, each being one insight. Example:
["User prefers comfort over style in footwear", "User has a budget range of 100-200 dollars"]

Respond with ONLY valid JSON array."""

MAX_REFLECTION_DEPTH = 2


async def should_generate_reflection(
    provider: MemoryProvider,
    user_id: uuid.UUID,
    since: datetime | None = None,
) -> bool:
    """Check if enough importance has accumulated to trigger reflection.

    Uses ``list_for_user`` rather than a query so the trigger check never
    bumps access counts or last_accessed on the scanned memories.
    """
    memories = await provider.list_for_user(user_id)

    recent_memories = []
    for mem in memories:
        # Reflections never feed the trigger — at any depth — so a burst of
        # generated insights cannot itself drive further reflection.
        if mem.memory_type == MemoryType.REFLECTION:
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
    memories = await provider.list_for_user(user_id)

    candidates: list[Memory] = []
    for mem in memories:
        depth = mem.metadata.get("reflection_depth", 0)
        if depth >= MAX_REFLECTION_DEPTH:
            continue
        if since and mem.created_at < since:
            continue
        candidates.append(mem)

    # Select the highest-importance memories rather than the most recent slice
    # so a burst of low-signal memories cannot crowd out significant ones.
    candidates.sort(key=lambda m: m.importance, reverse=True)
    input_memories = candidates[:max_input_memories]

    if not input_memories:
        return []

    max_depth = max(
        (m.metadata.get("reflection_depth", 0) for m in input_memories), default=0
    )

    memory_texts = "\n".join(
        f"- [{m.memory_type.value}] (importance: {m.importance:.1f}) {m.content}"
        for m in input_memories
    )
    prompt = render_with_untrusted(REFLECTION_PROMPT, memory_texts)

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


async def maybe_run_reflection(
    provider: MemoryProvider,
    llm: LLMClient,
    embedder: EmbeddingClient,
    user_id: uuid.UUID,
    pool: object | None = None,
) -> int:
    """Trigger reflection if the importance-sum threshold has been crossed.

    Returns the number of reflection memories persisted. When the trigger
    does not fire, returns 0 without invoking the LLM.

    When *pool* is supplied the reflection watermark is honoured: only memories
    created after the last reflection count toward the trigger and feed
    generation, and the watermark is advanced once generation runs. Without a
    pool (e.g. the in-memory runner) the historical unwatermarked behaviour is
    preserved.
    """
    since: datetime | None = None
    if pool is not None:
        try:
            since = await get_last_reflected_at(pool, user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "reflection watermark read failed for user %s: %s", user_id, exc
            )
            since = None

    if not await should_generate_reflection(provider, user_id, since=since):
        return 0

    reflections = await generate_reflections(
        provider=provider,
        user_id=user_id,
        llm_client=llm,
        embedder=embedder,
        since=since,
    )

    if pool is not None:
        # Advance the watermark whenever the trigger fired and generation ran,
        # even if no insights were produced, so the same accumulated set does
        # not re-trigger on every maintenance tick.
        try:
            await set_last_reflected_at(pool, user_id, datetime.now(timezone.utc))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "reflection watermark write failed for user %s: %s", user_id, exc
            )

    return len(reflections)
