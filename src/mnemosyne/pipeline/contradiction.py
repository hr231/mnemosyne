from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import StrEnum

from mnemosyne.db.models.memory import Memory, MemoryType
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.scoring import MultiSignalScorer

logger = logging.getLogger(__name__)

# Contradiction candidate band — sits between dedup threshold (0.90) and
# "too dissimilar to compare" (< 0.70). Dedup owns >= 0.90; contradiction
# owns [0.70, 0.89].
CONTRADICTION_SIMILARITY_MIN: float = 0.70
CONTRADICTION_SIMILARITY_MAX: float = 0.89

# NLI contradiction score thresholds.
# >= HIGH: confirmed contradiction, resolve via LLM.
# [LOW, HIGH): low-confidence, still escalate to LLM but log it.
# < LOW: not a contradiction, skip the pair.
NLI_CONTRADICTION_HIGH: float = 0.7
NLI_CONTRADICTION_LOW: float = 0.4

ADJUDICATION_PROMPT = """\
Memory A (older): {old_content}
Memory B (newer): {new_content}

Do these memories contradict each other? If so, which is more likely current?

Respond with ONLY one of these actions:
- SUPERSEDE: B replaces A (A is outdated)
- KEEP_BOTH: Both are valid in different contexts
- MERGE: Combine into a single updated memory
- KEEP_OLD: A is still correct, B is wrong

Response:\
"""


class ContradictionAction(StrEnum):
    SUPERSEDE = "supersede"
    KEEP_BOTH = "keep_both"
    MERGE = "merge"
    KEEP_OLD = "keep_old"


async def detect_contradictions(
    new_memory: Memory,
    provider: MemoryProvider,
    embedder: EmbeddingClient,
    use_nli: bool = True,
) -> list[tuple[Memory, float]]:
    """Find existing memories that may contradict *new_memory*.

    Searches the provider for semantically similar memories belonging to the
    same user, then narrows to the contradiction band (cosine 0.70–0.89,
    below the dedup threshold of 0.90). For each candidate, an NLI
    contradiction score is computed (falls back to cosine similarity when NLI
    is unavailable).

    Returns a list of (existing_memory, contradiction_score) pairs sorted by
    descending score. Only pairs whose contradiction score >= NLI_CONTRADICTION_LOW
    are returned.
    """
    if new_memory.embedding is None:
        return []

    hits = await provider.search(
        new_memory.embedding,
        user_id=new_memory.user_id,
        limit=20,
    )

    candidates: list[tuple[Memory, float]] = []

    for h in hits:
        existing = h.memory
        if existing.memory_id == new_memory.memory_id:
            continue
        if existing.embedding is None:
            continue

        sim = MultiSignalScorer._cosine_sim(new_memory.embedding, existing.embedding)
        if sim < CONTRADICTION_SIMILARITY_MIN or sim > CONTRADICTION_SIMILARITY_MAX:
            continue

        # Default: use cosine similarity as proxy for contradiction score.
        contradiction_score = sim

        if use_nli:
            try:
                from mnemosyne.pipeline.nli import predict_nli
                nli_result = predict_nli(new_memory.content, existing.content)
                contradiction_score = nli_result.contradiction
            except ImportError:
                pass  # torch/transformers not installed; fall back to similarity

        if contradiction_score >= NLI_CONTRADICTION_LOW:
            candidates.append((existing, contradiction_score))

    candidates.sort(key=lambda x: -x[1])
    return candidates


def _parse_action(response: str) -> ContradictionAction:
    """Parse an LLM adjudication response into a ContradictionAction.

    Scans the response for the first matching action token (case-insensitive).
    Defaults to KEEP_BOTH when no recognised action is found.
    """
    upper = response.strip().upper()
    # Check in specificity order to avoid KEEP_OLD matching on "KEEP_BOTH" etc.
    for action in (
        ContradictionAction.SUPERSEDE,
        ContradictionAction.KEEP_BOTH,
        ContradictionAction.MERGE,
        ContradictionAction.KEEP_OLD,
    ):
        if action.value.upper() in upper:
            return action
    return ContradictionAction.KEEP_BOTH


async def _execute_action(
    action: ContradictionAction,
    new_memory: Memory,
    old_memory: Memory,
    provider: MemoryProvider,
    embedder: EmbeddingClient,
) -> None:
    """Apply the resolved contradiction action to the provider."""
    if action == ContradictionAction.SUPERSEDE:
        await provider.invalidate(old_memory.memory_id, reason="contradiction_superseded")

    elif action == ContradictionAction.KEEP_OLD:
        await provider.invalidate(new_memory.memory_id, reason="contradiction_rejected")

    elif action == ContradictionAction.MERGE:
        merged_content = f"{old_memory.content}; updated: {new_memory.content}"
        merged_embedding = await embedder.embed(merged_content)
        merged = Memory(
            user_id=new_memory.user_id,
            content=merged_content,
            memory_type=new_memory.memory_type,
            importance=max(old_memory.importance, new_memory.importance),
            embedding=merged_embedding,
            source_memory_ids=[old_memory.memory_id, new_memory.memory_id],
            metadata={
                "merged_from": [str(old_memory.memory_id), str(new_memory.memory_id)],
            },
        )
        await provider.add(merged)
        await provider.invalidate(old_memory.memory_id, reason="contradiction_merged")
        await provider.invalidate(new_memory.memory_id, reason="contradiction_merged")

    elif action == ContradictionAction.KEEP_BOTH:
        pass  # Both memories remain valid; no provider writes needed.


async def resolve_contradiction(
    new_memory: Memory,
    old_memory: Memory,
    contradiction_score: float,
    provider: MemoryProvider,
    llm_client: LLMClient,
    embedder: EmbeddingClient,
) -> ContradictionAction:
    """Adjudicate a single contradiction pair via LLM and execute the action.

    Returns the chosen ContradictionAction. On LLM failure, defaults to
    KEEP_BOTH so no data is accidentally discarded.
    """
    prompt = ADJUDICATION_PROMPT.format(
        old_content=old_memory.content,
        new_content=new_memory.content,
    )

    try:
        response = await llm_client.complete(prompt)
        action = _parse_action(response)
    except Exception as exc:
        logger.warning(
            "LLM adjudication failed for pair (%s, %s): %s — defaulting to KEEP_BOTH",
            old_memory.memory_id,
            new_memory.memory_id,
            exc,
        )
        action = ContradictionAction.KEEP_BOTH

    await _execute_action(action, new_memory, old_memory, provider, embedder)

    logger.info(
        "Contradiction resolved: action=%s old=%s new=%s score=%.2f",
        action.value,
        old_memory.memory_id,
        new_memory.memory_id,
        contradiction_score,
    )
    return action


async def run_contradiction_check(
    provider: MemoryProvider,
    user_id: uuid.UUID,
    llm_client: LLMClient,
    embedder: EmbeddingClient,
    use_nli: bool = True,
    created_after: datetime | None = None,
) -> int:
    """Check recent memories for contradictions and resolve each found pair.

    Iterates over the user's recent memories (up to 50), runs
    detect_contradictions for each, and calls resolve_contradiction on pairs
    above the low-confidence threshold. Tracks which pairs have been checked
    to avoid double-processing.

    Returns the count of contradiction pairs resolved.
    """
    dummy_vec = await embedder.embed("contradiction check")
    hits = await provider.search(dummy_vec, user_id=user_id, limit=50)

    resolved = 0
    checked_pairs: set[tuple[uuid.UUID, uuid.UUID]] = set()

    for h in hits:
        mem = h.memory
        if created_after is not None:
            mem_created = mem.created_at
            if mem_created.tzinfo is None:
                mem_created = mem_created.replace(tzinfo=timezone.utc)
            cutoff = created_after
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            if mem_created < cutoff:
                continue

        contradictions = await detect_contradictions(mem, provider, embedder, use_nli=use_nli)

        for old_mem, score in contradictions:
            # Canonical pair key so (A, B) and (B, A) map to the same entry.
            pair_key: tuple[uuid.UUID, uuid.UUID] = tuple(
                sorted([mem.memory_id, old_mem.memory_id])
            )  # type: ignore[assignment]
            if pair_key in checked_pairs:
                continue
            checked_pairs.add(pair_key)

            if score >= NLI_CONTRADICTION_LOW:
                if score < NLI_CONTRADICTION_HIGH:
                    logger.info(
                        "Low-confidence contradiction (%.2f) between %s and %s, escalating to LLM",
                        score,
                        mem.memory_id,
                        old_mem.memory_id,
                    )
                await resolve_contradiction(
                    mem, old_mem, score, provider, llm_client, embedder
                )
                resolved += 1

    return resolved
