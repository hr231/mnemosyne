from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from mnemosyne.db.models.memory import Memory
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.scoring import MultiSignalScorer
from mnemosyne.utils import content_hash

logger = logging.getLogger(__name__)

# Importance-sum threshold from Stanford Generative Agents.
# Importance ∈ [0, 1] is scaled ×10 to the [0, 10] range used in the paper.
# Reflection triggers when the accumulated sum reaches this value.
REFLECTION_IMPORTANCE_SUM_THRESHOLD: float = 150.0

# Semantic similarity threshold above which two memories are considered
# duplicates and the lower-importance one is invalidated.
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.90


def compute_content_hash(content: str) -> str:
    """Return the SHA-256 hex digest of normalised *content*.

    Normalisation: strip surrounding whitespace, lowercase.
    Two strings that differ only in case or leading/trailing whitespace
    produce the same hash.

    Delegates to :func:`mnemosyne.utils.content_hash` so there is a single
    canonical implementation.
    """
    return content_hash(content)


def find_semantic_duplicates(
    memories: list[Memory],
    similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> list[tuple[Memory, Memory]]:
    """Find pairs of memories with cosine similarity >= *similarity_threshold*.

    Only pairs belonging to the same user are compared.  Memories without
    embeddings are skipped.

    Threshold semantics
    -------------------
    - >= 0.95  Near-certain duplicate — auto-merge.
    - 0.90–0.95  Safe to merge; higher-importance memory wins.
    - < 0.90   Distinct — keep both.
    """
    pairs: list[tuple[Memory, Memory]] = []

    for i, a in enumerate(memories):
        if a.embedding is None:
            continue
        for b in memories[i + 1 :]:
            if b.embedding is None:
                continue
            if a.user_id != b.user_id:
                continue
            sim = MultiSignalScorer._cosine_sim(a.embedding, b.embedding)
            if sim >= similarity_threshold:
                pairs.append((a, b))

    return pairs


def accumulated_importance_sum(memories: list[Memory]) -> float:
    """Return the accumulated importance sum scaled to [0, 10] per memory.

    Mirrors the Stanford Generative Agents mechanism where importance is
    rated 1–10.  Since Mnemosyne stores importance ∈ [0, 1], we multiply
    by 10.  Mundane memories (importance ~0.1–0.2) accumulate slowly;
    significant ones (importance ~0.8–1.0) trigger reflection quickly.
    """
    return sum(m.importance * 10.0 for m in memories)


def needs_reflection(
    recent_memories: list[Memory],
    threshold: float = REFLECTION_IMPORTANCE_SUM_THRESHOLD,
) -> bool:
    """Return True if the accumulated importance sum meets the threshold.

    Callers are responsible for passing only memories created since the
    last reflection.
    """
    return accumulated_importance_sum(recent_memories) >= threshold


async def run_dedup(
    provider: MemoryProvider,
    user_id: uuid.UUID,
    created_after: datetime | None = None,
) -> int:
    """Run three-tier deduplication for *user_id*.

    Tier 1 — Exact hash: scan for memories with the same ``content_hash``
    and invalidate all but the one with the highest importance.

    Tier 2 — Fuzzy (pg_trgm): Postgres-only.  Uses
    ``dedup.find_fuzzy_duplicates`` when the provider exposes a ``_pool``.

    Tier 3 — Semantic: In-memory cosine comparison for ``InMemoryProvider``;
    ``dedup.find_semantic_duplicates`` for Postgres.

    Returns the total count of memories invalidated.
    """
    merged = 0

    if hasattr(provider, "_memories"):
        merged += await _dedup_in_memory(provider, user_id)
    elif hasattr(provider, "_pool"):
        merged += await _dedup_postgres(provider, user_id, created_after)
    else:
        logger.warning(
            "run_dedup: unknown provider type %s — skipping",
            type(provider).__name__,
        )

    if merged:
        logger.info("run_dedup: invalidated %d duplicate memories for user %s", merged, user_id)
    return merged


async def _dedup_in_memory(provider: MemoryProvider, user_id: uuid.UUID) -> int:
    """Exact-hash + semantic dedup for InMemoryProvider."""
    merged = 0

    # Collect active memories for this user
    active: list[Memory] = [
        m
        for m in provider._memories.values()  # type: ignore[attr-defined]
        if m.user_id == user_id and m.valid_until is None
    ]

    # --- Tier 1: exact-hash dedup ---
    hash_groups: dict[str, list[Memory]] = {}
    for mem in active:
        ch = mem.content_hash or compute_content_hash(mem.content)
        hash_groups.setdefault(ch, []).append(mem)

    for ch, group in hash_groups.items():
        if len(group) < 2:
            continue
        # Keep the one with the highest importance; invalidate the rest
        group.sort(key=lambda m: m.importance, reverse=True)
        winner = group[0]
        for loser in group[1:]:
            await provider.invalidate(loser.memory_id, reason="dedup_merge")
            merged += 1

    # Re-collect active after exact dedup
    active = [
        m
        for m in provider._memories.values()  # type: ignore[attr-defined]
        if m.user_id == user_id and m.valid_until is None
    ]

    # --- Tier 3: semantic dedup ---
    sem_pairs = find_semantic_duplicates(active, SEMANTIC_SIMILARITY_THRESHOLD)
    invalidated: set[uuid.UUID] = set()
    for a, b in sem_pairs:
        if a.memory_id in invalidated or b.memory_id in invalidated:
            continue
        loser = b if a.importance >= b.importance else a
        await provider.invalidate(loser.memory_id, reason="dedup_merge")
        invalidated.add(loser.memory_id)
        merged += 1

    return merged


async def _dedup_postgres(
    provider: MemoryProvider,
    user_id: uuid.UUID,
    created_after: datetime | None,
) -> int:
    """Three-tier dedup for PostgresMemoryProvider via dedup repository."""
    from mnemosyne.db.repositories import dedup as dedup_repo

    merged = 0
    pool = provider._pool  # type: ignore[attr-defined]

    async with pool.acquire() as conn:
        # --- Tier 1: exact-hash ---
        exact_groups = await dedup_repo.find_exact_duplicates(conn, user_id, created_after)
        for group in exact_groups:
            memory_ids: list[uuid.UUID] = group["memory_ids"]
            if len(memory_ids) < 2:
                continue
            # Load all memories in the group to find the winner
            mems = []
            for mid in memory_ids:
                row = await conn.fetchrow(
                    "SELECT memory_id, importance, valid_until FROM memory.memories WHERE memory_id = $1",
                    mid,
                )
                if row and row["valid_until"] is None:
                    mems.append(row)
            if len(mems) < 2:
                continue
            mems.sort(key=lambda r: r["importance"], reverse=True)
            winner_id = mems[0]["memory_id"]
            for loser in mems[1:]:
                await provider.invalidate(loser["memory_id"], reason="dedup_merge")
                merged += 1

        # --- Tier 2: fuzzy (pg_trgm) ---
        fuzzy_pairs = await dedup_repo.find_fuzzy_duplicates(conn, user_id, 0.8, created_after)
        invalidated: set[uuid.UUID] = set()
        for pair in fuzzy_pairs:
            id_a, id_b = pair["id_a"], pair["id_b"]
            if id_a in invalidated or id_b in invalidated:
                continue
            imp_a, imp_b = pair["imp_a"], pair["imp_b"]
            loser_id = id_b if imp_a >= imp_b else id_a
            await provider.invalidate(loser_id, reason="dedup_merge")
            invalidated.add(loser_id)
            merged += 1

        # --- Tier 3: semantic ---
        sem_pairs = await dedup_repo.find_semantic_duplicates(
            conn, user_id, SEMANTIC_SIMILARITY_THRESHOLD, created_after
        )
        for pair in sem_pairs:
            id_a, id_b = pair["id_a"], pair["id_b"]
            if id_a in invalidated or id_b in invalidated:
                continue
            imp_a, imp_b = pair["imp_a"], pair["imp_b"]
            loser_id = id_b if imp_a >= imp_b else id_a
            await provider.invalidate(loser_id, reason="dedup_merge")
            invalidated.add(loser_id)
            merged += 1

    return merged
