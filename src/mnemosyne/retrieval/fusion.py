"""Reciprocal Rank Fusion for combining ranked candidate lists.

Used by entity-aware search (vector vs entity-expanded lists) and by the
Postgres hybrid leg (vector vs full-text lists). The two input lists are
treated symmetrically: a memory appearing in both lists at good ranks wins.
"""

from __future__ import annotations

import uuid

from mnemosyne.db.models.memory import ScoredMemory

RRF_K = 60


def rrf_score(ranks: list[int], k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score from 1-indexed ranks."""
    return sum(1.0 / (k + rank) for rank in ranks)


def fuse_rrf(
    left: list[ScoredMemory],
    right: list[ScoredMemory],
    limit: int,
    k: int = RRF_K,
) -> list[ScoredMemory]:
    """Fuse two ranked ScoredMemory lists with Reciprocal Rank Fusion.

    Ranks are derived from list position (1-indexed). The returned objects
    carry ``score`` set to the RRF score and a ``score_breakdown`` augmented
    with ``rrf_score``/``in_left``/``in_right`` while preserving any
    pre-existing breakdown keys from the first-seen ScoredMemory.
    """
    left_ranks: dict[uuid.UUID, int] = {
        sm.memory.memory_id: i + 1 for i, sm in enumerate(left)
    }
    right_ranks: dict[uuid.UUID, int] = {
        sm.memory.memory_id: i + 1 for i, sm in enumerate(right)
    }

    first_seen: dict[uuid.UUID, ScoredMemory] = {}
    for sm in left:
        first_seen.setdefault(sm.memory.memory_id, sm)
    for sm in right:
        first_seen.setdefault(sm.memory.memory_id, sm)

    fused: list[tuple[float, ScoredMemory]] = []
    for mid, sm in first_seen.items():
        ranks: list[int] = []
        if mid in left_ranks:
            ranks.append(left_ranks[mid])
        if mid in right_ranks:
            ranks.append(right_ranks[mid])

        score = rrf_score(ranks, k)
        merged = ScoredMemory(
            memory=sm.memory,
            score=score,
            score_breakdown={
                **sm.score_breakdown,
                "rrf_score": score,
                "in_left": 1.0 if mid in left_ranks else 0.0,
                "in_right": 1.0 if mid in right_ranks else 0.0,
            },
            score_breakdown_explain=sm.score_breakdown_explain,
        )
        fused.append((score, merged))

    fused.sort(key=lambda x: -x[0])
    return [sm for _, sm in fused[:limit]]
