from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from mnemosyne.db.models.memory import ScoredMemory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.scoring import ScoringWeights

if TYPE_CHECKING:
    from mnemosyne.db.repositories.entity import EntityStore

logger = logging.getLogger(__name__)

# RRF constant (standard value from the literature)
RRF_K = 60

# Max mentions to expand per entity
MAX_MENTIONS_PER_ENTITY = 50


def _rrf_score(ranks: list[int], k: int = RRF_K) -> float:
    """Compute Reciprocal Rank Fusion score from a list of ranks (1-indexed)."""
    return sum(1.0 / (k + rank) for rank in ranks)


async def entity_aware_search(
    provider: MemoryProvider,
    entity_store: "EntityStore | None",
    query_text: str,
    query_embedding: list[float],
    user_id: uuid.UUID,
    embedder: EmbeddingClient | None = None,
    limit: int = 10,
    weights: ScoringWeights | None = None,
) -> list[ScoredMemory]:
    """Search memories using both vector similarity and entity expansion.

    When entity_store is None or NER is unavailable, falls back to
    pure vector search via provider.search().

    Steps:
    1. Run standard vector search
    2. Extract entities from query (spaCy + GLiNER, no LLM for latency)
    3. Look up matching entities in entity_store
    4. Expand via entity_mentions to related memory_ids (capped at 50 per entity)
    5. Fetch those memories
    6. Fuse via Reciprocal Rank Fusion
    """
    # Step 1: Vector search (always runs)
    vector_results = await provider.search(
        query_embedding, user_id=user_id, limit=limit * 3, weights=weights
    )

    # If no entity store, return vector-only results
    if entity_store is None:
        return vector_results[:limit]

    # Step 2: Extract entities from query
    entity_results = await _get_entity_memories(
        query_text, user_id, entity_store, provider, embedder
    )

    if not entity_results:
        return vector_results[:limit]

    # Step 3: RRF fusion
    fused = _fuse_rrf(vector_results, entity_results, limit)
    return fused


async def _get_entity_memories(
    query_text: str,
    user_id: uuid.UUID,
    entity_store: "EntityStore",
    provider: MemoryProvider,
    embedder: EmbeddingClient | None = None,
) -> list[ScoredMemory]:
    """Extract entities from query and expand to related memories."""
    # Try NER extraction (graceful degradation if not installed)
    raw_entities = []
    try:
        from mnemosyne.pipeline.ner.spacy_extractor import extract_entities_spacy
        raw_entities.extend(extract_entities_spacy(query_text))
    except Exception:
        pass

    try:
        from mnemosyne.pipeline.ner.gliner_extractor import extract_entities_gliner
        raw_entities.extend(extract_entities_gliner(query_text))
    except Exception:
        pass

    if not raw_entities:
        # Try simple name-based lookup as fallback: each word that is long
        # enough might be a known entity name.
        for word in query_text.split():
            if len(word) < 3:
                continue
            for entity_type in ("person", "organization", "product", "brand", "location"):
                found = await entity_store.find_by_name(user_id, word, entity_type)
                if found is not None:
                    raw_entities.append(_SimpleEntity(
                        name=word,
                        entity_type=entity_type,
                    ))
                    break

    if not raw_entities:
        # Last resort: embedding similarity if embedder is available
        if embedder:
            try:
                query_emb = await embedder.embed(query_text)
                similar_entities = await entity_store.find_by_embedding(
                    user_id, query_emb, threshold=0.80, limit=5
                )
                for ent in similar_entities:
                    raw_entities.append(_SimpleEntity(
                        name=ent.entity_name,
                        entity_type=ent.entity_type,
                    ))
            except Exception:
                pass

    if not raw_entities:
        return []

    # Look up entities and expand mentions
    all_memory_ids: list[uuid.UUID] = []
    seen_ids: set[uuid.UUID] = set()

    for raw in raw_entities:
        # Find matching entity by exact normalised name
        entity = await entity_store.find_by_name(user_id, raw.name, raw.entity_type)
        if entity is None:
            continue

        # Get related memory_ids (capped at MAX_MENTIONS_PER_ENTITY)
        mention_ids = await entity_store.find_mentions_for_entity(entity.entity_id)
        for mid in mention_ids[:MAX_MENTIONS_PER_ENTITY]:
            if mid not in seen_ids:
                all_memory_ids.append(mid)
                seen_ids.add(mid)

    if not all_memory_ids:
        return []

    # Fetch memories and wrap as ScoredMemory
    entity_scored: list[ScoredMemory] = []
    for mid in all_memory_ids:
        mem = await provider.get_by_id(mid)
        if mem is None or mem.valid_until is not None:
            continue
        entity_scored.append(ScoredMemory(
            memory=mem,
            score=0.0,  # score replaced by RRF fusion
            score_breakdown={"entity_expanded": 1.0},
        ))

    return entity_scored


def _fuse_rrf(
    vector_results: list[ScoredMemory],
    entity_results: list[ScoredMemory],
    limit: int,
) -> list[ScoredMemory]:
    """Fuse two ranked lists using Reciprocal Rank Fusion."""
    # Build rank maps (1-indexed)
    vector_ranks: dict[uuid.UUID, int] = {}
    for i, sm in enumerate(vector_results):
        vector_ranks[sm.memory.memory_id] = i + 1

    entity_ranks: dict[uuid.UUID, int] = {}
    for i, sm in enumerate(entity_results):
        entity_ranks[sm.memory.memory_id] = i + 1

    # Collect all unique memory_ids, preserving first-seen ScoredMemory object
    all_memories: dict[uuid.UUID, ScoredMemory] = {}
    for sm in vector_results:
        all_memories[sm.memory.memory_id] = sm
    for sm in entity_results:
        if sm.memory.memory_id not in all_memories:
            all_memories[sm.memory.memory_id] = sm

    # Compute RRF scores and build output list
    rrf_scores: list[tuple[float, ScoredMemory]] = []
    for mid, sm in all_memories.items():
        ranks = []
        if mid in vector_ranks:
            ranks.append(vector_ranks[mid])
        if mid in entity_ranks:
            ranks.append(entity_ranks[mid])

        rrf = _rrf_score(ranks)
        fused_sm = ScoredMemory(
            memory=sm.memory,
            score=rrf,
            score_breakdown={
                **sm.score_breakdown,
                "rrf_score": rrf,
                "in_vector": 1.0 if mid in vector_ranks else 0.0,
                "in_entity": 1.0 if mid in entity_ranks else 0.0,
            },
        )
        rrf_scores.append((rrf, fused_sm))

    # Sort by RRF score descending
    rrf_scores.sort(key=lambda x: -x[0])
    return [sm for _, sm in rrf_scores[:limit]]


class _SimpleEntity:
    """Minimal entity placeholder used when NER is unavailable."""

    __slots__ = ("name", "entity_type")

    def __init__(self, name: str, entity_type: str) -> None:
        self.name = name
        self.entity_type = entity_type
