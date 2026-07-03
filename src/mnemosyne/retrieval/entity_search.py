"""Entity-aware retrieval.

Public surface: ``entity_aware_search`` is a STANDALONE function.
The ``MemoryProvider.search()`` signature is unchanged. Callers pick explicitly:

- Pure vector search: ``provider.search(...)``
- Entity-boosted search: ``entity_aware_search(provider, entity_store, ...)``

Reflections are not a special case for retrieval — they are persisted as
regular memories with ``memory_type=MemoryType.REFLECTION`` and ranked via
the same vector + RRF path as any other memory.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from mnemosyne.db.models.memory import ScoredMemory
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.providers.base import MemoryProvider
from mnemosyne.retrieval.fusion import RRF_K, fuse_rrf
from mnemosyne.retrieval.fusion import rrf_score as _rrf_score
from mnemosyne.retrieval.scoring import ScoringWeights

if TYPE_CHECKING:
    from mnemosyne.db.repositories.entity import EntityStore

logger = logging.getLogger(__name__)

# Max mentions to expand per entity
MAX_MENTIONS_PER_ENTITY = 50


async def entity_aware_search(
    provider: MemoryProvider,
    entity_store: "EntityStore | None",
    query_text: str,
    query_embedding: list[float],
    user_id: uuid.UUID,
    embedder: EmbeddingClient | None = None,
    limit: int = 10,
    weights: ScoringWeights | None = None,
    entity_filter: list[str] | None = None,
    *,
    out_entity_names: list[str] | None = None,
) -> list[ScoredMemory]:
    """Search memories using both vector similarity and entity expansion.

    When entity_store is None or NER is unavailable, falls back to
    pure vector search via provider.search().

    When ``entity_filter`` is provided, expansion is restricted to entities
    whose normalised name matches one of the supplied values — the query-side
    NER pass is skipped and only the explicit filter is used.

    When ``out_entity_names`` is supplied it is populated (in place) with the
    deduped entity names discovered from the query, so callers such as context
    assembly can render an entity summary without re-running NER.

    Steps:
    1. Run standard vector search
    2. Extract entities from query (spaCy + GLiNER, no LLM for latency) or
       honour ``entity_filter`` when set
    3. Look up matching entities in entity_store
    4. Expand via entity_mentions to related memory_ids (capped at 50 per entity)
    5. Fetch those memories
    6. Fuse via Reciprocal Rank Fusion
    """
    # Step 1: Vector search (always runs). Over-fetch without the access bump
    # so candidates that never reach the returned slice are not touched; the
    # final slice is bumped once below.
    vector_results = await provider.search(
        query_embedding,
        user_id=user_id,
        limit=limit * 3,
        weights=weights,
        bump_access=False,
    )

    # If no entity store, return vector-only results
    if entity_store is None:
        result = vector_results[:limit]
        await _bump_returned(provider, result)
        return result

    # Step 2: Extract entities from query (or honour explicit filter)
    entity_results = await _get_entity_memories(
        query_text,
        query_embedding,
        user_id,
        entity_store,
        provider,
        embedder,
        entity_filter=entity_filter,
        out_entity_names=out_entity_names,
    )

    if not entity_results:
        result = vector_results[:limit]
        await _bump_returned(provider, result)
        return result

    # Step 3: RRF fusion
    fused = _fuse_rrf(vector_results, entity_results, limit)
    await _bump_returned(provider, fused)
    return fused


async def _bump_returned(
    provider: MemoryProvider, results: list[ScoredMemory]
) -> None:
    """Record a single access bump on exactly the returned memories.

    Routes through the batched ``bump_access`` so context injection issues one
    UPDATE for the whole slice and never writes a ``memory_history`` row.
    """
    ids = [sm.memory.memory_id for sm in results]
    if not ids:
        return
    try:
        await provider.bump_access(ids)
    except Exception:
        # Bookkeeping must never fail retrieval; the ranking is still valid.
        logger.debug("access bump skipped for %d memories", len(ids))
        return
    now = datetime.now(timezone.utc)
    for sm in results:
        sm.memory.access_count += 1
        sm.memory.last_accessed = max(now, sm.memory.last_accessed)


async def _get_entity_memories(
    query_text: str,
    query_embedding: list[float],
    user_id: uuid.UUID,
    entity_store: "EntityStore",
    provider: MemoryProvider,
    embedder: EmbeddingClient | None = None,
    entity_filter: list[str] | None = None,
    out_entity_names: list[str] | None = None,
) -> list[ScoredMemory]:
    """Extract entities from query (or honour filter) and expand to related memories.

    The query embedding is supplied by the caller and reused for the
    embedding-similarity entity fallback — the query is never re-embedded.

    When ``out_entity_names`` is supplied it is filled with the deduped names
    of the entities discovered from the query, in discovery order.
    """
    raw_entities: list[_SimpleEntity] = []

    # Explicit filter: skip NER, honour the caller's restriction verbatim.
    if entity_filter is not None:
        for name in entity_filter:
            if not name or not name.strip():
                continue
            # Try every known entity_type — caller does not have to specify.
            for entity_type in (
                "person", "organization", "product", "brand", "location", "concept",
            ):
                found = await entity_store.find_by_name(user_id, name, entity_type)
                if found is not None:
                    raw_entities.append(
                        _SimpleEntity(name=name, entity_type=entity_type)
                    )
                    break
        if not raw_entities and embedder is not None:
            for name in entity_filter:
                try:
                    name_emb = await embedder.embed(name)
                    cands = await entity_store.find_by_embedding(
                        user_id, name_emb, threshold=0.80, limit=3
                    )
                    for ent in cands:
                        raw_entities.append(
                            _SimpleEntity(
                                name=ent.entity_name, entity_type=ent.entity_type
                            )
                        )
                except Exception:
                    continue
    else:
        # NER extraction is CPU-bound and synchronous — run it off the event
        # loop. Graceful degradation when the extractors are not installed.
        try:
            from mnemosyne.pipeline.ner.spacy_extractor import extract_entities_spacy
            raw_entities.extend(
                await asyncio.to_thread(extract_entities_spacy, query_text)
            )
        except Exception:
            pass

        try:
            from mnemosyne.pipeline.ner.gliner_extractor import extract_entities_gliner
            raw_entities.extend(
                await asyncio.to_thread(extract_entities_gliner, query_text)
            )
        except Exception:
            pass

        if not raw_entities:
            # Simple name-based fallback: each word long enough may be an entity.
            for word in query_text.split():
                if len(word) < 3:
                    continue
                for entity_type in (
                    "person", "organization", "product", "brand", "location",
                ):
                    found = await entity_store.find_by_name(user_id, word, entity_type)
                    if found is not None:
                        raw_entities.append(
                            _SimpleEntity(name=word, entity_type=entity_type)
                        )
                        break

        if not raw_entities:
            # Last resort: embedding similarity against registered entities.
            # Reuse the query embedding supplied by the caller — no re-embed.
            try:
                similar_entities = await entity_store.find_by_embedding(
                    user_id, query_embedding, threshold=0.80, limit=5
                )
                for ent in similar_entities:
                    raw_entities.append(
                        _SimpleEntity(
                            name=ent.entity_name, entity_type=ent.entity_type
                        )
                    )
            except Exception:
                pass

    if out_entity_names is not None:
        seen_names: set[str] = set()
        for raw in raw_entities:
            key = raw.name.strip().lower()
            if key and key not in seen_names:
                seen_names.add(key)
                out_entity_names.append(raw.name)

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

    # Batch-fetch active memories in one round-trip; get_by_ids already filters
    # invalidated rows in SQL. Preserve mention order for stable RRF ranks.
    fetched = await provider.get_by_ids(all_memory_ids)
    by_id = {mem.memory_id: mem for mem in fetched}
    return [
        ScoredMemory(
            memory=by_id[mid],
            score=0.0,  # score replaced by RRF fusion
            score_breakdown={"entity_expanded": 1.0},
        )
        for mid in all_memory_ids
        if mid in by_id
    ]


def _fuse_rrf(
    vector_results: list[ScoredMemory],
    entity_results: list[ScoredMemory],
    limit: int,
) -> list[ScoredMemory]:
    """Fuse the vector and entity-expanded lists with Reciprocal Rank Fusion.

    Delegates the core fusion to :func:`mnemosyne.retrieval.fusion.fuse_rrf`
    and relabels the generic ``in_left``/``in_right`` provenance flags to the
    domain-specific ``in_vector``/``in_entity`` keys callers expect.
    """
    fused = fuse_rrf(vector_results, entity_results, limit)
    for sm in fused:
        bd = sm.score_breakdown
        bd["in_vector"] = bd.pop("in_left", 0.0)
        bd["in_entity"] = bd.pop("in_right", 0.0)
    return fused


class _SimpleEntity:
    """Minimal entity placeholder used when NER is unavailable."""

    __slots__ = ("name", "entity_type")

    def __init__(self, name: str, entity_type: str) -> None:
        self.name = name
        self.entity_type = entity_type
