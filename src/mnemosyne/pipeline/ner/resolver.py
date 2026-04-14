from __future__ import annotations

import logging
import uuid

from mnemosyne.db.models.entity import Entity, EntityMention
from mnemosyne.db.repositories.entity import EntityStore
from mnemosyne.embedding.base import EmbeddingClient
from mnemosyne.llm.base import LLMClient
from mnemosyne.pipeline.ner.spacy_extractor import RawEntity

logger = logging.getLogger(__name__)


async def resolve_entities(
    raw_entities: list[RawEntity],
    user_id: uuid.UUID,
    memory_id: uuid.UUID,
    entity_store: EntityStore,
    embedder: EmbeddingClient,
    llm_client: LLMClient | None = None,
) -> list[Entity]:
    """Resolve raw entities against the entity store.

    For each RawEntity:
    1. Normalize name (lowercase, strip)
    2. Exact match lookup in entity_store
    3. If no exact match, embed name and search by similarity (>= 0.85)
    4. If multiple fuzzy matches, use LLM disambiguation (if available)
    5. Create or update entity, add EntityMention
    """
    resolved = []

    for raw in raw_entities:
        # 1. Exact match
        existing = await entity_store.find_by_name(user_id, raw.name, raw.entity_type)

        if existing is not None:
            # Update existing entity
            existing.mention_count += 1
            if memory_id not in existing.source_memory_ids:
                existing.source_memory_ids.append(memory_id)
            await entity_store.upsert_entity(existing)

            # Add mention
            mention = EntityMention(
                entity_id=existing.entity_id,
                memory_id=memory_id,
                mention_text=raw.mention_text,
                context=raw.context,
            )
            await entity_store.add_mention(mention)
            resolved.append(existing)
            continue

        # 2. Embedding similarity search
        try:
            name_embedding = await embedder.embed(raw.name)
            similar = await entity_store.find_by_embedding(
                user_id, name_embedding, threshold=0.85, limit=5
            )
        except Exception:
            similar = []

        if similar:
            # 3. If single high-confidence match, use it
            if len(similar) == 1:
                best = similar[0]
                best.mention_count += 1
                if memory_id not in best.source_memory_ids:
                    best.source_memory_ids.append(memory_id)
                await entity_store.upsert_entity(best)
                mention = EntityMention(
                    entity_id=best.entity_id,
                    memory_id=memory_id,
                    mention_text=raw.mention_text,
                    context=raw.context,
                )
                await entity_store.add_mention(mention)
                resolved.append(best)
                continue

            # 4. Multiple matches — LLM disambiguation if available
            if llm_client and len(similar) > 1:
                best = await _llm_disambiguate(raw.name, similar, llm_client)
                if best is not None:
                    best.mention_count += 1
                    await entity_store.upsert_entity(best)
                    mention = EntityMention(
                        entity_id=best.entity_id,
                        memory_id=memory_id,
                        mention_text=raw.mention_text,
                        context=raw.context,
                    )
                    await entity_store.add_mention(mention)
                    resolved.append(best)
                    continue

        # 5. Create new entity
        try:
            entity_embedding = await embedder.embed(raw.name)
        except Exception:
            entity_embedding = None

        new_entity = Entity(
            user_id=user_id,
            entity_name=raw.name,
            entity_type=raw.entity_type,
            embedding=entity_embedding,
            source_memory_ids=[memory_id],
            mention_count=1,
        )
        entity_id = await entity_store.upsert_entity(new_entity)
        new_entity.entity_id = entity_id

        mention = EntityMention(
            entity_id=entity_id,
            memory_id=memory_id,
            mention_text=raw.mention_text,
            context=raw.context,
        )
        await entity_store.add_mention(mention)
        resolved.append(new_entity)

    return resolved


async def _llm_disambiguate(
    query_name: str,
    candidates: list[Entity],
    llm_client: LLMClient,
) -> Entity | None:
    """Use LLM to pick the best matching entity from candidates."""
    candidate_descriptions = "\n".join(
        f"- {c.entity_name} (type: {c.entity_type}, mentions: {c.mention_count})"
        for c in candidates
    )
    prompt = (
        f"Is '{query_name}' the same as any of these entities?\n"
        f"{candidate_descriptions}\n\n"
        f"Respond with ONLY the entity name that matches, or 'NONE' if no match."
    )
    try:
        response = await llm_client.complete(prompt)
        response = response.strip()
        if response.upper() == "NONE":
            return None
        for candidate in candidates:
            if candidate.entity_name.lower() == response.lower():
                return candidate
            if candidate.normalized_name == response.strip().lower():
                return candidate
    except Exception as exc:
        logger.warning("LLM disambiguation failed: %s", exc)
    return None
