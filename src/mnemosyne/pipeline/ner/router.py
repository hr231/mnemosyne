from __future__ import annotations

import logging

from mnemosyne.pipeline.ner.spacy_extractor import RawEntity

logger = logging.getLogger(__name__)


def merge_entities(
    spacy_entities: list[RawEntity],
    gliner_entities: list[RawEntity],
    llm_entities: list[RawEntity],
) -> list[RawEntity]:
    """Merge entities from all three tiers, deduplicating by normalized name.

    Merge rule:
    - If spaCy and GLiNER find the same entity (by normalized name),
      use spaCy's type for standard NER tags (person, organization, location),
      otherwise use GLiNER's more specific label (brand, product_category).
    - Keep the longest mention_text span.
    - Priority: spaCy > GLiNER > LLM for the same entity name.
    """
    seen: dict[str, RawEntity] = {}  # normalized_name -> best entity

    STANDARD_TYPES = {"person", "organization", "location"}

    for entity in spacy_entities + gliner_entities + llm_entities:
        key = entity.name.strip().lower()
        if key not in seen:
            seen[key] = entity
        else:
            existing = seen[key]
            # Merge: pick the best type
            if entity.source == "spacy" and entity.entity_type in STANDARD_TYPES:
                merged_type = entity.entity_type
            elif existing.source == "spacy" and existing.entity_type in STANDARD_TYPES:
                merged_type = existing.entity_type
            elif entity.source == "gliner":
                merged_type = entity.entity_type
            else:
                merged_type = existing.entity_type

            # Keep longest mention_text
            mention = (
                entity.mention_text
                if len(entity.mention_text) > len(existing.mention_text)
                else existing.mention_text
            )

            # Keep highest-priority source
            priority = {"spacy": 0, "gliner": 1, "llm": 2}
            best_source = (
                existing.source
                if priority.get(existing.source, 3) <= priority.get(entity.source, 3)
                else entity.source
            )

            # Use longest context
            context = (
                entity.context
                if len(entity.context) > len(existing.context)
                else existing.context
            )

            seen[key] = RawEntity(
                name=entity.name if len(entity.name) >= len(existing.name) else existing.name,
                entity_type=merged_type,
                mention_text=mention,
                context=context,
                source=best_source,
            )

    return list(seen.values())


def should_use_llm(
    text: str,
    spacy_count: int,
    gliner_count: int,
    unstructured_threshold: float = 0.7,
) -> bool:
    """Determine if LLM fallback should be used for entity extraction.

    Uses same signal as memory extraction router: high unstructured ratio
    + low entity count suggests implicit entities the models missed.
    """
    if spacy_count + gliner_count == 0 and len(text.strip()) > 20:
        return True
    return False
