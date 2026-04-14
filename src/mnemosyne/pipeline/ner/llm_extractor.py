from __future__ import annotations

import json
import logging

from mnemosyne.llm.base import LLMClient
from mnemosyne.pipeline.ner.spacy_extractor import RawEntity

logger = logging.getLogger(__name__)

ENTITY_PROMPT = """Extract named entities from the following text.
Return a JSON array of objects, each with:
- "name": the entity name
- "type": one of "person", "organization", "product", "brand", "location", "concept"
- "context": the sentence containing the entity

Text: {text}

Respond with ONLY valid JSON array."""


async def extract_entities_llm(text: str, llm_client: LLMClient) -> list[RawEntity]:
    """Extract entities using LLM as fallback."""
    try:
        prompt = ENTITY_PROMPT.format(text=text)
        raw = await llm_client.complete(prompt)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [l for l in cleaned.split("\n") if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        items = json.loads(cleaned)
        if not isinstance(items, list):
            return []

        entities = []
        for item in items:
            if not isinstance(item, dict) or "name" not in item:
                continue
            entities.append(RawEntity(
                name=item["name"],
                entity_type=item.get("type", "concept"),
                mention_text=item["name"],
                context=item.get("context", ""),
                source="llm",
            ))
        return entities
    except Exception as exc:
        logger.warning("LLM entity extraction failed: %s", exc)
        return []
