from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded: en_core_web_sm")
        except (ImportError, OSError) as exc:
            logger.warning("spaCy not available: %s", exc)
            return None
    return _NLP


@dataclass
class RawEntity:
    name: str
    entity_type: str
    mention_text: str
    context: str
    source: str  # "spacy", "gliner", "llm"


# Map spaCy NER labels to our entity types
_SPACY_TYPE_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "PRODUCT": "product",
    "FAC": "location",
    "NORP": "organization",
}


def extract_entities_spacy(text: str) -> list[RawEntity]:
    """Extract entities from text using spaCy NER."""
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entity_type = _SPACY_TYPE_MAP.get(ent.label_)
        if entity_type is None:
            continue
        # Get surrounding sentence for context
        sent = ent.sent.text if ent.sent else text
        entities.append(RawEntity(
            name=ent.text,
            entity_type=entity_type,
            mention_text=ent.text,
            context=sent.strip(),
            source="spacy",
        ))
    return entities
