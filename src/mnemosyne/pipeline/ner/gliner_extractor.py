from __future__ import annotations

import logging

from mnemosyne.pipeline.ner.spacy_extractor import RawEntity

logger = logging.getLogger(__name__)

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from gliner import GLiNER
            _MODEL = GLiNER.from_pretrained("urchade/gliner_base")
            logger.info("GLiNER model loaded: urchade/gliner_base")
        except (ImportError, OSError) as exc:
            logger.warning("GLiNER not available: %s", exc)
            return None
    return _MODEL


# Default domain-specific labels
DEFAULT_LABELS = ["brand", "product_category", "material", "style", "color", "feature"]


def extract_entities_gliner(
    text: str,
    labels: list[str] | None = None,
    threshold: float = 0.5,
) -> list[RawEntity]:
    """Extract domain-specific entities using GLiNER zero-shot NER."""
    model = _get_model()
    if model is None:
        return []

    use_labels = labels or DEFAULT_LABELS
    predictions = model.predict_entities(text, use_labels, threshold=threshold)

    entities = []
    for pred in predictions:
        entities.append(RawEntity(
            name=pred["text"],
            entity_type=pred["label"],
            mention_text=pred["text"],
            context=text[:200],  # truncated context
            source="gliner",
        ))
    return entities
