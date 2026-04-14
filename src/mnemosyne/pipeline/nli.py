from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None


@dataclass
class NLIResult:
    entailment: float
    contradiction: float
    neutral: float


def _load_nli_model():
    """Lazy-load the DeBERTa NLI model. Raises ImportError if torch/transformers not installed."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Contradiction detection requires torch + transformers. "
            "Install: pip install torch --index-url https://download.pytorch.org/whl/cpu && "
            "pip install transformers sentencepiece"
        ) from exc

    model_name = "cross-encoder/nli-deberta-v3-base"
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(model_name)
    _MODEL.eval()
    logger.info("NLI model loaded: %s", model_name)
    return _MODEL, _TOKENIZER


def predict_nli(text_a: str, text_b: str) -> NLIResult:
    """Run NLI prediction on a text pair. Returns entailment/contradiction/neutral scores."""
    model, tokenizer = _load_nli_model()

    import torch

    inputs = tokenizer(text_a, text_b, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0]

    # Resolve label order from model config (not assumed to be fixed)
    id2label = model.config.id2label
    result: dict[str, float] = {}
    for idx, label in id2label.items():
        result[label.lower()] = float(scores[idx])

    return NLIResult(
        entailment=result.get("entailment", 0.0),
        contradiction=result.get("contradiction", 0.0),
        neutral=result.get("neutral", 0.0),
    )
