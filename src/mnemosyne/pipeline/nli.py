from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None
_MODEL_LOCK = threading.Lock()


@dataclass
class NLIResult:
    entailment: float
    contradiction: float
    neutral: float


def _load_nli_model():
    """Lazy-load the DeBERTa NLI model. Raises ImportError if torch/transformers not installed.

    The load is guarded by a lock with double-checked initialisation so
    concurrent callers (predict_nli runs under ``asyncio.to_thread`` on the
    default thread-pool) cannot race two model downloads/instantiations.
    """
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _TOKENIZER

        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Contradiction detection requires torch + transformers. "
                "Install: pip install torch --index-url https://download.pytorch.org/whl/cpu && "
                "pip install transformers sentencepiece"
            ) from exc

        model_name = "cross-encoder/nli-deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        # Publish the tokenizer first, then the model — callers gate on _MODEL,
        # so it must be assigned last to avoid exposing a half-initialised pair.
        _TOKENIZER = tokenizer
        _MODEL = model
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
