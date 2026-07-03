from __future__ import annotations

import functools

import tiktoken

# Default tokenizer. cl100k_base backs gpt-4/gpt-3.5 and is a reasonable
# host-agnostic estimator; override per deployment via ``encoding_name``.
DEFAULT_ENCODING = "cl100k_base"


@functools.lru_cache(maxsize=8)
def get_encoding(name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding by name.

    Falls back to :data:`DEFAULT_ENCODING` when *name* is unknown so a
    misconfigured host tokenizer degrades to estimation rather than raising.
    """
    try:
        return tiktoken.get_encoding(name)
    except (KeyError, ValueError):
        return tiktoken.get_encoding(DEFAULT_ENCODING)


class TokenBudget:
    """Tracks remaining token budget and handles truncation."""

    def __init__(
        self,
        max_tokens: int,
        encoding: tiktoken.Encoding | None = None,
        encoding_name: str = DEFAULT_ENCODING,
    ):
        self._encoding = encoding or get_encoding(encoding_name)
        self._max_tokens = max_tokens
        self._used = 0

    @property
    def remaining(self) -> int:
        return max(0, self._max_tokens - self._used)

    @property
    def used(self) -> int:
        return self._used

    def fits(self, text: str) -> bool:
        return len(self._encoding.encode(text)) <= self.remaining

    def consume(self, text: str) -> tuple[str, int]:
        """Consume as much of *text* as fits in the budget.

        Returns (fitted_text, tokens_used). If the full text fits,
        fitted_text == text. Otherwise it is truncated to the remaining
        budget at a token boundary.
        """
        tokens = self._encoding.encode(text)
        available = self.remaining
        if len(tokens) <= available:
            self._used += len(tokens)
            return text, len(tokens)
        truncated_tokens = tokens[:available]
        fitted = self._encoding.decode(truncated_tokens)
        self._used += available
        return fitted, available
