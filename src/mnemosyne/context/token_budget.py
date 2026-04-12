from __future__ import annotations

import tiktoken

_ENC = tiktoken.encoding_for_model("gpt-4")


class TokenBudget:
    """Tracks remaining token budget and handles truncation."""

    def __init__(self, max_tokens: int, encoding: tiktoken.Encoding | None = None):
        self._encoding = encoding or _ENC
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
