from __future__ import annotations

import tiktoken
import pytest

from mnemosyne.context.token_budget import TokenBudget


@pytest.fixture
def enc() -> tiktoken.Encoding:
    return tiktoken.encoding_for_model("gpt-4")


def test_full_text_fits(enc: tiktoken.Encoding) -> None:
    text = "hello world"
    token_count = len(enc.encode(text))
    budget = TokenBudget(max_tokens=token_count + 10, encoding=enc)
    fitted, used = budget.consume(text)
    assert fitted == text
    assert used == token_count
    assert budget.used == token_count


def test_truncation(enc: tiktoken.Encoding) -> None:
    # Build a text longer than the budget
    text = "alpha beta gamma delta epsilon"
    full_tokens = enc.encode(text)
    assert len(full_tokens) > 3, "need at least 4 tokens for this test"
    budget = TokenBudget(max_tokens=3, encoding=enc)
    fitted, used = budget.consume(text)
    assert used == 3
    assert budget.used == 3
    # Decoded text is a valid token-boundary prefix of the original
    assert enc.decode(enc.encode(text)[:3]) == fitted


def test_remaining_decreases(enc: tiktoken.Encoding) -> None:
    budget = TokenBudget(max_tokens=100, encoding=enc)
    assert budget.remaining == 100
    budget.consume("hello")
    assert budget.remaining < 100
    assert budget.remaining == 100 - budget.used


def test_empty_text(enc: tiktoken.Encoding) -> None:
    budget = TokenBudget(max_tokens=50, encoding=enc)
    fitted, used = budget.consume("")
    assert fitted == ""
    assert used == 0
    assert budget.used == 0
    assert budget.remaining == 50


def test_fits_check(enc: tiktoken.Encoding) -> None:
    short_text = "hi"
    long_text = " ".join(["word"] * 200)
    short_tokens = len(enc.encode(short_text))

    budget = TokenBudget(max_tokens=short_tokens, encoding=enc)
    assert budget.fits(short_text) is True
    assert budget.fits(long_text) is False


def test_remaining_never_negative(enc: tiktoken.Encoding) -> None:
    budget = TokenBudget(max_tokens=2, encoding=enc)
    budget.consume("alpha beta gamma delta")
    # After exhausting budget, remaining should be 0, not negative
    assert budget.remaining == 0
    # Additional consume yields no tokens
    _, used = budget.consume("more text here")
    assert used == 0
