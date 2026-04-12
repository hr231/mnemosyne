from __future__ import annotations

import pytest

from mnemosyne.context.assembly import ContextBlock
from mnemosyne.integration.prompt_builder import build_system_prompt_memory_block


def test_builds_memory_block():
    block = ContextBlock(text="- User prefers dark mode\n- User is based in Berlin\n", token_count=15)
    result = build_system_prompt_memory_block(block)
    assert result.startswith("## What you remember about this user")
    assert "User prefers dark mode" in result
    assert "User is based in Berlin" in result


def test_empty_context_returns_empty():
    block = ContextBlock(text="", token_count=0)
    result = build_system_prompt_memory_block(block)
    assert result == ""


def test_whitespace_only_returns_empty():
    block = ContextBlock(text="   ", token_count=0)
    result = build_system_prompt_memory_block(block)
    assert result == ""
