from __future__ import annotations

from mnemosyne.context.assembly import ContextBlock


def build_system_prompt_memory_block(context_block: ContextBlock) -> str:
    """Format a ContextBlock into a system prompt section.

    Returns an empty string if the context block has no content,
    so callers can skip injection when there's nothing to remember.
    """
    if not context_block.text.strip():
        return ""

    return (
        "## What you remember about this user\n\n"
        f"{context_block.text}\n"
    )
