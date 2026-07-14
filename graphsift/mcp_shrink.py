"""MCP tool description compression — shrink tool descriptions for token efficiency.

Inspired by caveman-shrink that wraps MCP servers to compress tool descriptions.
Reduces description tokens by 40-60% while preserving essential parameter info.

Usage::
    from graphsift.mcp_shrink import shrink_description, shrink_tools
    compressed = shrink_description("Long verbose description here")
"""

from __future__ import annotations

import re
from typing import Any

# Words/phrases to remove from descriptions
_FILLER_RE = re.compile(
    r"\b(?:"
    r"please|kindly|note that|it should be noted that|"
    r"this function|this tool|this method|"
    r"allows you to|enables you to|lets you|"
    r"used for|useful for|designed for|"
    r"in order to|for the purpose of|"
    r"you can|you may|you might want to|"
    r"essentially|basically|actually|simply|"
    r"importantly|crucially|significantly|"
    r"effectively|efficiently|easily"
    r")\b",
    re.IGNORECASE,
)


def shrink_description(description: str, max_words: int = 30) -> str:
    """Compress a single tool description string.

    Strips filler words, shortens sentences, preserves parameter meaning.

    Args:
        description: Original tool description.
        max_words: Maximum word count after compression (default 30).

    Returns:
        Compressed description.
    """
    # Remove filler phrases
    text = _FILLER_RE.sub("", description)

    # Remove parenthetical asides
    text = re.sub(r"\([^)]*\)", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to max_words if needed
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]) + "."

    return text


def shrink_tools(tools: list[dict[str, Any]], max_desc_words: int = 30) -> list[dict[str, Any]]:
    """Compress descriptions in a list of MCP tool definitions.

    Each tool dict should have at least a ``description`` key.
    Preserves all other keys unchanged.

    Args:
        tools: List of MCP tool dicts (name, description, inputSchema).
        max_desc_words: Max words per compressed description.

    Returns:
        Same list with compressed descriptions.
    """
    for tool in tools:
        if "description" in tool:
            tool["description"] = shrink_description(
                tool["description"], max_words=max_desc_words
            )
    return tools


def shrink_tool_registry(registry: dict[str, dict[str, Any]], max_desc_words: int = 30) -> dict[str, dict[str, Any]]:
    """Compress descriptions in a graphsift _TOOLS-style registry.

    Each entry should have a ``description`` key with a string value.

    Args:
        registry: Dict of tool name -> tool spec (with 'description').
        max_desc_words: Max words per compressed description.

    Returns:
        Same dict with compressed descriptions.
    """
    for name, spec in registry.items():
        if "description" in spec and isinstance(spec["description"], str):
            spec["description"] = shrink_description(
                spec["description"], max_words=max_desc_words
            )
    return registry
