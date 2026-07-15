"""Tool budget system — per-tool line caps to limit token consumption.

Enforces maximum line limits on tool output to prevent context window
overflow. Each tool type has a configurable cap (defaults: bash=80,
read=300, grep=120).

Processing pipeline:
    1. Strip ANSI escape sequences (colored output)
    2. Collapse repeated blank lines (3+ → 2)
    3. Optionally extract structured JSON/XML content
    4. Cap at tool's line limit with head/tail summary notation

Usage::
    budget = ToolBudget()
    capped = budget.apply("bash", long_output)
    capped = budget.apply("read", file_content)
    budget.set_budget("bash", 50)  # tighten budget
"""

from __future__ import annotations

import re
from typing import Optional

# Default per-tool line caps
_DEFAULT_BUDGETS: dict[str, int] = {
    "bash": 80,
    "read": 300,
    "grep": 120,
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

_STRUCTURED_EXTRACTORS: dict[str, str] = {
    "json": r"(\{.*\}|\[.*\])",
    "xml": r"(<[^>]*>.*?</[^>]*>)",
}


class ToolBudget:
    """Per-tool output budget enforcer."""

    def __init__(self, budgets: Optional[dict[str, int]] = None) -> None:
        self.budgets = {**_DEFAULT_BUDGETS, **(budgets or {})}

    def apply(self, tool: str, text: str, extract_structured: bool = False) -> str:
        """Apply budget constraints to *text* for the given *tool*.

        Steps:
        1. Strip ANSI escape sequences
        2. Collapse repeated blank lines
        3. Optionally extract JSON/XML content
        4. Cap at the tool's line limit
        """
        text = _ANSI_RE.sub("", text)
        text = self._collapse_blanks(text)

        if extract_structured:
            extracted = self._try_extract(text)
            if extracted is not None:
                text = extracted

        max_lines = self.budgets.get(tool, 9999)
        lines = text.split("\n")
        if len(lines) > max_lines:
            head = max_lines // 2
            tail = max_lines - head
            omitted = len(lines) - max_lines
            lines = lines[:head] + [f"  ... ({omitted} lines omitted by tool budget) ..."] + lines[-tail:]

        return "\n".join(lines)

    def get_budget(self, tool: str) -> int:
        """Return the line cap for *tool*."""
        return self.budgets.get(tool, 9999)

    def set_budget(self, tool: str, max_lines: int) -> None:
        """Set a custom line cap for *tool*."""
        self.budgets[tool] = max_lines

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _collapse_blanks(text: str) -> str:
        """Replace runs of 3+ blank lines with exactly 2."""
        lines = text.split("\n")
        result: list[str] = []
        blank_run = 0
        for line in lines:
            if line.strip() == "":
                blank_run += 1
                if blank_run <= 2:
                    result.append(line)
            else:
                blank_run = 0
                result.append(line)
        return "\n".join(result)

    @staticmethod
    def _try_extract(text: str) -> Optional[str]:
        """Try to extract JSON or XML structure from text."""
        for pattern in _STRUCTURED_EXTRACTORS.values():
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return m.group(1)
        return None
