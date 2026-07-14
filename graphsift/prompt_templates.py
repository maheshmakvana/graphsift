"""Prompt templates for production coding — fix/add/refactor.

Each template specifies exact JSON output, reducing both token count
and hallucination risk by constraining the response format.

Usage::
    from graphsift.prompt_templates import FixBugTemplate
    prompt = FixBugTemplate().render(bug="...", file="...", line=42)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FixBugTemplate:
    """Template for bug fix requests — structured, terse, verifiable."""

    def render(
        self,
        bug: str,
        file: str,
        line: Optional[int] = None,
        expected: str = "",
        actual: str = "",
    ) -> str:
        lines = [f"BUG: {bug}"]
        if file:
            lines.append(f"FILE: {file}" + (f":{line}" if line else ""))
        if expected:
            lines.append(f"EXPECTED: {expected}")
        if actual:
            lines.append(f"ACTUAL: {actual}")
        lines.append("")
        lines.append("Output ONLY valid JSON:")
        lines.append('{"root_cause": "...", "fix": "...", "file": "...", "line": N, "test_change": ""}')
        return "\n".join(lines)


@dataclass
class AddFeatureTemplate:
    """Template for feature addition — spec-driven, diff-focused."""

    def render(
        self,
        feature: str,
        files: Optional[list[str]] = None,
        acceptance_criteria: Optional[list[str]] = None,
    ) -> str:
        lines = [f"FEATURE: {feature}"]
        if files:
            lines.append(f"FILES: {', '.join(files)}")
        if acceptance_criteria:
            lines.append("ACCEPTANCE:")
            for ac in acceptance_criteria:
                lines.append(f"  - {ac}")
        lines.append("")
        lines.append("Output ONLY valid JSON:")
        lines.append(
            '{"approach": "...", "changes": [{"file": "...", "diff": "..."}], "tests": ["..."]}'
        )
        return "\n".join(lines)


@dataclass
class RefactorTemplate:
    """Template for refactoring — behavior-preserving, impact-aware."""

    def render(
        self,
        target: str,
        goal: str = "",
        files: Optional[list[str]] = None,
    ) -> str:
        lines = [f"REFACTOR: {target}"]
        if goal:
            lines.append(f"GOAL: {goal}")
        if files:
            lines.append(f"FILES: {', '.join(files)}")
        lines.append("")
        lines.append(
            "CONSTRAINT: Behavior must not change — "
            "no API signature changes without explicit approval."
        )
        lines.append("")
        lines.append("Output ONLY valid JSON:")
        lines.append(
            '{"changes": [{"file": "...", "before": "...", "after": "..."}], '
            '"verification": "...", "risk": "low|medium|high"}'
        )
        return "\n".join(lines)


def get_template(name: str):
    """Return a template by name: 'fix', 'add', 'refactor'."""
    registry = {
        "fix": FixBugTemplate(),
        "add": AddFeatureTemplate(),
        "refactor": RefactorTemplate(),
    }
    tpl = registry.get(name)
    if tpl is None:
        raise ValueError(
            f"Unknown template: {name}. Options: {list(registry)}"
        )
    return tpl
