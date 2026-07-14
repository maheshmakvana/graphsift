"""Evidence citation checker — validates file:line references in responses.

Scans generated text for file references and verifies they point to real
files with valid line numbers. Catches hallucinated file references before
they compound.

Usage::
    checker = EvidenceChecker(project_root="/path/to/repo")
    violations = checker.check_response(response_text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Citation:
    """A single file:line citation found in text."""
    raw: str
    file_path: str
    line: Optional[int] = None
    valid: bool = False
    error: str = ""


_FILE_LINE_RE = re.compile(
    r"""
    (?:
        [`'\"]([^`'\"]+?)        # quoted path
        [:#](\d+)                # colon/at + line number
        |
        (\S+?\.\w+)              # unquoted file path
        [:#](\d+)                # colon/at + line number
    )
    """,
    re.VERBOSE,
)

_SRC_DIRS = {"src", "lib", "app", "graphsift", "tests", "packages"}


class EvidenceChecker:
    """Validates file:line citations against the actual filesystem."""

    def __init__(self, project_root: str = "") -> None:
        self.project_root = Path(project_root or ".").resolve()

    def check_response(self, text: str) -> list[Citation]:
        """Find and validate all file:line citations in *text*.

        Returns a list of Citation objects with validity status.
        """
        citations: list[Citation] = []
        seen: set[str] = set()

        for m in _FILE_LINE_RE.finditer(text):
            quoted_path = m.group(1)
            quoted_line = m.group(2)
            unquoted_path = m.group(3)
            unquoted_line = m.group(4)

            path = quoted_path or unquoted_path or ""
            line = int(quoted_line or unquoted_line or 0)

            key = f"{path}:{line}"
            if key in seen:
                continue
            seen.add(key)

            citation = Citation(raw=m.group(0), file_path=path, line=line)
            self._validate(citation)
            citations.append(citation)

        return citations

    def _validate(self, citation: Citation) -> None:
        """Check if the cited file:line actually exists."""
        # Try direct path first
        candidate = self.project_root / citation.file_path
        if candidate.exists():
            if citation.line:
                citation.valid = self._line_exists(candidate, citation.line)
                if not citation.valid:
                    citation.error = f"Line {citation.line} exceeds file length"
            else:
                citation.valid = True
            return

        # Try relative to src dirs
        for src in _SRC_DIRS:
            candidate = self.project_root / src / citation.file_path
            if candidate.exists():
                if citation.line:
                    citation.valid = self._line_exists(candidate, citation.line)
                    if not citation.valid:
                        citation.error = f"Line {citation.line} exceeds file length"
                else:
                    citation.valid = True
                return

        # Try suffix-based glob match
        matches = list(self.project_root.rglob(citation.file_path))
        if matches:
            candidate = matches[0]
            if citation.line:
                citation.valid = self._line_exists(candidate, citation.line)
                if not citation.valid:
                    citation.error = f"Line {citation.line} exceeds file length"
            else:
                citation.valid = True
            return

        citation.error = "File not found"

    @staticmethod
    def _line_exists(path: Path, line: int) -> bool:
        """Check if *path* has at least *line* lines."""
        try:
            total = sum(1 for _ in open(path, encoding="utf-8"))
            return 1 <= line <= total
        except Exception:
            return False
