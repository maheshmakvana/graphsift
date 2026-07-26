"""Evidence citation checker — validates file:line references in responses.

Scans generated text for file references and verifies they point to real
files with valid line numbers. Catches hallucinated file references before
they compound into downstream errors.

The checker searches for patterns like ``src/main.py:42`` or
``'src/main.py:42'`` in text, resolves them relative to a project root
(including common source directories), and verifies the referenced file
and line number exist on disk.

Usage::
    checker = EvidenceChecker(project_root="/path/to/repo")
    violations = checker.check_response(response_text)

    for citation in violations:
        if not citation.valid:
            print(f"Hallucinated reference: {citation.raw} — {citation.error}")
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Citation:
    """A single file:line citation found in text."""
    raw: str
    file_path: str
    line: Optional[int] = None
    valid: bool = False
    error: str = ""


class EnforceMode(str, enum.Enum):
    """How to handle unverifiable claims in enforce_text()."""
    MARK = "mark"        # Append [UNKNOWN] after unverifiable claims
    STRIP = "strip"      # Remove unverifiable claims from text
    REPORT = "report"    # Return report without modifying text
    ENFORCE = "enforce"  # Auto-correct: strip invalid AND return report


@dataclass
class EnforceResult:
    """Result of evidence enforcement on a text."""
    text: str
    verified_claims: list[Citation] = field(default_factory=list)
    unverified_claims: list[Citation] = field(default_factory=list)
    total_claims: int = 0
    unverified_count: int = 0

    @property
    def summary(self) -> str:
        return (
            f"EnforceResult: {self.total_claims} claims, "
            f"{self.unverified_count} unverified"
        )


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evidence Checker
# ---------------------------------------------------------------------------


class EvidenceChecker:
    """Validates file:line citations against the actual filesystem."""

    def __init__(self, project_root: str = "") -> None:
        self.project_root = Path(project_root or ".").resolve()

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Enforce
    # ------------------------------------------------------------------

    def enforce_text(
        self,
        text: str,
        mode: EnforceMode = EnforceMode.MARK,
    ) -> EnforceResult:
        """Verify every file:line claim and enforce citation rules.

        Steps:
        1. Find all citations via ``check_response()``.
        2. For each citation:
           - Verified → keep as-is.
           - Unverified → handle per mode.
        3. Return ``EnforceResult`` with processed text and counts.

        Args:
            text: Text containing file:line claims.
            mode: ``EnforceMode`` (MARK, STRIP, REPORT, or ENFORCE).

        Returns:
            ``EnforceResult`` with modified text and claim breakdown.
        """
        citations = self.check_response(text)
        verified: list[Citation] = [c for c in citations if c.valid]
        unverified: list[Citation] = [c for c in citations if not c.valid]

        result_text = text

        if mode == EnforceMode.MARK:
            # Append [UNKNOWN] at the end of each unverified claim
            for c in unverified:
                marker = f" [UNKNOWN]"
                # Insert marker after the raw citation text
                result_text = result_text.replace(c.raw, c.raw + marker, 1)

        elif mode in (EnforceMode.STRIP, EnforceMode.ENFORCE):
            # Remove unverified citation text from output
            for c in unverified:
                result_text = result_text.replace(c.raw, "", 1)
            # Clean up double spaces left by removals
            result_text = re.sub(r"  +", " ", result_text)
            # Clean up "in " at end of sentences
            result_text = re.sub(r"\bin\s+\.", ".", result_text)

        # REPORT mode: leave text untouched

        return EnforceResult(
            text=result_text,
            verified_claims=verified,
            unverified_claims=unverified,
            total_claims=len(citations),
            unverified_count=len(unverified),
        )

    # ------------------------------------------------------------------
    # Validation internals
    # ------------------------------------------------------------------

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


__all__ = [
    "Citation",
    "EnforceMode",
    "EnforceResult",
    "EvidenceChecker",
]
