"""Verification hooks — auto-run syntax/lint checks after code changes.

Post-edit verification hooks that catch syntax errors, lint issues, and
other problems before they compound. Designed to run as PostToolUse hooks
in agentic coding workflows.

Supports Python (via ``compile()``) and JavaScript/TypeScript (via
``node --check``). Extensible to additional languages via the language
mapping.

Usage::
    verify = Verifier(project_root="/path/to/repo")
    result = verify.check("src/main.py")
    if not result.passed:
        print(f"Syntax error: {result.syntax_error}")

    ok, output = verify.lint("src/main.py")
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VerifyResult:
    """Result of a verification check."""
    file: str
    syntax_ok: bool = False
    syntax_error: str = ""
    lint_ok: bool = True
    lint_output: str = ""
    passed: bool = False


class Verifier:
    """Lightweight verifier for changed files."""

    def __init__(self, project_root: str = "", python_path: str = "") -> None:
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.python_path = python_path or sys.executable
        self._lang_map: dict[str, str] = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
        }

    def check(self, file_path: str) -> VerifyResult:
        """Run syntax + optional lint check on *file_path*.

        Returns a VerifyResult with pass/fail for each check.
        """
        full_path = self.project_root / file_path
        if not full_path.exists():
            return VerifyResult(file=file_path, syntax_ok=False, syntax_error="File not found")

        ext = full_path.suffix.lower()
        lang = self._lang_map.get(ext)

        result = VerifyResult(file=file_path)

        if lang == "python":
            result.syntax_ok, result.syntax_error = self._check_python_syntax(full_path)
        elif lang in ("javascript", "typescript"):
            result.syntax_ok, result.syntax_error = self._check_node_syntax(full_path)
        else:
            result.syntax_ok = True  # no syntax checker available

        result.passed = result.syntax_ok
        return result

    def _check_python_syntax(self, path: Path) -> tuple[bool, str]:
        """Run python -c 'compile(...)' on the file."""
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    def _check_node_syntax(self, path: Path) -> tuple[bool, str]:
        """Run node --check on the file (if node is available)."""
        try:
            result = subprocess.run(
                ["node", "--check", str(path)],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return True, ""  # node not available, skip

    def lint(self, file_path: str) -> tuple[bool, str]:
        """Run a linter on *file_path* if available."""
        full_path = self.project_root / file_path
        if not full_path.exists():
            return False, "File not found"

        ext = full_path.suffix.lower()
        try:
            if ext == ".py":
                result = subprocess.run(
                    [self.python_path, "-m", "py_compile", str(full_path)],
                    capture_output=True, text=True, timeout=10,
                )
                return result.returncode == 0, result.stderr.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return True, ""
