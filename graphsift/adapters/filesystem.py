"""Filesystem adapter for graphsift.

Provides helpers for callers to load source files from disk into the
source_map format required by ContextBuilder. The library never opens
files directly — this adapter is caller-supplied I/O.

Example::

    from graphsift.adapters.filesystem import load_source_map, walk_repo

    source_map = load_source_map("./my_repo", extensions=[".py", ".ts"])
    builder.index_files(source_map)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".mjs", ".ts", ".tsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".rb", ".php",
}

_DEFAULT_EXCLUDES = {
    "venv", ".venv", "node_modules", "dist", "build",
    "__pycache__", ".git", ".tox", ".mypy_cache",
    ".pytest_cache", "*.egg-info",
}


def load_source_map(
    root: str,
    extensions: set[str] | None = None,
    exclude_dirs: set[str] | None = None,
    max_file_bytes: int = 500_000,
    encoding: str = "utf-8",
) -> dict[str, str]:
    """Walk a directory tree and load source files into a dict.

    Args:
        root: Root directory path.
        extensions: File extensions to include (defaults to all supported).
        exclude_dirs: Directory names to skip.
        max_file_bytes: Files larger than this are skipped (default 500KB).
        encoding: File encoding (default utf-8).

    Returns:
        Dict mapping absolute file path → source text.
    """
    exts = extensions or _DEFAULT_EXTENSIONS
    excl = exclude_dirs or _DEFAULT_EXCLUDES
    source_map: dict[str, str] = {}
    root_path = Path(root).resolve()

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        # Skip excluded directories
        if any(part in excl for part in path.parts):
            continue
        if path.suffix.lower() not in exts:
            continue
        if path.stat().st_size > max_file_bytes:
            logger.debug("graphsift: skipping large file %s", path)
            continue
        try:
            source_map[str(path)] = path.read_text(encoding=encoding, errors="replace")
        except OSError as exc:
            logger.warning(
                "graphsift: could not read file",
                extra={"path": str(path), "error": str(exc)},
            )

    logger.info(
        "graphsift: loaded source map",
        extra={"root": str(root_path), "files": len(source_map)},
    )
    return source_map


def walk_repo(
    root: str,
    extensions: set[str] | None = None,
    exclude_dirs: set[str] | None = None,
) -> list[str]:
    """Return a list of all source file paths in a repo (no reading).

    Args:
        root: Root directory.
        extensions: File extensions to include.
        exclude_dirs: Directories to skip.

    Returns:
        List of absolute file path strings.
    """
    exts = extensions or _DEFAULT_EXTENSIONS
    excl = exclude_dirs or _DEFAULT_EXCLUDES
    paths: list[str] = []
    root_path = Path(root).resolve()

    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in excl for part in path.parts):
            continue
        if path.suffix.lower() in exts:
            paths.append(str(path))

    return paths


def load_changed_files(
    changed_paths: list[str],
    encoding: str = "utf-8",
) -> dict[str, str]:
    """Load only the changed files into a source map.

    Args:
        changed_paths: List of file paths to read.
        encoding: File encoding.

    Returns:
        Dict mapping path → source text.
    """
    result: dict[str, str] = {}
    for p in changed_paths:
        try:
            result[p] = Path(p).read_text(encoding=encoding, errors="replace")
        except OSError as exc:
            logger.warning(
                "graphsift: could not read changed file",
                extra={"path": p, "error": str(exc)},
            )
    return result
