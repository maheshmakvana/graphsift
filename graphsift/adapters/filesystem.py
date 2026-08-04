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
import os
from pathlib import Path

from graphsift.read_cache import SafeFileIO

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".mjs", ".ts", ".tsx",
    ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".rb", ".php",
}

_DEFAULT_EXCLUDES = {
    # Non-dot dependency dirs (dot dirs like .next, .git are auto-skipped)
    "node_modules", "vendor", "Pods", "bower_components", "jspm_packages",
    "dist", "build", "target", "out", "cdk.out",
    "__pycache__", "*.egg-info", "coverage", "htmlcov",
}


def _prune_predicates(exclude_dirs: set[str]):
    """Split an exclude set into exact names and glob suffixes (e.g. ``*.egg-info``).

    Returns ``(exact, suffixes)`` where ``suffixes`` are the parts after ``*``.
    """
    exact = {e for e in exclude_dirs if not e.startswith("*")}
    suffixes = {e[1:] for e in exclude_dirs if e.startswith("*")}
    return exact, suffixes


def _iter_source_paths(
    root_path: Path,
    extensions: set[str],
    exclude_dirs: set[str],
):
    """Yield source file paths under *root_path*.

    Uses ``os.walk`` and prunes hidden (dot) and excluded directories *while
    descending*, so huge tooling trees (``.venv``, ``.git``, ``node_modules``)
    are never traversed.  ``rglob`` was the previous implementation and walked
    every file in those trees before filtering — e.g. a 45k-file ``.venv``
    added ~33s to every build.
    """
    exact_excl, suffix_excl = _prune_predicates(exclude_dirs)
    exts = extensions

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune hidden + excluded directories in-place so os.walk never descends
        keep: list[str] = []
        for d in dirnames:
            if d.startswith("."):
                continue
            if d in exact_excl:
                continue
            if any(d.endswith(sfx) for sfx in suffix_excl):
                continue
            keep.append(d)
        dirnames[:] = keep

        for fname in filenames:
            if Path(fname).suffix.lower() in exts:
                yield os.path.join(dirpath, fname)


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

    for fp in _iter_source_paths(root_path, exts, excl):
        try:
            if os.path.getsize(fp) > max_file_bytes:
                logger.debug("graphsift: skipping large file %s", fp)
                continue
            source_map[fp] = SafeFileIO.read(fp, encoding=encoding)
        except OSError as exc:
            logger.warning(
                "graphsift: could not read file",
                extra={"path": str(fp), "error": str(exc)},
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
    root_path = Path(root).resolve()

    return list(_iter_source_paths(root_path, exts, excl))


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
            result[p] = SafeFileIO.read(Path(p), encoding=encoding)
        except OSError as exc:
            logger.warning(
                "graphsift: could not read changed file",
                extra={"path": p, "error": str(exc)},
            )
    return result
