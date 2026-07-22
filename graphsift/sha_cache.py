"""SHA-256 cache for incremental graph builds.

Persists file content hashes + stat metadata between builds so
unchanged files can be skipped during re-parsing without re-reading
content from disk.

Cache entry format::

    {"sha": "<sha256 hex>", "mtime": 1234567890.0, "size": 1234}

The cache is stored alongside the SQLite DB in
``~/.graphsift/<repo-hash>/sha_cache.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def sha_cache_path(root: str) -> str:
    """Return the path to the SHA cache JSON file for *root*.

    The path mirrors the DB location::

        ~/.graphsift/<sha1-of-root[:12]>/sha_cache.json
    """
    key = hashlib.sha1(root.encode()).hexdigest()[:12]
    cache_dir = Path.home() / ".graphsift" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / "sha_cache.json")


def load_sha_cache(root: str) -> dict[str, Any]:
    """Load the SHA cache from disk.

    Args:
        root: Repo root path (used to determine cache location).

    Returns:
        Dict mapping file path → SHA-256 hex digest, or empty dict if
        no cache file exists or it's corrupt.
    """
    path = sha_cache_path(root)
    if not os.path.exists(path):
        logger.debug("graphsift: no SHA cache at %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("graphsift: SHA cache is not a dict, ignoring")
            return {}
        logger.debug("graphsift: loaded SHA cache with %d entries", len(data))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("graphsift: failed to load SHA cache: %s", exc)
        return {}


def save_sha_cache(root: str, cache: dict[str, Any]) -> None:
    """Save the SHA cache to disk.

    Args:
        root: Repo root path.
        cache: Dict mapping file path → SHA-256 hex digest (str v1) or
               dict with sha/mtime/size (v2).
    """
    path = sha_cache_path(root)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        logger.debug("graphsift: saved SHA cache with %d entries to %s", len(cache), path)
    except OSError as exc:
        logger.warning("graphsift: failed to save SHA cache: %s", exc)


def compute_sha(source: str) -> str:
    """Compute the SHA-256 hex digest of source text.

    Args:
        source: File source text.

    Returns:
        SHA-256 hex digest string.
    """
    return hashlib.sha256(source.encode(errors="replace")).hexdigest()


def stat_match(path: str, cache: dict[str, Any]) -> bool:
    """Fast check if a file is unchanged using mtime + size (no content read).

    Only works with the dict-based cache format (v2+).  Returns False
    for entries in the legacy str-only format so callers fall back to
    content-hash checking.

    Args:
        path: Absolute file path.
        cache: Loaded SHA cache dict.

    Returns:
        True if the file's mtime and size match the cached values.
    """
    entry = cache.get(path)
    if not entry or not isinstance(entry, dict):
        return False
    try:
        st = os.stat(path)
    except OSError:
        return False
    return entry.get("mtime") == st.st_mtime and entry.get("size") == st.st_size


def make_cache_entry(path: str, source: str) -> dict[str, Any]:
    """Build a cache dict entry with SHA-256, mtime, and file size.

    Args:
        path: Absolute file path.
        source: File source text.

    Returns:
        Dict with keys ``sha``, ``mtime``, ``size``.
    """
    st = os.stat(path)
    return {
        "sha": compute_sha(source),
        "mtime": st.st_mtime,
        "size": st.st_size,
    }


def has_changed(path: str, source: str, cache: dict[str, Any]) -> bool:
    """Check if a file has changed since it was cached.

    Handles both the legacy str-only format (v1) and the current
    dict-with-metadata format (v2).

    Args:
        path: File path (cache key).
        source: Current file source text.
        cache: Loaded SHA cache dict.

    Returns:
        True if the file is new or modified; False if unchanged.
    """
    entry = cache.get(path)
    if entry is None:
        return True  # new file
    sha = compute_sha(source)
    if isinstance(entry, dict):
        return sha != entry.get("sha")
    return sha != entry  # legacy str format
