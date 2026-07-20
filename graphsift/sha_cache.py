"""SHA-256 cache for incremental graph builds.

Persists file content hashes between builds so unchanged files can be
skipped during re-parsing. The cache is stored alongside the SQLite DB
in ``~/.graphsift/<repo-hash>/sha_cache.json``.
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


def load_sha_cache(root: str) -> dict[str, str]:
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


def save_sha_cache(root: str, cache: dict[str, str]) -> None:
    """Save the SHA cache to disk.

    Args:
        root: Repo root path.
        cache: Dict mapping file path → SHA-256 hex digest.
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


def has_changed(path: str, source: str, cache: dict[str, str]) -> bool:
    """Check if a file has changed since it was cached.

    Args:
        path: File path (cache key).
        source: Current file source text.
        cache: Loaded SHA cache dict.

    Returns:
        True if the file is new or modified; False if unchanged.
    """
    cached = cache.get(path)
    if cached is None:
        return True  # new file
    return compute_sha(source) != cached
