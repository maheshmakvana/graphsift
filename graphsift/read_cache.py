"""Read deduplication — fingerprint files and return stubs on repeat reads.

Prevents the same file content from entering the context window twice.
Uses SHA-256 content fingerprinting to detect identical reads within
the same session.

On first read, the file content is hashed and cached. On subsequent reads
of the same file with the same fingerprint, a short stub is returned
instead of the full content — saving significant tokens for files that
are repeatedly accessed during agentic workflows.

Supports invalidation when files change (call ``invalidate(path)`` after
modification), and full session reset via ``clear()``.

Usage::
    cache = ReadCache()
    content = cache.read("src/main.py", lambda: open("src/main.py").read())
    # First call: returns full content
    # Second call: returns "[graphsift] ... same content ... Skipping."
"""

from __future__ import annotations

import hashlib
from typing import Callable


class ReadCache:
    """Session-scoped file read cache with fingerprint dedup."""

    def __init__(self) -> None:
        self._fingerprints: dict[str, str] = {}
        self._stub_count: int = 0

    def read(self, path: str, reader: Callable[[], str]) -> str:
        """Read *path* via *reader*, returning a stub if content is unchanged.

        Args:
            path: File path (used as cache key).
            reader: Callable that returns the file content.

        Returns:
            Full file content on first read, stub on duplicate reads.
        """
        content = reader()
        fp = self._fingerprint(content)

        if path in self._fingerprints:
            if self._fingerprints[path] == fp:
                self._stub_count += 1
                return (
                    f"[graphsift] {path} — same content as earlier read "
                    f"(fingerprint match). Skipping."
                )
            # Content changed — update fingerprint and return new content
            self._fingerprints[path] = fp
            return content

        self._fingerprints[path] = fp
        return content

    def invalidate(self, path: str) -> None:
        """Remove *path* from cache (call after file modification)."""
        self._fingerprints.pop(path, None)

    def clear(self) -> None:
        """Clear all cached fingerprints (new session)."""
        self._fingerprints.clear()
        self._stub_count = 0

    @property
    def stubs_served(self) -> int:
        """Number of duplicate reads that returned stubs."""
        return self._stub_count

    @staticmethod
    def _fingerprint(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
