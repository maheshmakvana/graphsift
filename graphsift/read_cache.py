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

SafeFileIO — encoding-safe file I/O with BOM detection/stripping::

    from graphsift.read_cache import SafeFileIO

    content = SafeFileIO.read("file.txt")       # auto-strips BOM, errors=replace
    SafeFileIO.write("file.txt", content)        # forced UTF-8, no BOM
    data = SafeFileIO.read_json("config.json")   # safe JSON read
    SafeFileIO.write_json("config.json", data)   # safe JSON write
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SafeFileIO — encoding-safe file I/O with BOM handling
# ---------------------------------------------------------------------------


class SafeFileIO:
    """Encoding-safe file read/write with BOM detection, stripping, and caching.

    Every read uses ``errors="replace"`` so no file ever crashes on non-UTF-8
    content. Every write forces UTF-8 with no BOM. BOM is auto-detected and
    stripped on read.

    Usage::

        from graphsift.read_cache import SafeFileIO

        # Read with safe encoding (auto BOM strip, errors=replace)
        text = SafeFileIO.read("myfile.txt")

        # Write with forced UTF-8, no BOM
        SafeFileIO.write("myfile.txt", text)

        # JSON helpers
        data = SafeFileIO.read_json("config.json")
        SafeFileIO.write_json("config.json", data)

        # Utilities
        cleaned = SafeFileIO.strip_bom(raw_text)
        has_bom = SafeFileIO.detect_bom("somefile.txt")
    """

    # UTF-8 BOM character (U+FEFF ZERO WIDTH NO-BREAK SPACE)
    UTF8_BOM = chr(0xFEFF)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def read(path: str | Path, encoding: str = "utf-8", errors: str = "replace") -> str:
        """Read file with BOM detection/stripping and safe encoding.

        Args:
            path: File path to read.
            encoding: Text encoding (default ``utf-8``).
            errors: Error handler for encoding issues (default ``replace``).

        Returns:
            File content as string, with BOM stripped if present.
        """
        p = Path(path)
        raw = p.read_text(encoding=encoding, errors=errors)
        return SafeFileIO.strip_bom(raw)

    @staticmethod
    def write(path: str | Path, content: str, encoding: str = "utf-8") -> None:
        """Write file with forced encoding and NFC normalization.

        Automatically creates parent directories, normalizes to NFC form,
        and strips any BOM from content.

        Args:
            path: File path to write.
            content: Text content to write.
            encoding: Text encoding (default ``utf-8``).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cleaned = SafeFileIO.strip_bom(SafeFileIO.normalize(content))
        p.write_text(cleaned, encoding=encoding)

    @staticmethod
    def read_json(path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
        """Read and parse JSON with safe encoding.

        Args:
            path: File path to read.
            encoding: Text encoding (default ``utf-8``).

        Returns:
            Parsed JSON dict (empty dict on failure).
        """
        try:
            raw = SafeFileIO.read(path, encoding=encoding)
            return json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read JSON from %s: %s", path, exc)
            return {}

    @staticmethod
    def write_json(path: str | Path, data: dict[str, Any], encoding: str = "utf-8") -> None:
        """Serialize dict to JSON and write with safe encoding.

        Args:
            path: File path to write.
            data: Dict to serialize.
            encoding: Text encoding (default ``utf-8``).
        """
        content = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        SafeFileIO.write(path, content, encoding=encoding)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def strip_bom(content: str) -> str:
        """Strip UTF-8 BOM from content if present.

        Args:
            content: Text that may start with BOM.

        Returns:
            Content with BOM removed (if present).
        """
        if content and content[0] == SafeFileIO.UTF8_BOM:
            return content[1:]
        return content

    @staticmethod
    def normalize(content: str) -> str:
        """NFC-normalize text content.

        Args:
            content: Text to normalize.

        Returns:
            NFC-normalized text.
        """
        return unicodedata.normalize("NFC", content)

    @staticmethod
    def detect_bom(path: str | Path) -> bool:
        """Check if file starts with UTF-8 BOM.

        Args:
            path: File path to check.

        Returns:
            True if file starts with UTF-8 BOM.
        """
        try:
            with open(path, "rb") as f:
                raw = f.read(3)
            return raw == b"\xef\xbb\xbf"
        except OSError:
            return False


# ---------------------------------------------------------------------------
# ReadCache — session-scoped file read cache with fingerprint dedup
# ---------------------------------------------------------------------------


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
        return hashlib.sha256(content.encode(errors="replace")).hexdigest()
