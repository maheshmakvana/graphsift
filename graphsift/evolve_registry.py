"""JSON-file-backed persistence store for evolved parameters.

Stores parameter sets produced by :class:`~graphsift.evolve.EvolutionOptimizer`
so successful evolutions can be recalled, compared, and reused across sessions.
Each entry is keyed by a **fingerprint** (SHA256 of source_map keys) and further
sub-keyed by **space_type** (e.g. ``"full"``, ``"ranker"``, ``"context"``).

The registry is backed by a single JSON file and is thread-safe via
:class:`threading.Lock`.

File format::

    {
        "<fingerprint>": {
            "full": {
                "params": {"bm25_weight": 0.6, "recency_weight": 0.3},
                "score": 0.5778,
                "timestamp": 1710800000.0,
                "rounds": 40
            },
            "ranker": {
                "params": {"recency_weight": 0.4},
                "score": 0.5123,
                "timestamp": 1710800100.0,
                "rounds": 25
            }
        }
    }

Usage::

    from graphsift.evolve_registry import EvolveRegistry

    reg = EvolveRegistry()
    reg.set("abc123def", "full", {"bm25_weight": 0.6}, score=0.5778)
    cached = reg.get("abc123def", "full")   # -> {"bm25_weight": 0.6} or None
    entries = reg.list_entries()
    reg.clear()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from graphsift.read_cache import SafeFileIO

logger = logging.getLogger(__name__)


class EvolveRegistry:
    """JSON-file-backed persistence store for evolved parameters.

    Each evolved parameter set is stored under a ``(fingerprint, space_type)``
    pair. The registry is thread-safe and handles missing or corrupted files
    gracefully.
    """

    def __init__(self, path: str | None = None) -> None:
        """Initialize registry.

        Args:
            path: Path to JSON file. If None, uses ``.graphsift/evolve_registry.json``
                  relative to the current working directory.
        """
        self._path = Path(path) if path else Path.cwd() / ".graphsift" / "evolve_registry.json"
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────

    def get(self, fingerprint: str, space_type: str = "full") -> dict | None:
        """Return cached params for a fingerprint and space_type, or None.

        Args:
            fingerprint: Unique identifier (caller computes SHA256 of
                source_map keys).
            space_type: Parameter space type (e.g. ``"full"``, ``"ranker"``).

        Returns:
            The params dict if found, or None if the fingerprint or
            space_type does not exist in the registry.
        """
        with self._lock:
            data = self._load()
            entry = data.get(fingerprint, {}).get(space_type)
            if entry is not None:
                params = entry.get("params")
                if params is not None:
                    return dict(params)
            return None

    def set(self, fingerprint: str, space_type: str, params: dict, score: float) -> None:
        """Store evolved parameters for a fingerprint and space_type.

        Args:
            fingerprint: Unique identifier (caller computes SHA256 of
                source_map keys).
            space_type: Parameter space type (e.g. ``"full"``, ``"ranker"``).
            params: The parameter dict to cache.
            score: The fitness score achieved by this parameter set.
        """
        with self._lock:
            data = self._load()
            data.setdefault(fingerprint, {})[space_type] = {
                "params": dict(params),
                "score": float(score),
                "timestamp": time.time(),
            }
            self._save(data)

    def list_entries(self) -> list[dict]:
        """Return all entries with metadata.

        Each entry dict contains ``fingerprint``, ``space_type``, ``params``,
        ``score``, and ``timestamp`` keys.

        Returns:
            A list of all registry entries as flat dicts with metadata.
        """
        with self._lock:
            data = self._load()
            entries: list[dict] = []
            for fingerprint, spaces in data.items():
                for space_type, entry in spaces.items():
                    entries.append({
                        "fingerprint": fingerprint,
                        "space_type": space_type,
                        "params": dict(entry.get("params", {})),
                        "score": entry.get("score", 0.0),
                        "timestamp": entry.get("timestamp", 0.0),
                    })
            return entries

    @property
    def path(self) -> str:
        """Return the registry file path as a string."""
        return str(self._path)

    def clear(self) -> None:
        """Delete all entries by removing the backing file."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    # ── Internal ────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        """Read the JSON file and return its contents.

        Returns an empty dict if the file does not exist or is corrupted.
        Logs a warning on corrupted data rather than raising.
        """
        if not self._path.exists():
            return {}
        try:
            raw = SafeFileIO.read(self._path)
            if not raw.strip():
                return {}
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
            logger.warning("EvolveRegistry root value is not a dict, resetting")
            return {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("EvolveRegistry corrupted, resetting: %s", exc)
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        """Write data to the JSON file, creating parent directories if needed.

        Args:
            data: The full registry data dict to persist.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        SafeFileIO.write_json(self._path, data)


__all__ = [
    "EvolveRegistry",
]
