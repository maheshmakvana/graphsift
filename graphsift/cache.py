"""LRU cache for parsed AST results with TTL. Uses disk + memory.

Provides ASTCache — a two-tier cache:
- Memory: OrderedDict LRU with configurable max entries and per-entry TTL.
- Disk: SQLite-backed persistent cache so parsed FileNodes survive restarts.

Usage::

    from graphsift.cache import ASTCache

    cache = ASTCache(max_memory=500, db_path="/tmp/ast_cache.db")
    cached = cache.get("sha256hex")      # FileNode or None
    cache.set("sha256hex", file_node)
    cache.invalidate("src/*.py")        # glob-style pattern
    cache.clear()
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from .models import FileNode, GraphNode, Language, NodeKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialisation helpers — convert FileNode <-> dict for SQLite storage
# ---------------------------------------------------------------------------

_FILE_NODE_SCHEMA_VERSION = 1


def _file_node_to_dict(fn: FileNode) -> dict[str, Any]:
    """Convert a FileNode (with nested GraphNodes) to a JSON-safe dict."""
    return {
        "__ast_cache_schema__": _FILE_NODE_SCHEMA_VERSION,
        "path": fn.path,
        "language": fn.language.value,
        "size_bytes": fn.size_bytes,
        "line_count": fn.line_count,
        "sha256": fn.sha256,
        "symbols": [
            {
                "node_id": s.node_id,
                "file_path": s.file_path,
                "kind": s.kind.value,
                "name": s.name,
                "qualified_name": s.qualified_name,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "language": s.language.value,
                "signature": s.signature,
                "decorators": s.decorators,
                "is_async": s.is_async,
                "is_dynamic": s.is_dynamic,
                "community_id": s.community_id,
                "metadata": dict(s.metadata),
            }
            for s in fn.symbols
        ],
        "imports": list(fn.imports),
        "dynamic_imports": list(fn.dynamic_imports),
        "token_estimate": fn.token_estimate,
        "metadata": dict(fn.metadata),
    }


def _dict_to_file_node(data: dict[str, Any]) -> FileNode | None:
    """Reconstruct a FileNode from a dict. Returns None on schema mismatch."""
    if data.get("__ast_cache_schema__") != _FILE_NODE_SCHEMA_VERSION:
        return None
    try:
        symbols = [
            GraphNode(
                node_id=s["node_id"],
                file_path=s["file_path"],
                kind=NodeKind(s["kind"]),
                name=s["name"],
                qualified_name=s["qualified_name"],
                line_start=s.get("line_start", 0),
                line_end=s.get("line_end", 0),
                language=Language(s.get("language", "unknown")),
                signature=s.get("signature", ""),
                decorators=s.get("decorators", []),
                is_async=s.get("is_async", False),
                is_dynamic=s.get("is_dynamic", False),
                community_id=s.get("community_id"),
                metadata=s.get("metadata", {}),
            )
            for s in data.get("symbols", [])
        ]
        return FileNode(
            path=data["path"],
            language=Language(data["language"]),
            size_bytes=data.get("size_bytes", 0),
            line_count=data.get("line_count", 0),
            sha256=data.get("sha256", ""),
            symbols=symbols,
            imports=data.get("imports", []),
            dynamic_imports=data.get("dynamic_imports", []),
            token_estimate=data.get("token_estimate", 0),
            metadata=data.get("metadata", {}),
        )
    except Exception as exc:
        logger.warning("graphsift: failed to deserialize FileNode from cache: %s", exc)
        return None


# ---------------------------------------------------------------------------
# ASTCache
# ---------------------------------------------------------------------------


class ASTCache:
    """Two-tier LRU cache for parsed AST results.

    **Memory tier** (LRU OrderedDict, configurable max size):
      - Fast in-process access for hot files.
      - Per-entry TTL to avoid serving stale results after a configurable
        duration.

    **Disk tier** (SQLite table):
      - Persists FileNode data across restarts.
      - Disk entries survive eviction from memory and are lazy-loaded back
        on miss.

    Typical usage for the ``ContextBuilder`` pipeline::

        cache = ASTCache(max_memory=500, db_path="/tmp/ast_cache.db")

        # During indexing
        fn = cache.get(file_sha256)
        if fn is None:
            fn = parser.parse_file(path, source)
            cache.set(file_sha256, fn)

        # When source changes
        cache.invalidate("src/auth/*.py")

    Args:
        max_memory: Maximum number of entries in the LRU memory cache.
        db_path: Path to the SQLite disk cache database. If empty or None,
            disk persistence is disabled.
        default_ttl: Default TTL in seconds for memory entries.
            ``None`` means no expiry.
    """

    def __init__(
        self,
        max_memory: int = 500,
        db_path: str = "",
        default_ttl: float | None = 3600.0,
    ) -> None:
        if max_memory < 1:
            raise ValueError("max_memory must be >= 1")

        self._max_memory = max_memory
        self._default_ttl = default_ttl
        self._mem_store: OrderedDict[str, tuple[FileNode, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits_mem = 0
        self._hits_disk = 0
        self._misses = 0
        self._evictions = 0
        self._warmed_count = 0

        # Disk cache
        self._db_path: str | None = db_path if db_path else None
        self._disk_conn: sqlite3.Connection | None = None
        if self._db_path:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._disk_conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._disk_conn.execute("PRAGMA journal_mode=WAL")
            self._disk_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ast_cache (
                    key         TEXT PRIMARY KEY,
                    value_json  TEXT NOT NULL,
                    created_at  TEXT DEFAULT (datetime('now'))
                )
                """
            )
            self._disk_conn.commit()

    def __repr__(self) -> str:
        return (
            f"ASTCache(mem={len(self._mem_store)}/{self._max_memory}, "
            f"disk={self._db_path is not None})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> FileNode | None:
        """Retrieve a cached FileNode by key (typically a SHA-256 hash).

        Checks memory first, then disk. On disk hit the entry is promoted
        to memory.

        Args:
            key: Cache key (e.g. file content SHA-256).

        Returns:
            Cached FileNode, or None if missing or expired.
        """
        if not key:
            return None

        # 1. Memory check
        with self._lock:
            if key in self._mem_store:
                entry, expires_at = self._mem_store[key]
                if self._default_ttl is None or time.monotonic() < expires_at:
                    self._mem_store.move_to_end(key)
                    self._hits_mem += 1
                    return entry
                # Expired — remove from memory
                del self._mem_store[key]
                self._evictions += 1

        # 2. Disk check
        if self._disk_conn is not None:
            try:
                row = self._disk_conn.execute(
                    "SELECT value_json FROM ast_cache WHERE key = ?", (key,)
                ).fetchone()
                if row is not None:
                    data = json.loads(row[0])
                    fn = _dict_to_file_node(data)
                    if fn is not None:
                        # Promote to memory
                        self.set(key, fn)
                        with self._lock:
                            self._hits_disk += 1
                        return fn
            except sqlite3.Error as exc:
                logger.warning("graphsift: ASTCache disk read error: %s", exc)

        with self._lock:
            self._misses += 1
        return None

    def set(self, key: str, value: FileNode) -> None:
        """Store a FileNode in cache (memory, and disk if configured).

        Args:
            key: Cache key (e.g. file content SHA-256).
            value: FileNode to cache.
        """
        exp = (
            (time.monotonic() + self._default_ttl)
            if self._default_ttl is not None
            else float("inf")
        )
        with self._lock:
            if key in self._mem_store:
                self._mem_store.move_to_end(key)
                self._mem_store[key] = (value, exp)
            else:
                if len(self._mem_store) >= self._max_memory:
                    self._mem_store.popitem(last=False)
                    self._evictions += 1
                self._mem_store[key] = (value, exp)

        # Persist to disk
        if self._disk_conn is not None:
            try:
                data = _file_node_to_dict(value)
                self._disk_conn.execute(
                    "INSERT OR REPLACE INTO ast_cache (key, value_json) VALUES (?, ?)",
                    (key, json.dumps(data)),
                )
                self._disk_conn.commit()
            except sqlite3.Error as exc:
                logger.warning("graphsift: ASTCache disk write error: %s", exc)

    def invalidate(self, pattern: str) -> None:
        """Invalidate cache entries matching a glob-style pattern on key or path.

        For example::

            cache.invalidate("src/*.py")       # match by key prefix or path pattern
            cache.invalidate("*test*")          # anything with 'test' in the key

        Args:
            pattern: Glob pattern to match against cache keys.
        """
        # Memory invalidation
        with self._lock:
            keys_to_remove = [k for k in self._mem_store if fnmatch(k, pattern)]
            for k in keys_to_remove:
                del self._mem_store[k]
                self._evictions += 1

        # Disk invalidation
        if self._disk_conn is not None:
            try:
                # Load all keys, match by pattern, delete
                rows = self._disk_conn.execute(
                    "SELECT key FROM ast_cache"
                ).fetchall()
                to_delete = [r[0] for r in rows if fnmatch(r[0], pattern)]
                for key in to_delete:
                    self._disk_conn.execute(
                        "DELETE FROM ast_cache WHERE key = ?", (key,)
                    )
                self._disk_conn.commit()
            except sqlite3.Error as exc:
                logger.warning("graphsift: ASTCache disk invalidation error: %s", exc)

    def clear(self) -> None:
        """Evict all entries from both memory and disk."""
        with self._lock:
            self._mem_store.clear()

        if self._disk_conn is not None:
            try:
                self._disk_conn.execute("DELETE FROM ast_cache")
                self._disk_conn.commit()
            except sqlite3.Error as exc:
                logger.warning("graphsift: ASTCache disk clear error: %s", exc)

    # ------------------------------------------------------------------
    # Cache warming (v2.4+)
    # ------------------------------------------------------------------

    def warm(self, keys: list[str]) -> int:
        """Pre-load entries from disk into memory cache.

        Useful after a build to warm cache for likely queries.

        Args:
            keys: Cache keys to warm (typically SHA-256 hashes).

        Returns:
            Number of entries warmed.
        """
        warmed = 0
        for key in keys:
            if key in self._mem_store:
                continue
            if self._disk_conn is not None:
                try:
                    row = self._disk_conn.execute(
                        "SELECT value_json FROM ast_cache WHERE key = ?",
                        (key,),
                    ).fetchone()
                    if row is not None:
                        data = json.loads(row[0])
                        fn = _dict_to_file_node(data)
                        if fn is not None:
                            self.set(key, fn)
                            warmed += 1
                            with self._lock:
                                self._warmed_count += 1
                except sqlite3.Error as exc:
                    logger.debug("Cache warm error for %s: %s", key, exc)
        return warmed

    def warm_from_paths(self, paths: list[str]) -> int:
        """Compute keys from file paths and warm cache.

        Args:
            paths: File paths to hash and warm.

        Returns:
            Number of entries warmed.
        """
        keys = []
        for path in paths:
            try:
                p = Path(path)
                if p.exists():
                    content = p.read_bytes()
                    key = hashlib.sha256(content).hexdigest()
                    keys.append(key)
            except Exception:
                pass
        return self.warm(keys)

    def predictive_warm(
        self, seed_paths: list[str], graph: object
    ) -> int:
        """Warm cache with files likely needed based on graph proximity.

        Uses the dependency graph's ``ranked_neighbors`` to find related
        files and pre-loads them into memory cache.

        Args:
            seed_paths: Starting file paths.
            graph: DependencyGraph with ``ranked_neighbors`` method.

        Returns:
            Number of entries warmed.
        """
        try:
            if hasattr(graph, "ranked_neighbors"):
                neighbors = graph.ranked_neighbors(seed_paths=seed_paths)
                related = list(neighbors.keys())[:50]
                return self.warm_from_paths(related)
        except Exception as exc:
            logger.debug("Predictive warm error: %s", exc)
        return 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits_mem + self._hits_disk + self._misses
            return {
                "hits_memory": self._hits_mem,
                "hits_disk": self._hits_disk,
                "misses": self._misses,
                "evictions": self._evictions,
                "memory_size": len(self._mem_store),
                "max_memory": self._max_memory,
                "disk_enabled": self._disk_conn is not None,
                "hit_rate": round(
                    (self._hits_mem + self._hits_disk) / max(total, 1), 4
                ),
                "warmed_entries": self._warmed_count,
            }

    def close(self) -> None:
        """Close the disk cache connection if open."""
        if self._disk_conn is not None:
            try:
                self._disk_conn.close()
            except sqlite3.Error:
                pass
            self._disk_conn = None
