"""Async engine for graphsift — parallel indexing, graph traversal, and search.

Provides async variants of the core pipeline methods that use
``asyncio.to_thread()`` for CPU-bound operations (parsing, ranking)
and ``asyncio.gather()`` for parallel I/O.  Also exposes a
:class:`AsyncContextBuilder` that wraps the synchronous
:class:`ContextBuilder` with a fully async interface.

Usage::

    from graphsift.async_engine import (
        async_index_files,
        async_build,
        async_search,
        AsyncContextBuilder,
    )

    # Standalone async functions:
    stats = await async_index_files(builder, source_map)
    result = await async_build(builder, diff_spec, source_map)
    results = await async_search(query, nodes, top_k=20)

    # Full async wrapper:
    async_builder = AsyncContextBuilder(builder)
    stats = await async_builder.index_files(source_map)
    result = await async_builder.build(diff_spec, source_map)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from typing import Any

from .core import ContextBuilder, DependencyGraph, RelevanceRanker
from .hybrid_search import HybridSearcher
from .models import (
    ContextConfig,
    ContextResult,
    DiffSpec,
    FileNode,
    GraphNode,
    IndexStats,
    ScoredFile,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Standalone async helpers
# ===========================================================================


async def _parse_file_async(
    builder: ContextBuilder, path: str, source: str
) -> FileNode:
    """Parse a single file in a thread pool."""
    return await asyncio.to_thread(builder.index_file, path, source)


async def async_index_files(
    builder: ContextBuilder,
    source_map: dict[str, str],
    *,
    concurrency: int = 8,
    incremental: bool = False,
) -> IndexStats:
    """Index multiple files asynchronously, parsing in parallel.

    Uses a ``ThreadPoolExecutor`` (via ``asyncio.to_thread``) to parse
    files concurrently, which is beneficial for CPU-bound AST parsing.

    Args:
        builder: Pre-configured ContextBuilder.
        source_map: Dict mapping file path → source text.
        concurrency: Maximum concurrent parse operations.
        incremental: If True, skip files whose SHA-256 matches the
            last indexed version (uses builder's internal SHA cache).

    Returns:
        IndexStats with counts.
    """
    t0 = time.monotonic()
    files_indexed = 0
    files_skipped = 0
    symbols = 0
    lang_counts: dict[str, int] = defaultdict(int)

    sem = asyncio.Semaphore(concurrency)

    async def _process(path: str, source: str) -> None:
        nonlocal files_indexed, files_skipped, symbols
        async with sem:
            if builder._should_skip(path):
                files_skipped += 1
                return

            # Incremental: skip if SHA matches
            if incremental:
                new_sha = hashlib.sha256(source.encode(errors="replace")).hexdigest()
                with builder._lock:
                    cached_sha = builder._sha_cache.get(path)
                if cached_sha == new_sha:
                    files_skipped += 1
                    return

            try:
                fn = await _parse_file_async(builder, path, source)
                files_indexed += 1
                symbols += len(fn.symbols)
                lang_counts[fn.language.value] += 1
                if incremental:
                    with builder._lock:
                        builder._sha_cache[path] = fn.sha256
            except Exception as exc:
                logger.warning(
                    "graphsift: async index skipped file",
                    extra={"path": path, "error": str(exc)},
                )
                files_skipped += 1

    tasks = [_process(path, source) for path, source in source_map.items()]
    await asyncio.gather(*tasks)

    # Build edges (CPU-bound, run in thread)
    total_edges = await asyncio.to_thread(
        _build_all_edges, builder._graph
    )

    duration = (time.monotonic() - t0) * 1000
    stats = IndexStats(
        files_indexed=files_indexed,
        files_skipped=files_skipped,
        symbols_extracted=symbols,
        edges_created=total_edges,
        duration_ms=round(duration, 2),
        languages=dict(lang_counts),
    )
    with builder._lock:
        builder._index_stats = stats

    logger.info(
        "graphsift: async index complete",
        extra={
            "files": files_indexed,
            "symbols": symbols,
            "edges": total_edges,
            "ms": round(duration, 2),
        },
    )
    return stats


def _build_all_edges(graph: DependencyGraph) -> int:
    """Build all edge types synchronously (runs in thread pool)."""
    import_edges = graph.build_import_edges()
    inherit_edges = graph.build_inheritance_edges()
    dec_edges = graph.build_decorator_edges()
    return import_edges + inherit_edges + dec_edges


async def async_build(
    builder: ContextBuilder,
    diff_spec: DiffSpec,
    source_map: dict[str, str],
) -> ContextResult:
    """Async graph traversal + ranking + selection.

    Offloads CPU-bound operations (graph traversal, ranking, rendering)
    to a thread pool via ``asyncio.to_thread``.

    Args:
        builder: Pre-indexed ContextBuilder.
        diff_spec: Diff specification.
        source_map: Dict mapping file path → source text.

    Returns:
        ContextResult with selected files and rendered context.
    """
    if not diff_spec.changed_files:
        from .exceptions import ValidationError
        raise ValidationError("DiffSpec must have at least one changed_file.")

    # Check session-memory cache (fast, no threading needed)
    diff_hash = hashlib.sha256(
        str(sorted(diff_spec.changed_files) + [diff_spec.query, diff_spec.commit_message]).encode()
    ).hexdigest()
    cached = await asyncio.to_thread(builder._check_cache, diff_spec, diff_hash)
    if cached is not None:
        return cached

    # Full build in thread pool
    result = await asyncio.to_thread(
        builder.build, diff_spec, source_map
    )
    return result


async def async_search(
    query: str,
    nodes: list[GraphNode],
    top_k: int = 20,
    searcher: HybridSearcher | None = None,
) -> list[tuple[GraphNode, float]]:
    """Async hybrid search over graph nodes.

    Offloads BM25 + TF-IDF scoring to a thread pool.

    Args:
        query: Free-text search query.
        nodes: Candidate GraphNode instances to score.
        top_k: Maximum results to return.
        searcher: Optional pre-configured HybridSearcher. Creates a
            default one if not provided.

    Returns:
        List of ``(node, score)`` tuples sorted descending by score.
    """
    if not query or not nodes:
        return []

    searcher = searcher or HybridSearcher(alpha=0.3)
    return await asyncio.to_thread(
        searcher.search, query, nodes, top_k
    )


async def async_search_rrf(
    query: str,
    nodes: list[GraphNode],
    top_k: int = 20,
    include_dense: bool = False,
    searcher: HybridSearcher | None = None,
) -> list[tuple[GraphNode, float]]:
    """Async RRF-based hybrid search.

    Args:
        query: Free-text search query.
        nodes: Candidate GraphNode instances.
        top_k: Maximum results to return.
        include_dense: If True, also run dense vector search.
        searcher: Optional pre-configured HybridSearcher.

    Returns:
        List of ``(node, rrf_score)`` sorted descending.
    """
    if not query or not nodes:
        return []

    searcher = searcher or HybridSearcher(alpha=0.3)
    return await asyncio.to_thread(
        searcher.search_rrf, query, nodes, top_k, include_dense
    )


# ===========================================================================
# AsyncContextBuilder — full async wrapper around ContextBuilder
# ===========================================================================


class AsyncContextBuilder:
    """Async wrapper around :class:`ContextBuilder`.

    Provides the same public API as ``ContextBuilder`` but with async
    methods that use ``asyncio.to_thread`` and ``asyncio.gather`` for
    parallelism.

    Usage::

        from graphsift.core import ContextBuilder, ContextConfig
        from graphsift.async_engine import AsyncContextBuilder

        builder = ContextBuilder(ContextConfig(token_budget=50_000))
        async_builder = AsyncContextBuilder(builder)

        stats = await async_builder.index_files(source_map)
        result = await async_builder.build(diff_spec, source_map)
    """

    def __init__(self, builder: ContextBuilder) -> None:
        self._builder = builder

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> ContextConfig:
        return self._builder._config

    @property
    def graph(self) -> DependencyGraph:
        return self._builder._graph

    @property
    def ranker(self) -> RelevanceRanker:
        return self._builder._ranker

    # ------------------------------------------------------------------
    # Async index methods
    # ------------------------------------------------------------------

    async def index_file(self, path: str, source: str) -> FileNode:
        """Parse and index a single source file asynchronously.

        Args:
            path: File path.
            source: Source text.

        Returns:
            Parsed FileNode.
        """
        return await _parse_file_async(self._builder, path, source)

    async def index_files(
        self,
        source_map: dict[str, str],
        *,
        concurrency: int = 8,
    ) -> IndexStats:
        """Index multiple files asynchronously.

        Args:
            source_map: Dict mapping path → source.
            concurrency: Max concurrent parses.

        Returns:
            IndexStats.
        """
        return await async_index_files(
            self._builder, source_map, concurrency=concurrency
        )

    async def index_files_incremental(
        self,
        source_map: dict[str, str],
        *,
        concurrency: int = 8,
    ) -> IndexStats:
        """Incrementally index files, skipping unchanged ones.

        Args:
            source_map: Dict mapping path → source.
            concurrency: Max concurrent parses.

        Returns:
            IndexStats.
        """
        return await async_index_files(
            self._builder, source_map, concurrency=concurrency, incremental=True
        )

    # ------------------------------------------------------------------
    # Async build
    # ------------------------------------------------------------------

    async def build(
        self,
        diff_spec: DiffSpec,
        source_map: dict[str, str],
    ) -> ContextResult:
        """Build ranked context asynchronously.

        Args:
            diff_spec: Diff specification.
            source_map: Dict mapping path → source.

        Returns:
            ContextResult.
        """
        return await async_build(self._builder, diff_spec, source_map)

    # ------------------------------------------------------------------
    # Async search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        nodes: list[GraphNode] | None = None,
        top_k: int = 20,
    ) -> list[tuple[GraphNode, float]]:
        """Async hybrid search over the builder's graph nodes.

        Args:
            query: Free-text query.
            nodes: Nodes to search. If None, loads all nodes from graph.
            top_k: Max results.

        Returns:
            List of ``(node, score)`` tuples.
        """
        if nodes is None:
            all_files = self._builder._graph.all_files()
            nodes = [
                sym
                for f in all_files
                for sym in f.symbols
            ]
        return await async_search(query, nodes, top_k=top_k)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def index_stats(self) -> IndexStats:
        """Return stats from last index_files call."""
        return self._builder.index_stats()

    def graph_stats(self) -> dict[str, int]:
        """Return current graph statistics."""
        return self._builder.graph_stats()

    def __repr__(self) -> str:
        return f"AsyncContextBuilder({self._builder!r})"
