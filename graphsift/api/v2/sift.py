"""Unified Sift API — modern replacement for ContextBuilder + advanced tools.

The ``Sift`` class combines indexing, search, context building, compression,
and analysis into a single clean interface.
"""

from __future__ import annotations

import time
from typing import Any

from graphsift.api.v2.exceptions import (
    BuildError,
    CompressError,
    ConfigError,
    IndexError,
    SearchError,
)
from graphsift.api.v2.models import (
    AnalysisResult,
    CompressResult,
    ContextResult,
    IndexResult,
    ScoredFile,
    SiftConfig,
)

# Lazy imports to avoid circular dependencies at import time
_core = None
_compress = None


def _get_core():
    global _core
    if _core is None:
        from graphsift import core as _core
    return _core


def _get_compress():
    global _compress
    if _compress is None:
        from graphsift import compress as _compress
    return _compress


class Sift:
    """Unified API for graphsift.

    Combines indexing, search, context building, compression, and analysis
    into a single clean interface with consistent naming and return types.

    Basic usage::

        from graphsift.api.v2 import Sift, SiftConfig

        sift = Sift(SiftConfig(token_budget=50_000))
        result = sift.index({"src/main.py": "def hello(): ..."})
        ctx = sift.build_context(["src/main.py"], query="explain this")
        print(ctx.rendered_context)
    """

    def __init__(self, config: SiftConfig | None = None):
        self._config = config or SiftConfig()
        core = _get_core()
        self._builder = core.ContextBuilder(
            core.ContextConfig(
                token_budget=self._config.token_budget,
                max_depth=self._config.max_depth,
                min_score=self._config.min_score,
                include_tests=self._config.include_tests,
                include_dynamic=self._config.include_dynamic,
                diff_aware_trimming=self._config.diff_aware_trimming,
                trimming_context_lines=self._config.trimming_context_lines,
                compress_low_score=self._config.compress_low_score,
                compression_ratio=self._config.compression_ratio,
                cache_aware=self._config.cache_aware,
                session_id=self._config.session_id,
                exclude_patterns=self._config.exclude_patterns,
                dedup_enabled=self._config.dedup_enabled,
            )
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, files: dict[str, str]) -> IndexResult:
        """Index source files for analysis.

        Args:
            files: Dictionary mapping file paths to source code content.

        Returns:
            IndexResult with indexing statistics.
        """
        try:
            stats = self._builder.index_files(files)
            return IndexResult(
                files_indexed=stats.files_indexed,
                files_skipped=stats.files_skipped,
                symbols_extracted=stats.symbols_extracted,
                edges_created=stats.edges_created,
                duration_ms=stats.duration_ms,
                languages=dict(stats.languages),
            )
        except Exception as exc:
            raise IndexError(f"Indexing failed: {exc}") from exc

    def search(self, query: str, top_k: int = 20) -> list[ScoredFile]:
        """Search the indexed codebase for relevant files.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return (default: 20).

        Returns:
            List of ScoredFile results ordered by relevance (highest first).
        """
        try:
            core = _get_core()
            from graphsift import hybrid_search

            searcher = hybrid_search.HybridSearcher(self._builder._graph)
            results = searcher.search(query, top_k=top_k)
            return [
                ScoredFile(
                    path=r.file_node.path,
                    score=r.score,
                    rank=i,
                    reasons=r.reasons,
                    depth=r.depth,
                )
                for i, r in enumerate(results)
            ]
        except Exception as exc:
            raise SearchError(f"Search failed: {exc}") from exc

    def build_context(
        self,
        changes: list[str],
        query: str = "",
    ) -> ContextResult:
        """Build an optimised context for a set of changes.

        Args:
            changes: List of changed file paths.
            query: Optional free-text question about the change.

        Returns:
            ContextResult with selected files and rendered context.
        """
        try:
            from graphsift.models import DiffSpec

            diff = DiffSpec(changed_files=changes, query=query)
            result = self._builder.build(diff)

            return ContextResult(
                selected_files=[
                    ScoredFile(
                        path=sf.file_node.path,
                        score=sf.score,
                        rank=sf.rank,
                        reasons=sf.reasons,
                        depth=sf.depth,
                    )
                    for sf in result.selected_files
                ],
                rendered_context=result.rendered_context,
                total_original_tokens=result.total_original_tokens,
                total_rendered_tokens=result.total_rendered_tokens,
                reduction_ratio=result.reduction_ratio,
                files_scanned=result.files_scanned,
                files_selected=result.files_selected,
            )
        except Exception as exc:
            raise BuildError(f"Context build failed: {exc}") from exc

    def compress(self, text: str, kind: str = "auto") -> str:
        """Compress text using graphsift's CLI output compressors.

        Args:
            text: Text to compress (e.g. command output).
            kind: Compression kind — ``"auto"`` (default), ``"pytest"``,
                  ``"bash"``, ``"git_diff"``, etc.

        Returns:
            Compressed text string.
        """
        try:
            cm = _get_compress()
            return cm.compress(text, kind)
        except Exception as exc:
            raise CompressError(f"Compression failed: {exc}") from exc

    def analyze(self, path: str) -> AnalysisResult:
        """Analyse a codebase path and return structural metrics.

        Args:
            path: File or directory path to analyse.

        Returns:
            AnalysisResult with structural information.
        """
        import os

        try:
            total_files = 0
            total_symbols = 0
            total_edges = 0
            languages: dict[str, int] = {}

            if os.path.isfile(path):
                core = _get_core()
                lang = core.detect_language(path)
                lang_str = lang.value if hasattr(lang, "value") else str(lang)
                languages[lang_str] = languages.get(lang_str, 0) + 1
                total_files = 1

            elif os.path.isdir(path):
                for root, _dirs, files in os.walk(path):
                    for fname in files:
                        if fname.endswith((".py", ".js", ".ts", ".rs", ".go")):
                            total_files += 1
                            core = _get_core()
                            lang = core.detect_language(fname)
                            lang_str = lang.value if hasattr(lang, "value") else str(lang)
                            languages[lang_str] = languages.get(lang_str, 0) + 1

            # If we have graph data, include it
            try:
                graph = self._builder._graph
                stats = graph.stats()
                total_symbols = sum(len(f.symbols) for f in graph.all_files())
                total_edges = stats.get("edges", 0)
            except Exception:
                pass

            summary = (
                f"Analysed {total_files} file(s) across {len(languages)} language(s). "
                f"Found {total_symbols} symbols and {total_edges} dependency edges."
            )

            return AnalysisResult(
                total_files=total_files,
                total_symbols=total_symbols,
                total_edges=total_edges,
                languages=languages,
                summary=summary,
            )
        except Exception as exc:
            from graphsift.api.v2.exceptions import AnalyzeError
            raise AnalyzeError(f"Analysis failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Property access to underlying objects
    # ------------------------------------------------------------------

    @property
    def config(self) -> SiftConfig:
        """Return the current configuration."""
        return self._config

    @property
    def builder(self):
        """Access the underlying ContextBuilder (advanced use)."""
        return self._builder


__all__ = [
    "Sift",
]
