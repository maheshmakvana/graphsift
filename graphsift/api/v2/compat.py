"""Adapter functions to convert between v1 and v2 models.

These allow gradual migration from the legacy API to the v2 unified API.
"""

from __future__ import annotations

from typing import Any

from graphsift.api.v2.models import (
    AnalysisResult,
    ContextResult,
    IndexResult,
    ScoredFile,
    SiftConfig,
)


# ---------------------------------------------------------------------------
# v1 → v2 Conversion
# ---------------------------------------------------------------------------


def v1_config_to_v2(v1_config: Any) -> SiftConfig:
    """Convert a v1 ContextConfig to a v2 SiftConfig.

    Args:
        v1_config: A ``graphsift.models.ContextConfig`` instance.

    Returns:
        Equivalent ``SiftConfig``.
    """
    return SiftConfig(
        token_budget=getattr(v1_config, "token_budget", 80_000),
        max_depth=getattr(v1_config, "max_depth", 4),
        min_score=getattr(v1_config, "min_score", 0.1),
        include_tests=getattr(v1_config, "include_tests", True),
        include_dynamic=getattr(v1_config, "include_dynamic", True),
        diff_aware_trimming=getattr(v1_config, "diff_aware_trimming", True),
        trimming_context_lines=getattr(v1_config, "trimming_context_lines", 10),
        compress_low_score=getattr(v1_config, "compress_low_score", True),
        compression_ratio=getattr(v1_config, "compression_ratio", 0.35),
        cache_aware=getattr(v1_config, "cache_aware", False),
        session_id=getattr(v1_config, "session_id", ""),
        exclude_patterns=getattr(
            v1_config,
            "exclude_patterns",
            ["venv", ".venv", "node_modules", "dist", "build", "__pycache__", ".git"],
        ),
        dedup_enabled=getattr(v1_config, "dedup_enabled", True),
    )


def v1_index_stats_to_v2(v1_stats: Any) -> IndexResult:
    """Convert a v1 IndexStats to a v2 IndexResult.

    Args:
        v1_stats: A ``graphsift.models.IndexStats`` instance.

    Returns:
        Equivalent ``IndexResult``.
    """
    return IndexResult(
        files_indexed=getattr(v1_stats, "files_indexed", 0),
        files_skipped=getattr(v1_stats, "files_skipped", 0),
        symbols_extracted=getattr(v1_stats, "symbols_extracted", 0),
        edges_created=getattr(v1_stats, "edges_created", 0),
        duration_ms=getattr(v1_stats, "duration_ms", 0.0),
        languages=dict(getattr(v1_stats, "languages", {})),
    )


def v1_scored_file_to_v2(v1_sf: Any) -> ScoredFile:
    """Convert a v1 ScoredFile to a v2 ScoredFile.

    Args:
        v1_sf: A ``graphsift.models.ScoredFile`` instance.

    Returns:
        Equivalent v2 ``ScoredFile``.
    """
    return ScoredFile(
        path=getattr(v1_sf, "file_node", getattr(v1_sf, "path", "")).path
        if hasattr(getattr(v1_sf, "file_node", None), "path")
        else getattr(v1_sf, "path", ""),
        score=getattr(v1_sf, "score", 0.0),
        rank=getattr(v1_sf, "rank", 0),
        reasons=getattr(v1_sf, "reasons", []),
        depth=getattr(v1_sf, "depth", 0),
    )


def v1_context_result_to_v2(v1_result: Any) -> ContextResult:
    """Convert a v1 ContextResult to a v2 ContextResult.

    Args:
        v1_result: A ``graphsift.models.ContextResult`` instance.

    Returns:
        Equivalent v2 ``ContextResult``.
    """
    v1_files = getattr(v1_result, "selected_files", [])
    return ContextResult(
        selected_files=[v1_scored_file_to_v2(sf) for sf in v1_files],
        rendered_context=getattr(v1_result, "rendered_context", ""),
        total_original_tokens=getattr(v1_result, "total_original_tokens", 0),
        total_rendered_tokens=getattr(v1_result, "total_rendered_tokens", 0),
        reduction_ratio=getattr(v1_result, "reduction_ratio", 0.0),
        files_scanned=getattr(v1_result, "files_scanned", 0),
        files_selected=getattr(v1_result, "files_selected", 0),
        metadata=dict(getattr(v1_result, "metadata", {})),
    )


# ---------------------------------------------------------------------------
# v2 → v1 Conversion
# ---------------------------------------------------------------------------


def v2_config_to_v1(v2_config: SiftConfig) -> Any:
    """Convert a v2 SiftConfig to a v1 ContextConfig.

    Returns:
        A ``graphsift.models.ContextConfig`` instance.
    """
    from graphsift.models import ContextConfig

    return ContextConfig(
        token_budget=v2_config.token_budget,
        max_depth=v2_config.max_depth,
        min_score=v2_config.min_score,
        include_tests=v2_config.include_tests,
        include_dynamic=v2_config.include_dynamic,
        diff_aware_trimming=v2_config.diff_aware_trimming,
        trimming_context_lines=v2_config.trimming_context_lines,
        compress_low_score=v2_config.compress_low_score,
        compression_ratio=v2_config.compression_ratio,
        cache_aware=v2_config.cache_aware,
        session_id=v2_config.session_id,
        exclude_patterns=v2_config.exclude_patterns,
        dedup_enabled=v2_config.dedup_enabled,
    )


def v2_index_result_to_v1(v2_result: IndexResult) -> Any:
    """Convert a v2 IndexResult to a v1 IndexStats.

    Returns:
        A ``graphsift.models.IndexStats`` instance.
    """
    from graphsift.models import IndexStats

    return IndexStats(
        files_indexed=v2_result.files_indexed,
        files_skipped=v2_result.files_skipped,
        symbols_extracted=v2_result.symbols_extracted,
        edges_created=v2_result.edges_created,
        duration_ms=v2_result.duration_ms,
        languages=v2_result.languages,
    )


def v2_context_result_to_v1(v2_result: ContextResult) -> Any:
    """Convert a v2 ContextResult back to v1 format.

    Returns:
        A dict structure compatible with v1 ``ContextResult`` attributes.
    """
    from graphsift.models import ScoredFile as V1ScoredFile
    from graphsift.models import FileNode

    v1_files = []
    for sf in v2_result.selected_files:
        v1_files.append(
            V1ScoredFile(
                file_node=FileNode(path=sf.path, language="unknown"),
                score=sf.score,
                rank=sf.rank,
                reasons=sf.reasons,
                depth=sf.depth,
            )
        )
    return {
        "selected_files": v1_files,
        "rendered_context": v2_result.rendered_context,
        "total_original_tokens": v2_result.total_original_tokens,
        "total_rendered_tokens": v2_result.total_rendered_tokens,
        "reduction_ratio": v2_result.reduction_ratio,
        "files_scanned": v2_result.files_scanned,
        "files_selected": v2_result.files_selected,
        "metadata": v2_result.metadata,
    }


__all__ = [
    "v1_config_to_v2",
    "v1_index_stats_to_v2",
    "v1_scored_file_to_v2",
    "v1_context_result_to_v2",
    "v2_config_to_v1",
    "v2_index_result_to_v1",
    "v2_context_result_to_v1",
]
