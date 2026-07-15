"""Lighter, cleaner Pydantic models for the v2 graphsift API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SiftConfig(BaseModel):
    """Configuration for the unified Sift API."""

    model_config = ConfigDict(frozen=True)

    token_budget: int = Field(
        default=80_000,
        ge=100,
        description="Hard token budget for total selected context.",
    )
    max_depth: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum graph traversal depth from changed files.",
    )
    min_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score to include a file.",
    )
    include_tests: bool = True
    include_dynamic: bool = True
    diff_aware_trimming: bool = True
    trimming_context_lines: int = Field(default=10, ge=0, le=100)
    compress_low_score: bool = True
    compression_ratio: float = Field(default=0.35, ge=0.1, le=1.0)
    cache_aware: bool = False
    session_id: str = ""
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "venv", ".venv", "node_modules", "dist", "build",
            "__pycache__", ".git", "*.egg-info",
        ]
    )
    dedup_enabled: bool = True


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class IndexResult(BaseModel):
    """Result of an indexing operation."""

    model_config = ConfigDict(frozen=True)

    files_indexed: int = 0
    files_skipped: int = 0
    symbols_extracted: int = 0
    edges_created: int = 0
    duration_ms: float = 0.0
    languages: dict[str, int] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"IndexResult(files={self.files_indexed}, "
            f"symbols={self.symbols_extracted}, "
            f"edges={self.edges_created})"
        )


class ScoredFile(BaseModel):
    """A file with its relevance score."""

    model_config = ConfigDict(frozen=True)

    path: str
    score: float = Field(ge=0.0, le=1.0)
    rank: int = 0
    reasons: list[str] = Field(default_factory=list)
    depth: int = 0


class ContextResult(BaseModel):
    """Result of a context building operation."""

    model_config = ConfigDict(frozen=True)

    selected_files: list[ScoredFile]
    rendered_context: str
    total_original_tokens: int
    total_rendered_tokens: int
    reduction_ratio: float
    files_scanned: int = 0
    files_selected: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ContextResult("
            f"selected={self.files_selected}/{self.files_scanned}, "
            f"tokens={self.total_rendered_tokens:,}, "
            f"saved={self.reduction_ratio:.0%})"
        )


class AnalysisResult(BaseModel):
    """Result of a codebase analysis operation."""

    model_config = ConfigDict(frozen=True)

    total_files: int = 0
    total_symbols: int = 0
    total_edges: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    cycles: int = 0
    dead_code_items: int = 0
    summary: str = ""


# ---------------------------------------------------------------------------
# Internal helper models (not exported)
# ---------------------------------------------------------------------------


@dataclass
class CompressResult:
    """Result from a compression operation."""

    text: str
    original_chars: int
    compressed_chars: int
    compression_ratio: float
    kind: str


__all__ = [
    "SiftConfig",
    "IndexResult",
    "ScoredFile",
    "ContextResult",
    "AnalysisResult",
    "CompressResult",
]
