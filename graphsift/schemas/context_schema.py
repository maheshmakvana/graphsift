"""Schema versions for ContextConfig and ContextResult with migration paths.

Version history:
  v1 — Original: excludes cache_aware, cache_provider, session_id, cache_ttl_days,
       depth_tier, exclude_patterns, dedup_enabled, trimming_context_lines,
       compress_low_score, compression_ratio.
  v2 — Current: adds all advanced cache, depth-tier, dedup, and trimming fields.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import OutputMode, DepthTier


# ---------------------------------------------------------------------------
# ContextConfig schemas
# ---------------------------------------------------------------------------


class ContextConfigV1(BaseModel):
    """Original ContextConfig — minimal cache/config fields."""

    model_config = ConfigDict(frozen=True)

    token_budget: int = Field(default=80_000, ge=100)
    max_depth: int = Field(default=4, ge=1, le=10)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    output_mode: OutputMode = OutputMode.SMART
    smart_threshold: float = 0.5
    hot_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    warm_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    include_tests: bool = True
    include_dynamic: bool = True
    diff_aware_trimming: bool = True


class ContextConfigV2(BaseModel):
    """Current ContextConfig — full set of fields with schema_version."""

    model_config = ConfigDict(frozen=True)

    token_budget: int = Field(default=80_000, ge=100)
    max_depth: int = Field(default=4, ge=1, le=10)
    min_score: float = Field(default=0.1, ge=0.0, le=1.0)
    output_mode: OutputMode = OutputMode.SMART
    smart_threshold: float = 0.5
    hot_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    warm_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    include_tests: bool = True
    include_dynamic: bool = True
    diff_aware_trimming: bool = True
    trimming_context_lines: int = Field(default=10, ge=0, le=100)
    compress_low_score: bool = True
    compression_ratio: float = Field(default=0.35, ge=0.1, le=1.0)
    cache_aware: bool = False
    cache_provider: str = "anthropic"
    session_id: str = ""
    cache_ttl_days: int = Field(default=7, ge=1, le=365)
    depth_tier: DepthTier = DepthTier.EXECUTION
    exclude_patterns: list[str] = Field(default_factory=lambda: [
        "venv", ".venv", "node_modules", "dist", "build",
        "__pycache__", ".git", "*.egg-info",
    ])
    dedup_enabled: bool = True
    schema_version: int = Field(default=2, ge=1)


# ---------------------------------------------------------------------------
# ContextResult schemas
# ---------------------------------------------------------------------------


class ContextResultV1(BaseModel):
    """Original ContextResult — basic fields only."""

    model_config = ConfigDict(frozen=True)

    diff_spec: dict
    selected_files: list[dict]
    rendered_context: str
    total_original_tokens: int
    total_rendered_tokens: int
    reduction_ratio: float


class ContextResultV2(BaseModel):
    """Current ContextResult — adds cache_breakpoints, files_scanned, metadata."""

    model_config = ConfigDict(frozen=True)

    diff_spec: dict
    selected_files: list[dict]
    rendered_context: str
    cache_breakpoints: int = 0
    total_original_tokens: int
    total_rendered_tokens: int
    reduction_ratio: float
    files_scanned: int = 0
    files_selected: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=2, ge=1)


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def migrate_config_v1_to_v2(data: dict) -> dict:
    """Migrate a ContextConfigV1 dict to ContextConfigV2 format."""
    result = dict(data)
    result.setdefault("trimming_context_lines", 10)
    result.setdefault("compress_low_score", True)
    result.setdefault("compression_ratio", 0.35)
    result.setdefault("cache_aware", False)
    result.setdefault("cache_provider", "anthropic")
    result.setdefault("session_id", "")
    result.setdefault("cache_ttl_days", 7)
    result.setdefault("depth_tier", "execution")
    result.setdefault("exclude_patterns", [
        "venv", ".venv", "node_modules", "dist", "build",
        "__pycache__", ".git", "*.egg-info",
    ])
    result.setdefault("dedup_enabled", True)
    result["schema_version"] = 2
    return result


def migrate_result_v1_to_v2(data: dict) -> dict:
    """Migrate a ContextResultV1 dict to ContextResultV2 format."""
    result = dict(data)
    result.setdefault("cache_breakpoints", 0)
    result.setdefault("files_scanned", 0)
    result.setdefault("files_selected", len(result.get("selected_files", [])))
    result.setdefault("metadata", {})
    result["schema_version"] = 2
    return result
