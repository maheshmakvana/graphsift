"""Schema versions for GraphNode and GraphEdge with migration paths.

Version history:
  v1 — Original: basic fields without community_id or schema_version.
  v2 — Current: adds community_id, schema_version, normalized decorators.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import EdgeKind, NodeKind, Language


# ---------------------------------------------------------------------------
# GraphNode schemas
# ---------------------------------------------------------------------------


class GraphNodeV1(BaseModel):
    """Original GraphNode — no community_id, no schema_version."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    file_path: str
    kind: NodeKind
    name: str
    qualified_name: str
    line_start: int = 0
    line_end: int = 0
    language: Language = Language.UNKNOWN
    signature: str = ""
    decorators: list[str] = Field(default_factory=list)
    is_async: bool = False
    is_dynamic: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNodeV2(BaseModel):
    """Current GraphNode — adds community_id, schema_version."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    file_path: str
    kind: NodeKind
    name: str
    qualified_name: str
    line_start: int = 0
    line_end: int = 0
    language: Language = Language.UNKNOWN
    signature: str = ""
    decorators: list[str] = Field(default_factory=list)
    is_async: bool = False
    is_dynamic: bool = False
    community_id: int | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=2, ge=1)


# ---------------------------------------------------------------------------
# GraphEdge schemas
# ---------------------------------------------------------------------------


class GraphEdgeV1(BaseModel):
    """Original GraphEdge — no schema_version."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    kind: EdgeKind
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeV2(BaseModel):
    """Current GraphEdge — adds schema_version for migration tracking."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    kind: EdgeKind
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=2, ge=1)


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


def migrate_graph_node_v1_to_v2(data: dict) -> dict:
    """Migrate a GraphNodeV1 dict to GraphNodeV2 format."""
    result = dict(data)
    result.setdefault("community_id", None)
    result["schema_version"] = 2
    return result


def migrate_graph_edge_v1_to_v2(data: dict) -> dict:
    """Migrate a GraphEdgeV1 dict to GraphEdgeV2 format."""
    result = dict(data)
    result["schema_version"] = 2
    return result
