"""Pydantic v2 data contracts for graphsift."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Language(str, Enum):
    """Supported source languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    RUBY = "ruby"
    PHP = "php"
    BASH = "bash"
    HCL = "hcl"         # Terraform / OpenTofu
    HELM = "helm"       # Helm chart templates (YAML+Go template)
    UNKNOWN = "unknown"


class NodeKind(str, Enum):
    """Type of a graph node."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    IMPORT = "import"
    VARIABLE = "variable"
    DECORATOR = "decorator"


class EdgeKind(str, Enum):
    """Type of a dependency edge."""

    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    DECORATES = "decorates"
    REFERENCES = "references"
    TEST_COVERS = "test_covers"
    DYNAMIC_IMPORT = "dynamic_import"


class OutputMode(str, Enum):
    """How to render selected files for LLM consumption."""

    FULL = "full"               # Raw source
    SIGNATURES = "signatures"  # Function/class signatures only
    COMPRESSED = "compressed"  # tokenpruner compression
    SMART = "smart"            # Full for high-score, signatures for low-score


# ---------------------------------------------------------------------------
# Graph nodes and edges
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A symbol in the codebase dependency graph."""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(description="Unique identifier: file::symbol_path")
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
    is_dynamic: bool = False  # True if detected via dynamic-import pattern
    community_id: int | None = Field(default=None, description="Community cluster ID assigned by community detection")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"GraphNode({self.kind.value}:{self.qualified_name})"


class GraphEdge(BaseModel):
    """A directed dependency between two nodes."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    kind: EdgeKind
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"GraphEdge({self.source_id} -{self.kind.value}-> {self.target_id})"


# ---------------------------------------------------------------------------
# File-level models
# ---------------------------------------------------------------------------


class FileNode(BaseModel):
    """Represents an indexed source file."""

    model_config = ConfigDict(frozen=True)

    path: str
    language: Language
    size_bytes: int = 0
    line_count: int = 0
    sha256: str = ""
    symbols: list[GraphNode] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    dynamic_imports: list[str] = Field(default_factory=list)
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"FileNode({self.path}, {self.language.value}, {len(self.symbols)} symbols)"


class ScoredFile(BaseModel):
    """A file with its relevance score for a given query/diff."""

    model_config = ConfigDict(frozen=True)

    file_node: FileNode
    score: float = Field(ge=0.0, le=1.0, description="Relevance score 0=irrelevant, 1=critical")
    rank: int = 0
    reasons: list[str] = Field(default_factory=list, description="Why this file was selected")
    depth: int = Field(default=0, description="Graph distance from changed files")
    output_mode: OutputMode = OutputMode.SMART

    def __repr__(self) -> str:
        return f"ScoredFile({self.file_node.path}, score={self.score:.3f}, rank={self.rank})"


# ---------------------------------------------------------------------------
# Query / context models
# ---------------------------------------------------------------------------


class DiffSpec(BaseModel):
    """Specification of changed files for a code review query."""

    model_config = ConfigDict(frozen=True)

    changed_files: list[str] = Field(description="Absolute or repo-relative paths of changed files")
    diff_text: str = Field(default="", description="Optional raw unified diff")
    commit_message: str = Field(default="")
    query: str = Field(default="", description="Free-text question about the change")

    def __repr__(self) -> str:
        return f"DiffSpec(changed={len(self.changed_files)} files)"


class ContextConfig(BaseModel):
    """Configuration for context selection and rendering."""

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
    output_mode: OutputMode = OutputMode.SMART
    smart_threshold: float = Field(
        default=0.5,
        description="Score above this → FULL; below → SIGNATURES (in SMART mode).",
    )
    include_tests: bool = True
    include_dynamic: bool = True
    compress_low_score: bool = Field(
        default=True,
        description="Use tokenpruner on low-score files to save budget.",
    )
    compression_ratio: float = Field(
        default=0.35,
        ge=0.1,
        le=1.0,
        description="tokenpruner target ratio for compressed files.",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "venv", ".venv", "node_modules", "dist", "build",
            "__pycache__", ".git", "*.egg-info",
        ]
    )


class ContextResult(BaseModel):
    """Final output: ranked selected files ready to send to an LLM."""

    model_config = ConfigDict(frozen=True)

    diff_spec: DiffSpec
    selected_files: list[ScoredFile]
    rendered_context: str = Field(description="Ready-to-paste LLM context string")
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


class IndexStats(BaseModel):
    """Statistics from a repository indexing run."""

    model_config = ConfigDict(frozen=True)

    files_indexed: int = 0
    files_skipped: int = 0
    symbols_extracted: int = 0
    edges_created: int = 0
    dynamic_imports_found: int = 0
    duration_ms: float = 0.0
    languages: dict[str, int] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"IndexStats(files={self.files_indexed}, "
            f"symbols={self.symbols_extracted}, "
            f"edges={self.edges_created})"
        )
