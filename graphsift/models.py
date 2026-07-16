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


class SourceConfidence(str, Enum):
    """Confidence level for code symbol extraction.

    ``EXTRACTED`` means the symbol was obtained via a deterministic AST parse
    (e.g. Python's ``ast`` module or tree-sitter). ``INFERRED`` means it was
    detected via a best-effort heuristic (regex / BM25 / vector search) and
    may be incomplete or inaccurate.
    """

    EXTRACTED = "extracted"
    INFERRED = "inferred"


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


class TierLevel(str, Enum):
    """Tier level for context selection."""

    HOT = "hot"        # Full source for highest-relevance files
    WARM = "warm"      # Signatures/headers for medium-relevance files
    COLD = "cold"      # Excluded from context


class DepthTier(str, Enum):
    """Context depth tier for different development phases.

    - PLANNING: High-level topology — file names, module descriptions, class signatures.
    - EXPLORATION: Interface-level — add function signatures, docstrings, type hints.
    - EXECUTION: Full implementation — complete function bodies and inline comments.
    """

    PLANNING = "planning"
    EXPLORATION = "exploration"
    EXECUTION = "execution"


class BudgetMode(str, Enum):
    """How to allocate token budget across selected files."""
    FIXED = "fixed"           # Equal share per file (current behavior)
    ADAPTIVE = "adaptive"     # Weighted by centrality/complexity
    PER_PHASE = "per_phase"   # Allocate per-plan-phase


class PruningStrategy(str, Enum):
    """How aggressively to prune redundant content from rendered context."""
    NONE = "none"             # No pruning (current behavior)
    LIGHT = "light"           # Only remove exact duplicate blocks
    BALANCED = "balanced"     # Remove duplicates + near-duplicates (SimHash)
    AGGRESSIVE = "aggressive" # Remove duplicates, near-duplicates, and boilerplate


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
    source_confidence: SourceConfidence = Field(
        default=SourceConfidence.EXTRACTED,
        description="Whether this symbol was extracted via deterministic AST (EXTRACTED) or regex/heuristic (INFERRED)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

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
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

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
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

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
    source_confidence: SourceConfidence = Field(
        default=SourceConfidence.EXTRACTED,
        description="Aggregate confidence for this file's symbols — EXTRACTED if all symbols are AST-parsed, INFERRED if any came from regex/heuristic",
    )
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

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
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

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
    hot_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Score above this → HOT (full source).",
    )
    warm_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Score above this → WARM (signatures). Below → COLD (excluded).",
    )
    include_tests: bool = True
    include_dynamic: bool = True
    diff_aware_trimming: bool = Field(
        default=True,
        description="Only include diff-relevant portions of files instead of full source.",
    )
    trimming_context_lines: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Lines of surrounding context to include around changed regions when trimming.",
    )
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
    cache_aware: bool = Field(
        default=False,
        description="Structure output with Anthropic/OpenAI cache_control breakpoints.",
    )
    cache_provider: str = Field(
        default="anthropic",
        description="Target LLM provider for cache markers: anthropic, openai, auto.",
    )
    session_id: str = Field(
        default="",
        description="Optional session identifier for cross-session memory reuse.",
    )
    cache_ttl_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="How many days a cached context entry remains valid.",
    )
    depth_tier: DepthTier = Field(
        default=DepthTier.EXECUTION,
        description="Context depth: planning (broad topology), exploration (interface contracts), execution (full implementation).",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "venv", ".venv", "node_modules", "dist", "build",
            "__pycache__", ".git", "*.egg-info",
        ]
    )
    dedup_enabled: bool = Field(
        default=True,
        description="Enable entropy-based deduplication of near-identical files to improve context diversity.",
    )
    auto_evolve: bool = Field(
        default=False,
        description="When True, ContextBuilder automatically runs EvolutionOptimizer to tune parameters for this codebase.",
    )
    evolve_rounds: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of evolution rounds when auto_evolve is enabled.",
    )
    evolve_population: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Population size per evolution round when auto_evolve is enabled.",
    )
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

    # -- Adaptive budgeting & pruning (v2.4+) --
    budget_mode: BudgetMode = Field(
        default=BudgetMode.FIXED,
        description="How to allocate token budget across selected files.",
    )
    pruning_strategy: PruningStrategy = Field(
        default=PruningStrategy.NONE,
        description="How aggressively to prune redundant content.",
    )
    centrality_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight given to graph centrality in adaptive budget allocation.",
    )
    overlap_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Token overlap ratio threshold that triggers dedup during pruning.",
    )


class ContextResult(BaseModel):
    """Final output: ranked selected files ready to send to an LLM."""

    model_config = ConfigDict(frozen=True)

    diff_spec: DiffSpec
    selected_files: list[ScoredFile]
    rendered_context: str = Field(description="Ready-to-paste LLM context string")
    cache_breakpoints: int = Field(
        default=0,
        description="Number of cache_control breakpoints in rendered output.",
    )
    total_original_tokens: int
    total_rendered_tokens: int
    reduction_ratio: float
    files_scanned: int = 0
    files_selected: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

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
    schema_version: int = Field(default=2, ge=1, description="Schema version for migration support")

    def __repr__(self) -> str:
        return (
            f"IndexStats(files={self.files_indexed}, "
            f"symbols={self.symbols_extracted}, "
            f"edges={self.edges_created})"
        )


# ---------------------------------------------------------------------------
# Cycle detection and dead code
# ---------------------------------------------------------------------------


class CycleInfo(BaseModel):
    """Information about a dependency cycle."""

    cycle_id: int
    files: list[str] = Field(description="File paths in the cycle")
    length: int = Field(description="Number of files in the cycle")
    severity: str = Field(default="warning", description="warning | error | info")


class CycleReport(BaseModel):
    """Result of cycle detection analysis."""

    cycles: list[CycleInfo]
    total_cycles: int
    max_cycle_length: int
    files_in_cycles: int = 0


class DeadCodeInfo(BaseModel):
    """Information about a potentially dead code element."""

    node_id: str
    file_path: str
    name: str
    kind: str = Field(description="function | class | method | variable")
    line_start: int = 0
    line_end: int = 0
    reason: str = Field(default="No callers found from entry points")


class DeadCodeReport(BaseModel):
    """Result of dead code detection."""

    entries: list[DeadCodeInfo]
    total_dead: int
    confidence: str = Field(default="medium", description="high | medium | low")


# ---------------------------------------------------------------------------
# Fix suggestions
# ---------------------------------------------------------------------------


class FixSeverity(str, Enum):
    """Severity level for auto-fix suggestions."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class FixSuggestion(BaseModel):
    """A single suggested fix for a code issue detected via graph analysis."""

    suggestion_id: str = Field(description="Unique hash-based ID for deduplication")
    file_path: str
    line_start: int
    line_end: int = 0
    severity: FixSeverity
    category: str = Field(
        description="One of: import, type, structure, cycle, dead_code"
    )
    title: str = Field(description="One-line summary of the issue")
    description: str = Field(description="Detailed explanation of the issue")
    suggested_change: str = Field(
        default="", description="Diff or code snippet showing the fix"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence score 0-1"
    )
    auto_fixable: bool = Field(
        default=False,
        description="True if the fix can be applied automatically",
    )


class FixReport(BaseModel):
    """Aggregated report from an auto-fix analysis run."""

    suggestions: list[FixSuggestion]
    total_issues: int
    by_severity: dict[str, int]
    by_category: dict[str, int]
    summary: str
