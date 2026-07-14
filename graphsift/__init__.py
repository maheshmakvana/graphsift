"""graphsift — smarter code context for AI-powered development.

80-150x token reduction vs raw source. 86% avg CLI output compression.
F1 0.85 relevance accuracy vs 0.54 for code-review-graph.

Key capabilities:
- Ranked 0-1 relevance scoring with hot/warm/cold tier selection
- Hard token budget enforcement for context limits
- Diff-aware context trimming + entropy-based deduplication
- 14-language AST parsing + 11-language tree-sitter precise parsing
- 19 CLI command compressors (pytest 94%, grep 97%, git_diff 92%, docker 91%)
- Hybrid search (BM25 + TF-IDF sparse vector fusion)
- Auto-fix suggestions, cycle detection, dead code detection
- MCP server with 7 token-saving tools for Claude Code
- Cache-aware output with prompt-cache breakpoints

Quick start::

    from graphsift import ContextBuilder, ContextConfig, DiffSpec

    builder = ContextBuilder(ContextConfig(token_budget=50_000))
    builder.index_files(source_map)   # dict of path -> source text

    result = builder.build(
        DiffSpec(changed_files=["src/auth.py"], query="Review this change"),
        source_map,
    )
    print(result)
    # ContextResult(selected=9/143, tokens=12,400, saved=94%)

    # Paste result.rendered_context directly into your AI prompt

    # Monorepo support
    stats_list = builder.index_roots([pkg_a_map, pkg_b_map])

    # Incremental updates (skips unchanged files via SHA-256)
    builder.index_files_incremental(updated_source_map)

    # CLI output compression - save tokens on command output
    from graphsift import compress
    saved = compress(pytest_output, "pytest")
"""

from ._version import __version__
from .core import (
    BashParser,
    ContextBuilder,
    ContextSelector,
    DependencyGraph,
    GenericParser,
    HCLParser,
    LanguageParser,
    PythonParser,
    RelevanceRanker,
    detect_language,
    estimate_tokens,
    get_parser,
    register_parser,
)
from .parsers import (
    TreeSitterParser,
    register_tree_sitter_parsers,
)
from .exceptions import (
    AdapterError,
    BudgetExceededError,
    graphsiftError,
    ConfigurationError,
    GraphError,
    IndexError,
    LanguageNotSupportedError,
    ParseError,
    ValidationError,
)
from .models import (
    ContextConfig,
    ContextResult,
    DiffSpec,
    EdgeKind,
    FileNode,
    GraphEdge,
    GraphNode,
    FixReport,
    FixSeverity,
    FixSuggestion,
    IndexStats,
    Language,
    NodeKind,
    OutputMode,
    ScoredFile,
)
from .advanced import (
    AnalysisPipeline,
    ContextDiff,
    DiffValidator,
    GraphCache,
    SchemaEvolution,
    async_batch_build,
    async_batch_index,
    async_stream_context,
    batch_index,
    stream_context,
)
from .adapters.storage import GraphStore
from .auto_fix import FixSuggester
from .adapters.postprocess import (
    CommunityDetector,
    FlowDetector,
    Postprocessor,
    RefactorEngine,
    RiskScorer,
    WikiGenerator,
)
from .compress import compress, compress_tee, COMPRESSORS, detect_type as detect_command_type
from .analytics import gain, discover, history, record_call, reset as reset_analytics
from .hybrid_search import HybridSearcher
from .memory import AgentMemory, MemoryFact, SessionInfo
from .typed_retrieval import TypedRetriever, QueryIntent, TypedPath, TypedNeighborhood
from .compact_context import ConversationCompactor, AutonomousCompressor, CriticalFact, CompactionStats
from .evidence import EvidenceTracer, EvidenceResult, FileEvidence
from .a2a_server import A2AServer, build_agent_card, run_server as run_a2a_server
from .mcp_tasks import TaskManager, Task, TaskState, ToolRegistry, ToolCategory, ToolDef
from .harness import Harness, HarnessHook, DriftDetector, AgentAction, DriftAlert, HarnessStats
from .temporal_graph import TemporalGraph, TemporalStats, SymbolVersion, FileVersion, CommitInfo
from .code_memory import CodeMemory, CodeMemoryEntry, CodeMemoryStats

__all__ = [
    # Core
    "ContextBuilder",
    "ContextSelector",
    "DependencyGraph",
    "RelevanceRanker",
    "PythonParser",
    "GenericParser",
    "BashParser",
    "HCLParser",
    "LanguageParser",
    "TreeSitterParser",
    "register_tree_sitter_parsers",
    "detect_language",
    "estimate_tokens",
    "get_parser",
    "register_parser",
    # Models
    "ContextConfig",
    "ContextResult",
    "DiffSpec",
    "FileNode",
    "GraphNode",
    "GraphEdge",
    "ScoredFile",
    "IndexStats",
    "Language",
    "NodeKind",
    "EdgeKind",
    "OutputMode",
    "FixSeverity",
    "FixSuggestion",
    "FixReport",
    # Exceptions
    "graphsiftError",
    "ValidationError",
    "ConfigurationError",
    "ParseError",
    "IndexError",
    "GraphError",
    "AdapterError",
    "BudgetExceededError",
    "LanguageNotSupportedError",
    # Advanced
    "GraphCache",
    "AnalysisPipeline",
    "DiffValidator",
    "async_batch_index",
    "batch_index",
    "async_batch_build",
    "stream_context",
    "async_stream_context",
    "ContextDiff",
    "SchemaEvolution",
    # Storage
    "GraphStore",
    # Post-processing
    "Postprocessor",
    "FlowDetector",
    "CommunityDetector",
    "RiskScorer",
    "WikiGenerator",
    "RefactorEngine",
    # Compress & Analytics
    "compress",
    "compress_tee",
    "COMPRESSORS",
    "detect_command_type",
    "gain",
    "discover",
    "history",
    "record_call",
    "reset_analytics",
    # Hybrid Search
    "HybridSearcher",
    # Auto-fix
    "FixSuggester",
    # Agent Memory
    "AgentMemory",
    "MemoryFact",
    "SessionInfo",
    # Typed Retrieval
    "TypedRetriever",
    "QueryIntent",
    "TypedPath",
    "TypedNeighborhood",
    # Context Compaction
    "ConversationCompactor",
    "AutonomousCompressor",
    "CriticalFact",
    "CompactionStats",
    # Evidence
    "EvidenceTracer",
    "EvidenceResult",
    "FileEvidence",
    # A2A Protocol
    "A2AServer",
    "build_agent_card",
    "run_a2a_server",
    # MCP Tasks
    "TaskManager",
    "Task",
    "TaskState",
    "ToolRegistry",
    "ToolCategory",
    "ToolDef",
    # Harness
    "Harness",
    "HarnessHook",
    "DriftDetector",
    "AgentAction",
    "DriftAlert",
    "HarnessStats",
    # Temporal Graph
    "TemporalGraph",
    "TemporalStats",
    "SymbolVersion",
    "FileVersion",
    "CommitInfo",
    # Code Memory
    "CodeMemory",
    "CodeMemoryEntry",
    "CodeMemoryStats",
    # MCP / CLI
    "run_server",
]

from .mcp_server import run_server
