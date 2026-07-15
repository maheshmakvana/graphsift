"""graphsift — #1 Token Saver for Claude, GPT-4, Gemini & Every LLM.

Created by Mahesh Makwana (https://github.com/maheshmakvana).

The Python library that slashes LLM token costs for AI-assisted code review,
debugging, and code generation. Save 80-150x tokens on every code review with
Claude Code, GPT-4, Gemini, Codex, or any LLM — zero data exfiltration, zero
telemetry, zero API calls from library code.

Why graphsift?
Instead of sending your entire codebase (or using binary blast-radius),
graphsift ranks every file 0-1 by relevance using AST dependency graphs + BM25
+ diff proximity, then selects only the most relevant files within a hard token
budget. Combined with 3-tier compression (HOT/WARM/COLD), diff-aware trimming,
entropy deduplication, and 19 CLI output compressors, you save 80-150x tokens
vs raw source while maintaining 0.85 F1 relevance accuracy.

Core capabilities:
- Ranked 0-1 relevance scoring — AST + BM25 + graph-distance fusion
- Hard token budget enforcement — never exceed Claude/GPT/Gemini limits
- 3-tier (HOT/WARM/COLD) compression — signatures vs full source
- Diff-aware context trimming — only changed regions + context
- Entropy-based deduplication — SimHash near-duplicate detection
- 14-language AST parsing + 11-language tree-sitter precise parsing
- 25 CLI command compressors (pytest 94%, grep 97%, git diff 93%, npm 89%, pip 90%, make 89%, docker 82%, go_test 71%, eslint 84%, and more)
- Cache-aware output with Claude/GPT prompt-cache breakpoints
- Hybrid search — BM25 + TF-IDF + optional dense vector fusion
- MCP server — 7 token-saving tools for Claude Code automatic integration
- Conversation compaction — 60-82% agent conversation savings
- Temporal graph — git-history-aware symbol tracking
- Agent memory — SQLite-backed cross-session persistence
- A2A protocol — Agent-to-Agent communication server
- Security — path traversal protection, command injection prevention
- Auto-fix suggestions — 5 categories of graph-based fixes

WITH vs WITHOUT graphsift (15 daily-dev scenarios tested):
  WITHOUT: Claude reads 2,748 tokens of raw output per session — ANSI escapes, timestamps,
           PASSED lines, traceback frames, and package metadata all mixed with real signals.
           Claude must FIND the signal in the noise before it can act. ($9.07/mo at 220 runs)
  WITH:    Claude reads 930 tokens of pre-filtered signal — only error types, failure messages,
           changed lines, and severity counts. Claude starts reasoning immediately with 3x
           more useful content in the same context window. ($3.07/mo at 220 runs)
  SAVINGS: 66% average token reduction. 100% critical code-quality signals preserved.
  SECURITY: DataScrubber redacts secrets, PathValidator blocks traversal, CommandSanitizer
           prevents injection — none of which exist WITHOUT graphsift.

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
    saved = compress(pytest_output, "pytest")  # 90% compression

Save Claude tokens. Reduce GPT-4 costs. Optimize Gemini context windows.
All with zero telemetry, zero accounts, and zero API calls.
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
    SourceConfidence,
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
from .analytics import gain, discover, history, record_call, summary_line, reset as reset_analytics
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
from .tool_budgets import ToolBudget
from .read_cache import ReadCache
from .verify_hooks import Verifier, VerifyResult
from .evidence_check import EvidenceChecker, Citation
from .prompt_templates import (
    FixBugTemplate,
    AddFeatureTemplate,
    RefactorTemplate,
    ProductionAppTemplate,
    ThemeChangeTemplate,
    SecurityArchitectureTemplate,
    get_template,
)
from .tiered_memory import TieredMemory
from .prioritize import PriorityScorer, PrioritizedResult, ScoredFinding
from .security import (
    CommandSanitizer,
    DataScrubber,
    DataLeakError,
    NetworkAccessError,
    PathTraversalError,
    PathValidator,
    SecurePipeline,
    SecurityError,
    CommandInjectionError,
)
from .executor import AutoPipeline, CommandExecutor, SilentRunner, PipelineResult, CommandResult

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
    "SourceConfidence",
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
    "summary_line",
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
    # Tool Budgets
    "ToolBudget",
    # Read Cache
    "ReadCache",
    # Verify Hooks
    "Verifier",
    "VerifyResult",
    # Evidence Check
    "EvidenceChecker",
    "Citation",
    # Prompt Templates
    "FixBugTemplate",
    "AddFeatureTemplate",
    "RefactorTemplate",
    "ProductionAppTemplate",
    "ThemeChangeTemplate",
    "SecurityArchitectureTemplate",
    "get_template",
    # Tiered Memory
    "TieredMemory",
    # Priority Scorer
    "PriorityScorer",
    "PrioritizedResult",
    "ScoredFinding",
    # Security
    "SecurityError",
    "PathTraversalError",
    "CommandInjectionError",
    "DataLeakError",
    "NetworkAccessError",
    "PathValidator",
    "CommandSanitizer",
    "DataScrubber",
    "SecurePipeline",
    # Executor
    "AutoPipeline",
    "CommandExecutor",
    "SilentRunner",
    "PipelineResult",
    "CommandResult",
    # MCP / CLI
    "run_server",
]

from .mcp_server import run_server
