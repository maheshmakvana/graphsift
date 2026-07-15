"""Type stubs for graphsift package."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from graphsift._version import __version__ as __version__: str

# ── Core ──────────────────────────────────────────────────────────────────

class ContextConfig:
    """Configuration for context building."""
    token_budget: int
    diff_aware_trimming: bool
    output_mode: str  # "SMART" | "FULL" | "SIGNATURES" | "COMPRESSED"
    def __init__(
        self,
        token_budget: int = 50_000,
        diff_aware_trimming: bool = False,
        output_mode: str = "SMART",
    ) -> None: ...

class DiffSpec:
    """Describes a code diff for context building."""
    changed_files: List[str]
    diff_text: str
    def __init__(
        self,
        changed_files: List[str],
        diff_text: str = "",
    ) -> None: ...

class ContextResult:
    """Result of a context build operation."""
    files_selected: int
    total_tokens: int
    savings_pct: float
    rendered_context: str
    file_scores: List["ScoredFile"]
    tier_breakdown: Dict[str, int]
    cache_hit: bool

class ScoredFile:
    """A file with its relevance score."""
    path: str
    score: float
    tier: str  # "hot" | "warm" | "cold"
    tokens: int
    def __init__(self, path: str, score: float, tier: str, tokens: int) -> None: ...

class IndexStats:
    """Statistics from an indexing operation."""
    files_indexed: int
    symbols_found: int
    edges_created: int
    time_ms: float

class ContextBuilder:
    """Builds ranked context for code diffs."""
    def __init__(self, config: ContextConfig) -> None: ...
    def index_files(self, source_map: Dict[str, str]) -> IndexStats: ...
    def index_files_incremental(self, source_map: Dict[str, str]) -> IndexStats: ...
    def index_roots(self, root_maps: List[Dict[str, str]]) -> List[IndexStats]: ...
    def build(self, diff_spec: DiffSpec, source_map: Optional[Dict[str, str]] = None) -> ContextResult: ...

class DependencyGraph:
    """AST dependency graph for a codebase."""
    def __init__(self) -> None: ...
    def build(self, source_map: Dict[str, str]) -> IndexStats: ...
    def get_dependents(self, filepath: str, depth: int = 1) -> List[str]: ...
    def get_dependencies(self, filepath: str) -> List[str]: ...
    def detect_cycles(self) -> List[List[str]]: ...
    def detect_dead_code(self, entry_points: Optional[List[str]] = None) -> List[str]: ...

class RelevanceRanker:
    """Scores files by relevance to a diff/query."""
    def __init__(self, graph: DependencyGraph) -> None: ...
    def rank(self, changed_files: List[str], source_map: Dict[str, str], query: str = "") -> List[ScoredFile]: ...

# ── Parsers ───────────────────────────────────────────────────────────────

class LanguageParser:
    """Base class for language parsers."""
    language: str
    def parse_file(self, path: str, content: str) -> List["FileNode"]: ...

class PythonParser(LanguageParser): ...
class GenericParser(LanguageParser): ...
class BashParser(LanguageParser): ...
class HCLParser(LanguageParser): ...

def detect_language(path: str, content: str = "") -> str: ...
def estimate_tokens(text: str) -> int: ...
def get_parser(language: str) -> Optional[LanguageParser]: ...
def register_parser(language: str, parser: LanguageParser) -> None: ...

# ── Models ────────────────────────────────────────────────────────────────

class FileNode:
    """A node in the dependency graph representing a file or symbol."""
    name: str
    file_path: str
    kind: "NodeKind"
    language: str
    line_start: int
    line_end: int

class GraphNode:
    """A generic graph node."""
    id: str
    label: str
    metadata: Dict[str, Any]

class GraphEdge:
    """A directed edge in the dependency graph."""
    source: str
    target: str
    kind: "EdgeKind"
    weight: float

class NodeKind:
    """Enum of node kinds."""
    FILE: "NodeKind"
    CLASS: "NodeKind"
    FUNCTION: "NodeKind"
    METHOD: "NodeKind"
    MODULE: "NodeKind"
    VARIABLE: "NodeKind"

class EdgeKind:
    """Enum of edge kinds."""
    CALLS: "EdgeKind"
    IMPORTS: "EdgeKind"
    INHERITS: "EdgeKind"
    DECORATES: "EdgeKind"
    REFERENCES: "EdgeKind"
    TEST_COVERS: "EdgeKind"
    DYNAMIC_IMPORT: "EdgeKind"

class Language:
    """Enum of supported languages."""
    PYTHON: "Language"
    JAVASCRIPT: "Language"
    TYPESCRIPT: "Language"
    GO: "Language"
    RUST: "Language"
    JAVA: "Language"
    CPP: "Language"
    C: "Language"
    RUBY: "Language"
    PHP: "Language"
    BASH: "Language"
    TERRAFORM: "Language"
    HELM: "Language"
    DOCKERFILE: "Language"

class OutputMode:
    """Enum of output modes."""
    SMART: "OutputMode"
    FULL: "OutputMode"
    SIGNATURES: "OutputMode"
    COMPRESSED: "OutputMode"

class FixSeverity:
    """Enum of fix severity levels."""
    CRITICAL: "FixSeverity"
    HIGH: "FixSeverity"
    MEDIUM: "FixSeverity"
    LOW: "FixSeverity"
    INFO: "FixSeverity"

class FixSuggestion:
    """A suggested fix for a code issue."""
    description: str
    severity: "FixSeverity"
    file: str
    line: int

class FixReport:
    """A report of fix suggestions."""
    suggestions: List[FixSuggestion]
    summary: str

# ── Exceptions ────────────────────────────────────────────────────────────

class graphsiftError(Exception): ...
class ValidationError(graphsiftError): ...
class ConfigurationError(graphsiftError): ...
class ParseError(graphsiftError): ...
class IndexError(graphsiftError): ...
class GraphError(graphsiftError): ...
class AdapterError(graphsiftError): ...
class BudgetExceededError(graphsiftError): ...
class LanguageNotSupportedError(graphsiftError): ...

# ── Advanced ──────────────────────────────────────────────────────────────

class GraphCache:
    """Cache for dependency graph lookups."""
    def __init__(self, max_size: int = 1000, ttl: int = 300) -> None: ...

class AnalysisPipeline:
    """Pipeline for batched code analysis."""
    def __init__(self, steps: List[Any]) -> None: ...
    def run(self, source_map: Dict[str, str]) -> Any: ...

class DiffValidator:
    """Validates diffs for consistency."""
    def validate(self, diff_text: str) -> Dict[str, Any]: ...

class ContextDiff:
    """Represents a diff between two context states."""
    pass

class SchemaEvolution:
    """Manages schema migrations for GraphStore."""
    def migrate(self, db_path: str) -> bool: ...

async def async_batch_build(builders: List[ContextBuilder], specs: List[DiffSpec]) -> List[ContextResult]: ...
async def async_batch_index(builders: List[ContextBuilder], root_maps: List[Dict[str, str]]) -> List[IndexStats]: ...
async def async_stream_context(builder: ContextBuilder, spec: DiffSpec) -> ContextResult: ...
def batch_index(builders: List[ContextBuilder], root_maps: List[Dict[str, str]]) -> List[IndexStats]: ...
def stream_context(builder: ContextBuilder, spec: DiffSpec) -> ContextResult: ...

# ── Storage ───────────────────────────────────────────────────────────────

class GraphStore:
    """SQLite-backed persistence for dependency graphs."""
    def __init__(self, db_path: str = "") -> None: ...
    def load(self) -> Optional[DependencyGraph]: ...
    def save(self, graph: DependencyGraph) -> None: ...
    def clear(self) -> None: ...

# ── Post-processing ───────────────────────────────────────────────────────

class Postprocessor:
    """Post-processing pipeline for analysis results."""
    def process(self, graph: DependencyGraph) -> Any: ...

class FlowDetector:
    """Detects data/control flows in the dependency graph."""
    def detect(self, graph: DependencyGraph, entry_points: List[str]) -> List[Dict[str, Any]]: ...

class CommunityDetector:
    """Detects module communities via graph clustering."""
    def detect(self, graph: DependencyGraph) -> List[List[str]]: ...

class RiskScorer:
    """Assigns risk scores to modules based on graph centrality."""
    def score(self, graph: DependencyGraph) -> Dict[str, float]: ...

class WikiGenerator:
    """Generates wiki documentation from graph analysis."""
    def generate(self, graph: DependencyGraph) -> str: ...

class RefactorEngine:
    """Suggests refactoring opportunities from graph analysis."""
    def analyze(self, graph: DependencyGraph) -> List[FixSuggestion]: ...

# ── Compression ───────────────────────────────────────────────────────────

COMPRESSORS: Dict[str, Callable[[str], str]]

def compress(text: str, cmd_type: str = "") -> str: ...
def compress_tee(text: str, cmd_type: str = "") -> Tuple[str, str, int]: ...
def detect_type(text: str) -> str: ...

# ── Analytics ─────────────────────────────────────────────────────────────

def gain(limit: int = 20) -> Dict[str, Any]: ...
def discover() -> Dict[str, Any]: ...
def history(days: int = 30) -> List[Dict[str, Any]]: ...
def record_call(tool: str, input_tokens: int, output_tokens: int, saved_tokens: int, cost: float = 0.0) -> None: ...
def reset_analytics() -> None: ...

# ── Hybrid Search ─────────────────────────────────────────────────────────

class HybridSearcher:
    """BM25 + TF-IDF hybrid search for code."""
    def __init__(self) -> None: ...
    def index(self, documents: Dict[str, str]) -> None: ...
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]: ...

# ── Agent Memory ──────────────────────────────────────────────────────────

class AgentMemory:
    """SQLite-backed agent memory for cross-session persistence."""
    def __init__(self, db_path: str = "") -> None: ...
    def store(self, fact: "MemoryFact") -> str: ...
    def recall(self, query: str, top_k: int = 10) -> List["MemoryFact"]: ...

class MemoryFact:
    """A single fact stored in agent memory."""
    id: str
    content: str
    importance: float
    created_at: str

class SessionInfo:
    """Metadata about an agent session."""
    session_id: str
    created_at: str
    memory_count: int

# ── Typed Retrieval ───────────────────────────────────────────────────────

class TypedRetriever:
    """PRISM-style typed graph traversal."""
    def __init__(self, graph: DependencyGraph) -> None: ...
    def query(self, symbols: List[str], intent: "QueryIntent", max_depth: int = 2) -> List[Dict[str, Any]]: ...

class QueryIntent:
    """Enum of query intents for typed retrieval."""
    SECURITY: "QueryIntent"
    REFACTOR: "QueryIntent"
    TEST: "QueryIntent"
    DEPENDENCY: "QueryIntent"
    ARCHITECTURE: "QueryIntent"
    GENERAL: "QueryIntent"

class TypedPath:
    """A typed path through the graph."""
    source: str
    target: str
    edges: List[Tuple[str, str]]

class TypedNeighborhood:
    """A typed neighborhood around a set of symbols."""
    center: str
    nodes: List[str]
    edges: List[Tuple[str, str, str]]

# ── Context Compaction ────────────────────────────────────────────────────

class ConversationCompactor:
    """Compresses agent conversations with multiple strategies."""
    def compress(self, messages: List[Dict[str, str]], strategy: str = "signature") -> List[Dict[str, str]]: ...

class AutonomousCompressor:
    """Auto-selects the best compression strategy."""
    def compress(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]: ...

class CriticalFact:
    """An important fact extracted from conversation."""
    content: str
    source: str

class CompactionStats:
    """Statistics from a compaction run."""
    original_tokens: int
    compressed_tokens: int
    savings_pct: float
    strategy_used: str

# ── Evidence ──────────────────────────────────────────────────────────────

class EvidenceTracer:
    """Creates audit trails for file selection decisions."""
    def __init__(self) -> None: ...
    def trace(self, file: str, reason: str, score: float) -> None: ...

class EvidenceResult:
    """A single audit trail entry."""
    file: str
    reason: str
    score: float

class FileEvidence:
    """Collection of evidence for a file."""
    file: str
    evidence: List[EvidenceResult]

# ── A2A Protocol ──────────────────────────────────────────────────────────

class A2AServer:
    """Agent-to-Agent protocol server using JSON-RPC over HTTP."""
    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None: ...
    def start(self) -> None: ...

def build_agent_card(name: str, capabilities: List[str]) -> Dict[str, Any]: ...
def run_a2a_server(host: str = "127.0.0.1", port: int = 8765) -> None: ...

# ── MCP Tasks ─────────────────────────────────────────────────────────────

class TaskManager:
    """Manages async MCP tasks with progress tracking."""
    def create_task(self, name: str, coro: Any) -> str: ...
    def cancel_task(self, task_id: str) -> bool: ...
    def get_task(self, task_id: str) -> Optional["Task"]: ...

class Task:
    """An async task managed by TaskManager."""
    id: str
    name: str
    state: "TaskState"
    progress: float

class TaskState:
    """Enum of task states."""
    PENDING: "TaskState"
    RUNNING: "TaskState"
    COMPLETED: "TaskState"
    FAILED: "TaskState"
    CANCELLED: "TaskState"

class ToolRegistry:
    """Registry of MCP tools."""
    def register(self, name: str, handler: Callable, category: str = "") -> None: ...

class ToolCategory:
    """Enum of tool categories."""
    CORE: "ToolCategory"
    ANALYSIS: "ToolCategory"
    MEMORY: "ToolCategory"
    SEARCH: "ToolCategory"
    UTILITY: "ToolCategory"

class ToolDef:
    """Definition of an MCP tool."""
    name: str
    description: str
    category: str
    handler: Callable

# ── Harness ───────────────────────────────────────────────────────────────

class Harness:
    """Agent harness with validation hooks."""
    def __init__(self, name: str = "default") -> None: ...
    def add_hook(self, hook: "HarnessHook") -> None: ...
    def run(self, action: "AgentAction") -> "HarnessStats": ...

class HarnessHook:
    """Pre/post validation hook."""
    def pre(self, action: "AgentAction") -> Optional[str]: ...
    def post(self, action: "AgentAction", result: Any) -> Optional[str]: ...

class DriftDetector:
    """Detects behavioral drift in agent actions."""
    def record(self, action: "AgentAction") -> None: ...
    def check(self) -> List["DriftAlert"]: ...

class AgentAction:
    """An action performed by an agent."""
    tool: str
    input: Dict[str, Any]
    output: Any
    duration_ms: float

class DriftAlert:
    """An alert about detected drift."""
    metric: str
    expected: float
    actual: float
    severity: str

class HarnessStats:
    """Statistics from a harness run."""
    total_actions: int
    total_duration_ms: float
    hooks_passed: int
    hooks_failed: int
    drift_alerts: List[DriftAlert]

# ── Temporal Graph ────────────────────────────────────────────────────────

class TemporalGraph:
    """Git-history-aware dependency graph with bi-temporal queries."""
    def __init__(self, repo_path: str, graph: Optional[DependencyGraph] = None) -> None: ...
    def index_history(self, max_commits: int = 500, branch: str = "HEAD") -> "TemporalStats": ...
    def symbols_at(self, commit_hash: str) -> List[str]: ...
    def introduced_in(self, symbol_name: str) -> Optional[str]: ...
    def last_modified(self, symbol_name: str) -> Optional[str]: ...
    def age_boost(self, filepath: str, current_commit: str = "HEAD") -> float: ...
    def diff_between(self, from_commit: str, to_commit: str) -> List[str]: ...
    def history_of(self, symbol_name: str) -> List["SymbolVersion"]: ...
    def file_history_of(self, filepath: str) -> List["FileVersion"]: ...

class TemporalStats:
    """Summary statistics from index_history()."""
    commits_indexed: int
    symbols_tracked: int
    files_tracked: int
    oldest_commit: str
    newest_commit: str
    time_span_days: int
    renames_detected: int

class SymbolVersion:
    """One commit-level change event for a symbol."""
    qualified_name: str
    commit_hash: str
    timestamp: Any  # datetime
    action: str
    filepath: str
    line_start: Optional[int]
    line_end: Optional[int]

class FileVersion:
    """One commit-level change event for a file."""
    filepath: str
    commit_hash: str
    timestamp: Any  # datetime
    action: str
    previous_path: Optional[str]

class CommitInfo:
    """Parsed metadata for a git commit."""
    hash: str
    author: str
    timestamp: Any  # datetime
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int

# ── Code Memory ───────────────────────────────────────────────────────────

class CodeMemory:
    """Code-anchored agent memory with SQLite persistence."""
    def __init__(self, db_path: Optional[str] = None, graph: Optional[DependencyGraph] = None) -> None: ...
    def remember(self, content: str, linked_symbols: Optional[List[str]] = None,
                 linked_files: Optional[List[str]] = None, memory_type: str = "note",
                 importance: float = 0.5, ttl_days: Optional[int] = None,
                 tags: Optional[List[str]] = None, session_id: Optional[str] = None) -> str: ...
    def recall_for_diff(self, changed_files: List[str], changed_symbols: Optional[List[str]] = None,
                        query: Optional[str] = None, top_k: int = 10) -> List["CodeMemoryEntry"]: ...
    def recall(self, query: str, top_k: int = 10) -> List["CodeMemoryEntry"]: ...
    def recall_for_symbol(self, symbol_name: str) -> List["CodeMemoryEntry"]: ...
    def recall_for_file(self, filepath: str) -> List["CodeMemoryEntry"]: ...
    def forget(self, memory_id: str) -> bool: ...
    def decay(self) -> int: ...
    def summarize_for_context(self, changed_files: List[str], max_tokens: int = 300) -> str: ...

class CodeMemoryEntry:
    """A single code-anchored memory entry."""
    memory_id: str
    content: str
    memory_type: str
    importance: float
    created_at: Any  # datetime
    last_accessed: Any  # datetime
    access_count: int
    valid_until: Any  # datetime
    invalid_at: Optional[Any]  # datetime
    session_id: Optional[str]
    linked_symbols: List[str]
    linked_files: List[str]
    tags: List[str]

class CodeMemoryStats:
    """Aggregate statistics for a CodeMemory store."""
    total_memories: int
    active_memories: int
    expired_memories: int
    by_type: Dict[str, int]
    avg_importance: float
    total_accesses: int

# ── Tool Budgets ──────────────────────────────────────────────────────────

class ToolBudget:
    """Per-tool output budget enforcer."""
    def __init__(self, budgets: Optional[Dict[str, int]] = None) -> None: ...
    def apply(self, tool: str, text: str, extract_structured: bool = False) -> str: ...
    def get_budget(self, tool: str) -> int: ...
    def set_budget(self, tool: str, max_lines: int) -> None: ...

# ── Read Cache ────────────────────────────────────────────────────────────

class ReadCache:
    """Session-scoped file read cache with fingerprint dedup."""
    def __init__(self) -> None: ...
    def read(self, path: str, reader: Callable[[], str]) -> str: ...
    def invalidate(self, path: str) -> None: ...
    def clear(self) -> None: ...

# ── Verify Hooks ──────────────────────────────────────────────────────────

class Verifier:
    """Lightweight verifier for changed files."""
    def __init__(self, project_root: str = "", python_path: str = "") -> None: ...
    def check(self, file_path: str) -> "VerifyResult": ...
    def lint(self, file_path: str) -> Tuple[bool, str]: ...

class VerifyResult:
    """Result of a verification check."""
    file: str
    syntax_ok: bool
    syntax_error: str
    lint_ok: bool
    lint_output: str
    passed: bool

# ── Evidence Check ────────────────────────────────────────────────────────

class EvidenceChecker:
    """Validates file:line citations against the filesystem."""
    def __init__(self, project_root: str = "") -> None: ...
    def check_response(self, text: str) -> List["Citation"]: ...

class Citation:
    """A single file:line citation found in text."""
    raw: str
    file_path: str
    line: Optional[int]
    valid: bool
    error: str

# ── Prompt Templates ──────────────────────────────────────────────────────

class FixBugTemplate:
    """Template for bug-fix prompts."""
    @staticmethod
    def render(context: str, bug_description: str) -> str: ...

class AddFeatureTemplate:
    """Template for feature addition prompts."""
    @staticmethod
    def render(context: str, feature_spec: str) -> str: ...

class RefactorTemplate:
    """Template for refactoring prompts."""
    @staticmethod
    def render(context: str, refactor_goal: str) -> str: ...

class ProductionAppTemplate:
    """Template for production-ready code prompts."""
    @staticmethod
    def render(context: str, requirements: str) -> str: ...

class ThemeChangeTemplate:
    """Template for theme/UI change prompts."""
    @staticmethod
    def render(context: str, theme_spec: str) -> str: ...

class SecurityArchitectureTemplate:
    """Template for security architecture prompts."""
    @staticmethod
    def render(context: str, threat_model: str) -> str: ...

def get_template(name: str) -> Any: ...

# ── Tiered Memory ─────────────────────────────────────────────────────────

class TieredMemory:
    """Manages tiered memory for token-efficient context loading."""
    repo_root: str
    axioms: List[str]
    rules_dir: str
    topic_files: Dict[str, str]
    def __init__(self, repo_root: str) -> None: ...
    def get_tier(self, tier: str) -> List[str]: ...
    def add_axiom(self, axiom: str) -> None: ...
    def load_topic(self, topic: str) -> Optional[str]: ...
    def estimate_tokens(self, tier: str) -> int: ...

# ── Priority Scorer ───────────────────────────────────────────────────────

class PriorityScorer:
    """Multi-signal priority scoring for findings."""
    def __init__(self) -> None: ...
    def score(self, finding: "ScoredFinding") -> float: ...

class PrioritizedResult:
    """A prioritized collection of findings."""
    findings: List["ScoredFinding"]
    total_score: float

class ScoredFinding:
    """A finding with a priority score."""
    description: str
    file: str
    line: int
    score: float
    signals: Dict[str, float]

# ── Security ──────────────────────────────────────────────────────────────

class SecurityError(Exception): ...
class PathTraversalError(SecurityError): ...
class CommandInjectionError(SecurityError): ...
class DataLeakError(SecurityError): ...
class NetworkAccessError(SecurityError): ...

class PathValidator:
    """Validates file paths to prevent traversal attacks."""
    def validate(self, path: str, base_dir: str) -> str: ...

class CommandSanitizer:
    """Sanitizes shell commands to prevent injection."""
    def sanitize(self, command: str) -> str: ...

class DataScrubber:
    """Scrubs sensitive data from output."""
    def scrub(self, text: str) -> str: ...

class SecurePipeline:
    """Composed security validation pipeline."""
    def run(self, path: str, command: str, output: str) -> Tuple[str, str, str]: ...

# ── Executor ──────────────────────────────────────────────────────────────

class AutoPipeline:
    """Auto-configuring execution pipeline."""
    def run(self, commands: List[str]) -> List["CommandResult"]: ...

class CommandExecutor:
    """Safe command execution with guards."""
    def execute(self, command: str, timeout: int = 30) -> "CommandResult": ...

class SilentRunner:
    """Silent command runner (PowerShell fallback on Windows)."""
    def run(self, command: str) -> "PipelineResult": ...

class PipelineResult:
    """Result from a pipeline run."""
    success: bool
    output: str
    error: str
    duration_ms: float

class CommandResult:
    """Result from a single command execution."""
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float

# ── MCP Server ────────────────────────────────────────────────────────────

def run_server() -> None: ...

# ── Auto-fix ──────────────────────────────────────────────────────────────

class FixSuggester:
    """Suggests fixes for code issues found in the graph."""
    def __init__(self, graph: DependencyGraph) -> None: ...
    def suggest(self) -> List[FixSuggestion]: ...

# ── Tree-sitter ───────────────────────────────────────────────────────────

class TreeSitterParser:
    """Tree-sitter based parser for precise AST parsing."""
    def parse(self, language: str, source: bytes) -> Any: ...

def register_tree_sitter_parsers() -> None: ...

# ── Context Selector ──────────────────────────────────────────────────────

class ContextSelector:
    """Selects optimal context based on relevance scores."""
    def select(self, scored_files: List[ScoredFile], budget: int) -> List[ScoredFile]: ...

# ── MCP Shrink ────────────────────────────────────────────────────────────

# No public exports from mcp_shrink
