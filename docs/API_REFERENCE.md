# graphsift API Reference

> **Version:** 4.8.0 · Created by [Mahesh Makwana](https://github.com/maheshmakvana) · [Source](https://github.com/maheshmakvana/graphsift/tree/master/graphsift)
>
> *Save Claude tokens. Reduce GPT costs. Optimize Gemini context windows. 80-150x token reduction.*

---

## Core

| Class / Function | Description | Source |
|---|---|---|
| `ContextBuilder` | Builds ranked context for a code diff from a source map | [`core.py`](../graphsift/core.py) |
| `ContextConfig` | Configuration for context building (token budget, tiers, trimming) | [`models.py`](../graphsift/models.py) |
| `DiffSpec` | Describes a code change to build context for | [`models.py`](../graphsift/models.py) |
| `ContextResult` | Result object with selected files, token count, rendered context | [`models.py`](../graphsift/models.py) |
| `ScoredFile` | A file with its 0–1 relevance score and tier assignment | [`models.py`](../graphsift/models.py) |
| `IndexStats` | Statistics from an indexing operation | [`models.py`](../graphsift/models.py) |
| `DependencyGraph` | AST dependency graph construction and querying | [`core.py`](../graphsift/core.py) |
| `RelevanceRanker` | Scores files 0–1 using AST + BM25 + graph proximity | [`core.py`](../graphsift/core.py) |
| `ContextSelector` | Selects optimal context from scored files within token budget | [`core.py`](../graphsift/core.py) |
| `LanguageParser` | Base class for language-specific parsers | [`core.py`](../graphsift/core.py) |
| `PythonParser` | Python AST parser | [`core.py`](../graphsift/core.py) |
| `GenericParser` | Fallback regex-based parser for languages without tree-sitter | [`core.py`](../graphsift/core.py) |
| `BashParser` | Shell script parser | [`core.py`](../graphsift/core.py) |
| `HCLParser` | HCL/Terraform parser | [`core.py`](../graphsift/core.py) |
| `detect_language()` | Detect programming language from file path and content | [`core.py`](../graphsift/core.py) |
| `estimate_tokens()` | Rough token count estimate (~4 chars per token) | [`core.py`](../graphsift/core.py) |
| `get_parser()` | Get parser for a given language | [`core.py`](../graphsift/core.py) |
| `register_parser()` | Register a custom language parser | [`core.py`](../graphsift/core.py) |

---

## Models

| Class / Function | Description | Source |
|---|---|---|
| `FileNode` | Node in the dependency graph (file or symbol) | [`models.py`](../graphsift/models.py) |
| `GraphNode` | Generic graph node with metadata | [`models.py`](../graphsift/models.py) |
| `GraphEdge` | Directed edge in the dependency graph | [`models.py`](../graphsift/models.py) |
| `Language` | Enum of supported languages (14 total) | [`models.py`](../graphsift/models.py) |
| `NodeKind` | Enum of node types (FILE, CLASS, FUNCTION, etc.) | [`models.py`](../graphsift/models.py) |
| `EdgeKind` | Enum of edge types (CALLS, IMPORTS, INHERITS, etc.) | [`models.py`](../graphsift/models.py) |
| `OutputMode` | Enum of output modes (SMART, FULL, SIGNATURES, COMPRESSED) | [`models.py`](../graphsift/models.py) |
| `FixSeverity` | Enum of fix severity levels | [`models.py`](../graphsift/models.py) |
| `FixSuggestion` | A suggested fix for a code issue | [`models.py`](../graphsift/models.py) |
| `FixReport` | A report containing fix suggestions | [`models.py`](../graphsift/models.py) |

---

## Compression & Analytics

| Class / Function | Description | Source |
|---|---|---|
| `compress()` | Compress CLI tool output (25 command types, auto-detect). Saves 25-95% tokens across pytest, git diff, npm, eslint, docker, kubectl, go test, cargo, terraform, pip, make, brew, dotnet, grep, git log/status, and more | [`compress.py`](../graphsift/compress.py) |
| `compress_tee()` | Compress and return both original and compressed | [`compress.py`](../graphsift/compress.py) |
| `COMPRESSORS` | Dict of 25 command-type → compressor function | [`compress.py`](../graphsift/compress.py) |
| `detect_command_type()` | Auto-detect command type from output text (100% accuracy on 15 tested types) | [`compress.py`](../graphsift/compress.py) |
| `gain()` | Show cumulative token savings analytics | [`analytics.py`](../graphsift/analytics.py) |
| `discover()` | Find missed token-saving opportunities | [`analytics.py`](../graphsift/analytics.py) |
| `history()` | Token savings history for the last N days | [`analytics.py`](../graphsift/analytics.py) |
| `record_call()` | Record a tool call for analytics | [`analytics.py`](../graphsift/analytics.py) |
| `reset_analytics()` | Reset all analytics data | [`analytics.py`](../graphsift/analytics.py) |

---

## Search

| Class / Function | Description | Source |
|---|---|---|
| `HybridSearcher` | BM25 + TF-IDF sparse vector fusion search | [`hybrid_search.py`](../graphsift/hybrid_search.py) |

---

## Agent Memory & Intelligence

| Class / Function | Description | Source |
|---|---|---|
| `AgentMemory` | SQLite-backed agent memory for cross-session persistence | [`memory.py`](../graphsift/memory.py) |
| `MemoryFact` | A single fact stored in agent memory | [`memory.py`](../graphsift/memory.py) |
| `SessionInfo` | Metadata about an agent session | [`memory.py`](../graphsift/memory.py) |
| `CodeMemory` | Code-anchored agent memory with SQLite persistence | [`code_memory.py`](../graphsift/code_memory.py) |
| `CodeMemoryEntry` | A single code-anchored memory entry | [`code_memory.py`](../graphsift/code_memory.py) |
| `CodeMemoryStats` | Aggregate store statistics | [`code_memory.py`](../graphsift/code_memory.py) |
| `TypedRetriever` | PRISM-style typed graph traversal (6 query intents) | [`typed_retrieval.py`](../graphsift/typed_retrieval.py) |
| `QueryIntent` | Enum of query intents (SECURITY, REFACTOR, TEST, etc.) | [`typed_retrieval.py`](../graphsift/typed_retrieval.py) |
| `TypedPath` | A typed path through the graph | [`typed_retrieval.py`](../graphsift/typed_retrieval.py) |
| `TypedNeighborhood` | A typed neighborhood around symbols | [`typed_retrieval.py`](../graphsift/typed_retrieval.py) |
| `TieredMemory` | Hierarchical memory (axioms → rules → topic → archive) | [`tiered_memory.py`](../graphsift/tiered_memory.py) |
| `ConversationCompactor` | Compress agent conversations (3 strategies) | [`compact_context.py`](../graphsift/compact_context.py) |
| `AutonomousCompressor` | Auto-selects best compression strategy | [`compact_context.py`](../graphsift/compact_context.py) |
| `CriticalFact` | Important fact extracted from conversation | [`compact_context.py`](../graphsift/compact_context.py) |
| `CompactionStats` | Statistics from a compaction run | [`compact_context.py`](../graphsift/compact_context.py) |

---

## Temporal & Graph Analysis

| Class / Function | Description | Source |
|---|---|---|
| `TemporalGraph` | Git-history-aware symbol tracking with bi-temporal queries | [`temporal_graph.py`](../graphsift/temporal_graph.py) |
| `TemporalStats` | Summary statistics from index_history() | [`temporal_graph.py`](../graphsift/temporal_graph.py) |
| `SymbolVersion` | One commit-level change event for a symbol | [`temporal_graph.py`](../graphsift/temporal_graph.py) |
| `FileVersion` | One commit-level change event for a file | [`temporal_graph.py`](../graphsift/temporal_graph.py) |
| `CommitInfo` | Parsed commit metadata | [`temporal_graph.py`](../graphsift/temporal_graph.py) |
| `PriorityScorer` | Multi-signal priority scoring for findings | [`prioritize.py`](../graphsift/prioritize.py) |
| `PrioritizedResult` | A prioritized collection of findings | [`prioritize.py`](../graphsift/prioritize.py) |
| `ScoredFinding` | A finding with a priority score | [`prioritize.py`](../graphsift/prioritize.py) |

---

## Evidence & Verification

| Class / Function | Description | Source |
|---|---|---|
| `EvidenceTracer` | Creates audit trails for file selection decisions | [`evidence.py`](../graphsift/evidence.py) |
| `EvidenceResult` | A single audit trail entry | [`evidence.py`](../graphsift/evidence.py) |
| `FileEvidence` | Collection of evidence for a file | [`evidence.py`](../graphsift/evidence.py) |
| `EvidenceChecker` | Validates file:line citations against filesystem | [`evidence_check.py`](../graphsift/evidence_check.py) |
| `Citation` | A single file:line citation found in text | [`evidence_check.py`](../graphsift/evidence_check.py) |
| `Verifier` | Post-change syntax/lint verification hooks | [`verify_hooks.py`](../graphsift/verify_hooks.py) |
| `VerifyResult` | Result of a verification check | [`verify_hooks.py`](../graphsift/verify_hooks.py) |

---

## Tool Budget & Caching

| Class / Function | Description | Source |
|---|---|---|
| `ToolBudget` | Per-tool output line caps with ANSI stripping | [`tool_budgets.py`](../graphsift/tool_budgets.py) |
| `ReadCache` | SHA-256 fingerprint dedup for file reads | [`read_cache.py`](../graphsift/read_cache.py) |

---

## Advanced / Async

| Class / Function | Description | Source |
|---|---|---|
| `GraphCache` | Cache for dependency graph lookups | [`advanced.py`](../graphsift/advanced.py) |
| `AnalysisPipeline` | Pipeline for batched code analysis | [`advanced.py`](../graphsift/advanced.py) |
| `DiffValidator` | Validates diffs for consistency | [`advanced.py`](../graphsift/advanced.py) |
| `ContextDiff` | Represents a diff between context states | [`advanced.py`](../graphsift/advanced.py) |
| `SchemaEvolution` | Manages GraphStore schema migrations | [`advanced.py`](../graphsift/advanced.py) |
| `async_batch_build()` | Async batch context building | [`advanced.py`](../graphsift/advanced.py) |
| `async_batch_index()` | Async batch indexing | [`advanced.py`](../graphsift/advanced.py) |
| `async_stream_context()` | Streaming async context building | [`advanced.py`](../graphsift/advanced.py) |
| `batch_index()` | Synchronous batch indexing | [`advanced.py`](../graphsift/advanced.py) |
| `stream_context()` | Streaming context building | [`advanced.py`](../graphsift/advanced.py) |

---

## Storage & Adapters

| Class / Function | Description | Source |
|---|---|---|
| `GraphStore` | SQLite-backed persistence for dependency graphs | [`adapters/storage.py`](../graphsift/adapters/storage.py) |
| `Postprocessor` | Post-processing pipeline for analysis results | [`adapters/postprocess.py`](../graphsift/adapters/postprocess.py) |
| `FlowDetector` | Detects data/control flows in the graph | [`adapters/postprocess.py`](../graphsift/adapters/postprocess.py) |
| `CommunityDetector` | Detects module communities via clustering | [`adapters/postprocess.py`](../graphsift/adapters/postprocess.py) |
| `RiskScorer` | Assigns risk scores based on graph centrality | [`adapters/postprocess.py`](../graphsift/adapters/postprocess.py) |
| `WikiGenerator` | Generates wiki docs from graph analysis | [`adapters/postprocess.py`](../graphsift/adapters/postprocess.py) |
| `RefactorEngine` | Suggests refactoring opportunities | [`adapters/postprocess.py`](../graphsift/adapters/postprocess.py) |

---

## Security

| Class / Function | Description | Source |
|---|---|---|
| `PathValidator` | Prevents path traversal attacks | [`security.py`](../graphsift/security.py) |
| `CommandSanitizer` | Sanitizes shell commands to prevent injection | [`security.py`](../graphsift/security.py) |
| `DataScrubber` | Scrubs sensitive patterns from output | [`security.py`](../graphsift/security.py) |
| `SecurePipeline` | Composed security validation pipeline | [`security.py`](../graphsift/security.py) |
| `SecurityError` | Base security exception | [`security.py`](../graphsift/security.py) |
| `PathTraversalError` | Path traversal detected | [`security.py`](../graphsift/security.py) |
| `CommandInjectionError` | Command injection detected | [`security.py`](../graphsift/security.py) |
| `DataLeakError` | Potential data leak detected | [`security.py`](../graphsift/security.py) |
| `NetworkAccessError` | Unauthorized network access attempt | [`security.py`](../graphsift/security.py) |

---

## Execution

| Class / Function | Description | Source |
|---|---|---|
| `CommandExecutor` | Safe command execution with guards | [`executor.py`](../graphsift/executor.py) |
| `ProcessRunner` | Cross-platform runner with tiered fallback | [`executor.py`](../graphsift/executor.py) |
| `SilentRunner` | Silent command runner (PowerShell fallback) | [`executor.py`](../graphsift/executor.py) |
| `AutoPipeline` | Auto-configuring execution pipeline | [`executor.py`](../graphsift/executor.py) |
| `PipelineResult` | Result from a pipeline run | [`executor.py`](../graphsift/executor.py) |
| `CommandResult` | Result from a single command | [`executor.py`](../graphsift/executor.py) |

## Smart Execution (v4.8.0+)

| Class / Function | Description | Source |
|---|---|---|
| `Daemon` | Persistent Python daemon — keeps modules loaded + caches results | [`daemon.py`](../graphsift/daemon.py) **NEW** |
| `hooks.pre_bash_hook` | PreToolUse hook — intercepts Bash/PowerShell, routes through daemon | [`hooks.py`](../graphsift/hooks.py) |
| `daemon.start()` / `stop()` | Start/stop the background Python daemon | [`daemon.py`](../graphsift/daemon.py) |
| `daemon.exec_code()` | Run Python code in cached daemon process (modules stay imported) | [`daemon.py`](../graphsift/daemon.py) |

The Smart Execution Engine auto-starts on `import graphsift` and works transparently:
- `pip install graphsift` → package installs
- First `import graphsift` → auto-configures `.claude/settings.json` + starts daemon
- Bash/PowerShell commands containing `cd <dir> && python ...` → intercepted → daemon runs them → ~0ms
- Non-Python commands (git, npm) → pass through unchanged
- `sleep N` → handled natively without Python execution

---

## Auto-Fix

| Class / Function | Description | Source |
|---|---|---|
| `FixSuggester` | Graph-based issue detection and fix proposals (5 categories) | [`auto_fix.py`](../graphsift/auto_fix.py) |

---

## Prompt Templates

| Class / Function | Description | Source |
|---|---|---|
| `FixBugTemplate` | Template for bug-fix prompts | [`prompt_templates.py`](../graphsift/prompt_templates.py) |
| `AddFeatureTemplate` | Template for feature addition prompts | [`prompt_templates.py`](../graphsift/prompt_templates.py) |
| `RefactorTemplate` | Template for refactoring prompts | [`prompt_templates.py`](../graphsift/prompt_templates.py) |
| `ProductionAppTemplate` | Template for production-ready code prompts | [`prompt_templates.py`](../graphsift/prompt_templates.py) |
| `ThemeChangeTemplate` | Template for theme/UI change prompts | [`prompt_templates.py`](../graphsift/prompt_templates.py) |
| `SecurityArchitectureTemplate` | Template for security architecture prompts | [`prompt_templates.py`](../graphsift/prompt_templates.py) |
| `get_template()` | Get a template by name | [`prompt_templates.py`](../graphsift/prompt_templates.py) |

---

## Exceptions

| Class | Description | Source |
|---|---|---|
| `graphsiftError` | Base exception for all graphsift errors | [`exceptions.py`](../graphsift/exceptions.py) |
| `ValidationError` | Configuration or input validation error | [`exceptions.py`](../graphsift/exceptions.py) |
| `ConfigurationError` | Invalid configuration | [`exceptions.py`](../graphsift/exceptions.py) |
| `ParseError` | Source code parsing error | [`exceptions.py`](../graphsift/exceptions.py) |
| `IndexError` | Indexing operation error | [`exceptions.py`](../graphsift/exceptions.py) |
| `GraphError` | Dependency graph error | [`exceptions.py`](../graphsift/exceptions.py) |
| `AdapterError` | Adapter layer error | [`exceptions.py`](../graphsift/exceptions.py) |
| `BudgetExceededError` | Token budget exceeded | [`exceptions.py`](../graphsift/exceptions.py) |
| `LanguageNotSupportedError` | Language not supported | [`exceptions.py`](../graphsift/exceptions.py) |

---

## Harness / Agent Framework

| Class / Function | Description | Source |
|---|---|---|
| `Harness` | Agent harness with pre/post validation hooks | [`harness.py`](../graphsift/harness.py) |
| `HarnessHook` | Pre/post validation hook interface | [`harness.py`](../graphsift/harness.py) |
| `DriftDetector` | Detects behavioral drift in agent actions | [`harness.py`](../graphsift/harness.py) |
| `AgentAction` | An action performed by an agent | [`harness.py`](../graphsift/harness.py) |
| `DriftAlert` | An alert about detected drift | [`harness.py`](../graphsift/harness.py) |
| `HarnessStats` | Statistics from a harness run | [`harness.py`](../graphsift/harness.py) |

---

## A2A Protocol

| Class / Function | Description | Source |
|---|---|---|
| `A2AServer` | Agent-to-Agent protocol server (JSON-RPC/HTTP) | [`a2a_server.py`](../graphsift/a2a_server.py) |
| `build_agent_card()` | Build an agent capability card | [`a2a_server.py`](../graphsift/a2a_server.py) |
| `run_a2a_server()` | Run the A2A server | [`a2a_server.py`](../graphsift/a2a_server.py) |

---

## MCP Tasks

| Class / Function | Description | Source |
|---|---|---|
| `TaskManager` | Manages async MCP tasks with progress tracking | [`mcp_tasks.py`](../graphsift/mcp_tasks.py) |
| `Task` | An async task managed by TaskManager | [`mcp_tasks.py`](../graphsift/mcp_tasks.py) |
| `TaskState` | Enum of task states | [`mcp_tasks.py`](../graphsift/mcp_tasks.py) |
| `ToolRegistry` | Registry of MCP tools | [`mcp_tasks.py`](../graphsift/mcp_tasks.py) |
| `ToolCategory` | Enum of tool categories | [`mcp_tasks.py`](../graphsift/mcp_tasks.py) |
| `ToolDef` | Definition of an MCP tool | [`mcp_tasks.py`](../graphsift/mcp_tasks.py) |

---

## Parsers

| Class / Function | Description | Source |
|---|---|---|
| `TreeSitterParser` | Tree-sitter based AST parser (11 languages) | [`parsers/tree_sitter_parser.py`](../graphsift/parsers/tree_sitter_parser.py) |
| `register_tree_sitter_parsers()` | Register all tree-sitter language parsers | [`parsers/__init__.py`](../graphsift/parsers/__init__.py) |
