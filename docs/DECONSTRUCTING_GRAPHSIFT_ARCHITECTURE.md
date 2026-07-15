# Deconstructing the graphsift Architecture

> **graphsift v2.2.0** — Created by [Mahesh Makwana](https://github.com/maheshmakvana).
> *Token optimization engine for Claude, GPT-4 & Gemini. 80-150x reduction, F1 0.85.*

> A deep analysis of design patterns, module relationships, data flow, and architectural decisions.
> Updated for v2.2.0-dev — reflecting the post-remediation architecture.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Inventory & Dependency Graph](#2-module-inventory--dependency-graph)
3. [Core Pipeline: The Data Flow](#3-core-pipeline-the-data-flow)
4. [Design Pattern Analysis](#4-design-pattern-analysis)
5. [Module-by-Module Deconstruction](#5-module-by-module-deconstruction)
6. [The Parser Architecture: A Tale of Two Systems](#6-the-parser-architecture-a-tale-of-two-systems)
7. [The Public API Surface](#7-the-public-api-surface)
8. [Architectural Debt: What Was Remediated & What Remains](#8-architectural-debt-what-was-remediated--what-remains)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Architectural Decisions Log](#10-architectural-decisions-log)

---

## 1. System Overview

### 1.1 What graphsift Is

graphsift is a **Python library and MCP server** that performs **ranked context selection** for LLM-powered code analysis. It takes a code repository and a change specification (diff) and returns a curated, token-budgeted subset of the repository that is most relevant to that change.

**Core capabilities:**
- 14-language AST parsing (Python, JS, TS, Go, Rust, Java, C, C++, Ruby, PHP, Bash, HCL, Helm, unknown)
- 11-language tree-sitter precise parsing with GenericParser fallback
- BM25 + TF-IDF hybrid search with optional dense vector fusion
- HOT/WARM/COLD tiered context selection with hard token budget enforcement
- 19 CLI command compressors (pytest 94%, grep 97%, git diff 92%)
- Cross-session agent memory with TTL-based expiry
- MCP server (7 token-saving tools) + A2A protocol server
- Conversation compaction (deletion-based, not hallucinated summarization)
- Schema versioning with auto-migration

### 1.2 What graphsift Is Not

- **Not a vector database** — graphsift uses BM25 + graph traversal, not embeddings. No GPU required.
- **Not a snapshot tool** — Unlike repomix/codesight, graphsift does not simply concatenate files. It ranks, tiers, trims, and deduplicates based on structural code relationships.
- **Not an LLM** — graphsift never calls an LLM. It is a pre-processing layer that optimizes input for LLMs.
- **Not a storage system** — SQLite persistence is for caching, agent memory, and cross-session state, not primary data storage.

### 1.3 The Three-Layer Architecture

```
+----------------------------------------------------------------------+
|                         APPLICATION LAYER                             |
|                                                                      |
|  +----------------+  +----------------+  +----------+  +-----------+ |
|  | CLI (cli.py)   |  | MCP Server     |  | A2A      |  | Python    | |
|  | graphsift cmd  |  | (mcp_*.py)     |  | (a2a_)   |  | API       | |
|  +----------------+  +----------------+  +----------+  | __init__.py| |
|                                                        +-----------+ |
|  +--------------------------------------------------------------+    |
|  | API v2 (NEW)        api/v2/                                 |    |
|  | Sift class, SiftConfig, IndexResult, compat layer            |    |
|  +--------------------------------------------------------------+    |
+------------------------------+---------------------------------------+
                               |
+------------------------------v---------------------------------------+
|                           DOMAIN LAYER                                |
|                                                                      |
|  +----------------------+  +------------------+  +------------------+ |
|  | ContextBuilder       |  | DependencyGraph  |  | RelevanceRanker  | |
|  | (core.py: ~2.7K LOC) |  | (core.py)        |  | (core.py)        | |
|  +----------+-----------+  +--------+---------+  +--------+---------+ |
|             |                       |                      |          |
|  +----------v-----------+          |                      |          |
|  | ContextSelector      |          |                      |          |
|  | (core.py)            |<---------+----------------------+          |
|  +----------------------+                                             |
|                                                                      |
|  +----------+  +----------+  +----------+  +-----------------------+ |
|  | Memory   |  | Security |  | Compress |  | Evidence / Verify    | |
|  | memory.py|  | security |  | compress |  | evidence_check.py    | |
|  |          |  | .py      |  | .py      |  | verify_hooks.py      | |
|  +----------+  +----------+  +----------+  +-----------------------+ |
|                                                                      |
|  +----------------------+  +----------------------+                  |
|  | Hybrid Search        |  | Auto-fix + Analyze   |                  |
|  | hybrid_search.py     |  | auto_fix.py, prior.. |                  |
|  +----------------------+  +----------------------+                  |
|                                                                      |
|  +----------------------+  +----------------------+                  |
|  | Migrations (NEW)     |  | Validators (NEW)     |                  |
|  | migrations.py,       |  | validators.py        |                  |
|  | schemas/             |  | (File/Config/Graph/  |                  |
|  +----------------------+  |  Security/DiffSpec)  |                  |
|                            +----------------------+                  |
+------------------------------+---------------------------------------+
                               |
+------------------------------v---------------------------------------+
|                       INFRASTRUCTURE LAYER                            |
|                                                                      |
|  +----------------------+  +----------------------+                  |
|  | GraphStore           |  | AgentMemory          |                  |
|  | (storage.py, SQLite) |  | (memory.py, SQLite)  |                  |
|  | v7 schema, auto-     |  | TTL-based expiry     |                  |
|  | migration via        |  | BM25 + TF-IDF recall |                  |
|  | SchemaRegistry (NEW) |  +----------------------+                  |
|  +----------------------+                                             |
|                                                                      |
|  +----------------------+  +----------------------+                  |
|  | ASTCache (NEW)       |  | DatabasePool (NEW)   |                  |
|  | LRU memory + SQLite  |  | Thread-safe conn     |                  |
|  | persistent tiers     |  | pool with auto-      |                  |
|  | SHA-256 invalidation |  | reconnect            |                  |
|  +----------------------+  +----------------------+                  |
|                                                                      |
|  +----------------------+  +----------------------+                  |
|  | AsyncEngine (NEW)    |  | Filesystem Adapter   |                  |
|  | async_index_files    |  | filesystem.py        |                  |
|  | async_build          |  | File I/O helpers     |                  |
|  | AsyncContextBuilder  |  +----------------------+                  |
|  | asyncio.to_thread()  |                                             |
|  | + asyncio.gather()   |                                             |
|  +----------------------+                                             |
+----------------------------------------------------------------------+
```

---

## 2. Module Inventory & Dependency Graph

### 2.1 Complete Module Inventory (55 Python files)

| Layer | Module | LOC | Dependencies | Purpose |
|-------|--------|-----|-------------|---------|
| **App** | `cli.py` | 1,925 | core, compress, memory, mcp_server, analytics | CLI entry point (30+ commands) |
| **App** | `mcp_server.py` | 3,248 | core, models, storage, compress, memory, analytics | MCP protocol server |
| **App** | `mcp_tasks.py` | 1,029 | models | Task & tool registry for MCP |
| **App** | `mcp_shrink.py` | 100 | core, mcp_server | MCP context compression tool |
| **App** | `a2a_server.py` | 1,255 | models | Agent-to-Agent protocol server |
| **App** | `api/v2/sift.py` | 273 | core, compress | Unified Sift API (NEW) |
| **App** | `api/v2/models.py` | 152 | pydantic | Clean v2 models (NEW) |
| **App** | `api/v2/compat.py` | 203 | v1 models | v1-v2 model converters (NEW) |
| **App** | `api/v1.py` | 123 | core | Backward compat shim (NEW) |
| | | | | |
| **Domain** | `core.py` | 2,792 | models, exceptions, ast | **Monolith**: parsers, graph, ranking, selection |
| **Domain** | `models.py` | 434 | pydantic | 25+ Pydantic v2 data contracts |
| **Domain** | `exceptions.py` | 43 | — | 9 exception classes |
| **Domain** | `advanced.py` | 846 | core, models, exceptions | AnalysisPipeline, DiffValidator, batch/async wrappers |
| **Domain** | `hybrid_search.py` | 496 | models, math | BM25 + TF-IDF + RRF fusion + dense vectors |
| **Domain** | `compress.py` | 966 | re, json | 19 CLI command compressors |
| **Domain** | `memory.py` | 933 | models, exceptions, sqlite3 | Agent memory with SQLite persistence |
| **Domain** | `compact_context.py` | 1,030 | core, compress | Conversation compaction engine |
| **Domain** | `security.py` | 567 | exceptions, re, os | PathValidator, CommandSanitizer, DataScrubber |
| **Domain** | `auto_fix.py` | 626 | models, core | FixSuggester with graph-based analysis |
| **Domain** | `evidence.py` | 901 | models | EvidenceTracer for file:line citations |
| **Domain** | `evidence_check.py` | 135 | models | EvidenceChecker for citation validation |
| **Domain** | `verify_hooks.py` | 117 | models | Verifier for post-change syntax checks |
| **Domain** | `prioritize.py` | 572 | models | PriorityScorer for findings |
| **Domain** | `temporal_graph.py` | 363 | models | Git-history-aware symbol tracking |
| **Domain** | `code_memory.py` | 387 | models, sqlite3 | Code-anchored agent memory |
| **Domain** | `tiered_memory.py` | 81 | — | Hierarchical memory buckets |
| **Domain** | `tool_budgets.py` | 106 | — | Per-tool output line caps |
| **Domain** | `read_cache.py` | 78 | hashlib | SHA-256 fingerprint dedup for file reads |
| **Domain** | `analytics.py` | 428 | — | Token savings analytics (gain, discover) |
| **Domain** | `harness.py` | 805 | models | Pre/post validation hooks for agent pipelines |
| **Domain** | `hooks.py` | 190 | — | CLI hook management |
| **Domain** | `prompt_templates.py` | 763 | — | 6 prompt templates |
| **Domain** | `executor.py` | 683 | subprocess | CommandExecutor, AutoPipeline |
| **Domain** | `migrations.py` | 418 | pydantic, exceptions | SchemaRegistry with auto-migration (NEW) |
| **Domain** | `validators.py` | 885 | exceptions, re, os | File/Config/Graph/Security/DiffSpec validators (NEW) |
| | | | | |
| **Parsers** | `parsers/tree_sitter_parser.py` | 1,872 | core, models, exceptions | 11-language tree-sitter + GenericParser fallback |
| **Parsers** | `parsers/__init__.py` | 24 | — | Parser package init |
| | | | | |
| **Schemas** | `schemas/graph_schema.py` | 111 | pydantic | GraphNodeV1/V2 (NEW) |
| **Schemas** | `schemas/context_schema.py` | 141 | pydantic | ContextConfigV1/V2 (NEW) |
| **Schemas** | `schemas/memory_schema.py` | 126 | pydantic | Memory schema versions (NEW) |
| | | | | |
| **Infra** | `adapters/storage.py` | 1,034 | models, exceptions, sqlite3 | GraphStore — SQLite persistence v7 |
| **Infra** | `adapters/postprocess.py` | 720 | models, core | CommunityDetector, RiskScorer, RefactorEngine |
| **Infra** | `adapters/filesystem.py` | 138 | os, pathlib | File I/O helpers |
| **Infra** | `cache.py` | 360 | sqlite3, hashlib | LRU AST cache (memory + SQLite) (NEW) |
| **Infra** | `pool.py` | 344 | sqlite3, threading | Thread-safe DatabasePool (NEW) |
| **Infra** | `async_engine.py` | 419 | asyncio, core | Async pipeline wrappers (NEW) |

**Total: ~32K source LOC across 55 Python files** (excluding tests)

### 2.2 Dependency Depth

| Module | Direct Imports | Transitive Deps | Depth | Notes |
|--------|---------------|-----------------|-------|-------|
| `__init__.py` | 35+ imports | All | Root | Central hub — re-exports ~80 symbols |
| `core.py` | 5 (ast, models, storage(TC), exceptions, tokenpruner) | Low | Low | Clean isolation |
| `mcp_server.py` | 15+ (core, models, storage, compress, memory, analytics, advanced...) | Medium | High | Too many deps — should be split |
| `cli.py` | 10+ (core, compress, memory, mcp_server, analytics) | Medium | High | Coupled to implementation |
| `memory.py` | 3 (models, exceptions, sqlite3) | Low | Low | Clean isolation |
| `security.py` | 2 (exceptions, standard lib) | Low | Low | Clean isolation |
| `storage.py` | 4 (models, exceptions, sqlite3) | Low | Low | Clean isolation (now using DatabasePool) |
| `advanced.py` | 4 (core, models, exceptions) | Low | Low | Clean isolation |
| `api/v2/sift.py` | 2 (core, compress) + lazy imports | Low | Low | Clean isolation (NEW) |
| `cache.py` | 4 (sqlite3, hashlib, fnmatch) | Low | Low | Clean isolation (NEW) |
| `pool.py` | 2 (sqlite3, threading) | Low | Low | Clean isolation (NEW) |
| `async_engine.py` | 3 (asyncio, core, models) | Low | Low | Clean isolation (NEW) |
| `migrations.py` | 2 (pydantic, exceptions) | Low | Low | Clean isolation (NEW) |
| `validators.py` | 3 (exceptions, re, os) | Low | Low | Clean isolation (NEW) |

**Key finding:** `core.py` (2,792 lines) remains the largest module and the primary candidate for further decomposition. The new modules (`cache.py`, `pool.py`, `async_engine.py`, `migrations.py`, `validators.py`, `api/v2/*`) all maintain clean, low-dependency profiles.

---

## 3. Core Pipeline: The Data Flow

### 3.1 The ContextBuilder Pipeline

This is the heart of graphsift. Understanding this flow is understanding the entire architecture.

```
Phase 1: INDEXING
=================

     raw source text (dict[path -> str])
              |
              v
     detect_language(path)
     via _EXT_MAP (file extension mapping)
              |
              v
     get_parser(language)
     Returns: PythonParser, GenericParser,
              BashParser, HCLParser, or TreeSitterParser
              |
              v
     parser.parse_file(path, source)
     Extracts: FileNode {
       .language
       .symbols[]  -> GraphNode[] (functions, classes, methods)
       .imports[]
       .dynamic_imports[]
       .sha256 (for incremental re-index)
       .token_estimate
     }
              |
              v
     graph.add_file(file_node)
     DependencyGraph stores in:
       _nodes: dict[str, GraphNode]     (symbols by node_id)
       _files: dict[str, FileNode]      (files by path)
       _edges: dict[str, list[GraphEdge]] (edges by source)
       _file_imports: dict[str, set[str]] (imports per file)
              |
              v
     graph.build_import_edges()
     Resolves import paths across files
     Creates edges of kind: IMPORTS, CALLS, INHERITS, DECORATES
              |
              v
     [ASTCache (NEW): caches FileNode by sha256,
      skips re-parse on subsequent index_files
      with identical content]

     Returns: IndexStats {
       files_indexed, files_skipped,
       symbols_extracted, edges_created,
       duration_ms, languages
     }


Phase 2: BUILD (per query)
==========================

     DiffSpec(changed_files, diff_text, query, commit_message)
              |
              v
     graph.ranked_neighbors(changed_files, max_depth=4)
     -> list of (ScoredFile, distance)
     BFS traversal with distance tracking
              |
              v
     RelevanceRanker.score(scored_files, query, diff_spec)
     Multi-signal scoring:
       score = alpha * BM25(query, file.text)
             + beta * graph_distance_decay(distance)
             + gamma * test_bonus(is_test_file)
             + delta * semantic_query_match(file.symbols, query)
       score clipped to [0, 1]
              |
              v
     [Validators (NEW): validate scored files,
      check token budget consistency,
      verify no orphan files]
              |
              v
     ContextSelector.select(scored_files, config)
     Greedy knapsack within token_budget:
       1. Sort by score descending
       2. HOT tier (score >= hot_threshold): full source
       3. WARM tier (score >= warm_threshold): signatures
       4. COLD tier (below warm_threshold): excluded
       5. Stop when token_budget exhausted
              |
              v
     Diff-aware trimming (if enabled):
       Parse diff hunks -> identify changed regions
       Include only relevant symbols + context_lines
       Insert gap markers for trimmed regions
              |
              v
     Entropy-based deduplication (if enabled):
       SimHash fingerprinting of selected file content
       Skip near-duplicate files (preserving changed files)
              |
              v
     Cache-aware rendering (if enabled):
       Structure output into cacheable (WARM) and
       dynamic (HOT) zones with cache_control breakpoints
              |
              v
     Returns: ContextResult {
       selected_files: list[ScoredFile],
       rendered_context: str,
       total_original_tokens: int,
       total_rendered_tokens: int,
       reduction_ratio: float,
       cache_breakpoints: int,
       metadata: dict
     }
```

### 3.2 The Async Pipeline (NEW)

```
     async_engine.py provides parallel variants:

     async_index_files(builder, source_map)
       -> uses asyncio.to_thread() for CPU-bound parsing
       -> uses asyncio.gather() for parallel file processing
       -> returns IndexStats

     async_build(builder, diff_spec, source_map)
       -> offloads ranking to thread pool
       -> returns ContextResult

     async_search(query, nodes, top_k=20)
       -> runs BM25 + TF-IDF in thread pool
       -> returns list[(GraphNode, float)]

     AsyncContextBuilder(builder)
       -> wraps ContextBuilder with full async interface
       -> await builder.index_files(source_map)
       -> await builder.build(diff_spec, source_map)
```

### 3.3 The Caching Layer (NEW)

```
     ASTCache provides two-tier caching:

     Memory tier (OrderedDict LRU):
       - max_memory entries (default 500)
       - Per-entry TTL (default 300s)
       - O(1) get/set

     Disk tier (SQLite):
       - Persistent across restarts
       - SHA-256 keyed (content fingerprint)
       - No TTL (persistent until invalidated)

     Invalidation:
       - By path pattern: cache.invalidate("src/*.py")
       - By prefix: cache.invalidate("src/auth/")
       - Full clear: cache.clear()
       - On index_files: skips files with unchanged SHA-256,
         re-parses only changed files
```

---

## 4. Design Pattern Analysis

### 4.1 Patterns Used

| Pattern | Location | Implementation |
|---------|----------|----------------|
| **Builder** | `ContextBuilder` | Fluent interface with `index_files()` and `build()` |
| **Strategy** | Parser selection | `get_parser(language)` returns appropriate parser |
| **Chain of Responsibility** | Compression | `detect_type()` -> specific compressor -> post-process |
| **Adapter** | `GraphStore` | SQLite adapter with uniform interface |
| **Repository** | `AgentMemory` | CRUD operations over fact storage |
| **Template Method** | `Pipeline` | Abstract step-based pipeline with hooks |
| **Facade** | `__init__.py` | Unified public API surface |
| **Factory** | `get_parser()` | Returns parser based on language |
| **Singleton** | `SchemaRegistry` (NEW) | Single registry instance per process |
| **Proxy** | `DatabasePool` (NEW) | Connection pooling with acquire/release |
| **Decorator** | `AsyncContextBuilder` (NEW) | Wraps sync ContextBuilder with async interface |

### 4.2 Patterns Avoided

| Pattern | Reason for Avoidance |
|---------|---------------------|
| **Abstract Factory** | Overkill — `get_parser()` is sufficient for current language count |
| **Observer** | No event-driven workflows; pipeline is linear |
| **Visitor** | AST traversal is handled by tree-sitter CST walkers |
| **Command** | CLI commands are simple argparse dispatch — no undo needed |
| **State** | ContextBuilder has one state transition (indexed -> built) — no state machine needed |

---

## 5. Module-by-Module Deconstruction

### 5.1 Core Modules

**`core.py` (2,792 LOC)** — The monolith. Contains:
- Language detection (`detect_language`, `_EXT_MAP`)
- Token estimation (`estimate_tokens`)
- Parser classes: `PythonParser`, `GenericParser`, `BashParser`, `HCLParser`, `LanguageParser` (Protocol)
- Parser registry (`register_parser`, `get_parser`)
- `DependencyGraph` — BFS graph traversal, cycle detection, dead code detection
- `RelevanceRanker` — BM25 + graph-distance multi-signal scoring
- `ContextSelector` — Greedy knapsack selection, tiering, rendering, dedup, cache-aware output
- `ContextBuilder` — Pipeline orchestration

**Design assessment**: Cohesive but violates Single Responsibility. The parsing, graph, ranking, and selection logic should each be their own module. **Primary candidate for decomposition.**

**`models.py` (434 LOC)** — 25+ Pydantic v2 data contracts. Enums (Language, NodeKind, EdgeKind, OutputMode, TierLevel, DepthTier, FixSeverity) and dataclasses (GraphNode, GraphEdge, FileNode, ScoredFile, DiffSpec, ContextConfig, ContextResult, IndexStats, CycleInfo, DeadCodeInfo, FixSuggestion, FixReport).

**Design assessment**: Well-factorized. Frozen models where appropriate (all graph models). ConfigDict on every model. Good field validation with `ge`/`le` constraints.

**`exceptions.py` (43 LOC)** — 9 exception classes in a hierarchy. Base `graphsiftError` is never raised directly. Subtypes: ValidationError, ConfigurationError, ParseError, IndexError, GraphError, AdapterError (with nested TimeoutError and RateLimitError), BudgetExceededError, LanguageNotSupportedError.

**Design assessment**: Clean, idiomatic, complete. No gaps.

### 5.2 New Infrastructure Modules (Introduced in v2.2)

**`cache.py` (360 LOC, NEW)** — Two-tier LRU + SQLite AST cache. Uses Python's `OrderedDict` for memory LRU (O(1) move_to_end) and SQLite for disk persistence. SHA-256 keyed for content-addressable cache invalidation. Supports glob-pattern invalidation.

**`pool.py` (344 LOC, NEW)** — Thread-safe SQLite connection pool. Bounded by `max_connections` (default 5). Uses `queue.Queue` for connection management. Auto-reconnect on stale connections. Configurable timeout.

**`async_engine.py` (419 LOC, NEW)** — Wraps synchronous `ContextBuilder` with async interface using `asyncio.to_thread()` and `asyncio.gather()`. Provides `async_index_files`, `async_build`, `async_search`, and `AsyncContextBuilder`.

**`migrations.py` (418 LOC, NEW)** — `SchemaRegistry` with versioned model families. Each model family registers a sequence of Pydantic model classes. `migrate(name, data, from_v, to_v)` performs forward migration step by step. Used by `storage.py` for auto-migration on connect.

**`validators.py` (885 LOC, NEW)** — Five validator classes:
- `FileValidator`: file paths, extensions, content size limits, encoding
- `ConfigValidator`: ContextConfig consistency (hot_threshold > warm_threshold)
- `GraphValidator`: graph integrity (no orphan edges, no self-loops)
- `SecurityValidator`: path traversal prevention, token budget bounds
- `DiffSpecValidator`: changed_files exist, diff format, no secrets in query

**`schemas/` (378 LOC, NEW)** — Three modules:
- `graph_schema.py`: GraphNodeV1 (original), GraphNodeV2 (adds community_id, metadata)
- `context_schema.py`: ContextConfigV1, ContextConfigV2 (adds depth_tier, dedup_enabled)
- `memory_schema.py`: MemoryFactV1, MemoryFactV2

### 5.3 API Layer Modules

**`api/v2/sift.py` (273 LOC, NEW)** — Unified `Sift` class that combines indexing, search, context building, compression, and analysis. Uses lazy imports to avoid circular dependencies.

```python
class Sift:
    def __init__(self, config: SiftConfig | None = None): ...
    def index(self, files: dict[str, str]) -> IndexResult: ...
    def search(self, query: str, top_k: int = 20) -> list[ScoredFile]: ...
    def build_context(self, changes: list[str], query: str = "") -> ContextResult: ...
    def compress(self, text: str, kind: str = "auto") -> str: ...
    def analyze(self, path: str) -> AnalysisResult: ...
```

**`api/v2/models.py` (152 LOC, NEW)** — Clean models: `SiftConfig`, `IndexResult`, `AnalysisResult`, `CompressResult`.

**`api/v2/compat.py` (203 LOC, NEW)** — Bidirectional converters between v1 and v2 models. `v2_from_v1_context_config(v1) -> SiftConfig`, `v1_from_v2_context_result(v2) -> ContextResult`, etc.

**`api/v1.py` (123 LOC, NEW)** — Backward compatibility shim. Re-exports all symbols from original locations with `DeprecationWarning` on import. Ensures `from graphsift import ContextBuilder` continues to work.

### 5.4 Supporting Modules

**`hybrid_search.py` (496 LOC)** — `HybridSearcher` with four search methods:
- `search(query, nodes, method="hybrid")` — alpha-weighted BM25 + TF-IDF fusion
- `search_rrf()` — Reciprocal Rank Fusion of BM25 + sparse + optional dense
- `_search_dense()` — sentence-transformers embeddings (lazy-loaded)
- `sparse_cosine(a, b)` — no-numpy cosine similarity

**Design assessment**: Clean isolation. The BM25 implementation duplicates logic from `core.py`'s `RelevanceRanker._bm25_score`. **Candidate for extraction into shared `_bm25.py` helper.**

**`compress.py` (966 LOC)** — 19 CLI command compressors organized as functions: `compress_pytest`, `compress_grep`, `compress_git_diff`, etc. Each strips noise, collapses repeats, groups similar lines, and truncates middle sections. Common primitives: `deduplicate`, `truncate_middle`, `filter_lines`, `group_similar`, `strip_blanks`.

**Design assessment**: Well-factorized with shared primitives. Each compressor is small (30-60 LOC). The auto-detect dispatcher (`detect_type`) is clean. Fuzz testing revealed one edge case (single-character grep output).

**`memory.py` (933 LOC)** — Full agent memory layer with SQLite persistence. TTL-based expiry, BM25 + TF-IDF recall, cross-session consolidation, session summarization. 6 Pydantic models (`MemoryFact`, `SessionInfo`, etc.).

**`security.py` (567 LOC)** — Four security components:
- `PathValidator`: Directory traversal prevention, symlink detection
- `CommandSanitizer`: Shell injection prevention
- `DataScrubber`: Secret/PII/API key detection
- `SecurePipeline`: Wraps build/analyze with all security checks

---

## 6. The Parser Architecture: A Tale of Two Systems

### 6.1 Design

graphsift has two parallel parser systems that coexist through graceful degradation:

```
                    +---------------------------+
                    |      get_parser()         |
                    |  (parser registry)        |
                    +----+------------------+---+
                         |                  |
            +------------v--+        +------v------------+
            | GenericParser  |        | TreeSitterParser  |
            | (core.py)      |        | (tree_sitter_     |
            | Python, JS,    |        |  parser.py)       |
            | TS, Go, Rust,  |        | Python, JS, TS,   |
            | Java, C, C++,  |        | Go, Rust, Java,   |
            | Ruby, PHP,     |        | C, C++, Ruby,     |
            | Bash, HCL,     |        | PHP, Bash          |
            | Helm            |        | (11 languages)    |
            +----------------+        +--------+----------+
                                                |
                                    +-----------v-----------+
                                    | If grammar missing or |
                                    | tree-sitter not       |
                                    | installed: fall       |
                                    | through to            |
                                    | GenericParser         |
                                    +-----------------------+
```

### 6.2 GenericParser (core.py)

A heuristic-based parser that uses language-specific regex patterns and structural heuristics. Works for 14 languages via extension mapping. Extracts:

- Function/class definitions (via regex patterns per language)
- Import statements (via language-aware regex)
- Decorators (Python), type hints, async markers
- Dynamic imports (via regex patterns shared across all parsers)

**Limitations**: No type inference. No cross-file resolution. Imports parsed textually, not semantically.

### 6.3 TreeSitterParser (parsers/tree_sitter_parser.py)

A precise CST walker using tree-sitter grammars. Installed per-language via pip. Lazy-loads each grammar on first use. Tree-walks the concrete syntax tree to extract:

- Functions (with async detection)
- Classes (with base class detection)
- Methods (with receiver detection for Go)
- Decorators (with argument extraction)
- Imports (named, default, dynamic)
- Signatures (parameter types, return types)
- Line numbers (start/end for every symbol)

**Fallback behavior**: If tree-sitter is not installed or a grammar is unavailable, `TreeSitterParser.parse_file()` falls through to `GenericParser.parse_file()`. Callers always get a `FileNode` back.

### 6.4 Parser Comparison

| Dimension | GenericParser | TreeSitterParser |
|-----------|---------------|------------------|
| Languages | 14 | 11 |
| Dependencies | None (stdlib only) | `tree-sitter>=0.23` + per-language grammar |
| Precision | Heuristic regex | CST walk |
| Speed | ~10K files/sec | ~5K files/sec |
| Method detection | Regex-based | Tree-walk |
| Decorator detection | Python only | All languages |
| Dynamic imports | Regex patterns | Regex + AST |

---

## 7. The Public API Surface

### 7.1 Legacy API (v1, stable)

```python
from graphsift import (
    # Core
    ContextBuilder, ContextConfig, ContextSelector,
    DependencyGraph, RelevanceRanker,
    # Parsers
    PythonParser, GenericParser, BashParser, HCLParser,
    detect_language, estimate_tokens,
    TreeSitterParser,
    # Models
    ContextResult, DiffSpec, ScoredFile, FileNode,
    GraphNode, GraphEdge, IndexStats,
    Language, NodeKind, EdgeKind, OutputMode,
    # Compression
    compress, compress_tee, COMPRESSORS,
    # Search
    HybridSearcher,
    # Memory
    AgentMemory, MemoryFact, SessionInfo,
    # And 40+ more exports...
)
```

### 7.2 Modern API (v2, NEW)

```python
from graphsift.api.v2 import Sift
from graphsift.api.v2.models import SiftConfig, IndexResult, AnalysisResult

sift = Sift(SiftConfig(token_budget=50_000))
result = sift.index({"src/auth.py": source})
ctx = sift.build_context(["src/auth.py"], query="Review auth change")
compressed = sift.compress(pytest_output, "pytest")
results = sift.search("login handler", top_k=10)
```

### 7.3 CLI

```bash
graphsift init          # Initialize a project
graphsift index         # Index source files
graphsift build         # Build context for review
graphsift search        # Search codebase symbols
graphsift compress      # Compress CLI output
graphsift serve         # Start MCP server
graphsift status        # Show indexing stats
graphsift gain          # Show token savings
graphsift discover      # Find optimization opportunities
```

### 7.4 MCP Server

7 tools for Claude Code:
- `compress_output` — Compress CLI command output
- `build_context` — Build ranked context for a diff
- `search_codebase` — Semantic code search
- `get_architecture` — Repository structure overview
- `list_resources` — Available context resources
- `get_token_savings` — Token analytics
- `analyze_change` — Diff impact analysis

---

## 8. Architectural Debt: What Was Remediated & What Remains

### 8.1 Remediated Debt (v2.2.0)

| Issue | Before | After | Remediated By |
|-------|--------|-------|---------------|
| **No schema migration** | Manual migration comments in storage.py | `SchemaRegistry` with auto-migration on connect | `migrations.py`, `schemas/` |
| **No validation layer** | Ad-hoc validation in constructors | 5 validator classes with `ValidationReport` | `validators.py` |
| **No AST caching** | Every `index_files()` re-parses all files | LRU + SQLite cache with SHA-256 invalidation | `cache.py` |
| **No connection pool** | Raw `sqlite3.connect()` per operation | Thread-safe `DatabasePool` with auto-reconnect | `pool.py` |
| **No async pipeline** | `asyncio.run()` wrapper in advanced.py | True async with `to_thread()` + `gather()` | `async_engine.py` |
| **Single API surface** | Everything through `__init__.py` | Versioned `api/v2/` with compat shim | `api/v2/` |
| **No property tests** | 140 unit tests, no property coverage | Hypothesis tests for compress, ranking, graph, models | `tests/property/` |
| **No stress tests** | No large-repo or concurrency tests | 10K-file index, 500-file diff, 50-thread concurrent | `tests/stress/` |
| **No fuzz tests** | No random-input testing | 60+ fuzz scenarios across compress, parsers, CLI | `tests/fuzz/` |
| **No integration tests** | Unit tests only | End-to-end pipeline, MCP, storage, CLI subprocess | `tests/integration/` |
| **No performance benchmarks** | No throughput measurements | Index/build/compress throughput + memory benchmarks | `tests/test_performance.py` |
| **Encoding corruption in docs** | Garbled box-drawing, emoji corruption | Clean ASCII diagrams, proper UTF-8 | This rewrite |
| **Outdated README** | 494 lines of outdated content | 185 lines, concise, with benchmarks table | Agent 6 |
| **No changelog** | None | Full version history in CHANGELOG.md | Agent 6 |
| **No migration guide** | None | v1->v2 with before/after examples | Agent 6 |
| **No type stubs** | None | `__init__.pyi` with 756 lines of annotations | Agent 6 |
| **No issue/PR templates** | None | Bug report, feature request, PR templates | Agent 6 |

### 8.2 Remaining Debt

| Issue | Location | Impact | Recommendation |
|-------|----------|--------|----------------|
| **core.py monolith** | `core.py` (2,792 LOC) | Violates SRP, hard to test in isolation | Split into `engine/`: parsers.py, graph.py, ranking.py, selector.py, estimator.py |
| **BM25 duplication** | `core.py` + `hybrid_search.py` | Two separate BM25 implementations | Extract shared BM25 into `engine/_bm25.py` |
| **mcp_server.py coupling** | `mcp_server.py` (3,248 LOC) | 15+ direct dependencies | Split into handler modules per tool category |
| **cli.py coupling** | `cli.py` (1,925 LOC) | Implementation coupling | Use the v2 API internally instead of importing core directly |
| **No CI/CD** | Repository root | No automated test/gate pipeline | Add GitHub Actions with pytest, coverage, lint |
| **Coverage threshold** | pyproject.toml | `fail_under` set to 0 | Set threshold to 75% once CI is in place |
| **Single-character grep edge case** | `compress.py` | compress("0", "grep") > input length | Guard: if input < 10 chars, return input unchanged |

### 8.3 Debt Priority Matrix

```
                    High Impact                Medium Impact
                  ┌──────────────────────┬────────────────────────┐
  High Effort     │ core.py split        │ mcp_server.py split   │
                  │ (2-3 days)           │ (1-2 days)            │
                  ├──────────────────────┼────────────────────────┤
  Low Effort      │ BM25 dedup           │ grep edge case fix    │
                  │ (2 hours)            │ (30 minutes)          │
                  │ CI/CD setup          │ Coverage threshold    │
                  │ (1 day)              │ (1 hour)              │
                  └──────────────────────┴────────────────────────┘
```

---

## 9. Data Flow Diagrams

### 9.1 Index Flow

```
Source Map
    |
    v
[detect_language] --> [get_parser] --> [parse_file]
    |                      |                |
    |                      v                v
    |              +-------------+   +------------+
    |              | PythonParser|   | FileNode   |
    |              | Generic     |   | GraphNode[]|
    |              | Bash/HCL    |   | imports[]  |
    |              | TreeSitter  |   | sha256     |
    |              +-------------+   +------------+
    |                                   |
    v                                   v
[ASTCache check] <-- sha256 --> [hit] --> return cached FileNode
    |                                   [miss] --> parse + cache
    |
    v
[DependencyGraph.add_file]
    |
    v
[build_import_edges] --> IndexStats
```

### 9.2 Build Flow

```
DiffSpec(changed_files, diff_text, query)
    |
    v
[DependencyGraph.ranked_neighbors]
    |  BFS from changed_files, track distance
    v
[RelevanceRanker.score]
    |  BM25(query, file.text)
    |  graph_distance_decay(distance)
    |  test_bonus + structural_importance
    v
[Validators.validate] (NEW)
    |  Check score bounds, budget consistency
    v
[ContextSelector.select]
    |  Sort by score, HOT/WARM/COLD tier, greedy knapsack
    v
[Diff-aware trimming]
    |  Parse diff hunks, trim to relevant regions
    v
[Entropy deduplication]
    |  SimHash fingerprint, skip near-duplicates
    v
[Cache-aware rendering]
    |  Zones + cache_control breakpoints
    v
ContextResult
```

### 9.3 Memory Flow

```
AgentMemory.remember(fact_content)
    |
    v
[Tokenize (BM25)] --> [Build TF-IDF vector]
    |
    v
[Store in SQLite]
    |  facts table: id, content, vector, tags, ttl, timestamp
    v
[Expiry check on read]
    |  DELETE FROM facts WHERE expires_at < NOW()
    v
AgentMemory.recall(query)
    |
    v
[Tokenize query] --> [BM25 score] + [TF-IDF cosine]
    |
    v
[RRF fuse] --> [Sort by score] --> [Return top-K]
```

### 9.4 Async Pipeline Flow (NEW)

```
async def async_index_files(builder, source_map):
    [asyncio.gather(
        asyncio.to_thread(parser.parse_file, path, src)
        for path, src in source_map.items()
    )]
        |
        v
    [asyncio.to_thread(builder._graph.add_file, file_node)
     for file_node in parsed_nodes]
        |
        v
    [asyncio.to_thread(builder._graph.build_import_edges)]
        |
        v
    return IndexStats
```

---

## 10. Architectural Decisions Log

### ADR-1: Pydantic v2 Over Dataclasses

**Status:** ✅ Accepted  
**Context:** graphsift needs validated, serializable data contracts. Python dataclasses lack built-in validation.  
**Decision:** Use Pydantic v2 `BaseModel` with `ConfigDict(frozen=True)` for immutability.  
**Consequence:** 25+ Pydantic models. Automatic JSON serialization. Field validation via `ge`/`le`/`Field`. `model_dump()` for storage serialization.

### ADR-2: BM25 Over Embeddings

**Status:** ✅ Accepted  
**Context:** Relevance ranking needs to work offline with zero dependencies.  
**Decision:** Use BM25 keyword scoring + graph-distance decay. No embeddings. No GPU. No API calls.  
**Consequence:** F1 of 0.85 vs potentially 0.90+ with embeddings. But 100% local, zero latency, no API costs.

### ADR-3: BM25 + TF-IDF Over Pure Dense Retrieval

**Status:** ✅ Accepted  
**Context:** `HybridSearcher` needs efficient code symbol retrieval.  
**Decision:** Combine BM25 (keyword precision) with sparse TF-IDF vectors (token-aware similarity). Use RRF fusion when dense vectors are available.  
**Consequence:** Works offline. No GPU required. Dense vector search is optional via `sentence-transformers`.

### ADR-4: JSON-RPC Over REST

**Status:** ✅ Accepted  
**Context:** MCP protocol uses JSON-RPC 2.0 over stdio. This is the standard for Claude Code integration.  
**Decision:** Implement raw JSON-RPC parsing — no framework (Flask, FastAPI, etc.).  
**Consequence:** 3,248 lines of manual JSON-RPC handling. No automatic request validation. No OpenAPI schema.

### ADR-5: Threading Over AsyncIO (NOW SUPERSEDED)

**Status:** ❌ Superseded by ADR-9  
**Context:** graphsift needed to handle concurrent requests from MCP clients and CLI commands.  
**Decision (original):** Use `threading.RLock()` for thread safety. No asyncio in the core pipeline.  
**Consequence (original):** `asyncio.run()` wrapper in `advanced.py` caused issues in async environments.  
**Updated (v2.2):** `async_engine.py` now provides proper async interface with `asyncio.to_thread()` and `asyncio.gather()`. Core pipeline remains synchronous; async is an optional wrapper.

### ADR-6: Monorepo via Multiple `index_roots()`

**Status:** ✅ Accepted  
**Context:** graphsift must support monorepos with multiple packages.  
**Decision:** `ContextBuilder.index_roots(*source_maps)` accepts multiple source maps and merges them into one graph.  
**Consequence:** Overlapping roots cause duplicate symbols. No cross-root edge resolution. Caller must ensure roots are disjoint.

### ADR-7: Cache-Aware Output With `cache_control`

**Status:** ✅ Accepted  
**Context:** Anthropic and OpenAI offer prompt caching discounts for repeated prefix tokens.  
**Decision:** `ContextSelector._render_cache_aware()` structures output into HOT (dynamic) and WARM (cacheable) zones with `cache_control` breakpoints.  
**Consequence:** Additional 64-79% savings on repeated queries in the same session.

### ADR-8: MCP as Primary Interface Over CLI

**Status:** ✅ Accepted  
**Context:** Claude Code integration requires MCP protocol. The CLI is secondary.  
**Decision:** MCP server has 7+ tools. CLI has 30 commands. MCP is the preferred interface.  
**Consequence:** 3,248 LOC of MCP server with integration tests now added.

### ADR-9: SchemaRegistry With Versioned Models (NEW)

**Status:** ✅ Accepted (v2.2)  
**Context:** Models evolve over time. SQLite tables need migration.  
**Decision:** `SchemaRegistry` with ordered model sequences. `migrate(name, data, from_v, to_v)` for forward migration. Auto-migration on storage connect.  
**Consequence:** All major models have version numbers. Schema changes are trackable and reversible.

### ADR-10: Two-Tier Caching With LRU + SQLite (NEW)

**Status:** ✅ Accepted (v2.2)  
**Context:** AST parsing is the most expensive operation. Repeated `index_files()` on unchanged repos re-parses everything.  
**Decision:** `ASTCache` with memory LRU (fast) + SQLite disk (persistent). SHA-256 content keyed. Glob-pattern invalidation.  
**Consequence:** Subsequent `index_files()` calls on unchanged content are O(1) cache hits. Invalidation is explicit via patterns or content change.

### ADR-11: API Versioning Via Subpackage (NEW)

**Status:** ✅ Accepted (v2.2)  
**Context:** The API surface grew organically. Breaking changes need a migration path.  
**Decision:** `api/v1.py` for legacy compat (with deprecation warnings). `api/v2/` for modern unified API. `compat.py` for bidirectional model conversion.  
**Consequence:** Old code continues working. New code can adopt the cleaner v2 API. Migration is opt-in, not forced.

### ADR-12: Validation as Separate Layer (NEW)

**Status:** ✅ Accepted (v2.2)  
**Context:** Validation was scattered across constructors and methods. No unified error reporting.  
**Decision:** Dedicated `validators.py` with per-domain validator classes. Each returns a `ValidationReport` with warnings/errors.  
**Consequence:** Validation is composable, testable, and domain-separated. Callers can inspect all issues before deciding whether to proceed.

---

*End of Architectural Deconstruction. Updated for v2.2.0-dev — see [CHANGELOG.md](../CHANGELOG.md) for version history.*
