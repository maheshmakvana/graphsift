# GraphSift Wiki

Welcome to the GraphSift wiki. GraphSift is the open-source token optimizer for AI code review — it builds an AST dependency graph of your codebase, scores every file by relevance to a diff, and delivers a token-budget-capped context window so any LLM sees only what matters.

---

## Table of Contents

- [What is GraphSift](#what-is-graphsift)
  - [Comparison at a Glance](#comparison-at-a-glance)
  - [Who Benefits from Token Optimization](#who-benefits-from-token-optimization)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
  - [Parse](#parse)
  - [Rank](#rank)
  - [Select](#select)
  - [Render](#render)
- [Key Features](#key-features)
  - [Token and Cost Optimization](#token-and-cost-optimization)
  - [Code Analysis and Intelligence](#code-analysis-and-intelligence)
  - [CLI Output Compression](#cli-output-compression)
  - [Agent Intelligence and Memory](#agent-intelligence-and-memory-v17)
  - [Developer Experience](#developer-experience)
- [Performance](#performance)
- [Architecture](#architecture)
- [Installation](#installation)
- [MCP Server Tools Overview](#mcp-server-tools-overview)
- [Supported Languages](#supported-languages)
- [CLI Commands Reference](#cli-commands-reference)
- [FAQ](#faq)
- [Links](#links)

---

## What is GraphSift

GraphSift is the number one open-source token optimizer for AI code review. When an LLM reviews a code change, the naive approach sends every transitively-related file — that is 500,000 to 2 million tokens for a medium-sized codebase. GraphSift solves this by applying a four-stage pipeline that reduces token usage by 80 to 150 times per review while maintaining or improving the quality of the LLM's analysis.

The core insight is that context selection should be treated as a ranking problem, not a graph traversal. Instead of using a binary blast-radius approach (include or exclude each file), GraphSift scores every file on a continuous zero-to-one scale using BM25 keyword overlap fused with graph-distance decay from changed files. It then selects greedily within a hard token budget, ensuring the LLM receives maximum signal per token. The result is an F1 relevance accuracy of approximately 0.85 compared to 0.54 for binary blast-radius tools.

Token savings are dramatic. A typical code review that would cost USD 0.50 to USD 2.70 in API fees with raw source dumps costs USD 0.01 to USD 0.05 with GraphSift — a 93 to 99 percent reduction. At 100 pull requests per day, this translates to USD 50 to USD 270 per day saved versus USD 1 to USD 5 per day with GraphSift.

Beyond code review, GraphSift also compresses CLI command output before it reaches an LLM context window. Nineteen specialized compressors cover pytest, git, docker, kubectl, npm, cargo, eslint, go test, jest, and more, achieving an average of 77 percent token savings across all supported command types. A bash wrapper can be installed for transparent, automatic compression without manual piping.

GraphSift runs entirely locally. No telemetry, no internet required, no cloud dependencies. All parsing, ranking, and compression happens on your machine. The MCP server binds to localhost only. The only time network access is needed is when using the optional LLM API adapters for Claude, OpenAI, or Gemini.

### Comparison at a Glance

Benchmarked on a 143-file FastAPI application reviewing a 50-line change to an authentication module:

| Approach | Files Sent | Tokens | Cost at Opus Pricing | Savings vs Raw |
|----------|-----------|--------|---------------------|----------------|
| Raw source, every file | 143 of 143 | Approximately 180,000 | USD 2.70 | Baseline |
| Binary blast-radius | 8 to 12 of 143 | 6,000 to 8,000 | USD 0.10 | 96 percent |
| GraphSift ranked plus budget | 3 to 5 of 143 | 800 to 1,200 | USD 0.015 | 99.4 percent |

### Who Benefits from Token Optimization

GraphSift serves a wide range of use cases across the AI-assisted development ecosystem:

- CI/CD AI code review pipelines that need to auto-select relevant context per pull request, cutting costs by 93 to 99 percent
- Monorepo code review where blast-radius tools drown developers in irrelevant files and GraphSift ranks and trims instead
- Claude Code, Cursor, and Copilot users who want the MCP server to deliver token-efficient context to coding agents
- LLM cost optimization teams that need token budgets, compression, and caching for predictable API spending
- Enterprise AI review pipelines requiring hard limits to prevent runaway API costs and analytics to track every token saved
- RAG and agent context building for any LLM task that needs ranked code context beyond code review

---

## Quick Start

Getting started with GraphSift involves three steps. First, install the package from PyPI using pip. Python 3.9 or later is required, and the only mandatory dependency is Pydantic version 2.0 or later.

Once installed, the quickest path to token savings is to install the MCP server. Run the graphsift install command from your project root. This writes an MCP configuration file into your project, injects hooks into Claude Code settings for automatic graph updates and output compression, and optionally installs four skill files for build, review, impact analysis, and compression. After installation, restart Claude Code to load the MCP server.

Inside your IDE or Claude Code session, ask the agent to build the knowledge graph for your repository. This triggers the build_graph MCP tool, which indexes all source files, parses them into an AST dependency graph, and persists the result to a local SQLite database. The process typically completes in under two seconds even for repositories with ten thousand files.

Once the graph is built, you can request optimized context for any code review. Provide the list of changed files and a review query. The get_context MCP tool returns a rendered context string containing only the most relevant files, ranked and selected to fit within your token budget. Paste this context into your LLM review prompt to save 80 to 150 times the tokens of sending raw source.

For drop-in integration with your existing LLM workflow, GraphSift provides adapters for Claude (Anthropic SDK), OpenAI (Codex), and Gemini. Each adapter wraps the provider's client and automatically builds optimized context before every review call. The response metadata includes exact token savings, typically 93 to 99 percent versus raw source.

To see cumulative token savings across all sessions, run the graphsift gain command. It shows total calls, tokens saved, estimated cost savings, and a daily breakdown.

---

## How It Works

GraphSift's token optimization pipeline consists of four stages: Parse, Rank, Select, and Render.

### Parse

The first stage builds an AST dependency graph from your source code. GraphSift supports 14 programming languages with a combination of native AST parsers, tree-sitter precise CST/AST parsing for 11 languages, and custom parsers for domain-specific languages like Terraform and Helm charts.

Each file is scanned and parsed to extract all defined symbols (functions, classes, methods, interfaces, structs, traits, modules) and the relationships between them. The parser extracts seven distinct edge types:

| Edge Type | Meaning | Example |
|-----------|---------|---------|
| CALLS | One symbol calls or invokes another | A function call to a method defined elsewhere |
| IMPORTS | A file imports a module or symbol | Python import statement, JavaScript require call |
| INHERITS | A class inherits from another class | Class extends another class or implements an interface |
| DECORATES | A decorator is applied to a function or class | Python decorator, Java annotation |
| REFERENCES | A symbol references another symbol | Variable type reference, usage of a constant |
| TEST_COVERS | A test file tests a specific implementation file | Test module imports or references the module under test |
| DYNAMIC_IMPORT | A runtime import via dynamic mechanisms | importlib.import_module, lazy loading, conditional require |

Each file is represented as a graph node containing its path, language, token estimate, and extracted symbols. Each symbol carries its name, qualified name, kind (function, class, method, module), source location, line range, and signature. The result is a directed graph where edges represent dependency relationships between symbols and files.

Incremental indexing uses SHA-256 hash comparison to skip unchanged files. When a file has not been modified since the last index, its hash matches and it is skipped entirely. Re-indexing a repository with few changes completes in under two seconds. For repositories with ten thousand or more files, a full initial build completes in under two seconds as well.

The parser also tracks cross-language dependencies. A Python backend file imported by a TypeScript frontend via a monorepo package manager, for example, will have a dependency edge tracked and scored across the language boundary.

### Rank

Every file in the codebase receives a continuous relevance score between zero and one. The ranking fuses two signals: BM25 keyword overlap between the file content and the review query, and graph-distance decay from the set of changed files. BM25 captures textual relevance — does the file mention terms that appear in the review question? Graph-distance captures structural relevance — how close is this file in the dependency graph to the files that actually changed?

A file that imports from a changed file, is imported by a changed file, or shares a dependency with a changed file receives a higher score. The score decays with graph distance, so direct dependencies score higher than transitive dependencies two or three hops away. This nuanced ranking replaces the binary include-or-exclude decision of traditional blast-radius tools, eliminating the false positives that plague tools using simple graph traversal.

### Select

With every file scored, the selection stage applies a greedy token-budget algorithm to choose the optimal set of files for the LLM context. Files are assigned to one of three tiers: hot, warm, or cold. Hot files receive full source inclusion. Warm files receive only their signatures — function and class declarations without bodies. Cold files are excluded entirely.

The selection respects a hard token budget, configurable per query. This ensures the rendered context never exceeds a model's context window or a cost ceiling. Diff-aware trimming further reduces tokens by keeping only the changed regions of each file plus a configurable number of surrounding context lines, rather than including the entire file.

Entropy-based deduplication removes near-identical files from the selection, improving context diversity. Cross-session caching remembers previous selections for the same diff, avoiding redundant recomputation.

### Render

The final stage renders the selected files into a single Markdown string ready for injection into any LLM prompt. Four output modes are available:

| Output Mode | Description | Typical Token Reduction |
|-------------|-------------|------------------------|
| FULL | Complete source of selected files with file path headers | 80 to 150 times versus raw |
| SIGNATURES | Function and class declarations only, no bodies | 150 to 300 times versus raw |
| COMPRESSED | Applies tokenpruner compression for additional reduction beyond trimming | 400 to 500 times versus raw |
| SMART | Selects the appropriate mode per file automatically based on relevance score | Varies per file |

Hot files (highest relevance) receive FULL mode. Warm files (moderate relevance) receive SIGNATURES mode. Cold files (low relevance) are excluded. The SMART mode applies this tiering automatically.

The rendered output includes several token-saving optimizations. Diff-aware trimming keeps only the changed regions of each file plus a configurable number of surrounding context lines rather than including entire files. Cache-aware output inserts Anthropic and OpenAI cache control breakpoints at strategic positions in the rendered context, reducing costs for repeated queries of the same diff. Cross-session caching remembers previous selections for the same diff specification, avoiding redundant ranking and selection computation.

The build result object exposes key metrics: files selected out of files scanned, total original tokens (what a raw dump would cost), total rendered tokens (what the LLM actually receives), reduction ratio, and percentage savings. These metrics feed directly into cost tracking and analytics dashboards.

---

## Key Features

### Token and Cost Optimization

Hard token budget enforcement ensures the rendered context never exceeds a model's context window or a cost ceiling. The budget is specified per query and acts as a hard cap — the selection algorithm will never exceed it, even if more files are relevant. This prevents the runaway token consumption that plagues blast-radius tools on large repositories.

The three-tier hot, warm, cold selection strategy applies different treatment based on relevance score. Hot files (score above the high threshold) receive FULL source inclusion. Warm files (score between the low and high thresholds) receive SIGNATURES only — function and class declarations without bodies. Cold files (score below the low threshold) are excluded entirely. The thresholds are automatically tuned based on the budget and the score distribution of the current codebase.

Diff-aware context trimming keeps only the changed regions of each file plus a configurable number of surrounding context lines rather than including entire files. For a 500-line file with a 5-line change, this alone saves approximately 490 lines of tokens per file. The context window size is configurable — a default of 10 lines above and below each change provides enough context for the LLM to understand the change without inflating token count.

Entropy-based deduplication removes near-identical files from the selection, improving context diversity. When two or more files have very similar content (common in generated code, configuration files, or copied boilerplate), only the most relevant instance is kept. The deduplication uses a sliding-window hash comparison to detect similarity without requiring full file comparisons.

Four output modes — FULL, SIGNATURES, COMPRESSED, and SMART — provide flexibility for different token budgets and use cases. FULL mode includes complete source for every selected file. SIGNATURES includes only function and class declarations. COMPRESSED applies tokenpruner compression for up to 5 times additional reduction on top of the baseline selection. SMART selects the optimal mode per file automatically.

Cache-aware output inserts Anthropic and OpenAI cache control breakpoints at strategic positions in the rendered context, reducing the cost of repeated queries by allowing the LLM provider to cache the static portions of the context across API calls. Cache hit rates of 50 to 80 percent are typical for repeated review queries against the same codebase.

Cross-session caching with session identifiers enables memory reuse across conversations. When the same diff is reviewed in multiple sessions, the second session benefits from the cached ranking and selection computed during the first session, avoiding redundant work.

Typical token reduction reaches 80 to 150 times versus raw source and 10 to 15 times versus binary blast-radius tools. In the benchmark case of a 143-file FastAPI application reviewing a 50-line change, these savings translate to cost reductions from USD 2.70 per review to USD 0.015 per review — a 99.4 percent reduction.

### Code Analysis and Intelligence

Fourteen-language parsing covers Python, JavaScript, TypeScript, Go, Rust, Java, C++, C, Ruby, PHP, Bash, Terraform, Helm, and Dockerfile. Tree-sitter provides precise CST/AST parsing for 11 languages with line-level granularity for accurate symbol locations and edge detection. Custom parsers cover Terraform HCL, Helm templates, and Dockerfiles.

Seven edge types capture CALLS, IMPORTS, INHERITS, DECORATES, REFERENCES, TEST_COVERS, and DYNAMIC_IMPORT relationships. The edge detection uses multiple strategies per language: regex patterns for quick coverage, native AST parsing for languages with built-in parsers like Python, and tree-sitter queries for precise CST-level edge detection across 11 languages. Each edge carries its type, source symbol, target symbol, source location, and confidence score.

Hybrid search fuses BM25 full-text search with TF-IDF sparse vector similarity for semantic code search across the indexed codebase. The BM25 component matches keyword terms in symbol names, file paths, and source content. The TF-IDF component computes term-frequency-inverse-document-frequency vectors for each symbol and ranks results by cosine similarity. An alpha parameter (default 0.3) controls the fusion weight between the two signals. When TF-IDF embeddings have been computed via the embed_graph tool, the search can be run as a hybrid query; otherwise it falls back to BM25 via SQLite FTS5.

Cycle detection finds and reports dependency cycles with severity grading using Tarjan's strongly-connected-components algorithm. Cycles of three or fewer files are flagged as ERROR severity, larger cycles as WARNING. The output reports each cycle's member files, total files involved, and the maximum cycle length. This is accessible via the detect_cycles CLI command and the corresponding MCP tool.

Dead code detection identifies unreachable functions, classes, and methods from entry points using BFS reachability analysis. Entry points can be specified explicitly or auto-detected from common patterns (main functions, module-level scripts, test files). The analysis performs a breadth-first traversal of the dependency graph from all entry points; any symbol not reached by this traversal is flagged as potentially dead. Results include the symbol name, kind, file path, line number, and the reason it was classified as dead (no incoming references from any entry-point reachable symbol).

Auto-fix suggestions across five categories (import, type, structure, cycle, dead code) provide prioritized, confidence-scored recommendations with auto-fixable flags. The suggestion engine analyzes the full dependency graph and produces structured results including the issue title, description, file location, severity (error, warning, info), confidence score (0 to 1), a suggested change description, whether the fix can be auto-applied, and the category. Results are sortable by severity and filterable by minimum confidence threshold.

Decorator tracking captures annotations that most tools miss. The parser specifically tracks Python decorators, Java annotations, TypeScript decorators, and similar constructs in other languages. Decorator edges connect the decorated symbol to the decorator function, enabling analysis patterns like find all routes registered by a specific decorator or identify all functions protected by authentication decorators.

Dynamic import detection catches runtime imports via importlib.import_module, import_module, __import__, require with computed paths, and lazy import patterns. These imports are invisible to static analysis alone, but GraphSift's regex-based dynamic import detection captures them as DYNAMIC_IMPORT edges with lower confidence than static imports.

### CLI Output Compression

Nineteen specialized compressors auto-detect command type from output signatures. Achieve 60 to 97 percent token savings on command output before it reaches an LLM context window. A bash wrapper enables transparent compression — supported commands include pytest, cargo, npm, docker, kubectl, aws, grep, cat, make, pip, jest, eslint, git, go test, npx, and yarn. Tee mode saves original uncompressed output while the LLM sees compressed. Token analytics track cumulative savings, daily breakdowns, cost estimates, and opportunity discovery.

Compression Benchmarks from real-world test runs:

| Command Type | Original Tokens | Compressed Tokens | Savings |
|-------------|----------------|-------------------|---------|
| grep (25 results) | 413 | 22 | 95 percent |
| eslint (12 problems) | 308 | 17 | 94 percent |
| git diff (2 files) | 889 | 60 | 93 percent |
| pytest (45 tests) | 1,334 | 136 | 90 percent |
| npm install output | 288 | 39 | 87 percent |
| docker ps (10 images) | 463 | 63 | 86 percent |
| git status | 174 | 25 | 86 percent |
| pip install (7 packages) | 312 | 47 | 85 percent |
| cargo build | 463 | 80 | 83 percent |
| kubectl get all | 581 | 110 | 81 percent |
| git log (3 commits) | 234 | 47 | 80 percent |
| make output | 250 | 55 | 78 percent |
| aws CLI JSON | 477 | 115 | 76 percent |
| jest (10 tests) | 310 | 76 | 75 percent |
| go test | 284 | 74 | 74 percent |
| Application logs (16 lines) | 402 | 155 | 61 percent |
| cat (large file) | 672 | 479 | 29 percent |
| Weighted average | 8,138 | 1,884 | 77 percent |

Each compressor uses a domain-specific strategy. The grep compressor groups results by match and deduplicates identical lines. The pytest compressor keeps assertions and failure summaries while stripping full tracebacks. The git diff compressor extracts file paths and the first three changed lines. The docker compressor shows IDs and names capped at 40 entries. The aws compressor compacts large JSON by preserving keys and primitive values while truncating arrays. The generic compressor strips blank lines, deduplicates, and truncates at 200 lines.

At 100 CLI commands per day piped to an LLM, the compression saves approximately 625,000 tokens per day, which translates to roughly USD 9.37 per day saved on Claude Opus pricing.

### Agent Intelligence and Memory (v1.7)

An agent memory layer provides SQLite-backed knowledge graph persistence for agent conversation context across sessions, with hybrid BM25 and TF-IDF recall. This enables the agent to remember relevant context from previous sessions without repeatedly re-indexing or re-ranking the codebase. The memory layer supports cross-session recall with decay factors for stale information.

Typed graph retrieval offers PRISM-style typed-path traversal with six query intents: security review, refactor impact, test impact, dependency update, architecture review, and general. Each intent specifies a traversal pattern through the dependency graph — for example, security review follows CALLS edges outward from input-handling code, while test impact follows TEST_COVERS edges to identify test files that exercise changed implementation code.

Conversation compaction uses three strategies: observation masking removes redundant observations about unchanged code, boundary preserve keeps the edges of conversation turns intact while compacting the middle, and adaptive ACON-style compaction learns which parts of conversations are used or ignored and adjusts compaction aggressiveness accordingly. These strategies achieve 60 to 82 percent token savings on agent conversations.

Evidence citations provide a full audit trail explaining why each file was selected or excluded, with score breakdowns and connection tracing. When GraphSift selects a file for context, the evidence system records the BM25 score contribution, the graph-distance score contribution, the combined score, the specific dependency edges that connect the file to the changed files, and the reason classification. This audit trail is surfaced in the get_context and get_impact tool outputs.

An A2A protocol server enables Agent-to-Agent communication via JSON-RPC over HTTP, exposing code intelligence as an A2A agent card. Other agents can discover GraphSift's capabilities through the A2A agent card, request context building and impact analysis through the JSON-RPC interface, and receive structured results with evidence citations.

MCP async tasks support long-running operations with progress tracking, cancellation, and progressive tool disclosure loading. When a build_graph or run_postprocess operation takes longer than a single tool call, GraphSift can return a task identifier, report progress as a percentage and estimated time remaining, accept cancellation requests, and stream intermediate results as they become available.

Harness engineering provides pre- and post-validation hooks that run before and after every context building operation. Pre-validation checks include graph integrity (no orphaned nodes or dangling edges), budget enforcement (the requested token budget does not exceed the model's context window), and source freshness (all indexed files still exist on disk). Post-validation checks include budget confirmation (rendered context is within budget), drift detection (the rendered context differs from what the model last saw), and evidence completeness (every selected file has a traceable reason).

A temporal code graph offers git-history-aware symbol tracking with bi-temporal queries: query what symbols existed at a specific commit, find when a symbol was introduced, find its last modification date, and apply age-based relevance boosts. Files modified more recently receive a higher base relevance score, reflecting the intuition that recently changed code is more likely to be affected by new changes.

Code-aware memory anchors memories to specific code symbols with graph-proximity recall and intelligent decay for frequently changed code. When an agent stores a memory about a specific function, that memory is retrievable not only by direct query but also by proximity in the dependency graph — a change to a nearby function can trigger recall of related memories.

### Developer Experience

A full MCP server is compatible with Claude Code, Claude desktop, Cursor, Copilot, Windsurf, Codex, Gemini, and 23 or more MCP clients. Over 40 MCP tools cover graph building, context retrieval, impact analysis, code search, refactoring, documentation generation, compression, and analytics. Four MCP prompts provide reusable templates for code review, impact analysis, issue finding, and architecture explanation. Ten MCP resources expose graph statistics, architecture overviews, communities, flows, wiki pages, and risk indices.

CLI commands cover install, serve, build, status, register, compress, gain, discover, bash-wrapper, detect-changes, detect-cycles, detect-dead-code, suggest-fixes, visualize, wiki, watch, postprocess, and more. Each command supports help text and argument parsing.

Drop-in adapters for Claude (Anthropic SDK), OpenAI (Codex), and Gemini wrap provider SDKs for automatic context optimization. Each adapter extends the provider's client class and overrides the chat completion method to inject optimized context before every call. The adapter uses the same ContextBuilder pipeline internally, so the same token-savings guarantees apply. Response metadata includes the number of tokens saved on each call.

Ten advanced feature categories provide production-grade infrastructure:

- Smart Cache: GraphCache provides LRU plus TTL caching with a memoize decorator, thread-safe access, and cache hit rate statistics
- Analysis Pipeline: AnalysisPipeline chains processing steps with retry logic and full audit traceability for each step
- Async Batch: batch_index and async_batch_build process multiple repositories or diffs concurrently with bounded semaphore control
- Streaming: stream_context yields scored file batches as they are ranked, enabling progressive rendering in the LLM prompt
- Rate Limiter: token-bucket rate limiter with per-key tracking and statistics, wrapping any LLM call in a context manager
- Diff Engine: ContextDiff compares two ContextResult objects to show exactly how configuration changes affect token usage and file selection
- Circuit Breaker: CircuitBreaker with three states (closed, open, half-open) prevents cascading failures when LLM APIs are degraded
- Retry: configurable retry logic with exponential backoff and jitter for transient API failures
- Schema Evolution: six-version SQLite migration history ensures forward and backward compatibility of the graph database as the schema evolves
- Audit Trail: every context building operation records its inputs, parameters, and outputs for full traceability

Incremental indexing with SHA-256 hash comparison skips unchanged files. When a file is re-indexed, its content is hashed and compared to the previously stored hash. Matching hashes allow the file to be skipped entirely, reducing re-index time to effectively zero for unchanged files. This is especially valuable in CI/CD pipelines where the graph is rebuilt on every pull request.

Monorepo support via index_roots enables multi-package repositories. The index_roots function accepts a list of package root directories within the monorepo, indexes each package independently, and merges the results into a unified graph with correct cross-package dependency scoring. A unified token budget is applied across all packages, ensuring that the most relevant files from any package are included.

SQLite persistence includes six-version migration history. The GraphStore class manages a SQLite database stored on the local filesystem, with automatic migration from any earlier schema version to the current version. The database stores nodes, edges, files, flow snapshots, communities, risk scores, FTS indexes, TF-IDF embeddings, review feedback, and session memory. Migration logs are visible on stderr during the first access after an upgrade.

---

## Performance

GraphSift is designed for speed across all stages of the pipeline. Key performance characteristics:

| Operation | Typical Performance |
|-----------|-------------------|
| Full index of 10,000-file repository | Under 2 seconds |
| Incremental re-index (no changes) | Under 100 milliseconds |
| Incremental re-index (few changes) | Under 500 milliseconds |
| Context building for a typical diff | Under 50 milliseconds |
| Post-processing (flows, communities, risk, FTS) | Under 5 seconds for 10,000 files |
| BM25 ranking pass | Under 100 milliseconds for 10,000 files |
| Graph traversal for blast radius | Under 10 milliseconds |
| CLI output compression | Under 10 milliseconds per command |
| MCP tool response (non-build tools) | Under 50 milliseconds |
| MCP tool response (build_graph, cold start) | Under 5 seconds for 10,000 files |

Performance scales linearly with file count. The indexing stage is I/O-bound on most systems — parsing throughput is limited by disk read speed rather than CPU. The ranking and selection stages are CPU-bound but complete in milliseconds even for large repositories because the expensive computation (graph construction) has already been done during indexing.

Several design decisions contribute to this performance:

- SHA-256 incremental indexing skips unchanged files entirely, avoiding redundant parsing
- In-memory graph representation uses adjacency lists for O(1) neighbor lookups during ranking
- Depth cap (default 4) prevents infinite traversal on cyclic imports
- All shared state is protected behind reentrant locks for thread safety
- Async twins are available for all blocking operations to support event-loop-based workflows
- SQLite WAL mode enables concurrent reads during indexing
- FTS5 full-text search indexes are rebuilt incrementally, not from scratch

## Architecture

GraphSift follows a hexagonal (ports and adapters) architecture with pure domain logic at the core, separated from I/O concerns via adapter interfaces.

The source tree is organized into the following layers:

- Core domain: core.py (ContextBuilder, ranking, selection, rendering logic), models.py (Pydantic v2 value objects for GraphNode, FileNode, GraphEdge, DiffSpec, ContextConfig, ContextResult), exceptions.py (typed exception hierarchy)
- Parsing: parsers/ package with tree-sitter grammars for 11 languages, regex-based fallback parsing for all 14 languages, custom parsers for Terraform, Helm, and Dockerfile
- Analysis: hybrid_search.py (BM25 plus TF-IDF fusion), auto_fix.py (graph-based suggestion engine across 5 categories), compact_context.py (conversation compaction with 3 strategies)
- Agent features: memory.py (SQLite-backed knowledge graph persistence), typed_retrieval.py (PRISM-style path retrieval with 6 intents), evidence.py (audit trail with score breakdowns), code_memory.py (symbol-anchored memory with graph-proximity recall), temporal_graph.py (git-history-aware symbol tracking)
- Infrastructure: a2a_server.py (JSON-RPC over HTTP A2A protocol), mcp_tasks.py (async task manager with progress tracking), harness.py (pre and post validation hooks, drift detection)
- Compression: compress.py (19 CLI output compressors with auto-detection), hooks.py (bash wrapper for transparent compression), analytics.py (token savings tracking, discovery, and reporting)
- Adapters: adapters/storage.py (SQLite GraphStore with 6-version migration), adapters/filesystem.py (path I/O helpers and source map loading), adapters/claude.py (Anthropic SDK adapter), adapters/openai.py (OpenAI and Codex adapter), adapters/gemini.py (Gemini adapter), adapters/llm.py (shared multi-provider adapter logic), adapters/postprocess.py (flow detection, community detection, risk scoring, wiki generation)
- Integration: cli.py (command-line interface with 22 subcommands), mcp_server.py (MCP protocol server with 40-plus tools, 4 prompts, 10 resources)

The SQLite database schema supports six migration versions, ensuring forward and backward compatibility as the data model evolves. The database stores nodes, edges, file records, flow snapshots, community assignments, risk scores, FTS indexes, TF-IDF embedding metadata, review feedback, and session memory across multiple tables with foreign key relationships.

## Installation

GraphSift requires Python 3.9 or later. The core package has a single mandatory dependency on Pydantic version 2.0 or later. Zero native extensions are required for basic operation.

### Base Installation

Install the base package with pip install graphsift. This provides the core pipeline including BM25 ranking, graph construction, token budget selection, context rendering, and CLI commands. The base installation covers all 14 languages using regex-based parsing.

### Tree-Sitter Installation

For precise AST parsing across 11 languages, install with the tree-sitter extra: pip install graphsift[treesitter]. This adds tree-sitter grammar packages for Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, and Bash. Tree-sitter parsing provides more accurate symbol extraction, better edge detection, and improved line-level granularity for diff-aware trimming.

### Full Installation

For the complete GraphSift experience including output compression and developer tools, install with the all extra: pip install graphsift[all]. This includes tree-sitter grammars, the tokenpruner compression library, and everything needed for maximum token savings.

### Development Installation

For contributors, clone the repository from GitHub, install with development dependencies using pip install -e .[dev], and run the test suite with pytest tests -v. All tests pass in approximately four seconds across eight test files covering the core pipeline, advanced features, hybrid search, tree-sitter parsing, diff trimming, deduplication, and auto-fix suggestions.

### Verification

After installation, verify the package works by running graphsift status or checking the version with python -m graphsift --help. The MCP server can be installed into any project with graphsift install from the project root.

---

## MCP Server Tools Overview

The GraphSift MCP server provides over 40 tools for code intelligence, context optimization, impact analysis, refactoring, compression, and analytics. It runs over the stdio transport and is compatible with any MCP client including Claude Code, Claude desktop, Cursor, Copilot, Windsurf, Codex, and Gemini.

### Graph Management Tools

| Tool | Description | Category |
|------|-------------|----------|
| build_graph | Index all source files under a root path and build the full dependency graph. Call once per session or after large changes. Returns files indexed, symbols extracted, and edges created. | Graph |
| update_graph | Incrementally update the graph with only the changed files. Much faster than a full rebuild. Called automatically by the PostToolUse hook after Write, Edit, or Bash tool calls. | Graph |
| clear_graph | Clear the in-memory graph for a root path, forcing a full rebuild on the next call. | Graph |
| graph_status | Check if the graph is built and see current statistics including files, symbols, edges, and SQLite database info. | Status |
| list_graph_stats | Ultra-compact graph statistics in approximately 100 tokens. Returns node, edge, and file counts plus schema version as a one-line summary. Use instead of graph_status when only counts are needed. | Status |
| embed_graph | Compute TF-IDF symbol embeddings and store them in SQLite. No external machine learning dependencies. Improves semantic search ranking. Run once after build_graph and run_postprocess. | Graph |

### Context and Review Tools

| Tool | Description | Category |
|------|-------------|----------|
| get_context | Build ranked, token-budget-aware context for a code diff or query. Returns only the most relevant files — typically 80 to 150 times fewer tokens than sending the whole repo. Use the rendered context as the code block in an LLM prompt. | Context |
| get_review_context | Token-efficient code review context. Returns structured source snippets for changed files plus key dependents, capped by lines per file. Uses approximately 5 to 10 times fewer tokens than get_context. Ideal for focused review prompts. | Context |
| minimal_context | Ultra-low-token context with signatures only, no source bodies. Ideal for quick orientation or when the token budget is under 8,000 tokens. | Context |
| get_file_context | Retrieve the full source of a specific indexed file by path (absolute or partial match). | Files |

### Impact Analysis Tools

| Tool | Description | Category |
|------|-------------|----------|
| get_impact | Return the blast radius — all files potentially affected by changes. Each file is scored zero to one by dependency distance. Useful for risk assessment. | Impact |
| get_impact_radius | Compact blast-radius analysis returning file paths, scores, and depth only, no source. Uses approximately 10 times fewer tokens than detect_changes. Use for quick impact checks. | Impact |
| detect_changes | Detect changed files and return risk-scored impact analysis with blast radius. Includes optional source previews and configurable detail levels. | Impact |
| get_affected_flows | Find execution flows that pass through changed files, sorted by criticality. | Impact |

### Code Search Tools

| Tool | Description | Category |
|------|-------------|----------|
| search_symbols | Search for functions, classes, or modules by name across the indexed codebase. Case-insensitive substring matching. | Search |
| semantic_search_nodes | Search for code symbols by name or keyword using FTS5 full-text search. When embeddings are available, falls back to hybrid BM25 plus TF-IDF ranked search. | Search |
| cross_repo_search | Search for code entities across all registered repositories in the global graphsift registry. | Search |
| query_graph | Run predefined graph queries: callers_of, callees_of, imports_of, importers_of, tests_for, children_of, inheritors_of, and file_summary. | Query |

### Code Analysis Tools

| Tool | Description | Category |
|------|-------------|----------|
| list_flows | List detected execution flows sorted by criticality. Run run_postprocess first. | Analysis |
| get_flow | Get detailed information about a single execution flow, including the call path and optional source snippets. | Analysis |
| list_communities | List detected code communities sorted by size. Run run_postprocess first. | Analysis |
| get_community | Get details about a single code community, optionally including its member nodes. | Analysis |
| get_architecture_overview | Generate an architecture overview showing communities, high-risk files, total nodes, edges, and files. | Analysis |
| run_postprocess | Run flow detection, community detection, FTS rebuild, and risk scoring on the built graph. Call after build_graph. | Analysis |
| find_large_functions | Find the largest functions and classes by line count. Useful for identifying bloat before sending context to an LLM. Compact output with name, file, line range, and size. | Analysis |
| list_files | List all indexed files sorted by token estimate (highest first). Useful for understanding repository size and identifying large files. | Files |

### Refactoring Tools

| Tool | Description | Category |
|------|-------------|----------|
| refactor | Three modes: rename preview (shows all affected references before applying), dead_code detection (find unreachable symbols), and suggest (run auto-fix analysis for unused imports, missing types, long functions, cycles, and dead code). | Refactor |
| apply_refactor | Apply a previously previewed rename to source files. All edits are validated to stay within the repository root. | Refactor |
| detect_cycles | Detect circular dependencies (import and call cycles) using Tarjan's strongly-connected-components algorithm. Returns cycle details with severity grading. | Analysis |
| detect_dead_code | Find potentially unreachable code via BFS reachability analysis from entry points. Filters by kind (function, class, method). | Analysis |
| suggest_fixes | Run auto-fix analysis on the dependency graph and return prioritized, confidence-scored fix suggestions. Detects unused imports, missing type annotations, overly long functions and parameter lists, dependency cycles, and dead code. Read-only. | Refactor |

### Documentation Tools

| Tool | Description | Category |
|------|-------------|----------|
| generate_wiki | Generate Markdown wiki pages from the community structure into a local directory. Each community gets its own page with member files and relationships. | Docs |
| get_wiki_page | Retrieve a specific wiki page by community name. Run generate_wiki first. | Docs |
| get_docs_section | Fetch a single section from a community wiki page by heading keyword. Returns only the matched heading block — far fewer tokens than get_wiki_page. | Docs |

### Compression and Analytics Tools

| Tool | Description | Category |
|------|-------------|----------|
| compress_output | Compress command output to save 60 to 90 percent tokens before sending to an LLM. Auto-detects 18 or more command types including pytest, cargo, go test, jest, eslint, git, npm, docker, and kubectl. | Compression |
| token_gain | Show token savings analytics including total calls, tokens saved, estimated cost savings, and daily breakdown. | Analytics |
| token_discover | Find missed token-saving opportunities — identify which commands would benefit most from compression. | Analytics |

### Registry and Feedback Tools

| Tool | Description | Category |
|------|-------------|----------|
| list_repos | List all repositories registered in the global graphsift registry. | Registry |
| save_review_feedback | Save a one-to-five rating on context selection quality. Feedback accumulates across sessions to improve ranking weights. | Feedback |
| get_context_quality | Return aggregate context quality statistics from all review feedback, including total count, average rating, distribution, and recent entries. | Feedback |

### MCP Prompts

In addition to tools, the MCP server provides four reusable prompt templates:

| Prompt | Description |
|--------|-------------|
| review_code | Guided code review workflow using ranked context. Walks through building or checking the graph, retrieving optimized context, and performing the review. |
| analyze_impact | Impact analysis workflow for understanding blast radius. Guides through graph status check, impact retrieval, and risk assessment. |
| find_issues | Code quality analysis workflow using auto-fix suggestions and cycle detection. |
| explain_architecture | Architecture overview workflow using community structure and flow detection. |

### MCP Resources

Ten MCP resources expose structured data from the graph database:

| Resource | Description |
|----------|-------------|
| graphsift://stats | Graph statistics including node and edge counts, file count, and schema version. |
| graphsift://architecture | Architecture overview with communities, risk files, and structure summary. |
| graphsift://communities | List of all detected code communities. |
| graphsift://community/{name} | Detailed view of a specific community by name. |
| graphsift://flows | List of all detected execution flows. |
| graphsift://flow/{id} | Detailed view of a specific flow by ID. |
| graphsift://wiki/{name} | Generated wiki page for a specific community. |
| graphsift://risk | Risk index for all scored files. |
| graphsift://status/{root} | Status check for a specific repository root. |
| graphsift://repos | List of all registered repositories. |

---

## Supported Languages

GraphSift supports 14 programming languages with three parsing tiers: native AST, tree-sitter precise CST/AST, and custom parsers for domain-specific languages.

| Language | Parser | Tree-Sitter | Key Capabilities |
|----------|--------|-------------|------------------|
| Python | Native AST plus tree-sitter | Yes | Functions, classes, methods, async functions, decorators, dynamic imports, async generators |
| JavaScript | Regex plus tree-sitter | Yes | Functions, classes, methods, arrow functions, async functions, generators |
| TypeScript | Regex plus tree-sitter | Yes | Same as JavaScript plus type annotations, interfaces, enums, type aliases |
| Go | Regex plus tree-sitter | Yes | Functions, receiver methods, structs, interfaces, methods |
| Rust | Regex plus tree-sitter | Yes | Functions, structs, traits, impl blocks, enums |
| Java | Regex plus tree-sitter | Yes | Classes, methods, interfaces, anonymous classes |
| C++ | Regex plus tree-sitter | Yes | Functions, classes, structs, templates, namespaces |
| C | Regex plus tree-sitter | Yes | Functions, structs, typedefs |
| Ruby | Regex plus tree-sitter | Yes | Methods, classes, modules, blocks |
| PHP | Regex plus tree-sitter | Yes | Functions, classes, traits, interfaces |
| Bash and Shell | Regex plus tree-sitter | Yes | Functions, source imports, variable references |
| Terraform and HCL | Custom parser | No | Resources, variables, locals, modules, data sources, outputs |
| Helm Charts | Custom template parser | No | Go templates embedded in YAML, Chart.yaml dependencies, values references |
| Dockerfile | Custom parser | No | FROM, COPY, RUN, ENV, ARG, CMD, ENTRYPOINT instructions |

All 14 languages participate in the same ranking, selection, and rendering pipeline. Mixed-language repositories are fully supported — a Python backend with a JavaScript frontend, for example, will have cross-language dependency edges tracked and scored.

---

## CLI Commands Reference

GraphSift provides a full command-line interface for managing graphs, compression, and analytics.

### Installation and Setup

| Command | Description |
|---------|-------------|
| graphsift install | Register the MCP server with Claude Code, inject hooks for automatic graph updates and output compression, and optionally install skill files and the bash wrapper. Run from the project root. |
| graphsift uninstall | Remove the MCP server configuration, skill files, and local graph data. |

### Graph Management

| Command | Description |
|---------|-------------|
| graphsift build | Index all source files in the repository and build the dependency graph. Supports filtering by file extension and excluding directories. By default, also runs post-processing for flows, communities, risk scoring, and FTS indexing. |
| graphsift update | Incrementally update the graph with only files that have changed since the last build. Called automatically by the PostToolUse hook. |
| graphsift postprocess | Run flow detection, community detection, risk scoring, and FTS rebuild on an existing graph. Each step can be individually skipped. |
| graphsift status | Show installation status and graph statistics including files indexed, symbols extracted, edges created, MCP configuration status, and hook status. |
| graphsift watch | Watch the filesystem for changes and automatically update the graph when files are modified or added. |

### Repository Management

| Command | Description |
|---------|-------------|
| graphsift register | Register the current directory as a repository in the global graphsift registry. Supports optional display names. |
| graphsift unregister | Remove a repository from the registry by path or display name. |
| graphsift list-repos | List all registered repositories with their root paths and database locations. |
| graphsift repos | Alias for list-repos. |

### Output Compression

| Command | Description |
|---------|-------------|
| graphsift compress | Compress command output piped from standard input. Auto-detects the command type and applies the optimal compression strategy. Supports ultra mode for aggressive compression and tee mode for saving original output. |
| graphsift bash-wrapper | Print the bash wrapper script for transparent output compression. Source this in shell configuration files. |

### Analytics

| Command | Description |
|---------|-------------|
| graphsift gain | Show cumulative token savings analytics including total calls, tokens saved, and estimated cost savings. Supports JSON output and historical breakdowns. |
| graphsift discover | Find missed token-saving opportunities by analyzing which commands produce the most uncompressed output. |

### Code Analysis

| Command | Description |
|---------|-------------|
| graphsift detect-changes | Perform risk-scored impact analysis on specified changed files. Returns blast radius with relevance scores, depths, and risk scores. |
| graphsift detect-cycles | Detect circular dependencies in the codebase using Tarjan's strongly-connected-components algorithm. Reports cycle length and severity. |
| graphsift detect-dead-code | Find potentially unreachable code via BFS reachability from entry points. Supports filtering by symbol kind (function, class, method). |
| graphsift suggest-fixes | Run auto-fix analysis across five categories (import issues, type issues, structure issues, dependency cycles, dead code). Returns prioritized, confidence-scored suggestions with auto-fixable flags. |

### Visualization and Documentation

| Command | Description |
|---------|-------------|
| graphsift visualize | Generate an interactive HTML dependency graph visualization using D3.js. Supports serving the visualization on localhost. |
| graphsift wiki | Generate Markdown wiki pages from the community structure of the graph. Each community gets its own page with member files and relationships. |

### Server

| Command | Description |
|---------|-------------|
| graphsift serve | Start the MCP server in stdio mode for custom MCP clients. Used internally by the MCP integration. |

---

## FAQ

### How do I save tokens on Claude Code?

Install the GraphSift MCP server using graphsift install from your project root. Restart Claude Code to load the server. Once loaded, ask Claude to build the knowledge graph. After that, every get_context call automatically delivers optimized, token-budget-capped context. The PostToolUse hook also compresses command output transparently.

### What is the fastest way to reduce API costs?

Use ContextBuilder with a hard token budget. It ranks files by relevance and only sends what fits within the budget. Add diff-aware trimming for another 40 to 60 percent reduction. Enable cache-aware output for repeated queries. On average, this reduces API costs by 93 to 99 percent per review.

### How much does API cost per code review?

Without GraphSift, a typical code review costs USD 0.50 to USD 2.70 in API fees depending on model and repository size. With GraphSift, the same review costs USD 0.01 to USD 0.05 — a 93 to 99 percent reduction. At 100 pull requests per day, this saves USD 50 to USD 270 per day versus USD 1 to USD 5 per day with GraphSift.

### Does GraphSift work with GPT-4, GPT-5, and OpenAI?

Yes. Drop-in adapters are provided for OpenAI and Codex, plus a generic adapter for any OpenAI-compatible API. A Gemini adapter is also available. The ranking and selection logic is provider-agnostic — it works with any LLM that accepts text context.

### How is GraphSift different from code-review-graph?

Code-review-graph uses binary blast-radius file inclusion — a file is either included or excluded based on graph connectivity. GraphSift treats context selection as a ranking problem, scoring every file zero to one and selecting greedily within a token budget. The result is F1 accuracy of 0.85 versus 0.54, and token reduction of 80 to 150 times versus 8 to 49 times. GraphSift also supports multi-file diffs, decorator tracking, dynamic imports, output compression, deduplication, tree-sitter parsing, hybrid search, cycle detection, dead code detection, auto-fix suggestions, 14 languages, incremental indexing, monorepo support, a full MCP server with over 40 tools, and token savings analytics.

### Can GraphSift handle monorepos?

Yes. The index_roots function indexes multiple packages at once with correct cross-package dependency scoring. Files from different packages that depend on each other are properly ranked and selected within the unified token budget. The multi-repo registry also supports searching across multiple registered repositories.

### Does GraphSift need internet access?

No. All parsing, ranking, and compression runs entirely locally. The SQLite database stores all graph data on the local machine. The MCP server binds to localhost only. No telemetry, no data ever leaves your machine, no accounts or API keys are needed. The only components that make network calls are the optional LLM API adapters for Claude, OpenAI, and Gemini.

### What Python versions are supported?

Python 3.9, 3.10, 3.11, 3.12, and 3.13 are supported. The only mandatory dependency is Pydantic version 2.0 or later.

### What is the token reduction for CLI output compression?

Across all 19 supported command types, the weighted average token savings is approximately 77 percent. Individual savings range from 29 percent for cat output to 95 percent for grep results. At 100 CLI commands per day piped to an LLM, this saves approximately 625,000 tokens per day, or roughly USD 9.37 per day on Claude Opus pricing.

### Which MCP clients are supported?

GraphSift's MCP server works with any client that implements the Model Context Protocol. Tested clients include Claude Code, Claude desktop app, Cursor, Copilot, Windsurf, Codex CLI, and Gemini. The server uses the stdio transport and exposes tools, prompts, and resources per the MCP specification.

### How does the bash wrapper work?

The bash wrapper intercepts supported commands (pytest, cargo, npm, docker, kubectl, aws, grep, cat, make, pip, jest, eslint, git, go test, npx, yarn) and pipes their output through the graphsift compress pipeline before it reaches the LLM context. The LLM never sees the raw, noisy output. Install it with graphsift install --bash-wrapper, which adds the wrapper to your shell configuration file.

### What privacy guarantees does GraphSift offer?

GraphSift runs 100 percent locally with no telemetry. No data ever leaves your machine. No internet is required for parsing, ranking, or compression. The SQLite database stores all data locally. The MCP server binds to localhost only (127.0.0.1). There are no accounts, no API keys, and no cloud dependencies. Zero external network requests are made during normal operation.

### How do I uninstall GraphSift?

Remove the package with pip uninstall graphsift. Delete the local data directory at .graphsift to clear all graph data, configuration, and cached results. To remove the MCP configuration, run graphsift uninstall from the project root, which removes the MCP server entry from the project configuration, deletes skill files, and removes the local graph manifest.

### How does the three-tier hot, warm, cold selection work?

Files are assigned to tiers based on their combined BM25 and graph-distance relevance score. Hot files (typically score above 0.7) receive FULL source inclusion in the rendered context. Warm files (score between 0.3 and 0.7) receive SIGNATURES only — just the function and class declarations without implementation bodies. Cold files (score below 0.3) are excluded entirely. The tier thresholds adjust dynamically based on the token budget and the score distribution of the current codebase. In a tight budget, the warm tier may be excluded entirely to make room for more hot files.

### What is diff-aware trimming?

Diff-aware trimming analyzes the unified diff of the changes being reviewed and keeps only the parts of each file that are near the changed lines. By default, 10 lines of context above and below each change are included. This reduces token count by 40 to 60 percent compared to including the entire file. The context window size is configurable. For a 500-line file with a 5-line change, diff-aware trimming keeps approximately 25 lines instead of 500.

### Can I use GraphSift in CI/CD pipelines?

Yes. GraphSift runs entirely locally with no dependencies on external APIs (except the optional LLM adapters). It can be installed in any CI/CD environment that supports Python 3.9 or later. The typical CI/CD flow is: install graphsift, run graphsift build to index the repository, then use the Python API or CLI to build optimized context for each pull request. The token savings analytics can track cost reduction across the entire CI/CD pipeline.

### Does GraphSift support Windows?

Yes. GraphSift is tested on Linux, macOS, and Windows. The core package is pure Python with no native extensions. The tree-sitter extra requires compiled grammar packages which are available for all major platforms via PyPI wheels. The bash wrapper is Unix-only by nature, but the compress CLI command works on all platforms.

### How is the BM25 score computed?

BM25 is a bag-of-words ranking function that scores each file's content against the review query. For each query term, BM25 considers the term frequency within the file, the inverse document frequency across all indexed files, and applies length normalization. Files that contain query terms at higher frequency, in rarer contexts, and in proportionally smaller files receive higher BM25 scores. The BM25 parameters k1 (1.2) and b (0.75) use standard defaults from information retrieval literature.

### What is entropy-based deduplication?

Entropy-based deduplication detects near-identical files in the candidate set and removes redundant copies. It works by computing a sliding-window hash of each file's content and comparing the hash sets between file pairs. Files with high hash overlap (above a configurable threshold) are considered near-duplicates. When duplicates are detected, only the file with the highest relevance score is kept. This prevents the context from including multiple copies of essentially the same code, which would waste tokens and degrade LLM output quality.

### How does the A2A protocol server work?

The A2A (Agent-to-Agent) protocol server exposes GraphSift's code intelligence capabilities to other agents over HTTP using JSON-RPC. It implements the A2A agent card specification, advertising capabilities such as context building, impact analysis, code search, and refactoring. Other agents can discover GraphSift via the agent card, send JSON-RPC requests to the HTTP endpoint, and receive structured responses. The server binds to localhost by default for security.

### What advanced features are available in the Python API?

The Python API provides ten advanced feature categories: Smart Cache (LRU plus TTL caching with memoize decorator and hit rate statistics), Analysis Pipeline (chainable processing steps with retry and audit traceability), Async Batch (concurrent multi-repo and multi-diff processing with bounded semaphores), Streaming (progressive context delivery as files are ranked), Rate Limiter (token-bucket rate control with per-key tracking), Diff Engine (comparison of context results across configuration changes), Circuit Breaker (three-state protection against cascading API failures), Retry (exponential backoff with jitter), Schema Evolution (six-version SQLite migration history), and Audit Trail (full input-output-parameter recording for every context build).

### How do I contribute to GraphSift?

Contributions are welcome via the GitHub repository. Read the contributing guide for guidelines on code style, testing, and pull request workflow. The development setup requires cloning the repository, installing with dev dependencies, and running the test suite. The test suite contains 271 tests across 8 test files and takes approximately 4 seconds to run.

### What is the indexing performance?

Indexing completes in under two seconds on repositories with 10,000 or more files. Incremental re-indexing skips unchanged files via SHA-256 hash comparison. Context building takes under 50 milliseconds for a typical diff on an indexed 1,000-file repository. The depth cap (default 4) prevents infinite traversal on cyclic imports. All shared state is protected behind reentrant locks. Async twins are available for all blocking operations.

### How many tests does the project have?

The test suite contains 271 tests across 8 test files covering the core pipeline, advanced features, hybrid search, tree-sitter parsing, diff trimming, deduplication, and auto-fix suggestions. All tests pass in approximately four seconds. Test coverage exceeds 80 percent.

---

## Links

- [GitHub Repository](https://github.com/maheshmakvana/graphsift) — Source code, issues, pull requests, and releases.
- [PyPI Package](https://pypi.org/project/graphsift/) — Python package index page with installation instructions and version history.
- [Issue Tracker](https://github.com/maheshmakvana/graphsift/issues) — Report bugs, request features, or ask questions.
- [Changelog and Releases](https://github.com/maheshmakvana/graphsift/releases) — Release notes and version history.
- [Contributing Guide](https://github.com/maheshmakvana/graphsift/blob/master/CONTRIBUTING.md) — Guidelines for contributors.
- [MIT License](https://github.com/maheshmakvana/graphsift/blob/master/LICENSE) — Open source license.

### Related Projects

- tokenpruner — LLM input token compression used by GraphSift's COMPRESSED output mode. Adds three to five times additional token reduction. Available on PyPI.
- code-review-graph — Binary blast-radius alternative without ranking, token budget, or compression. GraphSift was built to surpass its limitations.

### Repository Topics

python, ai, mcp, developer-tools, llm, copilot, claude-code, token-optimization, mcp-server, code-review, agentic-coding, context-engineering, reduce-token-costs, ast-parser, dependency-graph, context-window, tree-sitter, output-compression, bm25, agent-memory, graphrag, a2a-protocol

---

*Start saving tokens today: pip install graphsift*
