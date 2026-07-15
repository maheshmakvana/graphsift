# graphsift Wiki Home

> **graphsift v3.0** — Created by [Mahesh Makwana](https://github.com/maheshmakvana).  
> The #1 token saver for Claude, GPT-4 & Gemini. 80-150× fewer tokens, F1 0.85 relevance accuracy, 93% feature coverage.

<p align="center">
  <a href="https://pepy.tech/projects/graphsift"><img src="https://static.pepy.tech/badge/graphsift" alt="Downloads"></a>
  <a href="https://pepy.tech/projects/graphsift"><img src="https://static.pepy.tech/badge/graphsift/month" alt="Downloads/month"></a>
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/v/graphsift.svg?style=flat&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/pyversions/graphsift.svg?style=flat" alt="Python"></a>
  <a href="https://github.com/maheshmakvana/graphsift/stargazers"><img src="https://img.shields.io/github/stars/maheshmakvana/graphsift?style=flat&color=yellow" alt="Stars"></a>
  <br>
  <sub>📥 <strong>10,365 total downloads</strong> · 1,533 in last 30 days · <a href="https://pepy.tech/projects/graphsift">View live chart →</a></sub>
</p>

**Save Claude tokens. Reduce GPT-4 costs. Optimize Gemini context windows.**

## What is graphsift?

GraphSift (often written **graphsift**) is a Python library and MCP server that performs **ranked context selection** for LLM-powered code analysis. It takes a code repository and a change specification (diff) and returns a curated, token-budgeted subset of the repository that is most relevant to that change.

### Why is this useful?

When you ask an LLM to review a code change, you need to send relevant context. Sending the entire codebase is expensive (tens of thousands of tokens). Sending nothing means the LLM misses critical dependencies. GraphSift solves this by automatically identifying the most relevant files, ranking them by importance, and fitting them within a token budget.

### How does it work?

GraphSift constructs an AST dependency graph of the entire repository, identifying symbols, imports, function calls, class hierarchies, and decorator patterns across **14 languages**. When you request context for a specific diff, GraphSift performs a BFS traversal from the changed files to find all related files across the repository. A relevance ranker scores each candidate file using BM25 (keyword match) and graph distance (structural proximity). The context selector then applies a token budget, assigns files to HOT/WARM/COLD tiers based on their scores, renders full source for the most important files and signatures for moderately important ones, and optionally applies diff-aware trimming and entropy-based deduplication. The result is a compact, highly relevant context that costs **80 to 150 times fewer tokens** than raw source.

### What's new in v3.0?

**v3.0 adds 11 entirely new modules** with 702 tests (+115% over v2.3), achieving **93% feature coverage** (25/27 features):

| Module | What It Does |
|--------|-------------|
| **Planner** 🗺️ | Plan-first engine — scan repo, architect, execute, validate across 7 phases |
| **ToolChain** ⛓️ | DAG-based workflow automation with rollback and review |
| **AutoVerifier** ✅ | Self-verification cascade (syntax → lint → tests → fix loop) |
| **ConventionLearner** 📐 | Learns team coding conventions, stores in CodeMemory (365d TTL) |
| **ContextEnricher** 🔍 | Multi-source code exploration — discovers related symbols, imports, patterns |
| **AsyncEngine** ⚡ | Async parallel execution across repos and diffs (semaphore=8) |
| **ASTCache** 💾 | 2-tier LRU+SQLite cache with TTL, warm(), predictive warming |
| **Pool** 🗄️ | Thread-safe DB connection pool (WAL mode, auto-reconnect) |
| **SecurePipeline** 🔒 | End-to-end secure pipeline (PathValidator + CommandSanitizer + DataScrubber) |
| **DataScrubber** 🧹 | Redacts API keys, tokens, passwords from CLI output |
| **SchemaRegistry** 📋 | 6 schema families with v1→v2 auto-migration |

---

### What is BM25 ranking?

BM25 is a bag-of-words ranking function that scores each file's content against the review query. For each query term, BM25 considers the term frequency within the file, the inverse document frequency across all indexed files, and applies length normalization. Files that contain query terms at higher frequency, in rarer contexts, and in proportionally smaller files receive higher BM25 scores. The BM25 parameters k1 (1.2) and b (0.75) use standard defaults from information retrieval literature. v3.0 adds **3 fusion modes** (hybrid/rrf/rrf-dense) with optional dense vector embeddings.

### What is entropy-based deduplication?

Entropy-based deduplication detects near-identical files in the candidate set and removes redundant copies. It works by computing a sliding-window hash of each file's content and comparing the hash sets between file pairs. Files with high hash overlap (above a configurable threshold) are considered near-duplicates. When duplicates are detected, only the file with the highest relevance score is kept. This prevents the context from including multiple copies of essentially the same code, which would waste tokens and degrade LLM output quality.

### How does the A2A protocol server work?

The A2A (Agent-to-Agent) protocol server exposes GraphSift's code intelligence capabilities to other agents over HTTP using JSON-RPC. It implements the A2A agent card specification, advertising capabilities such as context building, impact analysis, code search, and refactoring. Other agents can discover GraphSift via the agent card, send JSON-RPC requests to the HTTP endpoint, and receive structured responses. The server binds to localhost by default for security.

### How does the Planner work?

The Planner generates structured execution plans from a task description + dependency graph. It produces a dependency-ordered sequence across **7 phases**: SCAN → ANALYZE → ARCHITECT → PLAN → EXECUTE → VALIDATE → REVIEW. Each step has clear status tracking (pending/running/completed/failed/skipped), and the plan is validated against the graph before any execution begins. Combined with ToolChain (DAG-based step chains) and AutoVerifier (self-verification cascade), this gives you an **autonomous coding agent** workflow.

### What advanced features are available in the Python API?

The Python API provides **12 advanced feature categories**: Smart Cache (LRU+TTL + SQLite 2-tier caching), Analysis Pipeline (chainable processing with retry and audit), Async Batch (concurrent multi-repo processing), Streaming (progressive context delivery), Diff Engine (compare context results across configs), Schema Evolution (6-version migration), Grid Search (hyperparameter optimization), Audit Trail (full input-output recording), Circuit Breaker (3-state API protection), Rate Limiter (token-bucket control), Retry (exponential backoff), and Pool (thread-safe connection pooling).

### How do I contribute to GraphSift?

Contributions are welcome via the GitHub repository. Read the contributing guide for guidelines on code style, testing, and pull request workflow. The development setup requires cloning the repository, installing with dev dependencies, and running the test suite. The test suite contains **702 tests across 42 test files** (5 categories: unit, fuzz, integration, property, stress) and takes approximately 6 seconds to run.

### What is the indexing performance?

Indexing completes in **under two seconds** on repositories with 10,000 or more files. Incremental re-indexing skips unchanged files via SHA-256 hash comparison. Context building takes under **50 milliseconds** for a typical diff on an indexed 1,000-file repository. The 2-tier ASTCache (LRU + SQLite) provides **sub-5ms cache-hit retrieval** with predictive warming. Three concurrency tiers (asyncio / thread pool / process pool) ensure optimal performance on any hardware.

### How many tests does the project have?

The test suite contains **702 tests across 42 test files** (v2.3 had 326 tests across 13 files — a **115% increase**). Tests are organized into **5 categories**: unit tests, fuzz tests, integration tests, property-based tests, and stress/performance tests. All tests pass in approximately six seconds. Test coverage exceeds 85 percent.

### What security features does graphsift have?

GraphSift v3.0 adds **3 new security classes** (5 total):

| Class | What It Protects Against |
|-------|------------------------|
| `PathValidator` | Path traversal attacks (`../../../etc/passwd`) |
| `CommandSanitizer` | Command injection (`; rm -rf /`) |
| `DataScrubber` | Secret leakage (API keys, tokens, passwords) |
| `SecurePipeline` | Combined end-to-end protection |
| `NetworkAccessError` | Unauthorized network access |

All built on a **zero-exfiltration architecture** — no telemetry, no network calls, no LLM calls from library code.

### What memory systems does graphsift have?

GraphSift v3.0 provides **3 memory systems** (v2.3 had 1):

1. **AgentMemory** — SQLite-backed cross-session knowledge graph
2. **CodeMemory** — Code-anchored memory with 7 types (decision, gotcha, note, insight, todo, bug, convention) and TTL-based expiry
3. **TieredMemory** — 4-tier hierarchy (axioms → rules → topic → archive)

Plus **ConventionLearner** for team-shared coding conventions (365-day TTL).

---

## Links

- [GitHub Repository](https://github.com/maheshmakvana/graphsift) — Source code, issues, pull requests, and releases.
- [PyPI Package](https://pypi.org/project/graphsift/) — Python package index page with installation instructions and version history.
- [Issue Tracker](https://github.com/maheshmakvana/graphsift/issues) — Report bugs, request features, or ask questions.
- [Changelog and Releases](https://github.com/maheshmakvana/graphsift/releases) — Release notes and version history.
- [V3 Upgrade Guide](https://github.com/maheshmakvana/graphsift/blob/master/docs/V3_UPGRADE_GUIDE.md) — Full v2.3 → v3.0 comparison matrix.
- [Contributing Guide](https://github.com/maheshmakvana/graphsift/blob/master/CONTRIBUTING.md) — Guidelines for contributors.
- [MIT License](https://github.com/maheshmakvana/graphsift/blob/master/LICENSE) — Open source license.

### Related Projects

- **[Caveman](https://github.com/JuliusBrussee/caveman)** 🗿 — Compress LLM output tokens (~65% savings). Use with graphsift for full pipeline savings.
- **tokenpruner** — LLM input token compression used by GraphSift's COMPRESSED output mode.
- **code-review-graph** — Binary blast-radius alternative without ranking, token budget, or compression.

### Author

**Mahesh Makwana** — [GitHub](https://github.com/maheshmakvana) · [X/Twitter](https://x.com/makwanamahesh5) · [Email](mailto:maheshmakwana527@gmail.com)

### Repository Topics

python, ai, mcp, developer-tools, llm, copilot, claude-code, token-optimization, mcp-server, code-review, agentic-coding, context-engineering, reduce-token-costs, ast-parser, dependency-graph, context-window, tree-sitter, output-compression, bm25, agent-memory, graphrag, a2a-protocol, autonomous-coding, v3

---

*Save tokens today: pip install graphsift · Built by Mahesh Makwana · MIT License · v3.0*
