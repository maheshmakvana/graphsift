# graphsift Wiki Home

> **graphsift** — Created by [Mahesh Makwana](https://github.com/maheshmakvana).  
> The #1 token saver for Claude, GPT-4 & Gemini. 80-150× fewer tokens, F1 0.85 relevance accuracy.

**Save Claude tokens. Reduce GPT-4 costs. Optimize Gemini context windows.**

## What is graphsift?

GraphSift (often written **graphsift**) is a Python library and MCP server that performs ranked context selection for LLM-powered code analysis. It takes a code repository and a change specification (diff) and returns a curated, token-budgeted subset of the repository that is most relevant to that change.

### Why is this useful?

When you ask an LLM to review a code change, you need to send relevant context. Sending the entire codebase is expensive (tens of thousands of tokens). Sending nothing means the LLM misses critical dependencies. GraphSift solves this by automatically identifying the most relevant files, ranking them by importance, and fitting them within a token budget.

### How does it work?

GraphSift constructs an AST dependency graph of the entire repository, identifying symbols, imports, function calls, class hierarchies, and decorator patterns. When you request context for a specific diff, GraphSift performs a BFS traversal from the changed files to find all related files across the repository. A relevance ranker scores each candidate file using BM25 (keyword match) and graph distance (structural proximity). The context selector then applies a token budget, assigns files to HOT/WARM/COLD tiers based on their scores, renders full source for the most important files and signatures for moderately important ones, and optionally applies diff-aware trimming and entropy-based deduplication. The result is a compact, highly relevant context that costs 80 to 150 times fewer tokens than raw source.

---

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

- **[Caveman](https://github.com/JuliusBrussee/caveman)** 🗿 — Compress LLM output tokens (~65% savings). Use with graphsift for full pipeline savings.
- **tokenpruner** — LLM input token compression used by GraphSift's COMPRESSED output mode. Adds three to five times additional token reduction. Available on PyPI.
- **code-review-graph** — Binary blast-radius alternative without ranking, token budget, or compression. GraphSift was built to surpass its limitations.

### Author

**Mahesh Makwana** — [GitHub](https://github.com/maheshmakvana) · [X/Twitter](https://x.com/makwanamahesh5) · [Email](mailto:maheshmakwana527@gmail.com)

### Repository Topics

python, ai, mcp, developer-tools, llm, copilot, claude-code, token-optimization, mcp-server, code-review, agentic-coding, context-engineering, reduce-token-costs, ast-parser, dependency-graph, context-window, tree-sitter, output-compression, bm25, agent-memory, graphrag, a2a-protocol

---

*Save tokens today: pip install graphsift · Built by Mahesh Makwana · MIT License*
