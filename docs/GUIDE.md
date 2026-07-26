# GraphSift Architecture Guide

> **graphsift v4.5.0** — Created by [Mahesh Makwana](https://github.com/maheshmakvana).  
> Token optimization engine for Claude, GPT-5 & Gemini. 80-150× reduction, F1 0.85.

---

## System Overview

GraphSift is a **Python library and MCP server** that performs **ranked context selection** for LLM-powered code analysis. It takes a code repository and a change specification (diff) and returns a curated, token-budgeted subset of the repository that is most relevant to that change.

### Core Pipeline

```
Source Code → AST Parser → Dependency Graph → BM25 Index → Relevance Ranker → Context Selector → Rendered Context
```

1. **AST Dependency Graph** — Parses source files across 14 languages, extracting symbols, imports, function calls, class hierarchies, and decorator patterns
2. **BM25+Graph Ranking** — Scores every file 0–1 using full-text search (BM25, k1=1.2, b=0.75) combined with graph distance from changed files
3. **3-Tier Context Selection** — HOT (full source), WARM (signatures only), COLD (excluded). Applies token budgets, diff-aware trimming, and SimHash deduplication
4. **Rendered Output** — Delivers 80–150× fewer tokens than raw source at F1 0.85 relevance

### Key Design Principles

- **Zero external dependencies** for core functionality (only `pydantic>=2.0`)
- **Local-first** — all processing happens on your machine, no network calls
- **SQLite-backed persistence** with WAL mode, mmap, and FTS5 covering indexes
- **Async-first** concurrency via asyncio, thread pools, and process pools

---

## Module Architecture

### Core Context Layer
- `ContextBuilder` — Token-budgeted context from diffs. HOT/WARM/COLD file tiers
- `RelevanceRanker` — BM25 + graph proximity scoring
- `DependencyGraph` — AST-based import/symbol graph across 14 languages

### Compression Layer
- `compress()` — 25 command-specific compressors (pytest, git, npm, docker, kubectl, etc.)
- `ConversationCompactor` — Agent dialogue compression (60–82%)
- `AutonomousCompressor` — Self-triggering at configurable thresholds

### Memory & Search
- `AgentMemory` — Cross-session knowledge graph (SQLite)
- `HybridSearcher` — BM25 + TF-IDF + optional dense vectors
- `TemporalGraph` — Git-history-aware symbol tracking

### Planning & Security
- `Planner` — 7-phase plan-then-execute engine
- `SecurePipeline` — PathValidator + CommandSanitizer + DataScrubber

---

## Frequently Asked Questions

### What is BM25 ranking?
BM25 is a bag-of-words ranking function that scores each file against the review query. Parameters use standard IR defaults (k1=1.2, b=0.75). GraphSift supports 3 fusion modes: hybrid, RRF, and RRF-dense (with optional sentence-transformers).

### What is entropy-based deduplication?
Detects near-identical files via sliding-window hash comparison. When duplicates are found, only the highest-scored file is kept — preventing token waste from redundant context.

### How does the A2A server work?
The Agent-to-Agent protocol server exposes GraphSift capabilities over HTTP using JSON-RPC. It implements the A2A agent card specification, allowing other agents to discover and use GraphSift's code intelligence.

### What indexing performance should I expect?
- Index 10,000+ files: < 2 seconds (single pass)
- Incremental re-index: < 0.5 seconds
- Context build (1k files): < 50 ms
- Cache-hit retrieval: < 5 ms

### What security features are included?
- `PathValidator` — Blocks path traversal attacks
- `CommandSanitizer` — Prevents command injection
- `DataScrubber` — Redacts API keys, tokens, passwords from output
- **Zero telemetry** — no analytics, no network calls, no code exfiltration

### What languages does the AST parser support?
Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Bash — 11 languages via optional tree-sitter parsers.

### Does this work with any LLM?
Yes — Claude Code, GPT-5, Gemini, Codex CLI, Copilot CLI, and any OpenAI-compatible API. Install via `graphsift install --all` for MCP integration with Claude.

---

## Use Cases

| Scenario | Without GraphSift | With GraphSift |
|----------|-----------------|----------------|
| Code review (50-line change, 143 files) | ~180K tokens, $2.70 (GPT-5) | 800–1,200 tokens, $0.015 |
| CI/CD debug (pytest output) | 3,200 raw tokens | ~480 tokens (85% savings) |
| Multi-file refactor (10 files) | ~15,000 tokens | ~1,500 tokens (90% savings) |
| 2-hour dev session | 140K–200K tokens, $3.00–$5.00 | 17K–23K tokens, $0.40–$0.70 |

---

## More Resources

- [README](../README.md) — Quick start, install, benchmark comparisons
- [API Reference](API_REFERENCE.md) — Full class/function reference
- [Prompt Benchmark](PROMPT_BENCHMARK_2026.md) — 4 prompt architectures compared
- [CHANGELOG](../CHANGELOG.md) — Release history

---

`pip install graphsift` · Zero telemetry · Zero accounts · MIT License
