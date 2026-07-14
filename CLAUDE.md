# graphsift — LLM Token Optimizer & Code Context Engine

graphsift saves tokens on AI code review, CLI output compression, and agent context — **80-150x token reduction** vs raw source. AST dependency graph + BM25+graph ranked relevance + hard token budget enforcement. Works with any LLM (Claude, GPT, Gemini, Codex).

## Quick start

```bash
pip install graphsift
graphsift build           # Index repo → dependency graph
graphsift compress        # Pipe CLI output for 60-97% token reduction
graphsift gain            # Show cumulative token savings
```

## Commands

| Command | Description |
|---------|-------------|
| `graphsift build` | Index repo and build AST dependency graph |
| `graphsift serve` | Start MCP stdio server (for Claude Code, Cursor, etc.) |
| `graphsift install` | Register graphsift MCP server with Claude Code |
| `graphsift compress` | Compress CLI output (auto-detect: pytest, docker, git, kubectl, npm...) |
| `graphsift gain` | Show token savings analytics |
| `graphsift discover` | Find missed token-saving opportunities |
| `graphsift detect-changes` | Risk-scored impact analysis for changed files |
| `graphsift detect-cycles` | Find circular import/call dependencies |
| `graphsift detect-dead-code` | Identify unreachable code from entry points |
| `graphsift suggest-fixes` | Prioritized auto-fix suggestions |
| `graphsift visualize` | Interactive HTML dependency graph |
| `graphsift wiki` | Generate markdown wiki from community structure |
| `graphsift status` | Installation and graph status |

## Test

```bash
pytest -xvs tests/        # 271+ tests, ~4s
```

## Build & publish

```bash
python -m build
python -m twine upload dist/*
```

## Code conventions

- Pure Python 3.9+, zero hard deps (only pydantic>=2.0 mandatory)
- Type hints everywhere, strict mypy where practical
- No LLM calls in library code — graphsift is a tool for LLMs, not powered by them
- Deletion-based compaction (no summarization — avoids hallucination)
- Hexagonal (ports & adapters) architecture
- Thread-safe: all shared state behind `threading.RLock`

## Architecture

```
graphsift/
├── __init__.py          # Public API: ContextBuilder, ContextConfig, DiffSpec
├── cli.py               # CLI entry point (click-style argparse)
├── mcp_server.py        # 25+ MCP tools, 4 prompts, 10 resources
├── core.py              # Domain logic: ranking, selection, rendering
├── models.py            # Pydantic models
├── compress.py          # 19 per-tool CLI output compressors (86% avg)
├── storage.py           # SQLite persistence (6-version migration)
├── analytics.py         # Token savings tracking
├── memory.py            # Agent memory (SQLite knowledge graph)
├── typed_retrieval.py   # PRISM-style typed graph traversal (6 intents)
├── compact_context.py   # Conversation compaction (60-82% savings)
├── evidence.py          # Audit trail for file selection
├── a2a_server.py        # Agent-to-Agent protocol (JSON-RPC/HTTP)
├── temporal_graph.py    # Git-history-aware symbol tracking
├── code_memory.py       # Code-anchored memories
├── harness.py           # Pre/post validation hooks
├── hybrid_search.py     # BM25 + TF-IDF fusion
├── auto_fix.py          # Graph-based fix suggestions
├── exceptions.py        # Typed error hierarchy
├── hooks.py             # Bash wrapper
└── parsers/             # Tree-sitter (11 langs) + custom (3 langs)
```

## Key design decisions

- **Ranked selection over blast-radius** — every file scored 0-1 via BM25 + graph-distance decay
- **Hard token budget** — greedy selection within limit (hot/warm/cold tiers)
- **Diff-aware trimming** — only changed regions + context lines, not full files
- **Entropy-based dedup** — remove near-identical files for context diversity
- **Local-first** — zero telemetry, no network calls, SQLite persistence
- **Cache-aware** — prompt-cache breakpoints for Anthropic/OpenAI

## Related

- [Caveman](https://github.com/JuliusBrussee/caveman) — output token compression by talking caveman (complementary: graphsift optimizes INPUT context, caveman compresses OUTPUT)
- [Caveman Code](https://github.com/slmingol/caveman) — full-agent output compression
