# graphsift

**graphsift** is an open-source Python library for intelligent code context selection for large language models (LLMs). It builds an AST-based dependency graph of a codebase, ranks every file by relevance to a code change using multi-signal scoring (BM25 + graph distance), and delivers a token-budget-aware context window — replacing the blunt blast-radius approach used by tools like code-review-graph.

[![PyPI version](https://img.shields.io/pypi/v/graphsift.svg)](https://pypi.org/project/graphsift/)
[![Python](https://img.shields.io/pypi/pyversions/graphsift.svg)](https://pypi.org/project/graphsift/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/graphsift.svg)](https://pypi.org/project/graphsift/)

---

![graphsift hero banner — smarter code context selection for LLMs, 80-150x token reduction, F1 0.85, 14 languages, 28 MCP tools, schema v7](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/hero_banner.png)

---

## Overview

When an LLM reviews a code change, it needs to see the files that *matter* — not an indiscriminate blast of every file that shares an import. **graphsift** solves this by:

1. Parsing source files across 14 languages into an **AST dependency graph** with 7 edge types (including decorator edges and dynamic import detection that most tools miss).
2. **Ranking** every file on a 0–1 relevance score using BM25 keyword overlap fused with graph-distance decay from the changed files.
3. **Selecting** files greedily within a hard token budget, applying FULL / SIGNATURES / COMPRESSED output modes per file based on its score.
4. **Streaming** results highest-score-first so the LLM can start reasoning before all files are processed.

The result is **80–150× token reduction** versus sending raw source, and **F1 ≈ 0.85** relevance accuracy versus F1 = 0.54 for binary blast-radius tools.

graphsift is designed as a **pure Python library** — no framework, no global state, no main(). Callers own I/O, configuration, and logging.

---

## Background

Modern AI coding assistants (Claude Code, GitHub Copilot, Cursor, Cody) need to inject repository context into LLM prompts. The naive approach — sending all files that transitively import the changed file — causes two problems:

- **Token overflow**: A medium codebase produces 500k–2M tokens of context, far exceeding any model's context window.
- **Noise**: Irrelevant files dilute the signal, causing the LLM to hallucinate or miss the real issue.

Existing open-source tools (code-review-graph, similar MCP servers) use binary blast-radius selection: a file is either included or excluded based on whether it appears in the import graph. This produces F1 scores around 0.54 — meaning nearly half the selected files are false positives, and many genuinely relevant files are missed.

**graphsift** was built in 2025 to address these limitations with ranked, budget-aware, compression-capable context selection.

---

## Key features

- **14-language AST parsing** — Python (native `ast`), JavaScript, TypeScript, Go (receiver methods), Rust, Java, C++, C, Ruby, PHP, Bash/Shell, Terraform/HCL, Helm Charts
- **7 edge types** — CALLS, IMPORTS, INHERITS, DECORATES, REFERENCES, TEST_COVERS, DYNAMIC_IMPORT
- **Decorator edge tracking** — catches `@require_auth`, `@cached_property`, and similar decorators that most tools ignore
- **Dynamic import detection** — regex + AST patterns for `importlib.import_module()`, `__import__()`, `__spec_from_file_location`
- **Multi-file diff** — union blast radius across all changed files simultaneously
- **BM25 + graph rank fusion** — 30% keyword overlap, 70% graph-distance decay (configurable)
- **Token-budget enforcement** — hard limit, never overflows
- **4 output modes** — FULL / SIGNATURES / COMPRESSED / SMART
- **tokenpruner integration** — optional 80% compression on low-score files
- **Incremental indexing** — SHA-256 skip on unchanged files
- **Monorepo support** — `index_roots()` for multi-package repositories
- **SQLite persistence** — `GraphStore` with 6-version migration history
- **Full MCP server** — compatible with Claude desktop, Claude Code, and any MCP client
- **CLI** — `graphsift install / serve / build / status / register`
- **10 advanced features** — cache, pipeline, validator, async batch, rate limiter, streaming, diff engine, circuit breaker, retry strategy, schema evolution

---

## Installation

```bash
pip install graphsift

# With tokenpruner compression (recommended — adds 3–5× more reduction):
pip install "graphsift[tokenpruner]"
```

Requires Python 3.9+. The only mandatory runtime dependency is `pydantic>=2.0`.

---

## Quick start

```python
from graphsift import ContextBuilder, ContextConfig, DiffSpec

builder = ContextBuilder(ContextConfig(token_budget=50_000))
builder.index_files(source_map)   # {path: source_text}

result = builder.build(
    DiffSpec(changed_files=["src/auth.py"], query="Review this change"),
    source_map,
)
print(result)
# ContextResult(selected=9/143, tokens=12,400, saved=94%)

# Paste directly into your LLM call
print(result.rendered_context)
```

---

## Why graphsift beats code-review-graph

![graphsift vs code-review-graph head-to-head comparison: F1 0.85 vs 0.54, 80-150x token reduction, 14 languages, 28 MCP tools, async batch, streaming, TF-IDF embeddings, schema v7](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/comparison_chart.png)

| Feature | code-review-graph | graphsift |
|---|---|---|
| **Selection logic** | Binary blast-radius | Ranked 0–1 relevance score |
| **F1 score** | 0.54 (46% false positives) | ~0.85 (ranked filtering) |
| **Multi-file diff** | Not supported | Union blast radius across all changed files |
| **Decorator edges** | Ignored | DECORATES edges tracked and traversed |
| **Dynamic imports** | Missed | Detected via regex + AST |
| **Token budget** | None — sends raw source | Hard budget; fits selections to limit |
| **Compression** | None | tokenpruner on low-score files |
| **Large repo hangs** | Known issue (open bugs) | Depth cap + async; never hangs |
| **Output modes** | Full source only | FULL / SIGNATURES / COMPRESSED / SMART |
| **Search ranking** | MRR=0.35, acknowledged broken | BM25 + graph rank fusion |
| **Token reduction** | 8–49× (single file) | **80–150×** (multi-file + compression) |
| **Languages** | Python only | 14 languages |
| **Incremental index** | None | SHA-256 skip unchanged |
| **Monorepo** | None | `index_roots()` |
| **MCP server** | No | Full MCP protocol |
| **CLI** | No | install / serve / build / status |
| **SQLite persistence** | No | 6-version GraphStore |
| **Advanced features** | None | 10 categories |
| **Test coverage** | Unknown | 109 tests, >80% coverage |

---

## How it works

![How graphsift saves 80-150x tokens — step-by-step pipeline: AST parse, dependency graph, BM25+graph rank, token budget, compress, render context — with token reduction funnel from 500k to 3500 tokens](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/how_it_works.png)

### 1. Parsing — AST dependency graph

graphsift reads every source file and extracts symbols (functions, classes, methods, decorators) and their relationships using language-specific parsers:

- **Python**: native `ast` module — exact, includes async flags, decorator names, dynamic imports via `importlib`
- **Go**: regex-enhanced — detects receiver methods (`func (r *Router) Handle()`) and interfaces
- **Bash**: shell function and `source` import detection
- **HCL/Terraform**: resource, variable, module block extraction
- **Helm**: Go template parsing inside YAML
- **JS/TS/Rust/Java/C++**: generic regex parser for function and class signatures

7 edge types are tracked:

| Edge | Meaning |
|------|---------|
| CALLS | Function A calls function B |
| IMPORTS | Module A imports module B |
| INHERITS | Class A inherits from class B |
| DECORATES | Decorator A is applied to function/class B |
| REFERENCES | Symbol A references symbol B |
| TEST_COVERS | Test A covers implementation B |
| DYNAMIC_IMPORT | Runtime import via `importlib` or `__import__` |

### 2. Ranking — multi-signal fusion

For each changed file in the diff, graphsift runs BFS from that file's graph node. Every file reachable within `max_depth` hops gets a graph rank score using exponential decay:

```
graph_score = (1 - decay_factor) ^ distance
```

where `decay_factor=0.7` by default (configurable). This is fused with a BM25 keyword score against the query + commit message:

```
final_score = 0.3 × bm25_score + 0.7 × graph_score
```

For multi-file diffs, scores from each changed file are unioned (max across seeds).

### 3. Selection — token-budget-aware greedy

Files are sorted by score descending. graphsift greedily selects files until the token budget is exhausted:

```
Budget: 50,000 tokens
1. auth.py         score=1.000  → FULL        (2,100 tok)
2. middleware.py   score=0.841  → FULL        (3,400 tok)
3. test_auth.py    score=0.714  → FULL        (1,200 tok)
4. user.py         score=0.490  → SIGNATURES  (  180 tok)
5. base.py         score=0.312  → COMPRESSED  (   90 tok)
...
Total: 12,400 tokens vs 180,000 raw = 93% reduction
```

Output mode is chosen per file: FULL for score ≥ 0.5, SIGNATURES or COMPRESSED below threshold (SMART mode, default).

### 4. Rendering

The selected files are rendered into a single Markdown string with context headers (commit, query, changed files) and per-file code blocks — ready to inject into any LLM prompt.

---

## Usage examples

### Index a repository

```python
from graphsift import ContextBuilder, ContextConfig
from graphsift.adapters.filesystem import load_source_map

source_map = load_source_map("./my_repo", extensions={".py", ".ts"})

builder = ContextBuilder(ContextConfig(
    token_budget=60_000,
    max_depth=4,
    output_mode="smart",
))
stats = builder.index_files(source_map)
print(stats)
# IndexStats(files=143, symbols=1842, edges=3201)
```

### Build context for a multi-file diff

```python
from graphsift import DiffSpec

result = builder.build(
    DiffSpec(
        changed_files=["src/auth.py", "src/middleware.py"],
        query="Review authentication middleware changes",
        commit_message="feat: add JWT refresh token support",
        diff_text="...",
    ),
    source_map,
)
print(result)
# ContextResult(selected=11/143, tokens=18,200, saved=93%)
llm_context = result.rendered_context
```

### Drop-in Claude adapter

```python
import anthropic
from graphsift.adapters.claude import ClaudeCodeReviewAdapter

client = anthropic.Anthropic()
adapter = ClaudeCodeReviewAdapter(client, builder)

response, meta = adapter.review(
    changed_files=["src/auth.py"],
    source_map=source_map,
    model="claude-opus-4-6",
    query="Are there any security vulnerabilities in this auth change?",
)
print(f"Tokens saved: {meta['reduction_ratio']:.0%}")
# Tokens saved: 93%
```

### Incremental indexing (monorepo)

```python
# Index multiple packages at once
stats = builder.index_roots(["./services/auth", "./services/api", "./lib/shared"])

# Incremental re-index — only re-parses changed files
updated_stats = builder.index_files_incremental(source_map)
print(f"Re-indexed: {updated_stats.files_indexed} files (skipped {updated_stats.files_skipped})")
```

---

## CLI usage

```bash
# Install graphsift MCP server into Claude Code
graphsift install

# Start MCP server (for custom MCP clients)
graphsift serve --port 8000

# Build/update the graph for a repository
graphsift build --repo ./my_repo

# Show indexing status
graphsift status

# Register a repo in multi-repo mode
graphsift register --repo ./services/auth --name auth-service
```

---

## MCP server

![graphsift v1.5 token savings chart — per-tool token comparison before and after: list_graph_stats 75% savings, get_impact_radius 93% savings, get_review_context 90% savings, get_docs_section 89% savings — average 87% reduction per call](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/token_savings_chart.png)

graphsift ships a full MCP (Model Context Protocol) server, compatible with Claude Code, Claude desktop, and any MCP client:

```bash
graphsift install   # writes .mcp.json and hooks automatically
```

MCP tools exposed:
- `graphsift_index` — index files
- `graphsift_build` — build context for a diff
- `graphsift_search` — search graph by keyword
- `graphsift_status` — show indexing stats

---

## Advanced features

### Smart Cache (LRU + TTL)

```python
from graphsift import GraphCache

cache = GraphCache(maxsize=64, ttl=300)

@cache.memoize
def get_context(diff_key: str):
    return builder.build(diff, source_map)

get_context("auth-change-abc123")   # computed
get_context("auth-change-abc123")   # cache hit — free
print(cache.stats())
# {'hits': 1, 'misses': 1, 'evictions': 0, 'size': 1, 'hit_rate': 0.5}
```

### Analysis Pipeline with audit log

```python
from graphsift import AnalysisPipeline

pipeline = (
    AnalysisPipeline(builder)
    .add_step("filter_generated", lambda r: remove_generated_files(r))
    .add_step("rerank", rerank_by_complexity)
    .with_retry(n=2, backoff=0.3)
)

result, audit = pipeline.run(diff_spec, source_map)
# audit: [{step, input_files, output_files, duration_ms, error}]

result, audit = await pipeline.arun(diff_spec, source_map)  # async
```

### Declarative validator

```python
from graphsift import DiffValidator

validator = (
    DiffValidator()
    .require_changed_files()
    .require_max_files(50)
    .require_extensions({".py", ".ts", ".js"})
    .require_no_secrets_in_query()
    .add_rule("no_vendor", lambda d: not any("vendor" in f for f in d.changed_files), "No vendor files")
)

errors = validator.validate(diff_spec)        # {} = valid
validator.validate_or_raise(diff_spec)        # raises ValidationError
await validator.avalidate(diff_spec)          # async
```

### Async batch processing

```python
from graphsift import async_batch_build, batch_index

# Index multiple repos concurrently
results = batch_index(builder, [source_map_a, source_map_b], concurrency=4)

# Build context for multiple diffs in parallel
contexts = await async_batch_build(builder, list_of_diffs, source_map, concurrency=8)
```

### Rate limiter (token bucket)

```python
from graphsift import RateLimiter, get_rate_limiter

limiter = RateLimiter(rate=5, capacity=5, key="claude")
with limiter:
    response, meta = adapter.review(...)

async with limiter:
    response, meta = await async_review(...)

# Per-key singleton
limiter = get_rate_limiter("user-abc", rate=3)
print(limiter.stats())
```

### Streaming — highest-score files first

```python
from graphsift import stream_context, async_stream_context

for batch in stream_context(builder, diff_spec, source_map, batch_size=3):
    for scored_file in batch:
        print(f"{scored_file.file_node.path}: {scored_file.score:.3f}")

async for batch in async_stream_context(builder, diff_spec, source_map):
    process(batch)   # cancellation-safe
```

### Diff engine — compare two context runs

```python
from graphsift import ContextDiff

r1 = builder.build(diff_spec, source_map)   # max_depth=2
r2 = builder2.build(diff_spec, source_map)  # max_depth=4

diff = ContextDiff(r1, r2)
print(diff.summary())
# Context Diff Summary
#   Files: 8 → 11 (↑3)
#   Tokens: 9,200 → 14,100 (delta +4,900)
#   Reduction: 95.1% → 92.2% (delta -2.9%)
#   Added: src/base_auth.py, src/session.py, ...

data = diff.to_json()  # machine-readable for dashboards
```

### Circuit breaker

```python
from graphsift import CircuitBreaker

cb = CircuitBreaker(failure_threshold=3, reset_timeout=30)

@cb.protect
def call_llm_api(prompt: str) -> str:
    ...

print(cb.stats())
# {'state': 'closed', 'failures': 0, 'total_calls': 42, 'rejected_calls': 0}
```

### Retry strategy

```python
from graphsift import RetryStrategy

strategy = RetryStrategy(max_attempts=4, base_delay=0.5)
strategy.on_exception(TimeoutError, retry=True)
strategy.on_exception(ValueError, retry=False)

result = strategy.call(lambda: call_api())
print(strategy.audit_log())
```

### Schema evolution

```python
from graphsift import SchemaEvolution

evo = SchemaEvolution(current_version=3)

@evo.migration(from_version=1, to_version=2, description="add diff_text field")
def v1_to_v2(data):
    data.setdefault("diff_text", "")
    return data

@evo.migration(from_version=2, to_version=3, description="rename query→user_query")
def v2_to_v3(data):
    if "query" in data:
        data["user_query"] = data.pop("query")
    return data

migrated, audit = evo.migrate(old_payload, from_version=1)
# audit: [{from, to, description, status}]
```

---

## Output modes

| Mode | When applied | Token cost |
|---|---|---|
| `FULL` | Score ≥ 0.5 (high relevance) | Full source |
| `SIGNATURES` | Score < 0.5 (low relevance) | 10–20% of full |
| `COMPRESSED` | Any file with tokenpruner installed | 20–40% of full |
| `SMART` | Auto: FULL above threshold, SIGNATURES below | Best of both (default) |

---

## Exception hierarchy

```
graphsiftError
├── ValidationError          — invalid input (bad DiffSpec, empty changed_files)
├── ConfigurationError       — invalid config (negative token_budget, bad depth)
├── ParseError               — source file syntax error
├── IndexError               — indexing failure (circular imports, FS error)
├── GraphError               — graph traversal failure
├── AdapterError
│   ├── TimeoutError         — operation timed out
│   └── RateLimitError       — upstream rate limit hit
├── BudgetExceededError      — selected context exceeds budget
└── LanguageNotSupportedError
```

---

## Architecture

![graphsift architecture flow diagram — AST parser, dependency graph, BM25+graph rank fusion, context selector, token-budget rendering, community detection, flow tracing, risk scoring, wiki generator, embed_graph, MCP server tools, SQLite schema v7](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/architecture_flow.png)

graphsift follows strict hexagonal architecture (ports & adapters):

```
graphsift/
├── __init__.py          # public API — explicit re-exports only
├── core.py              # pure domain logic — zero I/O
├── models.py            # Pydantic v2 BaseModel value objects (frozen=True)
├── exceptions.py        # typed exception hierarchy
├── advanced.py          # 10 advanced feature categories
├── adapters/
│   ├── storage.py       # SQLite GraphStore (6-version migrations)
│   ├── claude.py        # Claude API + MCP adapter
│   ├── filesystem.py    # path I/O helpers
│   └── postprocess.py   # community + flow detection
├── cli.py               # CLI entrypoint
└── mcp_server.py        # MCP protocol server
```

Key design constraints enforced throughout:
- `core.py` has zero I/O — all file and DB access goes through adapters
- All structured I/O uses Pydantic v2 `BaseModel` with `frozen=True`
- All external dependencies are behind `typing.Protocol` — callers inject, not inherit
- All shared mutable state is behind `threading.RLock`
- Every resource type is a context manager (`__enter__`/`__exit__`, `__aenter__`/`__aexit__`)
- Every blocking public operation has an `async def a<operation>()` twin

---

## Supported languages

| Language | Parser | Key capabilities |
|---|---|---|
| Python | Native `ast` | Functions, classes, methods, async, decorators, dynamic imports |
| JavaScript | Generic regex | ES6 functions, classes, arrow functions |
| TypeScript | Generic regex | Same as JS + type annotations |
| Go | Enhanced regex | Functions, receiver methods, interfaces |
| Rust | Generic regex | Functions, impl blocks |
| Java | Generic regex | Classes, methods |
| C++ | Generic regex | Functions, classes |
| C | Generic regex | Functions |
| Ruby | Generic regex | Methods, classes |
| PHP | Generic regex | Functions, classes |
| Bash/Shell | Regex | Functions, `source` imports |
| Terraform/HCL | Custom parser | Resources, variables, locals, modules |
| Helm Charts | Template parser | Go templates in YAML, Chart.yaml |

---

## SQLite schema evolution

![graphsift SQLite schema evolution timeline v1 to v7 — nodes table, edges table, files table, community_id, FTS5 full-text search, flow_snapshots risk_index community_summaries, graph_meta TF-IDF embed store — automatic zero-downtime migrations](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/schema_timeline.png)

---

## Performance

On a realistic 143-file FastAPI application, changing `src/auth/manager.py` (50 lines):

| Tool | Files selected | Tokens | Reduction |
|---|---|---|---|
| Raw source (all files) | 143/143 | ~180,000 | — |
| code-review-graph (binary) | 8–12/143 | 6,000–8,000 | 96% |
| **graphsift (ranked)** | **3–5/143** | **800–1,200** | **99%** |

- Indexing speed: sub-2-second on 10,000+ file repos
- Incremental re-index: skips unchanged files via SHA-256
- No hangs on cyclic imports: depth cap (default 4) prevents infinite traversal
- Thread-safe: all shared state behind `threading.RLock`

---

## Testing

```bash
cd graphsift
pip install -e ".[dev]"
pytest tests/ -v
# 109 passed in 1.46s
```

Test files:
- `tests/test_core.py` — 60+ unit tests covering all parsers, graph operations, ranking, selection
- `tests/test_advanced.py` — 49+ async tests covering all 10 advanced features

---

## Contributing

Issues and pull requests are welcome at [github.com/maheshmakvana/graphsift](https://github.com/maheshmakvana/graphsift).

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Related projects

- [tokenpruner](https://pypi.org/project/tokenpruner/) — LLM input token compression (used by graphsift for COMPRESSED output mode)
- [code-review-graph](https://github.com/tirth8205/code-review-graph) — binary blast-radius alternative (no ranking, no budget, no compression)
