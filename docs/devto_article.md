---
title: "graphsift: How We Got 80-150x Token Reduction for LLM Code Review"
published: true
description: "Binary blast-radius context selection for LLMs produces F1=0.54. Here's how ranked AST graph scoring with decorator edges, dynamic import detection, and token-budget-aware selection gets to F1≈0.85 and 80-150x reduction."
tags: python, llm, ai, opensource
cover_image: https://github.com/maheshmakvana/graphsift/raw/master/docs/cover.png
canonical_url: https://dev.to/graphsift/graphsift-80-150x-token-reduction-llm-code-review
---

# graphsift: How We Got 80–150× Token Reduction for LLM Code Review

When you wire Claude or GPT-4 into a code review workflow, you immediately hit the same wall: your repo has hundreds of files, the model has a finite context window, and sending everything is both expensive and noisy.

The standard approach — "blast radius" selection — takes the changed file, traverses the import graph, and includes every reachable file. It works at a surface level but produces a lot of irrelevant context.

We measured it: **F1 = 0.54**. Nearly half the files sent are irrelevant. Genuinely important files (connected via decorators or runtime imports) get missed.

[graphsift](https://github.com/maheshmakvana/graphsift) is an open-source Python library that fixes this with ranked relevance scoring, budget-aware selection, and edges that actually capture real dependencies.

---

## The problem with binary blast-radius

Here's what binary selection looks like:

```
Changed: src/auth/manager.py

Import graph traversal:
  manager.py → imports → models.py      ✓ included
  manager.py → imports → utils.py       ✓ included
  manager.py → imports → constants.py   ✓ included
  constants.py → imports → base.py      ✓ included
  base.py → imports → typing_compat.py  ✓ included  ← 3 hops, barely relevant
  ...
  @require_auth (decorator) → middleware.py  ✗ MISSED  ← actually critical
  importlib.import_module("plugins.cache")  ✗ MISSED   ← actually critical
```

Two real problems:

1. **False positives** — files at 3–4 import hops away are almost never relevant to a specific change, but binary selection includes them anyway.
2. **False negatives** — decorator relationships and dynamic imports are invisible to naive import-graph traversal. These are often the most important dependencies.

---

## How graphsift works

### Step 1: Build a richer dependency graph

graphsift parses source files into an AST dependency graph with **7 edge types**:

| Edge | Example |
|------|---------|
| CALLS | `auth_manager()` calls `hash_password()` |
| IMPORTS | `from models import User` |
| INHERITS | `class AdminUser(User)` |
| **DECORATES** | `@require_auth` applied to `create_post()` |
| REFERENCES | `config.MAX_RETRIES` used in `retry_handler` |
| TEST_COVERS | `test_auth.py` tests `auth/manager.py` |
| **DYNAMIC_IMPORT** | `importlib.import_module("plugins.cache")` |

The DECORATES and DYNAMIC_IMPORT edges are what most tools miss. In real codebases, decorators are a primary mechanism for cross-cutting concerns — auth, caching, rate limiting, logging. If you're reviewing an auth change, you need to see every function that uses `@require_auth`, not just files that import the auth module.

It supports **14 languages**: Python (native `ast`), JavaScript, TypeScript, Go (including receiver methods like `func (r *Router) Handle()`), Rust, Java, C++, C, Ruby, PHP, Bash/Shell, Terraform/HCL, and Helm Charts.

### Step 2: Rank every file 0–1

For each changed file, graphsift runs BFS from that node. Every reachable file gets a score using exponential distance decay:

```
graph_score = (1 - 0.7) ^ distance
```

Distance 1 → score 0.30, distance 2 → score 0.09, distance 3 → score 0.027. Files far from the change naturally get low scores.

This is fused with a BM25 keyword score against the query and commit message:

```
final_score = 0.3 × bm25_score + 0.7 × graph_score
```

For multi-file diffs, scores from each changed file are unioned (max across seeds). This gives a proper union blast radius across all changed files simultaneously.

### Step 3: Select within a hard token budget

Files are sorted by score descending. graphsift greedily picks files until the token budget is exhausted:

```
Budget: 50,000 tokens

1. auth/manager.py     score=1.000  → FULL        (2,100 tok)
2. middleware.py       score=0.841  → FULL        (3,400 tok)
3. tests/test_auth.py  score=0.714  → FULL        (1,200 tok)
4. models/user.py      score=0.490  → SIGNATURES  (  180 tok)  ← stubs only
5. utils/base.py       score=0.312  → COMPRESSED  (   90 tok)  ← pruned
...
Total: 12,400 tokens vs 180,000 raw = 93% reduction
```

Output mode is chosen per file:
- **FULL** — raw source, for high-score files
- **SIGNATURES** — function/class stubs only (10–20% of full), for low-score files
- **COMPRESSED** — tokenpruner compression, for any file when installed
- **SMART** — auto: FULL above score threshold, SIGNATURES below (default)

---

## Results on a real codebase

We tested on a 143-file FastAPI application. The change: `src/auth/manager.py` (50 lines modified, adding JWT refresh token support).

| Approach | Files selected | Tokens | F1 |
|---|---|---|---|
| Raw source (send everything) | 143/143 | ~180,000 | — |
| Binary blast-radius | 8–12/143 | 6,000–8,000 | 0.54 |
| **graphsift** | **3–5/143** | **800–1,200** | **~0.85** |

The F1 improvement comes primarily from:
- Decorator edges catching `@require_auth` usages across middleware and views
- Score decay eliminating irrelevant transitive imports (typing helpers, constants files)
- SIGNATURES mode letting low-relevance files contribute stubs without burning budget

---

## Using it

```bash
pip install graphsift

# Optional: 3-5x more compression on low-score files
pip install "graphsift[tokenpruner]"
```

Basic usage:

```python
from graphsift import ContextBuilder, ContextConfig, DiffSpec

builder = ContextBuilder(ContextConfig(token_budget=50_000))
builder.index_files(source_map)  # {path: source_text}

result = builder.build(
    DiffSpec(
        changed_files=["src/auth/manager.py", "src/auth/middleware.py"],
        query="Review JWT refresh token implementation",
        commit_message="feat: add JWT refresh token support",
    ),
    source_map,
)

print(result)
# ContextResult(selected=5/143, tokens=4,200, saved=98%)

# Ready to inject into your LLM call
llm_context = result.rendered_context
```

Drop-in Claude adapter:

```python
import anthropic
from graphsift.adapters.claude import ClaudeCodeReviewAdapter

client = anthropic.Anthropic()
adapter = ClaudeCodeReviewAdapter(client, builder)

response, meta = adapter.review(
    changed_files=["src/auth/manager.py"],
    source_map=source_map,
    model="claude-opus-4-6",
    query="Any security issues in this JWT implementation?",
)
print(f"Tokens saved: {meta['reduction_ratio']:.0%}")
# Tokens saved: 97%
```

MCP server (Claude Code integration):

```bash
graphsift install
# Writes .mcp.json and hooks automatically — no manual config
```

---

## Advanced features

graphsift ships with 10 production-grade utilities in `advanced.py`:

**Smart cache** — LRU + TTL, `.memoize()` decorator, thread-safe:
```python
cache = GraphCache(maxsize=64, ttl=300)

@cache.memoize
def get_context(diff_key: str):
    return builder.build(diff, source_map)
```

**Analysis pipeline** — chainable steps, `.arun()` async, per-step audit:
```python
pipeline = (
    AnalysisPipeline(builder)
    .add_step("filter_generated", remove_generated_files)
    .with_retry(n=2, backoff=0.3)
)
result, audit = pipeline.run(diff_spec, source_map)
```

**Rate limiter** — token bucket, per-key, sync + async context manager:
```python
limiter = RateLimiter(rate=5, capacity=5, key="claude")
with limiter:
    response = adapter.review(...)
```

**Streaming** — highest-score files first, cancellation-safe:
```python
async for batch in async_stream_context(builder, diff_spec, source_map):
    process(batch)   # start LLM on high-score files immediately
```

**Circuit breaker** — CLOSED → OPEN → HALF_OPEN, configurable threshold:
```python
cb = CircuitBreaker(failure_threshold=3, reset_timeout=30)

@cb.protect
def call_llm(prompt): ...
```

**Diff engine** — compare two context runs, `.summary()` + `.to_json()`:
```python
diff = ContextDiff(result_before, result_after)
print(diff.summary())
# Files: 8 → 11 (+3), Tokens: 9,200 → 14,100 (+4,900)
```

---

## Architecture

graphsift is a pure library — no framework, no global state, no `main()`. One mandatory dependency: `pydantic>=2.0`.

It follows strict hexagonal architecture:
- `core.py` — pure domain logic, zero I/O
- `adapters/` — SQLite persistence, Claude API, filesystem helpers
- `models.py` — Pydantic v2 `BaseModel` with `frozen=True` for all structured I/O
- `exceptions.py` — typed exception hierarchy (`graphsiftError` → `ValidationError`, `ParseError`, `AdapterError`, ...)

All external dependencies are behind `typing.Protocol` — callers inject, not inherit. All shared state is behind `threading.RLock`. Every blocking public method has an `async def a<method>()` twin.

---

## Status

- Version: 1.4.1
- Python: 3.9–3.12
- Tests: 109 passing (pytest, pytest-asyncio)
- License: MIT

GitHub: [maheshmakvana/graphsift](https://github.com/maheshmakvana/graphsift)
PyPI: [pypi.org/project/graphsift](https://pypi.org/project/graphsift/)

Issues, PRs, and feedback welcome.
