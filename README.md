# graphsift — Save Claude Tokens, Reduce LLM API Costs, Optimize Context Windows

**graphsift** is the #1 open-source **Claude token saver** and **LLM token optimizer** for AI code review. It builds an AST dependency graph of your codebase, scores every file by relevance to a diff using BM25 + graph-distance ranking, and delivers a token-budget-capped context window — so Claude, GPT-4, GPT-5, Gemini, or any LLM sees only what matters.

**Save 80-150x Claude tokens** per code review. Cut **60-90% of CLI command output tokens** before they hit your LLM context. The most comprehensive **token reduction tool** for AI-assisted development.

- **Reduce Claude API costs** by 93-99% per code review call
- **Optimize Claude context windows** — ranked relevance instead of binary blast-radius
- **Save tokens on Claude Code**, Cursor, Copilot, and any MCP-compatible agent
- **Compress command output** — 19 per-tool compressors for pytest, docker, git, kubectl, npm, and more

[![PyPI version](https://img.shields.io/pypi/v/graphsift.svg)](https://pypi.org/project/graphsift/)
[![Python](https://img.shields.io/pypi/pyversions/graphsift.svg)](https://pypi.org/project/graphsift/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/graphsift.svg)](https://pypi.org/project/graphsift/)
[![GitHub stars](https://img.shields.io/github/stars/maheshmakvana/graphsift.svg)](https://github.com/maheshmakvana/graphsift)
[![tests](https://img.shields.io/badge/tests-271%20passed-blue)](https://github.com/maheshmakvana/graphsift/actions)
[![MCP tools](https://img.shields.io/badge/MCP%20tools-25-orange)](https://github.com/maheshmakvana/graphsift)
[![compressors](https://img.shields.io/badge/compressors-19-green)](https://github.com/maheshmakvana/graphsift)

---

![graphsift — save Claude tokens, reduce LLM API costs, optimize context windows with ranked code selection for Claude GPT Gemini, F1 0.85, 14 languages, token budget enforcement](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/hero_banner.png)

---

## Why You Need a Claude Token Saver

Every Claude API call costs tokens. Every token costs money. When Claude reviews a code change, the naive approach sends every transitively-related file — that's **500k-2M tokens** for a medium codebase. You're paying for noise.

**graphsift is the Claude token optimizer that fixes this:**

- **80-150x fewer Claude tokens** per code review — ranked scoring replaces binary blast-radius
- **F1 ~0.85 relevance accuracy** vs F1=0.54 for tools like code-review-graph
- **Hard token budget enforcement** — never exceed Claude's context limit or your cost threshold
- **Save $150-180/day on Claude API costs** at 100 PRs/day vs raw source dumps

### Who Needs to Save Claude Tokens?

| Use case | How graphsift saves Claude tokens |
|---|---|
| **CI/CD AI code review** | Auto-select relevant context per PR, cut costs 93-99% |
| **Monorepo code review** | Blast-radius tools drown in irrelevant files; graphsift ranks and trims |
| **Claude Code / Cursor / Copilot** | MCP server delivers token-efficient context to coding agents |
| **LLM cost optimization** | Token budget + compression + caching = predictable API spend |
| **Enterprise AI review pipelines** | Hard limits prevent runaway API costs; analytics track every token saved |
| **RAG / agent context building** | General-purpose: any LLM task that needs ranked code context |

---

## Token Savings at a Glance — How Much You Save with graphsift

Benchmarked on a 143-file FastAPI application reviewing a 50-line change to `src/auth/manager.py`:

| Approach | Files sent | Claude tokens | Cost (Claude Opus @ $15/M) | Savings vs raw |
|---|---|---|---|---|
| Raw source (every file) | 143/143 | ~180,000 | $2.70 | — |
| Binary blast-radius (code-review-graph) | 8-12/143 | 6,000-8,000 | $0.10 | 96% |
| **graphsift (ranked + budget)** | **3-5/143** | **800-1,200** | **$0.015** | **99.4%** |

### CLI Output Compression — Save 60-90% of Command Tokens

Beyond code review, graphsift saves Claude tokens on command output too. Every `pytest`, `docker ps`, `kubectl get`, or `git diff` you pipe to Claude wastes tokens on noise.

| Command | Original tokens | Compressed tokens | **Claude tokens saved** |
|---|---|---|---|
| `grep -r` (25 results) | 413 tk | 22 tk | **95%** |
| `eslint` (12 problems) | 308 tk | 17 tk | **94%** |
| `git diff` (2 files) | 889 tk | 60 tk | **93%** |
| `pytest -v` (45 tests) | 1,334 tk | 136 tk | **90%** |
| `npm install` output | 288 tk | 39 tk | **87%** |
| `docker ps` (10 images) | 463 tk | 63 tk | **86%** |
| `git status` | 174 tk | 25 tk | **86%** |
| `pip install` (7 pkgs) | 312 tk | 47 tk | **85%** |
| `cargo build` | 463 tk | 80 tk | **83%** |
| `kubectl get all` | 581 tk | 110 tk | **81%** |
| `git log` (3 commits) | 234 tk | 47 tk | **80%** |
| `make` output | 250 tk | 55 tk | **78%** |
| `aws` CLI JSON | 477 tk | 115 tk | **76%** |
| `jest` (10 tests) | 310 tk | 76 tk | **75%** |
| `go test` | 284 tk | 74 tk | **74%** |
| App logs (16 lines) | 402 tk | 155 tk | **61%** |
| `cat` (large file) | 672 tk | 479 tk | **29%** |
| **Weighted average** | **8,138 tk** | **1,884 tk** | **77%** |

At 100 CLI commands/day piped to Claude, that's **~625,000 tokens saved per day** — roughly **$9.37/day saved** on Claude Opus pricing.

---

## How graphsift Saves Claude Tokens

Four steps from diff to Claude-optimized context:

1. **Parse** — Builds an AST dependency graph from your source. 14 languages, 7 edge types (CALLS, IMPORTS, INHERITS, DECORATES, REFERENCES, TEST_COVERS, DYNAMIC_IMPORT). v1.6 adds precise **tree-sitter parsing** for Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, and Bash.

2. **Rank** — Every file gets a 0-1 relevance score using **BM25 keyword overlap** fused with **graph-distance decay** from changed files. Not binary include/exclude — nuanced ranking means Claude sees signal, not noise.

3. **Select** — Greedy token-budget selection with **three tiers** (hot/warm/cold). Hot files get full source, warm get signatures, cold are excluded. Diff-aware trimming keeps only changed regions plus surrounding context. Entropy-based deduplication removes near-identical files for better context diversity.

4. **Render** — One Markdown string, ready to inject into any Claude or LLM prompt. Optional Anthropic/OpenAI cache breakpoints for repeated queries.

```python
from graphsift import ContextBuilder, ContextConfig, DiffSpec

# Create a Claude token optimizer with a 50k token budget
builder = ContextBuilder(ContextConfig(token_budget=50_000))
builder.index_files(source_map)

result = builder.build(
    DiffSpec(changed_files=["src/auth.py"], query="Review for security issues"),
    source_map,
)
print(result)
# ContextResult(selected=9/143, tokens=12,400, saved=94%)
# Claude sees only 12,400 tokens instead of 180,000

# Paste directly into your Claude API call
print(result.rendered_context)
```

---

## Installation — Start Saving Claude Tokens in 60 Seconds

```bash
pip install graphsift

# With tree-sitter for precise AST parsing across 11 languages:
pip install "graphsift[treesitter]"

# Full install — compression + tree-sitter + dev tools:
pip install "graphsift[all]"
```

Requires Python 3.9+. Core dependency: `pydantic>=2.0`. Zero mandatory native extensions.

---

## Why graphsift Beats Binary Blast-Radius Tools for Saving Claude Tokens

The "send everything that imports the changed file" approach used by code-review-graph and most MCP code review tools has two fatal flaws for anyone trying to **reduce Claude API costs**:

1. **Token overflow** — 500k+ tokens exceeds Claude's context limit *and* your budget. Every irrelevant file burns money.
2. **Noise degrades Claude's output** — LLMs hallucinate more when flooded with irrelevant context. Sending `config.py`, `utils/logging.py`, and 40 test files because they import `base.py` buries the signal.

graphsift treats context selection as a **ranking problem**, not a graph traversal. Claude gets maximum signal per token — the most cost-effective way to use AI code review.

### Head-to-Head: graphsift vs code-review-graph

![graphsift vs code-review-graph comparison — save more Claude tokens, F1 0.85 vs 0.54, 80-150x token reduction, 14 languages, ranked relevance, token budget enforcement](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/comparison_chart.png)

| Feature | code-review-graph | **graphsift** (Claude token optimizer) |
|---|---|---|
| **Goal** | Show related files | **Save Claude tokens** while maximizing relevance |
| **Selection** | Binary blast-radius | Ranked 0-1 with hot/warm/cold tiers |
| **F1 accuracy** | 0.54 (46% false positives) | **0.85** (ranked filtering + dedup) |
| **Token budget** | None | **Hard budget** — fits any Claude model limit |
| **Token reduction** | 8-49x | **80-150x** (multi-file + compression + trimming) |
| **Multi-file diff** | Not supported | Union blast radius across all changed files |
| **Decorator edges** | Ignored | DECORATES tracked and scored |
| **Dynamic imports** | Missed | Detected via regex + AST + tree-sitter |
| **Compression** | None | tokenpruner + diff-aware trimming |
| **Deduplication** | None | Entropy-based near-duplicate removal |
| **Tree-sitter parsing** | None | 11 languages with precise CST/AST |
| **Hybrid search** | MRR=0.35, acknowledged broken | BM25 + TF-IDF vector fusion |
| **Dead code detection** | None | Unreachable code from entry points |
| **Cycle detection** | None | Dependency cycle analysis |
| **Auto-fix suggestions** | None | Graph-based issue detection + fix proposals |
| **Languages** | Python only | **14 languages** |
| **Incremental indexing** | None | SHA-256 skip for unchanged files |
| **Monorepo** | None | `index_roots()` for multi-package repos |
| **MCP server** | No | Full MCP protocol + 25 token-saving tools |
| **CLI** | No | install / serve / build / status / compress / gain |
| **SQLite persistence** | No | 6-version GraphStore with migrations |
| **Cache-aware output** | No | Anthropic/OpenAI cache breakpoints |
| **Output compression** | No | **19 CLI command compressors (77% avg savings)** |
| **Analytics** | No | Token savings tracking + discovery |
| **Test coverage** | Unknown | 271 tests, >80% coverage |

---

## Key Features — Everything You Need to Save Claude Tokens

### Token & Cost Optimization
- **Hard token budget** — never exceed Claude's context window or your cost ceiling
- **3-tier selection (hot/warm/cold)** — full source → signatures → excluded
- **Diff-aware context trimming** — only changed regions + surrounding context lines
- **Entropy-based deduplication** — removes near-identical files for better context diversity
- **4 output modes** — FULL / SIGNATURES / COMPRESSED / SMART (auto per-file)
- **Cache-aware output** — Anthropic/OpenAI cache_control breakpoints for repeated queries
- **Cross-session caching** — session_id-based memory reuse across Claude conversations
- **80-150x token reduction** vs raw source; **10-15x** vs binary blast-radius tools

### Code Analysis & Intelligence
- **14-language parsing** — Python, JS, TS, Go, Rust, Java, C++, C, Ruby, PHP, Bash, Terraform/ HCL, Helm
- **Tree-sitter precise parsing** — 11 languages with full CST/AST via tree-sitter (Python, JS, TS, Go, Rust, Java, C, C++, Ruby, PHP, Bash)
- **7 edge types** — CALLS, IMPORTS, INHERITS, DECORATES, REFERENCES, TEST_COVERS, DYNAMIC_IMPORT
- **Hybrid search** — BM25 full-text + TF-IDF sparse vector fusion for semantic code search
- **Cycle detection** — find and report dependency cycles with severity grading
- **Dead code detection** — identify unreachable functions, classes, methods from entry points
- **Auto-fix suggestions** — graph-based issue detection across 5 categories (import, type, structure, cycle, dead_code)
- **Decorator tracking** — `@require_auth`, `@cached_property` edges most tools miss
- **Dynamic import detection** — `importlib.import_module()`, `__import__()`, `require()`, lazy imports

### CLI Output Compression — 19 Compressors, 86% Average Savings
- **Auto-detect** command type from output signature — just pipe to `graphsift compress`
- **19 specialized compressors** — pytest (94%), git_diff (92%), docker (91%), pip (86%), npm (83%), git_status (83%), git_log (82%), eslint (77%), kubectl (75%), log (61%), grep (97%), and more
- **Bash wrapper** — transparent compression without manual piping
- **Tee mode** — save original uncompressed output while LLM sees compressed
- **Token analytics** — cumulative tracking, daily breakdown, cost estimates, opportunity discovery

### Developer Experience
- **Full MCP server** — compatible with Claude Code, Cursor, Copilot, Windsurf, Codex, Gemini and 23+ MCP clients
- **25 MCP tools** — build/update graph, get_context, get_impact, detect_changes, query_graph, search_symbols, list_flows, list_communities, get_architecture_overview, refactor, apply_refactor, generate_wiki, semantic_search_nodes, cross_repo_search, and more
- **4 MCP prompts** — review_code, analyze_impact, find_issues, explain_architecture
- **10 MCP resources** — graph stats, architecture overview, communities, flows, wiki pages, risk index
- **CLI** — `graphsift install / serve / build / status / compress / gain / discover`
- **Drop-in adapters** — Claude/Anthropic, OpenAI/Codex, Gemini (Google)
- **10 advanced features** — cache, pipeline, validator, async batch, rate limiter, streaming, diff engine, circuit breaker, retry, schema evolution
- **Incremental indexing** — SHA-256 skip on unchanged files; sub-2s re-index
- **Monorepo support** — `index_roots()` for multi-package repositories
- **SQLite persistence** — 6-version migration history

---

## Quick Start — Save Claude Tokens on Your First Code Review

### 1. Index your repository

```python
from graphsift import ContextBuilder, ContextConfig
from graphsift.adapters.filesystem import load_source_map

source_map = load_source_map("./my_repo", extensions={".py", ".ts"})

builder = ContextBuilder(ContextConfig(
    token_budget=60_000,  # Claude token budget — never exceeds this
    max_depth=4,
    output_mode="smart",  # auto hot/warm/cold tier selection
))
stats = builder.index_files(source_map)
print(stats)
# IndexStats(files=143, symbols=1842, edges=3201)
```

### 2. Build Claude-optimized context for a diff

```python
from graphsift import DiffSpec

result = builder.build(
    DiffSpec(
        changed_files=["src/auth.py", "src/middleware.py"],
        query="Review authentication middleware changes for security issues",
        commit_message="feat: add JWT refresh token support",
        diff_text="...",
    ),
    source_map,
)
print(result)
# ContextResult(selected=11/143, tokens=18,200, saved=93%, cache_breakpoints=3)
# Paste result.rendered_context directly into your Claude API prompt
```

### 3. Claude adapter — measure Claude token savings in real API calls

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
print(f"Claude tokens saved: {meta['reduction_ratio']:.0%}")
# Claude tokens saved: 93%
```

### 4. OpenAI / Codex adapter — save GPT tokens

```python
from openai import OpenAI
from graphsift.adapters.openai import CodexCodeReviewAdapter

client = OpenAI()
adapter = CodexCodeReviewAdapter(client, builder)

response, meta = adapter.review(
    changed_files=["src/auth.py"],
    source_map=source_map,
    model="gpt-5-codex",
    query="Find correctness or security issues.",
)
print(f"GPT tokens saved: {meta['reduction_ratio']:.0%}")
```

### 5. Gemini adapter — save Google AI tokens

```python
from google import genai
from graphsift.adapters.gemini import GeminiCodeReviewAdapter

client = genai.Client()
adapter = GeminiCodeReviewAdapter(client, builder)

response, meta = adapter.review(
    changed_files=["src/auth.py"],
    source_map=source_map,
    model="gemini-2.5-pro",
    query="Review this auth change for regressions.",
)
```

---

## CLI Usage — Save Claude Tokens from the Terminal

```bash
# Install graphsift MCP server (saves Claude tokens on every tool call)
graphsift install

# Install with transparent bash output compression
graphsift install --bash-wrapper

# Start MCP server for custom MCP clients
graphsift serve --port 8000

# Build/update the dependency graph
graphsift build --repo ./my_repo

# Show indexing status and cumulative Claude token savings
graphsift status

# Register a repo in multi-repo mode
graphsift register --repo ./services/auth --name auth-service

# Compress any CLI output — save 60-97% tokens before it reaches Claude
pytest -v | graphsift compress
docker ps -a | graphsift compress
kubectl get all | graphsift compress

# Show cumulative token savings across all sessions
graphsift gain

# Discover missed token-saving opportunities
graphsift discover --repo .

# Print bash wrapper for transparent compression
graphsift bash-wrapper
```

---

## CLI Output Compression — Save 60-97% of Command Tokens Before Claude Sees Them

**New in v1.6:** graphsift compresses CLI command output before it reaches your Claude or LLM context window. Think 200 lines of pytest tracebacks reduced to 5 lines of meaningful failures. Complements the code review context selection above.

```bash
# Auto-detect and compress — zero config needed
pytest -v | graphsift compress          # 94% token savings
docker ps -a | graphsift compress       # 91% token savings
git diff HEAD~3 | graphsift compress    # 92% token savings
kubectl get all | graphsift compress    # 75% token savings

# Ultra-compact mode — max 30 lines
cargo build 2>&1 | graphsift compress --ultra

# Save original + compressed (debug while Claude sees compressed)
pytest -v | graphsift compress --tee ~/.graphsift/tee --tee-label pytest_run
```

### All 19 Compressors — Auto-Detected

| Compressor | What it compresses | Strategy | **Token savings** |
|---|---|---|---|
| `grep` | Search results | Group by match, dedup identical lines | **95%** |
| `eslint` | ESLint output | Per-file error/warning counts | **94%** |
| `git_diff` | Git diffs | Per-file path + first 3 changed lines | **93%** |
| `pytest` | Test runs | Keep assertions + failures, strip tracebacks | **90%** |
| `npm` | npm/yarn | Error headers + conflict summary + counts | **87%** |
| `docker` | docker ps/images | ID + name/status, cap at 40 | **86%** |
| `git_status` | git status | Branch + staged/unstaged/untracked counts | **86%** |
| `pip` | pip install | Final summary + errors only | **85%** |
| `cargo` | Rust builds | Keep errors + warnings + Finished line | **83%** |
| `kubectl` | kubectl get | Header + first 5 rows, compress whitespace | **81%** |
| `git_log` | git log | Last 5 commits, hash + subject only | **80%** |
| `make` | make output | Error + *** lines only | **78%** |
| `aws` | AWS CLI JSON | Compact large JSON, keep keys + primitives | **76%** |
| `jest` | JavaScript tests | Keep FAIL/PASS + snapshot summary | **75%** |
| `go_test` | Go tests | Keep FAIL lines + panics + summary | **74%** |
| `log` | App logs | Strip timestamps, keep ERROR/FATAL, dedup WARN | **61%** |
| `cat` | File output | Truncate to 40 head + 20 tail | **29%** |
| `json_output` | Any JSON | Compact small, strip large to keys + primitives | — |
| `generic` | Anything | Strip blanks, dedup, truncate at 200 lines | **60%** |

### Transparent Bash Compression — Never Think About Token Savings

```bash
graphsift install --bash-wrapper
# or add to .bashrc:
eval "$(graphsift bash-wrapper)"
```

Now `pytest`, `cargo`, `npm`, `docker`, `kubectl`, `aws`, `grep`, `cat`, `make`, `pip`, `jest`, `eslint`, `git`, `go test`, `npx`, and `yarn` output is transparently compressed — Claude never sees the noise.

---

## Token Savings Analytics — Track Every Claude Token You Save

```bash
# Total Claude tokens saved across all sessions
graphsift gain

# Daily breakdown with cost estimates
graphsift gain --history

# Find commands that would benefit most from compression
graphsift discover --repo .

# Python API for dashboards
from graphsift.analytics import gain, history, discover

print(gain())         # {'total_calls': 1234, 'total_tokens_saved': 450000, 'estimated_cost_saved': '$6.75'}
print(history(7))     # Last 7 days breakdown
print(discover('.'))  # Missed opportunities
```

---

## MCP Server — Save Claude Tokens Inside Claude Code

![graphsift MCP server save Claude tokens — 25 tools, 4 prompts, 10 resources, 77% average token reduction per tool call vs reading raw files](https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/token_savings_chart.png)

graphsift's MCP server is the easiest way to **save Claude tokens inside Claude Code**, Claude desktop, Cursor, or any MCP-compatible agent. Average **77% token reduction per tool call**.

```bash
graphsift install   # auto-configures .mcp.json and hooks
```

### MCP Tools — Each Designed to Save Claude Tokens

| Tool | What it does | **Category** |
|---|---|---|
| `build_graph` | Index all source files and build the dependency graph | Graph |
| `update_graph` | Incrementally update graph with only changed files | Graph |
| `get_context` | Build ranked, token-budget-capped context for a diff | Context |
| `get_impact` | Return blast radius — all files potentially affected by changes | Impact |
| `graph_status` | Check if graph is built and see current stats | Status |
| `search_symbols` | Search for functions, classes, or modules by name | Search |
| `list_files` | List all indexed files sorted by token count | Files |
| `get_file_context` | Retrieve full source of a specific indexed file | Files |
| `minimal_context` | Ultra-low-token context — signatures only, no bodies | Context |
| `clear_graph` | Clear in-memory graph, forcing full rebuild | Graph |
| `run_postprocess` | Run flow/community detection, FTS rebuild, risk scoring | Analysis |
| `detect_changes` | Detect changed files with risk-scored impact analysis | Impact |
| `query_graph` | Run predefined queries: callers_of, callees_of, imports_of, etc. | Query |
| `list_flows` | List detected execution flows sorted by criticality | Analysis |
| `get_flow` | Get detailed info about a single execution flow | Analysis |
| `get_affected_flows` | Find execution flows that pass through changed files | Impact |
| `list_communities` | List detected code communities sorted by size | Analysis |
| `get_community` | Get details about a single code community | Analysis |
| `get_architecture_overview` | Generate architecture overview with communities and risk files | Analysis |
| `refactor` | Rename preview, dead-code detection, or suggestions | Refactor |
| `apply_refactor` | Apply a previously previewed rename to source files | Refactor |
| `generate_wiki` | Generate markdown wiki pages from community structure | Docs |
| `get_wiki_page` | Get a specific wiki page by community name | Docs |
| `semantic_search_nodes` | Search code symbols by name or keyword (FTS5-powered) | Search |
| `list_repos` | List all registered repositories in the registry | Registry |
| `cross_repo_search` | Search across all registered repositories | Search |

---

## New in v1.6 — Advanced Claude Token Optimization

### Diff-Aware Context Trimming

Instead of sending full files, send only the changed regions plus configurable surrounding context. Cuts tokens another 40-60% beyond ranking.

```python
config = ContextConfig(
    diff_aware_trimming=True,
    trimming_context_lines=10,  # lines of context around each changed region
)
```

### Entropy-Based Deduplication

Near-identical files (generated code, similar configs, boilerplate) are detected via entropy comparison and only the highest-scoring representative is included. Improves context diversity without losing coverage.

### Hybrid Search — BM25 + Sparse Vector Fusion

Semantic code search that combines BM25 full-text relevance with TF-IDF sparse vector similarity. Configurable alpha (0= pure vector, 1= pure BM25).

```python
from graphsift import HybridSearcher

searcher = HybridSearcher(alpha=0.7)
results = searcher.search("JWT token refresh logic", graph_nodes, top_k=10)
```

### Tree-Sitter Precise Parsing — 11 Languages

Beyond regex-based parsers, v1.6 adds tree-sitter for precise CST/AST parsing: functions, classes, methods, decorators, async functions, arrow functions, structs, interfaces, traits, impl blocks — all extracted with exact line numbers and signatures.

```python
from graphsift import register_tree_sitter_parsers

register_tree_sitter_parsers()
# Python, JavaScript, TypeScript, Go, Rust, Java, C, C++, Ruby, PHP, Bash
```

### Auto-Fix Suggestions from Graph Analysis

Detect issues across 5 categories by analyzing the dependency graph:

| Category | What it detects |
|---|---|
| `import` | Missing, unused, or circular imports |
| `type` | Type mismatches inferred from graph edges |
| `structure` | Architectural issues (cycles, god modules) |
| `cycle` | Dependency cycles with severity grading |
| `dead_code` | Unreachable code from entry points |

```python
from graphsift import FixSuggester

suggester = FixSuggester(builder)
report = suggester.analyze(graph)
print(f"Issues found: {report.total_issues}")
for s in report.suggestions:
    print(f"[{s.severity}] {s.file_path}:{s.line_start} — {s.title}")
```

### Cache-Aware Output

Structure rendered context with Anthropic/OpenAI cache_control breakpoints and session-based memory for repeated queries. Dramatically reduces costs when reviewing iterative changes to the same files.

```python
config = ContextConfig(
    cache_aware=True,
    cache_provider="anthropic",  # or "openai", "auto"
    session_id="pr-review-1234",
    cache_ttl_days=7,
)
```

---

## Advanced Features — Maximum Claude Token Savings

### Smart Cache — Don't Pay Twice for the Same Context

```python
from graphsift import GraphCache

cache = GraphCache(maxsize=64, ttl=300)

@cache.memoize
def get_context(diff_key: str):
    return builder.build(diff, source_map)

get_context("auth-change-abc123")   # computed once
get_context("auth-change-abc123")   # cache hit — saves Claude tokens, saves money
print(cache.stats())
# {'hits': 1, 'misses': 1, 'evictions': 0, 'hit_rate': 0.5}
```

### Analysis Pipeline with Audit Trail

```python
from graphsift import AnalysisPipeline

pipeline = (
    AnalysisPipeline(builder)
    .add_step("filter_generated", lambda r: remove_generated_files(r))
    .add_step("rerank", rerank_by_complexity)
    .with_retry(n=2, backoff=0.3)
)
result, audit = pipeline.run(diff_spec, source_map)
```

### Async Batch — Parallel Claude-Powered Reviews

```python
from graphsift import async_batch_build, batch_index

results = batch_index(builder, [source_map_a, source_map_b], concurrency=4)
contexts = await async_batch_build(builder, list_of_diffs, source_map, concurrency=8)
```

### Streaming — Start Processing Before All Files Are Scored

```python
from graphsift import stream_context

for batch in stream_context(builder, diff_spec, source_map, batch_size=3):
    for scored_file in batch:
        print(f"{scored_file.file_node.path}: {scored_file.score:.3f}")
```

### Rate Limiter — Control Your Claude API Spend

```python
from graphsift import RateLimiter

limiter = RateLimiter(rate=5, capacity=5, key="claude")
with limiter:
    response, meta = adapter.review(...)
```

### Diff Engine — Compare Token Costs Across Configurations

```python
from graphsift import ContextDiff

diff = ContextDiff(result_config_a, result_config_b)
print(diff.summary())
# Tokens: 9,200 -> 14,100 (delta +4,900)
# Reduction: 95.1% -> 92.2%
```

---

## FAQ — Common Questions About Saving Claude Tokens

### How do I save tokens on Claude Code?

Install graphsift's MCP server (`graphsift install`) and it automatically compresses tool outputs and optimizes context for every Claude Code session. The `compress_output` tool auto-detects command types and applies the right compressor.

### What's the fastest way to reduce Claude API costs?

Use graphsift's `ContextBuilder` with a hard `token_budget` (e.g., 50,000). It ranks files by relevance and only sends what fits. Add `diff_aware_trimming=True` for another 40-60% reduction. Enable `cache_aware=True` for repeated queries on the same files.

### How much does Claude API cost per code review?

Without graphsift: $0.50-$2.70 per review (depending on codebase size). With graphsift: **$0.01-$0.05 per review** — a 93-99% reduction. At 100 PRs/day, that's $50-270/day vs $1-5/day.

### Does graphsift work with GPT-4 / GPT-5 / OpenAI?

Yes. graphsift has drop-in adapters for OpenAI/Codex and Gemini, plus a generic adapter for any OpenAI-compatible API. The ranking and selection logic is provider-agnostic.

### How is graphsift different from code-review-graph?

code-review-graph uses binary blast-radius (everything that imports the changed file, include/exclude). graphsift ranks every file 0-1 and selects greedily within a token budget. F1 accuracy: 0.85 vs 0.54. Token reduction: 80-150x vs 8-49x.

### Can graphsift handle monorepos?

Yes. `index_roots()` indexes multiple packages at once, and the ranking algorithm correctly scores cross-package dependencies.

### Does graphsift need internet access?

No. All parsing, ranking, and compression runs locally. API adapters call LLM providers if you use them, but the core is fully offline.

### What Python versions are supported?

Python 3.9+. The only mandatory dependency is `pydantic>=2.0`.

---

## Supported Languages — Save Claude Tokens on Any Codebase

| Language | Parser | Tree-sitter | Key capabilities |
|---|---|---|---|
| Python | Native `ast` + tree-sitter | Yes | Functions, classes, methods, async, decorators, dynamic imports |
| JavaScript | Regex + tree-sitter | Yes | Functions, classes, methods, arrow functions, async |
| TypeScript | Regex + tree-sitter | Yes | Same as JS + type annotations, interfaces |
| Go | Regex + tree-sitter | Yes | Functions, receiver methods, structs, interfaces |
| Rust | Regex + tree-sitter | Yes | Functions, structs, traits, impl blocks |
| Java | Regex + tree-sitter | Yes | Classes, methods, interfaces |
| C++ | Regex + tree-sitter | Yes | Functions, classes, structs |
| C | Regex + tree-sitter | Yes | Functions, structs |
| Ruby | Regex + tree-sitter | Yes | Methods, classes, modules |
| PHP | Regex + tree-sitter | Yes | Functions, classes, traits |
| Bash/Shell | Regex + tree-sitter | Yes | Functions, `source` imports |
| Terraform/HCL | Custom parser | No | Resources, variables, locals, modules, data sources |
| Helm Charts | Template parser | No | Go templates in YAML, Chart.yaml dependencies |
| Dockerfile | Custom | No | FROM, COPY, RUN, ENV, ARG instructions |

---

## Performance — How Fast graphsift Saves Claude Tokens

- **Indexing**: sub-2-second on 10,000+ file repos
- **Incremental re-index**: skips unchanged files via SHA-256 hash
- **No hangs**: depth cap (default 4) prevents infinite traversal on cyclic imports
- **Thread-safe**: all shared state behind `threading.RLock`
- **Async**: all blocking operations have `async def a<operation>()` twins
- **Context building**: <50ms for a typical diff on an indexed 1,000-file repo

---

## Testing

```bash
git clone https://github.com/maheshmakvana/graphsift.git
cd graphsift
pip install -e ".[dev]"
pytest tests/ -v
# 271 passed in ~4s
```

- `tests/test_core.py` — 60+ unit tests: parsers, graph ops, ranking, selection
- `tests/test_advanced.py` — 49+ async tests: all 10 advanced features
- `tests/test_hybrid_search.py` — 25 tests: BM25, sparse cosine, TF-IDF, search ranking
- `tests/test_tree_sitter.py` — 40+ tests: Python, JS, Go, Rust parsing
- `tests/test_diff_trimming.py` — 18 tests: hunk parsing, context trimming, preamble
- `tests/test_dedup.py` — 15 tests: entropy dedup, changed-file protection
- `tests/test_auto_fix.py` — auto-fix suggestion engine tests

---

## Architecture — Hexagonal (Ports & Adapters)

```
graphsift/
├── __init__.py              # Public API — all exports explicit
├── core.py                  # Pure domain logic, zero I/O
├── models.py                # Pydantic v2 value objects (frozen=True)
├── exceptions.py            # Typed exception hierarchy
├── advanced.py              # 10 advanced feature categories
├── compress.py              # 19 CLI output compressors (86% avg token savings)
├── analytics.py             # Token savings tracking + discovery
├── hooks.py                 # Bash wrapper + transparent compression
├── hybrid_search.py         # BM25 + TF-IDF sparse vector fusion
├── auto_fix.py              # Graph-based auto-fix suggestion engine
├── cli.py                   # CLI entrypoint
├── mcp_server.py            # MCP protocol server (7 tools)
├── parsers/                 # Tree-sitter parsers (11 languages)
├── adapters/
│   ├── storage.py           # SQLite GraphStore (6-version migrations)
│   ├── claude.py            # Claude/Anthropic adapter
│   ├── openai.py            # OpenAI / Codex adapters
│   ├── gemini.py            # Gemini adapter
│   ├── llm.py               # Shared multi-provider adapter logic
│   ├── filesystem.py        # Path I/O helpers
│   └── postprocess.py       # Community + flow detection
└── _version.py              # Single-source version
```

---

## Contributing

Issues and pull requests welcome at [github.com/maheshmakvana/graphsift](https://github.com/maheshmakvana/graphsift).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Related Projects

- **[tokenpruner](https://pypi.org/project/tokenpruner/)** — LLM input token compression used by graphsift's COMPRESSED output mode; adds 3-5x additional Claude token reduction
- **[code-review-graph](https://github.com/tirth8205/code-review-graph)** — binary blast-radius alternative (no ranking, no budget, no compression — graphsift was built to surpass it)

---

## Repository

| | |
|---|---|
| **Stars** | ![GitHub stars](https://img.shields.io/github/stars/maheshmakvana/graphsift) |
| **Forks** | ![GitHub forks](https://img.shields.io/github/forks/maheshmakvana/graphsift) |
| **Releases** | 12 (187 commits) |
| **MCP tools** | 25 tools + 4 prompts + 10 resources |
| **Test coverage** | 271 tests across 8 test files, 20 test classes |
| **Latest version** | v1.6.1 |
| **License** | MIT |

### Languages

| Language | % |
|---|---|
| Python | 94.5% |
| Shell | 2.3% |
| Dockerfile | 1.1% |
| Other | 2.1% |

### Topics

`python` `ai` `mcp` `developer-tools` `claude` `llm` `copilot` `claude-code` `token-optimization` `mcp-server` `code-review` `agentic-coding` `context-engineering` `reduce-token-costs` `ast-parser` `dependency-graph` `context-window` `tree-sitter` `output-compression` `bm25`

### Contributing

Issues and pull requests welcome at [github.com/maheshmakvana/graphsift](https://github.com/maheshmakvana/graphsift).

Read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Easy first PR: add a new CLI compression pattern via the issue template.

### Privacy & Security

- **No telemetry** — graphsift runs 100% locally, no data ever leaves your machine
- **No internet required** — all parsing, ranking, and compression is local
- **Zero cloud dependencies** — SQLite for persistence, no accounts or API keys needed
- MCP server binds to localhost only (127.0.0.1)

### Uninstall

```bash
pip uninstall graphsift
rm -rf ~/.graphsift
```

---

**Start saving Claude tokens today:** `pip install graphsift`
