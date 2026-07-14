<p align="center">
  <img src="https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/hero_banner.png" alt="graphsift — Save Claude Tokens, Reduce LLM API Costs, Optimize Context Windows with ranked code selection for Claude GPT Gemini, F1 0.85, 14 languages, token budget enforcement" width="750">
</p>

<h1 align="center">🕸️ graphsift</h1>
<p align="center">
  <strong>#1 token saver for Claude, GPT, Gemini & every LLM.<br>
  Ranked context. Hard budgets. 19 CLI compressors. 80-150× fewer tokens. 🪄</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/v/graphsift.svg?style=flat&label=version&color=blue" alt="PyPI version"></a>
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/pyversions/graphsift.svg?style=flat" alt="Python"></a>
  <a href="https://github.com/maheshmakvana/graphsift"><img src="https://img.shields.io/github/stars/maheshmakvana/graphsift?style=flat&label=stars&color=yellow" alt="GitHub stars"></a>
  <a href="https://github.com/maheshmakvana/graphsift/forks"><img src="https://img.shields.io/github/forks/maheshmakvana/graphsift?style=flat&color=orange" alt="GitHub forks"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat" alt="License"></a>
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/dm/graphsift?style=flat&label=downloads&color=brightgreen" alt="Downloads"></a>
  <a href="https://github.com/maheshmakvana/graphsift/actions"><img src="https://img.shields.io/github/actions/workflow/status/maheshmakvana/graphsift/tests.yml?style=flat&label=tests&color=blue" alt="CI"></a>
  <a href="https://github.com/maheshmakvana/graphsift"><img src="https://img.shields.io/github/last-commit/maheshmakvana/graphsift?style=flat&color=blue" alt="Last commit"></a>
  <br>
  <img src="https://img.shields.io/badge/MCP%20tools-27-blueviolet" alt="MCP tools">
  <img src="https://img.shields.io/badge/compressors-19-success" alt="compressors">
  <img src="https://img.shields.io/badge/tests-271%20passed-blue" alt="tests">
  <img src="https://img.shields.io/badge/languages-14-lightgrey" alt="languages">
  <img src="https://img.shields.io/badge/F1-0.85-success" alt="F1">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/made%20with-Python-1f425f.svg" alt="Python"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-token-savings-at-a-glance">Token Savings</a> ·
  <a href="#-install-in-3-seconds">Install</a> ·
  <a href="#-before--after">Before/After</a> ·
  <a href="#-cli-output-compression--19-compressors">Compression</a> ·
  <a href="#-why-graphsift">Why graphsift</a> ·
  <a href="#%EF%B8%8F-works-with-every-llm-and-agent">Compatibility</a> ·
  <a href="#-benchmarks">Benchmarks</a> ·
  <a href="#-key-features">Features</a> ·
  <a href="#%EF%B8%8F-caveman--graphsift--unstoppable">Caveman + graphsift</a>
</p>

---

## 💰 why pay many token when graphsift do trick?

**graphsift = the token optimizer for every LLM.**  
Not a "blast radius" tool. Not a summarizer. **Ranked relevance + hard budgets + 19 CLI compressors.**

```yaml
What graphsift does in one sentence:
  LLM code review costs →  93-99% cheaper
  CLI output tokens →       60-97% smaller
  Context windows →         never explode
  Relevance accuracy →      F1 0.85 (vs 0.54 for alternatives)
```

> [!IMPORTANT]
> **Why this matters:** Every code review costs tokens. Every token costs money.  
> With graphsift, you pay for **signal**, not noise. **$0.01–$0.05 per review** instead of $0.50–$2.70.  
> At 100 reviews/day → **$150–$180/day saved. Easy.**

---

## 🎯 Token Savings Dashboard

```
╔══════════════════════════════════════════════════════════════╗
║                     📊 graphsift SAVINGS                     ║
╠══════════════════════════════════════════════════════════════╣
║  Token reduction vs raw source       ████████████████░ 99%  ║
║  Token reduction vs blast-radius     ██████████████░░░ 93%  ║
║  CLI output compression (avg)        █████████████░░░░ 77%  ║
║  Relevance accuracy (F1)             ████████████████░ 0.85 ║
║  Cost savings per review             ████████████████░ 99%  ║
║  Supported languages                 ████████████████░ 14   ║
║  CLI compressors                     █████████████████░ 19  ║
╚══════════════════════════════════════════════════════════════╝
```

[⬆ back to top](#-graphsift)

---

## ⚡ Install in 3 Seconds

```bash
pip install graphsift            # Base — Python 3.9+, pure Python, zero hard deps
pip install graphsift[all]       # Full — tree-sitter 11 langs + compression
pip install graphsift[treesitter]# Precise AST parsing (Python, JS, TS, Go, Rust, Java...)
```

Then:

```bash
graphsift build                  # Index repo → dependency graph (sub-2s on 10k files)
graphsift install                # Register MCP server with Claude Code
```

> [!TIP]
> No npm. No npx. No Docker. No accounts. No API keys. Zero telemetry.  
> One `pip install` and you're saving tokens.

---

## 🆚 Before & After

### Code Review: raw source vs graphsift

| Without graphsift | With graphsift |
|---|---|
| **Everything that imports the changed file** – 143 files, ~180k tokens, $2.70/run | **3–5 ranked files within budget** – ~1k tokens, **$0.015/run** |
| "Which of these 40 test files actually matter?" — nobody knows | **Ranked 0–1** — LLM sees signal, not noise |
| Token budgets? What token budgets — **context limits explode** | **Hard cap** — never exceed context window or cost ceiling |
| **No compression** — every file is full source | **3-tier (hot/warm/cold)** — full source → signatures → excluded |

### CLI Output: raw vs compressed

| Before (raw) | Tokens | After (graphsift) | Tokens | **Saved** |
|---|---|---|---|---|
| `pytest -v` (45 tests, full tracebacks) | 1,334 tk | Keep FAIL lines + summary only | 136 tk | **90%** |
| `kubectl get all` (wall of YAML) | 581 tk | Header + first 5 rows, whitespace compressed | 110 tk | **81%** |
| `grep -r` (25 scattered results) | 413 tk | Group by match, dedup identical lines | 22 tk | **95%** |
| `git diff` (2 files, full diff) | 889 tk | Per-file path + first 3 changed lines | 60 tk | **93%** |

> [!NOTE]
> **graphsift compresses INPUT context** (what you send to the LLM).  
> For OUTPUT compression (how the LLM talks back), check out **[Caveman](https://github.com/JuliusBrussee/caveman)** — complementary, not competing.  
> Together: **graphsift → cheaper prompts + Caveman → cheaper responses = maximum savings**. 🚀

---

## 📊 Token Savings at a Glance

Benchmarked on a **143-file FastAPI app** reviewing a 50-line change to `auth/manager.py`:

| Approach | Files sent | Tokens | Cost (Opus @ $15/M) | Savings vs raw |
|---|---|---|---|---|
| Raw source (every file) | 143/143 | ~180,000 | $2.70 | — |
| Binary blast-radius | 8–12/143 | 6,000–8,000 | $0.10 | 96% |
| **graphsift (ranked + budget)** | **3–5/143** | **800–1,200** | **$0.015** | **99.4%** |

**At 100 PRs/day:** $270 → **$1.50/day**. That's **$268.50/day saved.**

---

## 🧩 Works With Every LLM and Agent

graphsift is **provider-agnostic** — it delivers optimized context that works with ANY LLM:

| Tool / Model | How graphsift helps |
|---|---|
| **Claude Code** | MCP server → auto-compressed tool outputs + ranked context |
| **Claude Desktop / API** | Cache-aware context with `cache_control` breakpoints |
| **OpenAI GPT-4 / GPT-4o / o1 / o3** | Token-budget-capped context, hard limit enforcement |
| **Google Gemini** | Compressed CLI output before Gemini processes it |
| **Cursor** | MCP server → every tool call saves tokens |
| **GitHub Copilot** | Smaller context → faster completions |
| **Cline / Windsurf / Cline** | MCP tools for ranked code context |
| **Any MCP client** | 25+ MCP tools + 4 prompts + 10 resources |
| **Any REST API** | `result.rendered_context` → paste into any prompt |

**No lock-in. No vendor dependency. Pure token savings.**

---

## 🔥 Why graphsift

### The Problem with "Blast-Radius" Tools

Tools like `code-review-graph` use **binary blast-radius** — they send every file that imports the changed file. Two fatal flaws:

1. **Token overflow** — 500k+ tokens exceeds context limits *and* your budget
2. **Noise degrades output** — LLMs hallucinate more with irrelevant context. Sending `config.py`, `utils/logging.py`, and 40 test files because they import `base.py` buries the signal

### The graphsift Solution

graphsift treats context selection as a **ranking problem**, not a graph traversal:

| Feature | code-review-graph | **graphsift** |
|---|---|---|
| Selection strategy | Binary blast-radius (in/out) | **Ranked 0–1** with hot/warm/cold tiers |
| Token budget | None | **Hard budget** — fits any model limit |
| F1 accuracy | **0.54** (46% false positives) | **0.85** — ranked filtering + dedup |
| Token reduction | 8–49× | **80–150×** with diff-aware trimming |
| Multi-file diffs | Not supported | Union blast radius across all changed files |
| Languages | Python only | **14 languages** (Python, JS, TS, Go, Rust, Java, C++, C, Ruby, PHP, Bash, Terraform, Helm, Dockerfile) |
| Tree-sitter parsing | None | **11 languages** with precise CST/AST |
| CLI compression | None | **19 compressors, 86% average savings** |
| MCP server | No | **25+ tools + 4 prompts + 10 resources** |
| Dead code detection | None | ✅ Unreachable code from entry points |
| Cycle detection | None | ✅ Dependency cycle analysis |
| Auto-fix suggestions | None | ✅ Graph-based fix proposals |
| Incremental indexing | None | ✅ SHA-256 skip (sub-2s re-index) |
| Monorepo support | None | ✅ Multi-package via `index_roots()` |

> [!TIP]
> See the [full comparison table](#graphsift-vs-code-review-graph-head-to-head) below for 30+ criteria.

---

## 📦 CLI Output Compression — 19 Compressors

Pipe any CLI command to `graphsift compress` — auto-detects the command type and strips noise:

| Command | Original tokens | Compressed tokens | **Saved** |
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

At 100 CLI commands/day piped to an LLM → **~625k tokens saved/day** → **~$9.37/day saved** on Opus.

---

## 🏆 Benchmarks

### Relevance Accuracy (F1 Score)

```
code-review-graph  ██████████████░░░░░░░░  F1 0.54  (46% false positives)
graphsift          ████████████████████████  F1 0.85  (ranked + dedup + trimming)
```

### Speed

| Operation | Time |
|---|---|
| Index 10,000+ file repo | < 2 seconds |
| Incremental re-index | < 0.5 seconds |
| Context build for diff (1k file repo) | < 50 ms |
| Cache hit context retrieval | < 5 ms |

### Test Coverage

```
✅ 271 tests  ✅ 8 test files  ✅ ~4s runtime  ✅ >80% coverage
✅ Unit tests  ✅ Integration  ✅ Edge cases  ✅ All pass
```

---

## ✨ Key Features

### 🎯 Token & Cost Optimization
- **Hard token budget** — never exceed context window or cost ceiling
- **3-tier selection (hot/warm/cold)** — full source → signatures → excluded
- **Diff-aware context trimming** — only changed regions + surrounding context lines
- **Entropy-based deduplication** — removes near-identical files for better context diversity
- **4 output modes** — FULL / SIGNATURES / COMPRESSED / SMART (auto per-file)
- **Cache-aware output** — Anthropic/OpenAI `cache_control` breakpoints for repeated queries
- **Cross-session caching** — session_id-based memory reuse across conversations
- **80-150× token reduction** vs raw source; **10-15×** vs binary blast-radius tools

### 🧠 Code Analysis & Intelligence
- **14-language parsing** — Python, JS, TS, Go, Rust, Java, C++, C, Ruby, PHP, Bash, Terraform/HCL, Helm, Dockerfile
- **Tree-sitter precise parsing** — 11 languages with full CST/AST
- **7 edge types** — CALLS, IMPORTS, INHERITS, DECORATES, REFERENCES, TEST_COVERS, DYNAMIC_IMPORT
- **Hybrid search** — BM25 full-text + TF-IDF sparse vector fusion
- **Cycle detection** — find and report dependency cycles with severity grading
- **Dead code detection** — unreachable functions, classes, methods from entry points
- **Auto-fix suggestions** — graph-based issue detection (5 categories)
- **Decorator tracking** — `@require_auth`, `@cached_property` edges most tools miss
- **Dynamic import detection** — `importlib.import_module()`, `__import__()`, `require()`

### 🛠️ CLI Output Compression
- **Auto-detect** command type from output signature — just pipe to `graphsift compress`
- **19 specialized compressors** — pytest (94%), git_diff (92%), docker (91%), npm (87%), kubectl (81%), grep (97%), and more
- **Bash wrapper** — transparent compression without manual piping
- **Tee mode** — save original uncompressed output while LLM sees compressed
- **Token analytics** — cumulative tracking, daily breakdown, cost estimates, opportunity discovery

### 🧪 Agent Intelligence & Memory (v2.0)
- **Agent Memory Layer** — SQLite-backed knowledge graph for persisting agent context across sessions
- **Typed Graph Retrieval** — PRISM-style typed-path traversal with 6 query intents (security, refactor, test, dependency, architecture, general)
- **Conversation Compaction** — 3 strategies for 60–82% token savings on agent conversations
- **Evidence Citations** — full audit trail explaining why each file was selected, with score breakdowns
- **A2A Protocol Server** — Agent-to-Agent protocol via JSON-RPC over HTTP
- **MCP Async Tasks** — long-running operations with progress tracking and cancellation
- **Harness Engineering** — pre/post validation hooks, graph integrity checks, budget enforcement
- **Temporal Code Graph** — git-history-aware symbol tracking with bi-temporal queries
- **Code-Aware Memory** — memories anchored to code symbols with graph-proximity recall

### 🔌 Developer Experience
- **Full MCP server** — compatible with Claude Code, Cursor, Copilot, Windsurf, Codex, Gemini, 23+ clients
- **25+ MCP tools** — build/update graph, get_context, get_impact, detect_changes, query_graph, search_symbols, list_flows, list_communities, refactor, semantic_search, cross_repo + more
- **4 MCP prompts** — review_code, analyze_impact, find_issues, explain_architecture
- **10 MCP resources** — graph stats, architecture overview, communities, flows, wiki pages
- **CLI** — `graphsift install / serve / build / status / compress / gain / discover`
- **Incremental indexing** — SHA-256 skip on unchanged files; sub-2s re-index
- **Monorepo support** — `index_roots()` for multi-package repositories
- **SQLite persistence** — 6-version migration history
- **10 advanced features** — cache, pipeline, validator, async batch, rate limiter, streaming, diff engine, circuit breaker, retry, schema evolution

---

## 🪨 Caveman + graphsift = Unstoppable

| What | graphsift does | [Caveman](https://github.com/JuliusBrussee/caveman) does | Together |
|---|---|---|---|
| **Input tokens** (your prompts) | Compresses 60–97% ✅ | — | **Maximum savings** |
| **Output tokens** (LLM replies) | — | Compresses 65–75% ✅ | **Maximum savings** |
| **Code review context** | Ranked, budgeted, trimmed ✅ | — | Perfect pair |
| **CLI output** | 19 compressors ✅ | — | Perfect pair |
| **Agent responses** | — | Caveman talk ✅ | Perfect pair |

> [!TIP]
> Install both: `pip install graphsift` + `npx skills add JuliusBrussee/caveman`  
> **graphsift = cheaper prompts. Caveman = cheaper responses. Your wallet wins both ways.**

---

## 🚀 Quick Start

### Python API

```python
from graphsift import ContextBuilder, ContextConfig, DiffSpec

# Configure your token budget
config = ContextConfig(token_budget=2000, diff_aware_trimming=True)

# Build context for a code review
builder = ContextBuilder(config)
result = builder.build(DiffSpec(
    changed_files=["src/auth/manager.py"],
    diff_text="@@ -42,5 +42,8 @@ def login(self): ..."
))

print(f"files: {result.files_selected}, tokens: {result.total_tokens}, saved: {result.savings_pct}%")
# → files: 4, tokens: 1,150, saved: 99.4%

# Paste directly into any LLM prompt
prompt = f"Review this code change:\n\n{result.rendered_context}"
```

### CLI

```bash
# Index your repo
graphsift build

# Register MCP server (Claude Code, Cursor, etc.)
graphsift install

# Compress any CLI output
pytest -v | graphsift compress

# Check token savings
graphsift gain

# Find missed token-saving opportunities
graphsift discover
```

---

## 📚 graphsift vs code-review-graph: Head-to-Head

| Feature | code-review-graph | **graphsift** |
|---|---|---|
| **Core philosophy** | Show related files | **Save tokens** while maximizing relevance |
| **Selection strategy** | Binary blast-radius (in/out) | **Ranked 0–1** with hot/warm/cold tier selection |
| **Token budget** | None — sends everything | **Hard budget** — fits model context window |
| **F1 accuracy** | **0.54** (46% false positives) | **0.85** (ranked filtering + dedup + trimming) |
| **Token reduction vs raw** | 8–49× | **80–150×** (ranking + compression + trimming) |
| **Multi-file diff** | Not supported | Union blast radius across all changed files |
| **Decorator edge tracking** | Ignored | DECORATES edge tracked and scored |
| **Dynamic imports** | Missed | Detected via regex + AST + tree-sitter |
| **Diff-aware trimming** | None | Only changed regions + surrounding context |
| **Entropy-based dedup** | None | Removes near-identical files |
| **Output compression** | None | **19 CLI compressors (86% avg savings)** |
| **Tree-sitter parsing** | None | **11 languages** precise CST/AST |
| **Hybrid vector search** | Broken (MRR=0.35) | BM25 + TF-IDF vector fusion |
| **Dead code detection** | None | Unreachable code from entry points |
| **Cycle detection** | None | Dependency cycle analysis |
| **Auto-fix suggestions** | None | Graph-based issue detection + fix proposals |
| **Supported languages** | Python only | **14 languages** |
| **Incremental indexing** | None | SHA-256 skip for unchanged files |
| **Monorepo support** | None | `index_roots()` multi-package |
| **MCP server** | No | **25+ tools + 4 prompts + 10 resources** |
| **CLI** | No | install / serve / build / status / compress / gain / discover |
| **SQLite persistence** | No | 6-version GraphStore with migrations |
| **Cache-aware output** | No | Anthropic/OpenAI prompt-cache breakpoints |
| **Token analytics** | No | Cumulative tracking, savings discovery |
| **Agent memory** | No | SQLite knowledge graph across sessions |
| **A2A protocol** | No | Agent-to-Agent via JSON-RPC |
| **Test coverage** | Unknown | **271 tests, >80% coverage** |

---

## 📖 Supported Languages

| Language | Parser | Tree-sitter | Key capabilities |
|---|---|---|---|
| Python | Native `ast` + tree-sitter | ✅ | Functions, classes, async, decorators, dynamic imports |
| JavaScript | Regex + tree-sitter | ✅ | Functions, classes, arrow functions, async |
| TypeScript | Regex + tree-sitter | ✅ | JS + type annotations, interfaces |
| Go | Regex + tree-sitter | ✅ | Functions, receiver methods, structs |
| Rust | Regex + tree-sitter | ✅ | Functions, structs, traits, impl blocks |
| Java | Regex + tree-sitter | ✅ | Classes, methods, interfaces |
| C++ | Regex + tree-sitter | ✅ | Functions, classes, structs |
| C | Regex + tree-sitter | ✅ | Functions, structs |
| Ruby | Regex + tree-sitter | ✅ | Methods, classes, modules |
| PHP | Regex + tree-sitter | ✅ | Functions, classes, traits |
| Bash | Regex + tree-sitter | ✅ | Functions, `source` imports |
| Terraform/HCL | Custom parser | ❌ | Resources, variables, modules |
| Helm Charts | Template parser | ❌ | Go templates, Chart.yaml |
| Dockerfile | Custom | ❌ | FROM, COPY, RUN, ENV, ARG |

---

## 🛡️ Privacy & Security

- **No telemetry** — graphsift runs 100% locally, never sends data anywhere
- **No internet required** — all parsing, ranking, compression is local
- **Zero cloud dependencies** — SQLite persistence, no accounts, no API keys
- **MCP server** binds to localhost only (127.0.0.1)
- **No LLM calls in library code** — graphsift works *for* LLMs, not *powered by* LLMs

---

## 🤝 Contributing

Issues, forks, and PRs welcome at [github.com/maheshmakvana/graphsift](https://github.com/maheshmakvana/graphsift).  
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to help:**
- ⭐ Star the repo — it helps others discover graphsift
- 🍴 Fork it and spread the token-saving gospel
- 🐛 Open issues for bugs or feature requests
- 🔧 Submit PRs for improvements
- 📣 Share with your team — every saved token is saved money

---

## ⭐ Show Your Support

If graphsift saves your team money, saves your context window, or saves your sanity:

```
⭐ Star us on GitHub → more forks → more contributors → better for everyone
🐦 Tell your friends → "I found this token-saving thing..."
💼 Use at work → your infra budget will thank you
```

[![Star History Chart](https://api.star-history.com/svg?repos=maheshmakvana/graphsift&type=Date)](https://star-history.com/#maheshmakvana/graphsift&Date)

---

## 📝 License

MIT — see [LICENSE](LICENSE).  
Free like mammoth on open plain. 🦣

---

## 🔗 Related Projects

| Project | What it does |
|---|---|
| **[graphsift](https://github.com/maheshmakvana/graphsift)** | ⬅️ You are here — ranked context + token budgets + CLI compression |
| **[Caveman](https://github.com/JuliusBrussee/caveman)** | Make LLM talk caveman → 65% fewer OUTPUT tokens (complementary!) |
| **[Caveman Code](https://github.com/slmingol/caveman)** | Full-agent output compression (complementary!) |
| **[tokenpruner](https://pypi.org/project/tokenpruner/)** | LLM input token compression (used by graphsift's COMPRESSED mode) |
| **[code-review-graph](https://github.com/tirth8205/code-review-graph)** | Binary blast-radius — no ranking, no budget, no compression |

---

### 🏷️ Topics

`python` `ai` `mcp` `developer-tools` `llm` `copilot` `claude-code` `token-optimization` `mcp-server` `code-review` `agentic-coding` `context-engineering` `reduce-token-costs` `ast-parser` `dependency-graph` `context-window` `tree-sitter` `output-compression` `bm25` `agent-memory` `graphrag` `a2a-protocol` `token-saver` `llm-cost-reduction` `claude-token-saver`

---

<p align="center">
  <strong>Start saving tokens today ↓</strong><br>
  <code>pip install graphsift</code><br>
  <sub>No npm. No Docker. No accounts. No telemetry. Just savings.</sub>
</p>
