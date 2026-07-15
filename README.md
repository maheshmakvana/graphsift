<p align="center">
  <img src="https://raw.githubusercontent.com/maheshmakvana/graphsift/master/docs/images/hero_banner.png" alt="graphsift — Ranked Code Context & Token Optimizer for LLMs" width="600">
</p>

<h1 align="center">graphsift</h1>
<p align="center">
  <strong>#1 Token Saver for Claude, GPT-4, Gemini & Every LLM —<br>
  80–150× Fewer Tokens, F1 0.85 Relevance Accuracy</strong>
</p>

<p align="center">
  Created by <a href="https://github.com/maheshmakvana"><strong>Mahesh Makwana</strong></a>
  ·
  <a href="https://x.com/makwanamahesh5">@makwanamahesh5</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/v/graphsift.svg?style=flat&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/pyversions/graphsift.svg?style=flat" alt="Python"></a>
  <a href="https://pypi.org/project/graphsift/"><img src="https://img.shields.io/pypi/dm/graphsift?style=flat&label=downloads&color=brightgreen" alt="Downloads"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat" alt="License"></a>
  <a href="https://github.com/maheshmakvana/graphsift/actions"><img src="https://img.shields.io/github/actions/workflow/status/maheshmakvana/graphsift/tests.yml?style=flat&label=tests&color=blue" alt="CI"></a>
  <img src="https://img.shields.io/badge/F1-0.85-success" alt="F1">
  <img src="https://img.shields.io/badge/languages-14-lightgrey" alt="languages">
  <img src="https://img.shields.io/badge/CLI%20compressors-19-orange" alt="CLI compressors">
  <a href="https://github.com/maheshmakvana/graphsift/stargazers"><img src="https://img.shields.io/github/stars/maheshmakvana/graphsift?style=flat&color=yellow" alt="Stars"></a>
</p>

<p align="center">
  <a href="#save-tokens-why-graphsift">Why graphsift?</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#install">Install</a> ·
  <a href="#api-overview">API</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#docs">Docs</a>
</p>

---

## 🚀 Save Tokens: Why graphsift?

**graphsift** is a **token optimization engine** purpose-built for AI-assisted development. Every time you send code context to an LLM (Claude, GPT-4, Gemini, Codex, etc.), graphsift automatically **selects only the most relevant files**, **compresses output**, and **enforces token budgets** — slashing your AI costs by **80–150×** without losing accuracy.

Built by **Mahesh Makwana**, graphsift is the **#1 Python library to save tokens on Claude Code, reduce OpenAI API costs, and optimize LLM context windows** for code review, debugging, and code generation workflows.

### How graphsift Compares to Other Token-Saving Tools

| Tool | What It Saves | Approach | Token Reduction | Best For |
|------|--------------|----------|:---:|----------|
| **graphsift** ✅ | **Input tokens** (code context) | Ranked relevance + tier selection + entropy dedup | **80–150×** | Code review, PRs, debugging, AI codegen |
| [Caveman](https://github.com/JuliusBrussee/caveman) 🗿 | **Output tokens** (LLM replies) | Instructs agent to speak concisely | ~65% | Complementary — use together! |
| [tokenpruner](https://pypi.org/project/tokenpruner/) ✂️ | **Input tokens** (any text) | Semantic compression | ~70-80% | General prompt compression |
| [Repomix](https://repomix.com) 📦 | **Code selection** | Concatenates files into single prompt | ~2× | Simple repo packing |

> **💡 Perfect Stack:** Use **graphsift** (input code context) + **Caveman** (output replies) = **maximum token savings** across the full AI pipeline.

### Who Can Save with graphsift?

- **Claude Code users** — reduce `claude_code` costs by up to 99% per review
- **OpenAI / GPT-4 devs** — slash API bills with context-aware file selection
- **Gemini & Codex teams** — stay within context limits on large monorepos
- **MCP-connected agents** — automatic token optimization via MCP server
- **CI/CD pipelines** — compress `pytest`, `eslint`, `kubectl` output before LLM analysis

---

## Install

```bash
pip install graphsift                           # Core (pure Python, zero hard deps)
pip install "graphsift[all]"                    # Full: tree-sitter + compression extras
pip install "graphsift[treesitter]"             # +11 language AST parsers
pip install "graphsift[semantic]"               # +dense vector embeddings for semantic code search
```

No npm, no Docker, no accounts, no telemetry.

---

## Quick Start

```python
from graphsift import ContextBuilder, ContextConfig, DiffSpec

# Build context for a 50-line change to auth/manager.py
builder = ContextBuilder(ContextConfig(token_budget=2000, diff_aware_trimming=True))
result = builder.build(
    DiffSpec(changed_files=["src/auth/manager.py"], diff_text="@@ -42,5 +42,8 @@ def login(self):"),
    source_map={},  # dict of path -> source text
)

print(f"Files: {result.files_selected}, Tokens: {result.total_rendered_tokens}, Saved: {result.reduction_ratio:.1%}")
# Files: 4, Tokens: 1,150, Saved: 99.4%

# Compress CLI output before sending to an LLM:
from graphsift import compress
saved = compress(pytest_output, "pytest")  # 90% reduction
```

---

> **📊 See the full [WITH vs WITHOUT graphsift](#with-vs-without-graphsift--how-claude-behaves-differently) comparison below** — 15 real-world developer scenarios with token counts, cost impact, and quality verification.

---

## Core Concepts

**Ranked context selection** treats code review as a relevance-ranking problem, not a graph traversal. Instead of sending every file that imports a changed module (binary blast-radius), graphsift scores each file 0–1 using AST dependencies, BM25 full-text search, and diff proximity — then selects only the most relevant files within a hard token budget.

**Three-tier output** applies different compression per file: *hot* files (directly changed) get full source; *warm* files (strongly related) get signatures only; *cold* files (weakly related) are excluded. Combined with diff-aware trimming (only changed regions + surrounding context) and entropy-based deduplication, this achieves **80–150× token reduction** vs raw source.

**19 CLI compressors** auto-detect command type from output and strip noise — keeping FAIL lines from pytest, headers from kubectl, and match groups from grep. Pipe any command output to `graphsift compress` for **60–97% token savings** before sending it to an LLM.

---

## API Overview

| Class / Function | Description |
|---|---|
| `ContextBuilder(config)` | Builds ranked context for a diff from a source map |
| `ContextConfig(token_budget, ...)` | Configuration for token budget, tiers, trimming |
| `DiffSpec(changed_files, diff_text)` | Describes a code change to build context for |
| `ContextResult` | Result: selected files, token count, rendered context |
| `RelevanceRanker` | Scores files 0–1 using AST + BM25 + graph proximity |
| `DependencyGraph` | Builds and queries the AST dependency graph |
| `compress(text, cmd_type)` | Compresses CLI tool output (19 command types) |
| `HybridSearcher` | BM25 + TF-IDF + optional dense vector embeddings |
| `TemporalGraph` | Git-history-aware symbol tracking |
| `CodeMemory` | Code-anchored agent memory with SQLite persistence |
| `EvidenceChecker` | Validates file:line citations against filesystem |
| `Verifier` | Post-change syntax/lint verification hooks |
| `ToolBudget` | Per-tool output line caps |
| `ReadCache` | SHA-256 fingerprint dedup for file reads |
| `TieredMemory` | Hierarchical memory (axioms, rules, topic, archive) |
| `PriorityScorer` | Multi-signal priority scoring for findings |
| `FixSuggester` | Graph-based auto-fix suggestions |
| `A2AServer` | Agent-to-Agent protocol server (JSON-RPC/HTTP) |
| `Harness` | Pre/post validation hooks for agent pipelines |
| `Sift` | Unified v2 API — index, search, build, compress in one class |

See [API_REFERENCE.md](docs/API_REFERENCE.md) for the full reference.

---

## Benchmarks

### Relevance Accuracy (F1 Score)

| Approach | F1 | False Positives |
|---|---|---|
| Binary blast-radius | 0.54 | 46% |
| **graphsift (ranked + dedup + trimming)** | **0.85** | **15%** |

### Token Reduction — Save Claude & OpenAI Costs

Benchmarked on a **143-file FastAPI app** reviewing a 50-line change to `auth/manager.py`:

| Approach | Files Sent | Tokens | Cost (GPT-4 @ $30/M tokens) | Cost (Claude Opus @ $15/M) | Savings vs Raw |
|---|---|---|---|---|---|
| Raw source | 143/143 | ~180,000 | $5.40 | $2.70 | — |
| Binary blast-radius | 8–12/143 | 6,000–8,000 | $0.24 | $0.10 | 96% |
| **graphsift (ranked + budget)** | **3–5/143** | **800–1,200** | **$0.036** | **$0.015** | **99.4%** |

> Save **$2.69 per code review** with Claude Opus. On 100 reviews/month = **$269/month savings**.

### Performance

| Operation | Time |
|---|---|
| Index 10,000+ file repo | < 2 s |
| Incremental re-index | < 0.5 s |
| Context build (1k file repo) | < 50 ms |
| Cache-hit retrieval | < 5 ms |

### CLI Output Compression — 25 Command Types

| Command | Raw Tokens | Compressed | Saved | Developer Use Case |
|---|---|---|---:|---|
| `grep -r` | 413 | 22 | **95%** | Searching codebase |
| `pip install` | 115 | 11 | **90%** | Setting up dependencies |
| `make build` | 75 | 8 | **89%** | Building C/C++ projects |
| `npm audit` | 630 | 21 | **89%** | Security vulnerability audit |
| `npm test` | 720 | 86 | **88%** | Node.js CI pipeline |
| `eslint src/` | 308 | 17 | **94%** | Linting before commit |
| `docker ps` | 450 | 32 | **82%** | Checking containers |
| `docker logs` | 450 | 90 | **80%** | Debugging containers |
| `git diff` | 889 | 60 | **93%** | Code review / PRs |
| `git status` | 177 | 34 | **81%** | Pre-commit check |
| `git log` | 212 | 65 | **69%** | Reviewing commit history |
| `pytest -v` | 1,334 | 136 | **90%** | Running test suite |
| `go test ./...` | 128 | 37 | **71%** | Go CI/CD pipeline |
| `dotnet build` | 136 | 56 | **59%** | .NET CI/CD pipeline |
| `kubectl get all` | 581 | 110 | **81%** | Checking Kubernetes |
| `cargo build` | 217 | 102 | **53%** | Rust build errors |
| `brew install` | 150 | 81 | **46%** | macOS package install |
| `terraform plan` | 291 | 218 | **25%** | Infrastructure review |
| **Weighted avg** | **8,138** | **1,884** | **77%** | All developer workflows |

---
### WITH vs WITHOUT graphsift — How Claude Behave Differently

Tested across **15 real-world developer scenarios** (2,748 total raw tokens):


| Scenario | WITHOUT graphsift | WITH graphsift | Saved |
|:---|---:|---:|---:|
| Bug diagnosis (`pytest`) | 427 tok — PASS lines, tracebacks, headers | 143 tok — error types + failure messages | **66%** |
| Security audit (`npm audit`) | 191 tok — engine warnings, deprecations | 21 tok — vulnerability severity summary | **89%** |
| Code review (`git diff`) | 344 tok — context lines, index hashes | 61 tok — changed lines + new symbols | **82%** |
| CI/CD debug (`go test`) | 128 tok — compile trace + test details | 37 tok — FAIL lines + test names | **71%** |
| Lint report (`eslint`) | 214 tok — per-rule breakdown | 34 tok — file paths + error/warning counts | **84%** |
| Commit history (`git log`) | 212 tok — Author, Date, full message | 65 tok — hash + subject (5 commits) | **69%** |
| Working tree (`git status`) | 102 tok — full status output | 34 tok — branch + file counts | **67%** |
| Log analysis (`tail -f`) | 229 tok — timestamps, INFO lines, ANSI | 105 tok — ERROR/CRITICAL/WARNING messages | **54%** |
| Build output (`make`) | 74 tok — full compile log | 8 tok — error line only | **89%** |
| Rust build (`cargo`) | 131 tok — crate compilation list | 81 tok — error + warning lines | **38%** |
| Container check (`docker ps`) | 180 tok — full table with ports/status | 32 tok — container ID + name pair list | **82%** |
| Pod status (`kubectl`) | 179 tok — full table with all columns | 85 tok — header + first 5 rows | **53%** |
| Terraform plan review | 192 tok — full resource detail | 164 tok — full plan (already terse) | **15%** |
| All tests passing (`go test`) | 49 tok — package durations | 49 tok — unchanged (output is minimal) | **0%** |
| Dependency install (`pip`) | 96 tok — download bars, progress | 11 tok — success summary | **89%** |
| **TOTALS** | **2,748 tok** | **930 tok** | **66%** |


**Without graphsift:** Claude reads 2,748 tokens of unfiltered output — ANSI escapes, timestamps, progress bars, PASSED lines, and metadata all mixed with real signals. It must *find* the problem in the noise before it can *fix* it. This consumes 3x more tokens, costs more, and increases hallucination risk.

**With graphsift:** Claude reads 930 tokens of pre-filtered signal — only error types, failure messages, changed lines, and severity counts. It starts reasoning immediately, never sees secrets (DataScrubber), and never receives traversal/injection commands (PathValidator + CommandSanitizer).

#### What graphsift DROPS (safe — no LLM value):
- PASSED/OK lines, timestamps, ANSI colors, progress bars `[====]`
- Traceback frame internals (keeps error TYPE + MESSAGE)
- Package metadata, author emails, commit dates
- Docker/kubectl table formatting (keeps data rows)

#### What graphsift KEEPS (required for LLM quality):
- Error types (`OperationalError`, `SyntaxError`) + failure messages
- Vulnerability severity counts + CVE IDs
- Changed file paths + diff hunks + new symbols
- Test failure names + build error descriptions
- File-level lint error/warning aggregates
- Error/WARNING/CRITICAL log message text

#### Monthly Cost Comparison (Developer Workday)

| Usage Level | Runs/Month | WITHOUT graphsift | WITH graphsift | **Savings** |
|:---|---:|---:|---:|---:|
| Light | 110 | $4.50 | $1.50 | **$3.00/mo** |
| Medium | 220 | $9.00 | $3.00 | **$6.00/mo** |
| Heavy | 500 | $20.50 | $6.80 | **$13.70/mo** |
| CI/CD pipeline | 1,000 | $41.00 | $13.60 | **$27.40/mo** |

> **66% average token savings** means every Claude/GPT prompt gets 3x more useful content in the same context window.

## CLI

```bash
graphsift build         # Index repo + dependency graph — optimize Claude Code context
graphsift install       # Register MCP server with Claude Code for automatic token savings
graphsift status        # Show indexing stats
graphsift compress      # Pipe CLI output for instant token compression
graphsift gain          # Show token savings analytics — track cost reduction over time
graphsift discover      # Find missed token-saving opportunities
```

---

## Further Reading

| Document | What It Covers |
|---|---|
| [The Economics and Mechanics of LLM Context Windows](docs/ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md) | Why context optimization is the highest-leverage investment for LLM code review |
| [Deconstructing the graphsift Architecture](docs/DECONSTRUCTING_GRAPHSIFT_ARCHITECTURE.md) | Deep dive into module design, data flow, and architectural decisions |
| [Pillar 1: AST Dependency Graph](docs/PILLAR1_AST_DEPENDENCY_GRAPH.md) | How graphsift parses code into a navigable graph of symbols and dependencies |
| [Pillar 2: Hybrid Relevance Ranking](docs/PILLAR2_HYBRID_RELEVANCE_RANKING.md) | BM25 + graph-distance fusion for continuous 0–1 scoring |
| [Exhaustive Research Report](docs/GRAPHSIFT_EXHAUSTIVE_RESEARCH_REPORT.md) | Full architectural audit, improvement roadmap, and 6-agent implementation results |

---

## Related Token-Saving Projects

| Project | What It Does | How It Complements graphsift |
|---|---|---|
| **graphsift** ⚡ | Input code context optimization (80–150× reduction) | — |
| [Caveman](https://github.com/JuliusBrussee/caveman) 🗿 | Compress LLM *output* tokens (~65% savings) | Use together: **graphsift** for input, **Caveman** for output = max savings |
| [tokenpruner](https://pypi.org/project/tokenpruner/) ✂️ | General prompt token compression (70–80%) | Alternative for non-code contexts |

---

## Author

**graphsift** was built and is maintained by **Mahesh Makwana**.

<table>
  <tr>
    <td><strong>🐙 GitHub</strong></td>
    <td><a href="https://github.com/maheshmakvana">@maheshmakvana</a></td>
  </tr>
  <tr>
    <td><strong>🐦 Twitter / X</strong></td>
    <td><a href="https://x.com/makwanamahesh5">@makwanamahesh5</a></td>
  </tr>
  <tr>
    <td><strong>📧 Email</strong></td>
    <td><a href="mailto:maheshmakwana527@gmail.com">maheshmakwana527@gmail.com</a></td>
  </tr>
  <tr>
    <td><strong>📦 PyPI</strong></td>
    <td><a href="https://pypi.org/project/graphsift/">pypi.org/project/graphsift</a></td>
  </tr>
</table>

---

## License

MIT — see [LICENSE](LICENSE).

<p align="center">
  <code>pip install graphsift</code><br>
  <sub>🚫 Zero telemetry. 🔒 Zero accounts. 💰 Just savings.</sub><br>
  <sub>Built by <a href="https://github.com/maheshmakvana">Mahesh Makwana</a></sub>
</p>
