# Contributing to graphsift

> **graphsift v3.0** — by [Mahesh Makwana](https://github.com/maheshmakvana).
> The #1 token saver for Claude, GPT-4 & Gemini. 80-150x fewer tokens, F1 0.85, 702 tests.

Thank you for contributing! This guide covers everything you need to set up, code, test, and submit changes.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing Requirements](#testing-requirements)
- [Pull Request Workflow](#pull-request-workflow)
- [Architecture Overview](#architecture-overview)

---

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Step-by-step

```bash
# 1. Clone
git clone https://github.com/maheshmakvana/graphsift.git
cd graphsift

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify
pytest -xvs tests/
```

The `[dev]` extra installs:

| Package | Purpose |
|---|---|
| `pytest>=7.0` | Test runner |
| `pytest-asyncio>=0.23` | Async test support |
| `pytest-cov>=4.0` | Coverage reporting |
| `tree-sitter` + grammars | Parser development |
| `tokenpruner>=1.0.0` | Optional token compression |

### Minimal setup (no tree-sitter)

If working only on core logic (ranking, selection, storage):

```bash
pip install -e ".[dev]" --no-deps
pip install pytest pytest-asyncio pytest-cov
```

---

## Code Style

### Ruff

graphsift uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Run before committing:

```bash
pip install ruff
ruff check graphsift/ tests/
ruff format --check graphsift/ tests/
```

Configuration is in `pyproject.toml` — all public functions must have complete type annotations. Private helpers should be annotated too.

### Mypy

Run static type checking:

```bash
pip install mypy
mypy graphsift/ --ignore-missing-imports
```

Code should pass with minimal `# type: ignore` escapes.

### Conventions

| Rule | Standard |
|---|---|
| Modules | `snake_case.py` |
| Classes | `PascalCase` |
| Functions | `snake_case` |
| Private helpers | `_leading_underscore` |
| Constants | `UPPER_SNAKE_CASE` |
| Docstrings | Google-style on all public APIs |
| Line length | 100 characters max |
| Thread safety | Use `threading.RLock` for shared state |
| No LLM calls | graphsift works *for* LLMs, not *powered by* them |

### Architecture: Ports & Adapters

graphsift follows hexagonal architecture:

- **Core** (`core.py`) — ranking, selection, rendering. No I/O.
- **Adapters** (`graphsift/adapters/`) — filesystem, storage, postprocessing.
- **Parsers** (`graphsift/parsers/`) — tree-sitter and custom parsers.

Domain logic must never import from adapters directly; depend on abstractions.

---

## Testing Requirements

### Running tests

```bash
pytest -xvs tests/                     # Fast feedback, stop on first failure
pytest tests/                          # Full suite
pytest -xvs tests/test_core.py         # Single test file
pytest --cov=graphsift tests/          # With coverage report
```

### Test conventions

- Each module → corresponding `tests/test_<module>.py`
- Tests must be **deterministic** — no network, random state, or timing
- Use fixtures from `tests/conftest.py` for shared setup
- New features **must** include tests

### Coverage targets

| Layer | Target |
|---|---|
| Domain logic (`core.py`, `models.py`) | ≥ 85% |
| Overall | ≥ 70% |

### Tools used

| Tool | Purpose |
|---|---|
| `pytest` | Test runner |
| `pytest-asyncio` | Async test support |
| `pytest-cov` | Coverage reports |
| `hypothesis` | Property-based testing (when applicable) |

---

## Pull Request Workflow

1. **Fork** and create a feature branch from `master`.
2. **Keep changes focused** — one feature/fix per PR.
3. **Run tests** before committing:

   ```bash
   pytest -xvs tests/
   ruff check graphsift/ tests/
   ```

4. **Write clear commit messages:**

   ```
   feat: add tree-sitter parser for Kotlin

   Kotlin is one of the most requested languages for Android code review.
   The parser handles functions, classes, properties, and companion objects.
   ```

5. **Update docs** if you change public APIs, add features, or modify CLI commands.
6. **Open the PR** against `master`. CI will run tests + code quality checks.
7. **Address review feedback** promptly.

### PR checklist

- [ ] All tests pass (`pytest -xvs tests/`)
- [ ] Type hints are complete
- [ ] Public APIs have docstrings
- [ ] No hard dependencies added beyond pydantic
- [ ] No LLM calls introduced in library code
- [ ] Changes are backward-compatible (or migration path provided)
- [ ] New tests cover the change

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI / MCP                            │
│  graphsift.cli      graphsift.mcp_server                   │
│  graphsift.a2a_server                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     Domain Logic                            │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Context-     │  │ Dependency-  │  │ Relevance-       │  │
│  │ Builder      │  │ Graph        │  │ Ranker           │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬──────────┘  │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌───────▼──────────┐  │
│  │ Diff-        │  │ Hybrid-      │  │ Priority-       │  │
│  │ Spec         │  │ Searcher     │  │ Scorer          │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Evidence-   │  │ Temporal-    │  │ Code-           │  │
│  │ Checker     │  │ Graph        │  │ Memory          │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└───────────┬──────────────────┬──────────────────┬──────────┘
            │                  │                  │
┌───────────▼──────────────────▼──────────────────▼──────────┐
│                      Adapters                              │
│                                                            │
│  ┌────────────┐  ┌───────────┐  ┌──────────────────────┐  │
│  │ GraphStore │  │ Filesystem│  │ Postprocessor        │  │
│  │ (SQLite)   │  │ I/O       │  │ (Community, Flow,    │  │
│  └────────────┘  └───────────┘  │  Wiki, Risk, Refactor)│  │
│                                 └──────────────────────┘  │
└───────────┬────────────────────────────────────────────────┘
            │
┌───────────▼────────────────────────────────────────────────┐
│                      Parsers                               │
│                                                            │
│  ┌──────────────────────┐  ┌──────────────────────────┐   │
│  │ Tree-sitter (11 lang)│  │ Custom (3 lang: HCL,     │   │
│  │ Python, JS, TS, Go,  │  │ Helm, Dockerfile)        │   │
│  │ Rust, Java, C++, C,  │  └──────────────────────────┘   │
│  │ Ruby, PHP, Bash      │                                 │
│  └──────────────────────┘                                 │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                      Utilities                             │
│                                                            │
│  compress.py  analytics.py  security.py  executor.py      │
│  hooks.py     models.py     exceptions.py  auto_fix.py    │
│  prompt_templates.py        tool_budgets.py  read_cache.py│
│  verify_hooks.py            evidence_check.py             │
│  tiered_memory.py           compact_context.py            │
│  typed_retrieval.py         memory.py                     │
│  harness.py                 advanced.py                   │
└────────────────────────────────────────────────────────────┘
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `core.py` | ContextBuilder, RelevanceRanker, DependencyGraph, language parsing |
| `models.py` | Pydantic models: ContextConfig, ContextResult, DiffSpec, etc. |
| `compress.py` | 19 CLI output compressors with auto-detection |
| `hybrid_search.py` | BM25 + TF-IDF hybrid search |
| `analytics.py` | Token savings tracking and analytics |
| `security.py` | PathValidator, CommandSanitizer, DataScrubber |
| `executor.py` | CommandExecutor, SilentRunner, AutoPipeline |
| `memory.py` | Agent memory layer (SQLite knowledge graph) |
| `code_memory.py` | Code-anchored agent memory |
| `temporal_graph.py` | Git-history-aware symbol tracking |
| `typed_retrieval.py` | PRISM-style typed graph traversal |
| `compact_context.py` | Conversation compaction (60-82% savings) |
| `evidence.py` | Audit trail for file selection |
| `evidence_check.py` | File:line citation validation |
| `verify_hooks.py` | Post-change syntax/lint verification |
| `tool_budgets.py` | Per-tool output line caps |
| `read_cache.py` | SHA-256 fingerprint dedup for reads |
| `tiered_memory.py` | Hierarchical memory tiers |
| `prioritize.py` | Multi-signal priority scoring |
| `prompt_templates.py` | Structured prompt templates |
| `auto_fix.py` | Graph-based fix suggestions |
| `harness.py` | Agent harness with drift detection |
| `advanced.py` | Batch/async operations, diff engine |
| `adapters/` | Filesystem, SQLite storage, postprocessing |
| `parsers/` | Tree-sitter + custom language parsers |
| `mcp_server.py` | MCP stdio server (25+ tools) |
| `cli.py` | CLI entry point |
| `a2a_server.py` | Agent-to-Agent protocol (JSON-RPC/HTTP) |

---

## Getting Help

- **Issues:** [github.com/maheshmakvana/graphsift/issues](https://github.com/maheshmakvana/graphsift/issues)
- **Security:** See [SECURITY.md](SECURITY.md) for responsible disclosure
- **Discussions:** [github.com/maheshmakvana/graphsift/discussions](https://github.com/maheshmakvana/graphsift/discussions)
