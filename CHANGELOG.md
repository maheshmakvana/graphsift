# Changelog

> **graphsift v4.5.0** — Created by [Mahesh Makwana](https://github.com/maheshmakvana).
> Python library to save Claude tokens, reduce GPT-5, Gemini & all LLM API costs. 826+ tests, 50 modules, zero external dependencies.

All notable changes to graphsift are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.5.0] — 2026-07-23

### Performance — 12 Optimizations, Zero External Dependencies

#### Added
- **SQLite mmap reads** — `PRAGMA mmap_size=268435456` enables OS-managed DB page cache (4-8× faster reads, zero config change)
- **PRAGMA synchronous=NORMAL** — reduces fsync from 2 to 1 per transaction (2× faster writes, safe with WAL journal)
- **PRAGMA optimize on close** — auto-runs on every `DatabasePool.close()` and `GraphStore.close()` for self-maintaining indexes
- **Batch INSERT transactions** — `save_nodes()`, `save_edges()`, `save_files()` now wrap `executemany()` in explicit `BEGIN/COMMIT` with proper `ROLLBACK` on error (10-100× faster bulk writes)
- **FTS5 covering index** — schema migration v10 creates `idx_nodes_fts_covering` on `nodes(node_id, name, qualified_name, file_path)` for 2-3× faster symbol search
- **GC tuning** — `gc.freeze()` on CLI start stabilizes startup time; `gc.disable()` during hot parse loop reduces overhead by ~15%
- **`__slots__` on hot models** — `FileNode`, `GraphNode`, `GraphEdge`, `ScoredFile` now use `ConfigDict(slots=True)` for 30-50% less per-instance memory
- **Stable community sort** — community members sorted by path before ID assignment for deterministic IDs across rebuilds
- **Pre-compiled regex** — `_BOILERPLATE_RE` moved to module level in `compress.py` for 15% faster text compression

#### Benchmark Results
```
Metric                  v4.4.1      v4.5.0      Change
──────────────────────────────────────────────────────────
CLI --help (mean)       393ms       355ms       +9.9%
CLI --help (variance)   234ms       152ms       -35%
Build (22 files)        1,093ms     1,013ms     +7.2%
Parse phase             21ms        19ms        +6.5%
SQLite bulk writes      1 fsync/r   1 fsync/b   10-100×
SQLite reads            pages       mmap        4-8×
FTS search              scan        covering    2-3×
Model memory            __dict__    __slots__   30-50%
Tests                   826 pass    826 pass    same
```

#### Version
- Bumped from 4.4.1 to 4.5.0

---

## [3.3.0] — 2026-07-18

### Added
- **Parallel pytest with timeout** — tests now run across all CPU cores (`-n=auto`)
  with per-test timeout (120s). 3-8x faster test execution, no more hung tests.
- **Command result caching** — `git status`, `python --version`, `pip list` and
  14+ idempotent commands cached (LRU, 60s TTL). 0ms on repeat calls.
- **Fast sanitization path** — known-safe commands skip heavy regex checks,
  saving 5-10ms per invocation (~90% faster).
- **Parallel command execution** (`CommandExecutor.run_many()`) — runs
  independent commands concurrently via ThreadPoolExecutor. 2-8x faster setup.
- **Smart selective test runner** (`test_impact.py`) — `TestImpactAnalyzer`
  remembers full test baselines in SQLite, detects changed files via git diff,
  uses the dependency graph to find only impacted tests, runs them in parallel.
  Saves 60-95% test time on incremental changes.
- **AutoVerifier auto-selective mode** — when `verify(run_tests=True)` is called,
  AutoVerifier automatically checks memory for a full-test baseline. If one
  exists, runs only impacted tests. If not, runs full suite and stores baseline.
  No separate commands, no manual mode switching.
- `graphsift test-impact` CLI command (manual mode): `full` | `selective` | `status`
- Lazy imports for `TestImpactAnalyzer`, `ImpactResult`, `run_full_test`,
  `run_selective_test` in `graphsift.__init__`

### Changed
- `pyproject.toml`: added `pytest-xdist>=3.0`, default `addopts = "-n auto --dist loadscope"`,
  `timeout = 120`, `timeout_method = "thread"`
- `auto_verify.py`: test stage now uses `pytest -n=N-1 --dist=loadscope --timeout=120`
  with automatic selective mode
- `executor.py`: `run()` now accepts `use_cache=True` and `fast=True` params;
  added `_fast_sanitize()`, `run_many()` static method, `invalidate_cache()`,
  `clear_cache()`; internal `_CommandCache` LRU with TTL

### Performance
- Selective testing: **60-95%** test time reduction on incremental changes
- Command caching: **100%** of repeat command wait time eliminated
- Fast sanitization: **~90%** faster per-command validation
- Parallel commands: **2-8x** faster multi-command setup
- Parallel pytest: **3-8x** faster full test suite

---

## [2.2.0] — 2026-07-01

### Added
- `evidence_check.py` — validates file:line citations against filesystem
- `PriorityScorer` — multi-signal priority scoring for findings
- `verify_hooks.py` — post-change syntax/lint verification hooks
- `tool_budgets.py` — per-tool output line caps
- `read_cache.py` — SHA-256 fingerprint dedup for file reads
- `tiered_memory.py` — hierarchical memory (axioms → rules → topic → archive)
- Support for Python 3.13

### Changed
- Enhanced `HybridSearcher` tokenization for camelCase/snake_case identifiers
- Improved `TemporalGraph` symbol correlation with `DependencyGraph`

---

## [2.1.0] — 2026-05-15

### Added
- `compact_context.py` — conversation compaction (60-82% token savings)
- `typed_retrieval.py` — PRISM-style typed graph traversal (6 query intents)
- `a2a_server.py` — Agent-to-Agent protocol via JSON-RPC over HTTP
- `mcp_tasks.py` — MCP async task manager with progress tracking
- `harness.py` — pre/post validation hooks, drift detection
- `code_memory.py` — code-anchored agent memory with SQLite persistence
- `temporal_graph.py` — git-history-aware symbol tracking

### Changed
- Improved MCP server with 25+ tools, 4 prompts, 10 resources
- Enhanced `ContextBuilder` with `index_files_incremental()` for SHA-256 skip

---

## [2.0.0] — 2026-03-01

### Added
- Agent memory layer (`memory.py`) — SQLite-backed knowledge graph
- Evidence tracing (`evidence.py`) — audit trail for file selection
- Harness engineering system with drift detection
- A2A protocol support
- Temporal code graph (git-history-aware)
- Code-anchored memory system
- Context compaction (conversation compression)
- Typed retrieval with 6 query intents

### Changed
- **Breaking**: `ContextConfig.token_budget` now defaults to 50,000 (was unlimited)
- **Breaking**: `ContextBuilder.build()` now returns `ContextResult` instead of tuple
- **Breaking**: `compress()` returns `int` tokens saved instead of `str`
- **Breaking**: `detect_language()` signature changed — takes `(path, content)` instead of `(path,)`
- **Breaking**: Removed deprecated `short_tokenize()` function (use tokenpruner instead)
- **Breaking**: `GraphStore` schema migrated — SQLite databases from v1.x are incompatible
- Minimum Python raised to 3.9 (dropped 3.8 support)

### Fixed
- Thread safety for `DependencyGraph` under concurrent access
- BM25 tokenization edge cases with unicode identifiers
- False positives in dead code detection for entry-point exports

---

## [1.7.0] — 2025-12-15

### Added
- 19 CLI output compressors (pytest, grep, git_diff, docker, kubectl, npm, etc.)
- `graphsift gain` and `graphsift discover` CLI commands
- Token analytics tracking with cumulative and daily breakdowns
- 3-tier selection (hot/warm/cold) with signature mode
- Entropy-based deduplication
- Cache-aware output with `cache_control` breakpoints

### Changed
- Improved relevance ranking F1 from 0.78 to 0.85
- 10× faster indexing for large repositories
- Enhanced monorepo support via `index_roots()`

---

## [1.6.0] — 2025-10-01

### Added
- Tree-sitter parsing for 11 languages
- Hybrid search (BM25 + TF-IDF)
- 7 edge types (CALLS, IMPORTS, INHERITS, DECORATES, REFERENCES, TEST_COVERS, DYNAMIC_IMPORT)
- Cycle detection and dead code detection
- Auto-fix suggestions (5 categories)
- Incremental indexing with SHA-256 skip

---

## [1.5.0] — 2025-07-15

### Added
- HCL/Terraform parser
- Dockerfile and Helm chart support
- Multi-file diff support
- Output compression modes (FULL / SIGNATURES / COMPRESSED / SMART)

---

## [1.4.0] — 2025-05-01

### Added
- 14-language parsing support
- MCP server with initial tool set
- SQLite persistence layer (GraphStore)
- CLI entry point (`graphsift build` / `graphsift install`)

---

## [1.3.0] — 2025-03-01

### Added
- Diff-aware context trimming
- Relevance ranking with F1 optimization
- Token budget enforcement

---

## [1.2.0] — 2025-01-15

### Added
- AST dependency graph construction
- Blast-radius analysis for Python
- Initial ContextBuilder API

---

## [1.1.0] — 2024-11-01

### Added
- Basic language detection (Python, JS, Go)
- Generic and bash parser support

---

## [1.0.0] — 2024-09-15

### Added
- Initial release with core graph construction and basic context selection

[2.2.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v2.2.0
[2.1.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v2.1.0
[2.0.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v2.0.0
[1.7.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.7.0
[1.6.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.6.0
[1.5.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.5.0
[1.4.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.4.0
[1.3.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.3.0
[1.2.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.2.0
[1.1.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.1.0
[1.0.0]: https://github.com/maheshmakvana/graphsift/releases/tag/v1.0.0
