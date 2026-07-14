# graphsift Feature Catalog (SSOT Snapshot)

Complete feature inventory as of v1.6.1. Last updated: 2026-05-27.

## Quick Stats

| Metric | Value |
|---|---|
| MCP tools | 25 |
| MCP prompts | 4 |
| MCP resources | 10 |
| CLI commands | 15 |
| Output compressors | 19 |
| Languages parsed | 14 |
| Tree-sitter languages | 11 |
| Edge types | 7 |
| Test count | 271 |
| Test classes | 20 |
| Test files | 8 |
| Advanced features | 10 |
| LLM adapters | 3 (Claude, OpenAI, Gemini) |
| Python support | 3.9+ |

---

## MCP Tools (25)

### Graph Management
- `build_graph` — Index all source files and build the dependency graph
- `update_graph` — Incremental update with only changed files (SHA-256 skip)
- `clear_graph` — Clear in-memory graph for clean rebuild
- `graph_status` — Current graph stats (files, symbols, edges)

### Context Building
- `get_context` — Ranked, token-budget-capped context for a diff (80-150x token reduction)
- `minimal_context` — Ultra-low-token mode — signatures only, no bodies (for <8K budgets)
- `get_file_context` — Retrieve full source of a specific indexed file

### Impact Analysis
- `get_impact` — Blast radius: all files potentially affected, scored 0-1 by distance
- `detect_changes` — Changed file detection with risk-scored impact analysis
- `get_affected_flows` — Find execution flows passing through changed files

### Graph Queries
- `query_graph` — Predefined queries: callers_of, callees_of, imports_of, importers_of, tests_for, children_of, inheritors_of, file_summary
- `search_symbols` — Search functions, classes, modules by name across the indexed codebase
- `semantic_search_nodes` — FTS5-powered code symbol search by keyword
- `cross_repo_search` — Search across all registered repositories

### Analysis & Visualization
- `run_postprocess` — Flow detection, community detection, FTS rebuild, risk scoring
- `list_flows` — List detected execution flows sorted by criticality
- `get_flow` — Detailed single flow info including full call path
- `list_communities` — List detected code communities sorted by size
- `get_community` — Single community details including all members
- `get_architecture_overview` — Architecture overview: communities, risk files, nodes/edges/files

### Refactoring
- `refactor` — Rename preview, dead-code detection, or refactoring suggestions
- `apply_refactor` — Apply a previously previewed rename to source files

### Documentation
- `generate_wiki` — Generate markdown wiki pages from community structure
- `get_wiki_page` — Get specific wiki page by community name

### Registry
- `list_files` — List all indexed files sorted by token count
- `list_repos` — List all registered repositories

---

## MCP Prompts (4)

- `review_code` — Review code changes using the dependency graph
- `analyze_impact` — Analyze blast radius and impact of changes
- `find_issues` — Search for code issues: cycles, dead_code, large_functions
- `explain_architecture` — Explain high-level architecture via community detection

---

## MCP Resources (10)

- `graphsift://graph-stats` — Current graph statistics
- `graphsift://architecture-overview` — Architecture overview with communities and risk scores
- `graphsift://communities` — Community list
- `graphsift://community/{name}` — Single community details with members
- `graphsift://wiki/{name}` — Wiki page for a community
- `graphsift://flows` — Execution flow list
- `graphsift://risk-index` — Risk index for the codebase
- `graphsift://cross-repo` — Cross-repository search index

---

## CLI Commands (15)

| Command | Description |
|---|---|
| `graphsift install` | Install MCP server config and optional bash wrapper |
| `graphsift serve` | Start MCP server on a port |
| `graphsift build` | Build/update the dependency graph for a repo |
| `graphsift register` | Register a repo in multi-repo mode |
| `graphsift list-repos` | List registered repositories |
| `graphsift status` | Show indexing status and token savings |
| `graphsift compress` | Compress any piped CLI output (19 compressors, auto-detect) |
| `graphsift gain` | Show cumulative token savings across all sessions |
| `graphsift gain --history` | Daily breakdown with cost estimates |
| `graphsift discover` | Find missed token-saving opportunities |
| `graphsift bash-wrapper` | Print bash wrapper for transparent compression |
| `graphsift watch` | Watch mode: auto-rebuild on file changes |
| `graphsift version` | Print graphsift version |
| `graphsift doctor` | Verify installation and dependencies |
| `graphsift uninstall` | Remove hooks and editor configs |

---

## Output Compressors (19)

| # | Compressor | Strategy | Token savings |
|---|---|---|---|
| 1 | `grep` | Group by match, dedup identical lines | 95% |
| 2 | `eslint` | Per-file error/warning counts | 94% |
| 3 | `git_diff` | Per-file path + first 3 changed lines | 93% |
| 4 | `pytest` | Keep assertions + failures, strip tracebacks | 90% |
| 5 | `npm` | Error headers + conflict summary + counts | 87% |
| 6 | `docker` | ID + name/status, cap at 40 | 86% |
| 7 | `git_status` | Branch + staged/unstaged/untracked counts | 86% |
| 8 | `pip` | Final summary + errors only | 85% |
| 9 | `cargo` | Keep errors + warnings + Finished line | 83% |
| 10 | `kubectl` | Header + first 5 rows, compress whitespace | 81% |
| 11 | `git_log` | Last 5 commits, hash + subject only | 80% |
| 12 | `make` | Error + *** lines only | 78% |
| 13 | `aws` | Compact large JSON, keep keys + primitives | 76% |
| 14 | `jest` | Keep FAIL/PASS + snapshot summary | 75% |
| 15 | `go_test` | Keep FAIL lines + panics + summary | 74% |
| 16 | `log` | Strip timestamps, keep ERROR/FATAL, dedup WARN | 61% |
| 17 | `cat` | Truncate to 40 head + 20 tail | 29% |
| 18 | `json_output` | Compact small, strip large to keys + primitives | — |
| 19 | `generic` | Strip blanks, dedup, truncate at 200 lines | 60% |

**Weighted average: 77% token savings** (real data, not estimates)

---

## Language Support

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

## Edge Types (7)

| Edge | Description | Detected via |
|---|---|---|
| CALLS | Function/method invocation | AST + tree-sitter |
| IMPORTS | Module import | AST + regex + tree-sitter |
| INHERITS | Class inheritance | AST + tree-sitter |
| DECORATES | Decorator application | AST + tree-sitter |
| REFERENCES | Variable/attribute reference | AST + tree-sitter |
| TEST_COVERS | Test-to-source relationship | Path matching + naming conventions |
| DYNAMIC_IMPORT | Runtime imports | Regex + AST (`importlib`, `__import__`, `require()`) |

---

## Advanced Features (10 categories)

| Category | Feature | Description |
|---|---|---|
| **Cache** | GraphCache | TTL-based memoization with LRU eviction |
| **Pipeline** | AnalysisPipeline | Composable step chain with audit trail |
| **Validation** | ContextValidator | Pre/post-build validation hooks |
| **Async** | async_batch_build | Parallel context building with configurable concurrency |
| **Streaming** | stream_context | Yield context batches as they're scored |
| **Diff Engine** | ContextDiff | Compare context results across configurations |
| **Schema Evolution** | SchemaEvolution | Forward-compatible graph schema migrations |

---

## Test Suite

| Test file | Test functions | Test classes | What it covers |
|---|---|---|---|
| `test_core.py` | 68 | — | Parsers, graph operations, ranking, selection |
| `test_advanced.py` | 58 | — | All 10 advanced features, async, streaming |
| `test_tree_sitter.py` | 43 | 4 | Python, JS, Go, Rust parsing |
| `test_hybrid_search.py` | 28 | — | BM25, sparse cosine, TF-IDF, search ranking |
| `test_dedup.py` | 23 | — | Entropy dedup, changed-file protection |
| `test_auto_fix.py` | 22 | — | Auto-fix suggestion engine (5 categories) |
| `test_diff_trimming.py` | 19 | — | Hunk parsing, context trimming, preamble |
| `test_adapters.py` | 10 | — | Claude, OpenAI, Gemini adapters |
| **Total** | **271** | **4** | All passing in ~4s |

---

## Architecture

```
graphsift/
├── __init__.py              # Public API — all exports explicit
├── core.py                  # Pure domain logic, zero I/O
├── models.py                # Pydantic v2 value objects (frozen=True)
├── exceptions.py            # Typed exception hierarchy
├── advanced.py              # 10 advanced feature categories
├── compress.py              # 19 CLI output compressors (77% avg token savings)
├── analytics.py             # Token savings tracking + discovery
├── hooks.py                 # Bash wrapper + transparent compression
├── hybrid_search.py         # BM25 + TF-IDF sparse vector fusion
├── auto_fix.py              # Graph-based auto-fix suggestion engine
├── cli.py                   # CLI entrypoint
├── mcp_server.py            # MCP protocol server (25 tools, 4 prompts)
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

## Version History

| Version | Date | Highlights |
|---|---|---|
| v1.6.1 | Current | Claude token optimizer, 7 new engines, 271 tests, SEO-optimized |
| v1.6.0 | — | Output compression engine (19 compressors, 60-90% token savings) |
| v1.5.0 | — | 6 token-saving MCP tools + schema v7 |
| v1.4.0 | — | Full code-review-graph feature parity — 25 MCP tools, 15 CLI commands |
| v1.3.0 | — | SQLite persistence, 6-migration schema, progress logging, register/list-repos |
| v1.2.0 | — | Bash/HCL/Helm parsing, Go receiver methods, monorepo, incremental index |
