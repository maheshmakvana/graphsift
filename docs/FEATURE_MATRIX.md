# graphsift v3.0 — Complete Feature Matrix

> **The full inventory of all 40 modules, 25 CLI compressors, 7 MCP tools, and every capability in graphsift v3.1.**
>
> Coverage: **95%** (27/28 features delivered). Tests: **767** (+9% from v3.0). Star on [GitHub](https://github.com/maheshmakvana/graphsift).

---

## 📑 Module Inventory — All 38 Modules

### Category Legend

| Icon | Category | Module Count | What It Does |
|:----:|----------|:-----------:|--------------|
| 🎯 | **Core Context** | 5 | Relevance ranking, token budgets, context building |
| 📉 | **Compression** | 5 | CLI output compression, conversation compaction |
| 🔍 | **Search** | 3 | Hybrid search, typed retrieval, temporal graph |
| 🧠 | **Memory** | 6 | Cross-session knowledge, team conventions, tiered storage |
| 🗺️ | **Planning** | 3 | Plan-first engine, DAG workflows, self-verification |
| 🔒 | **Security** | 4 | Path validation, command sanitization, data scrubbing |
| ✍️ | **Prompts** | 6 | Fable5 anti-hallucination templates |
| 🤖 | **Agents** | 5 | Agent-to-agent protocol, task management, drift detection |
| 🌐 | **Graph & Schema** | 4 | Graph persistence, schema evolution, postprocessing |
| ⚡ | **Async & Performance** | 5 | Async engines, process pooling, caching, budgets |

---

### 🎯 Core Context (5 modules)

| Module | Class/Function | Lines | Description | CLI Command |
|--------|---------------|:----:|-------------|:-----------:|
| `core.py` | `ContextBuilder` | ~500 | Builds ranked context for a diff from a source map. 3-tier output (hot/warm/cold). | `graphsift build` |
| `core.py` | `ContextConfig` | ~80 | Token budget, output tiers, trimming, diff awareness configuration | — |
| `core.py` | `DiffSpec` | ~30 | Describes changed files + diff text | — |
| `core.py` | `ContextResult` | ~40 | Result object: selected files, token count, rendered context, reduction ratio | — |
| `core.py` | `Sift` | ~200 | Unified v3 API — index, search, build, compress in one class | `graphsift sift` |

### 📉 Compression (5 modules)

| Module | Class/Function | Lines | Description | CLI Command |
|--------|---------------|:----:|-------------|:-----------:|
| `compress.py` | `compress()` | ~50 | Routes CLI output to 1 of 25 command-specific compressors | `graphsift compress` |
| `compress.py` | `ultra_compress()` | ~80 | 3-pass maximum compression (light/balanced/ultra levels) | `graphsift compress --ultra` |
| `compress.py` | `ConversationCompactor` | ~150 | Compresses agent conversation history — 60–82% savings | — |
| `compress.py` | `AutonomousCompressor` | ~100 | Self-triggering compaction at configurable token thresholds | — |
| `compress.py` | `CompressionLevel` | ~15 | Enum: LIGHT, BALANCED, ULTRA | — |

### 🔍 Search & Retrieval (3 modules)

| Module | Class/Function | Lines | Description |
|--------|---------------|:----:|-------------|
| `core.py` | `HybridSearcher` | ~180 | BM25 + TF-IDF + optional dense vector fusion (3 modes) |
| `core.py` | `TypedRetriever` | ~120 | PRISM-style typed graph traversal — 6 query intents |
| `core.py` | `TemporalGraph` | ~160 | Git-history-aware symbol tracking across commits |

### 🧠 Memory & Persistence (6 modules)

| Module | Class/Function | Lines | Description | TTL/Retention |
|--------|---------------|:----:|-------------|:------------:|
| `code_memory.py` | `CodeMemory` | ~300 | Code-anchored memory: 7 types (decision, gotcha, note, insight, todo, bug, convention) | Per-item TTL |
| `code_memory.py` | `ConventionLearner` | ~250 | Learns team coding conventions from codebase | 365 days |
| `code_memory.py` | `ContextEnricher` | ~180 | Multi-source exploration — discovers related symbols, imports, patterns | — |
| `cache.py` | `ASTCache` | ~200 | 2-tier LRU+SQLite cache with TTL, predictive warm, glob invalidation | Configurable |
| `cache.py` | `ReadCache` | ~80 | SHA-256 fingerprint dedup for file reads | Session |
| `models.py` | `TieredMemory` | ~150 | 4-tier hierarchy: axioms → rules → topic → archive | Configurable |

### 🗺️ Planning & Execution (3 modules)

| Module | Class/Function | Lines | Description | CLI Command |
|--------|---------------|:----:|-------------|:-----------:|
| `planner.py` | `Planner` | ~350 | Plan-first engine — 7 phases (scan → analyze → architect → plan → execute → validate → review) | — |
| `planner.py` | `ExecutionPlan` | ~100 | Structured plan with ordered phases and topological steps | — |
| `toolchain.py` | `ToolChain` | ~250 | DAG-based step chains — build, review, run with rollback | — |

### 🔄 Loop Engineering (2 new modules)

| Module | Class/Function | Lines | Description | CLI Command |
|--------|---------------|:----:|-------------|:-----------:|
| `loop_engineering.py` | `LoopEngine` | ~450 | Main orchestrator — struggle-aware loop execution | `graphsift loop` |
| `loop_engineering.py` | `StruggleDetector` | ~80 | Detects repeated failures, frustration, approach changes | — |
| `loop_engineering.py` | `LoopState` | ~100 | Persistent JSON state + run ledger | — |
| `loop_engineering.py` | `CircuitBreaker` | ~60 | Auto-stops after 5 consecutive failures | `loop reset-breaker` |
| `loop_engineering.py` | `HumanGate` | ~50 | L1/L2/L3 maturity safety model | — |
| `loop_engineering.py` | `LoopCostBudgeter` | ~60 | 500K tokens/day per-pattern cost cap | `loop cost` |
| `loop_engineering.py` | `WorktreeManager` | ~70 | Git worktree isolation | — |
| `loop_engineering.py` | `7 Pattern Types` | ~80 | Daily Triage, PR, CI, Deps, Changelog, Cleanup, Issues | `loop run <pattern>` |
| `loop_config.py` | `LoopConfig` | ~80 | JSON config at `.graphsift/loop-config.json` | `loop init` |
| — | SessionStart | — | One-shot ~12K tok diagnostic at session start | `loop session-start` |
| — | Audit Readiness | — | Loop readiness score (0-100) | `loop audit` |

### 🔒 Security (4 modules)

| Module | Class/Function | Lines | Description | Protects Against |
|--------|---------------|:----:|-------------|:---------------:|
| `verify_hooks.py` | `PathValidator` | ~80 | Blocks path traversal attacks in file operations | `../../../etc/passwd` |
| `verify_hooks.py` | `CommandSanitizer` | ~80 | Prevents command injection from untrusted input | `; rm -rf /` |
| `compress.py` | `DataScrubber` | ~120 | Redacts API keys, tokens, passwords, secrets from output | Secret leakage |
| `verify_hooks.py` | `SecurePipeline` | ~100 | End-to-end pipeline combining all 3 security layers | Combined |

### ✍️ Prompt Engineering — Fable5 Templates (6 modules)

| Template | File | Lines | Use Case | Anti-Hallucination Features |
|----------|------|:----:|----------|:--------------------------:|
| `FixBugTemplate` | `prompt_templates.py` | ~120 | Bug diagnosis with prior-art search | 5 confidence tiers, source quality hierarchy |
| `AddFeatureTemplate` | `prompt_templates.py` | ~100 | Feature addition with dependency analysis | `[VERIFIED-REAL]` markers, evidence checks |
| `RefactorTemplate` | `prompt_templates.py` | ~100 | Refactoring with semantic preservation checks | Coherence guard, structural integrity check |
| `ProductionAppTemplate` | `prompt_templates.py` | ~120 | Production-grade scaffolding | Phased guard, coverage checklist, validation theater block |
| `ThemeChangeTemplate` | `prompt_templates.py` | ~130 | Large-scale architecture changes | Full component inventory, knowledge currency check |
| `SecurityArchitectureTemplate` | `prompt_templates.py` | ~110 | Security review with attack-tree analysis | Attack-tree analysis, 4-tier risk assessment |

### 🤖 Agent Infrastructure (5 modules)

| Module | Class/Function | Lines | Description |
|--------|---------------|:----:|-------------|
| `async_engine.py` | `A2AServer` | ~200 | Agent-to-Agent protocol server (JSON-RPC/HTTP) |
| `async_engine.py` | `TaskManager` | ~150 | MCP async task manager with progress tracking |
| `verify_hooks.py` | `Harness` | ~160 | Pre/post validation hooks for agent pipelines |
| `drift_detector.py` | `DriftDetector` | ~100 | Detects output drift from expected patterns |
| `prioritize.py` | `PriorityScorer` | ~90 | 5-signal priority scoring (critical → high → medium → low → info) |

### 🌐 Graph & Schema (4 modules)

| Module | Class/Function | Lines | Description | Features |
|--------|---------------|:----:|-------------|:--------:|
| `models.py` | `GraphStore` | ~300 | SQLite-backed graph persistence | Nodes, edges, weighted relationships |
| `models.py` | `SchemaRegistry` | ~200 | 6 schema families, v1→v2 auto-migration | Naming convention discovery |
| `models.py` | `SchemaEvolution` | ~150 | Version-aware migration with auto-discover | Backward-compatible |
| `models.py` | `Postprocessor` | ~120 | Community detection, flow detection, risk scoring | Graph algorithms |

### ⚡ Async & Performance (5 modules)

| Module | Class/Function | Lines | Description | Concurrency Model |
|--------|---------------|:----:|-------------|:-----------------:|
| `async_engine.py` | `AsyncEngine` | ~200 | Async parallel execution with bounded semaphores | asyncio (semaphore=8) |
| `async_engine.py` | `ProcessPoolExecutor` | ~120 | Multi-process chunked file parsing | multiprocessing (50-file threshold) |
| `models.py` | `DatabasePool` | ~100 | Thread-safe SQLite pool | WAL mode + auto-reconnect |
| `cache.py` | `ASTCache` | ~200 | 2-tier LRU+SQLite with TTL + predictive warming | In-memory + disk |
| `cache.py` | `ReadCache` | ~80 | SHA-256 dedup for file reads | In-memory |

### Tools & Auto-Fix (3 modules)

| Module | Class/Function | Lines | Description |
|--------|---------------|:----:|-------------|
| `auto_verify.py` | `AutoVerifier` | ~250 | Self-verification cascade: syntax → lint → tests → fix (up to 3 retries) |
| `auto_verify.py` | `AutoPipeline` | ~100 | Combines executor + verifier into end-to-end pipeline |
| `toolchain.py` | `ToolBudget` | ~60 | Per-tool output line caps |

---

## 🖥️ CLI Commands — Full Reference

| Command | Purpose | Usage |
|---------|---------|-------|
| `graphsift build` | Index repo + dependency graph | `graphsift build [--path .]` |
| `graphsift install` | Register MCP server with Claude Code | `graphsift install` |
| `graphsift status` | Show indexing stats | `graphsift status` |
| `graphsift compress` | Pipe CLI output for token compression | `pytest -v \| graphsift compress` |
| `graphsift gain` | Show token savings analytics | `graphsift gain` |
| `graphsift discover` | Find missed token-saving opportunities | `graphsift discover` |

---

## 🎯 25 CLI Output Compressors — Complete List

| # | Compressor | File Lines | Command/Pattern | Avg Tokens Saved | Best For |
|:-:|:----------:|:--------:|:---------------:|:----------------:|:---------|
| 1 | `compress_pytest` | ~40 | pytest output | **90%** | Test failure analysis |
| 2 | `compress_cargo` | ~15 | Cargo build/test | **85%** | Rust builds |
| 3 | `compress_go_test` | ~15 | Go test output | **71%** | Go CI/CD |
| 4 | `compress_jest` | ~25 | Jest test output | **88%** | JS/TS test failures |
| 5 | `compress_eslint` | ~15 | ESLint output | **94%** | Lint review |
| 6 | `compress_git_status` | ~15 | `git status` | **81%** | Daily dev |
| 7 | `compress_git_diff` | ~20 | `git diff` | **93%** | Code review |
| 8 | `compress_git_log` | ~15 | `git log` | **69%** | History analysis |
| 9 | `compress_grep` | ~10 | `grep -r` | **95%** | Code search results |
| 10 | `compress_npm` | ~15 | npm install/audit | **89%** | JS dependency management |
| 11 | `compress_docker` | ~15 | docker ps/logs | **81%** | Container debugging |
| 12 | `compress_kubectl` | ~15 | kubectl get/describe | **81%** | Kubernetes debugging |
| 13 | `compress_aws` | ~15 | AWS CLI output | **80%** | Cloud debugging |
| 14 | `compress_json_output` | ~15 | Any JSON output | **85%** | API responses |
| 15 | `compress_make` | ~10 | make output | **89%** | Build debugging |
| 16 | `compress_pip` | ~10 | pip install | **90%** | Python dependency install |
| 17 | `compress_log` | ~10 | Log files | **87%** | Error log analysis |
| 18 | `compress_cat` | ~10 | File contents | **80%** | Configuration review |
| 19 | `compress_terraform` | ~15 | terraform plan/apply | **75%** | Infrastructure review |
| 20 | `compress_gh` | ~5 | gh CLI output | **80%** | GitHub operations |
| 21 | `compress_az` | ~5 | az CLI output | **80%** | Azure operations |
| 22 | `compress_gcloud` | ~10 | gcloud CLI output | **75%** | GCP operations |
| 23 | `compress_brew` | ~10 | brew output | **80%** | macOS package management |
| 24 | `compress_dotnet` | ~10 | dotnet build/test | **85%** | .NET builds |
| 25 | `compress_generic` | ~20 | Fallback for unknown types | **60%** | Any unrecognized output |

**Weighted average across all 25 compressors: 77% token reduction.**

---

## 🔧 MCP Tools — 7 Tools for Claude Code

When installed via `graphsift install`, these tools register in Claude Code automatically:

| Tool Name | Description | Token Savings |
|-----------|-------------|:------------:|
| `build_graph` | Index the repo into a searchable knowledge graph | — |
| `get_context` | Get ranked context for a specific file/change | **80–150×** |
| `search_code` | Semantic code search across the indexed repo | **Targeted** |
| `compress_output` | Compress CLI output before sending to context | **60–97%** |
| `check_evidence` | Validate that code changes match actual source | **Anti-hallucination** |
| `review_changes` | Review code changes with token-optimized context | **90%** |
| `check_conventions` | Check code against learned team conventions | **Memory-based** |

---

## 📊 Feature Coverage — v2.3 vs v3.0

| Category | v2.3 | v3.0 | Improvement |
|----------|:----:|:----:|:-----------:|
| Hallucination Prevention | 2 features / templates | 8 features / templates | **+300%** |
| Tests | 326 | 702 | **+115%** |
| Test categories | 1 (unit) | 5 (unit, property, stress, perf, integration) | **+400%** |
| Security classes | 2 | 5 | **+150%** |
| Memory systems | 1 | 3 | **+200%** |
| Concurrency tiers | 0 | 3 | **+300%** |
| Output savings | ~70% | ~85-90% | **+15-20%** |
| Source files | 31 | 48 | **+55%** |
| Overall coverage | 48% | 93% | **+45%** |

---

## 🔒 Security & Privacy Matrix

| Feature | Status | Notes |
|---------|:------:|-------|
| Zero telemetry | ✅ | No analytics pings, no usage tracking |
| Zero network calls | ✅ | Parsing, indexing, compression all local |
| Zero code exfiltration | ✅ | AST nodes never leave your machine |
| Zero LLM calls in library | ✅ | Tool *for* LLMs, not powered by them |
| Zero third-party SDKs | ✅ | No embedded analytics or error-reporting |
| DataScrubber | ✅ | Redacts API keys, tokens, passwords |
| PathValidator | ✅ | Blocks directory traversal |
| CommandSanitizer | ✅ | Blocks shell injection |
| SecurePipeline | ✅ | All 3 layers combined |
| MIT license | ✅ | Free for any use |

---

## 📈 Performance Benchmarks

| Operation | Time | Notes |
|-----------|:----:|-------|
| Index 10,000+ file repo | < 2 s | Single pass |
| Incremental re-index | < 0.5 s | Only changed files |
| Context build (1k file repo) | < 50 ms | With relevance ranking |
| Cache-hit retrieval | < 5 ms | In-memory |
| pytest output compression | < 1 ms | 1,334→136 tokens |
| npm audit compression | < 1 ms | 630→21 tokens |
| Conversation compaction | < 2 ms | 60–82% reduction |

---

## 🔗 Quick Links

| Resource | Description |
|----------|-------------|
| [Main README](../README.md) | Quick start, install, full API overview |
| [Why graphsift Detailed](WHY_GRAPHSIFT_DETAILED.md) | In-depth explanation with benchmarks and cost analysis |
| [API Reference](API_REFERENCE.md) | Complete reference for all public classes |
| [Architecture Deep Dive](DECONSTRUCTING_GRAPHSIFT_ARCHITECTURE.md) | Module design, data flow, decisions |
| [Economics of LLM Context Windows](ECONOMICS_MECHANICS_OF_LLM_CONTEXT_WINDOWS.md) | Why context optimization is the highest-leverage AI investment |
| [V3 Upgrade Guide](V3_UPGRADE_GUIDE.md) | Full v2.3 → v3.0 comparison |
| [Changelog](../CHANGELOG.md) | Full release history |
| [Wiki Home](wiki_home.md) | Community wiki — FAQ, use cases |

---

> **graphsift v3.0 — 38 modules, 25 compressors, 702 tests, 93% feature coverage.**
>
> `pip install graphsift` · Created by [Mahesh Makwana](https://github.com/maheshmakvana)
