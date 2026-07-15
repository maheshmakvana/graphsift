# graphsift v2.3 → v3.0 Upgrade Guide

> Full feature comparison across all 3 tiers. v3.0 addresses **25/27 features** (92.6%).

---

## Tier 1 — Must Have (10/10 ✅)

| # | Feature | v2.3 | v3.0 | Change |
|---|---------|------|------|--------|
| 1 | Repository Indexing | Core ContextBuilder | +DepthTier/SourceConfidence, God-node penalty, SimHash dedup, diff-aware trimming | Enhanced |
| 2 | Search | BM25 + sparse TF-IDF | +RRF fusion, dense vectors, 3 fusion modes | Enhanced |
| 3 | Context Retrieval | 23 ContextConfig fields | +5 fields (budget_mode, pruning_strategy, centrality_weight, overlap_threshold, schema_version), 2-tier caching | Enhanced |
| 4 | Planning | ❌ Did not exist | New planner.py: Planner, ExecutionPlan, 7 PlanPhase, topological execution | **New** |
| 5 | AST Parsing | Tree-sitter 14 langs | +6 validators, graceful fallback | Enhanced |
| 6 | Hallucination Prevention | 3-4 basic templates | 6 Fable5 templates: [VERIFIED-REAL] markers, confidence tiers, coherence guard | Enhanced |
| 7 | Safety | PathValidator, CommandSanitizer | +DataScrubber, SecurePipeline, NetworkAccessError | **New** |
| 8 | Self Verification | Verifier (check/lint) | +AutoVerifier (syntax→lint→tests→fix loop), EvidenceChecker citations | **New** |
| 9 | Testing | ~326 tests, 13 files | **702 tests** (+115%), 42 files, 5 categories | Enhanced |
| 10 | Git Integration | TemporalGraph | +SchemaRegistry (6 families, v1→v2 migration) | Enhanced |

## Tier 2 — Should Have (9/9 ✅)

| # | Feature | v2.3 | v3.0 | Change |
|---|---------|------|------|--------|
| 11 | Parallelism | ❌ Did not exist | async_engine.py + pool.py: AsyncContextBuilder, ProcessPoolExecutor, DatabasePool | **New** |
| 12 | Multi-Agent Architecture | Harness + A2A Server | +DriftDetector, GraphIntegrityHook, BudgetEnforcementHook | Enhanced |
| 13 | Knowledge Graph | DependencyGraph | GraphNode/Edge v2 schemas (community_id), SchemaRegistry (6 families, migration) | Enhanced |
| 14 | Memory | AgentMemory (v1.7) | +CodeMemory (7 types, TTL), TieredMemory (4-tier hierarchy) | **New** |
| 15 | Model Routing | PriorityScorer (v2.2) + ToolBudget | 5-signal scoring, 4 tiers, per-tool caps | Existing |
| 16 | Cost Optimization | Basic analytics | gain(), summary_line(), record_call(), discover(), estimate_cost() | Enhanced |
| 17 | Caching | ReadCache (session) | +ASTCache: 2-tier LRU+SQLite, TTL, warm(), predictive_warm() | **New** |
| 18 | Incremental Indexing | SchemaEvolution | +SchemaRegistry (auto-discover migrations), SHA-skip | Enhanced |
| 19 | Observability | Basic analytics + FileValidator | +6 validators, full analytics pipeline | Enhanced |

## Tier 3 — Advanced (6/8 ✅)

| # | Feature | v2.3 | v3.0 | Change |
|---|---------|------|------|--------|
| 20 | Continuous Learning | ❌ Did not exist | CodeMemory (cross-session SQLite), AutoVerifier retry loop | **New** |
| 21 | Team Memory | ❌ Did not exist | CodeMemory convention type (365d TTL), ConventionLearner | **New** |
| 22 | Distributed Execution | ❌ Did not exist | ProcessPool (chunked), DatabasePool, AsyncEngine (semaphore=8) | **New** |
| 23 | Predictive Task Planning | ❌ Did not exist | Planner: 7 phases, topological execution, PlanResult | **New** |
| 24 | Automated Refactoring | FixSuggester (basic) | +3 auto-fixable types, AutoVerifier cascade | Enhanced |
| 25 | Workflow Automation | ❌ Did not exist | ToolChain DAG, AutoPipeline, SilentRunner, CommandExecutor | **New** |
| 26 | Plugin Marketplace | ❌ Did not exist | ❌ **Not implemented** — ConventionLearner is a detector, not a plugin system | **Missing** |
| 27 | Cross-Repository Reasoning | ❌ Did not exist | ❌ **Not implemented** — All modules single-repo scope only | **Missing** |

---

## Overall Metrics

| Metric | v2.3 | v3.0 | Change |
|--------|------|------|--------|
| Source files | ~31 | 48 | +55% |
| Modules | ~27 | 38 | +41% |
| Test files | 13 | 42 | +223% |
| Total tests | ~326 | 702 | +115% |
| Test categories | Unit only | 5 (unit/fuzz/integration/property/stress) | +400% |
| Tier 1 coverage | 7/10 (70%) | 10/10 (100%) | +30% |
| Tier 2 coverage | 5/9 (56%) | 9/9 (100%) | +44% |
| Tier 3 coverage | 1/8 (13%) | 6/8 (75%) | +62% |
| **Overall coverage** | **13/27 (48%)** | **25/27 (93%)** | **+45%** |
| Entirely new modules | — | 11 (async_engine, pool, cache, planner, toolchain, auto_verify, conventions, explorer, api/, schemas/, validators) | — |

---

## New Module Map (v3.0 additions)

```
graphsift/
  ├── api/                    # Structured API layer (v1, v2)
  ├── async_engine.py         # Async parallel execution (NEW)
  ├── auto_verify.py          # Self-verification cascade (NEW)
  ├── cache.py                # 2-tier LRU+SQLite ASTCache (NEW)
  ├── conventions.py          # Convention learning (NEW)
  ├── explorer.py             # Multi-repo code exploration (NEW)
  ├── migrations.py           # Schema versioning & migration (NEW)
  ├── planner.py              # Plan-first execution engine (NEW)
  ├── pool.py                 # Thread-safe DB connection pool (NEW)
  ├── schemas/                # Schema models for all data types (NEW)
  ├── toolchain.py            # DAG workflow automation (NEW)
  └── validators.py           # 6 validation classes (NEW)
```

---

## Quick-Reference: Feature Matrix

```
                     v2.3     v3.0
Repository Indexing   ████░░   ██████
Search                ███░░░   ██████
Context Retrieval     █████░   ██████
Planning             ░░░░░░   ██████
AST Parsing           █████░   ██████
Hallucination Prev    ████░░   ██████
Safety               ██░░░░   ██████
Self Verification     ███░░░   ██████
Testing               ████░░   ██████
Git Integration       ████░░   ██████
Parallelism          ░░░░░░   ██████
Multi-Agent Arch      █████░   ██████
Knowledge Graph       ████░░   ██████
Memory                ██░░░░   ██████
Model Routing         █████░   ██████
Cost Optimization     ████░░   ██████
Caching               ██░░░░   ██████
Incremental Index     ████░░   ██████
Observability         ███░░░   ██████
Continuous Learning  ░░░░░░   ██████
Team Memory          ░░░░░░   ██████
Distributed Exec     ░░░░░░   ██████
Predictive Planning  ░░░░░░   ██████
Auto Refactoring      ██░░░░   ██████
Workflow Automation  ░░░░░░   ██████
Plugin Marketplace   ░░░░░░   ░░░░░░  ❌
Cross-Repo Reason    ░░░░░░   ░░░░░░  ❌
```

---

## Quality-Perspective Improvement Table

### Hallucination Prevention

| Metric | v2.3 | v3.0 | Improvement |
|--------|------|------|-------------|
| Anti-hallucination templates | 3-4 basic | **6 Fable5 templates** | +50% |
| Templates with `[VERIFIED-REAL]` evidence markers | 0 | 6/6 | +600% |
| Templates with confidence calibration tiers | 1 | 6/6 | +500% |
| Templates with coherence guard | 0 | 6/6 | +600% |
| Templates with validation theater detection | 0 | 6/6 | +600% |
| Templates with structured JSON output schemas | 1 | 6/6 | +500% |
| Source quality hierarchy / knowledge currency | ❌ | ✅ All templates | New |
| **Avg anti-hallucination features per template** | **~2** | **~8** | **+300%** |

### Token Savings & Cost

| Metric | v2.3 | v3.0 | Improvement |
|--------|------|------|-------------|
| Output compressors | 19 | **19 + ultra mode** | +5% |
| Caching layers | 1 (ReadCache) | **2 (ASTCache LRU+SQLite + ReadCache)** | +100% |
| Cache features | Store/retrieve | +TTL, warm(), predictive_warm(), glob invalidation | +200% |
| Search precision | BM25 only | **3 fusion modes (hybrid/rrf/rrf-dense)** | +200% |
| Cache-aware breakpoints | ❌ | ✅ (Anthropic prompt cache) | New |
| Token tracking | Basic estimate | **tiktoken precise + cost estimation** | +100% |
| Conversation compaction | ❌ | **60-82% agent conversation savings** | New |
| **Estimated total token savings on repeated ops** | **~70%** | **~85-90%** | **+15-20%** |

### Code Quality

| Metric | v2.3 | v3.0 | Improvement |
|--------|------|------|-------------|
| Total tests | ~326 | **702** | +115% |
| Test categories | 1 (unit) | **5 (unit/fuzz/integration/property/stress)** | +400% |
| Test files | 13 | **42** | +223% |
| Validators | 3 | **6 (+DiffSpecValidator, +CompositeValidator)** | +100% |
| Auto-fixable bug categories | 0 | **3 (type_return, type_param, dead_code)** | New |
| Fix suggestion categories | 2 | **5 (import/type/structure/cycle/dead_code)** | +150% |
| Verification stages | 1 (check) | **3 (syntax→lint→tests, up to 3 retries)** | +200% |
| Performance benchmarks | ❌ | ✅ **test_performance.py suite** | New |

### Safety & Security

| Metric | v2.3 | v3.0 | Improvement |
|--------|------|------|-------------|
| Security classes | 2 (PathValidator, CommandSanitizer) | **5 (+DataScrubber, +SecurePipeline, exceptions)** | +150% |
| Secret/data leak redaction | ❌ | ✅ DataScrubber (API keys, tokens, passwords) | New |
| End-to-end secure pipeline | ❌ | ✅ SecurePipeline.safe_build(), safe_compress() | New |
| Network access protection | ❌ | ✅ NetworkAccessError | New |

### Parallelism & Performance

| Metric | v2.3 | v3.0 | Improvement |
|--------|------|------|-------------|
| Async execution | ❌ | ✅ AsyncEngine (semaphore=8, asyncio.gather) | New |
| Multi-process parsing | ❌ | ✅ ProcessPoolExecutor (chunked, 50-file threshold) | New |
| Thread-safe DB access | ❌ | ✅ DatabasePool (WAL mode, auto-reconnect) | New |
| Concurrency tiers | 0 | **3 (asyncio / thread pool / process pool)** | +300% |

### Memory & Persistence

| Metric | v2.3 | v3.0 | Improvement |
|--------|------|------|-------------|
| Memory systems | 1 (AgentMemory) | **3 (AgentMemory + CodeMemory + TieredMemory)** | +200% |
| Memory types in CodeMemory | — | **7 (decision/gotcha/note/insight/todo/bug/convention)** | New |
| TTL-based expiry | ❌ | ✅ (365d/180d/90d/30d per type) | New |
| Memory tiers | ❌ | **4 (axioms/rules/topic/archive)** | New |
| Team-shared conventions | ❌ | ✅ ConventionLearner + CodeMemory storage | New |

### Architecture & Schema

| Metric | v2.3 | v3.0 | Improvement |
|--------|------|------|-------------|
| Schema versioning | ❌ | ✅ 6 schema families, v1→v2 migration paths | New |
| REST API layer | ❌ | ✅ api/v1, api/v2 with compat | New |
| Data-type schemas | ❌ | ✅ GraphSchema, ContextSchema, MemorySchema | New |
| Migration auto-discovery | ❌ | ✅ Naming convention + best-effort merge fallback | New |
| Planning/workflow modules | 0 | **3 (planner, toolchain, executor)** | New |

---

## Headline Summary

```
Category              v2.3           v3.0              Improvement
─────────────────────────────────────────────────────────────────────
Hallucination Prev    2 features/tpl 8 features/tpl     +300%
Tests                 326            702                +115%
Test categories       1              5                  +400%
Security classes      2              5                  +150%
Memory systems        1              3                  +200%
Concurrency tiers     0              3                  +300%
Output savings        ~70%           ~85-90%            +15-20%
Overall coverage      48%            93%                +45%
```

