# graphsift Benchmarks

Real-world compression and token-savings benchmarks. Reproducible on any machine.

Latest snapshot: 2026-05-27 (v1.6.1, 271 tests, 19 compressors)

## Reproduce

```bash
git clone https://github.com/maheshmakvana/graphsift.git
cd graphsift
pip install -e .

# Run the benchmark suite
python benchmark/run_benchmarks.py

# See live savings
graphsift gain
graphsift gain --history
```

## Output Compression Benchmarks

### Methodology

Each compressor was tested against a **realistic, representative sample** of CLI output: a 45-test pytest run with 4 failures, a multi-file git diff, a 10-container docker ps, a 3-commit git log, etc. Tokens were estimated using the standard 4-chars-per-token heuristic (matching tiktoken's cl100k_base within 5%).

### Results (sorted by savings)

| Rank | Compressor | Original chars | Compressed chars | Original tokens | Compressed tokens | **Savings** |
|---|---|---|---|---|---|---|
| 1 | `grep` | 1,654 | 91 | 413 | 22 | **94.7%** |
| 2 | `eslint` | 1,234 | 68 | 308 | 17 | **94.5%** |
| 3 | `git_diff` | 3,558 | 243 | 889 | 60 | **93.3%** |
| 4 | `pytest` | 5,336 | 547 | 1,334 | 136 | **89.8%** |
| 5 | `npm` | 1,152 | 159 | 288 | 39 | **86.5%** |
| 6 | `docker` | 1,852 | 255 | 463 | 63 | **86.4%** |
| 7 | `git_status` | 698 | 100 | 174 | 25 | **85.6%** |
| 8 | `pip` | 1,249 | 189 | 312 | 47 | **84.9%** |
| 9 | `cargo` | 1,854 | 320 | 463 | 80 | **82.7%** |
| 10 | `kubectl` | 2,327 | 442 | 581 | 110 | **81.1%** |
| 11 | `git_log` | 939 | 191 | 234 | 47 | **79.9%** |
| 12 | `make` | 1,002 | 223 | 250 | 55 | **78.0%** |
| 13 | `aws` | 1,911 | 462 | 477 | 115 | **75.9%** |
| 14 | `jest` | 1,241 | 305 | 310 | 76 | **75.5%** |
| 15 | `go_test` | 1,136 | 296 | 284 | 74 | **73.9%** |
| 16 | `log` | 1,610 | 620 | 402 | 155 | **61.4%** |
| 17 | `cat` | 2,689 | 1,917 | 672 | 479 | **28.7%** |
| 18 | `json_output` | 1,138 | 1,138 | 284 | 284 | **0.0%** |
| **TOTAL** | | **31,630** | **7,566** | **8,138** | **1,884** | **76.8%** |

### Per-category analysis

| Category | Compressors | Avg savings | Best | Worst |
|---|---|---|---|---|
| **Test runners** | pytest, jest, go_test | 79.7% | pytest 89.8% | go_test 73.9% |
| **Git** | git_diff, git_status, git_log | 86.3% | git_diff 93.3% | git_log 79.9% |
| **Containers** | docker, kubectl | 83.8% | docker 86.4% | kubectl 81.1% |
| **Package managers** | npm, pip, cargo | 84.6% | npm 86.5% | cargo 82.7% |
| **Linting** | eslint | 94.5% | eslint 94.5% | eslint 94.5% |
| **Search** | grep | 94.7% | grep 94.7% | grep 94.7% |
| **Infrastructure** | aws, make, log | 71.7% | make 78.0% | log 61.4% |
| **Pass-through** | cat, json_output | 14.4% | cat 28.7% | json_output 0.0% |

### Key Findings

1. **Best performers** (90%+): grep, eslint, git_diff, pytest — these compressors strip highly repetitive structure while preserving the critical signal (matches, errors, changed lines, assertions).

2. **Strong performers** (80-89%): npm, docker, git_status, pip, cargo, kubectl — well-structured output that benefits from field extraction and summarization.

3. **Solid performers** (70-79%): git_log, make, aws, jest, go_test — more variable output, but still significant compression wins.

4. **Moderate** (60-69%): log — app logs are inherently variable and timestamps are only one part of the noise.

5. **Low** (<30%): cat, json_output — these are pass-through compressors by design. cat already has the content you want; JSON is already compact. The value here is in the structured access, not compression.

---

## Code Review Context Benchmarks

Benchmarked on a 143-file FastAPI application reviewing a 50-line change to `src/auth/manager.py`:

| Approach | Files sent | Tokens | Cost (Opus @ $15/M) | Reduction vs raw |
|---|---|---|---|---|
| Raw source (every file) | 143/143 | ~180,000 | $2.70 | — |
| Binary blast-radius (code-review-graph) | 8-12/143 | 6,000-8,000 | $0.10 | 96% |
| **graphsift (FULL mode)** | **4-7/143** | **2,500-4,000** | **$0.05** | **98%** |
| **graphsift (SMART mode)** | **3-5/143** | **800-1,200** | **$0.015** | **99.4%** |
| **graphsift (COMPRESSED mode)** | **2-3/143** | **400-600** | **$0.007** | **99.7%** |

### F1 Accuracy

| Tool | Precision | Recall | F1 | False positive rate |
|---|---|---|---|---|
| code-review-graph | 0.56 | 0.52 | 0.54 | 44% |
| **graphsift** | 0.88 | 0.82 | **0.85** | **12%** |

---

## Test Suite Performance

```
271 passed in ~4s
8 test files, 20 test classes
test_core.py ................ 68 tests (parsers, graph, ranking, selection)
test_advanced.py ............ 58 tests (cache, pipeline, async, streaming, retry)
test_tree_sitter.py ......... 43 tests (Python, JS, Go, Rust tree-sitter)
test_hybrid_search.py ....... 28 tests (BM25, TF-IDF, sparse vectors)
test_dedup.py ............... 23 tests (entropy, changed-file protection)
test_auto_fix.py ............ 22 tests (5 issue categories)
test_diff_trimming.py ....... 19 tests (hunk parsing, context trimming)
test_adapters.py ............ 10 tests (Claude, OpenAI, Gemini)
```

---

## Indexing Performance

| Repo size | Files | Initial index | Incremental (1 file changed) |
|---|---|---|---|
| Small (<100 files) | 50 | <0.5s | <0.1s |
| Medium (100-1000 files) | 500 | <2s | <0.2s |
| Large (1000-10000 files) | 5,000 | <10s | <0.5s |
| Monorepo (10000+ files) | 20,000 | <30s | <1s |
