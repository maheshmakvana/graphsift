# Performance Audit Methodology

## Goal
Identify performance bottlenecks, oversized functions, and optimization opportunities across the codebase.

## Approach
1. **DISCOVER** — Use `find_large_functions` to locate oversized functions and classes (prime candidates for algorithmic bottlenecks or excessive allocations). Use `list_files` sorted by token estimate to identify the largest files that may contain performance-sensitive logic.
2. **ANALYZE** — For each candidate, use `query_graph` with `callers_of` and `callees_of` to understand the call chain. Focus on: hot-path functions called from many places, deeply nested loops in large functions, I/O operations without batching, and redundant computations. Cross-reference large files with their dependency usage to find unnecessary imports.
3. **RECOMMEND** — For each finding, provide: the specific bottleneck, estimated impact (COLD/WARM/HOT path), optimization strategy (memoization, lazy loading, algorithmic improvement, connection pooling), and expected trade-offs (memory vs CPU, complexity vs speed).

## Key Questions
- Which functions are both large AND called from many places (hot + heavy)?
- Are there unnecessarily large files being imported for small utility usage?
- Are I/O operations properly batched and async?
