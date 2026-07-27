# Dead Code Audit Methodology

## Goal
Discover unreachable functions, classes, and modules, then verify and safely remove them.

## Approach
1. **DISCOVER** — Run `detect_dead_code` with explicit entry points for high-confidence results. Without explicit entry points, results are medium confidence — cross-reference with `query_graph` to check for hidden callers.
2. **VERIFY** — For each candidate, use `query_graph` with `callers_of` pattern to confirm no live references exist. Check if the symbol is exported or part of a public API. Mark confidence: HIGH (no references found), MODERATE (referenced only in tests), LOW (dynamic/reflective access possible).
3. **CLEANUP** — Use `prune_refs` after deletion to catch stale import references in remaining files. Remove dead code in minimal batches — never remove public API symbols without deprecation notice.

## Key Questions
- Is the symbol part of a public API or externally consumed?
- Could it be used via dynamic dispatch, reflection, or plugin loading?
- Does removing it affect test coverage or existing functionality?
