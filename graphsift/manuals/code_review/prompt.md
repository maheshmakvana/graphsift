# Code Review Methodology

## Goal
Review code changes with full dependency-aware context to catch correctness, security, and maintainability issues.

## Approach
1. **CONTEXT** — Use `get_review_context` with the changed files to get targeted snippets of the changed code plus their key dependents. For token-constrained scenarios, use `minimal_context` for signatures-only orientation, or `get_context` for full ranked context with a token budget.
2. **REVIEW** — Examine changes in order: correctness first (logic errors, edge cases, race conditions), then security (injection, authz, data leakage), then maintainability (duplication, naming, complexity). For every finding, trace the dependency chain to verify the issue is real. Use `check_evidence` to validate all file:line references before reporting.
3. **VERIFY** — After review, verify each finding: confirm the vulnerable path exists through the dependency graph, check if existing tests cover the changed lines, and ensure the fix suggestion respects the module's architectural boundaries.

## Key Questions
- What dependents will be affected by this change?
- Are there edge cases in the dependency chain that the diff doesn't handle?
- Does every file:line claim trace back to the actual diff?
