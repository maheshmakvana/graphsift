# Refactor Planning Methodology

## Goal
Identify refactoring opportunities with full impact analysis and risk assessment before making changes.

## Approach
1. **DISCOVER** — Use `query_graph` to find highly-coupled modules, large functions, and repeated patterns. Look for files with excessive dependencies (high fan-in + high fan-out) as prime candidates for extraction.
2. **IMPACT** — For each candidate target, run `get_impact_radius` to map the full blast radius. Categorize reach: LOW (same-file, isolated), MEDIUM (same-module, few callers), HIGH (cross-module, many callers). Use `get_impact` for detailed dependency analysis on HIGH-risk candidates.
3. **PLAN** — Design the refactor incrementally: extract interface first, migrate callers one by one, then replace implementation. Use `get_context` to understand the full scope before starting each phase.
4. **VERIFY** — After each phase, verify no behavior changed. Check that the dependency graph improved (cycles removed, coupling reduced, cohesion increased).

## Key Questions
- What is the blast radius of this refactor?
- Can it be done incrementally or does it require a flag day?
- Does the refactor improve actual architectural metrics (coupling, cohesion)?
