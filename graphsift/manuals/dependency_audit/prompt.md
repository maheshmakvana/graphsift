# Dependency Audit Methodology

## Goal
Trace dependency chains, identify circular dependencies, and measure module coupling across the codebase.

## Approach
1. **SCAN** — Use `detect_cycles` to find all circular dependency chains. Use `query_graph` with `importers_of` / `imports_of` patterns to map module relationships.
2. **ANALYZE** — For each cycle or tight coupling found, trace the import chain end-to-end. Categorize severity: cycles of length ≤3 are errors; longer cycles are warnings. Use `list_flows` to understand how dependencies propagate through execution paths.
3. **REPORT** — For each finding, provide the exact file chain, impact on maintainability, and a concrete resolution strategy (extract shared interface, invert dependency, split module).

## Key Questions
- Which modules have the highest fan-in / fan-out?
- Are there any cycles that cross architectural boundaries?
- Do any dependency chains violate the intended layering?
