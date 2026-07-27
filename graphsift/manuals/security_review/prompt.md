# Security Review Methodology

## Goal
Systematically identify vulnerabilities, insecure patterns, and compliance gaps across the codebase.

## Approach
1. **RECON** — Map the attack surface: identify entry points (CLI commands, API handlers, file readers), data flow across modules using `query_graph`, and high-risk files via `suggest_fixes` for security-sensitive patterns.
2. **ANALYZE** — For each entry point, trace the full data path. Check for: unvalidated input, path traversal, command injection, hardcoded secrets, missing authz checks, unsafe deserialization. Tag every finding with severity (CRITICAL/HIGH/MEDIUM/LOW) and evidence markers.
3. **VERIFY** — Use `check_evidence` to validate all file:line claims. Run `detect_dead_code` to check if dormant code hides backdoor risks. Cross-reference with dependency graph to assess blast radius of each vulnerability.
4. **REPORT** — Prioritize findings by severity and exploitability. For each, provide: exact location, root cause, exploit scenario, fix recommendation, and evidence level.

## Key Questions
- What are all the external entry points to the system?
- Is there input validation at every trust boundary?
- Are secrets, tokens, or credentials ever hardcoded?
- Which files have the highest security risk score?
