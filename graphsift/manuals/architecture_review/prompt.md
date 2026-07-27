# Architecture Review Methodology

## Goal
Provide a high-level architectural overview of the codebase using community detection and flow analysis.

## Approach
1. **OVERVIEW** — Use `get_architecture_overview` to get the top-level picture: total nodes, edges, files, communities, and high-risk files. Establish the overall structural health.
2. **COMMUNITIES** — Use `list_communities` to enumerate all detected modules. For each significant community (size > 5%), use `get_wiki_page` if available or examine members to determine responsibility. Map community boundaries against intended architecture.
3. **FLOWS** — Use `list_flows` to identify key execution pathways. Cross-reference with communities to see which modules participate in each flow. Identify unnatural couplings (flows that span too many communities).
4. **REPORT** — Synthesize findings into: architectural diagram description, module responsibilities, inter-module communication patterns, architectural violations, and concrete improvement recommendations.

## Key Questions
- Does the detected community structure match the intended architecture?
- Are there cross-cutting concerns that span too many modules?
- Which flows are the most critical and what modules do they touch?
- Are there any god modules with disproportionate influence?
