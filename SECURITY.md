# Security Policy

> **graphsift** — Created by [Mahesh Makwana](https://github.com/maheshmakvana).
> Save Claude tokens, reduce LLM costs. 80-150x token reduction, zero telemetry.

## Local-First, Zero-Exfiltration Design

graphsift is built on a **zero-exfiltration, local-first** architecture. The tool is designed to operate entirely on your machine with no data leaving the host under any circumstance.

### What graphsift does NOT do

- **No telemetry.** graphsift contains zero telemetry, analytics pings, or usage tracking. No data is transmitted during or after execution.
- **No network calls during parsing.** AST parsing, dependency graph construction, BM25 indexing, and context selection are all performed locally. graphsift makes no outbound network connections during `build`, `compress`, `gain`, `discover`, or any other command.
- **No code exfiltration.** AST nodes, dependency graphs, source code excerpts, token counts, and rendered context are never transmitted off the host machine. All data stays in process memory or local SQLite storage.
- **No LLM calls in library code.** graphsift is a tool *for* LLMs, not powered by them. No source code or derived data is ever sent to an external API by graphsift's library code.
- **No third-party data sharing.** graphsift does not embed third-party analytics SDKs, error-reporting services, or external dependencies that make network calls.

### What graphsift DOES with data

| Data | Use | Persistence |
|------|-----|-------------|
| Source code text | Parsed into AST, indexed by BM25, stored in local SQLite | Local only |
| Dependency graph | Constructed from AST, used for ranked relevance scoring | Local SQLite |
| Token counts | Recorded for savings analytics (`graphsift gain`) | Local SQLite |
| CLI tool output | Compressed in-memory, optionally logged | Local only |
| Agent memory facts | Stored for cross-session retrieval | Local SQLite |

### Data at rest

graphsift stores all indexed data in a local SQLite database under your project directory. No data from this database is ever transmitted. You can delete it at any time by removing the `.graphsift/` directory from your project root.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.x     | Active development and security patches |
| 1.x     | Critical security patches only |
| < 1.0   | End of life |

## Reporting a Vulnerability

If you discover a security vulnerability in graphsift, please report it privately by emailing the maintainer at **mahesh.m.makvana@gmail.com**.

Please do **not** file a public GitHub issue for security vulnerabilities.

We will acknowledge receipt within 48 hours and provide an initial assessment within 5 business days. We ask that you allow us a reasonable window to address the issue before any public disclosure.

### What to include

- A clear description of the vulnerability
- Steps to reproduce (conceptual or concrete)
- Affected versions
- Any potential impact or exploit scenario
- Your name/affiliation for acknowledgement (optional)

### Scope

The following are considered in scope for this security policy:

- The `graphsift` Python package (all modules under `graphsift/`)
- The MCP server implementation (`graphsift/mcp_server.py`)
- The CLI entry point (`graphsift/cli.py`)
- Local SQLite storage layer (`graphsift/storage.py`)

The following are out of scope:

- Third-party dependencies (report to their respective maintainers)
- LLM provider APIs that consume graphsift's output (graphsift has no control over how output is handled downstream)
- Misconfiguration of `SecurePipeline` or `PathValidator` by the user

## Security Features Built Into graphsift

graphsift includes a dedicated security module (`graphsift/security.py`) with the following protections:

- **PathValidator** — prevents path traversal attacks by validating that resolved paths stay within allowed base directories
- **CommandSanitizer** — sanitizes shell commands to prevent injection through CLI compressors
- **DataScrubber** — scrubs sensitive patterns (API keys, tokens, credentials) from output before rendering
- **SecurePipeline** — composes the above three into a single validation pipeline for end-to-end security
- **CommandExecutor** (with SilentRunner) — executes commands with safety guards, available as a PowerShell fallback on Windows

These components are opt-in and can be composed into custom pipelines for your specific security requirements.

## Dependency Supply Chain

- **Zero hard dependencies beyond pydantic.** graphsift's runtime core has no transitive attack surface beyond `pydantic>=2.0`.
- **Optional tree-sitter grammars** are installed explicitly via the `[treesitter]` or `[all]` extras — only the languages you need.
- **All dependencies** are pulled from PyPI. We recommend pinning versions in production and reviewing dependency diffs on upgrades.
