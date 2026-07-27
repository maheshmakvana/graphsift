## Dependency Scanner Plugin

Scans Python dependencies using `safety check` for known vulnerabilities.

**Usage**: Run with the path to a requirements.txt or pyproject.toml.

**Returns**: Found vulnerabilities with CVE IDs, severity levels, and patched versions.

**When to use**: Before deploying or when auditing third-party dependencies.
