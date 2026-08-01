#!/usr/bin/env sh
# Portable launcher for the graphsift MCP server (stdio).
# Avoids hardcoding any interpreter path. Probes, in order:
#   .venv/bin/python  venv/bin/python  python3  python
# Each candidate must be able to import graphsift before it is used.
set -eu
cd "$(dirname "$0")/.." || exit 1

for py in ".venv/bin/python" "venv/bin/python" "python3" "python"; do
    if command -v "$py" >/dev/null 2>&1 && "$py" -c "import graphsift" >/dev/null 2>&1; then
        exec "$py" -m graphsift.mcp_server "$@"
    fi
done

echo "graphsift-mcp: no Python with graphsift installed was found." >&2
echo "Install it with: pip install -e .   (or: pip install graphsift)" >&2
exit 1
