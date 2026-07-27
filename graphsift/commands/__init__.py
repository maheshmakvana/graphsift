"""External command plugin system for graphsift.

Allows third-party analysis tools to register as graphsift commands
via manifest-driven JSON stdin/stdout protocol.

Protocol
--------
1. graphsift sends JSON to plugin's stdin::

    {"kind": "execute", "payload": {
        "arguments": {...},
        "session_dir": "...",
        "call_id": "..."
    }}

2. Plugin returns JSON on stdout::

    {"kind": "result", "payload": {
        "call_id": "...",
        "success": true,
        "output": "...",
        "error": null
    }}
"""

from graphsift.commands.registry import PluginManifest, PluginRegistry

__all__ = ["PluginManifest", "PluginRegistry"]
