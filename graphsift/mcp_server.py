"""graphsift MCP server — exposes graphsift tools to Claude Code via stdio."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal MCP stdio server (no external dep — pure stdlib)
# Protocol: https://spec.modelcontextprotocol.io/specification/
# ---------------------------------------------------------------------------

_JSONRPC = "2.0"


def _send(obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _ok(req_id: Any, result: Any) -> None:
    _send({"jsonrpc": _JSONRPC, "id": req_id, "result": result})


def _err(req_id: Any, code: int, message: str) -> None:
    _send({"jsonrpc": _JSONRPC, "id": req_id, "error": {"code": code, "message": message}})


# ---------------------------------------------------------------------------
# SQLite store — one DB per repo, stored at ~/.graphsift/<repo_hash>/graph.db
# ---------------------------------------------------------------------------

_store_lock = threading.RLock()
_stores: dict[str, Any] = {}  # root_path -> GraphStore


def _db_path_for(root: str) -> str:
    """Compute the DB path for a given repo root."""
    key = hashlib.sha1(root.encode()).hexdigest()[:12]
    home = Path.home() / ".graphsift" / key
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "graph.db")


def _get_store(root: str) -> Any:
    """Return (creating if absent) the GraphStore for *root*.

    Runs SQLite migrations on first open — migration progress is logged to
    stderr so the caller sees the same INFO lines as code-review-graph.
    """
    from graphsift.adapters.storage import GraphStore

    with _store_lock:
        if root not in _stores:
            db_path = _db_path_for(root)
            _stores[root] = GraphStore(db_path)
        return _stores[root]


# ---------------------------------------------------------------------------
# Graphsift state — one builder per working directory
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_builders: dict[str, Any] = {}   # root_path -> ContextBuilder
_source_maps: dict[str, dict[str, str]] = {}  # root_path -> source_map


def _get_builder(root: str) -> tuple[Any, dict[str, str]]:
    """Return (builder, source_map) for *root*, creating if absent."""
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    with _lock:
        if root not in _builders:
            _builders[root] = ContextBuilder(ContextConfig())
            _source_maps[root] = {}
        return _builders[root], _source_maps[root]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_build_graph(params: dict) -> dict:
    """Index all source files under root_path and build the dependency graph."""
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.core import ContextBuilder, estimate_tokens
    from graphsift.models import ContextConfig, FileNode, GraphEdge, GraphNode

    root = params.get("root_path", os.getcwd())
    extensions_raw = params.get("extensions")
    extensions = set(extensions_raw) if extensions_raw else {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"
    }
    exclude_dirs = set(params.get("exclude_dirs", [
        "venv", ".venv", "node_modules", ".git", "__pycache__",
        "dist", "build", ".mypy_cache", ".pytest_cache",
    ]))
    progress_interval = int(params.get("progress_interval", 200))

    # -- Ensure SQLite DB is open and migrated (logs migration steps to stderr)
    store = _get_store(root)

    source_map = load_source_map(root, extensions=extensions, exclude_dirs=exclude_dirs)
    total_files = len(source_map)

    with _lock:
        from graphsift.models import ContextConfig
        _builders[root] = ContextBuilder(ContextConfig())
        builder = _builders[root]
        _source_maps[root] = source_map

    # -- Index with per-file progress logging
    all_paths = list(source_map.keys())
    parsed_count = 0
    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    all_file_nodes: list[FileNode] = []

    for path in all_paths:
        source = source_map[path]
        try:
            with _lock:
                builder.index_file(path, source)
        except Exception as exc:  # noqa: BLE001
            logger.debug("build_graph: skipped %s: %s", path, exc)

        parsed_count += 1
        if progress_interval > 0 and parsed_count % progress_interval == 0:
            logger.info(
                "INFO: Progress: %d/%d files parsed", parsed_count, total_files
            )

    logger.info("INFO: Progress: %d/%d files parsed", total_files, total_files)

    # -- Gather stats from builder graph
    with _lock:
        stats = builder.index_files(source_map)
        graph = getattr(builder, "_graph", None)

    # -- Persist nodes + edges + files to SQLite
    if graph is not None:
        try:
            # Collect nodes
            from graphsift.models import NodeKind as _NodeKind
            for file_node in graph.all_files():
                all_file_nodes.append(file_node)
                for sym in file_node.symbols:
                    if hasattr(sym, "node_id"):
                        all_nodes.append(sym)
                    else:
                        all_nodes.append(
                            GraphNode(
                                node_id=f"{file_node.path}::{sym}",
                                file_path=file_node.path,
                                kind=_NodeKind.FUNCTION,
                                name=str(sym),
                                qualified_name=str(sym),
                                language=file_node.language,
                            )
                        )
            store.save_nodes(all_nodes)
            store.save_files(all_file_nodes)
            logger.info(
                "INFO: Persisted %d nodes, %d files to SQLite",
                len(all_nodes), len(all_file_nodes),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("build_graph: SQLite persist failed: %s", exc)

    return {
        "status": "indexed",
        "root": root,
        "files_indexed": stats.files_indexed,
        "files_skipped": stats.files_skipped,
        "symbols_extracted": stats.symbols_extracted,
        "edges_created": stats.edges_created,
        "duration_ms": stats.duration_ms,
        "db_path": _db_path_for(root),
    }


def _tool_update_graph(params: dict) -> dict:
    """Incrementally update the graph with changed files only."""
    from graphsift.adapters.filesystem import load_changed_files

    root = params.get("root_path", os.getcwd())
    changed = params.get("changed_files", [])

    if not changed:
        return {"status": "no_changes", "files_updated": 0}

    builder, source_map = _get_builder(root)
    new_sources = load_changed_files(changed)

    with _lock:
        source_map.update(new_sources)
        for path, source in new_sources.items():
            try:
                builder.index_file(path, source)
            except Exception as exc:  # noqa: BLE001
                logger.warning("update_graph: skipped %s: %s", path, exc)

    return {"status": "updated", "files_updated": len(new_sources)}


def _tool_get_context(params: dict) -> dict:
    """Build ranked context for a code diff / query."""
    from graphsift.models import ContextConfig, DiffSpec
    from graphsift.core import ContextBuilder

    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])
    query = params.get("query", "")
    token_budget = int(params.get("token_budget", 60_000))
    diff_text = params.get("diff_text", "")
    commit_message = params.get("commit_message", "")

    builder, source_map = _get_builder(root)

    if not source_map:
        return {
            "error": "Graph not built yet. Call build_graph first.",
            "rendered_context": "",
            "files_selected": 0,
            "token_savings_pct": 0,
        }

    from graphsift.models import ContextConfig
    builder_fresh = ContextBuilder(ContextConfig(token_budget=token_budget))
    builder_fresh.index_files(source_map)

    diff = DiffSpec(
        changed_files=changed_files,
        query=query,
        diff_text=diff_text,
        commit_message=commit_message,
    )
    result = builder_fresh.build(diff, source_map)

    return {
        "rendered_context": result.rendered_context,
        "files_selected": result.files_selected,
        "files_scanned": result.files_scanned,
        "total_original_tokens": result.total_original_tokens,
        "total_rendered_tokens": result.total_rendered_tokens,
        "reduction_ratio": round(result.reduction_ratio, 3),
        "token_savings_pct": round((1 - result.reduction_ratio) * 100, 1),
    }


def _tool_get_impact(params: dict) -> dict:
    """Return the blast radius (affected files) for a set of changed files."""
    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])
    max_depth = int(params.get("max_depth", 3))

    builder, source_map = _get_builder(root)
    graph = builder.graph if hasattr(builder, "graph") else None

    if not graph or not source_map:
        return {"error": "Graph not built yet.", "affected_files": []}

    scores = graph.ranked_neighbors(
        seed_paths=changed_files,
        include_dynamic=True,
    )
    affected = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {
        "changed_files": changed_files,
        "affected_files": [
            {"path": p, "score": round(s, 3)} for p, s in affected[:50]
        ],
        "total_affected": len(affected),
    }


def _tool_graph_status(params: dict) -> dict:
    """Return current graph statistics including SQLite DB stats."""
    root = params.get("root_path", os.getcwd())
    builder, source_map = _get_builder(root)

    db_stats: dict[str, Any] = {}
    try:
        store = _get_store(root)
        db_stats = store.stats()
    except Exception:
        pass

    if not source_map:
        return {
            "status": "empty",
            "message": "No graph built yet. Run build_graph.",
            "db": db_stats,
        }

    stats = builder.graph_stats() if hasattr(builder, "graph_stats") else {}
    return {
        "status": "ready",
        "root": root,
        "files_in_source_map": len(source_map),
        "db": db_stats,
        **stats,
    }


def _tool_search_symbols(params: dict) -> dict:
    """Search for symbols (functions/classes) matching a query string."""
    root = params.get("root_path", os.getcwd())
    query = params.get("query", "").lower()
    limit = int(params.get("limit", 20))

    builder, source_map = _get_builder(root)
    graph = builder.graph if hasattr(builder, "graph") else None

    if not graph:
        return {"error": "Graph not built yet.", "symbols": []}

    results = []
    for file_node in graph.all_files():
        for sym in file_node.symbols:
            if query in sym.lower() or query in file_node.path.lower():
                results.append({"symbol": sym, "file": file_node.path})
                if len(results) >= limit:
                    break
        if len(results) >= limit:
            break

    return {"query": query, "symbols": results, "total": len(results)}


def _tool_list_files(params: dict) -> dict:
    """List all indexed files with their token estimates."""
    root = params.get("root_path", os.getcwd())
    builder, source_map = _get_builder(root)
    graph = builder.graph if hasattr(builder, "graph") else None

    if not graph:
        return {"error": "Graph not built yet.", "files": []}

    files = [
        {
            "path": f.path,
            "language": f.language.value if hasattr(f.language, "value") else str(f.language),
            "token_estimate": f.token_estimate,
            "symbols": len(f.symbols),
        }
        for f in graph.all_files()
    ]
    files.sort(key=lambda x: x["token_estimate"], reverse=True)
    return {"files": files[:100], "total_files": len(files)}


def _tool_get_file_context(params: dict) -> dict:
    """Return the source of a specific file from the indexed source map."""
    root = params.get("root_path", os.getcwd())
    file_path = params.get("file_path", "")

    _, source_map = _get_builder(root)
    source = source_map.get(file_path)
    if source is None:
        # Try relative match
        for k in source_map:
            if k.endswith(file_path) or file_path.endswith(k):
                source = source_map[k]
                file_path = k
                break

    if source is None:
        return {"error": f"File not found in index: {file_path}"}

    from graphsift.core import estimate_tokens
    return {
        "file_path": file_path,
        "source": source,
        "token_estimate": estimate_tokens(source),
        "lines": source.count("\n") + 1,
    }


def _tool_minimal_context(params: dict) -> dict:
    """Ultra-minimal context — just file paths + signatures, no source bodies."""
    from graphsift.models import ContextConfig, DiffSpec, OutputMode
    from graphsift.core import ContextBuilder

    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])
    query = params.get("query", "")

    _, source_map = _get_builder(root)
    if not source_map:
        return {"error": "Graph not built yet.", "rendered_context": ""}

    builder = ContextBuilder(ContextConfig(
        token_budget=8_000,
        output_mode=OutputMode.SIGNATURES,
    ))
    builder.index_files(source_map)

    result = builder.build(
        DiffSpec(changed_files=changed_files, query=query),
        source_map,
    )
    return {
        "rendered_context": result.rendered_context,
        "files_selected": result.files_selected,
        "total_rendered_tokens": result.total_rendered_tokens,
        "token_savings_pct": round((1 - result.reduction_ratio) * 100, 1),
    }


def _tool_clear_graph(params: dict) -> dict:
    """Clear the in-memory graph for a root path (forces rebuild on next call)."""
    root = params.get("root_path", os.getcwd())
    with _lock:
        _builders.pop(root, None)
        _source_maps.pop(root, None)
    return {"status": "cleared", "root": root}


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS = {
    "build_graph": {
        "fn": _tool_build_graph,
        "description": (
            "Index all source files under root_path and build the dependency graph. "
            "Call once per session (or after large changes). "
            "Returns: files_indexed, symbols_extracted, edges_created."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root directory (default: cwd)"},
                "extensions": {"type": "array", "items": {"type": "string"}, "description": "File extensions to index (default: .py .js .ts .go .rs .java)"},
                "exclude_dirs": {"type": "array", "items": {"type": "string"}, "description": "Directories to skip"},
            },
        },
    },
    "update_graph": {
        "fn": _tool_update_graph,
        "description": (
            "Incrementally update the graph with only the changed files. "
            "Much faster than full rebuild. Called automatically by PostToolUse hook."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}, "description": "Absolute paths of changed files"},
            },
        },
    },
    "get_context": {
        "fn": _tool_get_context,
        "description": (
            "Build ranked, token-budget-aware context for a code diff or query. "
            "Returns only the most relevant files — typically 80-150x fewer tokens than sending the whole repo. "
            "Use rendered_context as the code context block in your LLM prompt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}, "description": "Files that changed"},
                "query": {"type": "string", "description": "What you want to know / review"},
                "token_budget": {"type": "integer", "description": "Max tokens to include (default 60000)"},
                "diff_text": {"type": "string", "description": "Raw unified diff text (optional)"},
                "commit_message": {"type": "string", "description": "Commit message (optional)"},
            },
        },
    },
    "get_impact": {
        "fn": _tool_get_impact,
        "description": (
            "Return the blast radius — all files potentially affected by changes to changed_files. "
            "Scored 0-1 by dependency distance. Useful for risk assessment."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
                "max_depth": {"type": "integer", "description": "Graph traversal depth (default 3)"},
            },
        },
    },
    "graph_status": {
        "fn": _tool_graph_status,
        "description": "Check if the graph is built and see current stats (files, symbols, edges).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
            },
        },
    },
    "search_symbols": {
        "fn": _tool_search_symbols,
        "description": "Search for functions, classes, or modules by name across the indexed codebase.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "query": {"type": "string", "description": "Symbol or filename substring to search"},
                "limit": {"type": "integer", "description": "Max results (default 20)"},
            },
            "required": ["query"],
        },
    },
    "list_files": {
        "fn": _tool_list_files,
        "description": "List all indexed files sorted by token count. Useful for understanding repo size.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
            },
        },
    },
    "get_file_context": {
        "fn": _tool_get_file_context,
        "description": "Retrieve the full source of a specific indexed file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "file_path": {"type": "string", "description": "Path to the file (absolute or partial match)"},
            },
            "required": ["file_path"],
        },
    },
    "minimal_context": {
        "fn": _tool_minimal_context,
        "description": (
            "Ultra-low-token context — signatures only, no bodies. "
            "Ideal for quick orientation or when token budget is tight (<8K tokens)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
                "query": {"type": "string"},
            },
        },
    },
    "clear_graph": {
        "fn": _tool_clear_graph,
        "description": "Clear the in-memory graph for root_path, forcing a full rebuild on next call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# MCP request handlers
# ---------------------------------------------------------------------------

def _handle_initialize(req_id: Any, params: dict) -> None:
    # Open the default-cwd store on startup so migrations run immediately
    # and the INFO migration lines appear in stderr (same as code-review-graph).
    try:
        _get_store(os.getcwd())
    except Exception as exc:
        logger.warning("graphsift: startup DB init failed: %s", exc)

    _ok(req_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "graphsift", "version": "1.3.0"},
    })


def _handle_tools_list(req_id: Any, params: dict) -> None:
    tools = []
    for name, spec in _TOOLS.items():
        tools.append({
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
        })
    _ok(req_id, {"tools": tools})


def _handle_tools_call(req_id: Any, params: dict) -> None:
    name = params.get("name", "")
    args = params.get("arguments", {})

    if name not in _TOOLS:
        _err(req_id, -32601, f"Unknown tool: {name}")
        return

    try:
        result = _TOOLS[name]["fn"](args)
        _ok(req_id, {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}],
        })
    except Exception as exc:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        _ok(req_id, {
            "content": [{"type": "text", "text": json.dumps({"error": str(exc)})}],
            "isError": True,
        })


_HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
}


# ---------------------------------------------------------------------------
# Main stdio loop
# ---------------------------------------------------------------------------

def run_server() -> None:
    """Run the graphsift MCP server over stdio."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )

    # Ensure stdout is line-buffered text
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        method = req.get("method", "")

        # Notifications (no id) — ignore
        if req_id is None:
            continue

        handler = _HANDLERS.get(method)
        if handler is None:
            _err(req_id, -32601, f"Method not found: {method}")
            continue

        try:
            handler(req_id, req.get("params") or {})
        except Exception as exc:  # noqa: BLE001
            logger.exception("handler %s failed", method)
            _err(req_id, -32603, str(exc))
