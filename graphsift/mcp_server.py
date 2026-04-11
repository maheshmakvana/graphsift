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
# Post-processing tools
# ---------------------------------------------------------------------------

def _tool_run_postprocess(params: dict) -> dict:
    """Run flow detection, community detection, FTS rebuild, and risk scoring on the graph."""
    from graphsift.adapters.postprocess import Postprocessor

    root = params.get("root_path", os.getcwd())
    do_flows = params.get("flows", True)
    do_communities = params.get("communities", True)
    do_fts = params.get("fts", True)
    do_risk = params.get("risk", True)

    builder, source_map = _get_builder(root)
    if not source_map:
        return {"error": "Graph not built yet. Call build_graph first."}

    graph = getattr(builder, "_graph", None)
    if graph is None:
        return {"error": "No graph available. Call build_graph first."}

    store = _get_store(root)
    pp = Postprocessor()
    result = pp.run(graph, store, source_map, flows=do_flows, communities=do_communities, risk=do_risk, fts=do_fts)
    return {"status": "done", **result}


def _tool_detect_changes(params: dict) -> dict:
    """Detect changed files and return risk-scored impact analysis."""
    from graphsift.adapters.postprocess import RiskScorer

    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])
    max_depth = int(params.get("max_depth", 2))
    include_source = params.get("include_source", False)
    detail_level = params.get("detail_level", "standard")

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph or not source_map:
        return {"error": "Graph not built yet. Call build_graph first."}

    store = _get_store(root)

    # Get blast radius
    scores = graph.ranked_neighbors(seed_paths=changed_files, include_dynamic=True)
    affected = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:50]

    # Risk from store
    risk_rows = store.load_risk_index(min_score=0.0)
    risk_by_path = {r["file_path"]: r["risk_score"] for r in risk_rows}

    result_files = []
    for fp, (score, depth, reasons) in affected:
        entry: dict[str, Any] = {
            "file": fp,
            "score": round(score, 3),
            "depth": depth,
            "reasons": reasons,
            "risk_score": round(risk_by_path.get(fp, 0.0), 3),
        }
        if include_source and detail_level == "standard":
            entry["source_preview"] = (source_map.get(fp, "")[:500] + "...") if source_map.get(fp) else ""
        result_files.append(entry)

    # Summary risk score = max risk among changed files
    max_risk = max((risk_by_path.get(f, 0.0) for f in changed_files), default=0.0)

    return {
        "changed_files": changed_files,
        "affected_count": len(affected),
        "max_risk_score": round(max_risk, 3),
        "affected_files": result_files,
    }


def _tool_query_graph(params: dict) -> dict:
    """Run predefined graph queries: callers_of, callees_of, imports_of, importers_of, tests_for, file_summary."""
    from graphsift.models import EdgeKind, NodeKind

    root = params.get("root_path", os.getcwd())
    pattern = params.get("pattern", "")
    target = params.get("target", "")
    limit = int(params.get("limit", 20))
    detail_level = params.get("detail_level", "standard")

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph:
        return {"error": "Graph not built yet. Call build_graph first."}

    with graph._lock:
        nodes = dict(graph._nodes)
        adj_out = {k: list(v) for k, v in graph._adj_out.items()}
        adj_in = {k: list(v) for k, v in graph._adj_in.items()}
        file_nodes = dict(graph._file_nodes)

    target_lower = target.lower()

    def _match_node(n: Any) -> bool:
        return (
            target_lower in n.name.lower()
            or target_lower in n.qualified_name.lower()
            or target_lower in n.file_path.lower()
        )

    matched = [n for n in nodes.values() if _match_node(n)][:5]

    results = []
    for seed in matched:
        nid = seed.node_id
        if pattern == "callers_of":
            hits = [nodes[e.source_id] for e in adj_in.get(nid, []) if e.source_id in nodes]
        elif pattern == "callees_of":
            hits = [nodes[e.target_id] for e in adj_out.get(nid, []) if e.target_id in nodes]
        elif pattern == "imports_of":
            hits = [nodes[e.target_id] for e in adj_out.get(nid, [])
                    if e.kind.value == "imports" and e.target_id in nodes]
        elif pattern == "importers_of":
            hits = [nodes[e.source_id] for e in adj_in.get(nid, [])
                    if e.kind.value == "imports" and e.source_id in nodes]
        elif pattern == "tests_for":
            hits = [nodes[e.source_id] for e in adj_in.get(nid, [])
                    if "test" in nodes.get(e.source_id, type("", (), {"file_path": ""})()).file_path.lower()
                    and e.source_id in nodes]
        elif pattern == "file_summary":
            fn = file_nodes.get(seed.file_path)
            return {
                "pattern": pattern,
                "target": target,
                "file": seed.file_path,
                "language": fn.language.value if fn else "unknown",
                "symbols": len(fn.symbols) if fn else 0,
                "token_estimate": fn.token_estimate if fn else 0,
            }
        elif pattern == "children_of":
            hits = [nodes[e.target_id] for e in adj_out.get(nid, []) if e.target_id in nodes]
        elif pattern == "inheritors_of":
            hits = [nodes[e.source_id] for e in adj_in.get(nid, [])
                    if e.kind.value == "inherits" and e.source_id in nodes]
        else:
            return {"error": f"Unknown pattern: {pattern}. Valid: callers_of, callees_of, imports_of, importers_of, tests_for, children_of, inheritors_of, file_summary"}

        for h in hits[:limit]:
            entry: dict[str, Any] = {
                "name": h.name,
                "qualified_name": h.qualified_name,
                "kind": h.kind.value,
                "file": h.file_path,
                "line": h.line_start,
            }
            results.append(entry)

    return {
        "pattern": pattern,
        "target": target,
        "results": results,
        "total": len(results),
    }


def _tool_list_flows(params: dict) -> dict:
    """List execution flows sorted by criticality."""
    root = params.get("root_path", os.getcwd())
    limit = int(params.get("limit", 50))
    sort_by = params.get("sort_by", "criticality")
    detail_level = params.get("detail_level", "standard")

    store = _get_store(root)
    with store._lock:
        try:
            rows = store._conn.execute(
                "SELECT * FROM flow_snapshots ORDER BY id DESC LIMIT ?", (limit * 2,)
            ).fetchall()
        except Exception:
            rows = []

    flows = []
    for row in rows:
        meta = json.loads(row["metadata"] or "{}")
        entry: dict[str, Any] = {
            "id": row["id"],
            "flow_name": row["flow_name"],
            "entry_point": row["entry_point"],
            "node_count": meta.get("node_count", 0),
            "file_count": meta.get("file_count", 0),
            "criticality": meta.get("criticality", 0.0),
        }
        flows.append(entry)

    # Sort
    key_map = {"criticality": "criticality", "node_count": "node_count", "file_count": "file_count", "name": "flow_name"}
    sort_key = key_map.get(sort_by, "criticality")
    flows.sort(key=lambda x: x.get(sort_key, 0), reverse=(sort_key != "flow_name"))

    return {"flows": flows[:limit], "total": len(flows)}


def _tool_get_flow(params: dict) -> dict:
    """Get detailed information about a single execution flow."""
    root = params.get("root_path", os.getcwd())
    flow_id = params.get("flow_id")
    flow_name = params.get("flow_name", "")
    include_source = params.get("include_source", False)

    store = _get_store(root)
    _, source_map = _get_builder(root)

    with store._lock:
        try:
            if flow_id is not None:
                row = store._conn.execute(
                    "SELECT * FROM flow_snapshots WHERE id=?", (flow_id,)
                ).fetchone()
            else:
                row = store._conn.execute(
                    "SELECT * FROM flow_snapshots WHERE flow_name LIKE ? LIMIT 1",
                    (f"%{flow_name}%",),
                ).fetchone()
        except Exception:
            row = None

    if not row:
        return {"error": "Flow not found."}

    nodes_json = json.loads(row["nodes_json"] or "[]")
    meta = json.loads(row["metadata"] or "{}")

    result: dict[str, Any] = {
        "id": row["id"],
        "flow_name": row["flow_name"],
        "entry_point": row["entry_point"],
        "node_count": meta.get("node_count", len(nodes_json)),
        "criticality": meta.get("criticality", 0.0),
        "nodes": nodes_json[:50],
    }

    if include_source and source_map:
        seen_files: set[str] = set()
        snippets = []
        for nid in nodes_json[:10]:
            fp = nid.split("::")[0] if "::" in nid else ""
            if fp and fp not in seen_files and fp in source_map:
                seen_files.add(fp)
                snippets.append({"file": fp, "source": source_map[fp][:300]})
        result["source_snippets"] = snippets

    return result


def _tool_get_affected_flows(params: dict) -> dict:
    """Find execution flows affected by changed files."""
    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])

    store = _get_store(root)

    with store._lock:
        try:
            rows = store._conn.execute("SELECT * FROM flow_snapshots").fetchall()
        except Exception:
            rows = []

    changed_set = set(changed_files)
    affected = []
    for row in rows:
        nodes_in_flow = json.loads(row["nodes_json"] or "[]")
        flow_files = {nid.split("::")[0] for nid in nodes_in_flow if "::" in nid}
        if flow_files & changed_set:
            meta = json.loads(row["metadata"] or "{}")
            affected.append({
                "id": row["id"],
                "flow_name": row["flow_name"],
                "entry_point": row["entry_point"],
                "criticality": meta.get("criticality", 0.0),
                "overlapping_files": list(flow_files & changed_set),
            })

    affected.sort(key=lambda x: -x["criticality"])
    return {"changed_files": changed_files, "affected_flows": affected, "total": len(affected)}


def _tool_list_communities(params: dict) -> dict:
    """List detected code communities."""
    root = params.get("root_path", os.getcwd())
    sort_by = params.get("sort_by", "size")
    min_size = int(params.get("min_size", 0))
    limit = int(params.get("limit", 50))
    detail_level = params.get("detail_level", "standard")

    store = _get_store(root)
    communities = store.load_communities()

    if min_size > 0:
        communities = [c for c in communities if c["node_count"] >= min_size]

    if sort_by == "name":
        communities.sort(key=lambda x: x["label"])
    else:
        communities.sort(key=lambda x: -x["node_count"])

    return {"communities": communities[:limit], "total": len(communities)}


def _tool_get_community(params: dict) -> dict:
    """Get detailed information about a single code community."""
    root = params.get("root_path", os.getcwd())
    community_name = params.get("community_name", "")
    community_id = params.get("community_id")
    include_members = params.get("include_members", False)

    store = _get_store(root)
    communities = store.load_communities()

    if community_id is not None:
        found = next((c for c in communities if c["community_id"] == community_id), None)
    else:
        name_lower = community_name.lower()
        found = next((c for c in communities if name_lower in c["label"].lower()), None)

    if not found:
        return {"error": "Community not found."}

    result: dict[str, Any] = {
        "community_id": found["community_id"],
        "label": found["label"],
        "node_count": found["node_count"],
    }
    if include_members:
        result["members"] = found.get("metadata", {}).get("members", [])

    return result


def _tool_get_architecture_overview(params: dict) -> dict:
    """Generate architecture overview based on community structure."""
    root = params.get("root_path", os.getcwd())

    store = _get_store(root)
    db_stats = store.stats()
    communities = store.load_communities()
    risk_index = store.load_risk_index(min_score=0.5)

    high_risk_files = [r["file_path"] for r in risk_index[:10]]

    overview = {
        "total_nodes": db_stats.get("nodes", 0),
        "total_edges": db_stats.get("edges", 0),
        "total_files": db_stats.get("files", 0),
        "total_communities": len(communities),
        "schema_version": db_stats.get("schema_version", 0),
        "communities": [
            {"id": c["community_id"], "label": c["label"], "size": c["node_count"]}
            for c in communities[:20]
        ],
        "high_risk_files": high_risk_files,
        "db_path": db_stats.get("db_path", ""),
    }
    return overview


def _tool_refactor(params: dict) -> dict:
    """Rename preview or dead-code detection across the graph."""
    from graphsift.adapters.postprocess import RefactorEngine

    root = params.get("root_path", os.getcwd())
    mode = params.get("mode", "rename")

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph:
        return {"error": "Graph not built yet. Call build_graph first."}

    engine = RefactorEngine()

    if mode == "rename":
        old_name = params.get("old_name", "")
        new_name = params.get("new_name", "")
        if not old_name or not new_name:
            return {"error": "old_name and new_name required for rename mode."}
        return engine.rename_preview(graph, old_name, new_name)

    elif mode == "dead_code":
        kind = params.get("kind")
        file_pattern = params.get("file_pattern")
        limit = int(params.get("limit", 50))
        dead = engine.find_dead_code(graph, kind=kind, file_pattern=file_pattern, limit=limit)
        return {"mode": "dead_code", "results": dead, "total": len(dead)}

    elif mode == "suggest":
        dead = engine.find_dead_code(graph, limit=10)
        return {
            "mode": "suggest",
            "suggestions": [
                f"Consider removing unused {d['kind']} '{d['name']}' in {d['file_path']}:{d['line_start']}"
                for d in dead[:10]
            ],
        }

    return {"error": f"Unknown mode: {mode}. Valid: rename, dead_code, suggest"}


def _tool_apply_refactor(params: dict) -> dict:
    """Apply a previously previewed rename to source files."""
    from graphsift.adapters.postprocess import RefactorEngine

    root = params.get("root_path", os.getcwd())
    refactor_id = params.get("refactor_id", "")
    if not refactor_id:
        return {"error": "refactor_id is required."}

    engine = RefactorEngine()
    return engine.apply_rename(refactor_id, root)


def _tool_generate_wiki(params: dict) -> dict:
    """Generate markdown wiki pages from community structure into .graphsift/wiki/."""
    from graphsift.adapters.postprocess import WikiGenerator

    root = params.get("root_path", os.getcwd())
    force = params.get("force", False)

    store = _get_store(root)
    communities = store.load_communities()
    risk_index = store.load_risk_index()

    if not communities:
        return {"error": "No communities found. Run run_postprocess first."}

    wiki_dir = str(Path(root) / ".graphsift" / "wiki")
    gen = WikiGenerator(wiki_dir)
    counts = gen.generate(communities, risk_index, force=force)
    return {"wiki_dir": wiki_dir, **counts}


def _tool_get_wiki_page(params: dict) -> dict:
    """Retrieve a specific wiki page by community name."""
    from graphsift.adapters.postprocess import WikiGenerator

    root = params.get("root_path", os.getcwd())
    community_name = params.get("community_name", "")

    wiki_dir = str(Path(root) / ".graphsift" / "wiki")
    gen = WikiGenerator(wiki_dir)
    content = gen.get_page(community_name)

    if content is None:
        return {"error": f"Wiki page not found for '{community_name}'. Run generate_wiki first."}
    return {"community_name": community_name, "content": content}


def _tool_semantic_search_nodes(params: dict) -> dict:
    """Search for code symbols by name, keyword, or file path."""
    root = params.get("root_path", os.getcwd())
    query = params.get("query", "")
    kind = params.get("kind")
    limit = int(params.get("limit", 20))

    store = _get_store(root)

    # Try FTS5 first, fall back to LIKE
    nodes = store.search_nodes(query, limit=limit * 2)

    if kind:
        kind_lower = kind.lower()
        nodes = [n for n in nodes if n.kind.value == kind_lower]

    results = [
        {
            "name": n.name,
            "qualified_name": n.qualified_name,
            "kind": n.kind.value,
            "file": n.file_path,
            "line": n.line_start,
            "language": n.language.value,
            "community_id": n.community_id,
        }
        for n in nodes[:limit]
    ]

    return {"query": query, "results": results, "total": len(results)}


def _tool_list_repos(params: dict) -> dict:
    """List all registered repositories in the graphsift registry."""
    registry_path = Path.home() / ".graphsift" / "registry.json"
    if not registry_path.exists():
        return {"status": "ok", "summary": "0 registered repository(ies)", "repos": []}

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        registry = {}

    repos = [
        {"root": root, "name": info.get("name", Path(root).name), "db_path": info.get("db_path", "")}
        for root, info in registry.items()
    ]
    return {
        "status": "ok",
        "summary": f"{len(repos)} registered repository(ies)",
        "repos": repos,
    }


def _compact(obj: Any, detail_level: str) -> Any:
    """Strip verbose keys when detail_level == 'minimal'."""
    if detail_level != "minimal" or not isinstance(obj, dict):
        return obj
    DROP = {"source_preview", "source", "rendered_context", "nodes", "metadata"}
    return {k: v for k, v in obj.items() if k not in DROP}


def _tool_get_review_context(params: dict) -> dict:
    """Return token-efficient source snippets for changed files + their key dependents.

    Unlike get_context (which returns a large rendered blob), this returns a
    structured list of file snippets capped by *lines_per_file* — ideal for
    passing individual snippets into a review prompt without blowing the budget.
    """
    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])
    query = params.get("query", "")
    max_depth = int(params.get("max_depth", 2))
    lines_per_file = int(params.get("lines_per_file", 120))
    detail_level = params.get("detail_level", "standard")
    include_signatures_only = params.get("include_signatures_only", False)

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph or not source_map:
        return {"error": "Graph not built yet. Call build_graph first.", "snippets": []}

    # Blast radius (scored)
    scores = graph.ranked_neighbors(seed_paths=changed_files, include_dynamic=True)
    affected = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    # Include changed files at the top (score=1.0)
    ordered: list[tuple[str, float, int, list[str]]] = []
    seen: set[str] = set()
    for cf in changed_files:
        if cf in source_map and cf not in seen:
            ordered.append((cf, 1.0, 0, ["changed"]))
            seen.add(cf)
    for fp, (score, depth, reasons) in affected:
        if fp not in seen and depth <= max_depth and fp in source_map:
            ordered.append((fp, score, depth, reasons))
            seen.add(fp)
        if len(ordered) >= 30:
            break

    from graphsift.core import estimate_tokens

    snippets = []
    total_tokens = 0
    for fp, score, depth, reasons in ordered:
        src = source_map.get(fp, "")
        if not src:
            continue

        if include_signatures_only or detail_level == "minimal":
            # Extract only def/class lines (signatures)
            lines = [
                ln for ln in src.splitlines()
                if ln.lstrip().startswith(("def ", "async def ", "class ", "func ", "fn "))
                or ln.startswith(("export ", "module ", "pub fn ", "interface "))
            ]
            body = "\n".join(lines[:lines_per_file])
        else:
            body_lines = src.splitlines()[:lines_per_file]
            body = "\n".join(body_lines)
            if len(src.splitlines()) > lines_per_file:
                body += f"\n... ({len(src.splitlines()) - lines_per_file} more lines)"

        tok = estimate_tokens(body)
        total_tokens += tok
        entry: dict[str, Any] = {
            "file": fp,
            "score": round(score, 3),
            "depth": depth,
            "tokens": tok,
            "source": body,
        }
        if detail_level == "standard":
            entry["reasons"] = reasons
        snippets.append(entry)

    return {
        "changed_files": changed_files,
        "query": query,
        "total_snippets": len(snippets),
        "total_tokens": total_tokens,
        "snippets": snippets,
    }


def _tool_get_impact_radius(params: dict) -> dict:
    """Return blast radius as a compact scored list — token-efficient alternative to get_impact.

    Returns file paths, scores, depth, and reason tags only (no source).
    Use detect_changes for full risk analysis with source previews.
    """
    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])
    max_depth = int(params.get("max_depth", 3))
    min_score = float(params.get("min_score", 0.0))
    limit = int(params.get("limit", 50))
    detail_level = params.get("detail_level", "standard")

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph or not source_map:
        return {"error": "Graph not built yet. Call build_graph first.", "affected_files": []}

    scores = graph.ranked_neighbors(seed_paths=changed_files, include_dynamic=True)
    affected_raw = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    affected = []
    for fp, (score, depth, reasons) in affected_raw:
        if depth > max_depth or score < min_score:
            continue
        entry: dict[str, Any] = {"file": fp, "score": round(score, 3), "depth": depth}
        if detail_level == "standard":
            entry["reasons"] = reasons
        affected.append(entry)
        if len(affected) >= limit:
            break

    return {
        "changed_files": changed_files,
        "affected_count": len(affected),
        "total_in_graph": len(affected_raw),
        "affected_files": affected,
    }


def _tool_list_graph_stats(params: dict) -> dict:
    """Return compact graph statistics — one-line summary of the repo's graph state.

    Token cost: ~100 tokens. Use instead of graph_status when you only need counts.
    """
    root = params.get("root_path", os.getcwd())
    builder, source_map = _get_builder(root)

    db_stats: dict[str, Any] = {}
    try:
        store = _get_store(root)
        db_stats = store.stats()
    except Exception:
        pass

    nodes = db_stats.get("nodes", 0)
    edges = db_stats.get("edges", 0)
    files = db_stats.get("files", 0)
    schema_v = db_stats.get("schema_version", 0)
    src_files = len(source_map)

    return {
        "summary": (
            f"Full build: {src_files} files, {nodes} nodes, {edges} edges "
            f"(schema_version={schema_v})"
        ),
        "files_in_source_map": src_files,
        "nodes": nodes,
        "edges": edges,
        "files_in_db": files,
        "schema_version": schema_v,
        "status": "ready" if source_map else "empty",
    }


def _tool_get_docs_section(params: dict) -> dict:
    """Retrieve a section from a generated wiki page by heading keyword.

    Returns only the matched section (not the entire page) to save tokens.
    Falls back to the full page if heading is not found.
    """
    root = params.get("root_path", os.getcwd())
    community_name = params.get("community_name", "")
    heading = params.get("heading", "").lower()
    max_chars = int(params.get("max_chars", 2000))

    from graphsift.adapters.postprocess import WikiGenerator
    wiki_dir = str(Path(root) / ".graphsift" / "wiki")
    gen = WikiGenerator(wiki_dir)
    content = gen.get_page(community_name)

    if content is None:
        return {"error": f"Wiki page not found for '{community_name}'. Run generate_wiki first."}

    if not heading:
        # Return beginning only
        snippet = content[:max_chars]
        if len(content) > max_chars:
            snippet += f"\n... ({len(content) - max_chars} more chars)"
        return {"community_name": community_name, "section": snippet, "full_length": len(content)}

    # Find heading in content (case-insensitive, markdown ## style)
    lines = content.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        if heading in line.lower() and line.startswith("#"):
            start_idx = i
            break

    if start_idx is None:
        snippet = content[:max_chars]
        return {
            "community_name": community_name,
            "heading_found": False,
            "section": snippet,
            "full_length": len(content),
        }

    # Extract until next same-level heading or end
    heading_level = len(lines[start_idx]) - len(lines[start_idx].lstrip("#"))
    section_lines = [lines[start_idx]]
    for line in lines[start_idx + 1:]:
        if line.startswith("#" * heading_level + " ") and not line.startswith("#" * (heading_level + 1)):
            break
        section_lines.append(line)

    section = "\n".join(section_lines)
    if len(section) > max_chars:
        section = section[:max_chars] + f"\n... ({len(section) - max_chars} more chars)"

    return {
        "community_name": community_name,
        "heading": heading,
        "heading_found": True,
        "section": section,
    }


def _tool_find_large_functions(params: dict) -> dict:
    """Find the largest functions/classes by line count — token-efficient dead-weight detector.

    Returns a compact ranked list. Use before sending context to an LLM to identify
    symbols worth splitting or skipping.
    """
    root = params.get("root_path", os.getcwd())
    limit = int(params.get("limit", 20))
    min_lines = int(params.get("min_lines", 30))
    kind_filter = params.get("kind")
    file_pattern = params.get("file_pattern", "")
    detail_level = params.get("detail_level", "standard")

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph:
        return {"error": "Graph not built yet. Call build_graph first.", "results": []}

    results = []
    for file_node in graph.all_files():
        if file_pattern and file_pattern not in file_node.path:
            continue
        for sym in file_node.symbols:
            if not hasattr(sym, "line_start") or not hasattr(sym, "line_end"):
                continue
            line_count = max(0, sym.line_end - sym.line_start)
            if line_count < min_lines:
                continue
            if kind_filter and hasattr(sym, "kind") and sym.kind.value != kind_filter.lower():
                continue
            entry: dict[str, Any] = {
                "name": sym.name if hasattr(sym, "name") else str(sym),
                "file": file_node.path,
                "line_start": sym.line_start,
                "line_end": sym.line_end,
                "line_count": line_count,
                "kind": sym.kind.value if hasattr(sym, "kind") else "unknown",
            }
            if detail_level == "standard" and hasattr(sym, "signature") and sym.signature:
                entry["signature"] = sym.signature
            results.append(entry)

    results.sort(key=lambda x: -x["line_count"])
    return {
        "total_found": len(results),
        "results": results[:limit],
    }


def _tool_embed_graph(params: dict) -> dict:
    """Compute and store lightweight TF-IDF-style symbol embeddings in SQLite.

    No external ML deps required — uses bag-of-words over symbol names and
    signatures. Enables ranked semantic search via semantic_search_nodes.
    Returns a summary of what was embedded.
    """
    root = params.get("root_path", os.getcwd())
    force = params.get("force", False)

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph:
        return {"error": "Graph not built yet. Call build_graph first."}

    store = _get_store(root)

    # Check if already embedded (presence of embed_version in db meta)
    try:
        with store._lock:
            row = store._conn.execute(
                "SELECT value FROM graph_meta WHERE key='embed_version' LIMIT 1"
            ).fetchone()
            if row and not force:
                return {
                    "status": "already_embedded",
                    "embed_version": row["value"],
                    "message": "Use force=true to re-embed.",
                }
    except Exception:
        pass

    # Ensure graph_meta table exists (v7 migration covers this; handle gracefully)
    try:
        with store._lock:
            store._conn.execute(
                "CREATE TABLE IF NOT EXISTS graph_meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            store._conn.commit()
    except Exception:
        pass

    # Build simple TF-IDF-like embeddings (token frequency over symbol corpus)
    import collections
    import math

    all_nodes = list(graph._nodes.values()) if hasattr(graph, "_nodes") else []
    if not all_nodes:
        all_nodes = store.load_nodes()

    # Build IDF: count docs (nodes) containing each token
    df: dict[str, int] = collections.Counter()
    doc_tokens: list[list[str]] = []
    for node in all_nodes:
        tokens = _tokenize_symbol(node)
        doc_tokens.append(tokens)
        for t in set(tokens):
            df[t] += 1

    N = max(len(all_nodes), 1)
    embedded = 0

    with store._lock:
        for node, tokens in zip(all_nodes, doc_tokens):
            if not tokens:
                continue
            tf: dict[str, float] = collections.Counter(tokens)
            # TF-IDF vector (sparse, stored as JSON)
            vec = {
                t: round(
                    (tf[t] / len(tokens)) * math.log((N + 1) / (df.get(t, 0) + 1)),
                    6,
                )
                for t in tf
            }
            try:
                store._conn.execute(
                    "UPDATE nodes SET metadata=json_patch(metadata, ?) WHERE node_id=?",
                    (json.dumps({"_tfidf": vec}), node.node_id),
                )
            except Exception:
                pass
            embedded += 1

        store._conn.execute(
            "INSERT OR REPLACE INTO graph_meta(key, value) VALUES('embed_version', '1')"
        )
        store._conn.commit()

    return {
        "status": "embedded",
        "nodes_embedded": embedded,
        "vocab_size": len(df),
        "embed_version": "1",
    }


def _tokenize_symbol(node: Any) -> list[str]:
    """Split a symbol node into bag-of-words tokens."""
    import re
    text = " ".join(filter(None, [
        getattr(node, "name", ""),
        getattr(node, "qualified_name", ""),
        getattr(node, "signature", ""),
        getattr(node, "file_path", ""),
    ]))
    # Split on non-alnum, camelCase, snake_case
    tokens = re.findall(r'[a-zA-Z][a-z]*|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', text)
    return [t.lower() for t in tokens if len(t) > 1]


def _tool_cross_repo_search(params: dict) -> dict:
    """Search for code entities across all registered repositories."""
    query = params.get("query", "")
    kind = params.get("kind")
    limit = int(params.get("limit", 20))

    registry_path = Path.home() / ".graphsift" / "registry.json"
    if not registry_path.exists():
        return {"error": "No repos registered. Run: graphsift register <path>", "results": []}

    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "Could not read registry.", "results": []}

    from graphsift.adapters.storage import GraphStore
    all_results = []

    for root, info in registry.items():
        db_path = info.get("db_path")
        if not db_path or not Path(db_path).exists():
            continue
        try:
            store = GraphStore(db_path)
            nodes = store.search_nodes(query, limit=limit)
            if kind:
                nodes = [n for n in nodes if n.kind.value == kind.lower()]
            for n in nodes[:limit]:
                all_results.append({
                    "repo": info.get("name", Path(root).name),
                    "root": root,
                    "name": n.name,
                    "qualified_name": n.qualified_name,
                    "kind": n.kind.value,
                    "file": n.file_path,
                    "line": n.line_start,
                })
        except Exception as exc:
            logger.warning("cross_repo_search: failed for %s: %s", root, exc)

    all_results.sort(key=lambda x: x["name"])
    return {"query": query, "results": all_results[:limit * len(registry)], "total": len(all_results)}


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
        "inputSchema": {"type": "object", "properties": {"root_path": {"type": "string"}}},
    },
    "run_postprocess": {
        "fn": _tool_run_postprocess,
        "description": "Run flow detection, community detection, FTS rebuild, and risk scoring on the built graph. Call after build_graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "flows": {"type": "boolean", "description": "Run flow detection (default true)"},
                "communities": {"type": "boolean", "description": "Run community detection (default true)"},
                "fts": {"type": "boolean", "description": "Rebuild FTS index (default true)"},
                "risk": {"type": "boolean", "description": "Compute risk scores (default true)"},
            },
        },
    },
    "detect_changes": {
        "fn": _tool_detect_changes,
        "description": "Detect changed files and return risk-scored impact analysis with blast radius.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
                "max_depth": {"type": "integer", "description": "BFS depth (default 2)"},
                "include_source": {"type": "boolean"},
                "detail_level": {"type": "string", "enum": ["standard", "minimal"]},
            },
        },
    },
    "query_graph": {
        "fn": _tool_query_graph,
        "description": "Run predefined graph queries: callers_of, callees_of, imports_of, importers_of, tests_for, children_of, inheritors_of, file_summary.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "pattern": {"type": "string", "description": "Query pattern: callers_of | callees_of | imports_of | importers_of | tests_for | children_of | inheritors_of | file_summary"},
                "target": {"type": "string", "description": "Symbol name, qualified name, or file path to query"},
                "limit": {"type": "integer"},
                "detail_level": {"type": "string", "enum": ["standard", "minimal"]},
            },
            "required": ["pattern", "target"],
        },
    },
    "list_flows": {
        "fn": _tool_list_flows,
        "description": "List detected execution flows sorted by criticality. Run run_postprocess first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "sort_by": {"type": "string", "enum": ["criticality", "node_count", "file_count", "name"]},
                "limit": {"type": "integer"},
                "detail_level": {"type": "string", "enum": ["standard", "minimal"]},
            },
        },
    },
    "get_flow": {
        "fn": _tool_get_flow,
        "description": "Get detailed information about a single execution flow including call path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "flow_id": {"type": "integer"},
                "flow_name": {"type": "string", "description": "Partial name match (used if flow_id omitted)"},
                "include_source": {"type": "boolean"},
            },
        },
    },
    "get_affected_flows": {
        "fn": _tool_get_affected_flows,
        "description": "Find execution flows that pass through changed files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "list_communities": {
        "fn": _tool_list_communities,
        "description": "List detected code communities sorted by size. Run run_postprocess first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "sort_by": {"type": "string", "enum": ["size", "name"]},
                "min_size": {"type": "integer"},
                "limit": {"type": "integer"},
                "detail_level": {"type": "string", "enum": ["standard", "minimal"]},
            },
        },
    },
    "get_community": {
        "fn": _tool_get_community,
        "description": "Get details about a single code community including members.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "community_name": {"type": "string"},
                "community_id": {"type": "integer"},
                "include_members": {"type": "boolean"},
            },
        },
    },
    "get_architecture_overview": {
        "fn": _tool_get_architecture_overview,
        "description": "Generate architecture overview: communities, risk files, total nodes/edges/files.",
        "inputSchema": {
            "type": "object",
            "properties": {"root_path": {"type": "string"}},
        },
    },
    "refactor": {
        "fn": _tool_refactor,
        "description": "Rename preview, dead-code detection, or suggestions. mode: rename | dead_code | suggest.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "mode": {"type": "string", "enum": ["rename", "dead_code", "suggest"]},
                "old_name": {"type": "string", "description": "For rename mode"},
                "new_name": {"type": "string", "description": "For rename mode"},
                "kind": {"type": "string", "description": "For dead_code: function | class | method"},
                "file_pattern": {"type": "string"},
            },
        },
    },
    "apply_refactor": {
        "fn": _tool_apply_refactor,
        "description": "Apply a previously previewed rename to source files. All edits validated to be within repo root.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "refactor_id": {"type": "string", "description": "ID from prior refactor(mode=rename) call"},
            },
            "required": ["refactor_id"],
        },
    },
    "generate_wiki": {
        "fn": _tool_generate_wiki,
        "description": "Generate markdown wiki pages from community structure into .graphsift/wiki/.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "force": {"type": "boolean", "description": "Regenerate all pages even if unchanged"},
            },
        },
    },
    "get_wiki_page": {
        "fn": _tool_get_wiki_page,
        "description": "Get a specific wiki page by community name. Run generate_wiki first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "community_name": {"type": "string", "description": "Community name (partial match)"},
            },
            "required": ["community_name"],
        },
    },
    "semantic_search_nodes": {
        "fn": _tool_semantic_search_nodes,
        "description": "Search for code symbols (functions, classes, modules) by name or keyword. Uses FTS5 when available.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "query": {"type": "string"},
                "kind": {"type": "string", "description": "Filter by kind: function | class | method | module"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    "list_repos": {
        "fn": _tool_list_repos,
        "description": "List all repositories registered in the graphsift registry (~/.graphsift/registry.json).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "cross_repo_search": {
        "fn": _tool_cross_repo_search,
        "description": "Search for code entities across all registered repositories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "kind": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    # ---- token-saving tools (new) ----
    "get_review_context": {
        "fn": _tool_get_review_context,
        "description": (
            "Token-efficient code review context. Returns structured source snippets "
            "for changed files + key dependents (capped by lines_per_file). "
            "~5-10x fewer tokens than get_context. Use for focused review prompts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
                "query": {"type": "string"},
                "max_depth": {"type": "integer", "description": "Graph traversal depth (default 2)"},
                "lines_per_file": {"type": "integer", "description": "Max lines per file snippet (default 120)"},
                "include_signatures_only": {"type": "boolean", "description": "Return only def/class lines (default false)"},
                "detail_level": {"type": "string", "enum": ["standard", "minimal"]},
            },
        },
    },
    "get_impact_radius": {
        "fn": _tool_get_impact_radius,
        "description": (
            "Compact blast-radius analysis — file paths + scores + depth only, no source. "
            "~10x fewer tokens than detect_changes. Use for quick impact checks."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "changed_files": {"type": "array", "items": {"type": "string"}},
                "max_depth": {"type": "integer", "description": "Max BFS depth (default 3)"},
                "min_score": {"type": "number", "description": "Minimum relevance score 0-1 (default 0.0)"},
                "limit": {"type": "integer", "description": "Max results (default 50)"},
                "detail_level": {"type": "string", "enum": ["standard", "minimal"]},
            },
        },
    },
    "list_graph_stats": {
        "fn": _tool_list_graph_stats,
        "description": (
            "Ultra-compact graph statistics (~100 tokens). "
            "Returns node/edge/file counts and schema version as a one-line summary. "
            "Use instead of graph_status when you only need counts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"root_path": {"type": "string"}},
        },
    },
    "get_docs_section": {
        "fn": _tool_get_docs_section,
        "description": (
            "Fetch a single section from a community wiki page by heading keyword. "
            "Returns only the matched heading block — far fewer tokens than get_wiki_page. "
            "Run generate_wiki first."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "community_name": {"type": "string", "description": "Community name (partial match)"},
                "heading": {"type": "string", "description": "Heading keyword to locate (case-insensitive)"},
                "max_chars": {"type": "integer", "description": "Max characters to return (default 2000)"},
            },
            "required": ["community_name"],
        },
    },
    "find_large_functions": {
        "fn": _tool_find_large_functions,
        "description": (
            "Find the largest functions/classes by line count. "
            "Compact output — name, file, line range, size. "
            "Useful for identifying bloat before sending context to an LLM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "limit": {"type": "integer", "description": "Max results (default 20)"},
                "min_lines": {"type": "integer", "description": "Min line count threshold (default 30)"},
                "kind": {"type": "string", "description": "Filter by kind: function | class | method"},
                "file_pattern": {"type": "string", "description": "Filter by file path substring"},
                "detail_level": {"type": "string", "enum": ["standard", "minimal"]},
            },
        },
    },
    "embed_graph": {
        "fn": _tool_embed_graph,
        "description": (
            "Compute TF-IDF symbol embeddings and store in SQLite. "
            "No external ML dependencies. Improves semantic_search_nodes ranking. "
            "Run once after build_graph + run_postprocess."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "force": {"type": "boolean", "description": "Re-embed even if already done (default false)"},
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
        "serverInfo": {"name": "graphsift", "version": "1.4.0"},
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
