"""GraphSift A2A (Agent-to-Agent) Protocol server.

Exposes GraphSift code intelligence as an A2A agent that other agents can query.
Uses Python stdlib only — no Flask, FastAPI, or external dependencies.

A2A Protocol (2026): https://a2a-protocol.org

Endpoints:
  GET  /.well-known/agent-card.json  — Agent card (capabilities discovery)
  POST /                             — JSON-RPC 2.0 endpoint
  POST /tasks                        — A2A task execution
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from graphsift._version import __version__

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# A2A JSON-RPC helpers
# ---------------------------------------------------------------------------

_JSONRPC = "2.0"


def _ok(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": _JSONRPC, "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": _JSONRPC, "id": req_id, "error": err}


# ---------------------------------------------------------------------------
# Agent card builder
# ---------------------------------------------------------------------------


def build_agent_card(
    base_url: str,
    repo_count: int = 0,
    supported_languages: list[str] | None = None,
) -> dict:
    """Return an A2A-compliant AgentCard JSON describing GraphSift capabilities.

    Args:
        base_url: The server's base URL (e.g. ``http://127.0.0.1:8889``).
        repo_count: Number of registered repositories.
        supported_languages: List of language names this agent understands.

    Returns:
        AgentCard dictionary per the A2A agent-card specification.
    """
    if supported_languages is None:
        supported_languages = [
            "Python", "JavaScript", "TypeScript", "Go", "Rust",
            "Java", "C++", "C", "Ruby", "PHP", "Bash", "HCL",
        ]

    return {
        "name": "GraphSift Code Intelligence",
        "description": (
            "Codebase-aware agent providing dependency analysis, impact assessment, "
            "architecture insights, and semantic code search across one or more "
            "registered repositories. Built on a full dependency graph with "
            "ranked relevance scoring, community detection, and flow analysis."
        ),
        "url": base_url.rstrip("/"),
        "version": __version__,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateful": True,
        },
        "metadata": {
            "repo_count": repo_count,
            "supported_languages": supported_languages,
            "features": [
                "impact_analysis",
                "dependency_analysis",
                "architecture_explanation",
                "semantic_search",
                "cycle_detection",
                "dead_code_analysis",
                "flow_analysis",
                "community_detection",
                "risk_scoring",
            ],
        },
        "skills": [
            {
                "id": "analyze_impact",
                "name": "Analyze Impact",
                "description": (
                    "Analyze the blast radius and impact of file changes. "
                    "Given a set of changed files, returns all potentially affected "
                    "files ranked by relevance score (0-1), traversal depth, and "
                    "the reasons for each connection. Supports configurable depth."
                ),
                "input": {
                    "type": "object",
                    "properties": {
                        "root_path": {"type": "string", "description": "Repository root path"},
                        "changed_files": {
                            "type": "array", "items": {"type": "string"},
                            "description": "Paths of changed files",
                        },
                        "max_depth": {
                            "type": "integer", "description": "Max BFS depth (default 3)",
                        },
                    },
                    "required": ["changed_files"],
                },
            },
            {
                "id": "find_dependencies",
                "name": "Find Dependencies",
                "description": (
                    "Find all dependencies (imports, calls, inheritance) and "
                    "dependents (callers, importers) of a file or symbol. "
                    "Returns structured callers, callees, imports, and test files."
                ),
                "input": {
                    "type": "object",
                    "properties": {
                        "root_path": {"type": "string", "description": "Repository root path"},
                        "target": {
                            "type": "string",
                            "description": "File path or symbol name to query",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["all", "dependencies", "dependents"],
                            "description": "Which direction to traverse (default all)",
                        },
                        "max_depth": {
                            "type": "integer", "description": "Traversal depth (default 2)",
                        },
                    },
                    "required": ["target"],
                },
            },
            {
                "id": "explain_architecture",
                "name": "Explain Architecture",
                "description": (
                    "Explain the architecture of this codebase or a specific subsystem. "
                    "Returns community structure, key modules, data flow patterns, "
                    "entry points, and architectural observations. Optionally focus "
                    "on a specific community or module."
                ),
                "input": {
                    "type": "object",
                    "properties": {
                        "root_path": {"type": "string", "description": "Repository root path"},
                        "community_name": {
                            "type": "string",
                            "description": "Focus on a specific community (optional)",
                        },
                        "detail_level": {
                            "type": "string",
                            "enum": ["summary", "standard", "detailed"],
                            "description": "Level of detail (default standard)",
                        },
                    },
                },
            },
            {
                "id": "search_codebase",
                "name": "Search Codebase",
                "description": (
                    "Search the codebase by natural language query or symbol name. "
                    "Uses hybrid BM25 + TF-IDF search when embeddings are available, "
                    "otherwise falls back to FTS5/LIKE search. Returns ranked results "
                    "with file paths, line numbers, and language information."
                ),
                "input": {
                    "type": "object",
                    "properties": {
                        "root_path": {"type": "string", "description": "Repository root path"},
                        "query": {"type": "string", "description": "Search query (natural language or symbol name)"},
                        "kind": {
                            "type": "string",
                            "enum": ["function", "class", "method", "module", "variable"],
                            "description": "Filter by symbol kind (optional)",
                        },
                        "limit": {"type": "integer", "description": "Max results (default 20)"},
                    },
                    "required": ["query"],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# GraphSift state — one builder per repo root
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_builders: dict[str, Any] = {}
_source_maps: dict[str, dict[str, str]] = {}
_repo_roots: list[str] = []
_active_root: str | None = None


def _get_builder(root: str) -> tuple[Any, dict[str, str]]:
    """Return (builder, source_map) for *root*, creating if absent."""
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    with _lock:
        if root not in _builders:
            _builders[root] = ContextBuilder(ContextConfig())
            _source_maps[root] = {}
        return _builders[root], _source_maps[root]


def register_repo_root(root: str) -> dict:
    """Register a repository root for A2A queries.

    Returns a status dict with the root path and a warning if the graph
    has not yet been built.
    """
    global _active_root  # noqa: PLW0603
    normalized = str(Path(root).resolve())
    with _lock:
        if normalized not in _repo_roots:
            _repo_roots.append(normalized)
        if _active_root is None:
            _active_root = normalized
            return {"status": "registered", "root": normalized, "graph_built": False}
        _active_root = normalized

    builder, source_map = _get_builder(normalized)
    graph = getattr(builder, "_graph", None)
    is_built = graph is not None and len(graph._file_nodes) > 0  # noqa: SLF001

    return {
        "status": "registered",
        "root": normalized,
        "graph_built": is_built,
        "warning": (
            "Graph not yet built. Use analyze_impact or search_codebase "
            "to trigger a build, or call build_graph via MCP server first."
            if not is_built else ""
        ),
    }


def set_active_root(root: str) -> dict:
    """Switch the active repository root for subsequent queries."""
    normalized = str(Path(root).resolve())
    with _lock:
        if normalized not in _builders:
            # Auto-register if not yet known
            return register_repo_root(normalized)
        global _active_root  # noqa: PLW0603
        _active_root = normalized
    return {"status": "active_root_set", "root": normalized}


def list_repo_roots() -> list[dict]:
    """List all registered repository roots with their build status."""
    results = []
    with _lock:
        for root in _repo_roots:
            _, source_map = _get_builder(root)
            results.append({
                "root": root,
                "graph_built": len(source_map) > 0,
                "is_active": root == _active_root,
            })
    return results


# ---------------------------------------------------------------------------
# SQLite store helper (same pattern as mcp_server.py)
# ---------------------------------------------------------------------------

_store_lock = threading.RLock()
_stores: dict[str, Any] = {}


def _get_store(root: str) -> Any:
    """Return (creating if absent) the GraphStore for *root*."""
    from graphsift.adapters.storage import GraphStore

    with _store_lock:
        if root not in _stores:
            key = hashlib.sha1(root.encode()).hexdigest()[:12]  # noqa: F821
            home = Path.home() / ".graphsift" / key
            home.mkdir(parents=True, exist_ok=True)
            db_path = str(home / "graph.db")
            _stores[root] = GraphStore(db_path)
        return _stores[root]


import hashlib  # noqa: E402 — needed by _get_store above


# ---------------------------------------------------------------------------
# A2A Task handlers
# ---------------------------------------------------------------------------


def handle_analyze_impact(params: dict) -> dict:
    """A2A task: analyze impact of file changes.

    Returns affected files with relevance scores, depth, and citation evidence.

    Args:
        params: Must contain ``changed_files`` (list of paths). May contain
            ``root_path`` and ``max_depth``.

    Returns:
        Dict with ``affected_files``, each containing path, score, depth,
        reasons (evidence), and source citations.
    """
    from graphsift.core import estimate_tokens

    root = params.get("root_path") or _active_root or os.getcwd()
    changed_files = params.get("changed_files", [])
    max_depth = int(params.get("max_depth", 3))

    if not changed_files:
        return {"error": "changed_files is required", "affected_files": []}

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)

    # Auto-build if graph is empty
    if graph is None or not source_map:
        _auto_build_graph(root, builder, source_map)
        graph = getattr(builder, "_graph", None)
        if graph is None:
            return {"error": "Failed to build graph", "affected_files": []}

    scores = graph.ranked_neighbors(
        seed_paths=changed_files,
        include_dynamic=True,
    )
    affected = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    # Build evidence with source citations
    total_tokens = 0
    affected_files = []
    for file_path, (score, depth, reasons) in affected[:50]:
        source = source_map.get(file_path, "")
        tok = estimate_tokens(source) if source else 0
        total_tokens += tok

        # Citation evidence: structured references
        citations = []
        for reason in reasons[:3]:
            citations.append({
                "type": "dependency_relationship",
                "description": reason,
                "file": file_path,
                "confidence": round(min(1.0, score + 0.1), 2),
            })

        affected_files.append({
            "path": file_path,
            "score": round(score, 3),
            "depth": depth,
            "token_estimate": tok,
            "evidence": citations,
            "reasons": reasons[:5],
        })

    return {
        "task": "analyze_impact",
        "root": root,
        "changed_files": changed_files,
        "max_depth": max_depth,
        "affected_files": affected_files,
        "total_affected": len(affected),
        "total_tokens_scanned": total_tokens,
        "summary": (
            f"Analysis complete: {len(affected)} files affected by "
            f"{len(changed_files)} changed files. "
            f"Top match: {affected_files[0]['path'] if affected_files else 'none'} "
            f"(score={affected_files[0]['score'] if affected_files else 0})."
        ),
    }


def handle_find_dependencies(params: dict) -> dict:
    """A2A task: find dependencies of a file or symbol.

    Returns structured callers, callees, imports, importers, and related test
    files with citation evidence.

    Args:
        params: Must contain ``target`` (file path or symbol name). May contain
            ``root_path``, ``direction``, and ``max_depth``.

    Returns:
        Dict with dependencies, dependents, and metadata.
    """
    from graphsift.models import EdgeKind, NodeKind

    root = params.get("root_path") or _active_root or os.getcwd()
    target = params.get("target", "")
    direction = params.get("direction", "all")
    max_depth = int(params.get("max_depth", 2))

    if not target:
        return {"error": "target is required", "dependencies": [], "dependents": []}

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)

    if graph is None or not source_map:
        _auto_build_graph(root, builder, source_map)
        graph = getattr(builder, "_graph", None)
        if graph is None:
            return {"error": "Graph not available", "dependencies": [], "dependents": []}

    target_lower = target.lower()

    # Find matching nodes
    with graph._lock:  # noqa: SLF001
        nodes = dict(graph._nodes)
        adj_out = {k: list(v) for k, v in graph._adj_out.items()}
        adj_in = {k: list(v) for k, v in graph._adj_in.items()}
        file_nodes = dict(graph._file_nodes)

    # Match by file path or symbol name
    matched_nodes = []
    for n in nodes.values():
        if (
            target_lower in n.file_path.lower()
            or target_lower in n.name.lower()
            or target_lower in n.qualified_name.lower()
        ):
            matched_nodes.append(n)

    if not matched_nodes:
        return {
            "error": f"No symbols found matching '{target}'",
            "dependencies": [],
            "dependents": [],
        }

    results: dict[str, Any] = {
        "task": "find_dependencies",
        "target": target,
        "root": root,
        "matches_found": len(matched_nodes),
        "matches": [],
    }

    for seed in matched_nodes[:5]:
        nid = seed.node_id

        # Dependencies (outgoing edges)
        deps = []
        for edge in adj_out.get(nid, []):
            if edge.target_id in nodes:
                tgt = nodes[edge.target_id]
                deps.append({
                    "name": tgt.name,
                    "qualified_name": tgt.qualified_name,
                    "kind": tgt.kind.value if hasattr(tgt.kind, "value") else str(tgt.kind),
                    "file": tgt.file_path,
                    "line": tgt.line_start,
                    "relationship": edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind),
                    "weight": edge.weight,
                })

        # Dependents (incoming edges)
        dependents = []
        for edge in adj_in.get(nid, []):
            if edge.source_id in nodes:
                src = nodes[edge.source_id]
                dependents.append({
                    "name": src.name,
                    "qualified_name": src.qualified_name,
                    "kind": src.kind.value if hasattr(src.kind, "value") else str(src.kind),
                    "file": src.file_path,
                    "line": src.line_start,
                    "relationship": edge.kind.value if hasattr(edge.kind, "value") else str(edge.kind),
                    "weight": edge.weight,
                })

        # Find related test files
        test_files = []
        for fp in file_nodes:
            if "/test" in fp or "\\test" in fp or fp.startswith("test_"):
                target_file = seed.file_path
                if target_file in fp or Path(fp).stem == Path(target_file).stem:
                    test_files.append(fp)

        # Build evidence
        evidence = []
        if deps:
            evidence.append({
                "type": "dependency_list",
                "count": len(deps),
                "description": f"{seed.name} depends on {len(deps)} symbols",
                "confidence": 0.95,
            })
        if dependents:
            evidence.append({
                "type": "dependent_list",
                "count": len(dependents),
                "description": f"{seed.name} is used by {len(dependents)} symbols",
                "confidence": 0.90,
            })

        entry: dict[str, Any] = {
            "symbol": {
                "name": seed.name,
                "qualified_name": seed.qualified_name,
                "kind": seed.kind.value if hasattr(seed.kind, "value") else str(seed.kind),
                "file": seed.file_path,
                "line": seed.line_start,
            },
            "evidence": evidence,
            "test_files": test_files,
        }

        if direction in ("all", "dependencies"):
            entry["dependencies"] = sorted(deps, key=lambda x: -x["weight"])[:30]
        if direction in ("all", "dependents"):
            entry["dependents"] = sorted(dependents, key=lambda x: -x["weight"])[:30]

        results["matches"].append(entry)

    results["summary"] = (
        f"Found {len(matched_nodes)} matching symbols for '{target}'. "
        f"Top match: {matched_nodes[0].qualified_name} in "
        f"{matched_nodes[0].file_path}."
    )
    return results


def handle_explain_architecture(params: dict) -> dict:
    """A2A task: explain how a subsystem is architected.

    Returns community structure, key modules, data flow, entry points, and
    architectural observations.

    Args:
        params: May contain ``root_path``, ``community_name``, and
            ``detail_level``.

    Returns:
        Dict with architecture overview, communities, and observations.
    """
    root = params.get("root_path") or _active_root or os.getcwd()
    community_name = params.get("community_name", "")
    detail_level = params.get("detail_level", "standard")

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)

    if graph is None or not source_map:
        _auto_build_graph(root, builder, source_map)
        graph = getattr(builder, "_graph", None)
        if graph is None:
            return {"error": "Graph not available"}

    store = _get_store(root)
    db_stats = store.stats()
    communities = store.load_communities()
    risk_index = store.load_risk_index(min_score=0.5)

    high_risk_files = [r["file_path"] for r in risk_index[:10]]

    # Community structure
    community_list = []
    for c in communities[:30]:
        entry: dict[str, Any] = {
            "id": c["community_id"],
            "label": c["label"],
            "size": c["node_count"],
        }
        if detail_level in ("standard", "detailed") and community_name:
            if community_name.lower() in c["label"].lower():
                entry["members"] = c.get("metadata", {}).get("members", [])[:20]

        community_list.append(entry)

    # Build observations
    observations = []
    if communities:
        largest = max(communities, key=lambda c: c["node_count"])
        observations.append({
            "type": "community_structure",
            "description": (
                f"Codebase has {len(communities)} communities. "
                f"Largest: '{largest['label']}' with {largest['node_count']} nodes."
            ),
            "confidence": 0.95,
        })
    if risk_index:
        avg_risk = sum(r["risk_score"] for r in risk_index) / max(len(risk_index), 1)
        observations.append({
            "type": "risk_profile",
            "description": (
                f"Average risk score: {avg_risk:.2f}. "
                f"High-risk files: {len(high_risk_files)}."
            ),
            "confidence": 0.85,
        })

    # Entry point detection
    entry_points = []
    for node in graph._nodes.values():  # noqa: SLF001
        if node.name == "main" and node.kind.value in ("function", "method"):
            entry_points.append({
                "name": node.qualified_name,
                "file": node.file_path,
                "kind": node.kind.value,
            })
    if not entry_points:
        entry_points = [
            {"name": f, "file": f, "kind": "module"}
            for f in list(source_map.keys())[:5]
        ]

    result: dict[str, Any] = {
        "task": "explain_architecture",
        "root": root,
        "detail_level": detail_level,
        "overview": {
            "total_nodes": db_stats.get("nodes", 0),
            "total_edges": db_stats.get("edges", 0),
            "total_files": db_stats.get("files", 0),
            "total_communities": len(communities),
            "schema_version": db_stats.get("schema_version", 0),
        },
        "communities": community_list,
        "entry_points": entry_points[:10],
        "high_risk_files": high_risk_files,
        "observations": observations[:10],
    }

    # Focus on a specific community if requested
    if community_name:
        focused = next(
            (c for c in communities if community_name.lower() in c["label"].lower()),
            None,
        )
        if focused:
            result["focused_community"] = {
                "label": focused["label"],
                "node_count": focused["node_count"],
                "members": focused.get("metadata", {}).get("members", []),
            }
            result["summary"] = (
                f"Architecture for '{focused['label']}' community: "
                f"{focused['node_count']} nodes across "
                f"{len(focused.get('metadata', {}).get('members', []))} files."
            )
        else:
            result["summary"] = (
                f"Codebase architecture: {db_stats.get('nodes', 0)} nodes, "
                f"{db_stats.get('edges', 0)} edges across "
                f"{len(communities)} communities."
            )
    else:
        result["summary"] = (
            f"Codebase architecture: {db_stats.get('nodes', 0)} nodes, "
            f"{db_stats.get('edges', 0)} edges across "
            f"{len(communities)} communities. "
            f"{len(high_risk_files)} high-risk files identified."
        )

    return result


def handle_search_codebase(params: dict) -> dict:
    """A2A task: search codebase by natural language query.

    Uses hybrid BM25 + TF-IDF search when embeddings are available, otherwise
    falls back to FTS5/LIKE search.

    Args:
        params: Must contain ``query``. May contain ``root_path``, ``kind``,
            and ``limit``.

    Returns:
        Dict with ranked search results, each with file path, line numbers,
        language, and evidence.
    """
    root = params.get("root_path") or _active_root or os.getcwd()
    query = params.get("query", "")
    kind_filter = params.get("kind", "")
    limit = int(params.get("limit", 20))

    if not query:
        return {"error": "query is required", "results": []}

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)

    if graph is None or not source_map:
        _auto_build_graph(root, builder, source_map)
        graph = getattr(builder, "_graph", None)
        if graph is None:
            return {"error": "Graph not available", "results": []}

    store = _get_store(root)

    # Try hybrid search first, fall back to FTS5/LIKE
    all_nodes = store.load_nodes()
    query_lower = query.lower()

    # Check for TF-IDF embeddings
    has_embeddings = _check_embed_version(store)

    if has_embeddings:
        from graphsift.hybrid_search import HybridSearcher  # noqa: PLC0415

        searcher = HybridSearcher(alpha=0.3)
        scored = searcher.search(query, all_nodes, top_k=limit * 2)

        if kind_filter:
            kind_lower = kind_filter.lower()
            scored = [(n, s) for n, s in scored if n.kind.value == kind_lower]

        results = []
        for n, s in scored[:limit]:
            results.append({
                "name": n.name,
                "qualified_name": n.qualified_name,
                "kind": n.kind.value,
                "file": n.file_path,
                "line": n.line_start,
                "language": n.language.value,
                "score": round(s, 4),
                "evidence": [
                    {
                        "type": "semantic_match",
                        "description": f"BM25+TF-IDF score {s:.3f}",
                        "file": n.file_path,
                        "confidence": round(min(1.0, s), 2),
                    }
                ],
            })
    else:
        # Fallback: FTS5 / LIKE search
        search_results = store.search_nodes(query, limit=limit * 2)

        if kind_filter:
            kind_lower = kind_filter.lower()
            search_results = [n for n in search_results if n.kind.value == kind_lower]

        results = []
        for n in search_results[:limit]:
            score = _like_relevance(query_lower, n.name, n.file_path)
            results.append({
                "name": n.name,
                "qualified_name": n.qualified_name,
                "kind": n.kind.value,
                "file": n.file_path,
                "line": n.line_start,
                "language": n.language.value,
                "score": round(score, 4),
                "evidence": [
                    {
                        "type": "keyword_match",
                        "description": f"Keyword match score {score:.3f}",
                        "file": n.file_path,
                        "confidence": round(score, 2),
                    }
                ],
            })

    return {
        "task": "search_codebase",
        "query": query,
        "root": root,
        "results": results,
        "total": len(results),
        "search_method": "hybrid" if has_embeddings else "keyword",
        "summary": f"Found {len(results)} results for '{query}' using "
        f"{'hybrid search' if has_embeddings else 'keyword search'}.",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_build_graph(
    root: str,
    builder: Any,
    source_map: dict[str, str],
) -> None:
    """Automatically build the graph if it's empty and source files exist.

    Scans the repository directory for supported source files, parses them,
    and builds the dependency graph in memory. This enables zero-config
    operation where the A2A server self-indexes on first query.
    """
    from graphsift.adapters.filesystem import load_source_map  # noqa: PLC0415
    from graphsift.core import ContextBuilder  # noqa: PLC0415

    logger.info("a2a: auto-building graph for %s", root)

    try:
        extensions = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java"}
        exclude_dirs = {
            "venv", ".venv", "node_modules", ".git", "__pycache__",
            "dist", "build", ".mypy_cache", ".pytest_cache",
        }
        new_map = load_source_map(root, extensions=extensions, exclude_dirs=exclude_dirs)
    except Exception as exc:
        logger.warning("a2a: auto-build load failed: %s", exc)
        return

    if not new_map:
        logger.warning("a2a: no source files found under %s", root)
        return

    source_map.update(new_map)
    builder.index_files(source_map)
    logger.info("a2a: auto-built graph with %d files", len(new_map))


def _check_embed_version(store: Any) -> bool:
    """Return True if the store has TF-IDF embeddings (embed_version == '1')."""
    try:
        with store._lock:  # noqa: SLF001
            row = store._conn.execute(
                "SELECT value FROM graph_meta WHERE key='embed_version' LIMIT 1",
            ).fetchone()
            return row is not None and row["value"] == "1"
    except Exception:
        return False


def _like_relevance(query: str, name: str, file_path: str) -> float:
    """Compute a simple relevance score for LIKE-based search results.

    Scores range from 0.0 to 1.0:
    - Exact match on name: 1.0
    - Substring match on name: 0.8
    - Match in qualified name: 0.6
    - Match in file path: 0.4
    """
    name_lower = name.lower()
    path_lower = file_path.lower()
    if query == name_lower:
        return 1.0
    if query in name_lower:
        return 0.8
    if f"{Path(name_lower).stem}" == query:
        return 0.7
    if query in path_lower:
        return 0.4
    return 0.1


# ---------------------------------------------------------------------------
# A2A task dispatcher
# ---------------------------------------------------------------------------

_TASK_HANDLERS: dict[str, callable] = {
    "analyze_impact": handle_analyze_impact,
    "find_dependencies": handle_find_dependencies,
    "explain_architecture": handle_explain_architecture,
    "search_codebase": handle_search_codebase,
}


def _handle_jsonrpc_request(body: dict) -> dict:
    """Process a JSON-RPC 2.0 request and return a response dict.

    Supports standard JSON-RPC methods plus A2A task execution.
    """
    req_id = body.get("id")
    method = body.get("method", "")
    params = body.get("params", {})

    if not method:
        return _err(req_id, -32600, "Invalid Request: method is required")

    # --- A2A task execution ---
    if method == "tasks/send":
        task_id = params.get("id", "")
        task_method = params.get("method", "")
        task_params = params.get("params", {})

        if task_method in _TASK_HANDLERS:
            try:
                result = _TASK_HANDLERS[task_method](task_params)
                return _ok(req_id, {
                    "id": task_id or task_method,
                    "status": "completed",
                    "result": result,
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception("task %s failed", task_method)
                return _err(req_id, -32603, f"Task execution failed: {exc}")
        else:
            return _err(
                req_id, -32601,
                f"Unknown task method: {task_method}",
                {"available_tasks": list(_TASK_HANDLERS.keys())},
            )

    # --- Direct method dispatch ---
    if method in _TASK_HANDLERS:
        try:
            result = _TASK_HANDLERS[method](params)
            return _ok(req_id, result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("method %s failed", method)
            return _err(req_id, -32603, str(exc))

    # --- Built-in RPC methods ---
    if method == "rpc.discover":
        return _ok(req_id, {
            "name": "GraphSift Code Intelligence",
            "version": __version__,
            "methods": list(_TASK_HANDLERS.keys()),
            "skills": [
                {
                    "id": sid,
                    "name": sid.replace("_", " ").title(),
                }
                for sid in _TASK_HANDLERS
            ],
        })

    if method == "rpc.list_roots":
        return _ok(req_id, {"roots": list_repo_roots()})

    if method == "rpc.register_root":
        root = params.get("root", os.getcwd())
        return _ok(req_id, register_repo_root(root))

    if method == "rpc.set_active_root":
        root = params.get("root", "")
        return _ok(req_id, set_active_root(root))

    if method == "rpc.status":
        with _lock:
            root = _active_root or ""
            _, source_map = _get_builder(root) if root else (None, {})
        return _ok(req_id, {
            "active_root": root,
            "repo_count": len(_repo_roots),
            "graph_built": len(source_map) > 0 if source_map else False,
            "files_indexed": len(source_map) if source_map else 0,
        })

    return _err(req_id, -32601, f"Method not found: {method}")


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------


class _A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the A2A server."""

    # Silence default logging (we use our own logger)
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("A2A HTTP: %s", format % args)

    def _send_json(
        self,
        status: int,
        data: Any,
        cors_origin: str = "*",
    ) -> None:
        """Send a JSON response with CORS headers."""
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", cors_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> dict:
        """Read and parse the JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"_parse_error": str(exc)}

    # ---- Route handlers ----

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self._send_json(204, {})

    def do_GET(self) -> None:
        """Handle GET requests.

        Routes:
          /.well-known/agent-card.json  — Return the agent card
        """
        if self.path == "/.well-known/agent-card.json":
            base_url = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"  # type: ignore[attr-defined]
            with _lock:
                repo_count = len(_repo_roots)
            card = build_agent_card(
                base_url=base_url,
                repo_count=repo_count,
            )
            self._send_json(200, card)
        else:
            self._send_json(404, {
                "jsonrpc": _JSONRPC,
                "error": {"code": -32000, "message": f"Not found: {self.path}"},
            })

    def do_POST(self) -> None:
        """Handle POST requests.

        Routes:
          /        — JSON-RPC 2.0 endpoint (all methods)
          /tasks   — A2A task execution endpoint
        """
        body = self._read_body()

        # Check for parse errors
        parse_error = body.pop("_parse_error", None)
        if parse_error:
            self._send_json(400, _err(None, -32700, f"Parse error: {parse_error}"))
            return

        if self.path == "/tasks":
            # A2A /tasks endpoint: expects {"id": ..., "method": ..., "params": ...}
            task_id = body.get("id", "")
            method = body.get("method", "")
            params = body.get("params", {})

            if method in _TASK_HANDLERS:
                try:
                    result = _TASK_HANDLERS[method](params)
                    response = {
                        "jsonrpc": _JSONRPC,
                        "id": task_id or method,
                        "result": {
                            "id": task_id or method,
                            "status": "completed",
                            "result": result,
                        },
                    }
                    self._send_json(200, response)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("task %s failed", method)
                    self._send_json(500, _err(task_id or None, -32603, str(exc)))
            else:
                self._send_json(404, _err(
                    task_id or None, -32601,
                    f"Unknown task: {method}",
                    {"available_tasks": list(_TASK_HANDLERS.keys())},
                ))
        else:
            # Default JSON-RPC endpoint
            response = _handle_jsonrpc_request(body)
            is_error = "error" in response
            self._send_json(200 if not is_error else 400, response)


# ---------------------------------------------------------------------------
# A2AServer
# ---------------------------------------------------------------------------


class A2AServer:
    """A2A Protocol server for GraphSift code intelligence.

    Exposes GraphSift's dependency graph, impact analysis, architecture
    explanations, and code search as an A2A agent that other agents (or
    HTTP clients) can query over HTTP.

    Uses Python's built-in ``http.server`` — no external dependencies.

    Args:
        host: Host address to bind to (default ``127.0.0.1``).
        port: Port to listen on (default ``8889``).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8889) -> None:
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Return the base URL of this server."""
        return f"http://{self._host}:{self._port}"

    @property
    def agent_card(self) -> dict:
        """Return the A2A agent card based on current server state.

        The card describes all skills, capabilities, and metadata that
        this agent exposes. Call this after registering repos for an
        accurate ``repo_count``.
        """
        with _lock:
            repo_count = len(_repo_roots)
        return build_agent_card(
            base_url=self.url,
            repo_count=repo_count,
        )

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread.

        The server runs until ``stop()`` is called. This method returns
        immediately; the server processes requests on a separate thread.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._running.is_set():
            raise RuntimeError("A2A server is already running")

        self._server = HTTPServer(
            (self._host, self._port),
            _A2ARequestHandler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="a2a-server",
            daemon=True,
        )
        self._thread.start()
        self._running.set()
        logger.info(
            "A2A server started on %s  (agent card: %s/.well-known/agent-card.json)",
            self.url,
            self.url,
        )

    def stop(self) -> None:
        """Stop the HTTP server and clean up.

        Safe to call even if the server was never started.
        """
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
            self._thread = None
        self._running.clear()
        logger.info("A2A server stopped")

    def register_repo(self, root: str) -> dict:
        """Register a repository root for A2A queries.

        Args:
            root: Filesystem path to the repository root.

        Returns:
            Status dict from ``register_repo_root()``.
        """
        return register_repo_root(root)

    def set_active_repo(self, root: str) -> dict:
        """Set the active repository for subsequent queries.

        Args:
            root: Filesystem path to the repository root.

        Returns:
            Status dict from ``set_active_root()``.
        """
        return set_active_root(root)

    def list_repos(self) -> list[dict]:
        """List all registered repository roots with their build status.

        Returns:
            List of dicts with ``root``, ``graph_built``, ``is_active``.
        """
        return list_repo_roots()


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_server(
    host: str = "127.0.0.1",
    port: int = 8889,
    register_repo: str | None = None,
    log_level: str = "INFO",
) -> A2AServer:
    """Create, start, and return an A2A server instance.

    This is the recommended entry point for most use cases::

        server = run_server(port=8889, register_repo="/path/to/repo")
        # ... server is running in background thread ...
        print(server.url)
        server.stop()

    Args:
        host: Host address to bind.
        port: Port to listen on.
        register_repo: Optional repo root to register on startup.
        log_level: Logging level (default ``INFO``).

    Returns:
        Running A2AServer instance.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
    )

    server = A2AServer(host=host, port=port)

    if register_repo:
        result = server.register_repo(register_repo)
        if result.get("warning"):
            logger.warning("a2a: %s", result["warning"])

    server.start()
    return server
