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

from graphsift.read_cache import SafeFileIO
from graphsift._version import __version__ as _GRAPHSIFT_VERSION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Production safeguards for auto-scan (mirrors cli.py constants)
# ---------------------------------------------------------------------------
_MAX_AUTO_SCAN_FILES = 5000        # skip if repo is larger
_MAX_AUTO_SCAN_DELETIONS = 10      # skip if batch delete is too large
_MIN_SCAN_INTERVAL_S = 30.0        # rate limit across MCP calls
_last_scan_times: dict[str, float] = {}  # root -> last scan timestamp


def _mcp_should_auto_scan(root: str, deleted_count: int, total_files_estimate: int) -> bool:
    """Production gates for auto-scan in the MCP server path."""
    import time as _time
    now = _time.monotonic()

    if total_files_estimate > _MAX_AUTO_SCAN_FILES:
        logger.debug("mcp auto-scan skipped: %d files > %d limit", total_files_estimate, _MAX_AUTO_SCAN_FILES)
        return False
    if deleted_count > _MAX_AUTO_SCAN_DELETIONS:
        logger.debug("mcp auto-scan skipped: %d deletions > %d batch limit", deleted_count, _MAX_AUTO_SCAN_DELETIONS)
        return False
    last = _last_scan_times.get(root, 0.0)
    if now - last < _MIN_SCAN_INTERVAL_S:
        logger.debug("mcp auto-scan skipped: rate-limited (%.1fs ago)", now - last)
        return False
    _last_scan_times[root] = now
    return True


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
_roots_by_hash: dict[str, str] = {}  # repo_hash -> root_path (for MCP resource URI resolution)
_manual_selectors: dict[str, Any] = {}  # root_path -> ManualSelector
_plugin_registries: dict[str, Any] = {}  # root_path -> PluginRegistry


def _db_path_for(root: str) -> str:
    """Compute the DB path for a given repo root."""
    key = hashlib.sha1(root.encode()).hexdigest()[:12]
    home = Path.home() / ".graphsift" / key
    home.mkdir(parents=True, exist_ok=True)
    return str(home / "graph.db")


def _session_id_for(root: str) -> str:
    """Compute a stable session identifier from a repo root.

    Used as the ``session_id`` for cross-session memory keying.
    """
    return hashlib.sha256(root.encode()).hexdigest()[:16]


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
            # Register reverse hash mapping for MCP resource URI resolution
            _roots_by_hash[hashlib.sha1(root.encode()).hexdigest()[:12]] = root
        return _stores[root]


# ---------------------------------------------------------------------------
# Fully-automated indexing — build any repo's graph on first use.
# Removes the manual ``graphsift build`` step for Claude Code.
# ---------------------------------------------------------------------------

_ensure_lock = threading.RLock()
_ensured_roots: set[str] = set()


def _ensure_graph(root: str) -> dict:
    """Build the graph for *root* on first access, if it isn't indexed yet.

    Idempotent per process: the first tool call that touches a repo with no
    indexed graph triggers a build; every later call is a cheap set lookup.
    Only runs on stderr-logged paths (never stdout), so it is safe to call
    from the background startup thread without corrupting the stdio protocol.
    """
    if root in _ensured_roots:
        return {}
    with _ensure_lock:
        if root in _ensured_roots:
            return {}
        try:
            store = _get_store(root)
            if store.stats().get("files", 0) > 0:
                _ensured_roots.add(root)  # already indexed — nothing to do
                return {}
        except Exception:
            pass
        logger.info("graphsift: auto-indexing %s (no graph found)", root)
        try:
            result = _tool_build_graph({"root_path": root})
        except Exception as exc:  # noqa: BLE001
            logger.warning("graphsift: auto-index failed for %s: %s", root, exc)
            result = {}
        _ensured_roots.add(root)
        return result


# ---------------------------------------------------------------------------
# Graphsift state — one builder per working directory
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_builders: dict[str, Any] = {}   # root_path -> ContextBuilder
_source_maps: dict[str, dict[str, str]] = {}  # root_path -> source_map


def _get_builder(root: str) -> tuple[Any, dict[str, str]]:
    """Return (builder, source_map) for *root*, creating if absent.

    Fully automated: if the repo has no indexed graph yet, it is built
    automatically on first access via ``_ensure_graph`` — Claude never has to
    run ``graphsift build`` by hand.
    """
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    with _lock:
        if root not in _builders:
            _builders[root] = ContextBuilder(ContextConfig())
            _source_maps[root] = {}
            # Register reverse hash mapping for MCP resource URI resolution
            _roots_by_hash[hashlib.sha1(root.encode()).hexdigest()[:12]] = root

    _ensure_graph(root)
    return _builders[root], _source_maps[root]


def _get_manual_selector(root: str) -> Any:
    """Return (creating if absent) the ManualSelector for *root*."""
    from graphsift.prompt_templates import ManualSelector

    with _lock:
        if root not in _manual_selectors:
            _manual_selectors[root] = ManualSelector()
        return _manual_selectors[root]


def _get_plugin_registry(root: str) -> Any:
    """Return (creating if absent) the PluginRegistry for *root*."""
    from graphsift.commands.registry import PluginRegistry

    with _lock:
        if root not in _plugin_registries:
            _plugin_registries[root] = PluginRegistry()
        return _plugin_registries[root]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_build_graph(params: dict) -> dict:
    """Index all source files under root_path and build the dependency graph."""
    from graphsift.adapters.filesystem import walk_repo
    from graphsift.core import ContextBuilder
    from graphsift.models import FileNode, GraphEdge, GraphNode
    from graphsift.sha_cache import load_sha_cache, save_sha_cache, stat_match

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

    # -- Ensure SQLite DB is open and migrated
    store = _get_store(root)

    # Version-aware cleanup: if the stored graph was built by an older graphsift,
    # stale nodes from the previous parser must be purged before re-indexing.
    _manifest_path = os.path.join(root, ".graphsift", "manifest.json")
    _stored_version = "0"
    try:
        if os.path.isfile(_manifest_path):
            with open(_manifest_path, "r", encoding="utf-8") as _mf:
                _stored_version = json.load(_mf).get("graphsift_version", "0")
    except Exception:
        _stored_version = "0"
    # Version-stale ⇒ clean rebuild automatically (no --force needed).
    _version_changed = os.path.isfile(_manifest_path) and _stored_version != _GRAPHSIFT_VERSION

    # -- Walk paths, stat-check unchanged, read all content for source_map
    sha_cache = load_sha_cache(root)
    walk_paths = walk_repo(root, extensions=extensions, exclude_dirs=exclude_dirs)
    total_files = len(walk_paths)
    source_map: dict[str, str] = {}
    needs_index: set[str] = set()

    for p in walk_paths:
        if sha_cache and stat_match(p, sha_cache):
            pass  # unchanged — skip indexing, but still read for source_map
        else:
            needs_index.add(p)
        try:
            source_map[p] = SafeFileIO.read(p)
        except OSError:
            pass

    fast_unchanged = total_files - len(needs_index)
    if fast_unchanged:
        logger.info("INFO: %d files unchanged (stat-match) — skip re-index", fast_unchanged)

    with _lock:
        from graphsift.models import ContextConfig
        _builders[root] = ContextBuilder(ContextConfig())
        builder = _builders[root]
        _source_maps[root] = source_map

    # -- Index only changed/new files
    parsed_count = 0
    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    all_file_nodes: list[FileNode] = []

    for path in walk_paths:
        source = source_map.get(path)
        if source is None or path not in needs_index:
            parsed_count += 1
            if progress_interval > 0 and parsed_count % progress_interval == 0:
                logger.info("INFO: Progress: %d/%d files", parsed_count, total_files)
            continue
        try:
            with _lock:
                builder.index_file(path, source)
        except Exception as exc:  # noqa: BLE001
            logger.debug("build_graph: skipped %s: %s", path, exc)

        parsed_count += 1
        if progress_interval > 0 and parsed_count % progress_interval == 0:
            logger.info("INFO: Progress: %d/%d files", parsed_count, total_files)

    logger.info("INFO: Progress: %d/%d files", total_files, total_files)

    # -- Gather stats from builder graph
    with _lock:
        stats = builder.index_files(source_map)
        graph = getattr(builder, "_graph", None)

    # -- Persist nodes + edges + files to SQLite
    if graph is not None:
        try:
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
            if _version_changed:
                purged = store.purge_all_graph_data()
                logger.info(
                    "INFO: graphsift %s → %s — cleaned %d stale graph records",
                    _stored_version, _GRAPHSIFT_VERSION, sum(purged.values()),
                )
            store.save_nodes(all_nodes)
            store.save_files(all_file_nodes)
            graph_edges = graph.all_edges()
            if graph_edges:
                store.save_edges(graph_edges)
            logger.info(
                "INFO: Persisted %d nodes, %d files, %d edges to SQLite",
                len(all_nodes), len(all_file_nodes), len(graph_edges),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("build_graph: SQLite persist failed: %s", exc)

    # -- Save SHA cache with mtime+size for future incremental builds
    if graph is not None:
        for file_node in graph.all_files():
            if hasattr(file_node, "sha256") and file_node.sha256:
                try:
                    st = os.stat(file_node.path)
                    sha_cache[file_node.path] = {
                        "sha": file_node.sha256,
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                    }
                except OSError:
                    sha_cache[file_node.path] = file_node.sha256  # plain str fallback
        save_sha_cache(root, sha_cache)

    # Write a manifest so the PostToolUse hook and status/CLI paths see a built
    # graph (previously only the CLI wrote one, so an MCP-built repo made the
    # hook think it was unbuilt and re-index everything).
    try:
        os.makedirs(os.path.join(root, ".graphsift"), exist_ok=True)
        SafeFileIO.write_json(
            _manifest_path,
            {
                "root": root,
                "files_indexed": stats.files_indexed,
                "symbols_extracted": stats.symbols_extracted,
                "edges_created": stats.edges_created,
                "duration_ms": stats.duration_ms,
                "graphsift_version": _GRAPHSIFT_VERSION,
                "files": [str(p) for p in walk_paths],
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("build_graph: manifest write failed: %s", exc)

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
    """Incrementally update the graph with changed files only.

    Detects both modified and deleted files.  Deleted files are
    purged from the graph database automatically.
    """
    from graphsift.adapters.filesystem import load_changed_files

    root = params.get("root_path", os.getcwd())

    # Version-stale ⇒ rebuild fully (same as --force); an incremental update on
    # top of an older parser's data would leave stale nodes behind.
    try:
        _mp = os.path.join(root, ".graphsift", "manifest.json")
        if os.path.isfile(_mp):
            with open(_mp, "r", encoding="utf-8") as _mf:
                if json.load(_mf).get("graphsift_version", "0") != _GRAPHSIFT_VERSION:
                    return _tool_build_graph({"root_path": root})
    except Exception:
        pass

    candidates = params.get("changed_files", [])

    if not candidates:
        return {"status": "no_changes", "files_updated": 0}

    # Separate candidates into modified (still exist) and deleted
    changed = [f for f in candidates if os.path.isfile(f)]
    deleted = [f for f in candidates if not os.path.isfile(f)]

    # Handle deletions from graph DB
    deleted_count = 0
    stale_findings = 0
    if deleted:
        try:
            from graphsift.adapters.storage import GraphStore
            store = GraphStore(_db_path_for(root))
            for fp in deleted:
                try:
                    store.delete_file_completely(fp)
                    deleted_count += 1
                except Exception as exc:
                    logger.warning("update_graph: delete failed for %s: %s", fp, exc)
        except Exception as exc:
            logger.warning("update_graph: could not open store for cleanup: %s", exc)

        # Auto-scan for stale source-code references (gated)
        total_files_estimate = len(candidates)  # rough proxy for size gate
        if _mcp_should_auto_scan(root, len(deleted), total_files_estimate):
            try:
                from graphsift.cleanup import StaleRefScanner
                from graphsift.adapters.filesystem import load_source_map
                source_map = load_source_map(root)
                scanner = StaleRefScanner(project_root=root)
                report = scanner.scan_after_deletion(deleted, source_map=source_map)
                stale_findings = report.total
                if stale_findings:
                    logger.warning(
                        "update_graph: %d stale reference(s) found for deleted file(s). "
                        "Use the prune_refs tool to inspect or fix.",
                        stale_findings,
                    )
            except Exception:
                pass

    if not changed:
        result: dict = {"status": "cleaned", "files_updated": 0, "files_deleted": deleted_count}
        if stale_findings:
            result["stale_references"] = stale_findings
        return result

    builder, source_map = _get_builder(root)
    new_sources = load_changed_files(changed)

    with _lock:
        source_map.update(new_sources)
        for path, source in new_sources.items():
            try:
                builder.index_file(path, source)
            except Exception as exc:  # noqa: BLE001
                logger.warning("update_graph: skipped %s: %s", path, exc)

    # Auto-scan modified files for removed exports (gated)
    stale_mod_findings = 0
    if changed and _mcp_should_auto_scan(root, len(changed), len(candidates)):
        try:
            from graphsift.cleanup import StaleRefScanner
            scanner = StaleRefScanner(project_root=root)
            report = scanner.scan_after_modification(changed, source_map=new_sources)
            stale_mod_findings = report.total
        except Exception:
            pass

    result: dict = {"status": "updated", "files_updated": len(new_sources)}
    if deleted_count:
        result["files_deleted"] = deleted_count
    if stale_findings:
        result["stale_references"] = stale_findings
    if stale_mod_findings:
        result["stale_references"] = result.get("stale_references", 0) + stale_mod_findings
    return result


def _tool_prune_refs(params: dict) -> dict:
    """Scan for stale references to deleted files and optionally auto-fix."""
    project_root = params.get("project_root", os.getcwd())
    deleted_paths = params.get("deleted_paths", [])
    fix = params.get("fix", False)

    from graphsift.adapters.filesystem import load_source_map
    try:
        from graphsift.cleanup import StaleRefScanner
    except ImportError:
        return {"error": "cleanup module not available", "findings": [], "total": 0}

    source_map = load_source_map(project_root)
    scanner = StaleRefScanner(project_root=project_root)
    report = scanner.scan_after_deletion(deleted_paths, source_map=source_map)
    if not report.findings:
        return {"findings": [], "total": 0, "message": "No stale references found"}

    result = report.model_dump()
    if fix:
        fix_result = scanner.apply_fixes(report, dry_run=False)
        result["fix_applied"] = fix_result
    return result


def _tool_get_context(params: dict) -> dict:
    """Build ranked context for a code diff / query.

    Uses cross-session memory when the SQLite store is available:
    cache key = hash of (sorted changed_files + query + commit_message).
    Returns ``from_cache`` metadata when a cached result is reused.
    """
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

    # Reuse existing graph to avoid redundant re-indexing
    existing_graph = getattr(builder, "_graph", None)

    # Enable cross-session memory via SQLite store
    store = _get_store(root)
    session_id = _session_id_for(root)
    config = ContextConfig(
        token_budget=token_budget,
        session_id=session_id,
    )

    builder_fresh = ContextBuilder(
        config=config,
        graph=existing_graph,
        store=store,
    )
    # Only index if graph is empty (first call before build_graph)
    if existing_graph is None or existing_graph.stats().get("files", 0) == 0:
        builder_fresh.index_files(source_map)

    # Warm in-memory cache from SQLite for fast subsequent lookups
    builder_fresh.warm_cache()

    diff = DiffSpec(
        changed_files=changed_files,
        query=query,
        diff_text=diff_text,
        commit_message=commit_message,
    )
    result = builder_fresh.build(diff, source_map)

    response: dict = {
        "rendered_context": result.rendered_context,
        "files_selected": result.files_selected,
        "files_scanned": result.files_scanned,
        "total_original_tokens": result.total_original_tokens,
        "total_rendered_tokens": result.total_rendered_tokens,
        "reduction_ratio": round(result.reduction_ratio, 3),
        "token_savings_pct": round((1 - result.reduction_ratio) * 100, 1),
    }

    # Expose cache hit metadata to the caller
    if result.metadata.get("from_cache"):
        response["from_cache"] = True
        response["cache_source"] = result.metadata.get("cache_source", "")

    return response


def _tool_get_impact(params: dict) -> dict:
    """Return the blast radius (affected files) for a set of changed files."""
    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files", [])
    max_depth = int(params.get("max_depth", 3))

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)

    if not graph or not source_map:
        return {"error": "Graph not built yet.", "affected_files": []}

    scores = graph.ranked_neighbors(
        seed_paths=changed_files,
        include_dynamic=True,
    )
    affected = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    return {
        "changed_files": changed_files,
        "affected_files": [
            {"path": p, "score": round(s[0], 3), "depth": s[1], "reasons": s[2][:3]} for p, s in affected[:50]
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
    graph = getattr(builder, "_graph", None)

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
    graph = getattr(builder, "_graph", None)

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
    """Rename preview, dead-code detection, or fix suggestions across the graph."""
    from graphsift.adapters.postprocess import RefactorEngine

    root = params.get("root_path", os.getcwd())
    mode = params.get("mode", "rename")

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph:
        return {"error": "Graph not built yet. Call build_graph first."}

    if mode == "suggest":
        from graphsift.auto_fix import FixSuggester  # noqa: PLC0415

        suggester = FixSuggester(graph, source_map=source_map)
        report = suggester.analyze()
        return {
            "mode": "suggest",
            "suggestions": [s.model_dump() for s in report.suggestions],
            "total_issues": report.total_issues,
            "by_severity": report.by_severity,
            "by_category": report.by_category,
            "summary": report.summary,
        }

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

        # Apply priority scoring when results are large
        prioritize = params.get("prioritize", True)
        if prioritize and dead:
            from .prioritize import PriorityScorer  # noqa: PLC0415
            scorer = PriorityScorer(graph=graph, source_map=source_map)
            ranked = scorer.score_dead_code(dead)
            return {
                "mode": "dead_code",
                "results": [e.to_dict() for e in ranked.entries],
                "total": len(dead),
                "tiers": ranked.tiers,
                "summary": ranked.summary,
                "truncated": ranked.truncated,
                "truncated_count": ranked.truncated_count,
            }

        return {"mode": "dead_code", "results": dead, "total": len(dead)}

    return {"error": f"Unknown mode: {mode}. Valid: rename, dead_code, suggest"}


def _tool_detect_cycles(params: dict) -> dict:
    """Detect circular dependencies (import/call cycles) in the codebase."""
    root = params.get("root", os.getcwd())
    builder, source_map = _get_builder(root)

    if not builder._graph or not builder._graph._nodes:
        return {"error": "Graph not built. Run build_graph first."}

    cycles = builder._graph.detect_cycles()

    # Build response
    cycle_infos = []
    for i, cycle in enumerate(cycles):
        cycle_infos.append({
            "cycle_id": i + 1,
            "files": cycle,
            "length": len(cycle),
            "severity": "error" if len(cycle) <= 3 else "warning",
        })

    all_cycle_files = set()
    for c in cycles:
        all_cycle_files.update(c)

    return {
        "cycles": cycle_infos,
        "total_cycles": len(cycles),
        "max_cycle_length": max((len(c) for c in cycles), default=0),
        "files_in_cycles": len(all_cycle_files),
        "root": root,
    }


def _tool_detect_dead_code(params: dict) -> dict:
    """Detect potentially unreachable/dead code via reachability analysis."""
    root = params.get("root", os.getcwd())
    kind = params.get("kind")  # None for all, or "function", "class", "method"
    entry_points = params.get("entry_points")  # Optional list of entry-point file paths
    prioritize = params.get("prioritize", True)  # Apply priority scoring
    max_results = params.get("max_results", 0)  # 0 = unlimited

    builder, source_map = _get_builder(root)

    if not builder._graph or not builder._graph._nodes:
        return {"error": "Graph not built. Run build_graph first."}

    dead = builder._graph.find_dead_code(entry_points=entry_points, kind=kind)

    result: dict = {
        "total_dead": len(dead),
        "confidence": "high" if entry_points else "medium",
        "root": root,
        "note": (
            "Auto-detected entry points may miss some. Provide explicit "
            "entry_points for high confidence."
        )
        if not entry_points
        else "",
    }

    # Apply priority scoring when results are large or scoring is requested
    if prioritize and dead:
        from .prioritize import PriorityScorer  # noqa: PLC0415

        scorer = PriorityScorer(
            graph=builder._graph, source_map=source_map
        )
        ranked = scorer.score_dead_code(dead)
        result["entries"] = [
            e.to_dict() for e in ranked.entries
        ]
        result["tiers"] = ranked.tiers
        result["summary"] = ranked.summary
        result["truncated"] = ranked.truncated
        result["truncated_count"] = ranked.truncated_count
    else:
        result["entries"] = dead

    if max_results > 0 and len(result.get("entries", [])) > max_results:
        result["entries"] = result["entries"][:max_results]
        result["truncated"] = True
        result["truncated_count"] = len(dead) - max_results

    return result


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
    """Search for code symbols by name, keyword, or file path.

    Uses hybrid BM25 + TF-IDF vector search when embeddings are available
    (run ``embed_graph`` first).  Falls back to FTS5 / LIKE when no TF-IDF
    vectors are found.
    """
    from graphsift.hybrid_search import HybridSearcher

    root = params.get("root_path", os.getcwd())
    query = params.get("query", "")
    kind = params.get("kind")
    limit = int(params.get("limit", 20))

    store = _get_store(root)

    # Quick probe: do any nodes have TF-IDF vectors?
    has_embeddings = _check_embed_version(store)

    if has_embeddings:
        # Load all nodes and use hybrid search.
        all_nodes = store.load_nodes()
        searcher = HybridSearcher(alpha=0.3)
        scored = searcher.search(query, all_nodes, top_k=limit * 2)

        if kind:
            kind_lower = kind.lower()
            scored = [(n, s) for n, s in scored if n.kind.value == kind_lower]

        results = [
            {
                "name": n.name,
                "qualified_name": n.qualified_name,
                "kind": n.kind.value,
                "file": n.file_path,
                "line": n.line_start,
                "language": n.language.value,
                "community_id": n.community_id,
                "score": round(s, 4),
            }
            for n, s in scored[:limit]
        ]
    else:
        # Fall back to FTS5 / LIKE (original behaviour).
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


def _check_embed_version(store: Any) -> bool:
    """Return True if the store has TF-IDF embeddings (embed_version == '1')."""
    try:
        from graphsift.adapters.storage import GraphStore

        with store._lock:
            row = store._conn.execute(
                "SELECT value FROM graph_meta WHERE key='embed_version' LIMIT 1"
            ).fetchone()
            return row is not None and row["value"] == "1"
    except Exception:
        return False


def _tool_list_repos(params: dict) -> dict:
    """List all registered repositories in the graphsift registry."""
    registry_path = Path.home() / ".graphsift" / "registry.json"
    if not registry_path.exists():
        return {"status": "ok", "summary": "0 registered repository(ies)", "repos": []}

    try:
        registry = json.loads(SafeFileIO.read(registry_path))
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
        registry = json.loads(SafeFileIO.read(registry_path))
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
# Session memory tools
# ---------------------------------------------------------------------------


def _tool_save_review_feedback(params: dict) -> dict:
    """Save user feedback on context quality (1-5 rating).

    Feedback accumulates over time to improve ranking weights.
    """
    root = params.get("root_path", os.getcwd())
    context_id = params.get("context_id")
    rating = params.get("rating")
    notes = params.get("notes", "")

    if context_id is None or rating is None:
        return {"error": "context_id and rating are required."}

    try:
        rating = int(rating)
    except (TypeError, ValueError):
        return {"error": "rating must be an integer 1-5."}

    if rating < 1 or rating > 5:
        return {"error": "rating must be between 1 and 5."}

    store = _get_store(root)
    try:
        store.save_review_feedback(context_id, rating, notes)
        return {"status": "saved", "context_id": context_id, "rating": rating}
    except Exception as exc:
        return {"error": str(exc)}


def _tool_get_context_quality(params: dict) -> dict:
    """Return aggregate quality stats from review feedback.

    Returns count, average rating, distribution, and recent feedback.
    """
    root = params.get("root_path", os.getcwd())
    store = _get_store(root)
    try:
        stats = store.get_context_quality_stats()
        return {"status": "ok", **stats}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Compress / Analytics tools
# ---------------------------------------------------------------------------


def _tool_compress_output(params: dict) -> dict:
    """Compress command output to save 60-90% tokens before sending to LLM."""
    from graphsift.compress import compress, detect_type as detect_command

    text = params.get("text", "")
    if not text:
        return {"error": "text parameter is required"}

    command_type = params.get("command_type", "auto")
    ultra = params.get("ultra", False)

    original_chars = len(text)

    if command_type == "auto":
        command_type = detect_command(text)

    compressed = compress(text, command=command_type, ultra=ultra)
    compressed_chars = len(compressed)
    savings_pct = round((1 - compressed_chars / max(original_chars, 1)) * 100, 1)

    return {
        "compressed": compressed,
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "savings_pct": savings_pct,
        "command_type": command_type,
    }


def _tool_token_gain(params: dict) -> dict:
    """Show token savings analytics — total calls, tokens saved, estimated cost savings, daily breakdown."""
    from graphsift.analytics import gain as analytics_gain

    root = params.get("root_path", os.getcwd())
    result = analytics_gain(project_root=root, format="json")
    return json.loads(result)


def _tool_token_discover(params: dict) -> dict:
    """Find missed token-saving opportunities — which commands would benefit most from compression."""
    from graphsift.analytics import discover as analytics_discover

    root = params.get("root_path", os.getcwd())
    return analytics_discover(project_root=root)


def _tool_suggest_fixes(params: dict) -> dict:
    """Run auto-fix analysis and return prioritized fix suggestions.

    Analyzes the dependency graph for:
      - Unused imports
      - Missing type annotations
      - Long functions, long parameter lists, large classes
      - Dependency cycles
      - Dead code

    All findings are read-only suggestions — no files are modified.
    """
    from graphsift.auto_fix import FixSuggester  # noqa: PLC0415

    root = params.get("root_path", os.getcwd())
    changed_files = params.get("changed_files")
    min_confidence = float(params.get("min_confidence", 0.0))

    builder, source_map = _get_builder(root)
    graph = getattr(builder, "_graph", None)
    if not graph or not source_map:
        return {"error": "Graph not built yet. Call build_graph first."}

    suggester = FixSuggester(graph, source_map=source_map)
    report = suggester.analyze(changed_files=changed_files)

    # Filter by min_confidence
    filtered = [
        s for s in report.suggestions
        if s.confidence >= min_confidence
    ]

    return {
        "suggestions": [s.model_dump() for s in filtered],
        "total_issues": len(filtered),
        "total_all": report.total_issues,
        "by_severity": report.by_severity,
        "by_category": report.by_category,
        "summary": report.summary,
    }


# ---------------------------------------------------------------------------
# New optimization tool implementations
# ---------------------------------------------------------------------------


def _tool_tool_budgets(params: dict) -> dict:
    """Apply tool budget line caps to bash/read/grep output."""
    from graphsift.tool_budgets import ToolBudget

    tool = params.get("tool", "bash")
    text = params.get("text", "")
    extract_structured = params.get("extract_structured", False)

    if not text:
        return {"error": "text parameter is required"}

    original_chars = len(text)
    budget = ToolBudget()
    capped = budget.apply(tool, text, extract_structured=extract_structured)

    return {
        "capped": capped,
        "original_chars": original_chars,
        "capped_chars": len(capped),
        "savings_pct": round((1 - len(capped) / max(original_chars, 1)) * 100, 1),
        "budget_applied": budget.get_budget(tool),
    }


def _tool_read_cache(params: dict) -> dict:
    """Fingerprint file reads and return stubs on duplicates."""
    from graphsift.read_cache import ReadCache

    _cache_inst: ReadCache = getattr(_tool_read_cache, "_cache", None)
    if _cache_inst is None:
        _cache_inst = ReadCache()
        _tool_read_cache._cache = _cache_inst

    path = params.get("path", "")
    content = params.get("content", "")

    if not path:
        return {"error": "path parameter is required"}

    result = _cache_inst.read(path, lambda: content)

    return {
        "result": result,
        "is_stub": "fingerprint match" in result,
        "stubs_served": _cache_inst.stubs_served,
    }


def _tool_verify_file(params: dict) -> dict:
    """Run syntax check on a changed file."""
    from graphsift.verify_hooks import Verifier

    file_path = params.get("file_path", "")
    project_root = params.get("project_root", "")

    if not file_path:
        return {"error": "file_path parameter is required"}

    verifier = Verifier(project_root=project_root)
    result = verifier.check(file_path)

    return {
        "file": result.file,
        "passed": result.passed,
        "syntax_ok": result.syntax_ok,
        "syntax_error": result.syntax_error,
    }


def _tool_check_evidence(params: dict) -> dict:
    """Scan text for file:line citations and validate them."""
    from graphsift.evidence_check import EvidenceChecker

    text = params.get("text", "")
    project_root = params.get("project_root", "")

    if not text:
        return {"error": "text parameter is required"}

    checker = EvidenceChecker(project_root=project_root)
    citations = checker.check_response(text)

    return {
        "total_citations": len(citations),
        "valid": [c for c in citations if c.valid],
        "invalid": [{"raw": c.raw, "file": c.file_path, "line": c.line, "error": c.error} for c in citations if not c.valid],
    }


def _tool_audit_strategy_claims(params: dict) -> dict:
    """Audit AI-generated trading strategy text for hallucinated claims.

    Extracts profit / win-rate / ROI / signal / guarantee claims and verifies
    them against real-time proven reference data. Catches the classic
    hallucination where a backtested figure (e.g. Rs.44,00,000) is presented
    as a live/proven result when the real proven P&L is far lower
    (e.g. Rs.4,00,000). Returns per-claim status (verified | synthetic |
    contradicted | unverifiable) plus a 0-100 hallucination score.
    """
    from graphsift.guard import JsonBacktestProvider, StrategyGuard

    text = params.get("text", "")
    reference = params.get("reference", "")

    if not text:
        return {"error": "text parameter is required"}

    guard = StrategyGuard(provider=JsonBacktestProvider(reference or None))
    report = guard.audit(text)
    data = report.to_dict()
    data["summary"] = report.summary()
    return data


def _tool_guard_strategy_text(params: dict) -> dict:
    """Enforce the strategy guard: rewrite text to neutralize risky claims.

    mode="mark" appends [UNVERIFIED]/[CONTRADICTED] to risky claims;
    mode="strip" removes them; mode="report" leaves text unchanged and only
    returns the audit. Returns rewritten_text + full audit report.
    """
    from graphsift.guard import JsonBacktestProvider, StrategyGuard

    text = params.get("text", "")
    reference = params.get("reference", "")
    mode = params.get("mode", "mark")

    if not text:
        return {"error": "text parameter is required"}

    guard = StrategyGuard(provider=JsonBacktestProvider(reference or None))
    rewritten, report = guard.enforce(text, mode=mode)
    data = report.to_dict()
    data["rewritten_text"] = rewritten
    return data


def _tool_build_strategy_prompt(params: dict) -> dict:
    """Build an anti-hallucination grounding prompt for strategy generation.

    Prepend the returned prompt to a strategy-generation call so the AI is
    forced to: only quote real/sourced figures, label every number
    [VERIFIED-REAL] or [UNKNOWN], never guarantee returns, give
    best/expected/worst scenarios, and refuse to inflate beyond reference data.
    """
    from graphsift.guard import JsonBacktestProvider, StrategyGuard

    strategy_request = params.get("strategy_request", "")
    reference = params.get("reference", "")
    include_reference = params.get("include_reference", True)

    guard = StrategyGuard(provider=JsonBacktestProvider(reference or None))
    prompt = guard.build_grounding_prompt(
        strategy_request=strategy_request,
        include_reference=bool(include_reference),
    )
    return {"prompt": prompt, "characters": len(prompt)}


def _tool_generate_fix_prompt(params: dict) -> dict:
    """PLANNING TOOL — reads the buggy file, returns a step-by-step execution
    plan with verification. YOU MUST EXECUTE each step in order. Does NOT
    modify files automatically."""
    from graphsift.prompt_templates import FixBugTemplate
    from pathlib import Path

    bug = params.get("bug", "")
    file = params.get("file", "")
    line = params.get("line")

    # Read actual file for context
    file_context = ""
    file_path = Path(file)
    if file_path.exists():
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            start = max(0, (line or 1) - 15)
            end = min(len(lines), (line or 1) + 15)
            file_context = "\n".join(lines[start:end])
        except Exception:
            file_context = "(could not read file)"

    # Build the test file name
    test_file = file.replace(".py", "_test.py").rsplit("/", 1)[-1]
    if not test_file.endswith(".py"):
        test_file = "test_" + test_file

    tpl = FixBugTemplate()
    prompt = tpl.render(
        bug=bug,
        file=file,
        line=line,
        expected=params.get("expected", ""),
        actual=params.get("actual", ""),
    )

    return {
        "status": "PLAN_READY",
        "action_required": True,
        "plan_summary": f"Fix bug in {file}: {bug[:80]}",
        "execution_steps": [
            {
                "step": 1,
                "action": "read",
                "target": file,
                "instruction": f"Read {file} around line {line or 'the relevant area'} to understand the code structure",
            },
            {
                "step": 2,
                "action": "edit",
                "target": file,
                "instruction": f"Apply the fix for: {bug[:200]}. Use Edit to modify the file.",
            },
            {
                "step": 3,
                "action": "write",
                "target": f"tests/{test_file}",
                "instruction": f"Add a regression test that reproduces the bug and confirms the fix",
            },
            {
                "step": 4,
                "action": "verify",
                "target": "ALL",
                "instruction": "Read back the modified file(s). Confirm the fix is correct and no side effects were introduced. Check the test captures the original bug.",
            },
        ],
        "verification_criteria": [
            f"The bug is fixed in {file}",
            "No public API signatures changed",
            "Regression test exists that reproduces the original bug",
            "No new imports or dependencies added without justification",
        ],
        "context": {
            "file": file,
            "file_exists": file_path.exists(),
            "surrounding_code": file_context,
        },
        "implementation_prompt": prompt,
    }


def _tool_generate_feature_prompt(params: dict) -> dict:
    """PLANNING TOOL — reads target files, returns a step-by-step execution
    plan with verification. YOU MUST EXECUTE each step in order. Does NOT
    create files automatically."""
    from graphsift.prompt_templates import AddFeatureTemplate
    from pathlib import Path

    file_list = params.get("files") or []
    feature = params.get("feature", "")

    # Read existing files for context
    file_contexts = {}
    for f in file_list:
        fp = Path(f)
        if fp.exists():
            try:
                file_contexts[f] = fp.read_text(encoding="utf-8")[:3000]
            except Exception:
                file_contexts[f] = "(could not read file)"
        else:
            file_contexts[f] = "(file does not exist yet — will be created)"

    tpl = AddFeatureTemplate()
    prompt = tpl.render(
        feature=feature,
        files=file_list,
        acceptance_criteria=params.get("acceptance_criteria"),
    )

    return {
        "status": "PLAN_READY",
        "action_required": True,
        "plan_summary": f"Add feature: {feature[:100]}",
        "execution_steps": [
            {
                "step": 1,
                "action": "read",
                "target": ", ".join(file_list),
                "instruction": f"Read existing files to understand current code patterns before writing",
            },
            {
                "step": 2,
                "action": "edit",
                "target": file_list[0] if file_list else "(new file)",
                "instruction": f"Implement {feature[:200]} following the patterns identified in step 1",
            },
            {
                "step": 3,
                "action": "write" if len(file_list) > 1 else "verify",
                "target": file_list[1] if len(file_list) > 1 else "",
                "instruction": f"Create/modify additional files as needed for the feature",
            },
            {
                "step": 4,
                "action": "verify",
                "target": "ALL",
                "instruction": "Read back modified files. Confirm the feature works and meets all acceptance criteria. Check error handling and edge cases.",
            },
        ],
        "verification_criteria": [
            f"Feature is implemented: {feature[:80]}",
            "All target files exist with correct content",
            "Error handling covers empty, null, error, and edge cases",
            "No breaking changes to existing public APIs",
        ],
        "context": {
            "files": file_list,
            "file_contents": file_contexts,
        },
        "implementation_prompt": prompt,
    }


def _tool_generate_refactor_prompt(params: dict) -> dict:
    """PLANNING TOOL — reads target files, returns a step-by-step execution
    plan with verification. YOU MUST EXECUTE each step in order. Does NOT
    modify files automatically."""
    from graphsift.prompt_templates import RefactorTemplate
    from pathlib import Path

    file_list = params.get("files") or []
    target = params.get("target", "")
    goal = params.get("goal", "")

    # Read existing files for context
    file_contexts = {}
    for f in file_list:
        fp = Path(f)
        if fp.exists():
            try:
                file_contexts[f] = fp.read_text(encoding="utf-8")[:3000]
            except Exception:
                file_contexts[f] = "(could not read file)"
        else:
            file_contexts[f] = "(file does not exist)"

    tpl = RefactorTemplate()
    prompt = tpl.render(
        target=target,
        goal=goal,
        files=file_list,
    )

    return {
        "status": "PLAN_READY",
        "action_required": True,
        "plan_summary": f"Refactor {target}: {goal[:100] if goal else 'improve structure'}",
        "execution_steps": [
            {
                "step": 1,
                "action": "read",
                "target": ", ".join(file_list),
                "instruction": f"Read all files to understand current structure, callers, and dependencies",
            },
            {
                "step": 2,
                "action": "edit",
                "target": file_list[0] if file_list else target,
                "instruction": f"Apply the refactoring: {goal[:200]}. Use Edit to modify files while preserving behavior.",
            },
            {
                "step": 3,
                "action": "edit" if len(file_list) > 1 else "verify",
                "target": file_list[1] if len(file_list) > 1 else "",
                "instruction": "Update all callers and references across affected files",
            },
            {
                "step": 4,
                "action": "verify",
                "target": "ALL",
                "instruction": "Read back all modified files. Confirm behavior is preserved. Check no references are broken. Verify all callers updated.",
            },
        ],
        "verification_criteria": [
            f"Refactoring complete: {target}",
            "Behavior is preserved — no API signature changes",
            "All existing callers are updated to use new code",
            "No dead code or stale references remain",
            "Tests still pass (if applicable)",
        ],
        "context": {
            "files": file_list,
            "file_contents": file_contexts,
        },
        "implementation_prompt": prompt,
    }


def _tool_set_task_type(params: dict) -> dict:
    """Activate a task-type-driven operation manual for the current session.

    Loads the manual matching *task_type*, expands its parent hierarchy
    recursively, and returns the active manual chain with prompts and
    enabled tools.  Subsequent tool calls can use ``list_manuals`` to
    see what's available and ``get_active_prompts`` for the methodology
    text.
    """
    from graphsift.prompt_templates import ManualSelector

    root = params.get("root_path", os.getcwd())
    task_type = params.get("task_type", "")

    if not task_type:
        return {"error": "task_type is required"}

    selector = _get_manual_selector(root)

    try:
        chain = selector.activate(task_type)
    except ValueError as exc:
        return {"error": str(exc)}

    return {
        "task_type": task_type,
        "active_manuals": [
            {
                "id": m["id"],
                "name": m.get("name", ""),
                "description": m.get("description", ""),
                "parent": m.get("parent"),
                "phases": m.get("phases", []),
            }
            for m in chain
        ],
        "tools_enabled": selector.get_active_tools(),
        "prompts": selector.get_active_prompts(),
    }


def _tool_list_manuals(params: dict) -> dict:
    """List all available operation manuals with descriptions.

    Returns every manual found under the ``manuals/`` directory
    regardless of current activation state.
    """
    root = params.get("root_path", os.getcwd())
    selector = _get_manual_selector(root)
    manuals = selector.list_manuals()

    return {"manuals": manuals, "total": len(manuals)}


def _tool_list_plugins(params: dict) -> dict:
    """List all registered external command plugins."""
    root = params.get("root_path", os.getcwd())
    registry = _get_plugin_registry(root)
    plugins = registry.list_plugins()
    return {"plugins": plugins, "total": len(plugins)}


def _tool_run_plugin(params: dict) -> dict:
    """Execute an external command plugin via JSON stdin/stdout protocol."""
    root = params.get("root_path", os.getcwd())
    plugin_id = params.get("plugin_id", "")
    arguments = params.get("arguments", {})
    timeout = params.get("timeout_ms", 30000)

    if not plugin_id:
        return {"success": False, "error": "Missing plugin_id"}

    registry = _get_plugin_registry(root)
    result = registry.execute(plugin_id, arguments, timeout_ms=timeout)
    return result


def _tool_terse_mode(params: dict) -> dict:
    """Generate terse-mode instructions for a given level."""
    level = params.get("level", "full")
    prompt = params.get("prompt", "")

    instructions = {
        "lite": (
            "[Terse:lite] Be brief but polite. Use short sentences. "
            "No filler words. Preserve code/commands/errors verbatim."
        ),
        "full": (
            "[Terse:full] Use sentence fragments, not full prose. "
            "No hedging, transitional phrases, or politeness. "
            "Preserve code/commands/errors verbatim."
        ),
        "ultra": (
            "[Terse:ultra] Minimum viable response. "
            "Just code, file paths, commands, and errors. "
            "No explanations unless explicitly asked."
        ),
    }

    instruction = instructions.get(level, instructions["full"])
    return {
        "full_prompt": f"{instruction}\n\n{prompt}",
        "level": level,
        "instruction": instruction,
    }


def _tool_should_compact(params: dict) -> dict:
    """Check whether conversation should be compacted."""
    from graphsift.compact_context import ConversationCompactor

    current = params.get("current_tokens", 0)
    max_context = params.get("max_context", 200000)
    threshold_pct = params.get("threshold_pct", 80)

    compactor = ConversationCompactor(max_context_tokens=max_context)
    should = compactor.should_compact(current, threshold_pct)

    threshold_tokens = max_context * threshold_pct // 100
    usage_pct = round((current / max(max_context, 1)) * 100, 1)

    return {
        "should_compact": should,
        "current_tokens": current,
        "max_context": max_context,
        "threshold_tokens": threshold_tokens,
        "usage_pct": usage_pct,
        "recommendation": "compact now" if should else "within budget",
    }


# ---------------------------------------------------------------------------
# Auto-trigger helpers
# ---------------------------------------------------------------------------


def _detect_command_type_from_text(text: str) -> str | None:
    """Detect CLI output type from content using regex.

    Tries the built-in ``graphsift.compress.detect_type`` first (18+ types),
    then falls back to additional pattern coverage for common tools.
    Returns the type string (e.g. ``"pytest"``, ``"git_diff"``) or ``None``.
    """
    from graphsift.compress import detect_type

    if not text or not text.strip():
        return None

    # Try the built-in detector first (covers 18+ types)
    try:
        detected = detect_type(text)
        if detected and detected != "generic":
            return detected
    except Exception:
        pass

    # Fallback: additional patterns for key CLI tools
    import re  # noqa: PLC0415

    head = text[:1000]

    # pytest
    if re.search(r"(?m)(?:FAILED|ERROR|PASSED|test_|assert)", head):
        return "pytest"
    # git diff
    if re.search(r"(?m)(?:^diff --git|^index |^--- a/)", head):
        return "git_diff"
    # git status
    if re.search(r"(?m)(?:^On branch|nothing to commit|^modified:|^new file:|^deleted:)", head):
        return "git_status"
    # eslint with line:col pattern
    if re.search(r"(?m)\d+:\d+\s+(?:error|warning|rule:)", head):
        return "eslint"
    # npm
    if re.search(r"(?m)(?:ERR!|npm ERR|npm WARN|npm notice)", head):
        return "npm"
    # docker
    if re.search(r"(?m)(?:CONTAINER|IMAGE ID|STATUS|^REPOSITORY)", head):
        return "docker"
    # grep -- file:line pattern
    if re.search(r"(?m)^[^:\n\r]+\.[a-zA-Z]{1,6}:\d+:", head):
        return "grep"

    return None


def _should_auto_verify(file_path: str) -> bool:
    """Check whether *file_path* is eligible for auto-verification.

    Returns ``True`` when the extension is one of ``.py``, ``.js``, ``.ts``,
    ``.go``, ``.rs`` **and** the path does not live inside a skipped directory
    (``.git``, ``node_modules``, ``__pycache__``, ``venv``, ``dist``).
    """
    if not file_path:
        return False

    valid_exts = {".py", ".js", ".ts", ".go", ".rs"}
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_exts:
        return False

    skip_dirs = {".git", "node_modules", "__pycache__", "venv", "dist"}
    parts = Path(file_path).parts
    if any(part in skip_dirs for part in parts):
        return False

    return True


def _tool_uiux_design_system(params: dict) -> dict:
    """Generate a complete UI/UX design system for a product/industry query.

    Delegates to the MIT-licensed ui-ux-pro-max-skill engine (installed via
    `graphsift uiux --install`). Returns style, WCAG-tested palette, typography,
    motion presets, anti-patterns, key effects and the pre-delivery checklist.
    """
    from graphsift.uiux import run_json

    query = params.get("query", "").strip()
    if not query:
        return {"error": "query parameter is required (e.g. 'saas analytics dashboard')"}

    argv = ["--design-system", "--json"]
    if params.get("project_name"):
        argv += ["--project-name", str(params["project_name"])]
    if params.get("format"):
        argv += ["--format", str(params["format"])]
    for dial, flag in (("variance", "--variance"), ("motion", "--motion"), ("density", "--density")):
        val = params.get(dial)
        if val is not None:
            try:
                argv += [flag, str(int(val))]
            except (TypeError, ValueError):
                return {"error": f"{dial} must be an integer 1-10"}

    result = run_json([query, *argv])
    if isinstance(result, dict) and "design_system" in result:
        return result["design_system"]
    return result


def _tool_uiux_search(params: dict) -> dict:
    """Search the UI/UX design database (styles, colors, typography, UX guidelines, charts, ...).

    `domain` selects a domain (style, color, chart, landing, product, ux,
    typography, google-fonts, icons, gsap, react, web); when omitted the engine
    auto-detects it. Returns ranked BM25 matches from the installed engine.
    """
    from graphsift.uiux import run_json

    query = params.get("query", "").strip()
    if not query:
        return {"error": "query parameter is required (e.g. 'glassmorphism landing page')"}

    argv = ["--json"]
    if params.get("domain"):
        argv += ["--domain", str(params["domain"])]
    if params.get("max_results"):
        try:
            argv += ["--max-results", str(int(params["max_results"]))]
        except (TypeError, ValueError):
            return {"error": "max_results must be an integer"}

    return run_json([query, *argv])


def _tool_uiux_stack_guide(params: dict) -> dict:
    """Framework/stack-specific UI guidelines (react, nextjs, shadcn, html-tailwind, ...).

    Returns Do/Don't guidance, severity and docs URLs for the requested stack.
    When `query` is omitted, the engine's BM25 search needs a query whose terms
    actually appear in the stack's guidelines, so we try a fallback chain
    (stack name, first token, then a generic term) and return the first match.
    """
    from graphsift.uiux import run_json

    stack = params.get("stack", "").strip()
    if not stack:
        return {"error": "stack parameter is required (e.g. 'react', 'nextjs', 'shadcn', 'html-tailwind')"}

    max_results = None
    if params.get("max_results"):
        try:
            max_results = int(params["max_results"])
        except (TypeError, ValueError):
            return {"error": "max_results must be an integer"}

    def _call(q: str) -> dict:
        argv = ["--stack", stack, "--json"]
        if max_results:
            argv += ["--max-results", str(max_results)]
        return run_json([q, *argv])

    query = params.get("query", "").strip()
    if query:
        # An explicit query is authoritative — never rewritten by fallback.
        result = _call(query)
        if isinstance(result, dict) and "query" not in result:
            result["query"] = query
        return result

    # No explicit query: BM25 needs terms present in the stack's guidelines, so
    # try a chain (stack name, first token, generic term) and return first hit.
    for q in (stack, stack.split("-", 1)[0], "components"):
        result = _call(q)
        if not isinstance(result, dict):
            return result
        if "error" in result and result.get("count", 0) == 0:
            return result  # engine error (e.g. missing install) — surface it
        if result.get("count", 0) > 0:
            result["query"] = q
            return result
    return {"error": f"no guidance found for stack '{stack}'", "count": 0}


def _tool_auto_process_output(params: dict) -> dict:
    """Auto-detect and compress CLI output.

    Detects the command type from *text*, then compresses it when the type is
    recognised and the text is longer than 500 characters.  Returns the
    compressed (or original) text together with metadata.
    """
    from graphsift.compress import compress

    text = params.get("text", "")
    if not text:
        return {"error": "text parameter is required"}

    original_chars = len(text)
    command_type = _detect_command_type_from_text(text)

    if command_type and original_chars > 500:
        try:
            compressed = compress(text, command=command_type)
            compressed_chars = len(compressed)
            savings_pct = round((1 - compressed_chars / max(original_chars, 1)) * 100, 1)
            return {
                "was_compressed": True,
                "original_chars": original_chars,
                "compressed_chars": compressed_chars,
                "savings_pct": savings_pct,
                "text": compressed,
            }
        except Exception:
            pass

    return {
        "was_compressed": False,
        "original_chars": original_chars,
        "compressed_chars": original_chars,
        "savings_pct": 0.0,
        "text": text,
    }


def _tool_auto_verify_and_fix(params: dict) -> dict:
    """Verify file syntax and return fix suggestions if errors are found.

    Runs a syntax check on *file_path* via ``verify_file``.  When the check
    fails, also runs ``suggest_fixes`` on the file and attaches the suggestions
    to the result.
    """
    file_path = params.get("file_path", "")
    project_root = params.get("project_root", "")

    if not file_path:
        return {"error": "file_path parameter is required"}

    if not _should_auto_verify(file_path):
        return {
            "error": "File type not supported or in excluded directory",
            "file_path": file_path,
        }

    try:
        verify_result = _tool_verify_file({
            "file_path": file_path,
            "project_root": project_root,
        })

        result = {
            "file": file_path,
            "passed": verify_result.get("passed", False),
            "syntax_ok": verify_result.get("syntax_ok", False),
            "syntax_error": verify_result.get("syntax_error"),
        }

        if not verify_result.get("passed", True):
            fix_result = _tool_suggest_fixes({
                "root_path": project_root or os.getcwd(),
                "changed_files": [file_path],
            })
            if "error" in fix_result:
                result["fix_suggestions"] = []
                result["total_fix_issues"] = 0
            else:
                result["fix_suggestions"] = fix_result.get("suggestions", [])
                result["total_fix_issues"] = fix_result.get("total_issues", 0)

        return result
    except Exception:
        return {"file": file_path, "passed": True, "syntax_ok": True, "syntax_error": None}


# ---------------------------------------------------------------------------
# Batch analysis — run multiple graph commands in a single MCP call
# ---------------------------------------------------------------------------


def _run_batch_command(tool_name: str, cmd_params: dict) -> tuple[str, dict, dict]:
    """Run a single tool command, returning (tool_name, original_params, result_dict).

    All exceptions are caught and returned as error results — never propagates
    so one failing command does not affect others in the same step.
    """
    if tool_name not in _TOOLS:
        return tool_name, cmd_params, {"error": f"Unknown tool: {tool_name}"}

    try:
        result = _TOOLS[tool_name]["fn"](cmd_params)
        return tool_name, cmd_params, result
    except Exception as exc:  # noqa: BLE001
        logger.exception("batch command %s failed", tool_name)
        return tool_name, cmd_params, {"error": str(exc)}


def _tool_batch_analyze(params: dict) -> dict:
    """Run multiple graph commands in one turn with step-based ordering.

    Commands are grouped by *step* number — all commands in the same step
    execute in parallel; steps execute sequentially (step 1 completes before
    step 2 starts).  If one command in a step fails, the other commands in
    that step still complete normally and the error is recorded in the results.

    Supported commands: any registered MCP tool (detect_cycles, detect_dead_code,
    get_impact, list_communities, list_flows, get_architecture_overview,
    graph_status, list_files, get_impact_radius, get_affected_flows, …).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

    root_path = params.get("root_path", "")
    commands = params.get("commands", [])

    if not commands:
        return {"error": "No commands provided.", "results": {}}

    # Group commands by step number
    steps: dict[int, list[dict]] = {}
    for cmd in commands:
        step = int(cmd.get("step", 1))
        steps.setdefault(step, []).append(cmd)

    results: dict[str, list[dict]] = {}

    for step_num in sorted(steps.keys()):
        step_cmds = steps[step_num]

        with ThreadPoolExecutor(max_workers=min(len(step_cmds), 8)) as executor:
            futures = []
            for cmd in step_cmds:
                tool_name = cmd.get("tool", "")
                cmd_params = dict(cmd.get("params", {}))

                # Inject root_path default from batch-level root_path
                if root_path:
                    cmd_params.setdefault("root_path", root_path)

                futures.append(
                    executor.submit(_run_batch_command, tool_name, cmd_params)
                )

            for future in as_completed(futures):
                try:
                    tool_name_res, cmd_params_res, cmd_result = future.result()
                except Exception as exc:  # noqa: BLE001
                    # Should not happen — _run_batch_command catches everything
                    tool_name_res = "unknown"
                    cmd_params_res = {}
                    cmd_result = {"error": f"Unexpected batch error: {exc}"}

                entry = {
                    "tool": tool_name_res,
                    "params": cmd_params_res,
                    "result": cmd_result,
                }
                results.setdefault(tool_name_res, []).append(entry)

    return {
        "results": results,
        "total_commands": len(commands),
        "total_steps": len(steps),
    }


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS = {
    "auto_process_output": {
        "fn": _tool_auto_process_output,
        "description": (
            "Auto-detect and compress CLI output, saving 60-97% tokens. "
            "Detects 20+ command types (pytest, git, eslint, npm, docker, grep, etc.)"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "CLI output text to compress"},
            },
            "required": ["text"],
        },
    },
    "uiux_design_system": {
        "fn": _tool_uiux_design_system,
        "description": (
            "Generate a complete UI/UX design system for a product/industry query "
            "(style, WCAG-tested palette, typography, motion presets, anti-patterns, "
            "pre-delivery checklist). Delegates to the MIT-licensed ui-ux-pro-max-skill "
            "engine. Use before writing UI code and when reviewing visual design quality."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Product/industry/keywords, e.g. 'saas analytics dashboard'"},
                "project_name": {"type": "string", "description": "Project name for the design system output"},
                "format": {"type": "string", "enum": ["ascii", "markdown"], "description": "Output format (default ascii)"},
                "variance": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Design variance dial: 1=centered/minimal, 10=bold/asymmetric"},
                "motion": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Motion intensity dial: 1=subtle, 10=complex (pulls a GSAP snippet)"},
                "density": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Visual density dial: 1=spacious, 10=dense/dashboard"},
            },
            "required": ["query"],
        },
    },
    "uiux_search": {
        "fn": _tool_uiux_search,
        "description": (
            "Search the UI/UX design database (styles, colors, typography, UX guidelines, "
            "charts, landing patterns, icons, GSAP motion). domain selects the domain; "
            "when omitted the engine auto-detects it. Returns ranked BM25 matches."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords, e.g. 'glassmorphism landing page'"},
                "domain": {"type": "string", "enum": ["style", "color", "chart", "landing", "product", "ux", "typography", "google-fonts", "icons", "gsap", "react", "web"], "description": "Domain to search (default: auto-detect)"},
                "max_results": {"type": "integer", "description": "Max results (default 3)"},
            },
            "required": ["query"],
        },
    },
    "uiux_stack_guide": {
        "fn": _tool_uiux_stack_guide,
        "description": (
            "Framework/stack-specific UI guidelines (react, nextjs, shadcn, html-tailwind, "
            "vue, svelte, flutter, swiftui, ...). Returns Do/Don't guidance, severity "
            "and docs URLs for the requested stack."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "stack": {"type": "string", "description": "Stack name, e.g. 'react', 'nextjs', 'shadcn', 'html-tailwind'"},
                "query": {"type": "string", "description": "Optional topic keyword (default: 'general best practices')"},
                "max_results": {"type": "integer", "description": "Max results (default 3)"},
            },
            "required": ["stack"],
        },
    },
    "auto_verify_and_fix": {
        "fn": _tool_auto_verify_and_fix,
        "description": (
            "Verify file syntax and return fix suggestions if errors are found. "
            "Combines syntax checking (Python compile / node --check) with "
            "graph-based auto-fix analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to verify"},
                "project_root": {"type": "string", "description": "Repo root (default: cwd)"},
            },
            "required": ["file_path"],
        },
    },
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
    "prune_refs": {
        "fn": _tool_prune_refs,
        "description": (
            "Scan for stale references to deleted files and optionally auto-fix. "
            "After files are deleted, detects import statements, symbol references, "
            "and path references in remaining source files that point to deleted components. "
            "Use fix=true to auto-remove stale import lines (creates .bak backups)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_root": {"type": "string", "description": "Repo root directory (default: cwd)"},
                "deleted_paths": {"type": "array", "items": {"type": "string"}, "description": "Absolute paths of deleted files"},
                "fix": {"type": "boolean", "description": "Auto-remove stale import lines (default false)"},
            },
            "required": ["deleted_paths"],
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
        "description": "Rename preview, dead-code detection, or suggestions. mode: rename | dead_code | suggest. Dead code results are priority-scored.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string"},
                "mode": {"type": "string", "enum": ["rename", "dead_code", "suggest"]},
                "old_name": {"type": "string", "description": "For rename mode"},
                "new_name": {"type": "string", "description": "For rename mode"},
                "kind": {"type": "string", "description": "For dead_code: function | class | method"},
                "file_pattern": {"type": "string"},
                "prioritize": {"type": "boolean", "description": "Apply priority scoring (default true)", "default": True},
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
    "compress_output": {
        "fn": _tool_compress_output,
        "description": (
            "Compress command output to save 60-90% tokens before sending to LLM. "
            "Supports auto-detection of 18+ command types (pytest, cargo, go test, "
            "jest, eslint, git, npm, docker, kubectl, aws, etc.)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Raw command output to compress"},
                "command_type": {"type": "string", "description": "Command type hint, e.g. pytest, cargo, git-status (default: auto)"},
                "ultra": {"type": "boolean", "description": "Ultra-compact mode — more aggressive filtering (default false)"},
            },
            "required": ["text"],
        },
    },
    "token_gain": {
        "fn": _tool_token_gain,
        "description": (
            "Show token savings analytics — total calls, tokens saved, "
            "estimated cost savings, daily breakdown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root directory (default: cwd)"},
            },
        },
    },
    "token_discover": {
        "fn": _tool_token_discover,
        "description": (
            "Find missed token-saving opportunities — which commands "
            "would benefit most from compression."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root directory (default: cwd)"},
            },
        },
    },
    "detect_cycles": {
        "fn": _tool_detect_cycles,
        "description": "Detect circular dependencies (import/call cycles) in the codebase using Tarjan's SCC algorithm.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Repository root path."},
            },
        },
    },
    "detect_dead_code": {
        "fn": _tool_detect_dead_code,
        "description": "Find potentially unreachable code via BFS reachability from entry points. Results are priority-scored when large.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "root": {"type": "string", "description": "Repository root path."},
                "kind": {"type": "string", "enum": ["function", "class", "method"], "description": "Filter by node kind."},
                "entry_points": {"type": "array", "items": {"type": "string"}, "description": "List of entry-point file paths."},
                "prioritize": {"type": "boolean", "description": "Apply priority scoring (default true).", "default": True},
                "max_results": {"type": "integer", "description": "Max results to return (0 = unlimited, default 0).", "default": 0},
            },
        },
    },
    "save_review_feedback": {
        "fn": _tool_save_review_feedback,
        "description": (
            "Save a 1-5 rating on context selection quality. "
            "Feedback accumulates across sessions to improve ranking weights."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root directory (default: cwd)"},
                "context_id": {"type": "integer", "description": "ID from session_memory (returned in get_context metadata)"},
                "rating": {"type": "integer", "description": "Quality rating 1-5 (5=best)"},
                "notes": {"type": "string", "description": "Optional free-text notes"},
            },
            "required": ["context_id", "rating"],
        },
    },
    "get_context_quality": {
        "fn": _tool_get_context_quality,
        "description": (
            "Return aggregate context quality stats from all review feedback. "
            "Includes total count, average rating, rating distribution, and recent feedback."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root directory (default: cwd)"},
            },
        },
    },
    "suggest_fixes": {
        "fn": _tool_suggest_fixes,
        "description": (
            "Run auto-fix analysis on the dependency graph and return prioritized fix suggestions. "
            "Detects unused imports, missing type annotations, overly long functions/param lists, "
            "dependency cycles, and dead code. Read-only — never modifies files. "
            "Results include confidence scores and auto-fixable flags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root directory (default: cwd)"},
                "changed_files": {"type": "array", "items": {"type": "string"}, "description": "Only analyze these files (optional)"},
                "min_confidence": {"type": "number", "description": "Minimum confidence 0-1 (default 0.0)"},
            },
        },
    },
    "tool_budgets": {
        "fn": _tool_tool_budgets,
        "description": "Apply tool budget line caps to bash/read/grep output — saves ~86% on tool output tokens. Caps bash=80, read=300, grep=120 lines, strips ANSI, collapses blanks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "Tool type: 'bash', 'read', or 'grep'"},
                "text": {"type": "string", "description": "Text to cap and compress"},
                "extract_structured": {"type": "boolean", "description": "Extract JSON/XML if detected (default false)"},
            },
            "required": ["tool", "text"],
        },
    },
    "read_cache": {
        "fn": _tool_read_cache,
        "description": "Fingerprint file reads and return stubs on repeat reads to avoid sending duplicate content. Saves ~99% on repeat file reads.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "content": {"type": "string", "description": "Current file content"},
            },
            "required": ["path", "content"],
        },
    },
    "verify_file": {
        "fn": _tool_verify_file,
        "description": "Run syntax check on a changed file. Supports Python (compile) and JS/TS (node --check). Catches errors immediately after code changes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to verify"},
                "project_root": {"type": "string", "description": "Repo root (default: cwd)"},
            },
            "required": ["file_path"],
        },
    },
    "check_evidence": {
        "fn": _tool_check_evidence,
        "description": "Scan text for file:line citations and validate they point to real files. Catches hallucinated file references.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Response text to scan"},
                "project_root": {"type": "string", "description": "Repo root (default: cwd)"},
            },
            "required": ["text"],
        },
    },
    "audit_strategy_claims": {
        "fn": _tool_audit_strategy_claims,
        "description": "Audit AI-generated trading strategy text for hallucinated claims. Extracts profit/win-rate/ROI/signal/guarantee claims and verifies them against real-time proven reference data. Catches the '44 lakh backtest presented as live 4 lakh' collapse. Returns per-claim status (verified|synthetic|contradicted|unverifiable) and a 0-100 hallucination score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "AI-generated trading strategy text to audit"},
                "reference": {"type": "string", "description": "Path to reference JSON (real backtest/live P&L stats). Optional — defaults to built-in demo reference."},
            },
            "required": ["text"],
        },
    },
    "guard_strategy_text": {
        "fn": _tool_guard_strategy_text,
        "description": "Enforce the strategy guard on text. mode=mark appends [UNVERIFIED]/[CONTRADICTED] to risky claims; mode=strip removes them; mode=report returns audit only. Returns rewritten_text + full audit report.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "AI-generated trading strategy text"},
                "reference": {"type": "string", "description": "Path to reference JSON (optional)"},
                "mode": {"type": "string", "enum": ["mark", "strip", "report", "enforce"], "description": "Enforcement mode (default: mark)"},
            },
            "required": ["text"],
        },
    },
    "build_strategy_prompt": {
        "fn": _tool_build_strategy_prompt,
        "description": "Build an anti-hallucination grounding prompt for strategy generation. Forces the AI to only quote real/sourced figures, label every number [VERIFIED-REAL]/[UNKNOWN], never guarantee returns, and give best/expected/worst scenarios. Prepend to a strategy-generation call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategy_request": {"type": "string", "description": "The strategy-generation request to embed"},
                "reference": {"type": "string", "description": "Path to reference JSON (optional)"},
                "include_reference": {"type": "boolean", "description": "Embed the reference stats in the prompt (default true)"},
            },
        },
    },
    "generate_fix_prompt": {
        "fn": _tool_generate_fix_prompt,
        "description": "PLANNING TOOL — reads the buggy file, returns a step-by-step execution plan with 4 steps: [read→edit→write test→verify]. YOU MUST EXECUTE each step in order using Write/Edit and verify before reporting done.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bug": {"type": "string", "description": "Description of the bug"},
                "file": {"type": "string", "description": "File path"},
                "line": {"type": "integer", "description": "Line number"},
                "expected": {"type": "string", "description": "Expected behavior"},
                "actual": {"type": "string", "description": "Actual behavior"},
            },
            "required": ["bug", "file"],
        },
    },
    "generate_feature_prompt": {
        "fn": _tool_generate_feature_prompt,
        "description": "PLANNING TOOL — reads target files, returns a step-by-step execution plan with 4 steps: [read→edit→write→verify]. YOU MUST EXECUTE each step in order using Write/Edit and verify before reporting done.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "feature": {"type": "string", "description": "Feature description"},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Files involved"},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "description": "Acceptance criteria"},
            },
            "required": ["feature"],
        },
    },
    "generate_refactor_prompt": {
        "fn": _tool_generate_refactor_prompt,
        "description": "PLANNING TOOL — reads target files, returns a step-by-step execution plan with 4 steps: [read→edit→update→verify]. YOU MUST EXECUTE each step in order using Edit and verify before reporting done.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Target to refactor"},
                "goal": {"type": "string", "description": "Goal of the refactor"},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Files involved"},
            },
            "required": ["target"],
        },
    },
    "set_task_type": {
        "fn": _tool_set_task_type,
        "description": (
            "Set the active task type to load the relevant operation manual(s). "
            "Manuals provide methodology-specific prompts, tool enablement lists, "
            "and phase guidance. Parent manuals are expanded recursively "
            "(e.g. security_review also loads dependency_audit). "
            "Use list_manuals to see available task types."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "description": (
                        "Task type identifier — one of: dependency_audit, "
                        "dead_code_audit, security_review, refactor_planning, "
                        "architecture_review, performance_audit, code_review"
                    ),
                },
                "root_path": {"type": "string", "description": "Repo root (default: cwd)"},
            },
            "required": ["task_type"],
        },
    },
    "list_manuals": {
        "fn": _tool_list_manuals,
        "description": (
            "List all available operation manuals and their descriptions. "
            "Manuals are task-type-driven methodology guides with prompts, "
            "tool enablement lists, and phase guidance. "
            "Activate one with set_task_type."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root (default: cwd)"},
            },
        },
    },
    "terse_mode": {
        "fn": _tool_terse_mode,
        "description": "Generate terse-mode instructions for a given level. Preserves code/commands/errors, strips filler/hedging/politeness.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "description": "Terseness level: 'lite', 'full' (default), or 'ultra'"},
                "prompt": {"type": "string", "description": "The actual prompt to prefix with terse instructions"},
            },
            "required": ["prompt"],
        },
    },
    "should_compact": {
        "fn": _tool_should_compact,
        "description": "Check whether conversation should be compacted based on current token count vs threshold. Triggers at 80% of max context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "current_tokens": {"type": "integer", "description": "Current conversation token count"},
                "max_context": {"type": "integer", "description": "Max context window size (default 200000)"},
                "threshold_pct": {"type": "integer", "description": "Threshold percentage (default 80)"},
            },
            "required": ["current_tokens"],
        },
    },
    "batch_analyze": {
        "fn": _tool_batch_analyze,
        "description": (
            "Run multiple graph commands in one turn. "
            "Commands are grouped by step number — same-step commands run in parallel, "
            "different steps execute sequentially. "
            "If one command in a step fails, others in that step still complete. "
            "Supports: detect_cycles, detect_dead_code, get_impact, list_communities, "
            "list_flows, get_architecture_overview, graph_status, list_files, and more."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repository root path (provides default for all commands)"},
                "commands": {
                    "type": "array",
                    "description": "List of commands to execute",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string", "description": "Tool name to call (any registered MCP tool)"},
                            "params": {"type": "object", "description": "Parameters to pass to the tool"},
                            "step": {"type": "integer", "description": "Step number for ordering (default: 1). Same-step commands run in parallel."},
                        },
                        "required": ["tool"],
                    },
                },
            },
            "required": ["commands"],
        },
    },
    "list_plugins": {
        "fn": _tool_list_plugins,
        "description": (
            "List all registered external command plugins. "
            "Plugins are third-party analysis tools that register via "
            "manifest-driven JSON stdin/stdout protocol."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root (default: cwd)"},
            },
        },
    },
    "run_plugin": {
        "fn": _tool_run_plugin,
        "description": (
            "Execute an external command plugin via JSON stdin/stdout subprocess protocol. "
            "The plugin receives a JSON request on stdin and returns JSON on stdout. "
            "Returns success, output, error, and duration_ms."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "root_path": {"type": "string", "description": "Repo root (default: cwd)"},
                "plugin_id": {"type": "string", "description": "Plugin ID to execute"},
                "arguments": {"type": "object", "description": "Arguments to pass to the plugin"},
                "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds (default: 30000)"},
            },
            "required": ["plugin_id"],
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
        "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
        "serverInfo": {"name": "graphsift", "version": _GRAPHSIFT_VERSION},
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


# ---------------------------------------------------------------------------
# Resource readers — each returns a dict with "content" and "mimeType"
# ---------------------------------------------------------------------------


def _read_resource_architecture(root: str) -> dict:
    """Read architecture overview resource."""
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
    }
    return {"content": json.dumps(overview, ensure_ascii=False, indent=2), "mimeType": "application/json"}


def _read_resource_graph_stats(root: str) -> dict:
    """Read graph statistics resource."""
    store = _get_store(root)
    db_stats = store.stats()
    return {"content": json.dumps(db_stats, ensure_ascii=False, indent=2), "mimeType": "application/json"}


def _read_resource_community(root: str, community_name: str) -> dict | None:
    """Read a single community resource by name (partial match)."""
    store = _get_store(root)
    communities = store.load_communities()
    name_lower = community_name.lower()
    found = next((c for c in communities if name_lower in c["label"].lower()), None)
    if not found:
        return None
    result = {
        "community_id": found["community_id"],
        "label": found["label"],
        "node_count": found["node_count"],
        "members": found.get("metadata", {}).get("members", []),
    }
    return {"content": json.dumps(result, ensure_ascii=False, indent=2), "mimeType": "application/json"}


def _read_resource_wiki(root: str, community_name: str) -> dict | None:
    """Read a wiki page resource by community name (partial match)."""
    from graphsift.adapters.postprocess import WikiGenerator

    wiki_dir = str(Path(root) / ".graphsift" / "wiki")
    gen = WikiGenerator(wiki_dir)
    content = gen.get_page(community_name)
    if content is None:
        return None
    return {"content": content, "mimeType": "text/markdown"}


def _read_resource_flows(root: str) -> dict:
    """Read execution flows resource."""
    store = _get_store(root)
    with store._lock:
        try:
            rows = store._conn.execute(
                "SELECT * FROM flow_snapshots ORDER BY id DESC LIMIT 100"
            ).fetchall()
        except Exception:
            rows = []

    flows = []
    for row in rows:
        meta = json.loads(row["metadata"] or "{}")
        flows.append({
            "id": row["id"],
            "flow_name": row["flow_name"],
            "entry_point": row["entry_point"],
            "node_count": meta.get("node_count", 0),
            "file_count": meta.get("file_count", 0),
            "criticality": meta.get("criticality", 0.0),
        })
    return {
        "content": json.dumps({"flows": flows, "total": len(flows)}, ensure_ascii=False, indent=2),
        "mimeType": "application/json",
    }


def _read_resource_risk(root: str) -> dict:
    """Read risk-scored files resource."""
    store = _get_store(root)
    risk_index = store.load_risk_index(min_score=0.0)
    return {
        "content": json.dumps({"risk_index": risk_index, "total": len(risk_index)}, ensure_ascii=False, indent=2),
        "mimeType": "application/json",
    }


# ---------------------------------------------------------------------------
# MCP Resource list — enumerate available resources per-root
# ---------------------------------------------------------------------------

_RESOURCE_URI_PREFIX = "graphsift://"


def _list_resources_for_root(root: str) -> list[dict]:
    """List all available resources for a given repo root."""
    repo_hash = hashlib.sha1(root.encode()).hexdigest()[:12]
    store = _get_store(root)
    resources: list[dict] = []

    # Architecture overview
    resources.append({
        "uri": f"graphsift://{repo_hash}/architecture",
        "name": f"Architecture Overview — {os.path.basename(root)}",
        "description": "High-level architecture overview: nodes, edges, communities, and high-risk files.",
        "mimeType": "application/json",
    })

    # Graph stats
    resources.append({
        "uri": f"graphsift://{repo_hash}/graph/stats",
        "name": f"Graph Statistics — {os.path.basename(root)}",
        "description": "Raw graph database statistics: node/edge/file counts and schema version.",
        "mimeType": "application/json",
    })

    # Communities
    communities = store.load_communities()
    for c in communities[:50]:
        resources.append({
            "uri": f"graphsift://{repo_hash}/community/{c['label']}",
            "name": f"Community: {c['label']}",
            "description": f"Community with {c['node_count']} nodes.",
            "mimeType": "application/json",
        })

    # Wiki pages
    wiki_dir = Path(root) / ".graphsift" / "wiki"
    if wiki_dir.is_dir():
        for wiki_file in sorted(wiki_dir.glob("*.md")):
            page_name = wiki_file.stem
            resources.append({
                "uri": f"graphsift://{repo_hash}/wiki/{page_name}",
                "name": f"Wiki: {page_name}",
                "description": f"Wiki page for community '{page_name}'.",
                "mimeType": "text/markdown",
            })

    # Execution flows
    resources.append({
        "uri": f"graphsift://{repo_hash}/flows",
        "name": f"Execution Flows — {os.path.basename(root)}",
        "description": "All detected execution flows sorted by criticality.",
        "mimeType": "application/json",
    })

    # Risk index
    resources.append({
        "uri": f"graphsift://{repo_hash}/risk",
        "name": f"Risk Index — {os.path.basename(root)}",
        "description": "Risk-scored files across the codebase.",
        "mimeType": "application/json",
    })

    return resources


# ---------------------------------------------------------------------------
# MCP resource handlers
# ---------------------------------------------------------------------------


_RESOURCE_READERS: dict[str, callable] = {
    "architecture": _read_resource_architecture,
    "graph/stats": _read_resource_graph_stats,
    "risk": _read_resource_risk,
    "flows": _read_resource_flows,
}


def _resolve_resource_uri(uri: str) -> tuple[str, str, dict | None] | tuple[None, None, str]:
    """Parse a resource URI into (root, path_parts, error_or_none).

    Returns:
        (root, resource_path, None) on success.
        (None, None, error_message) on failure.
    """
    if not uri.startswith(_RESOURCE_URI_PREFIX):
        return None, None, f"Invalid URI scheme: expected '{_RESOURCE_URI_PREFIX}...'"

    path_part = uri[len(_RESOURCE_URI_PREFIX):]
    parts = path_part.split("/", 1)
    if len(parts) < 1 or not parts[0]:
        return None, None, "Missing repo hash in URI"

    repo_hash = parts[0]
    root = _roots_by_hash.get(repo_hash)
    if root is None:
        return None, None, f"Unknown repo hash: {repo_hash}"

    resource_path = parts[1] if len(parts) > 1 else ""
    if not resource_path:
        return None, None, "Missing resource path in URI"

    return root, resource_path, None


def _handle_resources_list(req_id: Any, params: dict) -> None:
    """Handle resources/list — enumerate available resources."""
    resources = []

    known_roots = set(_stores.keys()) | set(_builders.keys())
    if not known_roots:
        known_roots = {os.getcwd()}

    for root in known_roots:
        try:
            resources.extend(_list_resources_for_root(root))
        except Exception as exc:
            logger.warning("graphsift: failed to list resources for %s: %s", root, exc)

    _ok(req_id, {"resources": resources})


def _handle_resources_read(req_id: Any, params: dict) -> None:
    """Handle resources/read — read a resource by URI."""
    uri = params.get("uri", "")
    if not uri:
        _err(req_id, -32602, "Missing 'uri' parameter")
        return

    root, resource_path, error = _resolve_resource_uri(uri)
    if error:
        _err(req_id, -32602, error)
        return

    try:
        if resource_path.startswith("community/"):
            community_name = resource_path[len("community/"):]
            result = _read_resource_community(root, community_name)
        elif resource_path.startswith("wiki/"):
            community_name = resource_path[len("wiki/"):]
            result = _read_resource_wiki(root, community_name)
        else:
            reader = _RESOURCE_READERS.get(resource_path)
            if reader is None:
                _err(req_id, -32602, f"Unknown resource: {resource_path}")
                return
            result = reader(root)

        if result is None:
            _err(req_id, -32602, f"Resource not found: {uri}")
            return

        _ok(req_id, {
            "contents": [{
                "uri": uri,
                "mimeType": result["mimeType"],
                "text": result["content"],
            }],
        })
    except Exception as exc:
        logger.exception("resource read failed: %s", uri)
        _err(req_id, -32602, str(exc))


# ---------------------------------------------------------------------------
# MCP Prompts
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, dict[str, Any]] = {
    "review_code": {
        "name": "review_code",
        "description": "Review code changes for a set of modified files using graphsift's dependency graph.",
        "arguments": [
            {
                "name": "root_path",
                "description": "Repository root directory",
                "required": False,
            },
            {
                "name": "changed_files",
                "description": "List of file paths that were changed",
                "required": True,
            },
            {
                "name": "diff_text",
                "description": "Raw unified diff text for the changes (optional)",
                "required": False,
            },
        ],
    },
    "analyze_impact": {
        "name": "analyze_impact",
        "description": "Analyze the blast radius and impact of changes across the codebase.",
        "arguments": [
            {
                "name": "root_path",
                "description": "Repository root directory",
                "required": False,
            },
            {
                "name": "changed_files",
                "description": "List of file paths that were changed",
                "required": True,
            },
            {
                "name": "max_depth",
                "description": "Maximum traversal depth for impact analysis (default 3)",
                "required": False,
            },
        ],
    },
    "find_issues": {
        "name": "find_issues",
        "description": "Search for potential code issues, dead code, cycles, and refactoring opportunities.",
        "arguments": [
            {
                "name": "root_path",
                "description": "Repository root directory",
                "required": False,
            },
            {
                "name": "focus",
                "description": "Issue focus: cycles, dead_code, large_functions, all (default all)",
                "required": False,
            },
        ],
    },
    "explain_architecture": {
        "name": "explain_architecture",
        "description": "Explain the high-level architecture of this codebase using graphsift's community detection.",
        "arguments": [
            {
                "name": "root_path",
                "description": "Repository root directory",
                "required": False,
            },
            {
                "name": "community_name",
                "description": "Focus on a specific community/module (optional)",
                "required": False,
            },
        ],
    },
}


def _build_review_code_prompt(args: dict) -> list[dict]:
    """Build a review_code prompt message (2026 anti-hallucination patterns)."""
    root = args.get("root_path", os.getcwd())
    changed_files = args.get("changed_files", [])
    diff_text = args.get("diff_text", "")

    text = (
        "# Role: senior code reviewer\n\n"
        "## Task: Review the following code changes\n\n"
        "--- ANTI-HALLUCINATION RULES ---\n"
        "- Tag every finding with [VERIFIED-REAL] (confirmed from diff) "
        "or [UNKNOWN] (speculative)\n"
        "- Do NOT flag issues that don't exist in the diff\n"
        "- If uncertain about a pattern's intent, mark it [UNKNOWN] "
        "rather than assuming it's a bug\n"
        "- After writing your review, self-check: is every claim "
        "traceable to the diff?\n\n"
        f"Repository root: {root}\n"
        f"Files changed: {json.dumps(changed_files, indent=2)}\n"
    )
    if diff_text:
        text += f"\nDiff:\n```diff\n{diff_text}\n```\n"

    text += (
        "\nReview dimensions (in order):\n"
        "1. Correctness — logic errors, race conditions, edge cases\n"
        "2. Security — injection, authz, data leakage, path traversal\n"
        "3. Performance — N+1 queries, unnecessary re-renders, memory\n"
        "4. Maintainability — duplication, naming, complexity\n"
        "5. Testing — adequate coverage for the change\n\n"
        "Output ONLY valid JSON:\n"
        '{"findings": [{"severity": "error|warning|info", '
        '"file": "...", "line": N, '
        '"issue": "[VERIFIED-REAL] description", '
        '"suggestion": "..."}], '
        '"self_review_passed": true}\n'
    )

    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": text,
            },
        },
    ]


def _build_analyze_impact_prompt(args: dict) -> list[dict]:
    """Build an analyze_impact prompt message (2026 anti-hallucination)."""
    root = args.get("root_path", os.getcwd())
    changed_files = args.get("changed_files", [])
    max_depth = args.get("max_depth", 3)

    text = (
        "# Role: senior software architect\n\n"
        "## Task: Analyze impact and blast radius\n\n"
        "--- ANTI-HALLUCINATION RULES ---\n"
        "- Tag every dependency claim with [VERIFIED-REAL] or [UNKNOWN]\n"
        "- Do NOT speculate about dependencies outside the provided context\n"
        "- If you cannot trace a dependency path, say so\n\n"
        f"Repository root: {root}\n"
        f"Changed files: {json.dumps(changed_files, indent=2)}\n"
        f"Max traversal depth: {max_depth}\n\n"
        "Analyze:\n"
        "1. Dependents — which files/modules import from changed files?\n"
        "2. Risk level per file — low/medium/high with justification\n"
        "3. Architectural concerns — coupling, circular deps, violated layers\n"
        "4. Testing priorities — what must be tested after this change\n\n"
        "Output ONLY valid JSON:\n"
        '{"impacted_files": [{"path": "...", "risk": "low|medium|high", '
        '"reason": "[VERIFIED-REAL] ..."}], '
        '"architectural_concerns": [], '
        '"testing_priorities": [], '
        '"self_review_passed": true}\n'
    )

    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": text,
            },
        },
    ]


def _build_find_issues_prompt(args: dict) -> list[dict]:
    """Build a find_issues prompt message (2026 anti-hallucination)."""
    root = args.get("root_path", os.getcwd())
    focus = args.get("focus", "all")

    text = (
        "# Role: senior code auditor\n\n"
        f"## Task: Find code issues in {root}\n"
        f"Focus: {focus}\n\n"
        "--- ANTI-HALLUCINATION RULES ---\n"
        "- Only report issues you can [VERIFY-REAL] from the codebase\n"
        "- Do NOT flag generated/placeholder code as production issues\n"
        "- For every finding, provide a file path and line number\n"
        "- If uncertain, mark [UNKNOWN]\n\n"
        "Search for:\n"
        "- Circular dependencies (import/call cycles) [VERIFIED-REAL]\n"
        "- Dead code (unused exports, functions, classes) [VERIFIED-REAL]\n"
        "- Large functions >50 lines or classes >300 lines\n"
        "- Missing error handling or type annotations\n"
        "- Security-sensitive patterns (hardcoded secrets, missing validation)\n\n"
        "Output ONLY valid JSON:\n"
        '{"issues": [{"file": "...", "line": N, '
        '"type": "cycle|dead_code|large_function|security|type", '
        '"severity": "error|warning|info", '
        '"description": "[VERIFIED-REAL] ...", '
        '"suggestion": "..."}], '
        '"total": 0}\n'
    )

    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": text,
            },
        },
    ]


def _build_explain_architecture_prompt(args: dict) -> list[dict]:
    """Build an explain_architecture prompt message (2026 patterns)."""
    root = args.get("root_path", os.getcwd())
    community_name = args.get("community_name", "")

    text = (
        "# Role: senior software architect\n\n"
        "## Task: Explain codebase architecture\n\n"
        "--- ANTI-HALLUCINATION RULES ---\n"
        "- Tag every architectural claim with [VERIFIED-REAL] "
        "(confirmed from graph analysis) or [UNKNOWN]\n"
        "- Do NOT invent architectural patterns that aren't visible in the graph\n"
        "- If a community has no clear responsibility, say so rather than guessing\n\n"
        f"Repository root: {root}\n"
    )
    if community_name:
        text += f"Focus on community/module: {community_name}\n\n"
    text += (
        "Cover:\n"
        "1. Main modules/communities and their responsibilities [VERIFIED-REAL]\n"
        "2. Data flow between modules (from dependency edges)\n"
        "3. Key entry points and their dependency chains\n"
        "4. Architectural patterns actually used (not aspirational)\n"
        "5. Concrete improvement areas with evidence\n\n"
        "Output ONLY valid JSON:\n"
        '{"modules": [{"name": "...", "responsibility": "...", '
        '"dependencies": [...], "entry_points": [...]}], '
        '"data_flow": "...", '
        '"patterns_used": ["..."], '
        '"improvements": [{"area": "...", "evidence": "[VERIFIED-REAL]", '
        '"suggestion": "..."}], '
        '"self_review_passed": true}\n'
    )

    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": text,
            },
        },
    ]


def _handle_prompts_list(req_id: Any, params: dict) -> None:
    """Handle prompts/list — return available prompts."""
    prompts = []
    for name, spec in _PROMPTS.items():
        prompts.append({
            "name": spec["name"],
            "description": spec["description"],
            "arguments": spec.get("arguments", []),
        })
    _ok(req_id, {"prompts": prompts})


def _handle_prompts_get(req_id: Any, params: dict) -> None:
    """Handle prompts/get — return a specific prompt with rendered messages."""
    name = params.get("name", "")
    arguments = params.get("arguments", {})

    if name not in _PROMPTS:
        _err(req_id, -32602, f"Unknown prompt: {name}")
        return

    _PROMPT_BUILDERS = {
        "review_code": _build_review_code_prompt,
        "analyze_impact": _build_analyze_impact_prompt,
        "find_issues": _build_find_issues_prompt,
        "explain_architecture": _build_explain_architecture_prompt,
    }

    builder = _PROMPT_BUILDERS.get(name)
    if builder is None:
        _err(req_id, -32602, f"No builder for prompt: {name}")
        return

    try:
        messages = builder(arguments)
        _ok(req_id, {
            "description": _PROMPTS[name]["description"],
            "messages": messages,
        })
    except Exception as exc:
        logger.exception("prompt %s build failed", name)
        _err(req_id, -32602, str(exc))


_HANDLERS = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
    "prompts/list": _handle_prompts_list,
    "prompts/get": _handle_prompts_get,
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

    # Ensure stdout is line-buffered text that can carry arbitrary UTF-8.
    # On Windows the locale codec (e.g. cp1252) cannot encode tool
    # descriptions containing chars like '→' (->), which crashes
    # tools/list and makes the whole server unresponsive to the MCP client.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
    # stdin may also deliver non-ASCII arguments from the client.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    # Fully automated: index the project Claude opened, in the background, so
    # the graph is ready before Claude asks for it. Only touches things that
    # look like a repo, and only writes to stderr (never stdout).
    try:
        cwd = os.getcwd()
        if os.path.isdir(cwd) and (
            os.path.isdir(os.path.join(cwd, ".git"))
            or os.path.isfile(os.path.join(cwd, ".graphsift", "manifest.json"))
        ):
            threading.Thread(target=_ensure_graph, args=(cwd,), daemon=True).start()
    except Exception:
        pass

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


if __name__ == "__main__":
    run_server()
