"""Post-processing adapter for graphsift.

Runs after graph indexing:
  - FlowDetector       -- traces execution paths (entry points -> call chains)
  - CommunityDetector  -- groups nodes into cohesive communities
  - RiskScorer         -- scores files/nodes by caller count, test coverage, security keywords
  - WikiGenerator      -- generates markdown pages per community
  - RefactorEngine     -- rename preview, dead-code detection

All operations are purely in-memory + SQLite; no I/O opened by this module.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FlowDetector
# ---------------------------------------------------------------------------


class FlowDetector:
    """Detect execution flows from entry-point nodes through the call graph.

    An entry point is any function/method that:
    - Has no incoming CALLS edges (top-level callers)
    - Is named main, run, execute, handle, __call__, or starts with test_
    - Is a CLI handler (decorated with @click.command, @app.route, etc.)

    Args:
        max_depth: Maximum call-chain depth to trace.
        max_flows: Maximum number of flows to detect.
    """

    _ENTRY_NAMES = re.compile(
        r'^(main|run|execute|handle|dispatch|start|serve|__call__|__main__)$|^test_',
        re.IGNORECASE,
    )
    _ENTRY_DECORATORS = re.compile(
        r'(command|route|get|post|put|delete|patch|task|celery|handler|endpoint)',
        re.IGNORECASE,
    )

    def __init__(self, max_depth: int = 15, max_flows: int = 500) -> None:
        self.max_depth = max_depth
        self.max_flows = max_flows

    def detect(self, graph: Any, store: Any) -> list[dict[str, Any]]:
        """Detect flows and persist them to the store.

        Args:
            graph: DependencyGraph instance.
            store: GraphStore instance.

        Returns:
            List of flow dicts with name, entry_point, nodes, criticality.
        """
        from graphsift.models import NodeKind, EdgeKind

        flows: list[dict[str, Any]] = []

        with graph._lock:
            nodes = dict(graph._nodes)
            adj_out = {k: list(v) for k, v in graph._adj_out.items()}
            adj_in = {k: list(v) for k, v in graph._adj_in.items()}

        # Find entry points
        entry_points = []
        for node_id, node in nodes.items():
            if node.kind not in (NodeKind.FUNCTION, NodeKind.METHOD, NodeKind.MODULE):
                continue
            incoming_calls = [
                e for e in adj_in.get(node_id, [])
                if e.kind.value == "calls"
            ]
            is_entry = (
                not incoming_calls
                or self._ENTRY_NAMES.match(node.name)
                or any(self._ENTRY_DECORATORS.search(d) for d in node.decorators)
            )
            if is_entry:
                entry_points.append(node)

        # Trace each entry point
        for ep in entry_points[:self.max_flows]:
            path_nodes: list[str] = []
            visited: set[str] = set()
            queue: deque[tuple[str, int]] = deque([(ep.node_id, 0)])

            while queue:
                nid, depth = queue.popleft()
                if nid in visited or depth > self.max_depth:
                    continue
                visited.add(nid)
                path_nodes.append(nid)
                for edge in adj_out.get(nid, []):
                    if edge.kind.value in ("calls", "imports", "inherits"):
                        queue.append((edge.target_id, depth + 1))

            if len(path_nodes) < 2:
                continue

            # Criticality = log(callers * depth * node_count)
            import math
            caller_count = len(adj_in.get(ep.node_id, []))
            criticality = round(
                math.log1p(caller_count + 1) * math.log1p(len(path_nodes)), 4
            )

            flow = {
                "flow_name": ep.name,
                "entry_point": ep.node_id,
                "nodes": path_nodes,
                "edges": [],
                "criticality": criticality,
                "node_count": len(path_nodes),
                "file_count": len({nodes[n].file_path for n in path_nodes if n in nodes}),
                "depth": min(self.max_depth, len(path_nodes)),
            }
            flows.append(flow)

            # Persist to SQLite
            try:
                store.save_flow_snapshot(
                    flow_name=ep.name,
                    entry_point=ep.node_id,
                    nodes=path_nodes,
                    edges=[],
                    metadata={
                        "criticality": criticality,
                        "node_count": flow["node_count"],
                        "file_count": flow["file_count"],
                    },
                )
            except Exception as exc:
                logger.warning("FlowDetector: failed to save flow %s: %s", ep.name, exc)

        logger.info("INFO: Flows detected: %d", len(flows))
        return flows


# ---------------------------------------------------------------------------
# CommunityDetector
# ---------------------------------------------------------------------------


class CommunityDetector:
    """Group nodes into communities by file-path proximity and import coupling.

    Uses a simple label-propagation approach:
    - Each file starts as its own community
    - Files that heavily import each other merge into the same community
    - Communities are labeled by their most common directory prefix

    Args:
        min_community_size: Minimum files per community (smaller ones stay solo).
        max_iterations: Max label-propagation rounds.
    """

    def __init__(self, min_community_size: int = 2, max_iterations: int = 10) -> None:
        self.min_community_size = min_community_size
        self.max_iterations = max_iterations

    def detect(self, graph: Any, store: Any) -> list[dict[str, Any]]:
        """Detect communities and persist them.

        Args:
            graph: DependencyGraph instance.
            store: GraphStore instance.

        Returns:
            List of community dicts with id, label, node_count, dominant_language.
        """
        with graph._lock:
            file_nodes = dict(graph._file_nodes)
            adj_out = {k: list(v) for k, v in graph._adj_out.items()}

        if not file_nodes:
            return []

        # Build file -> file import coupling
        file_paths = list(file_nodes.keys())
        path_to_idx = {p: i for i, p in enumerate(file_paths)}
        label = list(range(len(file_paths)))  # each file = its own community

        # Build adjacency: file_idx -> set of file_idxs it imports
        file_adj: dict[int, set[int]] = defaultdict(set)
        for i, fp in enumerate(file_paths):
            fn = file_nodes[fp]
            mod_id = f"{fp}::__module__"
            for edge in adj_out.get(mod_id, []):
                target_fp = edge.target_id.split("::")[0] if "::" in edge.target_id else ""
                if target_fp in path_to_idx:
                    file_adj[i].add(path_to_idx[target_fp])
                    file_adj[path_to_idx[target_fp]].add(i)

        # Label propagation
        for _ in range(self.max_iterations):
            changed = False
            for i in range(len(file_paths)):
                neighbors = file_adj[i]
                if not neighbors:
                    continue
                neighbor_labels = [label[j] for j in neighbors]
                # Most common neighbor label
                from collections import Counter
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                if most_common != label[i]:
                    label[i] = most_common
                    changed = True
            if not changed:
                break

        # Group by label
        communities: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(label):
            communities[lbl].append(i)

        # Filter small communities and assign sequential IDs
        result = []
        community_id = 0
        for lbl, member_idxs in sorted(communities.items(), key=lambda x: -len(x[1])):
            if len(member_idxs) < self.min_community_size:
                continue

            member_paths = [file_paths[i] for i in member_idxs]
            # Dominant language
            lang_counts: dict[str, int] = defaultdict(int)
            for p in member_paths:
                lang = file_nodes[p].language.value
                lang_counts[lang] += 1
            dominant_lang = max(lang_counts, key=lang_counts.__getitem__)

            # Label = common directory prefix
            parts_list = [Path(p).parts for p in member_paths]
            common = _common_prefix(parts_list)
            label_str = "/".join(common) if common else f"community_{community_id}"

            comm = {
                "community_id": community_id,
                "label": label_str,
                "node_count": len(member_paths),
                "dominant_language": dominant_lang,
                "members": member_paths,
            }
            result.append(comm)

            # Persist community record
            try:
                store.save_community(
                    community_id=community_id,
                    label=label_str,
                    node_count=len(member_paths),
                    metadata={"dominant_language": dominant_lang, "members": member_paths[:20]},
                )
            except Exception as exc:
                logger.warning("CommunityDetector: save_community failed: %s", exc)

            # Assign community_id to nodes in these files
            try:
                for p in member_paths:
                    fn = file_nodes[p]
                    for sym in fn.symbols:
                        nid = sym.node_id if hasattr(sym, "node_id") else f"{p}::{sym}"
                        store.assign_community(nid, community_id)
            except Exception as exc:
                logger.warning("CommunityDetector: assign_community failed: %s", exc)

            community_id += 1

        logger.info("INFO: Communities detected: %d", len(result))
        return result


def _common_prefix(parts_list: list[tuple[str, ...]]) -> list[str]:
    """Return common directory path prefix from a list of path parts."""
    if not parts_list:
        return []
    min_len = min(len(p) for p in parts_list)
    common = []
    for i in range(min_len - 1):  # -1 to exclude filename
        val = parts_list[0][i]
        if all(p[i] == val for p in parts_list):
            common.append(val)
        else:
            break
    return common


# ---------------------------------------------------------------------------
# RiskScorer
# ---------------------------------------------------------------------------


_SECURITY_KEYWORDS = re.compile(
    r'\b(password|secret|token|api_key|auth|crypt|hash|jwt|oauth|bearer|'
    r'credential|private_key|sign|verify|encrypt|decrypt|sanitize|escape|'
    r'sql|query|execute|inject|xss|csrf|cors|permission|sudo|root|admin)\b',
    re.IGNORECASE,
)


class RiskScorer:
    """Score nodes by risk: caller count, test coverage, security keywords.

    Risk formula:
        caller_score   = min(1.0, callers / 20)
        security_score = 1.0 if security keywords found else 0.0
        test_score     = 0.0 if tested else 0.5
        risk = (caller_score * 0.4) + (security_score * 0.4) + (test_score * 0.2)
    """

    def score(self, graph: Any, store: Any, source_map: dict[str, str]) -> list[dict[str, Any]]:
        """Score all files and persist to risk_index.

        Args:
            graph: DependencyGraph instance.
            store: GraphStore instance.
            source_map: Dict of file_path -> source text.

        Returns:
            List of risk dicts sorted by risk_score descending.
        """
        with graph._lock:
            file_nodes = dict(graph._file_nodes)
            adj_in = {k: list(v) for k, v in graph._adj_in.items()}
            nodes = dict(graph._nodes)

        results = []
        for fp, fn in file_nodes.items():
            reasons: list[str] = []

            # Caller count: count all incoming edges to any node in this file
            file_node_ids = {sym.node_id for sym in fn.symbols if hasattr(sym, "node_id")}
            caller_count = sum(len(adj_in.get(nid, [])) for nid in file_node_ids)
            caller_score = min(1.0, caller_count / 20)
            if caller_count > 10:
                reasons.append(f"high caller count ({caller_count})")

            # Security keywords
            source = source_map.get(fp, "")
            security_hits = len(_SECURITY_KEYWORDS.findall(source))
            security_score = min(1.0, security_hits / 5)
            if security_hits > 0:
                reasons.append(f"security-relevant ({security_hits} keywords)")

            # Test coverage: is there a test file for this file?
            stem = Path(fp).stem
            has_test = any(
                f"test_{stem}" in Path(other).stem or f"{stem}_test" in Path(other).stem
                for other in file_nodes
                if other != fp
            )
            test_score = 0.0 if has_test else 0.5
            if not has_test:
                reasons.append("no test coverage detected")

            risk = round((caller_score * 0.4) + (security_score * 0.4) + (test_score * 0.2), 4)

            results.append({
                "file_path": fp,
                "risk_score": risk,
                "reasons": reasons,
            })

            try:
                store.upsert_risk(fp, risk, reasons)
            except Exception as exc:
                logger.warning("RiskScorer: upsert_risk failed for %s: %s", fp, exc)

        results.sort(key=lambda x: -x["risk_score"])
        logger.info("INFO: Risk scores computed: %d files", len(results))
        return results


# ---------------------------------------------------------------------------
# WikiGenerator
# ---------------------------------------------------------------------------


class WikiGenerator:
    """Generate markdown wiki pages from community structure.

    Writes one .md file per community into <repo>/.graphsift/wiki/.

    Args:
        output_dir: Directory to write wiki pages into.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)

    def generate(self, communities: list[dict[str, Any]], risk_index: list[dict[str, Any]], force: bool = False) -> dict[str, int]:
        """Write community wiki pages.

        Args:
            communities: List of community dicts from CommunityDetector.
            risk_index: List of risk dicts from RiskScorer.
            force: Regenerate even if page already exists.

        Returns:
            Dict with pages_generated, pages_updated, pages_unchanged counts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        risk_by_path = {r["file_path"]: r for r in risk_index}

        generated = updated = unchanged = 0

        for comm in communities:
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', comm["label"] or f"community_{comm['community_id']}")
            page_path = self.output_dir / f"{safe_name}.md"

            members = comm.get("members", [])
            high_risk = [
                m for m in members
                if risk_by_path.get(m, {}).get("risk_score", 0) >= 0.6
            ]
            key_symbols = comm.get("key_symbols", [])

            content = _render_wiki_page(comm, members, high_risk, key_symbols)

            if page_path.exists() and not force:
                existing = page_path.read_text(encoding="utf-8")
                if existing == content:
                    unchanged += 1
                    continue
                page_path.write_text(content, encoding="utf-8")
                updated += 1
            else:
                page_path.write_text(content, encoding="utf-8")
                generated += 1

        return {"pages_generated": generated, "pages_updated": updated, "pages_unchanged": unchanged}

    def get_page(self, community_name: str) -> str | None:
        """Return the content of a wiki page by community name.

        Args:
            community_name: Community label (partial match allowed).

        Returns:
            Markdown string or None if not found.
        """
        if not self.output_dir.exists():
            return None
        name_lower = community_name.lower()
        for page in self.output_dir.glob("*.md"):
            if name_lower in page.stem.lower():
                return page.read_text(encoding="utf-8")
        return None


def _render_wiki_page(
    comm: dict[str, Any],
    members: list[str],
    high_risk: list[str],
    key_symbols: list[str],
) -> str:
    lines = [
        f"# {comm['label'] or 'Community ' + str(comm['community_id'])}",
        "",
        f"**Community ID**: {comm['community_id']}  ",
        f"**Size**: {comm['node_count']} files  ",
        f"**Language**: {comm.get('dominant_language', 'unknown')}",
        "",
        "## Files",
        "",
    ]
    for m in sorted(members)[:50]:
        risk = "  `[HIGH RISK]`" if m in high_risk else ""
        lines.append(f"- `{m}`{risk}")
    if len(members) > 50:
        lines.append(f"- _...and {len(members) - 50} more_")

    if key_symbols:
        lines += ["", "## Key Symbols", ""]
        for sym in key_symbols[:20]:
            lines.append(f"- `{sym}`")

    if high_risk:
        lines += ["", "## High Risk Files", ""]
        for f in high_risk:
            lines.append(f"- `{f}`")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# RefactorEngine
# ---------------------------------------------------------------------------


class RefactorEngine:
    """Graph-powered rename preview and dead-code detection.

    Args:
        None
    """

    _previews: dict[str, Any] = {}
    _lock = threading.Lock()

    def rename_preview(self, graph: Any, old_name: str, new_name: str) -> dict[str, Any]:
        """Preview a rename across all files.

        Args:
            graph: DependencyGraph instance.
            old_name: Symbol name to rename.
            new_name: Replacement name.

        Returns:
            Dict with refactor_id, edits (list of {file, line, old, new}).
        """
        import uuid
        with graph._lock:
            nodes = dict(graph._nodes)

        edits = []
        for node in nodes.values():
            if node.name == old_name or node.qualified_name == old_name:
                edits.append({
                    "file": node.file_path,
                    "line": node.line_start,
                    "old": old_name,
                    "new": new_name,
                    "node_id": node.node_id,
                })

        refactor_id = str(uuid.uuid4())[:8]
        import time
        with self._lock:
            self._previews[refactor_id] = {
                "old_name": old_name,
                "new_name": new_name,
                "edits": edits,
                "created_at": time.time(),
            }

        return {"refactor_id": refactor_id, "edits": edits, "total_occurrences": len(edits)}

    def apply_rename(self, refactor_id: str, repo_root: str) -> dict[str, Any]:
        """Apply a previously previewed rename to source files.

        Args:
            refactor_id: ID from rename_preview.
            repo_root: Repo root (edits validated to be inside this dir).

        Returns:
            Dict with applied_edits count, status.
        """
        import time
        with self._lock:
            preview = self._previews.get(refactor_id)

        if not preview:
            return {"error": "Refactor ID not found or expired.", "applied_edits": 0}

        if time.time() - preview["created_at"] > 600:
            return {"error": "Refactor preview expired (10 min limit).", "applied_edits": 0}

        root = Path(repo_root).resolve()
        applied = 0
        errors = []

        for edit in preview["edits"]:
            fp = Path(edit["file"]).resolve()
            # Security: only allow edits inside repo root
            try:
                fp.relative_to(root)
            except ValueError:
                errors.append(f"Blocked: {fp} is outside repo root")
                continue
            if not fp.exists():
                errors.append(f"File not found: {fp}")
                continue
            try:
                content = fp.read_text(encoding="utf-8")
                new_content = content.replace(edit["old"], edit["new"])
                if new_content != content:
                    fp.write_text(new_content, encoding="utf-8")
                    applied += 1
            except OSError as exc:
                errors.append(f"Error writing {fp}: {exc}")

        return {"applied_edits": applied, "errors": errors, "status": "done" if not errors else "partial"}

    def find_dead_code(self, graph: Any, kind: str | None = None, file_pattern: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Find symbols with zero incoming edges (unused code).

        Args:
            graph: DependencyGraph instance.
            kind: Filter by NodeKind value (e.g. "function", "class").
            file_pattern: Filter by file path substring.
            limit: Max results.

        Returns:
            List of dicts with node_id, name, file_path, kind, line_start.
        """
        from graphsift.models import NodeKind

        with graph._lock:
            nodes = dict(graph._nodes)
            adj_in = {k: list(v) for k, v in graph._adj_in.items()}

        results = []
        for node_id, node in nodes.items():
            if node.kind == NodeKind.MODULE:
                continue
            if kind and node.kind.value != kind.lower():
                continue
            if file_pattern and file_pattern not in node.file_path:
                continue
            if not adj_in.get(node_id):
                results.append({
                    "node_id": node_id,
                    "name": node.name,
                    "qualified_name": node.qualified_name,
                    "file_path": node.file_path,
                    "kind": node.kind.value,
                    "line_start": node.line_start,
                })
            if len(results) >= limit:
                break

        return results


# ---------------------------------------------------------------------------
# Postprocessor — orchestrates all post-processing steps
# ---------------------------------------------------------------------------


class Postprocessor:
    """Orchestrate flow detection, community detection, FTS rebuild, and risk scoring.

    Args:
        max_flow_depth: Max depth for flow tracing.
        max_flows: Max flows to detect.
        min_community_size: Min files per community.
    """

    def __init__(
        self,
        max_flow_depth: int = 15,
        max_flows: int = 500,
        min_community_size: int = 2,
    ) -> None:
        self._flow_detector = FlowDetector(max_depth=max_flow_depth, max_flows=max_flows)
        self._community_detector = CommunityDetector(min_community_size=min_community_size)
        self._risk_scorer = RiskScorer()

    def run(
        self,
        graph: Any,
        store: Any,
        source_map: dict[str, str],
        flows: bool = True,
        communities: bool = True,
        risk: bool = True,
        fts: bool = True,
    ) -> dict[str, Any]:
        """Run all post-processing steps.

        Args:
            graph: DependencyGraph instance.
            store: GraphStore instance.
            source_map: Dict of file_path -> source text.
            flows: Whether to run flow detection.
            communities: Whether to run community detection.
            risk: Whether to score risk.
            fts: Whether to rebuild the FTS index.

        Returns:
            Dict with flows_detected, communities_detected, files_scored, fts_indexed.
        """
        result: dict[str, Any] = {
            "flows_detected": 0,
            "communities_detected": 0,
            "files_scored": 0,
            "fts_indexed": 0,
        }

        if flows:
            logger.info("INFO: Running flow detection ...")
            detected = self._flow_detector.detect(graph, store)
            result["flows_detected"] = len(detected)

        if communities:
            logger.info("INFO: Running community detection ...")
            comms = self._community_detector.detect(graph, store)
            result["communities_detected"] = len(comms)
            result["communities"] = comms

        if risk:
            logger.info("INFO: Computing risk scores ...")
            scored = self._risk_scorer.score(graph, store, source_map)
            result["files_scored"] = len(scored)

        if fts:
            logger.info("INFO: Rebuilding FTS index ...")
            try:
                store._conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
                store._conn.commit()
                row_count = store._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
                result["fts_indexed"] = row_count
                logger.info("INFO: FTS index rebuilt: %d rows indexed", row_count)
            except Exception as exc:
                logger.warning("Postprocessor: FTS rebuild failed: %s", exc)

        logger.info("INFO: Post-processing complete")
        return result
