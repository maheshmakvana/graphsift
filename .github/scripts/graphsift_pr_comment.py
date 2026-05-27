"""Generate a markdown PR comment with graphsift blast radius analysis.

Called by the graphsift-review GitHub Action workflow.  Uses the graphsift
Python API directly to build the dependency graph, compute the blast radius
for changed files, detect circular dependencies, and write a structured
markdown comment to RUNNER_TEMP/graphsift_comment.md for the next workflow
step to post via actions/github-script.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

try:
    from graphsift._version import __version__ as _graphsift_version
except ImportError:
    _graphsift_version = "unknown"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_changed_files(base_sha: str, head_sha: str) -> list[str]:
    """Return list of file paths changed between two git refs."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_sha}...{head_sha}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


# ---------------------------------------------------------------------------
# graphsift analysis
# ---------------------------------------------------------------------------


def _db_path_for_root(root: str) -> str:
    """Compute the per-repo SQLite DB path (same algorithm as graphsift.cli)."""
    key = hashlib.sha1(root.encode()).hexdigest()[:12]
    db_dir = Path.home() / ".graphsift" / key
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "graph.db")


def _estimate_tokens(file_path: str) -> int:
    """Rough token estimate: 1 token ~= 4 characters."""
    try:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return len(text) // 4
    except OSError:
        return 0


def analyze(changed_files: list[str]) -> dict:
    """Build dependency graph and run blast-radius + cycle analysis."""
    # Lazy-import graphsift internals so the script fails early if missing
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.adapters.storage import GraphStore
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = Path.cwd()

    # --- Build in-memory graph ---
    source_map = load_source_map(str(root))
    builder = ContextBuilder(ContextConfig())
    stats = builder.index_files(source_map)
    graph = getattr(builder, "_graph", None)

    if graph is None:
        return {"error": "Failed to build dependency graph"}

    # --- Blast radius: ranked neighbors ---
    scores = graph.ranked_neighbors(seed_paths=changed_files, include_dynamic=True)
    affected = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    # --- Risk scores from DB (populated by ``graphsift build``) ---
    risk_by_path: dict[str, float] = {}
    db_path = _db_path_for_root(str(root))
    if Path(db_path).exists():
        try:
            store = GraphStore(db_path)
            risk_by_path = {
                r["file_path"]: r["risk_score"]
                for r in store.load_risk_index(min_score=0.0)
            }
        except Exception:
            pass  # risk index missing — proceed without it

    # --- Token estimates ---
    total_tokens = 0
    hot_warm_tokens = 0
    for fp, (score, _depth, _reasons) in affected:
        t = _estimate_tokens(fp)
        total_tokens += t
        if score >= 0.3:  # HOT + WARM
            hot_warm_tokens += t

    token_savings_pct = (
        round((1 - hot_warm_tokens / total_tokens) * 100)
        if total_tokens > 0
        else 0
    )

    # --- Cycle detection ---
    cycles = graph.detect_cycles()

    # --- Tier buckets ---
    hot_files: list[str] = []
    warm_files: list[str] = []
    cold_files: list[str] = []
    for fp, (score, _depth, _reasons) in affected:
        if score >= 0.7:
            hot_files.append(fp)
        elif score >= 0.3:
            warm_files.append(fp)
        else:
            cold_files.append(fp)

    return {
        "changed_file_list": changed_files,
        "changed_count": len(changed_files),
        "files_scanned": stats.files_indexed,
        "symbols": stats.symbols_extracted,
        "edges": stats.edges_created,
        "duration_ms": stats.duration_ms,
        "affected_count": len(affected),
        "affected_top": [
            {
                "path": fp,
                "score": round(score, 4),
                "risk": round(risk_by_path.get(fp, 0.0), 4),
                "depth": depth,
            }
            for fp, (score, depth, _reasons) in affected[:30]
        ],
        "total_tokens": total_tokens,
        "hot_warm_tokens": hot_warm_tokens,
        "token_savings_pct": token_savings_pct,
        "tier_hot": len(hot_files),
        "tier_warm": len(warm_files),
        "tier_cold": len(cold_files),
        "hot_files": hot_files[:10],
        "warm_files": warm_files[:10],
        "cycles": [
            {
                "files": cycle,
                "stems": [Path(f).stem for f in cycle],
                "count": len(cycle),
            }
            for cycle in cycles
        ],
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _fmt_file_list(paths: list[str], max_shown: int = 6) -> str:
    """Format a short comma-separated file list with ellipsis if truncated."""
    parts = [f"`{Path(p).stem}`" for p in paths[:max_shown]]
    if len(paths) > max_shown:
        parts.append(f"and {len(paths) - max_shown} more")
    return ", ".join(parts) if parts else chr(8212)


def format_comment(data: dict) -> str:
    """Return the full markdown PR comment."""
    lines: list[str] = []
    ap = lambda: lines.append("")  # noqa: E731 — blank-line helper

    lines.append("## graphsift Blast Radius Analysis")
    ap()

    # ── Summary table ──────────────────────────────────────────────────────
    total_affected = data["affected_count"]
    selected = data["tier_hot"] + data["tier_warm"]
    file_savings = int((1 - selected / total_affected) * 100) if total_affected > 0 else 0

    lines.append("### Summary")
    ap()
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Files changed | {data['changed_count']} |")
    lines.append(f"| Files scanned | {data['files_scanned']} |")
    lines.append(f"| Files in blast radius | {total_affected} |")
    lines.append(f"| Symbols in graph | {data['symbols']:,} |")
    lines.append(f"| Dependency edges | {data['edges']:,} |")
    lines.append(f"| Rough token estimate (all affected) | {data['total_tokens']:,} |")
    lines.append(f"| Token savings (HOT+WARM focus) | ~{data['token_savings_pct']}% |")
    lines.append(f"| File savings (HOT+WARM focus) | ~{file_savings}% |")
    ap()

    # ── Changed files ──────────────────────────────────────────────────────
    changed = data["changed_file_list"]
    lines.append("### Changed Files")
    ap()
    for f in changed[:15]:
        lines.append(f"- `{f}`")
    if len(changed) > 15:
        lines.append(f"- *... and {len(changed) - 15} more files*")
    ap()

    # ── Top affected files ──────────────────────────────────────────────────
    top = data["affected_top"]
    if top:
        lines.append("### Top 10 Affected Files (Blast Radius)")
        ap()
        lines.append("| # | File | Score | Risk | Depth |")
        lines.append("|---|------|-------|------|-------|")
        for i, entry in enumerate(top[:10], 1):
            fp = entry["path"]
            short = fp if len(fp) <= 54 else f"...{fp[-51:]}"
            lines.append(
                f"| {i} | `{short}` | {entry['score']:.3f} | "
                f"{entry['risk']:.3f} | {entry['depth']} |"
            )
        ap()

    # ── Tier distribution ──────────────────────────────────────────────────
    lines.append("### Tier Distribution")
    ap()
    hot_str = _fmt_file_list(data["hot_files"])
    warm_str = _fmt_file_list(data["warm_files"])
    lines.append("| Tier | Count | Examples |")
    lines.append("|------|-------|---------|")
    lines.append(
        f"| :fire: **HOT** (score ≥ 0.7) | {data['tier_hot']} | {hot_str} |"
    )
    lines.append(
        f"| :large_orange_diamond: **WARM** (0.3 ≤ score < 0.7) "
        f"| {data['tier_warm']} | {warm_str} |"
    )
    lines.append(
        f"| :snowflake: **COLD** (score < 0.3) | {data['tier_cold']} | {chr(8212)} |"
    )
    ap()

    # ── Circular dependency warnings ───────────────────────────────────────
    cycles = data["cycles"]
    if cycles:
        lines.append("### Circular Dependency Warnings")
        ap()
        lines.append(
            f"Found **{len(cycles)}** circular dependenc"
            f"{'y' if len(cycles) == 1 else 'ies'}:"
        )
        ap()
        for i, cycle in enumerate(cycles[:5], 1):
            severity = "HIGH" if cycle["count"] <= 3 else "LOW"
            label = f":warning: **Cycle {i}** ({severity}, {cycle['count']} files)"
            stems = cycle["stems"][:6]
            path_str = " → ".join(stems)
            if len(cycle["stems"]) > 6:
                path_str += " → ..."
            lines.append(f"- {label}: `{path_str}`")
        if len(cycles) > 5:
            lines.append(
                f"- *... and {len(cycles) - 5} more cycle{'s' if len(cycles) - 5 != 1 else ''}*"
            )
        ap()
    else:
        lines.append("### Circular Dependencies")
        ap()
        lines.append(":white_check_mark: No circular dependencies detected.")
        ap()

    # ── Footer ─────────────────────────────────────────────────────────────
    lines.append("---")
    ap()
    lines.append(
        f"_Generated by graphsift v{_graphsift_version} "
        f"— {data['duration_ms']:.0f}ms analysis_"
    )
    ap()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    base_sha = os.environ.get("BASE_SHA")
    head_sha = os.environ.get("HEAD_SHA")

    if not base_sha or not head_sha:
        print("Missing BASE_SHA or HEAD_SHA environment variables.", file=sys.stderr)
        sys.exit(1)

    changed = get_changed_files(base_sha, head_sha)
    if not changed:
        print("No changed files to analyze.")
        sys.exit(0)

    print(f"Analyzing {len(changed)} changed file(s) ...")
    data = analyze(changed)

    if "error" in data:
        print(f"Error: {data['error']}", file=sys.stderr)
        sys.exit(1)

    comment = format_comment(data)

    runner_temp = os.environ.get(
        "RUNNER_TEMP", os.environ.get("TMPDIR", "/tmp")
    )
    out_path = Path(runner_temp) / "graphsift_comment.md"
    out_path.write_text(comment, encoding="utf-8")
    print(f"Blast radius comment saved to {out_path}")


if __name__ == "__main__":
    main()
