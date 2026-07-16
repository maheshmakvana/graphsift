"""graphsift CLI - install, serve, build, update, status, register, list-repos."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys

from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cwd() -> str:
    return os.getcwd()


def _find_claude_settings(project_root: Path) -> Path:
    """Return path to .claude/settings.json, creating dirs if needed."""
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    return claude_dir / "settings.json"


def _find_mcp_json(project_root: Path) -> Path:
    return project_root / ".mcp.json"


def _python_executable() -> str:
    return sys.executable


# ---------------------------------------------------------------------------
# install command
# ---------------------------------------------------------------------------

def cmd_install(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    print(f"[graphsift] Installing into {project_root}")

    # 1. Write / merge .mcp.json
    mcp_path = _find_mcp_json(project_root)
    mcp_config: dict = {}
    if mcp_path.exists():
        try:
            mcp_config = json.loads(mcp_path.read_text(encoding="utf-8"))
        except Exception:
            mcp_config = {}

    # Top-level key is "mcpServers" per Claude Code spec
    mcp_config.setdefault("mcpServers", {})
    mcp_config["mcpServers"]["graphsift"] = {
        "command": _python_executable(),
        "args": ["-m", "graphsift.mcp_server"],
        "env": {},
    }
    mcp_path.write_text(json.dumps(mcp_config, indent=2), encoding="utf-8")
    print(f"[graphsift] Wrote {mcp_path}")

    # 2. Inject hooks into .claude/settings.json
    if not args.no_hooks:
        settings_path = _find_claude_settings(project_root)
        settings: dict = {}
        if settings_path.exists():
            try:
                settings = json.loads(settings_path.read_text(encoding="utf-8"))
            except Exception:
                settings = {}

        settings.setdefault("hooks", {})

        # SessionStart - prime Claude with graph awareness
        settings["hooks"].setdefault("SessionStart", [])
        session_hook = {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": (
                        f"{_python_executable()} -c \""
                        "import graphsift, os; "
                        "print('[graphsift] Knowledge graph available. "
                        "Use build_graph tool to index repo, then get_context for token-efficient code context.')"
                        "\""
                    ),
                }
            ],
        }
        # Only add if not already present
        existing_cmds = [
            h.get("command", "")
            for entry in settings["hooks"]["SessionStart"]
            for h in entry.get("hooks", [])
        ]
        if not any("graphsift" in c for c in existing_cmds):
            settings["hooks"]["SessionStart"].append(session_hook)

        # PostToolUse - auto-update graph after Write/Edit/Bash
        settings["hooks"].setdefault("PostToolUse", [])
        post_hook = {
            "matcher": "Write|Edit|Bash",
            "hooks": [
                {
                    "type": "command",
                    "command": (
                        f"{_python_executable()} -m graphsift.cli update "
                        f"--project-root \"{project_root}\" 2>/dev/null || true"
                    ),
                }
            ],
        }
        existing_post = [
            h.get("command", "")
            for entry in settings["hooks"]["PostToolUse"]
            for h in entry.get("hooks", [])
        ]
        if not any("graphsift" in c for c in existing_post):
            settings["hooks"]["PostToolUse"].append(post_hook)

        # PostToolUse for Bash — compress command output to save tokens
        bash_post_hook = {
            "matcher": "Bash",
            "hooks": [
                {
                    "type": "command",
                    "command": (
                        f"{_python_executable()} -c \""
                        "import sys, os; "
                        "from graphsift.compress import compress; "
                        "from graphsift.analytics import record_call; "
                        "text = sys.stdin.read(); "
                        "if text and len(text) > 200: "
                        "    compressed = compress(text); "
                        "    record_call(tokens_saved=(len(text)-len(compressed))//4, command_type='bash', original_chars=len(text), compressed_chars=len(compressed)); "
                        "    sys.stdout.write(compressed) "
                        "else: "
                        "    sys.stdout.write(text or '')"
                        "\""
                    ),
                }
            ],
        }
        if not any("graphsift.compress" in h.get("command", "") for entry in settings["hooks"]["PostToolUse"] for h in entry.get("hooks", [])):
            settings["hooks"]["PostToolUse"].append(bash_post_hook)

        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"[graphsift] Wrote hooks -> {settings_path}")

    # 3. Write skill files
    if not args.no_skills:
        _write_skills(project_root)

    # 4. Install bash wrapper (auto-compress commands)
    if args.bash_wrapper:
        from .hooks import get_bash_wrapper_script
        bashrc_path = Path.home() / ".bashrc"
        wrapper_script = get_bash_wrapper_script(python_path=_python_executable())

        # Check if already installed
        existing = bashrc_path.read_text(encoding="utf-8") if bashrc_path.exists() else ""
        if "# graphsift: transparent output compression" not in existing:
            with open(bashrc_path, "a", encoding="utf-8") as f:
                f.write(f"\n# graphsift: transparent output compression\n")
                f.write(f'eval "$({_python_executable()} -m graphsift.cli bash-wrapper)"\n')
            print(f"[graphsift] Installed bash wrapper -> {bashrc_path}")
        else:
            print(f"[graphsift] Bash wrapper already installed in {bashrc_path}")

    print("[graphsift] Installation complete.")
    print()
    print("  Next steps:")
    print("  1. Restart Claude Code (to load the MCP server)")
    print("  2. Ask Claude: 'Build the graphsift graph for this repo'")
    print("     or run:  graphsift build")
    print()
    return 0


# ---------------------------------------------------------------------------
# serve command  (starts the MCP stdio server)
# ---------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace) -> int:
    from graphsift.mcp_server import run_server
    run_server()
    return 0


# ---------------------------------------------------------------------------
# build command  (index repo from CLI)
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:  # noqa: C901
    import time
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.adapters.storage import GraphStore
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig, GraphNode, NodeKind

    root = Path(args.project_root).resolve()
    extensions = set(args.extensions) if args.extensions else {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    }
    exclude_dirs = set(args.exclude_dirs) if args.exclude_dirs else {
        "venv", ".venv", "node_modules", ".git", "__pycache__",
        "dist", "build", ".mypy_cache", ".pytest_cache",
    }
    progress_interval = int(getattr(args, "progress_interval", 200))

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("  graphsift: building knowledge graph")
    print(f"  repo   : {root}")
    print()

    # ── Step 1: Open / migrate SQLite DB ─────────────────────────────────────
    print("  [1/5] Opening database ...")
    db_path = _db_path_for_root(str(root))
    _t0 = time.monotonic()

    class _MigrationPrinter:
        """Redirect graphsift storage logger to stdout during migration."""
        def write(self, msg: str) -> None:
            msg = msg.strip()
            if msg:
                print(f"        {msg}")
        def flush(self) -> None:
            pass

    import logging as _logging
    _storage_handler = _logging.StreamHandler(_MigrationPrinter())  # type: ignore[arg-type]
    _storage_handler.setFormatter(_logging.Formatter("%(message)s"))
    _storage_logger = _logging.getLogger("graphsift.adapters.storage")
    _storage_logger.setLevel(_logging.INFO)
    _storage_logger.addHandler(_storage_handler)
    _storage_logger.propagate = False

    store = GraphStore(db_path)
    db_stats = store.stats()
    print(f"        schema version : {db_stats['schema_version']}")
    print(f"        db path        : {db_path}")
    print()

    # ── Step 2: Discover files ────────────────────────────────────────────────
    print("  [2/5] Scanning files ...")
    source_map = load_source_map(str(root), extensions=extensions, exclude_dirs=exclude_dirs)
    total_files = len(source_map)

    # Count by extension
    from collections import Counter
    ext_counts: Counter = Counter(Path(p).suffix.lower() for p in source_map)
    print(f"        found {total_files} files")
    for ext, cnt in ext_counts.most_common(8):
        print(f"          {ext or '(no ext)':10s}  {cnt}")
    print()

    # ── Step 3: Parse & index ─────────────────────────────────────────────────
    print(f"  [3/5] Parsing {total_files} files ...")
    from graphsift.models import DepthTier  # noqa: PLC0415
    depth_tier_val = DepthTier(getattr(args, 'depth', 'execution'))
    builder = ContextBuilder(ContextConfig(depth_tier=depth_tier_val))
    all_paths = list(source_map.keys())
    skipped = 0
    t_parse_start = time.monotonic()

    for i, path in enumerate(all_paths, 1):
        try:
            builder.index_file(path, source_map[path])
        except Exception:
            skipped += 1
        if progress_interval > 0 and i % progress_interval == 0:
            elapsed = time.monotonic() - t_parse_start
            rate = i / elapsed if elapsed > 0 else 0
            pct = i * 100 // total_files
            print(f"        Progress: {i:>6}/{total_files}  [{pct:>3}%]  {rate:.0f} files/s")

    if total_files % progress_interval != 0 or total_files == 0:
        elapsed = time.monotonic() - t_parse_start
        rate = total_files / elapsed if elapsed > 0 else 0
        print(f"        Progress: {total_files:>6}/{total_files}  [100%]  {rate:.0f} files/s")

    parse_ms = (time.monotonic() - t_parse_start) * 1000
    print(f"        done in {parse_ms:.0f} ms  ({skipped} skipped)")
    print()

    # ── Step 4: Build final graph stats ──────────────────────────────────────
    print("  [4/5] Building dependency graph ...")
    t_graph = time.monotonic()
    stats = builder.index_files(source_map)
    graph_ms = (time.monotonic() - t_graph) * 1000

    # Language breakdown from stats
    lang_counts = stats.languages
    print(f"        files indexed  : {stats.files_indexed}")
    print(f"        files skipped  : {stats.files_skipped}")
    print(f"        symbols        : {stats.symbols_extracted}")
    print(f"        edges          : {stats.edges_created}")
    print(f"        dynamic imports: {stats.dynamic_imports_found}")
    if lang_counts:
        print(f"        languages      :", ", ".join(f"{k}:{v}" for k, v in sorted(lang_counts.items(), key=lambda x: -x[1])[:6]))
    print(f"        time           : {graph_ms:.0f} ms")
    print()

    # ── Step 5: Persist to SQLite ─────────────────────────────────────────────
    print("  [5/5] Persisting to database ...")
    t_db = time.monotonic()
    graph_obj = getattr(builder, "_graph", None)
    nodes_saved = 0
    files_saved = 0

    if graph_obj is not None:
        all_nodes: list[GraphNode] = []
        all_file_nodes = []
        for file_node in graph_obj.all_files():
            all_file_nodes.append(file_node)
            for sym in file_node.symbols:
                if hasattr(sym, "node_id"):
                    # sym is already a GraphNode
                    all_nodes.append(sym)
                else:
                    # sym is a string name
                    all_nodes.append(
                        GraphNode(
                            node_id=f"{file_node.path}::{sym}",
                            file_path=file_node.path,
                            kind=NodeKind.FUNCTION,
                            name=str(sym),
                            qualified_name=str(sym),
                            language=file_node.language,
                        )
                    )
        store.save_nodes(all_nodes)
        store.save_files(all_file_nodes)
        nodes_saved = len(all_nodes)
        files_saved = len(all_file_nodes)

    db_ms = (time.monotonic() - t_db) * 1000
    print(f"        nodes saved    : {nodes_saved}")
    print(f"        files saved    : {files_saved}")
    print(f"        time           : {db_ms:.0f} ms")
    print()

    # ── Step 6: Post-processing (flows, communities, risk, FTS) ───────────────
    pp_result: dict = {}
    if not getattr(args, "skip_postprocess", False):
        print("  [6/6] Post-processing (flows, communities, risk, FTS) ...")
        t_pp = time.monotonic()
        from graphsift.adapters.postprocess import Postprocessor

        class _PPPrinter:
            def write(self, msg: str) -> None:
                msg = msg.strip()
                if msg:
                    print(f"        {msg}")
            def flush(self) -> None:
                pass

        import logging as _logging
        _pp_handler = _logging.StreamHandler(_PPPrinter())  # type: ignore[arg-type]
        _pp_handler.setFormatter(_logging.Formatter("%(message)s"))
        _pp_logger = _logging.getLogger("graphsift.adapters.postprocess")
        _pp_logger.setLevel(_logging.INFO)
        _pp_logger.addHandler(_pp_handler)
        _pp_logger.propagate = False

        if graph_obj is not None:
            pp = Postprocessor()
            pp_result = pp.run(graph_obj, store, source_map)
        pp_ms = (time.monotonic() - t_pp) * 1000
        print(f"        time           : {pp_ms:.0f} ms")
        print()

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest = {
        "root": str(root),
        "files_indexed": stats.files_indexed,
        "symbols_extracted": stats.symbols_extracted,
        "edges_created": stats.edges_created,
        "duration_ms": stats.duration_ms,
        "files": [str(p) for p in source_map.keys()],
    }
    manifest_path = root / ".graphsift" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_ms = (time.monotonic() - _t0) * 1000

    # ── Summary ───────────────────────────────────────────────────────────────
    print("  " + "-" * 45)
    print(f"  Build complete in {total_ms:.0f} ms")
    print(f"  {stats.files_indexed} files  |  {stats.symbols_extracted} symbols  |  {stats.edges_created} edges")
    if pp_result:
        print(f"  flows    : {pp_result.get('flows_detected', 0)}  |  communities: {pp_result.get('communities_detected', 0)}  |  fts rows: {pp_result.get('fts_indexed', 0)}")
    print(f"  db       : {db_path}")
    print(f"  manifest : {manifest_path}")
    # Analytics summary
    try:
        from .analytics import summary_line
        sl = summary_line(str(root))
        if "Run" not in sl:
            print(f"  {sl}")
    except Exception:
        pass
    print()

    return 0


def _db_path_for_root(root: str) -> str:
    """Compute the per-repo DB path, stored under ~/.graphsift/<sha1>/graph.db."""
    key = hashlib.sha1(root.encode()).hexdigest()[:12]
    db_dir = Path.home() / ".graphsift" / key
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "graph.db")


# ---------------------------------------------------------------------------
# update command  (incremental - called by PostToolUse hook)
# ---------------------------------------------------------------------------

def cmd_update(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    manifest_path = root / ".graphsift" / "manifest.json"

    if not manifest_path.exists():
        # Silent - no graph built yet, nothing to update
        return 0

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    # Find files newer than manifest
    manifest_mtime = manifest_path.stat().st_mtime
    changed: list[str] = []
    for file_path in manifest.get("files", []):
        p = Path(file_path)
        if p.exists() and p.stat().st_mtime > manifest_mtime:
            changed.append(str(p))

    if not changed:
        return 0

    from graphsift.adapters.filesystem import load_changed_files
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    new_sources = load_changed_files(changed)
    builder = ContextBuilder(ContextConfig())
    for path, source in new_sources.items():
        try:
            builder.index_file(path, source)
        except Exception:
            pass

    # Touch manifest to update mtime
    manifest["files_updated"] = changed
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.project_root).resolve()
    manifest_path = root / ".graphsift" / "manifest.json"
    mcp_path = _find_mcp_json(root)
    settings_path = _find_claude_settings(root)

    print(f"[graphsift] Status for {root}")
    print()

    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"  Graph     : built ({m.get('files_indexed', '?')} files, "
                  f"{m.get('symbols_extracted', '?')} symbols, "
                  f"{m.get('edges_created', '?')} edges)")
        except Exception:
            print("  Graph     : manifest unreadable")
    else:
        print("  Graph     : not built  (run: graphsift build)")

    print(f"  MCP config: {'found' if mcp_path.exists() else 'missing'} ({mcp_path})")
    print(f"  Hooks     : {'found' if settings_path.exists() else 'missing'} ({settings_path})")

    skills_dir = root / ".claude" / "skills"
    skill_count = len(list(skills_dir.glob("*/SKILL.md"))) if skills_dir.exists() else 0
    print(f"  Skills    : {skill_count} installed")
    print()

    # Evolution status
    try:
        from graphsift.evolve_registry import EvolveRegistry
        registry = EvolveRegistry()
        entries = registry.list_entries()
        if entries:
            print(f"Evolution cache: {len(entries)} entries")
            for e in entries:
                print(f"  {e.get('fingerprint', '?')[:16]}: score={e.get('score', 0):.4f}")
        else:
            print("Evolution cache: empty")
    except ImportError:
        pass  # evolve module not available

    return 0


# ---------------------------------------------------------------------------
# uninstall command
# ---------------------------------------------------------------------------

def cmd_uninstall(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()

    # Remove from .mcp.json
    mcp_path = _find_mcp_json(project_root)
    if mcp_path.exists():
        try:
            cfg = json.loads(mcp_path.read_text(encoding="utf-8"))
            cfg.get("mcpServers", {}).pop("graphsift", None)
            mcp_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"[graphsift] Removed MCP entry from {mcp_path}")
        except Exception as exc:
            print(f"[graphsift] Warning: could not update {mcp_path}: {exc}")

    # Remove skills
    skills_dir = project_root / ".claude" / "skills"
    for skill_dir in ["graphsift-build", "graphsift-review", "graphsift-impact", "graphsift-compress"]:
        import shutil
        target = skills_dir / skill_dir
        if target.exists():
            shutil.rmtree(target)
    print("[graphsift] Removed skill files.")

    # Remove manifest
    import shutil
    gs_dir = project_root / ".graphsift"
    if gs_dir.exists():
        shutil.rmtree(gs_dir)
        print(f"[graphsift] Removed {gs_dir}")

    print("[graphsift] Uninstalled. Restart Claude Code to apply.")
    return 0


# ---------------------------------------------------------------------------
# Skill file writer
# ---------------------------------------------------------------------------

def _write_skills(project_root: Path) -> None:
    skills_root = project_root / ".claude" / "skills"

    _write_skill(
        skills_root / "graphsift-build" / "SKILL.md",
        title="graphsift: Build Graph",
        description="Build or rebuild the graphsift dependency graph for this repo.",
        steps=[
            "Call the `build_graph` MCP tool with root_path set to the repo root.",
            "Report back: files indexed, symbols extracted, edges created.",
            "Tell the user the graph is ready and they can now use get_context for token-efficient reviews.",
        ],
        example="Build the graphsift graph",
    )

    _write_skill(
        skills_root / "graphsift-review" / "SKILL.md",
        title="graphsift: Code Review",
        description="Review changed files using graphsift's ranked context selection - minimal tokens, maximum relevance.",
        steps=[
            "Call `graph_status` to check if the graph is built. If not, call `build_graph` first.",
            "Call `get_context` with the changed_files list and a query describing what to review.",
            "Use the returned rendered_context as the code block for your review.",
            "Report token_savings_pct to show how many tokens were saved vs sending the whole repo.",
        ],
        example="Review the changes in src/auth.py using graphsift",
    )

    _write_skill(
        skills_root / "graphsift-impact" / "SKILL.md",
        title="graphsift: Impact Analysis",
        description="Find all files affected by a change - blast radius analysis with relevance scores.",
        steps=[
            "Call `get_impact` with the changed_files list.",
            "Present the top affected files sorted by score (0-1).",
            "Highlight any high-score (>0.7) files as high-risk blast radius.",
        ],
        example="What is the blast radius of changes to src/auth.py?",
    )

    _write_skill(
        skills_root / "graphsift-compress" / "SKILL.md",
        title="graphsift: Compress Output",
        description="Compress CLI command output to save 60-90% tokens before they reach the LLM context window.",
        steps=[
            "After running a Bash command that produced large output, call `compress_output` with the output text.",
            "The tool auto-detects the command type (git, npm, pytest, etc.) and applies the optimal compression strategy.",
            "Use the compressed output instead of the raw output in your LLM analysis.",
            "Check `token_gain` periodically to see cumulative token savings.",
        ],
        example="Compress this pytest output before analyzing it",
    )

    print(f"[graphsift] Wrote 4 skill files -> {skills_root}")


def _write_skill(path: Path, title: str, description: str, steps: list[str], example: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps_md = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    path.write_text(
        f"# {title}\n\n"
        f"{description}\n\n"
        f"## Steps\n\n{steps_md}\n\n"
        f"## Example trigger\n\n> {example}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Registry helpers  (~/.graphsift/registry.json)
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path.home() / ".graphsift" / "registry.json"


def _load_registry() -> dict[str, dict]:
    if _REGISTRY_PATH.exists():
        try:
            return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_registry(registry: dict[str, dict]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# register command
# ---------------------------------------------------------------------------

def cmd_register(args: argparse.Namespace) -> int:
    root = str(Path(args.project_root).resolve())
    registry = _load_registry()
    registry[root] = {
        "root": root,
        "db_path": _db_path_for_root(root),
        "name": args.name or Path(root).name,
    }
    _save_registry(registry)
    print(f"[graphsift] Registered repo: {root}")
    print(f"[graphsift] Registry      -> {_REGISTRY_PATH}")
    return 0


# ---------------------------------------------------------------------------
# list-repos command
# ---------------------------------------------------------------------------

def cmd_list_repos(args: argparse.Namespace) -> int:
    registry = _load_registry()
    if not registry:
        print("[graphsift] No repos registered. Run: graphsift register")
        return 0

    count = len(registry)
    print(f"[graphsift] {count} registered repo(s):\n")
    for i, (root, info) in enumerate(registry.items(), 1):
        name = info.get("name", Path(root).name)
        db = info.get("db_path", "?")
        print(f"  {i}. {name}")
        print(f"     root   : {root}")
        print(f"     db     : {db}")
        print()
    return 0


# ---------------------------------------------------------------------------
# postprocess command
# ---------------------------------------------------------------------------

def cmd_postprocess(args: argparse.Namespace) -> int:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s", stream=sys.stdout)

    from graphsift.adapters.filesystem import load_source_map
    from graphsift.adapters.postprocess import Postprocessor
    from graphsift.adapters.storage import GraphStore
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = Path(args.project_root).resolve()
    db_path = _db_path_for_root(str(root))

    print(f"\nGraphsift: running post-processing for {root}\n")

    manifest_path = root / ".graphsift" / "manifest.json"
    if not manifest_path.exists():
        print("[graphsift] No graph built yet. Run: graphsift build")
        return 1

    # Re-index for in-memory graph
    print("  Loading source map ...")
    source_map = load_source_map(str(root))
    builder = ContextBuilder(ContextConfig())
    builder.index_files(source_map)
    graph = getattr(builder, "_graph", None)
    if graph is None:
        print("[graphsift] Failed to build graph.")
        return 1

    store = GraphStore(db_path)
    pp = Postprocessor()

    result = pp.run(
        graph, store, source_map,
        flows=not args.no_flows,
        communities=not args.no_communities,
        risk=not args.no_risk,
        fts=not args.no_fts,
    )

    print()
    print("  Post-processing results:")
    print(f"    flows detected     : {result['flows_detected']}")
    print(f"    communities found  : {result['communities_detected']}")
    print(f"    files risk-scored  : {result['files_scored']}")
    print(f"    fts entries        : {result['fts_indexed']}")
    print()
    return 0


# ---------------------------------------------------------------------------
# watch command
# ---------------------------------------------------------------------------

def cmd_watch(args: argparse.Namespace) -> int:
    import time
    from graphsift.adapters.filesystem import load_changed_files
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = Path(args.project_root).resolve()
    manifest_path = root / ".graphsift" / "manifest.json"
    print(f"[graphsift] Watching {root} for changes (Ctrl+C to stop) ...")

    last_mtimes: dict[str, float] = {}

    def _scan_mtimes() -> dict[str, float]:
        mtimes: dict[str, float] = {}
        for ext in [".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java"]:
            for p in root.rglob(f"*{ext}"):
                skip = any(d in p.parts for d in ["venv", ".venv", "node_modules", ".git", "__pycache__", "dist", "build"])
                if not skip:
                    try:
                        mtimes[str(p)] = p.stat().st_mtime
                    except OSError:
                        pass
        return mtimes

    last_mtimes = _scan_mtimes()

    try:
        while True:
            time.sleep(2)
            current = _scan_mtimes()
            changed = [p for p, mtime in current.items()
                       if p not in last_mtimes or last_mtimes[p] != mtime]
            removed = [p for p in last_mtimes if p not in current]

            if changed or removed:
                print(f"[graphsift] {len(changed)} changed, {len(removed)} removed — updating graph ...")
                if changed:
                    new_sources = load_changed_files(changed)
                    builder = ContextBuilder(ContextConfig())
                    for path, source in new_sources.items():
                        try:
                            builder.index_file(path, source)
                        except Exception:
                            pass
                    print(f"[graphsift] Updated {len(changed)} files.")
                last_mtimes = current
    except KeyboardInterrupt:
        print("\n[graphsift] Watch stopped.")
    return 0


# ---------------------------------------------------------------------------
# detect-changes command
# ---------------------------------------------------------------------------

def cmd_detect_changes(args: argparse.Namespace) -> int:
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.adapters.postprocess import RiskScorer
    from graphsift.adapters.storage import GraphStore
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = Path(args.project_root).resolve()
    changed_files = [str(Path(f).resolve()) for f in args.files] if args.files else []

    if not changed_files:
        print("[graphsift] No files specified. Use: graphsift detect-changes file1.py file2.py")
        return 1

    source_map = load_source_map(str(root))
    builder = ContextBuilder(ContextConfig())
    builder.index_files(source_map)
    graph = getattr(builder, "_graph", None)

    if not graph:
        print("[graphsift] No graph built.")
        return 1

    store = GraphStore(_db_path_for_root(str(root)))
    risk_rows = store.load_risk_index(min_score=0.0)
    risk_by_path = {r["file_path"]: r["risk_score"] for r in risk_rows}

    scores = graph.ranked_neighbors(seed_paths=changed_files, include_dynamic=True)
    affected = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:30]

    max_risk = max((risk_by_path.get(f, 0.0) for f in changed_files), default=0.0)

    print(f"\n  Changed files  : {len(changed_files)}")
    print(f"  Affected files : {len(affected)}")
    print(f"  Max risk score : {max_risk:.2f}")
    print()
    print(f"  {'File':<60} {'Score':>6}  {'Risk':>5}  Reasons")
    print("  " + "-" * 80)
    for fp, (score, depth, reasons) in affected[:20]:
        rsk = risk_by_path.get(fp, 0.0)
        reason_str = ", ".join(reasons[:2])
        short_fp = fp[-55:] if len(fp) > 55 else fp
        print(f"  {short_fp:<60} {score:>6.3f}  {rsk:>5.2f}  {reason_str}")
    print()
    return 0


# ---------------------------------------------------------------------------
# visualize command
# ---------------------------------------------------------------------------

def cmd_visualize(args: argparse.Namespace) -> int:
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = Path(args.project_root).resolve()
    output_path = root / ".graphsift" / "graph.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[graphsift] Building visualization for {root} ...")
    source_map = load_source_map(str(root))
    builder = ContextBuilder(ContextConfig())
    builder.index_files(source_map)
    graph = getattr(builder, "_graph", None)

    if not graph:
        print("[graphsift] No graph built.")
        return 1

    with graph._lock:
        file_nodes = list(graph._file_nodes.values())
        edges = list(graph._edges)

    # Build minimal D3-ready JSON
    nodes_js = [
        {"id": fn.path, "label": Path(fn.path).name, "lang": fn.language.value, "tokens": fn.token_estimate}
        for fn in file_nodes[:300]
    ]
    node_ids = {n["id"] for n in nodes_js}
    links_js = [
        {"source": e.source_id.split("::")[0], "target": e.target_id.split("::")[0], "kind": e.kind.value}
        for e in edges
        if e.source_id.split("::")[0] in node_ids and e.target_id.split("::")[0] in node_ids
    ][:1000]

    html = _render_graph_html(nodes_js, links_js, str(root))
    output_path.write_text(html, encoding="utf-8")

    print(f"[graphsift] Graph visualization -> {output_path}")
    if args.serve:
        import http.server
        import webbrowser
        port = 8765
        os.chdir(str(output_path.parent))
        print(f"[graphsift] Serving at http://localhost:{port}/graph.html  (Ctrl+C to stop)")
        webbrowser.open(f"http://localhost:{port}/graph.html")
        http.server.HTTPServer(("", port), http.server.SimpleHTTPRequestHandler).serve_forever()
    return 0


def _render_graph_html(nodes: list, links: list, title: str) -> str:
    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>graphsift: {title}</title>
<style>
body {{ margin:0; background:#1a1a2e; font-family:monospace; color:#eee; }}
svg {{ width:100vw; height:100vh; }}
.node circle {{ stroke:#fff; stroke-width:1.5px; cursor:pointer; }}
.link {{ stroke:#555; stroke-opacity:0.5; }}
.label {{ font-size:10px; fill:#ccc; pointer-events:none; }}
#info {{ position:fixed; top:10px; right:10px; background:#16213e; padding:12px; border-radius:6px; max-width:300px; font-size:12px; }}
</style></head>
<body>
<div id="info"><b>graphsift</b><br>{title}<br>{len(nodes)} files &nbsp; {len(links)} edges</div>
<svg id="graph"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const nodes = {nodes_json};
const links = {links_json};
const w = window.innerWidth, h = window.innerHeight;
const svg = d3.select('#graph').attr('width',w).attr('height',h);
const g = svg.append('g');
svg.call(d3.zoom().on('zoom', e => g.attr('transform', e.transform)));
const colors = d3.scaleOrdinal(d3.schemeTableau10);
const sim = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d=>d.id).distance(80))
  .force('charge', d3.forceManyBody().strength(-120))
  .force('center', d3.forceCenter(w/2, h/2));
const link = g.append('g').selectAll('line').data(links).join('line').attr('class','link');
const node = g.append('g').selectAll('g').data(nodes).join('g')
  .call(d3.drag().on('start',(e,d)=>{{if(!e.active)sim.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y}})
    .on('drag',(e,d)=>{{d.fx=e.x;d.fy=e.y}})
    .on('end',(e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}}));
node.append('circle').attr('r',6).attr('fill',d=>colors(d.lang));
node.append('text').attr('class','label').attr('dx',8).attr('dy',4).text(d=>d.label);
node.append('title').text(d=>d.id+'\\n'+d.lang+' | '+d.tokens+' tokens');
sim.on('tick',()=>{{
  link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
  node.attr('transform',d=>`translate(${{d.x}},${{d.y}})`);
}});
</script></body></html>"""


# ---------------------------------------------------------------------------
# wiki command
# ---------------------------------------------------------------------------

def cmd_wiki(args: argparse.Namespace) -> int:
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.adapters.postprocess import CommunityDetector, RiskScorer, WikiGenerator
    from graphsift.adapters.storage import GraphStore
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = Path(args.project_root).resolve()
    db_path = _db_path_for_root(str(root))
    store = GraphStore(db_path)

    communities = store.load_communities()
    risk_index = store.load_risk_index()

    if not communities:
        print("[graphsift] No communities found. Run: graphsift postprocess")
        return 1

    wiki_dir = root / ".graphsift" / "wiki"
    gen = WikiGenerator(str(wiki_dir))
    counts = gen.generate(communities, risk_index, force=args.force)

    print(f"[graphsift] Wiki generated -> {wiki_dir}")
    print(f"  generated: {counts['pages_generated']}")
    print(f"  updated  : {counts['pages_updated']}")
    print(f"  unchanged: {counts['pages_unchanged']}")
    return 0


# ---------------------------------------------------------------------------
# unregister command
# ---------------------------------------------------------------------------

def cmd_unregister(args: argparse.Namespace) -> int:
    registry = _load_registry()
    target = args.path_or_name

    # Try exact path match first, then name match
    key_to_remove = None
    for key, info in registry.items():
        if key == target or str(Path(target).resolve()) == key or info.get("name") == target:
            key_to_remove = key
            break

    if key_to_remove is None:
        print(f"[graphsift] Not found in registry: {target}")
        return 1

    del registry[key_to_remove]
    _save_registry(registry)
    print(f"[graphsift] Unregistered: {key_to_remove}")
    return 0


# ---------------------------------------------------------------------------
# repos command (alias for list-repos with nicer output)
# ---------------------------------------------------------------------------

def cmd_repos(args: argparse.Namespace) -> int:
    return cmd_list_repos(args)


# ---------------------------------------------------------------------------
# gain command  (token savings analytics)
# ---------------------------------------------------------------------------

def cmd_gain(args: argparse.Namespace) -> int:
    if args.history:
        from .analytics import history
        result = history(project_root=args.project_root, limit=20)
        print(json.dumps(result, indent=2, default=str))
        return 0
    from .analytics import gain
    print(gain(project_root=args.project_root, format="json" if args.json else "text"))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show cumulative token savings and cost estimate."""
    from .analytics import summary_line
    print(summary_line(project_root=args.project_root))
    return 0


# ---------------------------------------------------------------------------
# compress command  (rtk-style output compression)
# ---------------------------------------------------------------------------

def cmd_compress(args: argparse.Namespace) -> int:
    if args.list:
        from .compress import COMPRESSORS
        print("Available compressors:")
        for name in sorted(COMPRESSORS):
            print(f"  {name}")
        return 0

    # Detect interactive terminal (no piped input) and show help
    if sys.stdin.isatty():
        print("graphsift compress: pipe command output into me to save tokens.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("  pytest -v | graphsift compress", file=sys.stderr)
        print("  git diff  | graphsift compress", file=sys.stderr)
        print("  docker ps | graphsift compress", file=sys.stderr)
        print("", file=sys.stderr)
        print("Available compressors:", file=sys.stderr)
        from .compress import COMPRESSORS
        for name in sorted(COMPRESSORS):
            print(f"  {name}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Transparent mode: eval \"$(graphsift bash-wrapper)\" to auto-compress all supported commands.",
              file=sys.stderr)
        return 1

    raw = sys.stdin.read()

    if args.tee:
        from .compress import set_tee_dir, compress_tee
        set_tee_dir(args.tee)
        result, tee_path = compress_tee(raw, command=args.type, ultra=args.ultra,
                                         label=args.tee_label)
    else:
        from .compress import compress
        result = compress(raw, command=args.type, ultra=args.ultra)

    sys.stdout.write(result)
    # Analytics summary to stderr (so it doesn't mix with piped output)
    try:
        from .analytics import summary_line
        sl = summary_line()
        if "Run" not in sl:
            print(sl, file=sys.stderr)
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# discover command  (find missed token-saving opportunities)
# ---------------------------------------------------------------------------

def cmd_discover(args: argparse.Namespace) -> int:
    from .analytics import discover
    return discover(args)


def cmd_bash_wrapper(args: argparse.Namespace) -> int:
    """Print bash shell wrapper script for transparent command compression.

    Source the output in .bashrc for automatic compression of 19 command types:

        eval "$(graphsift bash-wrapper)"
    """
    from .hooks import get_bash_wrapper_script
    print(get_bash_wrapper_script(python_path=_python_executable()))
    return 0


# ---------------------------------------------------------------------------
# detect-cycles command
# ---------------------------------------------------------------------------


def cmd_detect_cycles(args: argparse.Namespace) -> int:
    """Detect circular dependencies (import/call cycles) in the codebase using Tarjan's SCC."""
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = args.root or os.getcwd()
    source_map = load_source_map(root)
    builder = ContextBuilder(ContextConfig())
    builder.index_files(source_map)

    graph = getattr(builder, "_graph", None)
    if not graph or not graph._nodes:
        print("Error: Graph is empty. Index your codebase first.", file=sys.stderr)
        return 1

    cycles = graph.detect_cycles()

    if not cycles:
        print("No circular dependencies found.")
        return 0

    print(f"Found {len(cycles)} circular dependencies:\n")
    for i, cycle in enumerate(cycles):
        severity = "ERROR" if len(cycle) <= 3 else "WARNING"
        print(f"  [{severity}] Cycle {i + 1} ({len(cycle)} files):")
        for f in cycle:
            print(f"    -> {f}")
        print()

    total_files = len(set(f for c in cycles for f in c))
    print(f"Total files in cycles: {total_files}")
    return 0


# ---------------------------------------------------------------------------
# detect-dead-code command
# ---------------------------------------------------------------------------


def cmd_suggest_fixes(args: argparse.Namespace) -> int:
    """Run auto-fix analysis and return prioritized fix suggestions."""
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING, format="%(message)s", stream=sys.stderr)

    from graphsift.adapters.filesystem import load_source_map
    from graphsift.auto_fix import FixSuggester
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = Path(args.project_root).resolve()
    changed_files = args.changed_files or None

    print(f"\nGraphsift: auto-fix suggestions for {root}\n")

    # Build graph
    source_map = load_source_map(str(root))
    builder = ContextBuilder(ContextConfig())
    builder.index_files(source_map)
    graph = getattr(builder, "_graph", None)

    if not graph:
        print("[graphsift] No graph built. Run: graphsift build")
        return 1

    suggester = FixSuggester(graph, source_map=source_map)
    report = suggester.analyze(changed_files=changed_files)

    if args.json:
        print(report.model_dump_json(indent=2))
        return 0

    if not report.suggestions:
        print("  No issues found. Your code looks clean!")
        return 0

    print(f"  {report.summary}")
    print()

    # Filter by min_confidence
    filtered = [s for s in report.suggestions if s.confidence >= args.min_confidence]

    # Group by severity
    for severity in ("error", "warning", "info"):
        group = [s for s in filtered if s.severity.value == severity]
        if not group:
            continue
        label = severity.upper()
        print(f"  [{label}]")
        print()
        for s in group:
            loc = f"{s.file_path}:{s.line_start}"
            auto_tag = "  [AUTO-FIXABLE]" if s.auto_fixable else ""
            conf_tag = f"  [conf={s.confidence:.2f}]"
            print(f"    {s.title}")
            print(f"      File: {loc}{auto_tag}{conf_tag}")
            if s.description:
                print(f"      {s.description}")
            if s.suggested_change:
                print(f"      --> {s.suggested_change}")
            print()
        print()

    print(f"  Total: {len(filtered)} suggestion(s) "
          f"({report.by_severity.get('error', 0)} errors, "
          f"{report.by_severity.get('warning', 0)} warnings, "
          f"{report.by_severity.get('info', 0)} info)")
    print()

    return 0


def cmd_detect_dead_code(args: argparse.Namespace) -> int:
    """Detect potentially unreachable code via BFS reachability analysis."""
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    root = args.root or os.getcwd()
    source_map = load_source_map(root)
    builder = ContextBuilder(ContextConfig())
    builder.index_files(source_map)

    graph = getattr(builder, "_graph", None)
    if not graph or not graph._nodes:
        print("Error: Graph is empty. Index your codebase first.", file=sys.stderr)
        return 1

    entry_points = args.entry_points.split(",") if args.entry_points else None
    dead = graph.find_dead_code(
        entry_points=entry_points,
        kind=args.kind or None,
    )

    if not dead:
        print("No dead code detected.")
        return 0

    # Apply priority scoring for large result sets (>20 entries)
    prioritize = getattr(args, "prioritize", True)
    show_all = getattr(args, "all", False)

    if prioritize and len(dead) > 5:
        from graphsift.prioritize import PriorityScorer  # noqa: PLC0415

        scorer = PriorityScorer(graph=graph, source_map=source_map)
        ranked = scorer.score_dead_code(dead)
        entries_to_show = (
            ranked.all_entries if show_all else ranked.entries
        )

        print(ranked.summary)
        print()

        if ranked.tiers:
            table_header = f"{'Tier':<12} {'Kind':<12} {'Name':<30} {'File':<40} {'Score':<8}"
            print(table_header)
            print("-" * len(table_header))
            for sf in entries_to_show:
                e = sf.entry
                print(
                    f"{sf.tier:<12} "
                    f"[{e.get('kind', '?'):<10}] "
                    f"{e.get('name', ''):<30} "
                    f"{e.get('file_path', '')[-39:]:<40} "
                    f"{sf.score:<8.3f}"
                )
        print()
        if ranked.truncated:
            print(
                f"Tip: {ranked.truncated_count} findings hidden. "
                f"Use --all to see everything."
            )
    else:
        print(f"Found {len(dead)} potentially unreachable elements:\n")
        for item in dead:
            print(f"  [{item['kind']}] {item['name']}")
            print(f"    File: {item['file_path']}:{item['line_start']}")
            print(f"    Reason: {item['reason']}")
            print()

    return 0


# ---------------------------------------------------------------------------
# terse command  (inline terse mode prefix)
# ---------------------------------------------------------------------------

def cmd_terse(args: argparse.Namespace) -> int:
    """Return a terse mode prefix for LLM prompts."""
    level = args.level
    prefix = f"[TERSE:{level}]"
    if args.prompt:
        prefix = f"{prefix} {args.prompt}"
    print(prefix)
    return 0


# ---------------------------------------------------------------------------
# fix command  (bug fix template)
# ---------------------------------------------------------------------------

def cmd_fix(args: argparse.Namespace) -> int:
    from graphsift.prompt_templates import FixBugTemplate

    tpl = FixBugTemplate()
    result = tpl.render(
        bug=args.bug,
        file=args.file,
        line=args.line,
        expected=args.expected or "",
        actual=args.actual or "",
    )
    if args.json:
        print(json.dumps({"prompt": result}))
    else:
        print(result)
    return 0


# ---------------------------------------------------------------------------
# add command  (feature template)
# ---------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> int:
    from graphsift.prompt_templates import AddFeatureTemplate

    tpl = AddFeatureTemplate()
    result = tpl.render(
        feature=args.feature,
        files=args.files or None,
        acceptance_criteria=args.acceptance_criteria or None,
    )
    if args.json:
        print(json.dumps({"prompt": result}))
    else:
        print(result)
    return 0


# ---------------------------------------------------------------------------
# refactor command  (refactor template)
# ---------------------------------------------------------------------------

def cmd_refactor(args: argparse.Namespace) -> int:
    from graphsift.prompt_templates import RefactorTemplate

    tpl = RefactorTemplate()
    result = tpl.render(
        target=args.target,
        goal=args.goal or "",
        files=args.files or None,
    )
    if args.json:
        print(json.dumps({"prompt": result}))
    else:
        print(result)
    return 0


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    from graphsift.verify_hooks import Verifier

    verifier = Verifier(project_root=args.project_root)
    result = verifier.check(args.file)
    print(f"File     : {result.file}")
    print(f"Syntax   : {'OK' if result.syntax_ok else 'FAIL'}")
    if result.syntax_error:
        print(f"Error    : {result.syntax_error}")
    print(f"Passed   : {result.passed}")
    return 0 if result.passed else 1


# ---------------------------------------------------------------------------
# tool-budgets command
# ---------------------------------------------------------------------------

def cmd_tool_budgets(args: argparse.Namespace) -> int:
    from graphsift.tool_budgets import ToolBudget

    budget = ToolBudget()
    if args.set_lines is not None and args.tool:
        budget.set_budget(args.tool, args.set_lines)
        print(f"Set budget for '{args.tool}' to {args.set_lines} lines.")
        return 0
    if args.tool:
        val = budget.get_budget(args.tool)
        print(f"{args.tool}: {val} lines")
        return 0
    # Show all budgets
    print("Tool budgets:")
    for tool, limit in sorted(budget.budgets.items()):
        print(f"  {tool}: {limit} lines")
    return 0


# ---------------------------------------------------------------------------
# read-cache command
# ---------------------------------------------------------------------------

_READ_CACHE: object = None  # module-level singleton


def _get_read_cache():
    global _READ_CACHE
    if _READ_CACHE is None:
        from graphsift.read_cache import ReadCache

        _READ_CACHE = ReadCache()
    return _READ_CACHE


def cmd_read_cache(args: argparse.Namespace) -> int:
    cache = _get_read_cache()
    if args.clear:
        cache.clear()
        print("Read cache cleared.")
        return 0
    if args.stats:
        print(f"Stubs served: {cache.stubs_served}")
        print(f"Cached files: {len(cache._fingerprints)}")
        return 0
    # Default: show stats
    print(f"Stubs served: {cache.stubs_served}")
    print(f"Cached files: {len(cache._fingerprints)}")
    return 0


# ---------------------------------------------------------------------------
# evidence command
# ---------------------------------------------------------------------------

def cmd_evidence(args: argparse.Namespace) -> int:
    from graphsift.evidence_check import EvidenceChecker

    checker = EvidenceChecker(project_root=args.project_root)
    citations = checker.check_response(args.text)
    if not citations:
        print("No citations found in text.")
        return 0
    valid = [c for c in citations if c.valid]
    invalid = [c for c in citations if not c.valid]
    print(f"Citations found: {len(citations)}")
    print(f"  Valid  : {len(valid)}")
    print(f"  Invalid: {len(invalid)}")
    if invalid:
        print()
        print("Invalid citations:")
        for c in invalid:
            loc = f"{c.file_path}:{c.line}" if c.line else c.file_path
            print(f"  {loc}  ({c.error})")
    return 1 if invalid else 0


# ---------------------------------------------------------------------------
# claude-md command  (auto-generate CLAUDE.md with project topology)
# ---------------------------------------------------------------------------

def _detect_build_tools(root: Path) -> dict[str, str]:
    """Detect common build/test tools from project config files.

    Returns dict with keys like "build", "test", "type_check",
    "lint" with detected commands.
    """
    tools: dict[str, str] = {}

    # pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib  # noqa: PLC0415
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            # Test command
            if "pytest" in str(data):
                tools["test"] = "pytest -xvs"
            # Build system
            build_sys = data.get("build-system", {})
            backend = build_sys.get("build-backend", "")
            if "setuptools" in backend:
                tools["build"] = "python -m build"
            elif "poetry" in backend or "poetry" in str(data):
                tools["build"] = "poetry build"
            # Scripts
            scripts = data.get("project", {}).get("scripts", {})
            if scripts:
                tools["scripts"] = ", ".join(scripts.keys())
        except Exception:
            pass

    # package.json
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            import json  # noqa: PLC0415
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            scripts = pkg.get("scripts", {})
            if "test" in scripts:
                tools["test"] = f"npm test  ({scripts['test']})"
            if "build" in scripts:
                tools["build"] = f"npm run build  ({scripts['build']})"
            if "lint" in scripts:
                tools["lint"] = f"npm run lint  ({scripts['lint']})"
        except Exception:
            pass

    # Cargo.toml
    cargo = root / "Cargo.toml"
    if cargo.exists():
        tools["build"] = "cargo build"
        tools["test"] = "cargo test"

    # go.mod
    go_mod = root / "go.mod"
    if go_mod.exists():
        tools["test"] = "go test ./..."
        tools["build"] = "go build ./..."

    # Makefile
    makefile = root / "Makefile"
    if makefile.exists():
        try:
            content = makefile.read_text(encoding="utf-8")
            if ".PHONY" in content or ":" in content:
                tools["has_makefile"] = "make"
                for target in ("test", "build", "lint", "fmt", "install"):
                    if re.search(rf"^{target}:", content, re.MULTILINE):
                        tools[target] = f"make {target}"
        except Exception:
            pass

    return tools


def _count_files_by_lang(root: Path, exclude_dirs: set[str]) -> dict[str, int]:
    """Count source files grouped by language extension."""
    counts: dict[str, int] = {}
    ext_map: dict[str, str] = {
        ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
        ".ts": "TypeScript", ".tsx": "TypeScript", ".go": "Go",
        ".rs": "Rust", ".java": "Java", ".rb": "Ruby",
        ".php": "PHP", ".c": "C", ".h": "C", ".cpp": "C++",
        ".hpp": "C++", ".sh": "Shell", ".bash": "Shell",
        ".tf": "Terraform", ".yaml": "YAML", ".yml": "YAML",
        ".md": "Markdown", ".json": "JSON", ".toml": "TOML",
    }
    extra_exclude = exclude_dirs | {".claude", ".graphsift"}
    for ext in ext_map:
        for fp in root.rglob(f"*{ext}"):
            parts = fp.relative_to(root).parts
            if any(d in parts for d in extra_exclude):
                continue
            lang = ext_map[ext]
            counts[lang] = counts.get(lang, 0) + 1
    return counts


def cmd_claude_md(args: argparse.Namespace) -> int:
    """Auto-generate CLAUDE.md with project topology for optimal AI context.

    Scans the project root, detects language distribution, build tools,
    and key entry points, then writes a structured CLAUDE.md.
    """
    root = Path(args.project_root).resolve()
    claude_md_path = root / "CLAUDE.md"

    if claude_md_path.exists() and not args.force:
        print(f"[graphsift] CLAUDE.md already exists at {claude_md_path}")
        print("  Use --force to overwrite.")
        return 0

    print(f"[graphsift] Scanning {root} for CLAUDE.md generation ...")

    exclude_dirs: set[str] = {
        "venv", ".venv", "node_modules", ".git", "__pycache__",
        "dist", "build", ".mypy_cache", ".pytest_cache",
    }

    # Detect project name
    project_name = root.name

    # Analyze language distribution
    lang_counts = _count_files_by_lang(root, exclude_dirs)
    sorted_langs = sorted(lang_counts.items(), key=lambda x: -x[1])
    total_files = sum(lang_counts.values())

    # Detect build tools
    tools = _detect_build_tools(root)

    # Detect key directories
    key_dirs: list[str] = []
    for d in ["src", "graphsift", "app", "lib", "core", "api", "cmd"]:
        if (root / d).is_dir():
            key_dirs.append(d)

    # Detect entry points (exclude .claude worktrees, venv, etc.)
    entry_points: list[str] = []
    main_patterns = [
        "**/__main__.py", "**/main.py", "**/main.go", "**/main.rs",
        "**/cli.py", "**/app.py", "**/index.js", "**/index.ts",
        "**/App.tsx", "**/app.tsx",
    ]
    extra_exclude = exclude_dirs | {".claude"}
    for pat in main_patterns:
        for fp in root.glob(pat):
            parts = fp.relative_to(root).parts
            if not any(d in parts for d in extra_exclude):
                entry_points.append(str(fp.relative_to(root)))

    # Detect dependency count
    dep_count = 0
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib  # noqa: PLC0415
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            deps = data.get("project", {}).get("dependencies", [])
            dep_count = len(deps)
        except Exception:
            pass

    # ── Build CLAUDE.md ──────────────────────────────────────────────────
    lines: list[str] = [
        f"# {project_name}",
        "",
        "Auto-generated by graphsift. Optimized for AI-assisted development.",
        "",
        "## Project Overview",
        "",
        f"- **Files**: {total_files} source files across {len(sorted_langs)} languages",
        f"- **Languages**: {', '.join(f'{lang} ({count})' for lang, count in sorted_langs[:8])}",
        f"- **Dependencies**: {dep_count}",
        f"- **Key dirs**: {', '.join(key_dirs) if key_dirs else 'flat layout'}",
        "",
    ]

    if entry_points:
        lines.append("## Entry Points")
        lines.append("")
        for ep in entry_points[:5]:
            lines.append(f"- `{ep}`")
        lines.append("")

    if tools:
        lines.append("## Build & Test")
        lines.append("")
        for key in ("build", "test", "lint", "type_check", "scripts"):
            if key in tools:
                lines.append(f"- **{key.replace('_', ' ').title()}**: `{tools[key]}`")
        if "has_makefile" in tools:
            lines.append(f"- **Make targets**: check `make help` or the Makefile")
        lines.append("")

    lines.extend([
        "## AI Context Rules",
        "",
        "When assisting with this codebase:",
        "",
        "1. **Graphsift is installed** — use it to save tokens on every operation:",
        f"   - `graphsift build` — index the repo (already done if CLAUDE.md exists)",
        "   - `graphsift compress` — pipe CLI output to save 60-97% tokens",
        "   - Use `get_context` MCP tool for token-efficient code reviews",
        "2. **Before reading files**, check if graphsift can provide targeted context",
        "3. **For review requests**, always ask for the specific files/functions",
        "4. **Token budget**: aim for <50k tokens per context request",
        "",
    ])

    content = "\n".join(lines)

    claude_md_path.parent.mkdir(parents=True, exist_ok=True)
    claude_md_path.write_text(content, encoding="utf-8")

    print()
    print(f"  Written: {claude_md_path}")
    print(f"  Files   : {total_files}")
    print(f"  Languages: {', '.join(f'{l}:{c}' for l, c in sorted_langs[:6])}")
    print(f"  Tools  : {', '.join(tools.keys()) if tools else 'none detected'}")
    print()
    print("  Next: share CLAUDE.md with your AI assistants for optimal context.")
    print()
    return 0


# ---------------------------------------------------------------------------
# loop-engineering commands
# ---------------------------------------------------------------------------


def cmd_loop_init(args: argparse.Namespace) -> int:
	"""Scaffold loop config for the project."""
	from graphsift.loop_config import LoopConfig

	root = Path(args.project_root).resolve()
	config = LoopConfig()
	config.save(str(root))
	print(f"[graphsift] Loop config initialized at {root / '.graphsift' / 'loop-config.json'}")
	print()
	# Run audit
	from graphsift.loop_engineering import LoopEngine
	engine = LoopEngine(repo_root=str(root))
	audit = engine.audit_readiness()
	print(f"  Loop Readiness Score: {audit['score']}/{audit['max_score']} ({audit['percentage']}%)")
	print(f"  Level: {audit['level']}")
	if audit['suggestions']:
		print("  Suggestions:")
		for s in audit['suggestions']:
			print(f"    - {s}")
	return 0


def cmd_loop_run(args: argparse.Namespace) -> int:
	"""Run a specific loop pattern."""
	from graphsift.loop_engineering import LoopEngine, PatternType

	engine = LoopEngine(repo_root=args.project_root)
	pattern = PatternType(args.pattern)

	method_map = {
		PatternType.DAILY_TRIAGE: engine.run_daily_triage,
		PatternType.PR_BABYSITTER: engine.run_pr_babysitter,
		PatternType.CI_SWEEPER: engine.run_ci_sweeper,
		PatternType.DEP_SWEEPER: engine.run_dep_sweeper,
		PatternType.CHANGELOG_DRAFT: engine.run_changelog,
		PatternType.POST_MERGE_CLEANUP: engine.run_cleanup,
		PatternType.ISSUE_TRIAGE: engine.run_issue_triage,
	}

	fn = method_map.get(pattern)
	if not fn:
		print(f"[graphsift] Unknown pattern: {args.pattern}")
		return 1

	result = fn()
	print(f"[graphsift] Loop run: {result.pattern.value}")
	print(f"  Status  : {result.status.value}")
	print(f"  Summary : {result.summary}")
	print(f"  Duration: {result.duration_ms:.0f}ms")
	print(f"  Tokens  : {result.tokens_used}")
	print(f"  Run ID  : {result.run_id[:12]}")
	if result.error:
		print(f"  Error   : {result.error}")
	return 0 if result.status.value == "success" else 1


def cmd_loop_status(args: argparse.Namespace) -> int:
	"""Show loop engine status."""
	from graphsift.loop_engineering import LoopEngine

	engine = LoopEngine(repo_root=args.project_root)
	report = engine.full_report()
	sched = report.get("scheduler", {})
	budget = report.get("budget", {})

	print(f"[graphsift] Loop Engine Status")
	print(f"  Scheduler     : {'running' if sched.get('running') else 'stopped'}")
	print(f"  Patterns      : {len(sched.get('patterns', {}))}")
	print(f"  Daily budget  : {budget.get('daily_limit', 500_000):,} tokens")
	print(f"  Today spend   : {budget.get('today_total', 0):,} tokens")
	print(f"  Budget remain : {budget.get('budget_remaining', 0):,} tokens")
	print(f"  Week spend    : {budget.get('week_total', 0):,} tokens")
	print(f"  Worktrees     : {len(report.get('active_worktrees', []))} active")
	drift = report.get("drift", [])
	if drift:
		for d in drift:
			print(f"  Drift         : {d.get('type')} ({d.get('from')} -> {d.get('to')})")
	return 0


def cmd_loop_report(args: argparse.Namespace) -> int:
	"""Full loop activity report."""
	import json

	from graphsift.loop_engineering import LoopEngine

	engine = LoopEngine(repo_root=args.project_root)
	report = engine.full_report()
	print(json.dumps(report, indent=2, default=str))
	return 0


def cmd_loop_schedule(args: argparse.Namespace) -> int:
	"""List scheduled patterns."""
	from graphsift.loop_engineering import LoopEngine, PatternType, PATTERN_REGISTRY

	engine = LoopEngine(repo_root=args.project_root)
	sched = engine.full_report().get("scheduler", {})
	patterns = sched.get("patterns", {})

	if not patterns:
		print("[graphsift] No patterns registered.")
		return 0

	print(f"{'Pattern':<25} {'Cadence':<12} {'Maturity':<10} {'Enabled':<10}")
	print("-" * 60)
	for name, info in sorted(patterns.items()):
		registry_info = PATTERN_REGISTRY.get(PatternType(name), {})
		cadence = registry_info.get("cadence", f"{info.get('cadence_seconds', 0)}s")
		print(f"{name:<25} {cadence:<12} {info.get('maturity', 'L1'):<10} {str(info.get('enabled', True)):<10}")
	return 0


def cmd_loop_cost(args: argparse.Namespace) -> int:
	"""Estimate token cost per pattern."""
	from graphsift.loop_engineering import LoopCostBudgeter, LoopEngine, MaturityLevel, PatternType

	budgeter = LoopCostBudgeter()
	pattern = PatternType(args.pattern)
	maturity = MaturityLevel(args.maturity)

	est = budgeter.estimate_cost(pattern, maturity)
	print(f"[graphsift] Cost estimate for {args.pattern} @ {args.maturity}")
	print(f"  Estimated tokens per run: {est:,}")
	print(f"  Daily budget: {budgeter.DEFAULT_DAILY_LIMIT:,}")
	print(f"  Max runs/day: {budgeter.DEFAULT_DAILY_LIMIT // max(est, 1)}")
	print(f"  Weekly estimate (daily): {est * 7:,}")
	return 0


def cmd_loop_audit(args: argparse.Namespace) -> int:
	"""Loop readiness score and suggestions."""
	from graphsift.loop_engineering import LoopEngine

	engine = LoopEngine(repo_root=args.project_root)
	audit = engine.audit_readiness()

	print(f"[graphsift] Loop Readiness Audit")
	print(f"  Score     : {audit['score']}/{audit['max_score']} ({audit['percentage']}%)")
	print(f"  Level     : {audit['level']}")
	print(f"  Badge     : {audit['badge']}")
	print()
	if audit['suggestions']:
		print("  Suggestions to improve:")
		for s in audit['suggestions']:
			print(f"    ☐ {s}")
	else:
		print("  No suggestions — loop setup looks complete!")
	return 0


def cmd_loop_session_start(args: argparse.Namespace) -> int:
	"""Run one-shot session start diagnostic. No background loops."""
	from graphsift.loop_engineering import LoopEngine

	engine = LoopEngine(repo_root=args.project_root)
	diag = engine.session_start()
	print("[graphsift] Session diagnostics complete")
	print(f"  Summary  : {diag['summary']}")
	print(f"  Patterns : {len(diag.get('patterns_run', []))}")
	for p in diag.get('patterns_run', []):
		print(f"    - {p['pattern']}: {p['status']} ({p['tokens']} tok, {p['duration_ms']:.0f}ms)")
	print(f"  Total    : {diag.get('total_tokens', 0)} tokens, {diag.get('total_duration_ms', 0):.0f}ms")
	if diag.get('drift'):
		for d in diag['drift']:
			print(f"  Drift    : {d.get('type')}")
	print()
	print("  Tip: run 'graphsift loop run <pattern>' for deeper analysis")
	print("  Tip: run 'graphsift loop diagnose' when you're stuck")
	return 0


def cmd_loop_diagnose(args: argparse.Namespace) -> int:
	"""Run comprehensive diagnostic — use when struggling with code."""
	import json
	from graphsift.loop_engineering import LoopEngine

	engine = LoopEngine(repo_root=args.project_root)
	result = engine.run_diagnostic()
	print(f"[graphsift] Diagnostic complete")
	print(f"  Status   : {result.status.value}")
	print(f"  Summary  : {result.summary}")
	print(f"  Duration : {result.duration_ms:.0f}ms")
	print(f"  Tokens   : {result.tokens_used}")
	if args.message:
		struggle = engine.detect_struggle(user_message=args.message)
		if struggle['triggered']:
			print(f"  Struggle : {struggle['reason']} (confidence: {struggle['confidence']:.1f})")
			print(f"  Suggest  : run 'graphsift loop run {struggle['suggested_pattern'].value}'")
	return 0


def cmd_loop_struggle(args: argparse.Namespace) -> int:
	"""Check for struggle signals in user message or failure patterns."""
	from graphsift.loop_engineering import LoopEngine

	engine = LoopEngine(repo_root=args.project_root)
	result = engine.detect_struggle(user_message=args.message, repeated_failures=args.failures)
	if result['triggered']:
		print(f"[graphsift] Struggling detected: {result['reason']}")
		print(f"  Confidence : {result['confidence']:.1f}")
		print(f"  Suggested  : graphsift loop run {result['suggested_pattern'].value}")
		print(f"  Or run     : graphsift loop diagnose")
	else:
		print("[graphsift] No struggle signals detected.")
		print("  If you ARE stuck, try: graphsift loop diagnose")
	return 0


def cmd_loop_reset_breaker(args: argparse.Namespace) -> int:
	"""Reset circuit breaker for a pattern."""
	from graphsift.loop_engineering import LoopEngine

	engine = LoopEngine(repo_root=args.project_root)
	engine._circuit_breaker.reset(args.pattern)
	print(f"[graphsift] Circuit breaker reset for pattern: {args.pattern}")
	return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    from ._version import __version__  # noqa: PLC0415
    from .loop_engineering import PatternType

    parser = argparse.ArgumentParser(
        prog="graphsift",
        description="graphsift - smarter code context for LLMs",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # install
    p_install = sub.add_parser("install", help="Register graphsift with Claude Code")
    p_install.add_argument("--project-root", default=_cwd(), help="Repo root (default: cwd)")
    p_install.add_argument("--no-hooks", action="store_true", help="Skip hook injection")
    p_install.add_argument("--no-skills", action="store_true", help="Skip skill file creation")
    p_install.add_argument("--bash-wrapper", action="store_true", help="Install transparent bash command compression")

    # serve
    sub.add_parser("serve", help="Start MCP stdio server (used by Claude Code)")

    # build
    p_build = sub.add_parser("build", help="Index repo and build dependency graph")
    p_build.add_argument("--project-root", default=_cwd())
    p_build.add_argument("--extensions", nargs="*", metavar="EXT")
    p_build.add_argument("--exclude-dirs", nargs="*", metavar="DIR")
    p_build.add_argument("--progress-interval", type=int, default=200,
                         help="Log progress every N files (default 200, 0=disable)")
    p_build.add_argument("--depth", choices=["planning", "exploration", "execution"],
                         default="execution", help="Context depth tier (default: execution)")
    p_build.add_argument("--skip-postprocess", action="store_true",
                         help="Skip flow/community/risk/FTS post-processing after indexing")

    # update
    p_update = sub.add_parser("update", help="Incrementally update graph (changed files only)")
    p_update.add_argument("--project-root", default=_cwd())

    # postprocess
    p_pp = sub.add_parser("postprocess", help="Run flow/community detection, risk scoring, FTS rebuild")
    p_pp.add_argument("--project-root", default=_cwd())
    p_pp.add_argument("--no-flows", action="store_true", help="Skip flow detection")
    p_pp.add_argument("--no-communities", action="store_true", help="Skip community detection")
    p_pp.add_argument("--no-risk", action="store_true", help="Skip risk scoring")
    p_pp.add_argument("--no-fts", action="store_true", help="Skip FTS rebuild")

    # status
    p_status = sub.add_parser("status", help="Show installation and graph status")
    p_status.add_argument("--project-root", default=_cwd())

    # watch
    p_watch = sub.add_parser("watch", help="Watch for file changes and auto-update graph")
    p_watch.add_argument("--project-root", default=_cwd())

    # claude-md
    p_claude = sub.add_parser("claude-md", help="Auto-generate CLAUDE.md with project topology")
    p_claude.add_argument("--project-root", default=_cwd())
    p_claude.add_argument("--force", action="store_true", help="Overwrite existing CLAUDE.md")

    # detect-changes
    p_dc = sub.add_parser("detect-changes", help="Show risk-scored impact analysis for changed files")
    p_dc.add_argument("--project-root", default=_cwd())
    p_dc.add_argument("files", nargs="*", metavar="FILE", help="Changed files to analyze")

    # visualize
    p_viz = sub.add_parser("visualize", help="Generate interactive HTML dependency graph")
    p_viz.add_argument("--project-root", default=_cwd())
    p_viz.add_argument("--serve", action="store_true", help="Serve on localhost:8765 after generating")

    # wiki
    p_wiki = sub.add_parser("wiki", help="Generate markdown wiki from community structure")
    p_wiki.add_argument("--project-root", default=_cwd())
    p_wiki.add_argument("--force", action="store_true", help="Regenerate all pages")

    # uninstall
    p_uninstall = sub.add_parser("uninstall", help="Remove graphsift from Claude Code config")
    p_uninstall.add_argument("--project-root", default=_cwd())

    # register
    p_register = sub.add_parser("register", help="Register a repo in the global graphsift registry")
    p_register.add_argument("--project-root", default=_cwd(), help="Repo root to register (default: cwd)")
    p_register.add_argument("--name", default="", help="Optional display name for this repo")

    # unregister
    p_unreg = sub.add_parser("unregister", help="Remove a repo from the graphsift registry")
    p_unreg.add_argument("path_or_name", help="Repo path or name to remove")

    # list-repos / repos
    sub.add_parser("list-repos", help="List all registered repos")
    sub.add_parser("repos", help="List all registered repos (alias for list-repos)")

    # gain
    p_gain = sub.add_parser("gain", help="Show token savings analytics")
    p_gain.add_argument("--json", action="store_true", help="Output as JSON")
    p_gain.add_argument("--history", action="store_true", help="Show analytics history")
    p_gain.add_argument("--project-root", default=_cwd(), help="Repo root (default: cwd)")

    # compress
    p_compress = sub.add_parser("compress", help="Compress command output (rtk-style)")
    p_compress.add_argument("--type", "-t", default="auto",
                            help="Output type (default: auto-detect)")
    p_compress.add_argument("--ultra", "-u", action="store_true",
                            help="Aggressive compression")
    p_compress.add_argument("--tee", "-e", default=None,
                            help="Directory to save original (uncompressed) output for tee recovery")
    p_compress.add_argument("--tee-label", default="output",
                            help="Filename label for tee save (default: output)")
    p_compress.add_argument("--list", "-l", action="store_true",
                            help="List available compressor types and exit")

    # discover
    p_discover = sub.add_parser("discover", help="Find missed token-saving opportunities")
    p_discover.add_argument("--project-root", default=_cwd(), help="Repo root (default: cwd)")

    # bash-wrapper
    sub.add_parser("bash-wrapper", help="Print bash wrapper script for transparent command compression")

    # detect-cycles
    p_cycles = sub.add_parser(
        "detect-cycles",
        help="Detect circular dependencies (import/call cycles)",
    )
    p_cycles.add_argument("--root", default=None, help="Repository root path")
    p_cycles.set_defaults(func=cmd_detect_cycles)

    # detect-dead-code
    p_dead = sub.add_parser(
        "detect-dead-code",
        help="Find potentially unreachable/dead code",
    )
    p_dead.add_argument("--root", default=None, help="Repository root path")
    p_dead.add_argument("--kind", choices=["function", "class", "method"], help="Filter by node kind")
    p_dead.add_argument("--entry-points", help="Comma-separated entry-point file paths")
    p_dead.add_argument("--prioritize", default=True, action="store_true",
                        help="Apply priority scoring (default: on)")
    p_dead.add_argument("--no-prioritize", dest="prioritize", action="store_false",
                        help="Skip priority scoring, show raw results")
    p_dead.add_argument("--all", action="store_true",
                        help="Show all findings, including low-priority ones")
    p_dead.set_defaults(func=cmd_detect_dead_code)

    # suggest-fixes
    p_suggest = sub.add_parser(
        "suggest-fixes",
        help="Run auto-fix analysis and return prioritized fix suggestions",
    )
    p_suggest.add_argument("--project-root", default=_cwd(), help="Repository root path (default: cwd)")
    p_suggest.add_argument("--changed-files", nargs="*", metavar="FILE",
                           help="Only show suggestions for these files")
    p_suggest.add_argument("--json", action="store_true", help="Output as JSON")
    p_suggest.add_argument("--min-confidence", type=float, default=0.0,
                           help="Minimum confidence threshold 0-1 (default 0.0)")
    p_suggest.set_defaults(func=cmd_suggest_fixes)

    # terse
    p_terse = sub.add_parser("terse", help="Return a terse mode prefix for LLM prompts")
    p_terse.add_argument("--level", choices=["lite", "full", "ultra"], default="lite",
                         help="Terseness level (default: lite)")
    p_terse.add_argument("--prompt", type=str, default="",
                         help="Optional prompt text to prefix")

    # fix
    p_fix = sub.add_parser("fix", help="Generate a bug-fix prompt template")
    p_fix.add_argument("--bug", required=True, help="Bug description")
    p_fix.add_argument("--file", required=True, help="File path containing the bug")
    p_fix.add_argument("--line", type=int, default=None, help="Line number of the bug")
    p_fix.add_argument("--expected", default="", help="Expected behavior")
    p_fix.add_argument("--actual", default="", help="Actual behavior")
    p_fix.add_argument("--json", action="store_true", help="Output as JSON")

    # add
    p_add = sub.add_parser("add", help="Generate a feature-addition prompt template")
    p_add.add_argument("--feature", required=True, help="Feature description")
    p_add.add_argument("--files", nargs="*", metavar="FILE", help="Files to modify")
    p_add.add_argument("--acceptance-criteria", nargs="*", metavar="CRITERION",
                       help="Acceptance criteria")
    p_add.add_argument("--json", action="store_true", help="Output as JSON")

    # refactor
    p_refactor = sub.add_parser("refactor", help="Generate a refactoring prompt template")
    p_refactor.add_argument("--target", required=True, help="Target to refactor")
    p_refactor.add_argument("--goal", default="", help="Refactoring goal")
    p_refactor.add_argument("--files", nargs="*", metavar="FILE", help="Files involved")
    p_refactor.add_argument("--json", action="store_true", help="Output as JSON")

    # verify
    p_verify = sub.add_parser("verify", help="Verify file syntax and lint")
    p_verify.add_argument("--file", required=True, help="File path to verify")
    p_verify.add_argument("--project-root", default=_cwd(), help="Project root (default: cwd)")

    # tool-budgets
    p_tb = sub.add_parser("tool-budgets", help="Show or set per-tool output line budgets")
    p_tb.add_argument("--show", action="store_true", help="Show all tool budgets")
    p_tb.add_argument("--tool", default=None, help="Tool name (bash, read, grep)")
    p_tb.add_argument("--set", type=int, default=None, dest="set_lines",
                      help="Set line cap for the specified --tool")

    # read-cache
    p_rc = sub.add_parser("read-cache", help="Show or clear the read-dedup cache")
    p_rc.add_argument("--stats", action="store_true", help="Show cache statistics")
    p_rc.add_argument("--clear", action="store_true", help="Clear the cache")

    # evidence
    p_ev = sub.add_parser("evidence", help="Check text for hallucinated file:line citations")
    p_ev.add_argument("--text", required=True, help="Text to check for citations")
    p_ev.add_argument("--project-root", default=_cwd(), help="Project root (default: cwd)")

    # evolve
    evolve_parser = sub.add_parser(
        "evolve",
        help="Run or manage evolutionary parameter optimization.",
    )
    evolve_sub = evolve_parser.add_subparsers(dest="evolve_action")

    # evolve run
    run_parser = evolve_sub.add_parser("run", help="Run evolution on the current codebase.")
    run_parser.add_argument("--rounds", type=int, default=40, help="Number of evolution rounds.")
    run_parser.add_argument("--population", type=int, default=6, help="Population per round.")
    run_parser.add_argument("--source-dir", type=str, default=".", help="Root directory of the codebase.")

    # evolve status
    evolve_sub.add_parser("status", help="Show cached evolution results.")

    # evolve clear
    evolve_sub.add_parser("clear", help="Clear all cached evolution results.")

    # -----------------------------------------------------------------------
    # loop command  (loop-engineering)
    # -----------------------------------------------------------------------
    loop_parser = sub.add_parser("loop", help="Loop-engineering: scheduled automation patterns")
    loop_sub = loop_parser.add_subparsers(dest="loop_action", required=True)

    # loop init
    p_loop_init = loop_sub.add_parser("init", help="Scaffold loop config for the project")
    p_loop_init.add_argument("--project-root", default=_cwd())

    # loop run
    p_loop_run = loop_sub.add_parser("run", help="Run a specific loop pattern")
    p_loop_run.add_argument("pattern", choices=[p.value for p in PatternType], help="Pattern to run")
    p_loop_run.add_argument("--project-root", default=_cwd())

    # loop status
    p_loop_status = loop_sub.add_parser("status", help="Show loop engine status")
    p_loop_status.add_argument("--project-root", default=_cwd())

    # loop report
    p_loop_report = loop_sub.add_parser("report", help="Full loop activity report")
    p_loop_report.add_argument("--project-root", default=_cwd())

    # loop schedule
    p_loop_sched = loop_sub.add_parser("schedule", help="List scheduled patterns")
    p_loop_sched.add_argument("--project-root", default=_cwd())

    # loop cost
    p_loop_cost = loop_sub.add_parser("cost", help="Estimate token cost per pattern")
    p_loop_cost.add_argument("--pattern", choices=[p.value for p in PatternType], required=True, help="Pattern type")
    p_loop_cost.add_argument("--maturity", choices=["L1", "L2", "L3"], default="L1", help="Maturity level")
    p_loop_cost.add_argument("--project-root", default=_cwd())

    # loop audit
    p_loop_audit = loop_sub.add_parser("audit", help="Loop readiness score and suggestions")
    p_loop_audit.add_argument("--project-root", default=_cwd())

    # loop start (one-shot session start diagnostic)
    p_loop_start = loop_sub.add_parser("session-start", help="Run one-shot session start diagnostic (~12K tokens, no background loops)")
    p_loop_start.add_argument("--project-root", default=_cwd())

    # loop diagnose (run when user is struggling)
    p_loop_diag = loop_sub.add_parser("diagnose", help="Run comprehensive diagnostic (use when struggling with code)")
    p_loop_diag.add_argument("--project-root", default=_cwd())
    p_loop_diag.add_argument("--message", default="", help="User message to analyze for struggle signals")

    # loop struggle (check for struggle signals)
    p_loop_struggle = loop_sub.add_parser("struggle", help="Check if user is showing struggle signals")
    p_loop_struggle.add_argument("--message", default="", help="User message to analyze")
    p_loop_struggle.add_argument("--failures", type=int, default=0, help="Number of repeated failures")
    p_loop_struggle.add_argument("--project-root", default=_cwd())

    # loop cost
    p_loop_breaker = loop_sub.add_parser("reset-breaker", help="Reset circuit breaker for a pattern")
    p_loop_breaker.add_argument("pattern", help="Pattern name to reset")
    p_loop_breaker.add_argument("--project-root", default=_cwd())

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Handle evolve command with sub-actions inline
    if args.command == "evolve":
        from graphsift.evolve_registry import EvolveRegistry
        registry = EvolveRegistry()

        if args.evolve_action == "status":
            entries = registry.list_entries()
            if not entries:
                print("No evolution results cached.")
                return
            print(f"Cached evolution results ({len(entries)}):")
            print(f"  {'Fingerprint':<20} {'Score':<10} {'Params':<60}")
            print(f"  {'-'*90}")
            for e in entries:
                fp = e.get("fingerprint", "?")[:16]
                score = f"{e.get('score', 0):.4f}"
                params_str = str(e.get("params", {}))
                if len(params_str) > 55:
                    params_str = params_str[:55] + "..."
                print(f"  {fp:<20} {score:<10} {params_str:<60}")

        elif args.evolve_action == "clear":
            registry.clear()
            print("Evolution cache cleared.")

        elif args.evolve_action == "run":
            # Collect source files
            from pathlib import Path
            from graphsift import PythonParser, DependencyGraph, DiffSpec

            src_dir = Path(args.source_dir).resolve()
            source_map = {}
            for py_file in src_dir.rglob("*.py"):
                if "venv" in str(py_file) or ".venv" in str(py_file) or "__pycache__" in str(py_file):
                    continue
                try:
                    source_map[str(py_file.relative_to(src_dir))] = py_file.read_text(encoding="utf-8")
                except Exception:
                    continue

            if not source_map:
                print(f"No Python files found in {args.source_dir}")
                return

            print(f"Found {len(source_map)} Python files. Running evolution...")
            print(f"  Rounds: {args.rounds}, Population: {args.population}")

            from graphsift.evolve import EvolutionOptimizer, ParameterSpace, make_evaluator

            diff = DiffSpec(changed_files=list(source_map.keys())[:3], query="Optimize params")
            space = ParameterSpace.full_space()
            optimizer = EvolutionOptimizer(space, seed=42, verbose=True)
            evaluator = make_evaluator(source_map, diff, space_type="full")

            result = optimizer.optimize(
                seed_params=space.defaults(),
                evaluator=evaluator,
                rounds=args.rounds,
                population=args.population,
            )

            print(f"\nEvolution complete!")
            print(f"  Best score: {result.best_score:.4f}")
            print(f"  Improvements: {result.improvements}/{result.rounds}")
            print(f"  Duration: {result.duration_s:.1f}s")
            print(f"\nBest params:")
            for k, v in result.best_params.items():
                print(f"  {k}: {v}")

        return

    # Handle loop command with sub-actions
    if args.command == "loop":
        loop_actions = {
            "init": cmd_loop_init,
            "run": cmd_loop_run,
            "status": cmd_loop_status,
            "report": cmd_loop_report,
            "schedule": cmd_loop_schedule,
            "cost": cmd_loop_cost,
            "audit": cmd_loop_audit,
            "session-start": cmd_loop_session_start,
            "diagnose": cmd_loop_diagnose,
            "struggle": cmd_loop_struggle,
            "reset-breaker": cmd_loop_reset_breaker,
        }
        fn = loop_actions.get(args.loop_action)
        if fn:
            sys.exit(fn(args))
        else:
            print(f"[graphsift] Unknown loop action: {args.loop_action}")
            sys.exit(1)

    commands = {
        "install": cmd_install,
        "serve": cmd_serve,
        "build": cmd_build,
        "update": cmd_update,
        "postprocess": cmd_postprocess,
        "status": cmd_status,
        "watch": cmd_watch,
        "detect-changes": cmd_detect_changes,
        "visualize": cmd_visualize,
        "wiki": cmd_wiki,
        "uninstall": cmd_uninstall,
        "register": cmd_register,
        "unregister": cmd_unregister,
        "list-repos": cmd_list_repos,
        "repos": cmd_repos,
        "gain": cmd_gain,
        "stats": cmd_stats,
        "compress": cmd_compress,
        "discover": cmd_discover,
        "bash-wrapper": cmd_bash_wrapper,
        "detect-cycles": cmd_detect_cycles,
        "detect-dead-code": cmd_detect_dead_code,
        "suggest-fixes": cmd_suggest_fixes,
        "terse": cmd_terse,
        "fix": cmd_fix,
        "add": cmd_add,
        "refactor": cmd_refactor,
        "verify": cmd_verify,
        "tool-budgets": cmd_tool_budgets,
        "read-cache": cmd_read_cache,
        "evidence": cmd_evidence,
        "claude-md": cmd_claude_md,
    }

    # Support func-based dispatch for new-style subcommands
    fn = getattr(args, "func", None) or commands.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(fn(args))


if __name__ == "__main__":
    main()
