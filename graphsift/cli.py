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

        # SessionInit - prime Claude with graph awareness
        settings["hooks"].setdefault("SessionInit", [])
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
            for entry in settings["hooks"]["SessionInit"]
            for h in entry.get("hooks", [])
        ]
        if not any("graphsift" in c for c in existing_cmds):
            settings["hooks"]["SessionInit"].append(session_hook)

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

        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        print(f"[graphsift] Wrote hooks -> {settings_path}")

    # 3. Write skill files
    if not args.no_skills:
        _write_skills(project_root)

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

def cmd_build(args: argparse.Namespace) -> int:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(message)s", stream=sys.stderr)

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

    # Open SQLite store — runs all pending migrations and logs them
    db_path = _db_path_for_root(str(root))
    store = GraphStore(db_path)

    print(f"[graphsift] Indexing {root} ...")
    source_map = load_source_map(str(root), extensions=extensions, exclude_dirs=exclude_dirs)
    total_files = len(source_map)

    builder = ContextBuilder(ContextConfig())

    # Index file-by-file with progress reporting
    all_paths = list(source_map.keys())
    for i, path in enumerate(all_paths, 1):
        try:
            builder.index_file(path, source_map[path])
        except Exception:
            pass
        if progress_interval > 0 and i % progress_interval == 0:
            print(f"INFO: Progress: {i}/{total_files} files parsed", file=sys.stderr)

    print(f"INFO: Progress: {total_files}/{total_files} files parsed", file=sys.stderr)

    stats = builder.index_files(source_map)

    # Persist nodes + files to SQLite
    graph = getattr(builder, "graph", None)
    if graph is not None:
        all_nodes: list[GraphNode] = []
        all_file_nodes = []
        for file_node in graph.all_files():
            all_file_nodes.append(file_node)
            for sym in file_node.symbols:
                all_nodes.append(
                    GraphNode(
                        node_id=f"{file_node.path}::{sym}",
                        file_path=file_node.path,
                        kind=NodeKind.FUNCTION,
                        name=sym,
                        qualified_name=sym,
                        language=file_node.language,
                    )
                )
        store.save_nodes(all_nodes)
        store.save_files(all_file_nodes)

    # Persist a lightweight index manifest (paths + token estimates only)
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

    print(f"[graphsift] Indexed {stats.files_indexed} files, "
          f"{stats.symbols_extracted} symbols, {stats.edges_created} edges "
          f"in {stats.duration_ms:.0f} ms")
    print(f"[graphsift] DB      -> {db_path}")
    print(f"[graphsift] Manifest-> {manifest_path}")
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
    for skill_dir in ["graphsift-build", "graphsift-review", "graphsift-impact"]:
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

    print(f"[graphsift] Wrote 3 skill files -> {skills_root}")


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
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphsift",
        description="graphsift - smarter code context for LLMs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # install
    p_install = sub.add_parser("install", help="Register graphsift with Claude Code")
    p_install.add_argument("--project-root", default=_cwd(), help="Repo root (default: cwd)")
    p_install.add_argument("--no-hooks", action="store_true", help="Skip hook injection")
    p_install.add_argument("--no-skills", action="store_true", help="Skip skill file creation")

    # serve
    sub.add_parser("serve", help="Start MCP stdio server (used by Claude Code)")

    # build
    p_build = sub.add_parser("build", help="Index repo and build dependency graph")
    p_build.add_argument("--project-root", default=_cwd())
    p_build.add_argument("--extensions", nargs="*", metavar="EXT")
    p_build.add_argument("--exclude-dirs", nargs="*", metavar="DIR")
    p_build.add_argument("--progress-interval", type=int, default=200,
                         help="Log progress every N files (default 200, 0=disable)")

    # update
    p_update = sub.add_parser("update", help="Incrementally update graph (changed files only)")
    p_update.add_argument("--project-root", default=_cwd())

    # status
    p_status = sub.add_parser("status", help="Show installation and graph status")
    p_status.add_argument("--project-root", default=_cwd())

    # uninstall
    p_uninstall = sub.add_parser("uninstall", help="Remove graphsift from Claude Code config")
    p_uninstall.add_argument("--project-root", default=_cwd())

    # register
    p_register = sub.add_parser("register", help="Register a repo in the global graphsift registry")
    p_register.add_argument("--project-root", default=_cwd(), help="Repo root to register (default: cwd)")
    p_register.add_argument("--name", default="", help="Optional display name for this repo")

    # list-repos
    sub.add_parser("list-repos", help="List all registered repos")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    commands = {
        "install": cmd_install,
        "serve": cmd_serve,
        "build": cmd_build,
        "update": cmd_update,
        "status": cmd_status,
        "uninstall": cmd_uninstall,
        "register": cmd_register,
        "list-repos": cmd_list_repos,
    }

    fn = commands.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(fn(args))


if __name__ == "__main__":
    main()
