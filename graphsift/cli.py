"""graphsift CLI - install, serve, build, update, status, register, list-repos."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import sys

from pathlib import Path

from graphsift.read_cache import SafeFileIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Production safeguards for auto-scan
# ---------------------------------------------------------------------------

# Size gate: skip auto-scan on repos larger than this (load_source_map is O(n))
_MAX_AUTO_SCAN_FILES = 5000

# Batch gate: skip auto-scan when more than this many files are deleted at once
# to avoid scanning after bulk operations like rm -rf src/
_MAX_AUTO_SCAN_DELETIONS = 10

# Rate limit: minimum seconds between auto-scans for the same repo root
_MIN_SCAN_INTERVAL_S = 30.0

# Track last scan time per project root (module-level) for rate limiting
_last_scan_times: dict[str, float] = {}


def _should_auto_scan(root: str, deleted_count: int, total_files: int) -> bool:
    """Check all production safeguards before running an auto-scan.

    Returns False when any gate is triggered, with a debug log explaining why.
    """
    import time as _time
    now = _time.monotonic()

    # Size gate
    if total_files > _MAX_AUTO_SCAN_FILES:
        logger.debug(
            "graphsift: auto-scan skipped — repo has %d files (max %d)",
            total_files, _MAX_AUTO_SCAN_FILES,
        )
        return False

    # Batch gate
    if deleted_count > _MAX_AUTO_SCAN_DELETIONS:
        logger.debug(
            "graphsift: auto-scan skipped — %d files deleted in batch (max %d)",
            deleted_count, _MAX_AUTO_SCAN_DELETIONS,
        )
        return False

    # Rate limit gate
    last = _last_scan_times.get(root, 0.0)
    elapsed = now - last
    if elapsed < _MIN_SCAN_INTERVAL_S:
        logger.debug(
            "graphsift: auto-scan skipped — last scan %.1fs ago (min %.0fs)",
            elapsed, _MIN_SCAN_INTERVAL_S,
        )
        return False

    _last_scan_times[root] = now
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cwd() -> str:
    return os.getcwd()


def _safe_print(*args, **kwargs) -> None:
    """Print with Unicode fallback for terminals that don't support it (Windows)."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Replace non-ASCII characters with ASCII equivalents
        sanitized = []
        for a in args:
            if isinstance(a, str):
                sanitized.append(a.encode("ascii", errors="replace").decode("ascii"))
            else:
                sanitized.append(str(a))
        print(*sanitized, **kwargs)


def _find_claude_settings(project_root: Path) -> Path:
    """Return path to .claude/settings.json, creating dirs if needed."""
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    return claude_dir / "settings.json"


def _find_mcp_json(project_root: Path) -> Path:
    return project_root / ".mcp.json"


def _python_executable() -> str:
    return sys.executable


def _mcp_server_command() -> tuple[str, list[str]]:
    """(command, args) used to launch the graphsift MCP server.

    Uses ``python -m graphsift.mcp_server`` rather than the ``graphsift-mcp``
    console script. On Windows a running MCP server holds the pip-managed
    console script open, so ``pip install -U graphsift`` fails with
    WinError 32 (file in use). A ``python -m`` process only locks python.exe
    (never replaced by pip), so upgrades succeed even while the server runs.
    The interpreter is resolved to the current env at write time; the
    auto-configure refreshes it idempotently on later imports.
    """
    return sys.executable, ["-m", "graphsift.mcp_server"]


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
            mcp_config = SafeFileIO.read_json(mcp_path)
        except Exception:
            mcp_config = {}

    # Top-level key is "mcpServers" per Claude Code spec
    mcp_config.setdefault("mcpServers", {})
    mcp_cmd, mcp_args = _mcp_server_command()
    mcp_config["mcpServers"]["graphsift"] = {
        "command": mcp_cmd,
        "args": mcp_args,
        "env": {},
    }
    SafeFileIO.write_json(mcp_path, mcp_config)
    print(f"[graphsift] Wrote {mcp_path}")

    # 2. Inject hooks into .claude/settings.json
    if not args.no_hooks:
        settings_path = _find_claude_settings(project_root)
        settings: dict = {}
        if settings_path.exists():
            try:
                settings = SafeFileIO.read_json(settings_path)
            except Exception:
                settings = {}

        settings.setdefault("hooks", {})

        # SessionStart — auto-start daemon + prime graph awareness
        settings["hooks"].setdefault("SessionStart", [])
        session_hook = {
            "matcher": "",
            "hooks": [
                {
                    "type": "command",
                    "command": (
                        f"\"{_python_executable()}\" -c \""
                        "import graphsift; "
                        "try: "
                        "  from graphsift.daemon import start; "
                        "  r = start(); "
                        "  print(f'[graphsift] Daemon ready (pid {r.get(\\\"pid\\\",\\\"?\\\")})'); "
                        "except Exception as e: "
                        "  print(f'[graphsift] Ready — daemon unavailable: {e}'); "
                        "print('[graphsift] Graph is auto-maintained — no manual build needed.')"
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
                        f"\"{_python_executable()}\" -m graphsift.cli update "
                        f"--project-root \"{project_root}\" 2>{os.devnull} || true"
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
                        f"\"{_python_executable()}\" -c \""
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

        # PreToolUse — auto-route Bash/PowerShell through daemon (smart execution)
        settings["hooks"].setdefault("PreToolUse", [])
        pre_hook = {
            "matcher": "Bash|PowerShell",
            "hooks": [
                {
                    "type": "command",
                    "command": (
                        f"\"{_python_executable()}\" -m graphsift.hooks pre-bash-hook"
                    ),
                }
            ],
        }
        existing_pre = [
            h.get("command", "")
            for entry in settings["hooks"]["PreToolUse"]
            for h in entry.get("hooks", [])
        ]
        if not any("pre-bash-hook" in c for c in existing_pre):
            settings["hooks"]["PreToolUse"].append(pre_hook)
            print(f"[graphsift] Installed PreToolUse hook (auto-route Python commands through daemon)")

        # Pre-approve all graphsift commands — zero permission prompts
        settings.setdefault("allow", [])
        graphsift_patterns = [
            "graphsift *",
            "python -m graphsift.*",
        ]
        for pat in graphsift_patterns:
            if pat not in settings["allow"]:
                settings["allow"].append(pat)
        print(f"[graphsift] Pre-approved graphsift commands (no permission prompts)")

        SafeFileIO.write_json(settings_path, settings)
        print(f"[graphsift] Wrote hooks -> {settings_path}")

    # 3. Write skill files (project-scoped only — never user-global)
    if not args.no_skills:
        _write_skills(project_root)
        _cleanup_legacy_global_skills()

    # 3b. UI/UX design engine — auto-install once when missing (skip with --no-uiux-engine)
    if not args.no_uiux_engine:
        try:
            from graphsift.uiux import find_search_script, install_engine
            if find_search_script() is None:
                print("[graphsift] UI/UX engine not found — installing the MIT-licensed "
                      "ui-ux-pro-max-skill now (one-time, npm required)...")
                code, msg = install_engine()
                if code == 0:
                    print(f"[graphsift] {msg}")
                else:
                    print(f"[graphsift] UI/UX engine auto-install failed: {msg}", file=sys.stderr)
                    print("[graphsift]   The graphsift-uiux skill will still auto-trigger and "
                          "retry the install when you make your first UI/UX request.")
            else:
                print("[graphsift] UI/UX design engine found — `graphsift uiux` is ready.")
        except Exception as exc:
            print(f"[graphsift] UI/UX engine setup skipped ({exc}). "
                  "Install it anytime with `graphsift uiux --install`.")

    # 4. Install bash wrapper (auto-compress commands)
    if args.bash_wrapper:
        from .hooks import get_bash_wrapper_script
        bashrc_path = Path.home() / ".bashrc"
        wrapper_script = get_bash_wrapper_script(python_path=_python_executable())

        # Check if already installed
        existing = SafeFileIO.read(bashrc_path) if bashrc_path.exists() else ""
        if "# graphsift: transparent output compression" not in existing:
            with open(bashrc_path, "a", encoding="utf-8") as f:
                f.write(f"\n# graphsift: transparent output compression\n")
                f.write(f'eval "$({_python_executable()} -m graphsift.cli bash-wrapper)"\n')
            print(f"[graphsift] Installed bash wrapper -> {bashrc_path}")
        else:
            print(f"[graphsift] Bash wrapper already installed in {bashrc_path}")

    print("[graphsift] Installation complete.")
    print()
    _print_cli_instructions(args, project_root)
    return 0


def _print_cli_instructions(args: argparse.Namespace, project_root: Path) -> None:
    """Print per-CLI instructions based on install flags."""
    install_all = getattr(args, 'all', False)
    targets = []
    if install_all:
        targets = ["claude-code", "claude-desktop", "cursor", "windsurf", "continue", "codex", "copilot"]
    else:
        for name in ["claude-code", "claude-desktop", "cursor", "windsurf", "continue", "codex", "copilot"]:
            if getattr(args, name.replace("-", "_"), False):
                targets.append(name)
    if not targets:
        # Default: show all
        targets = ["claude-code", "claude-desktop", "cursor", "windsurf", "continue", "codex", "copilot"]

    mcp_path = _find_mcp_json(project_root)
    mcp_cmd, mcp_args = _mcp_server_command()

    instructions = {
        "claude-code": (
            "  Claude Code:  ✅ Auto (MCP + hooks already installed)\n"
            f"                MCP config: {mcp_path}\n"
            "                PostToolUse hooks auto-fire on every file change."
        ),
        "claude-desktop": (
            "  Claude Desktop:  ⚠️  Manual setup needed\n"
            "                 1. Open Claude Desktop → Settings → Developer → Edit Config\n"
            "                 2. Add to claude_desktop_config.json:\n"
            '                   { "mcpServers": { "graphsift": {'
            f' "command": "{mcp_cmd}",'
            f' "args": {json.dumps(mcp_args)}'
            " } } }\n"
            "                 3. Restart Claude Desktop\n"
            "                 4. No auto-hooks — run manually:\n"
            "                    'Build the graphsift graph' then use prune_refs tool"
        ),
        "cursor": (
            "  Cursor:  ✅ MCP auto-detected\n"
            f"          graphsift MCP server registered in {mcp_path}\n"
            "          Cursor reads .mcp.json automatically.\n"
            "          For auto-cleanup on file changes, run:\n"
            f"          graphsift watch --daemon --project-root {project_root}"
        ),
        "windsurf": (
            "  Windsurf:  ✅ MCP auto-detected (same .mcp.json)\n"
            f"            Config: {mcp_path}\n"
            "            For auto-cleanup, run:\n"
            f"            graphsift watch --daemon --project-root {project_root}"
        ),
        "continue": (
            "  Continue.dev:  ✅ MCP auto-detected\n"
            f"                Config: {mcp_path}\n"
            "                Continue reads .mcp.json automatically.\n"
            "                For auto-cleanup, run:\n"
            f"                graphsift watch --daemon --project-root {project_root}"
        ),
        "codex": (
            "  Codex CLI (OpenAI):  ⚠️  No MCP support\n"
            "                      Use pipe syntax:\n"
            "                        graphsift build\n"
            "                        pytest -v | graphsift compress\n"
            "                        graphsift prune-refs\n"
            "                      For auto-cleanup, run:\n"
            f"                        graphsift watch --daemon --project-root {project_root}"
        ),
        "copilot": (
            "  GitHub Copilot CLI:  ⚠️  No MCP support\n"
            "                      Use CLI commands directly:\n"
            "                        graphsift build\n"
            "                        graphsift prune-refs [--fix]\n"
            "                      For auto-cleanup, run:\n"
            f"                        graphsift watch --daemon --project-root {project_root}"
        ),
    }

    print("  Supported CLI / Agent integration:")
    print()
    for t in targets:
        if t in instructions:
            print(instructions[t])
            print()
    if any(t in targets for t in ["cursor", "windsurf", "continue", "codex", "copilot"]):
        print("  NOTE: For CLIs without PostToolUse hooks, the watch daemon")
        print("  provides the same auto-cleanup on file changes (2s poll).")
        print()
    print("  Next steps:")
    print("  1. Build the graph:              graphsift build")
    print("  2. Start auto-watch (optional):   graphsift watch --daemon")
    print("  3. Scan for stale refs:           graphsift prune-refs")
    print("  4. Fix stale refs (with backup):  graphsift prune-refs --fix")
    print()


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
        # Dot dirs (.*) are auto-skipped in load_source_map
        # Keep only explicit non-dot build/dep dirs
        "node_modules", "vendor", "Pods", "bower_components", "jspm_packages",
        "dist", "build", "target", "out", "cdk.out",
        "__pycache__", "*.egg-info", "coverage", "htmlcov",
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

    # Version-aware cleanup: if this repo's graph was built by an older
    # graphsift, stale nodes/edges from the previous parser must not linger —
    # force a clean rebuild instead of reusing the old data.
    from graphsift._version import __version__ as _GRAPHSIFT_VERSION
    _manifest_path = root / ".graphsift" / "manifest.json"
    _stored_version = "0"
    try:
        if _manifest_path.exists():
            _stored_version = SafeFileIO.read_json(_manifest_path).get("graphsift_version", "0")
    except Exception:
        _stored_version = "0"
    # Version-stale ⇒ automatically behave like --force (full clean rebuild) so
    # the user never has to pass --force themselves. Only applies when a build
    # already exists (a fresh repo just builds normally).
    version_changed = _manifest_path.exists() and _stored_version != _GRAPHSIFT_VERSION

    from graphsift.sha_cache import load_sha_cache, save_sha_cache
    sha_cache = load_sha_cache(str(root))
    incremental = bool(sha_cache) and not getattr(args, "force", False) and not version_changed
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

    if incremental:
        # ── Fast incremental path: walk paths, stat-check, only read changed ──
        from graphsift.adapters.filesystem import walk_repo
        from graphsift.sha_cache import stat_match
        walk_paths = walk_repo(str(root), extensions=extensions, exclude_dirs=exclude_dirs)
        total_files = len(walk_paths)
        all_paths = walk_paths  # full file list for progress & manifest

        from collections import Counter
        ext_counts = Counter(Path(p).suffix.lower() for p in all_paths)
        print(f"        found {total_files} files")
        for ext, cnt in ext_counts.most_common(8):
            print(f"          {ext or '(no ext)':10s}  {cnt}")
        print()

        # Stat-check every file against cache — zero content reads for unchanged
        source_map = {}
        for p in all_paths:
            if not stat_match(p, sha_cache):
                try:
                    source_map[p] = SafeFileIO.read(p)
                except OSError:
                    pass

        fast_unchanged = total_files - len(source_map)
        if fast_unchanged:
            print(f"        {fast_unchanged} files unchanged (mtime+size match) — content read skipped")
            print()
    else:
        # ── Full scan: read everything into memory ──
        source_map = load_source_map(str(root), extensions=extensions, exclude_dirs=exclude_dirs)
        total_files = len(source_map)
        all_paths = list(source_map.keys())

        from collections import Counter
        ext_counts = Counter(Path(p).suffix.lower() for p in source_map)
        print(f"        found {total_files} files")
        for ext, cnt in ext_counts.most_common(8):
            print(f"          {ext or '(no ext)':10s}  {cnt}")
        print()

    # ── Purge stale files (excluded dirs, deleted files) ──────────────
    purged = store.purge_stale_files(set(all_paths))
    if purged["files"]:
        print(f"        purged {purged['files']} stale files ({purged['nodes']} nodes, {purged['edges']} edges, {purged['risk']} risk)")
        print()

    # ── Step 3: Parse & index ─────────────────────────────────────────────────
    print(f"  [3/5] Parsing {total_files} files ...")
    from graphsift.models import DepthTier  # noqa: PLC0415
    depth_tier_val = DepthTier(getattr(args, 'depth', 'execution'))
    builder = ContextBuilder(ContextConfig(depth_tier=depth_tier_val))
    if incremental:
        builder._sha_cache = sha_cache
    skipped = 0
    unchanged = 0
    changed = 0  # tracks files that were actually re-parsed

    # Check if tqdm is available for a proper progress bar
    try:
        from tqdm import tqdm  # noqa: PLC0415
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    t_parse_start = time.monotonic()

    # Disable GC during hot parse loop — objects are short-lived; let OS reclaim
    gc.disable()
    try:
        if _HAS_TQDM:
            pbar = tqdm(all_paths, desc="        Parsing", unit="files", ncols=80)
            for path in pbar:
                if path not in source_map:
                    unchanged += 1
                    continue
                try:
                    builder.index_file(path, source_map[path])
                    changed += 1
                except Exception:
                    skipped += 1
            pbar.close()
        else:
            for i, path in enumerate(all_paths, 1):
                if path not in source_map:
                    unchanged += 1
                    continue
                try:
                    builder.index_file(path, source_map[path])
                    changed += 1
                except Exception:
                    skipped += 1
                if progress_interval > 0 and i % progress_interval == 0:
                    elapsed = time.monotonic() - t_parse_start
                    rate = i / elapsed if elapsed > 0 else 0
                    remaining = (total_files - i) / rate if rate > 0 else 0
                    pct = i * 100 // total_files
                    bar_len = 20
                    filled = int(bar_len * i / total_files)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"        [{bar}] {pct:>3}% | Processing file {i:>6}/{total_files} | ETA: {remaining:>5.0f}s")

            if total_files == 0 or (progress_interval > 0 and total_files % progress_interval != 0):
                elapsed = time.monotonic() - t_parse_start
                rate = total_files / elapsed if elapsed > 0 else 0
                bar_len = 20
                bar = "█" * bar_len
                print(f"        [{bar}] 100% | Processing file {total_files:>6}/{total_files} | {rate:.0f} files/s")

        parse_ms = (time.monotonic() - t_parse_start) * 1000
        print(f"        done in {parse_ms:.0f} ms  ({skipped} failed, {unchanged} unchanged)")
        print()
    finally:
        gc.enable()

    # ── Step 4: Build final graph stats ──────────────────────────────────────
    print("  [4/5] Building dependency graph ...")
    t_graph = time.monotonic()
    stats = builder.index_files_incremental(source_map) if incremental else builder.index_files(source_map)
    graph_ms = (time.monotonic() - t_graph) * 1000

    # Incremental builds (no-op or partial) report only what was re-indexed
    # this run, which is a misleading delta vs the whole repo. Always surface
    # the stored graph totals from the DB so the output reflects reality.
    noop = incremental and stats.files_indexed == 0 and total_files > 0
    db_pre = store.stats() if incremental else None

    if noop:
        print(f"        graph is up to date — {db_pre['files']} files | "
              f"{db_pre['nodes']} nodes | {db_pre['edges']} edges (no changes since last build)")
    elif incremental:
        print(f"        files indexed  : {stats.files_indexed} (of {total_files})")
        print(f"        symbols        : {stats.symbols_extracted} (this run)")
        print(f"        edges          : {stats.edges_created} (this run)")
        print(f"        graph total    : {db_pre['files']} files | {db_pre['nodes']} nodes | {db_pre['edges']} edges")
    else:
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
    edges_saved = 0

    if graph_obj is not None:
        if version_changed:
            # Built by an older graphsift — the previous parser's output may
            # no longer be valid, so start clean.
            purged = store.purge_all_graph_data()
            _n_cleaned = sum(purged.values())
            print(f"        version change {_stored_version} → {_GRAPHSIFT_VERSION} — "
                  f"cleaned {_n_cleaned} stale graph records")
        elif not incremental and db_stats.get("nodes", 0) > 0:
            # Full rebuild (--force): symbols deleted from files since the last
            # build must not linger in the DB.
            purged = store.purge_all_graph_data()
            _n_cleaned = sum(purged.values())
            if _n_cleaned:
                print(f"        cleaned {_n_cleaned} stale graph records (full rebuild)")
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
        all_edges = graph_obj.all_edges()
        store.save_nodes(all_nodes)
        store.save_files(all_file_nodes)
        if all_edges:
            store.save_edges(all_edges)
            edges_saved = len(all_edges)
        nodes_saved = len(all_nodes)
        files_saved = len(all_file_nodes)

    db_ms = (time.monotonic() - t_db) * 1000
    if noop:
        print(f"        graph unchanged — {db_pre['nodes']} nodes / {db_pre['files']} files / "
              f"{db_pre['edges']} edges already in DB (nothing new to save)")
    else:
        print(f"        nodes saved    : {nodes_saved}")
        print(f"        files saved    : {files_saved}")
        if edges_saved:
            print(f"        edges saved    : {edges_saved}")
    print(f"        time           : {db_ms:.0f} ms")
    print()

    # ── Step 6: Post-processing (flows, communities, risk, FTS) ───────────────
    pp_result: dict = {}
    if getattr(args, "postprocess", False):
        if incremental:
            # The in-memory graph only holds this run's re-indexed files, so
            # flows/communities computed here would be wrong. Post-processing
            # only makes sense on a full build.
            print("  [6/6] Post-processing skipped on incremental build (run `graphsift build --force --postprocess` to refresh)")
            print()
        else:
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
    else:
        print("  [6/6] Post-processing skipped (use --postprocess to enable)")
        print()

    # ── Manifest ──────────────────────────────────────────────────────────────
    # Persist real counts. On incremental builds the in-memory stats reflect
    # only this run's delta — write the actual stored graph totals (queried
    # after persist) so the manifest never records a misleading partial or
    # empty build.
    db_after = store.stats() if incremental else None
    if incremental:
        manifest_indexed = db_after["files"]
        manifest_symbols = db_after["nodes"]
        manifest_edges = db_after["edges"]
    else:
        manifest_indexed = stats.files_indexed
        manifest_symbols = stats.symbols_extracted
        manifest_edges = stats.edges_created

    manifest = {
        "root": str(root),
        "files_indexed": manifest_indexed,
        "symbols_extracted": manifest_symbols,
        "edges_created": manifest_edges,
        "duration_ms": stats.duration_ms,
        "graphsift_version": _GRAPHSIFT_VERSION,
        "files": [str(p) for p in all_paths],
    }
    manifest_path = _manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    SafeFileIO.write_json(manifest_path, manifest)

    # ── Save SHA cache for next incremental build ──
    if graph_obj is not None:
        for file_node in graph_obj.all_files():
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
        save_sha_cache(str(root), sha_cache)

    total_ms = (time.monotonic() - _t0) * 1000

    # ── Summary ───────────────────────────────────────────────────────────────
    print("  " + "-" * 45)
    print(f"  Build complete in {total_ms:.0f} ms")
    if incremental:
        note = " (unchanged — graph up to date)" if stats.files_indexed == 0 else f" ({stats.files_indexed} file(s) updated)"
        print(f"  {db_after['files']} files  |  {db_after['nodes']} nodes  |  {db_after['edges']} edges{note}")
    else:
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

def _hook_build_namespace(root: Path) -> argparse.Namespace:
    """Build a ``cmd_build`` Namespace for the auto-build path in ``cmd_update``.

    The PostToolUse hook runs on every Write/Edit/Bash; when a repo has no
    graph yet, the hook turns into a full build so indexing is fully automatic.
    """
    return argparse.Namespace(
        project_root=str(root),
        extensions=None,
        exclude_dirs=None,
        depth="execution",
        progress_interval=200,
        force=False,
        postprocess=False,
    )


def cmd_update(args: argparse.Namespace) -> int:
    """Incremental graph update — called by PostToolUse hook.  Never raises."""
    try:
        root = Path(args.project_root).resolve()
        manifest_path = root / ".graphsift" / "manifest.json"

        if not manifest_path.exists():
            # Fully automated: no graph for this repo yet — the first edit
            # auto-creates it. This is what removes the manual `graphsift
            # build` step for the PostToolUse hook.
            try:
                return cmd_build(_hook_build_namespace(root))
            except Exception:
                return 0

        try:
            manifest = SafeFileIO.read_json(manifest_path)
        except Exception:
            return 0

        # Version-aware: if the graph was built by an older graphsift, purge it
        # and rebuild with the current version instead of incrementally updating
        # on top of data the previous parser produced.
        from graphsift._version import __version__ as _GRAPHSIFT_VERSION
        if manifest.get("graphsift_version", "0") != _GRAPHSIFT_VERSION:
            return cmd_build(_hook_build_namespace(root))

        manifest_files: list[str] = manifest.get("files", [])
        manifest_mtime = manifest_path.stat().st_mtime

        # ── 1. Detect deleted files ──────────────────────────────────────────
        deleted: list[str] = []
        changed: list[str] = []
        for file_path in manifest_files:
            p = Path(file_path)
            if not p.exists():
                deleted.append(str(p))
            elif p.stat().st_mtime > manifest_mtime:
                changed.append(str(p))

        # ── 2. Clean up deleted files from DB ────────────────────────────────
        if deleted:
            from graphsift.adapters.storage import GraphStore
            store = GraphStore(_db_path_for_root(str(root)))
            for fp in deleted:
                try:
                    store.delete_file_completely(fp)
                except Exception:
                    pass
            logger.info("graphsift: purged %d deleted files from graph DB", len(deleted))

        # ── 3. Auto-scan for stale source-code references ────────────────────
        if deleted and _should_auto_scan(str(root), len(deleted), len(manifest_files)):
            try:
                from graphsift.cleanup import StaleRefScanner
                from graphsift.adapters.filesystem import load_source_map
                source_map = load_source_map(str(root))
                scanner = StaleRefScanner(project_root=str(root))
                report = scanner.scan_after_deletion(deleted, source_map=source_map)
                if report.findings:
                    high = report.by_severity.get("HIGH", 0)
                    med = report.by_severity.get("MEDIUM", 0)
                    logger.warning(
                        "graphsift: %d deleted file(s) have %d stale reference(s) "
                        "(%d HIGH, %d MEDIUM) in remaining source code. "
                        "Run 'graphsift prune-refs' to inspect or '--fix' to clean up.",
                        len(deleted), report.total, high, med,
                    )
            except Exception:
                pass

        # ── 4. Update changed files ──────────────────────────────────────────
        if changed:
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

            # Persist the re-indexed files to the DB — otherwise the hook only
            # updates the manifest and the graph goes stale on every edit.
            try:
                from graphsift.adapters.storage import GraphStore
                from graphsift.models import GraphNode, NodeKind
                store = GraphStore(_db_path_for_root(str(root)))
                graph_obj = getattr(builder, "_graph", None)
                if graph_obj is not None:
                    all_nodes: list[GraphNode] = []
                    all_file_nodes = []
                    for file_node in graph_obj.all_files():
                        all_file_nodes.append(file_node)
                        for sym in file_node.symbols:
                            if hasattr(sym, "node_id"):
                                all_nodes.append(sym)
                            else:
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
            except Exception:
                pass

        # ── 5. Auto-scan modified files for removed exports ─────────────────
        if changed and _should_auto_scan(str(root), len(changed), len(manifest_files)):
            try:
                from graphsift.cleanup import StaleRefScanner
                from graphsift.adapters.filesystem import load_source_map
                source_map = load_source_map(str(root))
                scanner = StaleRefScanner(project_root=str(root))
                report = scanner.scan_after_modification(changed, source_map=source_map)
                if report.findings:
                    high = report.by_severity.get("HIGH", 0)
                    med = report.by_severity.get("MEDIUM", 0)
                    logger.warning(
                        "graphsift: %d modified file(s) had %d symbol(s) removed "
                        "(%d HIGH, %d MEDIUM) that may break dependents. "
                        "Run 'graphsift prune-refs' to inspect.",
                        len(changed), report.total, high, med,
                    )
            except Exception:
                pass

        # ── 6. Update manifest ───────────────────────────────────────────────
        if deleted or changed:
            manifest["files"] = [f for f in manifest_files if f not in deleted]
            if changed:
                manifest["files_updated"] = changed
            if deleted:
                manifest["files_deleted"] = deleted
            SafeFileIO.write_json(manifest_path, manifest)
    except Exception:
        pass
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
            m = SafeFileIO.read_json(manifest_path)
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
            cfg = SafeFileIO.read_json(mcp_path)
            cfg.get("mcpServers", {}).pop("graphsift", None)
            SafeFileIO.write_json(mcp_path, cfg)
            print(f"[graphsift] Removed MCP entry from {mcp_path}")
        except Exception as exc:
            print(f"[graphsift] Warning: could not update {mcp_path}: {exc}")

    # Remove skills (project + any stale user-global leftovers)
    skills_dir = project_root / ".claude" / "skills"
    for skill_dir in ["graphsift-build", "graphsift-review", "graphsift-impact", "graphsift-compress", "graphsift-uiux"]:
        import shutil
        target = skills_dir / skill_dir
        if target.exists():
            shutil.rmtree(target)
    _cleanup_legacy_global_skills()
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

def _cleanup_legacy_global_skills() -> int:
    """Remove stale user-global graphsift skills/commands left by older versions.

    Older releases installed skills into ``~/.claude/skills`` (and commands),
    so the same slash commands appeared twice — once user-global, once
    project-scoped. Current releases only install project-scoped skills; this
    removes the legacy global leftovers for anyone upgrading.

    Handles both directory-form (``~/.claude/skills/graphsift-build/``) and
    single-file legacy (``~/.claude/skills/graphsift-build.md``) leftovers.
    Returns the number of paths removed.
    """
    import shutil
    names = ["graphsift-build", "graphsift-compress", "graphsift-impact",
             "graphsift-review", "graphsift-uiux"]
    removed = 0
    for base in (Path.home() / ".claude" / "skills", Path.home() / ".claude" / "commands"):
        if not base.is_dir():
            continue
        for name in names:
            target = base / name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed += 1
        for legacy in base.glob("graphsift-*.md"):
            try:
                if legacy.is_dir():
                    shutil.rmtree(legacy, ignore_errors=True)
                else:
                    legacy.unlink()
                removed += 1
            except OSError:
                pass
    return removed


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

    # Auto-triggering UI/UX design skill. Frontmatter description drives Claude
    # Code's skill router, so it activates automatically on UI/UX/frontend work
    # without the user running `graphsift uiux` by hand.
    _write_skill(
        skills_root / "graphsift-uiux" / "SKILL.md",
        title="graphsift: UI/UX Design Intelligence",
        description="UI/UX and frontend design intelligence for web and mobile. Use it whenever designing, building, styling, or reviewing any UI - landing pages, dashboards, SaaS apps, components, motion/animation, color palettes, typography, dark/light themes, responsive layouts. Auto-run `graphsift uiux \"<product/keywords>\" --design-system` for a complete design system (style, WCAG-tested colors, font pairing, motion presets, anti-patterns, pre-delivery checklist) and `graphsift uiux \"<keyword>\" --domain style|color|ux|typography|chart|...` for targeted design searches.",
        steps=[
            "Generate the design system first: run `graphsift uiux \"<product_type> <industry> <keywords>\" --design-system -p \"<Project>\"` to get style, palette, typography, motion, anti-patterns and a pre-delivery checklist. Tune with `--variance`, `--motion`, `--density` (1-10).",
            "Supplement with targeted searches when needed: `--domain` (style, color, ux, typography, chart, landing, icons, gsap, react, web) and `--stack` (react, nextjs, shadcn, html-tailwind, ...) for framework-specific guidelines.",
            "Apply the design system verbatim when writing UI code: exact WCAG-tested palette, font pairing, spacing scale and motion presets.",
            "Before delivering, run the pre-delivery checklist: no emoji icons (use SVG), hover states 150-300ms, focus-visible rings, 4.5:1 text contrast, prefers-reduced-motion respected, responsive at 375/768/1024/1440.",
            "If the engine is not installed, run `graphsift uiux --install` once (npm required). graphsift delegates to the MIT-licensed ui-ux-pro-max-skill; it ships no upstream code.",
        ],
        example="Design a landing page for a SaaS analytics tool",
        frontmatter={
            "name": "graphsift-uiux",
            "user-invocable": False,
            "description": ("UI/UX and frontend design intelligence for web and mobile. "
                            "Use it whenever designing, building, styling, or reviewing any UI - "
                            "landing pages, dashboards, SaaS apps, components, motion/animation, "
                            "color palettes, typography, dark/light themes, responsive layouts. "
                            "Auto-run `graphsift uiux \"<product/keywords>\" --design-system` for a "
                            "complete design system (style, WCAG-tested colors, font pairing, motion "
                            "presets, anti-patterns, pre-delivery checklist) and "
                            "`graphsift uiux \"<keyword>\" --domain style|color|ux|typography|chart` "
                            "for targeted design searches."),
        },
    )

    print(f"[graphsift] Wrote 5 skill files -> {skills_root}")


def _write_skill(
    path: Path,
    title: str,
    description: str,
    steps: list[str],
    example: str,
    frontmatter: dict | None = None,
) -> None:
    """Write a Claude Code skill file.

    If `frontmatter` is given (e.g. ``{"name": ..., "description": ...}``), a YAML
    frontmatter block is prepended so the skill auto-triggers based on its
    description. Skills without frontmatter are user-invocable only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    steps_md = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    body = (
        f"# {title}\n\n"
        f"{description}\n\n"
        f"## Steps\n\n{steps_md}\n\n"
        f"## Example trigger\n\n> {example}\n"
    )
    if frontmatter:
        fm_lines = []
        for k, v in frontmatter.items():
            if isinstance(v, bool):
                v = "true" if v else "false"
            fm_lines.append(f"{k}: {v}")
        content = "---\n" + "\n".join(fm_lines) + "\n---\n\n" + body
    else:
        content = body
    SafeFileIO.write(path, content)


# ---------------------------------------------------------------------------
# Registry helpers  (~/.graphsift/registry.json)
# ---------------------------------------------------------------------------

_REGISTRY_PATH = Path.home() / ".graphsift" / "registry.json"


def _load_registry() -> dict[str, dict]:
    if _REGISTRY_PATH.exists():
        try:
            return SafeFileIO.read_json(_REGISTRY_PATH)
        except Exception:
            pass
    return {}


def _save_registry(registry: dict[str, dict]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SafeFileIO.write_json(_REGISTRY_PATH, registry)


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


def _watch_loop(root: Path, manifest_path: Path) -> None:
    """Scan for file changes, update graph, auto-clean deleted files, scan stale refs.

    Designed for CLIs without PostToolUse hooks (Cursor, Windsurf, Continue.dev,
    Claude Desktop, Codex CLI, Copilot CLI).  Runs a 2-second poll loop.
    Never raises.
    """
    import time
    from graphsift.adapters.filesystem import load_changed_files
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

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

    try:
        last_mtimes = _scan_mtimes()
        while True:
            time.sleep(2)
            current = _scan_mtimes()
            changed = [p for p, mtime in current.items()
                       if p not in last_mtimes or last_mtimes[p] != mtime]
            removed = [p for p in last_mtimes if p not in current]

            if changed or removed:
                print(f"[graphsift] {len(changed)} changed, {len(removed)} removed — updating graph ...")

                # Handle deleted files: DB cleanup + stale ref scan
                if removed and _should_auto_scan(str(root), len(removed), len(current)):
                    try:
                        db_path = _db_path_for_root(str(root))
                        from graphsift.adapters.storage import GraphStore
                        store = GraphStore(db_path)
                        for fp in removed:
                            try:
                                store.delete_file_completely(fp)
                            except Exception:
                                pass
                        # Scan for stale references
                        from graphsift.cleanup import StaleRefScanner
                        from graphsift.adapters.filesystem import load_source_map
                        source_map = load_source_map(str(root))
                        scanner = StaleRefScanner(project_root=str(root))
                        report = scanner.scan_after_deletion(removed, source_map=source_map)
                        if report.findings:
                            print(f"[graphsift] WARNING: {report.total} stale reference(s) found "
                                  f"({report.by_severity.get('HIGH', 0)} HIGH). "
                                  f"Run: graphsift prune-refs --fix")
                    except Exception:
                        pass

                # Handle changed files: re-index + scan for removed exports
                if changed:
                    new_sources = load_changed_files(changed)
                    builder = ContextBuilder(ContextConfig())
                    for path, source in new_sources.items():
                        try:
                            builder.index_file(path, source)
                        except Exception:
                            pass
                    print(f"[graphsift] Updated {len(changed)} files.")
                    # Scan for symbols removed from modified files
                    if _should_auto_scan(str(root), len(changed), len(current)):
                        try:
                            from graphsift.cleanup import StaleRefScanner
                            from graphsift.adapters.filesystem import load_source_map
                            source_map = load_source_map(str(root))
                            scanner = StaleRefScanner(project_root=str(root))
                            report = scanner.scan_after_modification(changed, source_map=source_map)
                            if report.findings:
                                print(f"[graphsift] WARNING: {report.total} removed symbol reference(s) "
                                      f"found in modified files. Run: graphsift prune-refs")
                        except Exception:
                            pass

                last_mtimes = current
    except Exception:
        pass


def cmd_watch(args: argparse.Namespace) -> int:
    import threading

    root = Path(args.project_root).resolve()
    manifest_path = root / ".graphsift" / "manifest.json"

    if getattr(args, 'daemon', False):
        t = threading.Thread(target=_watch_loop, args=(root, manifest_path), daemon=True)
        t.start()
        print(f"[graphsift] Watch daemon started for {root}")
        return 0

    print(f"[graphsift] Watching {root} for changes (Ctrl+C to stop) ...")
    try:
        _watch_loop(root, manifest_path)
    except KeyboardInterrupt:
        print("\n[graphsift] Watch stopped.")
    return 0


# ---------------------------------------------------------------------------
# auto-guide functions
# ---------------------------------------------------------------------------


def auto_guide(
    task: str,
    changed_files: list[str] | None = None,
    project_root: str = "",
) -> dict:
    """Build focused code context for a given task. Never raises.

    Args:
        task: Free-text description of what you want to review or understand.
        changed_files: Optional list of changed file paths to anchor context.
        project_root: Repo root path (default: cwd).

    Returns:
        Dict with keys ``context`` (str) and ``files_selected`` (int).
    """
    try:
        from graphsift.adapters.filesystem import load_source_map
        from graphsift.core import ContextBuilder
        from graphsift.models import ContextConfig, DiffSpec

        root = project_root or os.getcwd()
        source_map = load_source_map(root)
        if not source_map:
            return {"context": "", "files_selected": 0}

        config = ContextConfig(token_budget=40_000)
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(
            changed_files=changed_files or list(source_map.keys())[:3],
            query=task,
        )
        result = builder.build(diff, source_map=source_map)
        return {
            "context": result.rendered_context,
            "files_selected": result.files_selected,
        }
    except Exception:
        return {"context": "", "files_selected": 0}


def cmd_guide(args: argparse.Namespace) -> int:
    """Print focused context for a task description."""
    task = " ".join(getattr(args, 'task', [])) or "Understand the codebase"
    result = auto_guide(
        task=task,
        project_root=getattr(args, 'project_root', os.getcwd()),
    )
    if result["files_selected"]:
        print(f"[graphsift] Selected {result['files_selected']} files for context")
        print()
    print(result["context"])
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
    SafeFileIO.write(output_path, html)

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


def cmd_daemon(args: argparse.Namespace) -> int:
    """Manage the persistent Python daemon (start/stop/status/cache)."""
    action = getattr(args, 'daemon_action', None) or 'status'
    from graphsift.daemon import start, stop, status, cache_stats, cache_clear

    actions = {
        'start': lambda: start(),
        'stop': lambda: stop(),
        'status': lambda: status(),
        'cache-stats': lambda: cache_stats(),
        'cache-clear': lambda: cache_clear(),
    }

    result_fn = actions.get(action)
    if result_fn is None:
        print(f"[graphsift] Unknown daemon action: {action}")
        print("  Available: start, stop, status, cache-stats, cache-clear")
        return 1

    result = result_fn()
    if action == 'status':
        st = result.get('status', 'unknown')
        pid = result.get('pid', '')
        if st == 'running':
            print(f"[graphsift] Daemon is RUNNING (pid {pid})")
        else:
            print("[graphsift] Daemon is STOPPED")
    elif action == 'cache-stats':
        print(f"[graphsift] In-process cache: {result.get('in_process', '?')} entries")
        print(f"[graphsift] Daemon cache:     {result.get('daemon', '?')}")
    else:
        st = result.get('status', result.get('ok', False))
        print(f"[graphsift] Daemon {action}: {st}")
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
# prune-refs command  (post-deletion reference cleanup)
# ---------------------------------------------------------------------------

def cmd_prune_refs(args: argparse.Namespace) -> int:
    """Scan for stale references to deleted files and optionally auto-fix.

    Detects import statements, symbol references, and string path references
    in remaining source files that point to deleted components.

    Use --fix to auto-remove stale import lines (creates .bak backups).
    """
    project_root = Path(getattr(args, 'project_root', os.getcwd())).resolve()
    deleted_paths = getattr(args, 'paths', None)
    fix = getattr(args, 'fix', False)

    try:
        from graphsift.cleanup import StaleRefScanner  # noqa: PLC0415
    except ImportError:
        print("[graphsift] cleanup module not available. Run: pip install graphsift")
        return 1

    from graphsift.adapters.filesystem import load_source_map  # noqa: PLC0415

    # If no paths given, detect deletions from manifest
    if not deleted_paths:
        manifest_path = project_root / ".graphsift" / "manifest.json"
        if manifest_path.exists():
            try:
                from graphsift.read_cache import SafeFileIO  # noqa: PLC0415
                manifest = SafeFileIO.read_json(manifest_path)
                manifest_files = manifest.get("files", [])
                deleted_paths = [
                    f for f in manifest_files
                    if not Path(f).exists()
                ]
                if not deleted_paths:
                    print("[graphsift] No deleted files found in manifest.")
                    return 0
            except Exception:
                print("[graphsift] Could not read manifest.")
                return 1
        else:
            print("[graphsift] No manifest found and no paths given.")
            print("  Usage: graphsift prune-refs [paths...] [--fix]")
            return 1

    # Scan
    source_map = load_source_map(str(project_root))
    scanner = StaleRefScanner(project_root=str(project_root))
    report = scanner.scan_after_deletion(deleted_paths, source_map=source_map)

    # Print report
    if not report.findings:
        print(f"[graphsift] No stale references found for {len(deleted_paths)} deleted file(s).")
        return 0

    _safe_print(f"\n  {'='*60}")
    _safe_print(f"  Stale Reference Report - {len(deleted_paths)} deleted file(s)")
    _safe_print(f"  {'='*60}")
    _safe_print(f"  Total findings: {report.total}")
    _safe_print(f"  By severity    : {report.by_severity.get('HIGH', 0)} HIGH, "
                f"{report.by_severity.get('MEDIUM', 0)} MEDIUM, "
                f"{report.by_severity.get('LOW', 0)} LOW")
    _safe_print(f"  By kind        : {report.by_kind}")
    _safe_print(f"  Auto-fixable   : {report.auto_fixable}")
    _safe_print(f"  {'='*60}\n")

    for severity in ("HIGH", "MEDIUM", "LOW"):
        group = [f for f in report.findings if f.severity == severity]
        if not group:
            continue
        _safe_print(f"  [{severity}] - {len(group)} finding(s)")
        _safe_print()
        for f in group[:20]:  # Show first 20
            _safe_print(f"    {f.file_path}:{f.line_number}")
            _safe_print(f"      {f.line_text.strip()}")
            if f.suggested_fix:
                _safe_print(f"      ==> {f.suggested_fix}")
            _safe_print()
        if len(group) > 20:
            _safe_print(f"    ... and {len(group) - 20} more")
            _safe_print()
        _safe_print()

    # Apply fixes if requested
    if fix:
        _safe_print("  Applying fixes...")
        result = scanner.apply_fixes(report, dry_run=False)
        _safe_print(f"  Files modified : {result.get('files_modified', 0)}")
        _safe_print(f"  Lines removed  : {result.get('lines_removed', 0)}")
        if result.get("files_backed_up"):
            _safe_print(f"  Backups created: {len(result['files_backed_up'])}")
        if result.get("errors"):
            _safe_print(f"  Errors         : {len(result['errors'])}")
            for e in result["errors"][:5]:
                _safe_print(f"    - {e}")
    else:
        fixable = sum(1 for f in report.findings if f.suggested_fix)
        if fixable:
            _safe_print(f"  Tip: {fixable} finding(s) are auto-fixable. Run with --fix to apply.")
        _safe_print()

    return 1 if report.total > 0 else 0


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
# test-impact command — smart selective test runner
# ---------------------------------------------------------------------------

def cmd_test_impact(args: argparse.Namespace) -> int:
    """Run full or selective tests with impact analysis."""
    from graphsift.test_impact import TestImpactAnalyzer

    analyzer = TestImpactAnalyzer(project_root=args.project_root)

    if args.mode == "status":
        snap = analyzer._memory.last_snapshot()
        if snap is None:
            print("No test snapshots found. Run `graphsift test-impact full` first.")
            return 1
        print(f"Test Snapshot #{snap.id}")
        print(f"  Mode       : {snap.mode}")
        print(f"  Status     : {snap.status}")
        print(f"  Commit     : {snap.commit_hash}")
        print(f"  Tests      : {snap.tests_run} run, "
              f"{snap.tests_passed} passed, {snap.tests_failed} failed")
        print(f"  Duration   : {snap.duration_ms:.0f}ms")
        print(f"  Timestamp  : {snap.created_at}")
        if snap.changed_files:
            print(f"  Changed    : {len(snap.changed_files)} files")
        if snap.impacted_tests:
            print(f"  Impacted   : {len(snap.impacted_tests)} test files")
        return 0 if snap.status == "passed" else 1

    if args.mode == "full":
        print("Running FULL test suite (baseline for selective mode)...")
        result = analyzer.run_full(
            pytest_args=args.pytest_args,
            timeout=args.timeout,
        )
    else:
        print("Running SELECTIVE tests (only impacted by changes)...")
        result = analyzer.run_selective(
            changed_files=args.changed_files,
            pytest_args=args.pytest_args,
            timeout=args.timeout,
        )

    print(f"\n{'='*60}")
    print(f"  {result.summary}")
    print(f"{'='*60}")
    if result.duration_ms > 0:
        print(f"  Time      : {result.duration_ms:.0f}ms")
    if result.tests_run > 0:
        print(f"  Tests     : {result.tests_run} total, "
              f"{result.tests_passed} passed, {result.tests_failed} failed")
    if result.skipped_tests > 0:
        print(f"  Skipped   : {result.skipped_tests} tests "
              f"({result.savings_pct:.0f}% savings)")
    if result.impacted_tests:
        print(f"  Files     : {len(result.impacted_tests)} impacted test files")
    if result.message:
        print(f"  Info      : {result.message}")

    return 0 if result.status == "passed" else 1


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
# guard command  (trading-strategy hallucination guard)
# ---------------------------------------------------------------------------

def cmd_guard(args: argparse.Namespace) -> int:
    """Audit / enforce AI-generated trading strategy text for hallucinated claims.

    Detects fabricated profit, win-rate, ROI, and guarantee claims by
    comparing them against real-time proven reference data (a real backtest
    or live P&L). This is the "44 lakh profit → 4 lakh real" collapse catcher.
    """
    from graphsift.guard import JsonBacktestProvider, StrategyGuard

    guard = StrategyGuard(provider=JsonBacktestProvider(args.reference))

    if args.action == "prompt":
        # Build the anti-hallucination grounding prompt (no audit).
        sys.stdout.write(guard.build_grounding_prompt(strategy_request=args.text))
        sys.stdout.write("\n")
        return 0

    if not args.text:
        print("error: --text is required for audit/enforce", file=sys.stderr)
        return 2

    if args.action == "audit":
        report = guard.audit(args.text)
        if args.json:
            import json as _json
            sys.stdout.write(_json.dumps(report.to_dict(), indent=2))
            sys.stdout.write("\n")
            return 1 if report.hallucination_score >= 50 else 0
        print(f"Claims found : {report.total_claims}")
        print(f"  verified   : {len(report.verified_claims)}")
        print(f"  synthetic  : {len(report.synthetic_claims)} (backtest, not live)")
        print(f"  contradicted: {len(report.contradicted_claims)}")
        print(f"  unverifiable: {len(report.unverifiable_claims)}")
        print(f"Hallucination score: {report.hallucination_score:.1f}/100")
        print(f"Risk level  : {report.risk_level}")
        if report.claims:
            print()
            print("Claims:")
            for c in report.claims:
                val = f"{c.value:,.0f}{c.unit}" if c.value is not None else "-"
                print(f"  [{c.status:13s}] {c.type:10s} {val:>12}  {c.raw!r}")
        return 1 if report.hallucination_score >= 50 else 0

    # enforce (mark / strip / enforce)
    cleaned, report = guard.enforce(args.text, mode=args.action)
    if args.json:
        import json as _json
        data = report.to_dict()
        data["rewritten_text"] = cleaned
        sys.stdout.write(_json.dumps(data, indent=2))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(cleaned)
        if not cleaned.endswith("\n"):
            sys.stdout.write("\n")
        sys.stderr.write(report.summary() + "\n")
    return 1 if report.hallucination_score >= 50 else 0


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
            pkg = SafeFileIO.read_json(pkg_json)
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
            content = SafeFileIO.read(makefile)
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
        # Dot dirs (.*) auto-skipped; list non-dot dirs only
        "node_modules", "vendor", "Pods", "bower_components", "jspm_packages",
        "dist", "build", "target", "out", "cdk.out",
        "__pycache__", "*.egg-info", "coverage", "htmlcov",
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
    SafeFileIO.write(claude_md_path, content)

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
# session commands  (durable session workspaces)
# ---------------------------------------------------------------------------


def _get_session_store():
	"""Create and return a SessionStore instance."""
	from graphsift.memory import SessionStore

	return SessionStore()


def _resolve_and_print(store, name_or_id):
	"""Resolve a session by name-or-id, printing an error on failure."""
	session = store.resolve_session(name_or_id)
	if session is None:
		print(f"[graphsift] Session not found: {name_or_id!r}", file=sys.stderr)
		return None
	return session


def _format_session(session):
	"""Format a SessionRecord for human-readable display."""
	lines = [
		f"  ID:          {session.session_id}",
		f"  Name:        {session.name}",
		f"  Description: {session.description or '(none)'}",
		f"  Created:     {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
		f"  Updated:     {session.updated_at.strftime('%Y-%m-%d %H:%M:%S')}",
		f"  Status:      {'Active' if session.is_active else 'Closed'}",
		f"  Repo root:   {session.repo_root or '(none)'}",
	]
	g = session.graph_hash
	if g:
		gh = g[:16] + "..." if len(g) > 16 else g
	else:
		gh = "(none)"
	lines.append(f"  Graph hash:  {gh}")
	return "\n".join(lines)


def cmd_session_create(args: argparse.Namespace) -> int:
	"""Create a new named analysis session."""
	store = _get_session_store()
	session = store.create_session(
		name=args.name,
		description=args.description,
		repo_root=args.repo if args.repo else "",
	)
	print(f"[graphsift] Session created:")
	print(_format_session(session))
	return 0


def cmd_session_list(args: argparse.Namespace) -> int:
	"""List all sessions."""
	store = _get_session_store()
	sessions = store.list_sessions(
		active_only=args.active_only, limit=args.limit
	)
	if not sessions:
		print("[graphsift] No sessions found.")
		return 0

	status_filter = "active" if args.active_only else "all"
	print(
		f"[graphsift] Sessions ({status_filter},"
		f" up to {args.limit} shown):"
	)
	print()
	print(f"  {'ID':<38} {'Name':<24} {'Status':<8} {'Updated':<22}")
	print(f"  {'-' * 92}")
	for s in sessions:
		sid_short = s.session_id[:12] + "..."
		status = "Active" if s.is_active else "Closed"
		updated = s.updated_at.strftime("%Y-%m-%d %H:%M:%S")
		print(
			f"  {sid_short:<38} {s.name:<24} {status:<8} {updated:<22}"
		)
	print()
	print(f"  Total: {len(sessions)} session(s)")
	return 0


def cmd_session_show(args: argparse.Namespace) -> int:
	"""Show session details."""
	store = _get_session_store()
	session = _resolve_and_print(store, args.name_or_id)
	if session is None:
		return 1

	print("[graphsift] Session details:")
	print(_format_session(session))
	print()

	if args.snapshots:
		snapshots = store.get_session_snapshots(session.session_id)
		if not snapshots:
			print("  No snapshots recorded.")
		else:
			print(f"  Snapshots ({len(snapshots)}):")
			print()
			for snap in snapshots:
				files = snap["files_affected"]
				files_str = ", ".join(files[:5])
				if len(files) > 5:
					files_str += f" ... (+{len(files) - 5} more)"
				print(
					f"    [{snap['analysis_type']}]"
					f" {snap['result_summary'][:120]}"
				)
				print(f"      Snapshot ID: {snap['snapshot_id'][:12]}...")
				print(f"      Time:        {snap['created_at']}")
				print(f"      Token cost:  {snap['token_cost']}")
				print(f"      Files:       {files_str or '(none)'}")
				print()
	return 0


def cmd_session_close(args: argparse.Namespace) -> int:
	"""Mark a session as inactive (soft-close)."""
	store = _get_session_store()
	session = _resolve_and_print(store, args.name_or_id)
	if session is None:
		return 1

	store.close_session(session.session_id)
	print(
		f"[graphsift] Session closed: {session.name}"
		f" ({session.session_id[:12]}...)"
	)
	return 0


def cmd_session_delete(args: argparse.Namespace) -> int:
	"""Permanently delete a session."""
	store = _get_session_store()
	session = _resolve_and_print(store, args.name_or_id)
	if session is None:
		return 1

	store.delete_session(session.session_id)
	print(
		f"[graphsift] Session deleted: {session.name}"
		f" ({session.session_id[:12]}...)"
	)
	return 0


def cmd_session_compare(args: argparse.Namespace) -> int:
	"""Compare two sessions."""
	store = _get_session_store()
	sa = _resolve_and_print(store, args.session_a)
	if sa is None:
		return 1
	sb = _resolve_and_print(store, args.session_b)
	if sb is None:
		return 1

	result = store.compare_sessions(sa.session_id, sb.session_id)

	print("[graphsift] Session comparison:")
	print()
	print(
		f"  Session A: {sa.name} ({sa.session_id[:12]}...)"
	)
	print(
		f"  Session B: {sb.name} ({sb.session_id[:12]}...)"
	)
	print()

	common = result.get("common_fields", {})
	if common:
		print("  Common fields:")
		for field, value in common.items():
			print(f"    {field}: {value}")
		print()

	counts = result.get("snapshot_counts", {})
	print(
		f"  Snapshot counts:  A={counts.get('a', 0)},"
		f"  B={counts.get('b', 0)}"
	)

	files_only_a = result.get("files_only_in_a", [])
	files_only_b = result.get("files_only_in_b", [])
	files_both = result.get("files_in_both", [])

	if files_only_a:
		print(f"  Files only in A:  {len(files_only_a)}")
		for f in files_only_a[:10]:
			print(f"    - {f}")
		if len(files_only_a) > 10:
			print(f"    ... and {len(files_only_a) - 10} more")
	if files_only_b:
		print(f"  Files only in B:  {len(files_only_b)}")
		for f in files_only_b[:10]:
			print(f"    - {f}")
		if len(files_only_b) > 10:
			print(f"    ... and {len(files_only_b) - 10} more")
	if files_both:
		print(f"  Files in both:    {len(files_both)}")
		for f in files_both[:10]:
			print(f"    - {f}")
		if len(files_both) > 10:
			print(f"    ... and {len(files_both) - 10} more")
	return 0


def cmd_session_prune(args: argparse.Namespace) -> int:
	"""Delete oldest sessions beyond keep count."""
	store = _get_session_store()
	deleted = store.prune_old_sessions(keep_count=args.keep)
	if deleted == 0:
		print(
			f"[graphsift] No sessions to prune"
			f" (keeping {args.keep} most recent)."
		)
	else:
		print(
			f"[graphsift] Pruned {deleted} old session(s)"
			f" (keeping {args.keep} most recent)."
		)
	return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

class _LazyPatternChoices:
    """Defer PatternType import (and thus loop_engineering module) until arg validation."""
    _values: list[str] | None = None

    def _load(self) -> list[str]:
        if self._values is None:
            from .loop_engineering import PatternType  # noqa: PLC0415
            _LazyPatternChoices._values = [p.value for p in PatternType]
        return self._values

    def __iter__(self):
        return iter(self._load())

    def __contains__(self, item):
        return item in self._load()

    def __len__(self):
        return len(self._load())


def _build_parser() -> argparse.ArgumentParser:
    from ._version import __version__  # noqa: PLC0415

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
    p_install.add_argument("--no-uiux-engine", action="store_true",
                           help="Skip auto-installing the ui-ux-pro-max design engine (npm -g install + uipro init)")
    p_install.add_argument("--bash-wrapper", action="store_true", help="Install transparent bash command compression")
    p_install.add_argument("--all", action="store_true", help="Show instructions for all supported CLIs")
    p_install.add_argument("--claude-code", action="store_true", help="Show Claude Code instructions (default)")
    p_install.add_argument("--claude-desktop", action="store_true", help="Show Claude Desktop instructions")
    p_install.add_argument("--cursor", action="store_true", help="Show Cursor instructions")
    p_install.add_argument("--windsurf", action="store_true", help="Show Windsurf instructions")
    p_install.add_argument("--continue", dest="continue_", action="store_true", help="Show Continue.dev instructions")
    p_install.add_argument("--codex", action="store_true", help="Show Codex CLI (OpenAI) instructions")
    p_install.add_argument("--copilot", action="store_true", help="Show Copilot CLI instructions")

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
    p_build.add_argument("--postprocess", action="store_true",
                         help="Run flow/community/risk/FTS post-processing (off by default for speed)")
    p_build.add_argument("--skip-postprocess", action="store_true",
                         help=argparse.SUPPRESS)  # deprecated no-op, kept for compat
    p_build.add_argument("--force", action="store_true",
                         help="Force full rebuild (ignore SHA cache)")

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
    p_watch.add_argument("--daemon", action="store_true", help="Run in background (no blocking)")

    # guide
    p_guide = sub.add_parser("guide", help="Pre-compute focused code context for agents")
    p_guide.add_argument("task", nargs="*", help="Task description")
    p_guide.add_argument("--project-root", default=_cwd())

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

    # daemon — manage persistent Python daemon
    p_daemon = sub.add_parser("daemon", help="Manage the persistent Python daemon (start/stop/status)")
    daemon_sub = p_daemon.add_subparsers(dest="daemon_action")
    daemon_sub.add_parser("start", help="Start the background Python daemon")
    daemon_sub.add_parser("stop", help="Stop the background Python daemon")
    daemon_sub.add_parser("status", help="Check if daemon is running")
    daemon_sub.add_parser("cache-stats", help="Show daemon cache statistics")
    daemon_sub.add_parser("cache-clear", help="Clear daemon caches")
    p_daemon.set_defaults(func=cmd_daemon)

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

    # prune-refs
    p_prune = sub.add_parser(
        "prune-refs",
        help="Scan for stale references to deleted files and optionally auto-fix",
    )
    p_prune.add_argument("--project-root", default=_cwd(), help="Repository root path (default: cwd)")
    p_prune.add_argument("--fix", action="store_true", help="Auto-remove stale import lines (creates .bak backups)")
    p_prune.add_argument("paths", nargs="*", metavar="PATH", help="Deleted file paths to scan (default: auto-detect from manifest)")
    p_prune.set_defaults(func=cmd_prune_refs)

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

    # guard — trading-strategy hallucination guard
    p_guard = sub.add_parser(
        "guard",
        help="Audit AI-generated trading strategy text for hallucinated "
             "claims (fake profit/win-rate/ROI vs real-time proven data). "
             "Catches the '44 lakh profit → 4 lakh real' collapse.",
    )
    p_guard.add_argument(
        "action", nargs="?",
        choices=["audit", "mark", "strip", "enforce", "prompt"],
        default="audit",
        help="audit (report only, default) | mark [UNVERIFIED]/[CONTRADICTED] | "
             "strip risky claims | enforce (strip + report) | prompt (build "
             "grounding prompt)",
    )
    p_guard.add_argument("--text", default="", help="AI-generated strategy text to audit")
    p_guard.add_argument(
        "--reference", default=None,
        help="Path to reference JSON (real backtest / live P&L stats). "
             "Defaults to built-in demo reference.",
    )
    p_guard.add_argument("--json", action="store_true", help="Output JSON report")

    # test-impact — smart selective test runner
    p_ti = sub.add_parser(
        "test-impact",
        help="Smart test runner: run only tests affected by changed files. "
             "Saves 60-95%% test time on incremental changes.",
    )
    p_ti.add_argument(
        "mode", nargs="?",
        choices=["full", "selective", "status"],
        default="selective",
        help="'full' = run all tests (baseline) | "
             "'selective' = run only impacted tests (default) | "
             "'status' = show last test snapshot",
    )
    p_ti.add_argument(
        "--changed-files", nargs="*", default=None,
        help="Space-separated file paths (auto-detected via git if omitted)",
    )
    p_ti.add_argument(
        "--project-root", default=_cwd(),
        help="Project root (default: cwd)",
    )
    p_ti.add_argument(
        "--pytest-args", default="--tb=short -q",
        help="Extra pytest arguments (default: '--tb=short -q')",
    )
    p_ti.add_argument(
        "--timeout", type=int, default=120,
        help="Test timeout in seconds (default: 120)",
    )

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
    # session command  (durable session workspaces)
    # -----------------------------------------------------------------------
    session_parser = sub.add_parser(
        "session",
        help="Manage durable analysis sessions (create, list, show, close, delete, compare, prune)",
    )
    session_sub = session_parser.add_subparsers(dest="session_action", required=True)

    # session create
    p_sess_create = session_sub.add_parser("create", help="Create a new named analysis session")
    p_sess_create.add_argument("name", help="Session name")
    p_sess_create.add_argument("--description", "-d", default="", help="Session description")
    p_sess_create.add_argument("--repo", "-r", default="", help="Repository root path")
    p_sess_create.set_defaults(func=cmd_session_create)

    # session list
    p_sess_list = session_sub.add_parser("list", help="List all sessions")
    p_sess_list.add_argument("--active-only", action="store_true", help="Show only active sessions")
    p_sess_list.add_argument("--limit", type=int, default=50, help="Max sessions to show (default: 50)")
    p_sess_list.set_defaults(func=cmd_session_list)

    # session show
    p_sess_show = session_sub.add_parser("show", help="Show session details")
    p_sess_show.add_argument("name_or_id", help="Session name or ID")
    p_sess_show.add_argument("--snapshots", action="store_true", help="Include snapshot history")
    p_sess_show.set_defaults(func=cmd_session_show)

    # session close
    p_sess_close = session_sub.add_parser("close", help="Mark a session as inactive (soft-close)")
    p_sess_close.add_argument("name_or_id", help="Session name or ID")
    p_sess_close.set_defaults(func=cmd_session_close)

    # session delete
    p_sess_delete = session_sub.add_parser("delete", help="Permanently delete a session")
    p_sess_delete.add_argument("name_or_id", help="Session name or ID")
    p_sess_delete.set_defaults(func=cmd_session_delete)

    # session compare
    p_sess_compare = session_sub.add_parser("compare", help="Compare two sessions")
    p_sess_compare.add_argument("session_a", help="First session name or ID")
    p_sess_compare.add_argument("session_b", help="Second session name or ID")
    p_sess_compare.set_defaults(func=cmd_session_compare)

    # session prune
    p_sess_prune = session_sub.add_parser("prune", help="Delete oldest sessions beyond keep count")
    p_sess_prune.add_argument("--keep", type=int, default=30, help="Number of sessions to keep (default: 30)")
    p_sess_prune.set_defaults(func=cmd_session_prune)

    # -----------------------------------------------------------------------
    # loop command  (loop-engineering)
    # -----------------------------------------------------------------------
    _lazy_pt = _LazyPatternChoices()
    loop_parser = sub.add_parser("loop", help="Loop-engineering: scheduled automation patterns")
    loop_sub = loop_parser.add_subparsers(dest="loop_action", required=True)

    # loop init
    p_loop_init = loop_sub.add_parser("init", help="Scaffold loop config for the project")
    p_loop_init.add_argument("--project-root", default=_cwd())

    # loop run
    p_loop_run = loop_sub.add_parser("run", help="Run a specific loop pattern")
    p_loop_run.add_argument("pattern", choices=_lazy_pt, help="Pattern to run")
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
    p_loop_cost.add_argument("--pattern", choices=_lazy_pt, required=True, help="Pattern type")
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

    # -----------------------------------------------------------------------
    # uiux command  (UI/UX design intelligence — vendored ui-ux-pro-max engine)
    # -----------------------------------------------------------------------
    p_uiux = sub.add_parser(
        "uiux",
        help="UI/UX design intelligence: search 84 styles, 192 palettes, 74 font "
             "pairings, 25 chart types and 98 UX guidelines, or generate a complete "
             "design system (style, WCAG-tested colors, typography, motion, "
             "anti-patterns, pre-delivery checklist).",
    )
    p_uiux.add_argument(
        "query", nargs="?", default="",
        help="Product/industry/keywords to design for (e.g. 'saas analytics dashboard')",
    )
    p_uiux.add_argument(
        "--design-system", "-ds", action="store_true",
        help="Generate a complete design system recommendation (takes priority over domain/stack search)",
    )
    p_uiux.add_argument(
        "--project-name", "-p", default=None,
        help="Project name for design-system output",
    )
    p_uiux.add_argument(
        "--domain", "-d",
        help="Search a specific domain: style, color, chart, landing, product, ux, "
             "typography, google-fonts, icons, gsap, react, web (default: auto-detect)",
    )
    p_uiux.add_argument(
        "--stack", "-s",
        help="Stack-specific guidelines (react, nextjs, shadcn, html-tailwind, ...)",
    )
    p_uiux.add_argument(
        "--max-results", "-n", type=int, default=3,
        help="Max results for domain/stack search (default: 3)",
    )
    p_uiux.add_argument(
        "--json", action="store_true", help="Output as JSON",
    )
    p_uiux.add_argument(
        "--full", action="store_true",
        help="Do not truncate long field values in text output",
    )
    p_uiux.add_argument(
        "--format", "-f", choices=["ascii", "markdown"], default="ascii",
        help="Output format for design system (ignored if --json)",
    )
    p_uiux.add_argument(
        "--persist", action="store_true",
        help="Save design system to design-system/<project-slug>/MASTER.md",
    )
    p_uiux.add_argument(
        "--page", default=None,
        help="Also create a page-specific override in design-system/<project-slug>/pages/",
    )
    p_uiux.add_argument(
        "--output-dir", "-o", default=None,
        help="Directory the design-system/ folder is created under (default: cwd)",
    )
    p_uiux.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing MASTER.md when persisting (default: skip if it exists)",
    )
    p_uiux.add_argument(
        "--variance", type=int, choices=range(1, 11), metavar="1-10",
        help="DESIGN_VARIANCE dial: 1=centered/minimal, 10=bold/asymmetric (only with --design-system)",
    )
    p_uiux.add_argument(
        "--motion", type=int, choices=range(1, 11), metavar="1-10",
        help="MOTION_INTENSITY dial: 1=subtle, 10=complex; pulls a matching GSAP snippet (only with --design-system)",
    )
    p_uiux.add_argument(
        "--density", type=int, choices=range(1, 11), metavar="1-10",
        help="VISUAL_DENSITY dial: 1=spacious, 10=dense/dashboard (only with --design-system)",
    )
    p_uiux.add_argument(
        "--list-domains", action="store_true", help="List available search domains",
    )
    p_uiux.add_argument(
        "--list-stacks", action="store_true", help="List available frontend stacks",
    )
    p_uiux.add_argument(
        "--validate-data", action="store_true",
        help="Run the installed skill's design-database integrity check (data guardrail)",
    )
    p_uiux.add_argument(
        "--install", action="store_true",
        help="Install the upstream ui-ux-pro-max-skill via its official npm CLI "
             "(npm install -g ui-ux-pro-max-cli && uipro init --ai claude)",
    )
    p_uiux.set_defaults(func=cmd_uiux)

    return parser


def cmd_uiux(args: argparse.Namespace) -> int:
    """UI/UX design intelligence — search the design DB or generate a design system.

    Thin wrapper: graphsift does not bundle the ui-ux-pro-max engine; it locates
    the officially-installed skill on this machine and shells out to its
    search.py (see graphsift.uiux). Install the engine once with
    `graphsift uiux --install`.
    """
    from graphsift.uiux import DOMAINS, STACKS, find_search_script, install_engine, run_cli

    if args.install:
        code, msg = install_engine()
        print(msg)
        return code

    if args.validate_data:
        script = find_search_script()
        if script is None:
            from graphsift.uiux import install_hint
            print(install_hint(), file=sys.stderr)
            return 1
        validator = script.parent / "validate_data.py"
        if not validator.is_file():
            print("error: validate_data.py not found next to the installed search.py", file=sys.stderr)
            return 1
        import subprocess as _sp
        proc = _sp.run([sys.executable, str(validator)])
        return proc.returncode if proc.returncode is not None else 1

    if args.list_domains:
        for d in DOMAINS:
            print(d)
        return 0

    if args.list_stacks:
        for s in STACKS:
            print(s)
        return 0

    if not args.query:
        print("error: uiux requires a query (e.g. 'graphsift uiux \"saas analytics dashboard\"')", file=sys.stderr)
        return 2

    # Map argparse Namespace -> upstream search.py argv.
    argv: list[str] = []
    if args.design_system:
        argv.append("--design-system")
    if args.project_name:
        argv += ["--project-name", args.project_name]
    if args.domain:
        argv += ["--domain", args.domain]
    if args.stack:
        argv += ["--stack", args.stack]
    if args.max_results:
        argv += ["--max-results", str(args.max_results)]
    if args.json:
        argv.append("--json")
    if args.full:
        argv.append("--full")
    if args.format:
        argv += ["--format", args.format]
    if args.persist:
        argv.append("--persist")
    if args.page:
        argv += ["--page", args.page]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.force:
        argv.append("--force")
    if args.variance:
        argv += ["--variance", str(args.variance)]
    if args.motion:
        argv += ["--motion", str(args.motion)]
    if args.density:
        argv += ["--density", str(args.density)]

    return run_cli(argv, args.query)


def main() -> None:
    # Ensure stdout/stdin handle arbitrary UTF-8 on all platforms. Windows
    # defaults to the locale codec (e.g. cp1252), which cannot encode
    # non-ASCII and crashes compress/output commands on those characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    gc.freeze()  # freeze pre-import objects — GC never rescans them
    # Self-heal: drop stale user-global graphsift skills/commands that
    # duplicate the project-scoped ones in the Claude Code slash menu.
    try:
        if _cleanup_legacy_global_skills():
            print("[graphsift] Removed stale user-global graphsift skills "
                  "(duplicate slash commands). Restart Claude Code to refresh the / menu.")
    except Exception:
        pass
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
                    source_map[str(py_file.relative_to(src_dir))] = SafeFileIO.read(py_file)
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
        "prune-refs": cmd_prune_refs,
        "terse": cmd_terse,
        "fix": cmd_fix,
        "add": cmd_add,
        "refactor": cmd_refactor,
        "verify": cmd_verify,
        "test-impact": cmd_test_impact,
        "tool-budgets": cmd_tool_budgets,
        "read-cache": cmd_read_cache,
        "evidence": cmd_evidence,
        "guard": cmd_guard,
        "uiux": cmd_uiux,
        "claude-md": cmd_claude_md,
        "guide": cmd_guide,
    }

    # Support func-based dispatch for new-style subcommands
    fn = getattr(args, "func", None) or commands.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(fn(args))


if __name__ == "__main__":
    main()
