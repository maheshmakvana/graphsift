"""graphsift.uiux — UI/UX design intelligence (thin wrapper, no vendored code).

graphsift does NOT bundle the ui-ux-pro-max-skill engine or its design
database. Instead, `graphsift uiux` locates the officially-installed skill on
this machine and shells out to its `search.py` (BM25 search over 84 styles,
192 palettes, 74 font pairings, 25 chart types, 98 UX guidelines and 22
stacks, plus full design-system generation).

The upstream skill is MIT-licensed (© 2024 Next Level Builder). Install it
once, then every graphsift command delegates to it:

    npm install -g ui-ux-pro-max-cli
    uipro init --ai claude

or via the Claude Code plugin marketplace:

    /plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill
    /plugin install ui-ux-pro-max@ui-ux-pro-max-skill

If the skill lives somewhere unusual, point GRAPHSIFT_UIUX_SKILL at its
`search.py` (or at the directory containing it) to skip discovery.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Static, factual API names of the upstream engine's search domains and stacks.
# These are plain names used to drive the CLI (--list-*, help text), not
# copyrighted code or data.
# ---------------------------------------------------------------------------
DOMAINS = [
    "style", "color", "chart", "landing", "product", "ux",
    "typography", "google-fonts", "icons", "gsap", "react", "web",
]
STACKS = [
    "react", "nextjs", "vue", "svelte", "astro", "swiftui", "react-native",
    "flutter", "nuxtjs", "nuxt-ui", "html-tailwind", "shadcn", "jetpack-compose",
    "threejs", "angular", "laravel", "javafx", "wpf", "winui", "avalonia",
    "uno", "uwp",
]

_SCRIPT_REL = Path(".claude/skills/ui-ux-pro-max/scripts/search.py")


def install_hint() -> str:
    """Human-readable instructions for installing the upstream skill once."""
    return (
        "The ui-ux-pro-max-skill design engine is not installed on this machine.\n"
        "Install it once (MIT-licensed, © 2024 Next Level Builder):\n"
        "  npm install -g ui-ux-pro-max-cli\n"
        "  uipro init --ai claude\n"
        "or from the Claude Code plugin marketplace:\n"
        "  /plugin marketplace add nextlevelbuilder/ui-ux-pro-max-skill\n"
        "  /plugin install ui-ux-pro-max@ui-ux-pro-max-skill\n"
        "Then re-run this command. If it is installed somewhere unusual, set\n"
        "GRAPHSIFT_UIUX_SKILL to its search.py path (or the directory holding it)."
    )


def _candidate_roots() -> list[Path]:
    """Directories that may contain (or nest) an installed ui-ux-pro-max skill."""
    cwd = Path.cwd()
    home = Path.home()
    roots: list[Path] = [cwd, cwd / ".claude", home]
    for sub in (".claude/plugins", ".claude/skills", ".agents/skills",
                ".config/agents/skills", ".claude/plugins/marketplaces"):
        roots.append(home / sub)
    return roots


def find_search_script() -> Path | None:
    """Locate the upstream search.py in known install locations.

    Search order: GRAPHSIFT_UIUX_SKILL override, then each candidate root at a
    few sensible nestings (direct, `.claude/skills/...`, and one-to-two levels
    of marketplace/plugin nesting). Returns the first hit or None.
    """
    env = os.environ.get("GRAPHSIFT_UIUX_SKILL")
    if env:
        p = Path(env).expanduser()
        if p.name == "search.py" and p.is_file():
            return p
        if (p / "search.py").is_file():
            return p / "search.py"

    for root in _candidate_roots():
        if not root.is_dir():
            continue
        candidates = [
            root / _SCRIPT_REL,
            root / "ui-ux-pro-max" / "scripts" / "search.py",
        ]
        # Marketplace/plugin nesting: <root>/<owner>/<skill>/.claude/skills/...
        candidates += sorted(root.glob("*/" + str(_SCRIPT_REL)))
        candidates += sorted(root.glob("*/*/" + str(_SCRIPT_REL)))
        for cand in candidates:
            try:
                if cand.is_file():
                    return cand
            except OSError:
                continue
    return None


def ensure_engine() -> tuple[Path | None, str | None]:
    """Return (search_script, None) or (None, error message)."""
    script = find_search_script()
    if script is None:
        return None, install_hint()
    return script, None


def _cmd(script: Path, argv: list[str]) -> list[str]:
    return [sys.executable, "-u", str(script), *argv]


def run_cli(argv: list[str], query: str) -> int:
    """Run the engine directly, streaming its output to the caller's stdout.

    Returns the engine's exit code (1 if the engine is not installed).
    """
    script, err = ensure_engine()
    if script is None:
        sys.stderr.write(err + "\n")
        return 1
    env = dict(os.environ)
    cmd = _cmd(script, [query, *argv])
    try:
        proc = subprocess.run(cmd, env=env)
    except OSError as exc:
        sys.stderr.write(f"error: could not run the ui-ux-pro-max engine: {exc}\n")
        return 1
    return proc.returncode if proc.returncode is not None else 1


def run_json(argv: list[str]) -> dict:
    """Run the engine and return its JSON output as a dict.

    Used by the MCP tools. Returns {"error": ...} (plus install hint) when the
    engine is missing or does not emit JSON.
    """
    script, err = ensure_engine()
    if script is None:
        return {"error": err}
    env = dict(os.environ)
    try:
        proc = subprocess.run(
            _cmd(script, argv), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=env,
        )
    except OSError as exc:
        return {"error": f"could not run the ui-ux-pro-max engine: {exc}"}
    if proc.returncode != 0:
        return {
            "error": proc.stderr.strip() or f"engine exited {proc.returncode}",
            "stdout": proc.stdout[:4000],
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": "engine did not return JSON", "stdout": proc.stdout[:4000]}


def install_engine() -> tuple[int, str]:
    """Try to install the upstream skill via its official npm CLI.

    Non-interactive: `npm install -g ui-ux-pro-max-cli` then `uipro init --ai claude`.
    Returns (exit_code, message).
    """
    npm = shutil.which("npm")
    if npm is None:
        return 1, "npm not found on PATH. Install Node.js first, then:\n" + install_hint()
    install_cmd = [npm, "install", "-g", "ui-ux-pro-max-cli"]
    print("$ " + " ".join(install_cmd))
    if subprocess.call(install_cmd) != 0:
        return 1, "npm install ui-ux-pro-max-cli failed."
    uipro = shutil.which("uipro") or shutil.which("ui-ux-pro-max-cli")
    if uipro is None:
        return 1, "Installed ui-ux-pro-max-cli, but 'uipro' is not on PATH yet."
    init_cmd = [uipro, "init", "--ai", "claude"]
    print("$ " + " ".join(init_cmd))
    if subprocess.call(init_cmd) != 0:
        return 1, "uipro init --ai claude failed. Run it manually."
    return 0, "ui-ux-pro-max-skill installed. Run `graphsift uiux ...` again."


__all__ = [
    "DOMAINS",
    "STACKS",
    "ensure_engine",
    "find_search_script",
    "install_engine",
    "install_hint",
    "run_cli",
    "run_json",
]
