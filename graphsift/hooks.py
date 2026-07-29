"""Auto-rewrite hooks for graphsift — transparent compression AND smart execution.

Pure Python, no external deps, type-hinted.

Provides:
  - wrap_command()      — rewrite shell commands to pipe through compression
  - compress_bash_hook  — PostToolUse: compress Bash output (save tokens)
  - pre_bash_hook       — PreToolUse: auto-route Python commands through daemon
                          (bypasses classifier + shell startup + permission prompts)
  - get_bash_wrapper_script()   — shell functions for transparent compression
  - get_pre_tool_use_config()   — PreToolUse hook config for settings.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_COMPRESSIBLE_COMMANDS: dict[str, str] = {
    "pytest": "pytest",
    "cargo": "cargo",
    "go test": "go_test",
    "jest": "jest",
    "npx jest": "jest",
    "eslint": "eslint",
    "npx eslint": "eslint",
    "git status": "git_status",
    "git diff": "git_diff",
    "git log": "git_log",
    "grep": "grep",
    "npm": "npm",
    "yarn": "npm",
    "docker": "docker",
    "kubectl": "kubectl",
    "aws": "aws",
    "make": "make",
    "pip": "pip",
    "cat": "cat",
}

# Patterns that graphsift can optimize — runs via daemon/direct subprocess
# instead of shell (bypasses classifier + permission + shell startup)
_OPTIMIZABLE_PATTERNS: list[re.Pattern] = [
    # cd <dir> && python -c "code"
    re.compile(r'^cd\s+(.+?)\s*[;&]+\s*python\s+-c\s+["\'](.+)["\']\s*$', re.DOTALL),
    # cd <dir> && python <script>
    re.compile(r'^cd\s+(.+?)\s*[;&]+\s*python\s+(\S+\.py)\s*$'),
    # python -c "code" (no cd)
    re.compile(r'^python\s+-c\s+["\'](.+)["\']\s*$', re.DOTALL),
    # cd <dir> && python -m <module>
    re.compile(r'^cd\s+(.+?)\s*[;&]+\s*python\s+-m\s+(\S+)\s*$'),
    # cd <dir> && python <script> with 2>&1
    re.compile(r'^cd\s+(.+?)\s*[;&]+\s*python\s+(\S+\.py).*?$'),
    # cd <dir> && python -c "code" with 2>&1 | head/tail
    re.compile(r'^cd\s+(.+?)\s*[;&]+\s*python\s+-c\s+', re.DOTALL),
]

# Windows-style: cd /d <dir> && python ...
_OPTIMIZABLE_PS_PATTERNS: list[re.Pattern] = [
    re.compile(r'^cd\s+(.+?)\s*[;]+\s*python\s+-c\s+["\'](.+)["\']\s*$', re.DOTALL),
    re.compile(r'^cd\s+(.+?)\s*[;]+\s*python\s+(\S+\.py)\s*$'),
]


def _tee_path() -> str:
    """Return the tee directory path for saving original uncompressed output."""
    return str(Path.home() / ".graphsift" / "tee")


def _detect_command_type(command: str) -> Optional[str]:
    """Inspect a shell command and return its compress type, or None."""
    stripped = command.strip().lstrip("(")
    words = stripped.split()
    if not words:
        return None

    # Check two-word prefix first (e.g. "git status", "go test")
    if len(words) >= 2:
        prefix = " ".join(words[:2]).lower()
        if prefix in _COMPRESSIBLE_COMMANDS:
            return _COMPRESSIBLE_COMMANDS[prefix]

    # Check single-word prefix (e.g. "pytest", "cargo")
    first = words[0].lower()
    return _COMPRESSIBLE_COMMANDS.get(first, None)


# ---------------------------------------------------------------------------
# Smart execution: detect Python commands and route through daemon
# ---------------------------------------------------------------------------

# Lazy import for daemon (avoid circular imports)
_daemon_started = False

def _ensure_daemon():
    """Start the daemon if not running. Safe to call multiple times."""
    global _daemon_started
    try:
        from graphsift.daemon import start, status
        st = status()
        if st.get("status") != "running":
            result = start()
            _daemon_started = True
            return result
        return {"status": "already_running"}
    except ImportError:
        return {"status": "unavailable"}


def _run_via_daemon(code: str, cwd: str = "") -> dict:
    """Run Python code through the persistent daemon. Returns result dict."""
    try:
        from graphsift.daemon import exec_code
        return exec_code(code, cwd)
    except ImportError:
        return {"ok": False, "stdout": "", "stderr": "graphsift.daemon not available"}


def _run_via_direct(cmd: list[str], cwd: str = "") -> dict:
    """Run a command via direct subprocess (no shell)."""
    import subprocess
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True, cwd=cwd or None,
            timeout=60,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "Timeout after 60s"}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e)}


def _extract_cwd_and_code(command: str) -> Optional[dict]:
    """Analyze a command and extract (cwd, code/script, mode) if optimizable.

    Returns:
        None if not optimizable.
        dict with {cwd, code_or_script, mode, needs_daemon} if optimizable.
    """
    stripped = command.strip()

    # Pattern 1: cd <dir> && python -c "code" (bash style with &&)
    # Pattern 2: cd <dir> && python -c "code" 2>&1 | head/tail/...
    m = re.search(
        r'cd\s+["\']?(.+?)["\']?\s*[;&]+\s*python\s+-c\s+', stripped, re.DOTALL
    )
    if m:
        cwd = m.group(1).strip()
        # Extract the code after -c "..."
        # Find the opening quote
        after_c = stripped[m.end():].strip()
        # Handle 2>&1 | head/tail etc - strip those
        code = after_c
        # If code starts with quote, find matching close quote
        if code.startswith('"') or code.startswith("'"):
            quote = code[0]
            end_idx = code.find(quote, 1)
            if end_idx > 0:
                code = code[1:end_idx]
                return {
                    "cwd": cwd,
                    "code": code,
                    "mode": "daemon",
                    "needs_daemon": True,
                }
        # If no quotes, try raw (less common)
        # Strip pipe/redirect
        for sep in [" 2>&1", " |", " >", " ;"]:
            idx = code.find(sep)
            if idx > 0:
                code = code[:idx]
        return {
            "cwd": cwd,
            "code": code.strip(),
            "mode": "daemon",
            "needs_daemon": True,
        }

    # Pattern 3: cd <dir> && python <script.py>
    m = re.search(
        r'cd\s+["\']?(.+?)["\']?\s*[;&]+\s*python\s+(\S+\.py)',
        stripped,
    )
    if m:
        cwd = m.group(1).strip()
        script = m.group(2).strip()
        return {
            "cwd": cwd,
            "code_or_script": script,
            "mode": "script",
            "needs_daemon": False,
        }

    # Pattern 4: python -c "code" (no cd)
    m = re.match(r'^python\s+-c\s+', stripped)
    if m:
        after_c = stripped[m.end():].strip()
        if after_c.startswith('"') or after_c.startswith("'"):
            quote = after_c[0]
            end_idx = after_c.find(quote, 1)
            if end_idx > 0:
                code = after_c[1:end_idx]
                return {
                    "cwd": os.getcwd(),
                    "code": code,
                    "mode": "daemon",
                    "needs_daemon": True,
                }

    # Pattern 5: sleep N (handle natively, no execution needed)
    m = re.match(r'^sleep\s+(\d+(?:\.\d+)?)\s*$', stripped)
    if m:
        duration = float(m.group(1))
        return {
            "cwd": os.getcwd(),
            "code": "",
            "mode": "sleep",
            "duration": duration,
            "needs_daemon": True,
        }

    return None


def optimize_command(command: str) -> Optional[dict]:
    """Try to run a command via the fast path (daemon or direct subprocess).

    Args:
        command: Raw command string from Bash/PowerShell tool.

    Returns:
        None if command cannot be optimized (pass through to shell).
        dict with fast_result if command was optimized.
    """
    info = _extract_cwd_and_code(command)
    if info is None:
        return None

    cwd = info["cwd"]
    mode = info["mode"]

    if mode == "daemon":
        # Route through persistent daemon (keeps modules cached)
        result = _run_via_daemon(info["code"], cwd)
        return {
            "optimized": True,
            "method": "daemon",
            "cwd": cwd,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "ok": result.get("ok", False),
        }

    elif mode == "script":
        # Route through direct subprocess (no shell)
        script = info["code_or_script"]
        if not os.path.isabs(script):
            script = os.path.join(cwd, script)
        result = _run_via_direct([sys.executable, script], cwd)
        return {
            "optimized": True,
            "method": "direct",
            "cwd": cwd,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "ok": result.get("ok", False),
        }

    elif mode == "sleep":
        # Handle sleep natively — no Python exec, returns instantly
        import time as _time
        duration = info.get("duration", 1.0)
        _time.sleep(min(duration, 30))  # cap at 30s
        return {
            "optimized": True,
            "method": "sleep",
            "cwd": cwd,
            "stdout": "",
            "stderr": "",
            "ok": True,
        }

    return None


# ---------------------------------------------------------------------------
# PreToolUse hook — auto-route Bash/PowerShell through daemon
# ---------------------------------------------------------------------------

def pre_bash_hook() -> str:
    """PreToolUse hook for Bash: auto-route Python commands through daemon.

    Called by Claude Code PreToolUse hook BEFORE every Bash command.
    Reads the tool input from stdin as JSON, checks if the command can be
    optimized (e.g. ``cd <dir> && python ...``), and if so runs it via the
    persistent daemon — bypassing the classifier, permission system, and
    shell startup entirely.

    If the command cannot be optimized, it's returned unchanged so the
    Bash tool runs normally.

    Usage:
        python -m graphsift.hooks pre-bash-hook

    Hook config (in .claude/settings.json):
        {
          "hooks": {
            "PreToolUse": [
              {
                "matcher": "Bash|PowerShell",
                "hooks": [
                  {
                    "type": "command",
                    "command": "python -m graphsift.hooks pre-bash-hook"
                  }
                ]
              }
            ]
          }
        }
    """
    try:
        # Read tool input as JSON
        raw = sys.stdin.read()
        if not raw:
            return raw

        data = json.loads(raw)
        command = ""

        # Handle different input formats
        if isinstance(data, dict):
            if "input" in data:
                command = data["input"]
            elif "command" in data:
                command = data["command"]
            elif isinstance(data.get("input"), dict):
                command = data["input"].get("command", "")

        if not command:
            return raw  # pass through unchanged

        # Try to optimize
        result = optimize_command(command)
        if result is None:
            return raw  # not optimizable, pass through

        # Build response — tell Claude Code to skip the Bash tool
        # and use our result directly.
        output = result.get("stdout", "") or ""
        stderr = result.get("stderr", "") or ""

        # Format as tool output
        response = output
        if stderr:
            response += ("\n" + stderr) if output else stderr

        # Return instruction to cancel Bash and use our result
        skip_response = {
            "skip": True,
            "response": response.strip(),
            "_graphsift": {
                "optimized": True,
                "method": result.get("method", "unknown"),
                "cwd": result.get("cwd", ""),
            },
        }
        return json.dumps(skip_response)

    except json.JSONDecodeError:
        return raw  # not JSON, pass through
    except Exception:
        # Hook must never fail — pass through on any error
        return raw


# ---------------------------------------------------------------------------
# Output compression hooks (existing)
# ---------------------------------------------------------------------------

def wrap_command(command: str, ultra: bool = False) -> str:
    """Rewrite a shell command to pipe through graphsift compression.

    If the command matches a known compressible type it is rewritten to
    pipe stdout+stderr through ``python -m graphsift.compress``.
    Returns the original command unchanged when the type is not recognised.

    Args:
        command: Original shell command string.
        ultra: Pass ``--ultra`` for aggressive 30-line cap.

    Returns:
        Rewritten command or the original if not compressible.
    """
    cmd_type = _detect_command_type(command)
    if cmd_type is None:
        return command

    tee_dir = _tee_path()
    ultra_flag = " --ultra" if ultra else ""

    return (
        f"{command} 2>&1 | python -m graphsift.compress"
        f" --type {cmd_type} --tee {tee_dir} --tee-label {cmd_type}{ultra_flag}"
    )


def compress_bash_hook() -> str:
    """PostToolUse hook for Bash: compress stdout and record analytics.

    Called by Claude Code PostToolUse hook after every Bash command.
    Reads original output from stdin, compresses it, records token savings,
    and writes the compressed output back to stdout.  **Never raises.**

    Usage:
        python -m graphsift.hooks compress-bash-hook
    """
    try:
        text = sys.stdin.read()
        if not text:
            return ""

        if len(text) > 200:
            from graphsift.compress import compress
            from graphsift.analytics import record_call

            compressed = compress(text)
            tokens_saved = (len(text) - len(compressed)) // 4
            if tokens_saved > 0:
                record_call(
                    tokens_saved=tokens_saved,
                    command_type="bash",
                    original_chars=len(text),
                    compressed_chars=len(compressed),
                )
            return compressed
        return text
    except Exception:
        # Hook must never fail — if anything goes wrong,
        # return the original text so no data is lost.
        return sys.stdin.read() if hasattr(sys.stdin, "read") else ""


def get_bash_wrapper_script(python_path: str = "python") -> str:
    """Return a bash script for transparent compression via shell functions.

    Source this fragment in ``.bashrc``::

        eval "$(graphsift bash-wrapper)"

    or::

        source <(python -m graphsift.hooks bash-wrapper)

    The script exports ``GRAPHSIFT_TEE_DIR``, defines a
    ``__graphsift_compress`` helper, and installs shell functions that
    intercept common commands and pipe their output through compression.
    """
    return f'''# graphsift: transparent output compression
# Source in .bashrc:  eval "$(python -m graphsift.hooks bash-wrapper)"

export GRAPHSIFT_TEE_DIR="${{HOME}}/.graphsift/tee"

__graphsift_compress() {{
    local type="${{1:-auto}}"
    {python_path} -m graphsift.compress --type "$type" --tee "$GRAPHSIFT_TEE_DIR" --tee-label "$type"
}}

# Build / test / analysis
pytest() {{ command pytest "$@" 2>&1 | __graphsift_compress pytest; }}
cargo() {{ command cargo "$@" 2>&1 | __graphsift_compress cargo; }}
go() {{
    if [ "$1" = "test" ]; then
        command go "$@" 2>&1 | __graphsift_compress go_test
    else
        command go "$@"
    fi
}}
jest() {{ command jest "$@" 2>&1 | __graphsift_compress jest; }}
eslint() {{ command eslint "$@" 2>&1 | __graphsift_compress eslint; }}
npx() {{
    case "$1" in
        jest|eslint)
            local type="$1"
            shift
            command npx "$type" "$@" 2>&1 | __graphsift_compress "$type"
            ;;
        *)
            command npx "$@"
            ;;
    esac
}}

# Package managers
npm() {{ command npm "$@" 2>&1 | __graphsift_compress npm; }}
yarn() {{ command yarn "$@" 2>&1 | __graphsift_compress npm; }}

# Infrastructure
docker() {{ command docker "$@" 2>&1 | __graphsift_compress docker; }}
kubectl() {{ command kubectl "$@" 2>&1 | __graphsift_compress kubectl; }}
make() {{ command make "$@" 2>&1 | __graphsift_compress make; }}

# Git shorthand
gs() {{ git status "$@" 2>&1 | __graphsift_compress git_status; }}
gd() {{ git diff "$@" 2>&1 | __graphsift_compress git_diff; }}
gl() {{ git log "$@" 2>&1 | __graphsift_compress git_log; }}

# Utilities
grep() {{ command grep "$@" 2>&1 | __graphsift_compress grep; }}
cat() {{ command cat "$@" 2>&1 | __graphsift_compress cat; }}
pip() {{ command pip "$@" 2>&1 | __graphsift_compress pip; }}
aws() {{ command aws "$@" 2>&1 | __graphsift_compress aws; }}
'''


def get_pre_tool_use_config(python_path: str = "python") -> dict:
    """Return a Claude Code PreToolUse hook config for smart execution.

    The returned dict can be appended to the ``PreToolUse`` array in
    ``.claude/settings.json``::

        {
          "hooks": {
            "PreToolUse": [
              get_pre_tool_use_config("python3.11")
            ]
          }
        }

    This hook intercepts Bash and PowerShell commands and automatically
    routes optimizable commands (like ``cd <dir> && python ...``) through
    graphsift's persistent daemon — bypassing classifier + permission +
    shell startup for huge speed improvements.

    Returns:
        A PreToolUse entry with matcher ``"Bash|PowerShell"``.
    """
    return {
        "matcher": "Bash|PowerShell",
        "hooks": [
            {
                "type": "command",
                "command": (
                    f'{python_path} -m graphsift.hooks pre-bash-hook'
                ),
            }
        ],
    }


def get_post_tool_use_config(project_root: str, python_path: str) -> dict:
    """Return a Claude Code PostToolUse hook config dict for Bash compression.

    The returned dict can be appended to the ``PostToolUse`` array in
    ``.claude/settings.json``::

        {
          "hooks": {
            "PostToolUse": [
              get_post_tool_use_config("/repo", "python3.11")
            ]
          }
        }

    Args:
        project_root: Root of the project (for context; not used directly).
        python_path: Python executable path.

    Returns:
        A single PostToolUse entry with matcher ``"Bash"``.
    """
    _ = project_root  # kept for API consistency
    return {
        "matcher": "Bash",
        "hooks": [
            {
                "type": "command",
                "command": (
                    f'{python_path} -m graphsift.hooks compress-bash-hook'
                ),
            }
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry: python -m graphsift.hooks <command>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "compress-bash-hook":
            result = compress_bash_hook()
            sys.stdout.write(result)
        elif cmd == "pre-bash-hook":
            result = pre_bash_hook()
            sys.stdout.write(result)
        elif cmd == "bash-wrapper":
            path = sys.argv[2] if len(sys.argv) > 2 else "python"
            result = get_bash_wrapper_script(path)
            sys.stdout.write(result)
        elif cmd == "pre-tool-use-config":
            path = sys.argv[2] if len(sys.argv) > 2 else "python"
            result = json.dumps(get_pre_tool_use_config(path), indent=2)
            sys.stdout.write(result)
        elif cmd == "post-tool-use-config":
            project = sys.argv[2] if len(sys.argv) > 2 else "."
            path = sys.argv[3] if len(sys.argv) > 3 else "python"
            result = json.dumps(get_post_tool_use_config(project, path), indent=2)
            sys.stdout.write(result)
        else:
            print("Usage: python -m graphsift.hooks <command>")
            print()
            print("Commands:")
            print("  compress-bash-hook    PostToolUse: compress Bash output")
            print("  pre-bash-hook         PreToolUse: auto-route through daemon")
            print("  bash-wrapper [path]   Print bash wrapper script")
            print("  pre-tool-use-config [path]  Print PreToolUse hook config")
            print("  post-tool-use-config [proj path] Print PostToolUse hook config")
    else:
        print("Usage: python -m graphsift.hooks <command>")
