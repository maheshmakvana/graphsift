"""Auto-rewrite hooks for graphsift -- mirrors rtk's transparent compression.

Pure Python, no external deps, type-hinted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
                    f'{python_path} -c "from graphsift.analytics import '
                    f"record_call; record_call(tokens_saved=30000, "
                    f"command_type='bash')\""
                ),
            }
        ],
    }
