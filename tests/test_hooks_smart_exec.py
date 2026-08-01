"""Tests for the PreToolUse smart-execution hook (graphsift.hooks).

Covers:
  - ``pre_bash_hook`` payload parsing for the documented Claude Code
    ``tool_input`` shape (and legacy ``input`` shapes).
  - ``_extract_cwd_and_code`` command-optimization detection.
  - ``optimize_command`` daemon / direct / sleep fast paths.

These are regression tests for a bug where the hook read ``data["input"]``
as a bare string, so the real ``{"command": ...}`` value was never extracted
and smart execution silently never fired (every Bash command paid Python
startup cost for zero benefit).
"""

from __future__ import annotations

import io
import json
import sys

import pytest

from graphsift.hooks import (
    _extract_cwd_and_code,
    optimize_command,
    pre_bash_hook,
)


def _run_hook(payload: dict) -> str:
    """Invoke pre_bash_hook with *payload* as stdin, returning its stdout."""
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        return pre_bash_hook()
    finally:
        sys.stdin = old_stdin


def _parsed_hook_output(out: str) -> dict:
    """Parse hook stdout; returns {} if it wasn't a JSON hook response."""
    try:
        value = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_optimized(out: str) -> bool:
    parsed = _parsed_hook_output(out)
    gs = parsed.get("_graphsift", {}) or {}
    return bool(gs.get("optimized") or gs.get("rewritten"))


# ---------------------------------------------------------------------------
# Payload parsing — the core regression
# ---------------------------------------------------------------------------


class TestPreBashHookPayloadParsing:
    """pre_bash_hook must extract the command from tool_input / input."""

    def test_tool_input_dict(self):
        """Documented Claude Code shape: tool_input = {"command": ...}."""
        out = _run_hook({
            "tool_name": "Bash",
            "tool_input": {
                "command": 'python -c "print(1+1)"',
                "description": "run python",
            },
        })
        assert _is_optimized(out), out

    def test_input_dict(self):
        """Legacy/other hosts: input = {"command": ...}."""
        out = _run_hook({
            "tool_name": "Bash",
            "input": {"command": 'python -c "print(1+1)"'},
        })
        assert _is_optimized(out), out

    def test_input_string(self):
        """Legacy: input is a bare command string."""
        out = _run_hook({
            "tool_name": "Bash",
            "input": 'python -c "print(1+1)"',
        })
        assert _is_optimized(out), out

    def test_tool_input_with_extra_fields(self):
        """Full documented payload with session fields."""
        out = _run_hook({
            "session_id": "abc123",
            "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {
                "command": 'cd C:/repo && python -c "print(42)"',
                "description": "run",
                "timeout": 120000,
            },
            "tool_use_id": "toolu_01ABC123",
        })
        assert _is_optimized(out), out

    def test_power_shell_tool(self):
        """PowerShell tool uses the same tool_input shape."""
        out = _run_hook({
            "tool_name": "PowerShell",
            "tool_input": {"command": 'python -c "print(1)"'},
        })
        assert _is_optimized(out), out

    def test_non_optimizable_command_passes_through(self):
        """Commands that can't be optimized must pass through unchanged."""
        payload = {
            "tool_name": "Bash",
            "tool_input": {"command": "git status"},
        }
        out = _run_hook(payload)
        assert not _is_optimized(out)
        # Pass-through returns the raw input (or nothing) — never a skip.
        assert out == json.dumps(payload) or out.strip() == ""

    def test_missing_command_passes_through(self):
        """No command at all → pass through without error."""
        out = _run_hook({"tool_name": "Bash", "tool_input": {}})
        assert not _is_optimized(out)

    def test_invalid_json_passes_through(self):
        """Non-JSON stdin must never crash the hook."""
        old_stdin = sys.stdin
        sys.stdin = io.StringIO("not json at all")
        try:
            out = pre_bash_hook()
        finally:
            sys.stdin = old_stdin
        assert out == "not json at all"


# ---------------------------------------------------------------------------
# Response format — modern Claude Code hook protocol
# ---------------------------------------------------------------------------


class TestPreBashHookResponseFormat:
    """Optimized results must rewrite the command via the modern
    ``updatedInput`` channel.

    The legacy ``{"skip": true, "response": ...}`` shape is not honored by
    current Claude Code, and the ``deny`` decision frames output as a
    failure (the model works around it). Rewriting the command with
    ``permissionDecision: "allow"`` runs the fast launcher instead, so the
    model sees a normal successful tool result.
    """

    def _run_with_rewrite(self, monkeypatch, rewritten: str):
        import graphsift.hooks as hooks

        monkeypatch.setattr(hooks, "_rewrite_command", lambda cmd, shell="bash": rewritten)
        return _run_hook({
            "tool_name": "Bash",
            "tool_input": {"command": 'python -c "print(40+2)"'},
        })

    def test_optimized_uses_hook_specific_output(self, monkeypatch):
        out = self._run_with_rewrite(monkeypatch, "LAUNCHER --codefile /tmp/c.py")
        parsed = _parsed_hook_output(out)
        hso = parsed.get("hookSpecificOutput")
        assert isinstance(hso, dict), out
        assert hso.get("hookEventName") == "PreToolUse"
        assert hso.get("permissionDecision") == "allow"

    def test_command_rewritten_via_updated_input(self, monkeypatch):
        out = self._run_with_rewrite(monkeypatch, "LAUNCHER --codefile /tmp/c.py")
        parsed = _parsed_hook_output(out)
        updated = parsed["hookSpecificOutput"].get("updatedInput", {})
        assert isinstance(updated, dict)
        assert updated.get("command") == "LAUNCHER --codefile /tmp/c.py"

    def test_rewrite_preserves_other_tool_input_fields(self, monkeypatch):
        import graphsift.hooks as hooks

        monkeypatch.setattr(hooks, "_rewrite_command", lambda cmd, shell="bash": "R")
        out = _run_hook({
            "tool_name": "Bash",
            "tool_input": {
                "command": "python -c \"print(1)\"",
                "description": "run python",
                "timeout": 120000,
            },
        })
        parsed = _parsed_hook_output(out)
        updated = parsed["hookSpecificOutput"].get("updatedInput", {})
        assert updated.get("description") == "run python"
        assert updated.get("timeout") == 120000
        assert updated.get("command") == "R"

    def test_no_deny_or_reason_fields(self, monkeypatch):
        out = self._run_with_rewrite(monkeypatch, "R")
        parsed = _parsed_hook_output(out)
        hso = parsed.get("hookSpecificOutput", {})
        assert "permissionDecisionReason" not in hso
        assert "denyReason" not in hso

    def test_legacy_skip_response_not_emitted(self, monkeypatch):
        out = self._run_with_rewrite(monkeypatch, "R")
        parsed = _parsed_hook_output(out)
        assert "skip" not in parsed
        assert "response" not in parsed


# ---------------------------------------------------------------------------
# Command rewriting — _rewrite_command
# ---------------------------------------------------------------------------


class TestRewriteCommand:
    """_rewrite_command turns optimizable commands into launcher commands."""

    def test_python_c_rewritten(self, monkeypatch):
        import graphsift.hooks as hooks
        import graphsift.launcher as launcher
        import graphsift.daemon as daemon_mod

        monkeypatch.setattr(daemon_mod, "status", lambda: {"status": "running"})
        monkeypatch.setattr(launcher, "build_launcher_command",
                            lambda **kw: "LAUNCHER " + str(kw))
        rewritten = hooks._rewrite_command('cd C:/repo && python -c "print(1)"', "bash")
        assert rewritten is not None
        assert rewritten.startswith("LAUNCHER")
        assert "codefile" in rewritten

    def test_script_rewritten(self, monkeypatch):
        import graphsift.hooks as hooks
        import graphsift.launcher as launcher

        monkeypatch.setattr(launcher, "build_launcher_command",
                            lambda **kw: "LAUNCHER " + str(kw))
        rewritten = hooks._rewrite_command("cd C:/repo && python run.py", "bash")
        assert rewritten is not None
        assert "script" in rewritten

    def test_sleep_rewritten(self, monkeypatch):
        import graphsift.hooks as hooks
        import graphsift.launcher as launcher

        monkeypatch.setattr(launcher, "build_launcher_command",
                            lambda **kw: "LAUNCHER " + str(kw))
        rewritten = hooks._rewrite_command("sleep 2", "bash")
        assert rewritten is not None
        assert "sleep" in rewritten

    def test_not_optimizable_returns_none(self, monkeypatch):
        import graphsift.hooks as hooks

        assert hooks._rewrite_command("git status", "bash") is None
        assert hooks._rewrite_command("npm test", "bash") is None

    def test_powershell_shell_passed(self, monkeypatch):
        import graphsift.hooks as hooks
        import graphsift.launcher as launcher

        captured = {}

        def fake(**kw):
            captured.update(kw)
            return "LAUNCHER"

        monkeypatch.setattr(launcher, "build_launcher_command", fake)
        hooks._rewrite_command("sleep 1", "powershell")
        assert captured.get("shell") == "powershell"


# ---------------------------------------------------------------------------
# Command extraction
# ---------------------------------------------------------------------------


class TestExtractCwdAndCode:
    def test_cd_and_python_c(self):
        info = _extract_cwd_and_code('cd C:/repo && python -c "print(42)"')
        assert info is not None
        assert info["cwd"] == "C:/repo"
        assert info["code"] == "print(42)"
        assert info["mode"] == "daemon"

    def test_python_c_no_cd(self):
        info = _extract_cwd_and_code('python -c "print(42)"')
        assert info is not None
        assert info["mode"] == "daemon"

    def test_python_script(self):
        info = _extract_cwd_and_code("cd C:/repo && python run.py")
        assert info is not None
        assert info["mode"] == "script"
        assert info["code_or_script"] == "run.py"

    def test_sleep(self):
        info = _extract_cwd_and_code("sleep 2")
        assert info is not None
        assert info["mode"] == "sleep"
        assert info["duration"] == 2.0

    def test_chained_command_not_optimized(self):
        """Commands with chained operators after the python part pass through."""
        assert _extract_cwd_and_code(
            'cd C:/repo && python -c "print(1)" && echo done'
        ) is None

    def test_git_status_not_optimizable(self):
        assert _extract_cwd_and_code("git status") is None


# ---------------------------------------------------------------------------
# optimize_command fast paths
# ---------------------------------------------------------------------------


class TestOptimizeCommand:
    def test_python_c_optimized(self):
        result = optimize_command('python -c "print(1+1)"')
        assert result is not None
        assert result["optimized"] is True
        assert result["ok"] is True
        assert "2" in result.get("stdout", "")

    def test_sleep_native(self):
        result = optimize_command("sleep 0.01")
        assert result is not None
        assert result["method"] == "sleep"
        assert result["ok"] is True

    def test_non_optimizable_returns_none(self):
        assert optimize_command("git status") is None
        assert optimize_command("npm test") is None
