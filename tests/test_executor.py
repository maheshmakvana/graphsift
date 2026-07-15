"""Tests for the silent executor with platform fallback."""

from __future__ import annotations

import sys

import pytest

from graphsift.executor import CommandExecutor, CommandResult, AutoPipeline, PipelineResult, _detect_shell


class TestCommandExecutor:
    """Tests for the cross-platform command executor."""

    def test_shell_detection_returns_string(self):
        """_detect_shell should return a non-empty string."""
        shell = _detect_shell()
        assert shell
        assert isinstance(shell, str)

    def test_run_simple_echo(self):
        """Running a simple echo command should succeed."""
        executor = CommandExecutor()
        result = executor.run("echo hello", check=False)
        assert result.exit_code == 0
        assert "hello" in result.stdout

    def test_run_failing_command(self):
        """A failing command should return non-zero exit code."""
        executor = CommandExecutor()
        if sys.platform == "win32":
            result = executor.run("exit 1", check=False)
        else:
            result = executor.run("false", check=False)
        assert result.exit_code != 0

    def test_run_check_raises(self):
        """With check=True, a failing command should raise."""
        executor = CommandExecutor()
        with pytest.raises(RuntimeError):
            if sys.platform == "win32":
                executor.run("exit 1", check=True)
            else:
                executor.run("false", check=True)

    def test_run_truncates_long_output(self):
        """Very long command output should be truncated."""
        executor = CommandExecutor(max_output_lines=5)
        # Generate 25 lines of output
        result = executor.run(
            'python -c "for x in range(25): print(str(x))"',
            check=False,
        )
        lines = result.stdout.strip().splitlines()
        assert len(lines) <= 10  # 5 max + truncation message

    def test_command_result_ok(self):
        """CommandResult.ok() should reflect exit code."""
        ok = CommandResult("", "", 0, "test")
        assert ok.ok() is True
        fail = CommandResult("", "", 1, "test")
        assert fail.ok() is False

    def test_max_retries_configurable(self):
        """Executor should accept custom retry config."""
        executor = CommandExecutor(max_retries=5, retry_delay=0.1)
        assert executor._max_retries == 5
        assert executor._retry_delay == 0.1


class TestAutoPipeline:
    """Tests for AutoPipeline structure."""

    def test_pipeline_result(self):
        """PipelineResult should store phase results."""
        result = AutoPipeline("/tmp").run(
            build=False, detect_dead_code=False,
            suggest_fixes=False, detect_cycles=False,
        )
        assert result.summary
        assert isinstance(result.failed(), bool)

    def test_pipeline_to_dict(self):
        """PipelineResult.to_dict() should serialize properly."""
        pr = PipelineResult(
            phases={"build": {"status": "ok"}},
            findings={"dead_code": {"count": 5}},
            applied=["fix1"],
            errors=[],
            summary="Done",
        )
        d = pr.to_dict()
        assert d["phases"]["build"]["status"] == "ok"
        assert len(d["applied"]) == 1
        assert d["errors"] == []
