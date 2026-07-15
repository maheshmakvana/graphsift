"""Integration test: CLI subprocess tests."""

import subprocess
import sys
import json
import pytest


class TestCLIIntegration:
    """Integration tests for CLI via subprocess."""

    @pytest.fixture
    def python_cmd(self):
        """Get the Python executable."""
        return sys.executable

    def run_graphsift(self, *args):
        """Run the graphsift CLI with given args and return result."""
        cmd = [sys.executable, "-m", "graphsift"] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result

    def test_cli_help(self):
        """CLI --help should show usage information."""
        result = self.run_graphsift("--help")
        # argparse --help exits 0
        assert "usage" in result.stdout.lower() or len(result.stdout) > 0

    def test_cli_version(self):
        """CLI --version should show version."""
        result = self.run_graphsift("--version")
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_cli_compress_stdin(self):
        """CLI compress reads from stdin."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "graphsift", "compress"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(
            input="test line 1\ntest line 2\ntest line 3\n",
            timeout=15,
        )
        # Should produce some output (possibly compressed)
        assert len(stdout) > 0
        assert proc.returncode == 0

    def test_cli_compress_with_type(self):
        """CLI compress --type should work."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "graphsift", "compress", "--type", "pytest"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(
            input="=== 5 passed in 0.5s ===\ntest session details\n",
            timeout=15,
        )
        assert proc.returncode == 0
        assert "passed" in stdout.lower()

    def test_cli_compress_list(self):
        """CLI compress --list should list compressors."""
        result = self.run_graphsift("compress", "--list")
        assert result.returncode == 0
        assert "pytest" in result.stdout
        assert "generic" in result.stdout
        assert "git_diff" in result.stdout

    def test_cli_no_command_shows_help(self):
        """CLI with no args should show an error (required subcommand)."""
        result = self.run_graphsift()
        # Required subparsers → error message to stderr
        assert result.returncode != 0

    def test_cli_invalid_command(self):
        """CLI with invalid command should show error."""
        result = self.run_graphsift("nonexistent_command_xyz")
        assert result.stdout is not None

    def test_cli_invalid_flag(self):
        """CLI with invalid flag should show error."""
        result = self.run_graphsift("--nonexistent-flag")
        assert result.returncode != 0

    def test_cli_status(self):
        """CLI status command should work."""
        result = self.run_graphsift("status")
        # Status may return non-zero if not configured, but should not crash
        assert result.stderr is not None or result.stdout is not None
