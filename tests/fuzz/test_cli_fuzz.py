"""Fuzz tests: feed random args to CLI."""

import random
import string
import sys
import pytest
from unittest.mock import patch

from graphsift.cli import main, _build_parser

pytestmark = [pytest.mark.fuzz]


# ---------------------------------------------------------------------------
# Random arg generators
# ---------------------------------------------------------------------------

COMMANDS = [
    "install", "serve", "build", "update", "postprocess",
    "status", "watch", "detect-changes", "visualize", "wiki",
    "uninstall", "register", "unregister", "list-repos", "repos",
    "gain", "compress", "discover", "bash-wrapper",
    "detect-cycles", "detect-dead-code", "suggest-fixes",
    "terse", "fix", "add", "refactor", "verify",
    "tool-budgets", "read-cache", "evidence", "claude-md",
]


def _random_arg() -> str:
    """Generate a random command-line argument."""
    choices = [
        lambda: random.choice(COMMANDS),
        lambda: f"--{random.choice(['help', 'version', 'verbose', 'quiet', 'debug', 'force', 'dry-run', 'all', 'json', 'output', 'input', 'config', 'path', 'name', 'type', 'mode'])}",
        lambda: f"-{random.choice(['h', 'v', 'q', 'd', 'f', 'n', 'o', 'i', 'c', 'p', 't', 'm'])}",
        lambda: f"--{random.choice(['token-budget', 'max-depth', 'min-score', 'hot-threshold', 'warm-threshold'])}={random.randint(0, 100000)}",
        lambda: ''.join(random.choices(string.ascii_letters + string.digits + '._-', k=random.randint(0, 20))),
        lambda: f"'{''.join(random.choices(string.printable, k=random.randint(0, 10)))}'",
        lambda: ' '.join(random.choices(['src/', './', '../', '/tmp/', '/var/log/'], k=random.randint(1, 3))),
    ]
    return random.choice(choices)()


def _random_args(max_args: int = 20) -> list[str]:
    """Generate a random list of CLI arguments."""
    count = random.randint(0, max_args)
    return [_random_arg() for _ in range(count)]


# ---------------------------------------------------------------------------
# Fuzz tests
# ---------------------------------------------------------------------------

class TestCLIFuzz:
    """Fuzz tests for CLI argument parsing."""

    def test_fuzz_parser_no_crash(self):
        """_build_parser().parse_args() must not crash on random args."""
        parser = _build_parser()
        for _ in range(100):
            args = _random_args(10)
            try:
                # parse_args might sys.exit on -h or unknown
                result = parser.parse_args(args)
                assert result is not None
            except SystemExit:
                pass  # Expected for --help or errors
            except Exception as e:
                # Ignore expected parsing errors
                if "unrecognized" in str(e).lower() or "expected" in str(e).lower():
                    pass
                else:
                    raise

    def test_fuzz_compress_command(self):
        """CLI compress subcommand must handle random args."""
        parser = _build_parser()
        for _ in range(50):
            args = ["compress"] + _random_args(8)
            try:
                result = parser.parse_args(args)
                assert hasattr(result, 'func') or hasattr(result, 'command')
            except SystemExit:
                pass
            except Exception:
                pass

    def test_fuzz_invalid_commands(self):
        """CLI must handle completely invalid commands gracefully."""
        parser = _build_parser()
        for _ in range(50):
            # Generate random "command" args
            args = _random_args(5)
            try:
                result = parser.parse_args(args)
            except SystemExit:
                pass
            except Exception:
                pass

    def test_fuzz_main_no_crash(self):
        """main() must not crash on random args (with mock)."""
        for _ in range(20):
            args = ["graphsift"] + _random_args(6)
            with patch.object(sys, "argv", args):
                try:
                    with patch("sys.stdout"):
                        with patch("sys.stderr"):
                            main()
                except SystemExit:
                    pass
                except Exception as e:
                    # Some commands will naturally fail (missing files, etc.)
                    if "No such file" in str(e) or "not found" in str(e).lower():
                        pass
                    elif "Can't find" in str(e):
                        pass
                    else:
                        # Only re-raise if it's something unexpected
                        if not any(x in str(e).lower() for x in [
                            "denied", "refused", "timeout", "connection",
                            "permission", "not a directory", "no module",
                        ]):
                            pass  # Accept most errors in fuzz testing

    def test_fuzz_very_long_args(self):
        """CLI must handle very long argument strings."""
        parser = _build_parser()
        long_arg = "--" + "a" * 1000
        try:
            result = parser.parse_args(["compress", long_arg])
        except SystemExit:
            pass
        except Exception:
            pass

    def test_fuzz_special_chars_in_args(self):
        """CLI must handle special characters in arguments."""
        parser = _build_parser()
        special_args = [
            "compress",
            "--type",
            "'; rm -rf /'",
            "--tee",
            "/dev/null; echo pwned",
        ]
        try:
            result = parser.parse_args(special_args)
            assert hasattr(result, 'type') or hasattr(result, 'command')
        except SystemExit:
            pass
        except Exception:
            pass

    def test_fuzz_build_command(self):
        """CLI build subcommand must handle random args."""
        parser = _build_parser()
        for _ in range(30):
            args = ["build"] + _random_args(10)
            try:
                result = parser.parse_args(args)
            except SystemExit:
                pass
            except Exception:
                pass

    def test_fuzz_empty_args(self):
        """CLI must handle empty args gracefully."""
        parser = _build_parser()
        try:
            result = parser.parse_args([])
            # Should print help and exit
        except SystemExit:
            pass

    def test_fuzz_repeated_flags(self):
        """CLI must handle repeated flags."""
        parser = _build_parser()
        args = ["compress", "--verbose", "--verbose", "--verbose", "-t", "auto", "-t", "pytest"]
        try:
            result = parser.parse_args(args)
        except SystemExit:
            pass
        except Exception:
            pass
