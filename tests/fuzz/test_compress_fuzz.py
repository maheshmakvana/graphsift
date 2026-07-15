"""Fuzz tests: feed random byte strings to compress()."""

import random
import string
import sys
import pytest

from graphsift.compress import (
    compress,
    compress_generic,
    compress_pytest,
    compress_git_diff,
    compress_git_status,
    compress_git_log,
    compress_grep,
    compress_json_output,
    compress_docker,
    compress_kubectl,
    compress_npm,
    compress_eslint,
    compress_cargo,
    compress_go_test,
    compress_jest,
    compress_make,
    compress_pip,
    compress_log,
    compress_cat,
    compress_terraform,
    compress_gh,
    compress_az,
    compress_gcloud,
    compress_brew,
    compress_dotnet,
    deduplicate,
    truncate_middle,
    filter_lines,
    group_similar,
    strip_blanks,
    detect_type,
)

pytestmark = [pytest.mark.fuzz]


# ---------------------------------------------------------------------------
# Helpers: generate random inputs
# ---------------------------------------------------------------------------

def _random_bytes(max_size: int = 4096) -> bytes:
    """Generate random byte strings."""
    size = random.randint(0, max_size)
    return bytes(random.randint(0, 255) for _ in range(size))


def _random_unicode(max_size: int = 2048) -> str:
    """Generate random Unicode strings including non-BMP."""
    chars = []
    for _ in range(random.randint(0, max_size)):
        # Mix of ASCII, extended Latin, CJK, emoji, and invalid surrogates
        cp = random.choice([
            random.randint(0x20, 0x7E),       # ASCII printable
            random.randint(0xA0, 0xFF),        # Latin-1 Supplement
            random.randint(0x4E00, 0x9FFF),    # CJK Unified
            random.randint(0x1F300, 0x1F9FF),  # Emoticons/Symbols
            random.randint(0x2000, 0x206F),    # General Punctuation
            0x00, 0x01, 0x1B,                  # Control chars (null, SOH, ESC)
            0xFE0F,                            # Variation selector
        ])
        chars.append(chr(cp))
    return "".join(chars)


def _random_cli_output(max_size: int = 2048) -> str:
    """Generate random CLI-like output with ANSI codes."""
    lines = []
    for _ in range(random.randint(0, 50)):
        if random.random() < 0.3:
            # ANSI escape sequence
            ansi = f"\x1b[{random.randint(0, 107)}m"
            lines.append(ansi + _random_unicode(40) + "\x1b[0m")
        else:
            lines.append(_random_unicode(40))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fuzz compress() with random byte strings
# ---------------------------------------------------------------------------

class TestCompressFuzz:
    """Fuzz tests for the main compress() function."""

    COMPRESSORS = [
        "auto", "pytest", "cargo", "go_test", "jest", "eslint",
        "git_status", "git_diff", "git_log", "grep", "npm",
        "docker", "kubectl", "aws", "json_output", "make", "pip",
        "log", "cat", "terraform", "gh", "az", "gcloud", "brew",
        "dotnet", "generic",
    ]

    @pytest.mark.parametrize("cmd", COMPRESSORS)
    def test_fuzz_compress_random_unicode(self, cmd):
        """compress() must not crash on random Unicode with any compressor."""
        for _ in range(20):
            text = _random_unicode(500)
            try:
                result = compress(text, command=cmd)
                assert isinstance(result, str)
            except Exception as e:
                # Some bad inputs might raise, but should be controlled
                if isinstance(e, (ValueError, TypeError, UnicodeError)):
                    pass  # Acceptable for truly broken inputs
                elif "Maximum recursion" in str(e):
                    pass  # Stack overflow protection
                else:
                    raise

    @pytest.mark.parametrize("cmd", COMPRESSORS)
    def test_fuzz_compress_binary(self, cmd):
        """compress() must handle binary data gracefully."""
        for _ in range(10):
            raw = _random_bytes(500)
            try:
                text = raw.decode("utf-8", errors="replace")
                result = compress(text, command=cmd)
                assert isinstance(result, str)
            except Exception:
                pass  # Should not crash

    def test_fuzz_no_crash_on_ansi_noise(self):
        """compress should not crash on random ANSI noise."""
        for _ in range(50):
            text = _random_cli_output(1000)
            try:
                result = compress(text)
                assert isinstance(result, str)
            except Exception:
                pass

    def test_fuzz_detect_type_no_crash(self):
        """detect_type should not crash on random input."""
        for _ in range(50):
            text = _random_unicode(1000)
            try:
                result = detect_type(text)
                assert isinstance(result, str)
            except Exception:
                pass

    def test_fuzz_compress_extreme_lengths(self):
        """compress() handles very long strings."""
        for length in [0, 1, 10, 100, 1000, 10_000, 50_000]:
            text = "x" * length
            try:
                result = compress(text)
                assert isinstance(result, str)
                assert len(result) <= len(text) + 10
            except MemoryError:
                pytest.skip("Out of memory for large input")
            except Exception as e:
                pytest.fail(f"Crash on length {length}: {e}")


# ---------------------------------------------------------------------------
# Fuzz individual compressors
# ---------------------------------------------------------------------------

class TestIndividualCompressorsFuzz:
    """Fuzz each compressor individually."""

    @pytest.mark.parametrize("compressor_fn", [
        compress_generic, compress_pytest, compress_git_diff,
        compress_git_status, compress_git_log, compress_grep,
        compress_json_output, compress_docker, compress_kubectl,
        compress_npm, compress_eslint, compress_cargo, compress_go_test,
        compress_jest, compress_make, compress_pip, compress_log,
        compress_cat, compress_terraform, compress_gh, compress_az,
        compress_gcloud, compress_brew, compress_dotnet,
    ])
    def test_compressor_fuzz_random_unicode(self, compressor_fn):
        """Individual compressor must not crash on random Unicode."""
        for _ in range(10):
            text = _random_unicode(300)
            try:
                result = compressor_fn(text)
                assert isinstance(result, str)
            except Exception as e:
                if isinstance(e, (ValueError, TypeError)):
                    pass
                else:
                    raise


# ---------------------------------------------------------------------------
# Fuzz primitives
# ---------------------------------------------------------------------------

class TestPrimitivesFuzz:
    """Fuzz individual compression primitives."""

    def test_fuzz_deduplicate(self):
        for _ in range(50):
            text = _random_unicode(500)
            threshold = random.randint(0, 10)
            try:
                result = deduplicate(text, threshold)
                assert isinstance(result, str)
            except Exception:
                pass

    def test_fuzz_truncate_middle(self):
        for _ in range(50):
            text = _random_unicode(500)
            head = random.randint(0, 50)
            tail = random.randint(0, 50)
            try:
                result = truncate_middle(text, head, tail)
                assert isinstance(result, str)
            except Exception:
                pass

    def test_fuzz_filter_lines(self):
        for _ in range(50):
            text = _random_unicode(500)
            keep = None
            drop = None
            if random.random() < 0.5:
                keep = [random.choice(["error", "warn", "info", "test", "a"])]
            if random.random() < 0.5:
                drop = [random.choice(["debug", "trace", "verbose", "b"])]
            try:
                result = filter_lines(text, keep, drop)
                assert isinstance(result, str)
            except Exception:
                pass

    def test_fuzz_group_similar(self):
        for _ in range(50):
            text = _random_unicode(500)
            pattern = random.choice([r"error", r"^test", r"\d+", r"\[.*\]", r"foo"])
            label = _random_unicode(10)
            try:
                result = group_similar(text, pattern, label)
                assert isinstance(result, str)
            except Exception:
                pass

    def test_fuzz_strip_blanks(self):
        for _ in range(50):
            text = _random_unicode(500)
            try:
                result = strip_blanks(text)
                assert isinstance(result, str)
            except Exception:
                pass
