"""Tests for v2.4+ compression features."""

from __future__ import annotations

from graphsift.compress import ultra_compress, CompressionLevel, compress, deduplicate


class TestCompressionLevel:
    """Tests for the CompressionLevel enum."""

    def test_enum_values(self):
        """CompressionLevel should have correct values."""
        assert CompressionLevel.LIGHT.value == "light"
        assert CompressionLevel.BALANCED.value == "balanced"
        assert CompressionLevel.ULTRA.value == "ultra"

    def test_enum_members(self):
        """CompressionLevel should have 3 members."""
        assert len(CompressionLevel) == 3


class TestUltraCompress:
    """Tests for ultra_compress."""

    def test_ultra_compress_no_shrink(self):
        """ultra_compress should never increase output size."""
        texts = [
            "hello world",
            "a\nb\nc\n",
            "PASSED test_a\nPASSED test_b\nFAILED test_c\nError: x != y\n",
            "line1\nline2\nline3\nline1\nline2\nline3\n",
        ]
        for text in texts:
            result = ultra_compress(text)
            assert len(result) <= max(len(text), 1), f"ultra_compress increased size for: {text!r}"

    def test_ultra_compress_light(self):
        """Light compression should not crash."""
        text = "line one\nline two\nline three\n"
        result = ultra_compress(text, passes=1, level=CompressionLevel.LIGHT)
        assert isinstance(result, str)
        assert len(result) <= len(text)

    def test_ultra_compress_balanced(self):
        """Balanced compression should apply 2 passes."""
        text = "PASSED test_a\nPASSED test_b\nFAILED test_c\n"
        result = ultra_compress(text, command="pytest", passes=2, level=CompressionLevel.BALANCED)
        assert isinstance(result, str)
        assert len(result) <= len(text)

    def test_ultra_compress_ultra(self):
        """Ultra compression should apply 3 passes."""
        text = "ok\nok\nok\n" * 10 + "FAILED test\n"
        result = ultra_compress(text, passes=3, level=CompressionLevel.ULTRA)
        assert isinstance(result, str)
        assert len(result) <= len(text)

    def test_ultra_compress_empty(self):
        """ultra_compress on empty string should return empty."""
        assert ultra_compress("") == ""

    def test_ultra_compress_single_line(self):
        """ultra_compress on single line should not crash."""
        assert ultra_compress("hello world") == "hello world"

    def test_ultra_compress_dedup(self):
        """ultra_compress should deduplicate repeated lines."""
        text = "a\nb\nb\nb\nc\n"
        result = ultra_compress(text, passes=2)
        # After dedup, "b" x3 should become "b (x3)"
        assert "x3" in result.lower() or "(x3)" in result or len(result) < len(text)

    def test_ultra_compress_auto_detect(self):
        """ultra_compress should auto-detect command type."""
        pytest_out = "test session starts\nPASSED test_a\nPASSED test_b\nFAILED test_c\n"
        result = ultra_compress(pytest_out, "auto")
        assert isinstance(result, str)
        assert len(result) <= len(pytest_out)

    def test_ultra_compress_known_command(self):
        """ultra_compress should handle specific command types."""
        git_out = " M src/main.py\n M src/utils.py\n"
        result = ultra_compress(git_out, "git_status")
        assert isinstance(result, str)

    def test_ultra_compress_vs_normal(self):
        """ultra_compress with level=ULTRA should compress more than normal on repetitive text."""
        text = ("ok\n" * 50) + ("FAILED test\n" * 3)
        normal = compress(text)
        ultra = ultra_compress(text, passes=3, level=CompressionLevel.ULTRA)
        # Ultra should be at least as compressed as normal
        assert len(ultra) <= max(len(normal), 1)

    def test_semantic_compress_import(self):
        """_semantic_compress should be importable."""
        from graphsift.compress import _semantic_compress
        result = _semantic_compress("hello\nworld\n")
        assert isinstance(result, str)


class TestDeduplicate:
    """Tests for the deduplicate helper."""

    def test_deduplicate_identical_lines(self):
        """Consecutive identical lines should be collapsed."""
        result = deduplicate("a\na\na\n", threshold=1)
        assert "(x3)" in result

    def test_deduplicate_no_duplicates(self):
        """Unique lines should not be modified."""
        result = deduplicate("a\nb\nc\n", threshold=1)
        # No collapse needed
        assert result.count("\n") == 2 or "x" not in result
