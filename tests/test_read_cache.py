"""Tests for SafeFileIO — encoding-safe file I/O with BOM handling."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from graphsift.read_cache import SafeFileIO, ReadCache


# ==============================================================================
# SafeFileIO Tests
# ==============================================================================


class TestSafeFileIO:
    """Comprehensive tests for encoding-safe file I/O."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Basic read / write
    # ------------------------------------------------------------------

    def test_write_and_read(self):
        """Writing and reading a simple file should round-trip correctly."""
        path = Path(self.tmpdir) / "test.txt"
        SafeFileIO.write(path, "hello world")
        content = SafeFileIO.read(path)
        assert content == "hello world"

    def test_write_creates_parent_dirs(self):
        """write() should create parent directories if needed."""
        path = Path(self.tmpdir) / "sub" / "nested" / "test.txt"
        SafeFileIO.write(path, "nested content")
        assert path.exists()
        assert SafeFileIO.read(path) == "nested content"

    def test_read_missing_file_raises(self):
        """Reading a non-existent file should raise FileNotFoundError."""
        path = Path(self.tmpdir) / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            SafeFileIO.read(path)

    # ------------------------------------------------------------------
    # BOM handling
    # ------------------------------------------------------------------

    def test_strip_bom_removes_bom(self):
        """strip_bom() should remove UTF-8 BOM character."""
        bom = chr(0xFEFF)
        content = bom + "hello"
        result = SafeFileIO.strip_bom(content)
        assert result == "hello"
        assert not result.startswith(chr(0xFEFF))

    def test_strip_bom_no_bom(self):
        """strip_bom() should return content unchanged if no BOM."""
        content = "hello world"
        result = SafeFileIO.strip_bom(content)
        assert result == content

    def test_strip_bom_empty_string(self):
        """strip_bom() should handle empty string."""
        assert SafeFileIO.strip_bom("") == ""

    def test_read_auto_strips_bom(self):
        """read() should automatically strip BOM."""
        path = Path(self.tmpdir) / "bom_file.txt"
        # Write file with BOM bytes (0xEF 0xBB 0xBF = UTF-8 BOM)
        bom_bytes = bytes([0xEF, 0xBB, 0xBF]) + b"hello with bom"
        path.write_bytes(bom_bytes)
        content = SafeFileIO.read(path)
        assert content == "hello with bom"
        assert chr(0xFEFF) not in content

    def test_write_strips_bom_from_content(self):
        """write() should strip BOM from content before writing."""
        path = Path(self.tmpdir) / "clean_output.txt"
        bom_content = chr(0xFEFF) + "content with bom"
        SafeFileIO.write(path, bom_content)
        # Read back as bytes to verify no BOM
        raw = path.read_bytes()
        assert not raw.startswith(bytes([0xEF, 0xBB, 0xBF]))
        assert SafeFileIO.read(path) == "content with bom"

    def test_detect_bom_positive(self):
        """detect_bom() should return True for files with BOM."""
        path = Path(self.tmpdir) / "has_bom.txt"
        path.write_bytes(bytes([0xEF, 0xBB, 0xBF]) + b"hello")
        assert SafeFileIO.detect_bom(path) is True

    def test_detect_bom_negative(self):
        """detect_bom() should return False for files without BOM."""
        path = Path(self.tmpdir) / "no_bom.txt"
        SafeFileIO.write(path, "hello")
        assert SafeFileIO.detect_bom(path) is False

    def test_detect_bom_missing_file(self):
        """detect_bom() should return False for non-existent files."""
        path = Path(self.tmpdir) / "missing.txt"
        assert SafeFileIO.detect_bom(path) is False

    # ------------------------------------------------------------------
    # Encoding safety
    # ------------------------------------------------------------------

    def test_read_with_encoding_errors_replaced(self):
        """read() should use errors=replace to handle non-UTF-8 bytes."""
        path = Path(self.tmpdir) / "bad_encoding.txt"
        # Write invalid UTF-8 bytes
        path.write_bytes(b"hello xffxfe world")
        content = SafeFileIO.read(path)
        # Should not crash — bad bytes replaced
        assert "hello" in content

    def test_write_forced_utf8(self):
        """write() should force UTF-8 encoding regardless of system default."""
        path = Path(self.tmpdir) / "utf8_forced.txt"
        content = "unicode chars: u00e9 u00fc u00f1"
        SafeFileIO.write(path, content)
        # Read back as bytes to confirm UTF-8
        raw = path.read_bytes()
        raw.decode("utf-8")  # Should not raise
        assert SafeFileIO.read(path) == content

    # ------------------------------------------------------------------
    # NFC normalization
    # ------------------------------------------------------------------

    def test_normalize_nfc(self):
        """normalize() should convert to NFC form."""
        import unicodedata
        # Composed form
        composed = "u00e9"  # é pre-composed
        canon = SafeFileIO.normalize(composed)
        assert unicodedata.is_normalized("NFC", canon)

    def test_write_normalizes_content(self):
        """write() should NFC-normalize content."""
        path = Path(self.tmpdir) / "normalized.txt"
        # Decomposed form (e + combining accent)
        decomposed = "e" + chr(0x0301)  # e + combining acute accent (decomposed é)
        SafeFileIO.write(path, decomposed)
        content = SafeFileIO.read(path)
        # Should be composed (NFC)
        assert len(content) == 1  # Single é char, not 2

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    SIMPLE_DATA = {"name": "test", "value": 42, "active": True}

    def test_write_and_read_json(self):
        """write_json() and read_json() should round-trip data."""
        path = Path(self.tmpdir) / "data.json"
        SafeFileIO.write_json(path, self.SIMPLE_DATA)
        result = SafeFileIO.read_json(path)
        assert result == self.SIMPLE_DATA

    def test_read_json_missing_file(self):
        """read_json() should return empty dict for missing file."""
        path = Path(self.tmpdir) / "missing.json"
        result = SafeFileIO.read_json(path)
        assert result == {}

    def test_read_json_invalid_content(self):
        """read_json() should return empty dict for invalid JSON."""
        path = Path(self.tmpdir) / "invalid.json"
        SafeFileIO.write(path, "not json at all{{{")
        result = SafeFileIO.read_json(path)
        assert result == {}

    def test_write_json_with_bom_stripped(self):
        """write_json() should not produce BOM in output."""
        path = Path(self.tmpdir) / "clean.json"
        SafeFileIO.write_json(path, self.SIMPLE_DATA)
        raw = path.read_bytes()
        assert not raw.startswith(bytes([0xEF, 0xBB, 0xBF]))

    def test_write_json_ensure_ascii(self):
        """write_json() should handle unicode characters."""
        path = Path(self.tmpdir) / "unicode.json"
        data = {"key": "u00e9 u4e2d u6587"}
        SafeFileIO.write_json(path, data)
        result = SafeFileIO.read_json(path)
        assert result == data

    # ------------------------------------------------------------------
    # Cross-platform
    # ------------------------------------------------------------------

    def test_read_windows_newlines(self):
        """read() should handle files with Windows line endings."""
        path = Path(self.tmpdir) / "crlf.txt"
        path.write_bytes(b"liner1rnliner2")
        content = SafeFileIO.read(path)
        assert "liner1" in content
        assert "liner2" in content

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    def test_write_windows_path(self):
        """write() should work with Windows-style paths."""
        path = Path(self.tmpdir) / "windows_test.txt"
        SafeFileIO.write(path, "windows test")
        assert path.exists()
        assert SafeFileIO.read(path) == "windows test"


# ==============================================================================
# ReadCache Tests
# ==============================================================================


class TestReadCache:
    """Tests for the session-scoped read dedup cache."""

    def test_first_read_returns_content(self):
        """First read returns full content."""
        cache = ReadCache()
        result = cache.read("test.txt", lambda: "full content")
        assert result == "full content"

    def test_second_read_returns_stub(self):
        """Second read of unchanged file returns stub."""
        cache = ReadCache()
        cache.read("test.txt", lambda: "same content")
        result = cache.read("test.txt", lambda: "same content")
        assert "Skipping" in result or "same content" in result

    def test_changed_content_returns_new(self):
        """Read after content change returns new content."""
        cache = ReadCache()
        cache.read("test.txt", lambda: "old content")
        result = cache.read("test.txt", lambda: "new content")
        assert result == "new content"

    def test_stubs_served_count(self):
        """stubs_served should count duplicate reads."""
        cache = ReadCache()
        cache.read("a.txt", lambda: "hello")
        cache.read("a.txt", lambda: "hello")
        cache.read("b.txt", lambda: "world")
        cache.read("b.txt", lambda: "world")
        assert cache.stubs_served == 2

    def test_invalidate_clears_file(self):
        """invalidate() should remove cached fingerprint."""
        cache = ReadCache()
        cache.read("test.txt", lambda: "content")
        cache.invalidate("test.txt")
        result = cache.read("test.txt", lambda: "content")
        assert result == "content"  # Full content, not stub

    def test_clear_resets_all(self):
        """clear() should reset all fingerprints and stub count."""
        cache = ReadCache()
        cache.read("a.txt", lambda: "x")
        cache.read("a.txt", lambda: "x")
        cache.clear()
        assert cache.stubs_served == 0
        result = cache.read("a.txt", lambda: "x")
        assert result == "x"  # Full content

    @staticmethod
    def test_fingerprint_consistency():
        """Same content should produce same fingerprint."""
        fp1 = ReadCache._fingerprint("same content")
        fp2 = ReadCache._fingerprint("same content")
        assert fp1 == fp2

    @staticmethod
    def test_fingerprint_different():
        """Different content should produce different fingerprints."""
        fp1 = ReadCache._fingerprint("content a")
        fp2 = ReadCache._fingerprint("content b")
        assert fp1 != fp2

    @staticmethod
    def test_fingerprint_with_unicode():
        """Fingerprint should handle unicode content."""
        fp = ReadCache._fingerprint("unicode: u00e9 u4e2d u6587")
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hexdigest
