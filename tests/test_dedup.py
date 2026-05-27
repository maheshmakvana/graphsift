"""Tests for entropy-based deduplication in ContextSelector."""

import hashlib
from pathlib import Path

import pytest

from graphsift import (
    ContextConfig,
    ContextSelector,
    DiffSpec,
    FileNode,
    Language,
    OutputMode,
    ScoredFile,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scored_file(path: str, source: str, score: float = 0.8) -> ScoredFile:
    """Create a ScoredFile for testing."""
    fn = FileNode(
        path=path,
        language=Language.PYTHON,
        size_bytes=len(source.encode()),
        line_count=len(source.splitlines()),
        token_estimate=estimate_tokens(source),
    )
    return ScoredFile(
        file_node=fn,
        score=score,
        rank=1,
        reasons=["test"],
        depth=0,
        output_mode=OutputMode.FULL,
    )


# ---------------------------------------------------------------------------
# _simhash
# ---------------------------------------------------------------------------


class TestSimHash:
    """Tests for the _simhash fingerprinting method."""

    def test_identical_sources_same_fingerprint(self):
        source = "def foo(): pass\n"
        fp1 = ContextSelector._simhash(source)
        fp2 = ContextSelector._simhash(source)
        assert fp1 == fp2

    def test_different_sources_different_fingerprints(self):
        fp1 = ContextSelector._simhash("def foo(): pass\n")
        fp2 = ContextSelector._simhash("class Bar:\n    pass\n")
        assert fp1 != fp2

    def test_empty_source_returns_default(self):
        fp = ContextSelector._simhash("")
        assert fp == "0" * 16

    def test_short_source_uses_direct_hash(self):
        """Source shorter than window uses direct MD5 (truncated to 64 bits)."""
        short = "short"
        fp = ContextSelector._simhash(short)
        expected = hashlib.md5(short.encode()).hexdigest()[:16]
        assert fp == expected

    def test_window_parameter_affects_result(self):
        source = "def hello(): return 1\n\ndef world(): return 2\n"
        fp_default = ContextSelector._simhash(source)
        fp_small = ContextSelector._simhash(source, window=8)
        assert fp_default != fp_small  # Different window sizes produce different fingerprints


# ---------------------------------------------------------------------------
# _hamming_distance
# ---------------------------------------------------------------------------


class TestHammingDistance:
    """Tests for Hamming distance computation."""

    def test_identical_fingerprints_zero_distance(self):
        fp = "abcd1234abcd1234"
        assert ContextSelector._hamming_distance(fp, fp) == 0

    def test_completely_different_fingerprints_large_distance(self):
        fp1 = "0" * 16
        fp2 = "f" * 16
        # All 64 bits differ
        assert ContextSelector._hamming_distance(fp1, fp2) == 64

    def test_single_bit_difference(self):
        # "1" vs "0" differs by 1 bit
        fp1 = "0000000000000001"
        fp2 = "0000000000000000"
        assert ContextSelector._hamming_distance(fp1, fp2) == 1

    def test_hex_digit_differences(self):
        # "f" (1111) vs "0" (0000) differs by 4 bits
        fp1 = "f000000000000000"
        fp2 = "0000000000000000"
        assert ContextSelector._hamming_distance(fp1, fp2) == 4

    def test_symmetric_distance(self):
        fp1 = "a000000000000000"
        fp2 = "0000000000000000"
        assert ContextSelector._hamming_distance(fp1, fp2) == ContextSelector._hamming_distance(fp2, fp1)


# ---------------------------------------------------------------------------
# _is_duplicate
# ---------------------------------------------------------------------------


class TestIsDuplicate:
    """Tests for the _is_duplicate method."""

    def test_identical_source_is_duplicate(self):
        selector = ContextSelector()
        source = "def hello(): pass\n"
        seen: set[str] = set()
        # First time: not a duplicate
        assert not selector._is_duplicate(source, seen)
        assert len(seen) == 1
        # Second time: duplicate
        assert selector._is_duplicate(source, seen)

    def test_different_sources_not_duplicate(self):
        selector = ContextSelector()
        seen: set[str] = set()
        assert not selector._is_duplicate("def foo(): pass\n", seen)
        assert not selector._is_duplicate("class Bar:\n    pass\n", seen)
        assert len(seen) == 2

    def test_fingerprint_only_added_for_novel_content(self):
        selector = ContextSelector()
        seen: set[str] = set()
        source = "some content\n"
        # Novel: not duplicate, fingerprint added
        assert not selector._is_duplicate(source, seen)
        assert len(seen) == 1
        # Duplicate: fingerprint NOT added
        assert selector._is_duplicate(source, seen)
        assert len(seen) == 1  # Still 1 — duplicate didn't add


# ---------------------------------------------------------------------------
# Integration: select_and_render dedup
# ---------------------------------------------------------------------------


class TestDedupInSelector:
    """Integration tests for dedup inside ContextSelector.select_and_render."""

    def test_identical_unrelated_files_deduped(self):
        """Two non-changed files with identical content: second should be deduped."""
        source = "def helper(): return 42\n"
        changed = _make_scored_file("main.py", "def main(): pass\n", score=1.0)
        other_a = _make_scored_file("util_a.py", source, score=0.8)
        other_b = _make_scored_file("util_b.py", source, score=0.8)

        selector = ContextSelector()
        diff = DiffSpec(changed_files=["main.py"])
        source_map = {
            "main.py": "def main(): pass\n",
            "util_a.py": source,
            "util_b.py": source,
        }
        selected, _, _, _ = selector.select_and_render(
            [changed, other_a, other_b], source_map, diff
        )
        paths = [sf.file_node.path for sf in selected]
        assert "main.py" in paths
        assert "util_a.py" in paths
        assert "util_b.py" not in paths  # deduped

    def test_different_files_both_selected(self):
        """Two completely different files should both be selected."""
        src_a = "def foo(): return 1\n"
        src_b = "class Bar:\n    def baz(self): return 2\n"
        sf1 = _make_scored_file("file_a.py", src_a)
        sf2 = _make_scored_file("file_b.py", src_b)
        selector = ContextSelector()
        diff = DiffSpec(changed_files=["file_a.py"])
        source_map = {"file_a.py": src_a, "file_b.py": src_b}
        selected, _, _, _ = selector.select_and_render([sf1, sf2], source_map, diff)
        paths = [sf.file_node.path for sf in selected]
        assert "file_a.py" in paths
        assert "file_b.py" in paths

    def test_changed_files_never_deduped(self):
        """Changed files should never be skipped even if identical."""
        source = "def process(): pass\n"
        sf1 = _make_scored_file("file_a.py", source, score=1.0)
        sf2 = _make_scored_file("file_b.py", source, score=1.0)
        selector = ContextSelector()
        diff = DiffSpec(changed_files=["file_a.py", "file_b.py"])
        source_map = {"file_a.py": source, "file_b.py": source}
        selected, _, _, _ = selector.select_and_render([sf1, sf2], source_map, diff)
        paths = [sf.file_node.path for sf in selected]
        assert "file_a.py" in paths
        assert "file_b.py" in paths  # also changed, so not deduped

    def test_dedup_disabled_via_config(self):
        """With dedup_enabled=False, no deduplication should occur."""
        source = "identical content\n"
        sf1 = _make_scored_file("file_a.py", source)
        sf2 = _make_scored_file("file_b.py", source)
        config = ContextConfig(dedup_enabled=False)
        selector = ContextSelector(config)
        diff = DiffSpec(changed_files=["file_a.py"])
        source_map = {"file_a.py": source, "file_b.py": source}
        selected, _, _, _ = selector.select_and_render([sf1, sf2], source_map, diff)
        paths = [sf.file_node.path for sf in selected]
        assert "file_a.py" in paths
        assert "file_b.py" in paths  # dedup disabled, so both pass

    def test_multiple_identical_files_deduped(self):
        """Multiple non-changed files with identical content: only first is selected."""
        source = "def helper(): return 42\n"
        changed = _make_scored_file("main.py", "def main(): pass\n", score=1.0)
        sfa = _make_scored_file("helper_a.py", source, score=0.7)
        sfb = _make_scored_file("helper_b.py", source, score=0.7)
        sfc = _make_scored_file("helper_c.py", source, score=0.7)

        selector = ContextSelector()
        diff = DiffSpec(changed_files=["main.py"])
        source_map = {
            "main.py": "def main(): pass\n",
            "helper_a.py": source,
            "helper_b.py": source,
            "helper_c.py": source,
        }
        selected, _, _, _ = selector.select_and_render(
            [changed, sfa, sfb, sfc], source_map, diff
        )
        paths = [sf.file_node.path for sf in selected]
        assert "main.py" in paths
        assert "helper_a.py" in paths  # first identical file passes
        assert "helper_b.py" not in paths  # deduped
        assert "helper_c.py" not in paths  # deduped

    def test_high_score_non_changed_still_skipped_if_duplicate(self):
        """Even high-scoring non-changed files get deduped if they are near-duplicates."""
        source = "shared utility code\n"
        sf1 = _make_scored_file("util_a.py", source, score=0.95)
        sf2 = _make_scored_file("util_b.py", source, score=0.95)
        changed = _make_scored_file("main.py", "def main(): pass\n", score=1.0)
        selector = ContextSelector()
        diff = DiffSpec(changed_files=["main.py"])
        source_map = {
            "main.py": "def main(): pass\n",
            "util_a.py": source,
            "util_b.py": source,
        }
        selected, _, _, _ = selector.select_and_render(
            [changed, sf1, sf2], source_map, diff
        )
        paths = [sf.file_node.path for sf in selected]
        assert "util_a.py" in paths
        assert "util_b.py" not in paths  # deduped despite high score

    def test_similar_files_with_long_shared_prefix_deduped(self):
        """Files that differ only at the very end should share enough common windows
        for the median fingerprint to be identical or within 3 bits."""
        # 400 chars of identical padding ensures the median window is well inside
        # the shared prefix region for any 64-char window.
        prefix = "# " + "x" * 397 + "\n"
        src_a = prefix + "def process(): return 42\n"
        src_b = prefix + "def compute(): return 100\n"

        sfa = _make_scored_file("process.py", src_a, score=0.7)
        sfb = _make_scored_file("compute.py", src_b, score=0.7)
        changed = _make_scored_file("main.py", "def main(): pass\n", score=1.0)

        selector = ContextSelector()
        diff = DiffSpec(changed_files=["main.py"])
        source_map = {
            "main.py": "def main(): pass\n",
            "process.py": src_a,
            "compute.py": src_b,
        }
        selected, _, _, _ = selector.select_and_render(
            [changed, sfa, sfb], source_map, diff
        )
        paths = [sf.file_node.path for sf in selected]
        assert "main.py" in paths
        # Only one of the two near-identical files should be selected
        similar_count = sum(1 for p in paths if p in ("process.py", "compute.py"))
        assert similar_count == 1, (
            f"Expected only 1 of the 2 near-identical files, got {paths}"
        )

    def test_dedup_does_not_block_changed_file_with_similar_unrelated(self):
        """A changed file's fingerprint should not block adding a similar non-changed file
        that was passed earlier (non-changed files are deduped against each other)."""
        src_a = "def handler(event):\n    print(event)\n"
        src_b = "def handler(event):\n    print(event)\n    return event\n"  # slightly different

        sfa = _make_scored_file("handler_v1.py", src_a, score=0.7)
        changed = _make_scored_file("handler_v2.py", src_b, score=1.0)

        selector = ContextSelector()
        diff = DiffSpec(changed_files=["handler_v2.py"])
        source_map = {
            "handler_v1.py": src_a,
            "handler_v2.py": src_b,
        }
        selected, _, _, _ = selector.select_and_render(
            [sfa, changed], source_map, diff
        )
        paths = [sf.file_node.path for sf in selected]
        # handler_v2 is changed, always included
        assert "handler_v2.py" in paths
        # handler_v1 is unrelated and different from handler_v2, should pass
        assert "handler_v1.py" in paths, (
            f"handler_v1 should be selected (unique content), got {paths}"
        )


# ---------------------------------------------------------------------------
# Config default
# ---------------------------------------------------------------------------


def test_dedup_enabled_default():
    """dedup_enabled should default to True."""
    config = ContextConfig()
    assert config.dedup_enabled is True


def test_dedup_enabled_can_be_disabled():
    """dedup_enabled can be set to False in config."""
    config = ContextConfig(dedup_enabled=False)
    assert config.dedup_enabled is False
