"""Simple benchmarks for graphsift operations.

Measures:
  - index_files() throughput (files/second)
  - build() latency
  - compress() throughput
  - Memory usage during large operations
"""

import gc
import time
import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec, OutputMode
from graphsift.compress import compress


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_benchmark_source_map(file_count: int) -> dict[str, str]:
    """Generate source files for benchmarking."""
    source_map = {}
    for i in range(file_count):
        mod = f"mod_{i:05d}"
        file_path = f"src/{mod}.py"

        imports = []
        if i > 0:
            for j in range(max(0, i - 5), i):
                imports.append(f"from mod_{j:05d} import HelperClass{j}")

        source = f'''"""{mod}."""
{chr(10).join(imports[:8])}

class HelperClass{i}:
    """Helper."""
    def __init__(self, val: int = {i}):
        self.val = val
    def run(self, x: int) -> int:
        return x + self.val

def top_func_{i}(x: int = 0) -> str:
    """Top-level function."""
    h = HelperClass{i}(x)
    return f"result-{{h.run(x)}}"
'''
        source_map[file_path] = source
    return source_map


# ---------------------------------------------------------------------------
# Index throughput benchmark
# ---------------------------------------------------------------------------

class TestIndexThroughput:
    """Measure index_files() throughput."""

    @pytest.mark.slow
    @pytest.mark.benchmark(min_rounds=1)
    def test_index_100_files(self, benchmark):
        """Index throughput for 100 files."""
        source_map = _generate_benchmark_source_map(100)

        def _do_index():
            builder = ContextBuilder(ContextConfig(token_budget=200_000))
            stats = builder.index_files(source_map)
            return stats.files_indexed

        result = benchmark(_do_index)
        assert result == 100

    @pytest.mark.slow
    @pytest.mark.benchmark(min_rounds=1)
    def test_index_500_files(self, benchmark):
        """Index throughput for 500 files."""
        source_map = _generate_benchmark_source_map(500)

        def _do_index():
            builder = ContextBuilder(ContextConfig(token_budget=500_000))
            stats = builder.index_files(source_map)
            return stats.files_indexed

        result = benchmark(_do_index)
        assert result == 500

    @pytest.mark.slow
    def test_index_2000_files_throughput(self):
        """Measure indexing throughput for 2000 files."""
        source_map = _generate_benchmark_source_map(2000)

        gc.collect()
        start = time.perf_counter()
        builder = ContextBuilder(ContextConfig(token_budget=1_000_000))
        stats = builder.index_files(source_map)
        elapsed = time.perf_counter() - start

        throughput = 2000 / elapsed if elapsed > 0 else 0
        print(f"\n  Index 2000 files: {elapsed:.2f}s ({throughput:.0f} files/sec)")

        assert stats.files_indexed == 2000
        assert throughput > 10, f"Throughput {throughput:.0f} files/sec too slow"


# ---------------------------------------------------------------------------
# Build latency benchmark
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed_builder_100():
    """Pre-indexed builder with 100 files."""
    source_map = _generate_benchmark_source_map(100)
    builder = ContextBuilder(ContextConfig(token_budget=100_000))
    builder.index_files(source_map)
    return builder, source_map


@pytest.fixture(scope="module")
def indexed_builder_500():
    """Pre-indexed builder with 500 files."""
    source_map = _generate_benchmark_source_map(500)
    builder = ContextBuilder(ContextConfig(token_budget=500_000))
    builder.index_files(source_map)
    return builder, source_map


class TestBuildLatency:
    """Measure build() latency."""

    @pytest.mark.benchmark(min_rounds=5)
    def test_build_latency_100(self, benchmark, indexed_builder_100):
        """Build latency with 100 files."""
        builder, source_map = indexed_builder_100
        diff = DiffSpec(
            changed_files=["src/mod_00000.py"],
            query="Review the changes",
        )

        result = benchmark(lambda: builder.build(diff, source_map))
        assert result.files_selected >= 1

    @pytest.mark.benchmark(min_rounds=3)
    def test_build_latency_500(self, benchmark, indexed_builder_500):
        """Build latency with 500 files."""
        builder, source_map = indexed_builder_500
        diff = DiffSpec(
            changed_files=["src/mod_00000.py"],
            query="Review the changes",
        )

        result = benchmark(lambda: builder.build(diff, source_map))
        assert result.files_selected >= 1

    @pytest.mark.slow
    def test_build_multiple_changed_files(self):
        """Build latency with varying changed file counts."""
        source_map = _generate_benchmark_source_map(300)
        builder = ContextBuilder(ContextConfig(token_budget=200_000))
        builder.index_files(source_map)

        for changed_count in [1, 5, 10, 30]:
            changed = [f"src/mod_{i:05d}.py" for i in range(changed_count)]
            diff = DiffSpec(changed_files=changed)

            gc.collect()
            start = time.perf_counter()
            result = builder.build(diff, source_map)
            elapsed = time.perf_counter() - start

            print(f"\n  Build with {changed_count} changed files: "
                  f"{elapsed:.4f}s, selected {result.files_selected} files")
            assert result.files_selected >= changed_count


# ---------------------------------------------------------------------------
# Compress throughput benchmark
# ---------------------------------------------------------------------------

class TestCompressThroughput:
    """Measure compress() throughput."""

    TEXT_SAMPLES = {
        "small": "Line 1\nLine 2\nLine 3\n",
        "medium": "\n".join(f"Line {i}" for i in range(100)),
        "large": "\n".join(f"Line {i} - some sample content for benchmarking purposes" for i in range(1000)),
        "pytest": "=== 5 passed, 1 failed in 0.5s ===\n" * 50,
        "git_diff": "diff --git a/foo.py b/foo.py\nindex abc..def\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old code\n+new code\n" * 20,
        "json": '{"key": "value", "nested": {"a": 1, "b": 2}, "items": [1, 2, 3, 4, 5]}',
    }

    @pytest.mark.benchmark(min_rounds=10)
    @pytest.mark.parametrize("sample_name", ["small", "medium", "pytest"])
    def test_compress_throughput(self, benchmark, sample_name):
        """Compress throughput for different sample sizes."""
        text = self.TEXT_SAMPLES[sample_name]

        result = benchmark(lambda: compress(text))
        assert isinstance(result, str)

    @pytest.mark.slow
    def test_compress_all_types(self):
        """Measure compress with all compressor types."""
        text = "Some sample CLI output\nwith multiple lines\nfor testing\n"
        types = ["auto", "pytest", "generic", "git_diff", "git_status",
                 "json_output", "log", "cat"]

        for cmd in types:
            gc.collect()
            start = time.perf_counter()
            for _ in range(100):
                result = compress(text, command=cmd)
            elapsed = time.perf_counter() - start

            ops_per_sec = 100 / elapsed if elapsed > 0 else 0
            print(f"\n  compress({cmd}): {ops_per_sec:.0f} ops/sec")
            assert isinstance(result, str)

    @pytest.mark.slow
    def test_compress_varying_lengths(self):
        """Compress performance with varying input lengths."""
        for length in [10, 100, 1000, 10_000, 50_000]:
            text = "x\n" * length

            gc.collect()
            start = time.perf_counter()
            result = compress(text)
            elapsed = time.perf_counter() - start

            print(f"\n  compress({length} lines): {elapsed*1000:.2f}ms, "
                  f"output {len(result.split(chr(10)))} lines")
            assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Memory usage benchmarks
# ---------------------------------------------------------------------------

class TestMemoryUsage:
    """Measure memory usage during operations."""

    @pytest.mark.slow
    def test_memory_during_index(self):
        """Measure memory during indexing of 500 files."""
        import tracemalloc
        tracemalloc.start()

        try:
            source_map = _generate_benchmark_source_map(500)
            gc.collect()

            snapshot_before = tracemalloc.take_snapshot()

            builder = ContextBuilder(ContextConfig(token_budget=500_000))
            builder.index_files(source_map)

            gc.collect()
            snapshot_after = tracemalloc.take_snapshot()

            stats = snapshot_after.compare_to(snapshot_before, "lineno")
            total_diff = sum(stat.size_diff for stat in stats)
            diff_mb = total_diff / (1024 * 1024)

            print(f"\n  Memory delta after indexing 500 files: {diff_mb:+.1f} MB")
            # Should not use more than 500 MB
            assert diff_mb < 500, f"Indexing used {diff_mb:.0f} MB"
        finally:
            tracemalloc.stop()

    @pytest.mark.slow
    def test_memory_during_build(self, indexed_builder_500):
        """Measure memory during build."""
        import tracemalloc
        tracemalloc.start()

        try:
            builder, source_map = indexed_builder_500
            diff = DiffSpec(changed_files=["src/mod_00000.py"])

            gc.collect()
            snapshot_before = tracemalloc.take_snapshot()

            result = builder.build(diff, source_map)

            gc.collect()
            snapshot_after = tracemalloc.take_snapshot()

            stats = snapshot_after.compare_to(snapshot_before, "lineno")
            total_diff = sum(stat.size_diff for stat in stats)
            diff_mb = total_diff / (1024 * 1024)

            print(f"\n  Memory delta during build: {diff_mb:+.1f} MB")
            assert result.files_selected >= 1
        finally:
            tracemalloc.stop()
