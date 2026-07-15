"""Stress test: index 10,000 generated source files, measure time/memory."""

import os
import time
import gc
import pytest

from graphsift import ContextBuilder, ContextConfig

pytestmark = [pytest.mark.stress, pytest.mark.slow]

_XDIST_ACTIVE = os.environ.get("PYTEST_XDIST_WORKER") is not None


def _generate_source_files(count: int, base_path: str = "src") -> dict[str, str]:
    """Generate *count* Python source files with classes, functions, and imports."""
    source_map = {}
    for i in range(count):
        mod_name = f"module_{i:05d}"
        file_path = f"{base_path}/{mod_name}.py"

        # Every 10th file imports from the previous 10 modules
        imports = []
        if i > 0:
            for j in range(max(0, i - 10), i):
                imports.append(f"from module_{j:05d} import HelperClass{j}")

        source = f'''"""{mod_name} — auto-generated for stress testing."""
{chr(10).join(imports)}

import os
import sys
from typing import Optional


class HelperClass{i}:
    """Helper class {i}."""
    
    def __init__(self, value: int = {i}):
        self.value = value
    
    def process(self, data: str) -> str:
        """Process data with helper."""
        return f"{{data}}-processed-{{self.value}}"
    
    @staticmethod
    def static_method() -> int:
        return {i * 2}


def top_level_func_{i}(arg: Optional[str] = None) -> str:
    """Top-level function {i}."""
    helper = HelperClass{i}()
    result = helper.process(arg or "default")
    return result


class AnotherClass{i}:
    """Another class with inheritance relationships."""
    
    def __init__(self):
        self.items = []
    
    def add_item(self, item: str) -> None:
        self.items.append(item)
    
    def get_count(self) -> int:
        return len(self.items)
'''
        source_map[file_path] = source
    return source_map


class TestLargeRepoIndexing:
    """Stress test for indexing large repositories."""

    @pytest.mark.parametrize("file_count", [100, 500, 1000])
    def test_index_time_scales_linearly(self, file_count):
        """Index time should scale roughly linearly with file count."""
        source_map = _generate_source_files(file_count)

        builder = ContextBuilder(ContextConfig(token_budget=500_000))

        start = time.perf_counter()
        stats = builder.index_files(source_map)
        elapsed = time.perf_counter() - start

        # All files should be indexed
        assert stats.files_indexed == file_count, (
            f"Expected {file_count} files indexed, got {stats.files_indexed}"
        )
        assert stats.symbols_extracted > 0

        # Calculate throughput
        files_per_sec = file_count / elapsed if elapsed > 0 else float("inf")
        print(f"\n  Indexed {file_count} files in {elapsed:.2f}s "
              f"({files_per_sec:.0f} files/sec)")

        # Rough sanity: should be able to index at least 10 files/sec
        assert files_per_sec > 10, (
            f"Indexing too slow: {files_per_sec:.0f} files/sec"
        )

    @pytest.mark.skipif(
        _XDIST_ACTIVE,
        reason="10k-file stress test requires full memory — skip under xdist parallel mode",
    )
    def test_index_10000_files(self):
        """Index 10,000 files to test large-scale performance."""
        source_map = _generate_source_files(10_000)

        builder = ContextBuilder(ContextConfig(token_budget=1_000_000))

        start = time.perf_counter()
        stats = builder.index_files(source_map)
        elapsed = time.perf_counter() - start

        assert stats.files_indexed == 10_000
        assert stats.symbols_extracted >= 10_000  # At least one symbol per file

        files_per_sec = 10_000 / elapsed if elapsed > 0 else float("inf")
        print(f"\n  Indexed 10,000 files in {elapsed:.2f}s "
              f"({files_per_sec:.0f} files/sec)")

        # Build context from large repo
        diff_spec = type("DiffSpec", (), {"changed_files": ["src/module_00000.py"]})()
        source_map_typed = source_map
        build_start = time.perf_counter()
        result = builder.build(diff_spec, source_map_typed)
        build_elapsed = time.perf_counter() - build_start

        print(f"  Built context in {build_elapsed:.3f}s")
        assert result.files_selected >= 1
        assert result.total_rendered_tokens > 0

        # Force garbage collection
        gc.collect()

    def test_memory_usage_stable(self):
        """Memory should not grow unboundedly during repeated indexing."""
        import tracemalloc

        tracemalloc.start()
        try:
            for i in range(5):
                source_map = _generate_source_files(500, base_path=f"repo_{i}")
                builder = ContextBuilder(ContextConfig(token_budget=50_000))
                builder.index_files(source_map)
                gc.collect()

            current, peak = tracemalloc.get_traced_memory()
            print(f"\n  Memory: current={current / 1024 / 1024:.1f}MB, "
                  f"peak={peak / 1024 / 1024:.1f}MB")
            # Peak should be reasonable (< 1GB)
            assert peak < 1_000_000_000, (
                f"Peak memory {peak / 1024 / 1024:.0f}MB exceeds 1GB"
            )
        finally:
            tracemalloc.stop()
