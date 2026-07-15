"""Stress test: 50 threads accessing ContextBuilder simultaneously."""

import threading
import time
import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec, OutputMode

pytestmark = [pytest.mark.stress, pytest.mark.slow]


def _make_source_map(size: int = 50) -> dict[str, str]:
    """Generate a small source map for concurrent access testing."""
    source_map = {}
    for i in range(size):
        file_path = f"src/module_{i:03d}.py"
        source = f'''"""{file_path}"""
import os
from typing import Optional

value_{i} = {i}

class Worker{i}:
    """Worker class."""
    def run(self, x: int = 0) -> int:
        return x + {i}

def compute_{i}(data: Optional[str] = None) -> str:
    """Compute function."""
    w = Worker{i}()
    result = w.run({i})
    return f"result-{{result}}"
'''
        source_map[file_path] = source
    return source_map


class TestConcurrentAccess:
    """Stress test for thread-safe access to ContextBuilder."""

    SOURCE_MAP = _make_source_map(50)
    CHANGED_FILES = [f"src/module_{i:03d}.py" for i in range(3)]

    def _build_context(self, builder, source_map, results, idx):
        """Thread worker: build context."""
        try:
            diff = DiffSpec(
                changed_files=self.CHANGED_FILES,
                query=f"Review thread {idx}",
            )
            result = builder.build(diff, source_map)
            results[idx] = {
                "success": True,
                "files_selected": result.files_selected,
                "tokens": result.total_rendered_tokens,
            }
        except Exception as e:
            results[idx] = {"success": False, "error": str(e)}

    def _index_files(self, builder, source_map, results, idx):
        """Thread worker: index files."""
        try:
            stats = builder.index_files(source_map)
            results[idx] = {
                "success": True,
                "files_indexed": stats.files_indexed,
            }
        except Exception as e:
            results[idx] = {"success": False, "error": str(e)}

    def test_concurrent_builds(self):
        """50 threads simultaneously building context."""
        builder = ContextBuilder(ContextConfig(token_budget=100_000))
        builder.index_files(self.SOURCE_MAP)

        num_threads = 50
        threads = []
        results = [None] * num_threads

        start = time.perf_counter()
        for i in range(num_threads):
            t = threading.Thread(
                target=self._build_context,
                args=(builder, self.SOURCE_MAP, results, i),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        # Check results
        successes = sum(1 for r in results if r and r.get("success"))
        failures = sum(1 for r in results if r and not r.get("success"))

        print(f"\n  {num_threads} concurrent builds: "
              f"{successes} success, {failures} failure, "
              f"in {elapsed:.2f}s")

        assert successes >= num_threads * 0.8, (
            f"Only {successes}/{num_threads} concurrent builds succeeded"
        )

    def test_concurrent_index_and_build(self):
        """Mix of indexing and building threads."""
        builder = ContextBuilder(ContextConfig(token_budget=100_000))
        # Pre-index
        builder.index_files(self.SOURCE_MAP)

        num_threads = 30
        threads = []
        results = [None] * num_threads

        # Half index, half build
        for i in range(num_threads):
            if i % 2 == 0:
                t = threading.Thread(
                    target=self._build_context,
                    args=(builder, self.SOURCE_MAP, results, i),
                )
            else:
                t = threading.Thread(
                    target=self._index_files,
                    args=(builder, self.SOURCE_MAP, results, i),
                )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        successes = sum(1 for r in results if r and r.get("success"))
        assert successes >= num_threads * 0.7, (
            f"Only {successes}/{num_threads} mixed operations succeeded"
        )

    def test_concurrent_with_different_configs(self):
        """Multiple ContextBuilder instances with different configs."""
        configs = [
            ContextConfig(token_budget=10_000, output_mode=OutputMode.FULL),
            ContextConfig(token_budget=50_000, output_mode=OutputMode.SMART),
            ContextConfig(token_budget=100_000, output_mode=OutputMode.SIGNATURES),
            ContextConfig(token_budget=200_000, output_mode=OutputMode.COMPRESSED),
            ContextConfig(token_budget=500_000, output_mode=OutputMode.SMART,
                          hot_threshold=0.9, warm_threshold=0.5),
        ]

        num_repeats = 10  # 5 configs * 10 = 50 operations
        threads = []
        results = [None] * (len(configs) * num_repeats)

        def _build_with_config(config, source_map, result_slot):
            try:
                builder = ContextBuilder(config)
                builder.index_files(source_map)
                diff = DiffSpec(changed_files=["src/module_000.py"])
                result = builder.build(diff, source_map)
                results[result_slot] = {
                    "success": True,
                    "tokens": result.total_rendered_tokens,
                }
            except Exception as e:
                results[result_slot] = {"success": False, "error": str(e)}

        idx = 0
        for _ in range(num_repeats):
            for config in configs:
                t = threading.Thread(
                    target=_build_with_config,
                    args=(config, self.SOURCE_MAP, idx),
                )
                threads.append(t)
                t.start()
                idx += 1

        for t in threads:
            t.join()

        successes = sum(1 for r in results if r and r.get("success"))
        total = len(configs) * num_repeats
        assert successes >= total * 0.75, (
            f"Only {successes}/{total} different-config builds succeeded"
        )
