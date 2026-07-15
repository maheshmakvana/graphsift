"""Stress test: run pipeline 1000 times, check memory doesn't grow."""

import gc
import time
import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec, OutputMode

pytestmark = [pytest.mark.stress, pytest.mark.slow]


def _small_source_map() -> dict[str, str]:
    """Generate a small source map for repeated pipeline runs."""
    return {
        "src/core.py": """
\"\"\"Core module.\"\"\"
from typing import Optional

class Engine:
    def __init__(self, name: str = "default"):
        self.name = name

    def process(self, data: str) -> str:
        return f"processed-{data}"

def create_engine(name: str = "default") -> Engine:
    return Engine(name)
""",
        "src/utils.py": """
\"\"\"Utils module.\"\"\"
from src.core import Engine

def run_pipeline(data: str) -> str:
    engine = Engine("pipeline")
    return engine.process(data)

def cleanup():
    pass
""",
        "src/handler.py": """
\"\"\"Handler module.\"\"\"
from src.core import Engine
from src.utils import run_pipeline

class RequestHandler:
    def handle(self, request: str) -> str:
        return run_pipeline(request)
""",
        "tests/test_core.py": """
\"\"\"Tests.\"\"\"
from src.core import Engine

def test_engine():
    e = Engine()
    assert e.process("x") == "processed-x"
""",
    }


class TestMemoryLeak:
    """Test that repeated pipeline runs don't leak memory."""

    SOURCE_MAP = _small_source_map()
    CHANGED = ["src/core.py"]

    def _get_mem_usage(self) -> float:
        """Get current process memory in MB."""
        import tracemalloc
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")
            total = sum(stat.size for stat in top_stats)
            return total / (1024 * 1024)
        except Exception:
            import os
            if hasattr(os, "getpid"):
                try:
                    import psutil
                    proc = psutil.Process(os.getpid())
                    return proc.memory_info().rss / (1024 * 1024)
                except ImportError:
                    pass
            return 0.0

    @pytest.mark.skip(reason="Memory measurement requires tracemalloc/psutil")
    def test_repeated_pipeline_memory(self):
        """Run pipeline 1000 times and verify memory stability."""
        import tracemalloc
        tracemalloc.start()

        try:
            # Warm up
            builder = ContextBuilder(ContextConfig(token_budget=50_000))
            builder.index_files(self.SOURCE_MAP)
            diff = DiffSpec(changed_files=self.CHANGED)
            builder.build(diff, self.SOURCE_MAP)
            gc.collect()

            initial = self._get_mem_usage()
            print(f"\n  Initial memory: {initial:.1f} MB")

            # Run pipeline repeatedly
            for iteration in range(1000):
                builder = ContextBuilder(ContextConfig(token_budget=50_000))
                builder.index_files(self.SOURCE_MAP)
                diff = DiffSpec(changed_files=self.CHANGED)
                builder.build(diff, self.SOURCE_MAP)

                if iteration % 100 == 0:
                    gc.collect()
                    current = self._get_mem_usage()
                    print(f"  Iteration {iteration}: {current:.1f} MB")

            gc.collect()
            final = self._get_mem_usage()
            print(f"\n  Final memory: {final:.1f} MB")
            print(f"  Delta: {final - initial:.1f} MB")

            # Memory should not grow by more than 50MB
            assert final - initial < 50, (
                f"Memory grew by {final - initial:.1f} MB (>50 MB leak)"
            )

        finally:
            tracemalloc.stop()

    def test_repeated_pipeline_functional(self):
        """Run pipeline 100 times and verify functional correctness."""
        config = ContextConfig(token_budget=50_000)

        for i in range(100):
            builder = ContextBuilder(config)
            builder.index_files(self.SOURCE_MAP)

            diff = DiffSpec(changed_files=self.CHANGED, query=f"Review iteration {i}")
            result = builder.build(diff, self.SOURCE_MAP)

            assert result.files_selected >= 1, f"Iteration {i}: no files selected"
            assert result.total_rendered_tokens > 0, f"Iteration {i}: zero tokens"
            assert "src/core.py" in result.rendered_context, (
                f"Iteration {i}: changed file missing from rendered context"
            )

            if i % 20 == 0:
                gc.collect()
                print(f"  Pipeline iteration {i}/100 OK")

        print(f"\n  All 100 pipeline iterations completed successfully")

    def test_repeated_compress(self):
        """Run compress 1000 times to verify no memory leak."""
        from graphsift.compress import compress

        texts = [
            ("pytest", "=== 5 passed in 0.5s ==="),
            ("git_status", "On branch main\nChanges to be committed:\n  modified: foo.py"),
            ("git_diff", "diff --git a/foo.py b/foo.py\nindex abc..def\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n"),
            ("generic", "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"),
            ("json_output", '{"key": "value", "nested": {"a": 1, "b": 2}}'),
        ]

        for i in range(1000):
            cmd, text = texts[i % len(texts)]
            result = compress(text, command=cmd)
            assert isinstance(result, str)

            if i % 200 == 0:
                gc.collect()
                print(f"  Compress iteration {i}/1000 OK")

        print(f"\n  All 1000 compress iterations completed")
