"""Stress test: build context with 500 changed files."""

import time
import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec, OutputMode

pytestmark = [pytest.mark.stress, pytest.mark.slow]


def _generate_large_source_map(file_count: int) -> dict[str, str]:
    """Generate *file_count* source files with cross-references."""
    source_map = {}
    for i in range(file_count):
        mod = f"module_{i:05d}"
        file_path = f"src/{mod}.py"

        # Each file imports from 5 neighboring modules (when available)
        neighbor_imports = []
        for j in range(max(0, i - 5), i):
            neighbor_imports.append(f"from module_{j:05d} import func_{j}")
        for j in range(i + 1, min(file_count, i + 6)):
            neighbor_imports.append(f"from module_{j:05d} import func_{j}")

        source = f'''"""{mod}."""
{chr(10).join(neighbor_imports[:10])}

def func_{i}(x: int = 0) -> int:
    """Function {i}."""
    return x + {i}

class Class_{i}:
    """Class {i}."""
    def method_{i}(self) -> str:
        return "result_{i}"
'''
        source_map[file_path] = source
    return source_map


class TestLargeDiff:
    """Stress test for building context with many changed files."""

    @pytest.mark.parametrize("changed_count", [10, 50, 100, 500])
    def test_large_diff_build(self, changed_count):
        """Build context with varying numbers of changed files."""
        total_files = max(changed_count * 2, 1000)
        source_map = _generate_large_source_map(total_files)

        config = ContextConfig(
            token_budget=500_000,
            output_mode=OutputMode.SMART,
            hot_threshold=0.8,
            warm_threshold=0.25,
        )
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        # Pick the first N files as changed
        changed = [f"src/module_{i:05d}.py" for i in range(changed_count)]
        diff = DiffSpec(changed_files=changed, query="Review all changes")

        start = time.perf_counter()
        result = builder.build(diff, source_map)
        elapsed = time.perf_counter() - start

        print(f"\n  {changed_count} changed files: "
              f"selected {result.files_selected}/{result.files_scanned} files, "
              f"{result.total_rendered_tokens:,} tokens, "
              f"in {elapsed:.2f}s")

        assert result.files_selected >= changed_count
        assert result.total_rendered_tokens > 0
        assert result.reduction_ratio >= 0.0

        # Should complete within reasonable time
        # Allow up to 30s for 500 files
        max_time = min(5 + changed_count * 0.05, 60)
        assert elapsed < max_time, (
            f"Build took {elapsed:.1f}s, max allowed {max_time:.1f}s"
        )

    def test_all_files_changed(self):
        """Stress test with ALL files as changed."""
        file_count = 200
        source_map = _generate_large_source_map(file_count)

        builder = ContextBuilder(ContextConfig(token_budget=500_000))
        builder.index_files(source_map)

        changed = list(source_map.keys())
        diff = DiffSpec(changed_files=changed)

        start = time.perf_counter()
        result = builder.build(diff, source_map)
        elapsed = time.perf_counter() - start

        print(f"\n  All {file_count} files changed: "
              f"selected {result.files_selected}, "
              f"tokens={result.total_rendered_tokens:,}, "
              f"in {elapsed:.2f}s")

        assert result.files_selected >= 1

    def test_large_diff_rendering_quality(self):
        """Large diff output should be well-formed."""
        source_map = _generate_large_source_map(300)
        changed = [f"src/module_{i:05d}.py" for i in range(30)]

        config = ContextConfig(
            token_budget=200_000,
            output_mode=OutputMode.SMART,
            cache_aware=True,
        )
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(changed_files=changed)
        result = builder.build(diff, source_map)

        context = result.rendered_context
        # Should contain tier labels
        assert "[HOT]" in context or "[WARM]" in context
        # Should not be empty
        assert len(context) > 0
        # Changed files should appear in context
        for cf in changed[:3]:
            assert cf in context, f"Changed file {cf} missing from context"
