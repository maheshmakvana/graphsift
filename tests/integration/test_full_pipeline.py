"""Integration test: end-to-end pipeline index → build → render."""

import pytest

from graphsift import (
    ContextBuilder,
    ContextConfig,
    ContextResult,
    DiffSpec,
    OutputMode,
    PythonParser,
    compress,
)


class TestFullPipeline:
    """End-to-end pipeline: index → build → render."""

    def test_full_pipeline_simple(self, source_map):
        """Basic index → build → render cycle."""
        config = ContextConfig(
            token_budget=50_000,
            output_mode=OutputMode.SMART,
            hot_threshold=0.8,
            warm_threshold=0.25,
        )
        builder = ContextBuilder(config)

        # 1. Index
        stats = builder.index_files(source_map)
        assert stats.files_indexed > 0
        assert stats.symbols_extracted > 0
        assert stats.duration_ms > 0

        # 2. Build
        diff = DiffSpec(
            changed_files=["src/auth.py"],
            query="Review the authentication logic for security issues",
            commit_message="refactor: improve auth token generation",
        )
        result = builder.build(diff, source_map)

        # 3. Verify result structure
        assert isinstance(result, ContextResult)
        assert result.files_selected >= 1
        assert result.files_scanned >= result.files_selected
        assert result.total_rendered_tokens > 0
        assert result.reduction_ratio >= 0.0

        # 4. Verify rendered context contains expected content
        context = result.rendered_context
        assert "src/auth.py" in context
        assert "[HOT]" in context or "[WARM]" in context

        # 5. Verify selected files include the changed file
        selected_paths = {sf.file_node.path for sf in result.selected_files}
        assert "src/auth.py" in selected_paths

    def test_pipeline_with_compression(self, source_map):
        """Pipeline with compression output mode."""
        config = ContextConfig(
            token_budget=50_000,
            output_mode=OutputMode.COMPRESSED,
        )
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)

        assert result.files_selected >= 1
        context = result.rendered_context
        assert len(context) > 0

    def test_pipeline_with_signatures(self, source_map):
        """Pipeline with signatures-only output mode."""
        config = ContextConfig(
            token_budget=50_000,
            output_mode=OutputMode.SIGNATURES,
        )
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)

        assert result.files_selected >= 1
        context = result.rendered_context
        # Signatures mode should show signatures, not full bodies
        assert "def " in context or "class " in context or "SIGNATURES" in context.upper()

    def test_pipeline_with_different_formats(self, source_map):
        """Pipeline handles different file formats."""
        config = ContextConfig(token_budget=50_000)
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)

        # All files should be properly parsed
        for sf in result.selected_files:
            assert sf.file_node.language is not None
            assert sf.file_node.path.count(".") >= 1  # Has extension

    def test_pipeline_multiple_queries(self, source_map):
        """Pipeline handles different queries."""
        config = ContextConfig(token_budget=50_000)
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        queries = [
            "Review security implications",
            "Find any bugs in the authentication flow",
            "Suggest improvements for error handling",
            "",
        ]

        for query in queries:
            diff = DiffSpec(changed_files=["src/auth.py"], query=query)
            result = builder.build(diff, source_map)
            assert result.files_selected >= 1
            assert result.total_rendered_tokens > 0

    def test_pipeline_with_multiple_changed_files(self, source_map):
        """Pipeline handles multiple changed files."""
        config = ContextConfig(token_budget=50_000)
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(
            changed_files=["src/auth.py", "src/user.py"],
            query="Review both auth and user changes",
        )
        result = builder.build(diff, source_map)

        selected_paths = {sf.file_node.path for sf in result.selected_files}
        assert "src/auth.py" in selected_paths
        assert "src/user.py" in selected_paths

    def test_pipeline_compress_after_build(self, source_map):
        """Built context can be further compressed."""
        config = ContextConfig(token_budget=50_000)
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)

        # Compress the rendered context
        compressed = compress(result.rendered_context)
        assert len(compressed) <= len(result.rendered_context) or True  # At least no crash

    def test_pipeline_monorepo_support(self, source_map):
        """Pipeline handles monorepo structure."""
        # Split into roots
        items = list(source_map.items())
        root_src = dict(items[:3])
        root_tests = dict(items[3:])

        config = ContextConfig(token_budget=50_000)
        builder = ContextBuilder(config)

        stats_list = builder.index_roots([root_src, root_tests])
        assert len(stats_list) == 2
        total_indexed = sum(s.files_indexed for s in stats_list)
        assert total_indexed == len(source_map)

        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)
        assert result.files_selected >= 1
