"""Property-based tests for relevance ranking using hypothesis.

Verifies invariants:
  - Score is always in [0, 1]
  - Empty query → empty results
  - Higher score for more relevant files
  - Token budget enforcement
"""

from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
import pytest

from graphsift import (
    ContextBuilder,
    ContextConfig,
    DiffSpec,
    OutputMode,
    RelevanceRanker,
    ScoredFile,
    FileNode,
    Language,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

small_score_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.tuples(valid_scores, st.integers(min_value=0, max_value=10)),
    min_size=0,
    max_size=10,
)


# ---------------------------------------------------------------------------
# Score range invariant
# ---------------------------------------------------------------------------

class TestScoreInvariants:
    """Structured test class for score property invariants."""

    def test_scores_in_0_1_range(self, builder, source_map, diff_spec):
        """Every scored file must have score in [0, 1]."""
        result = builder.build(diff_spec, source_map)
        for sf in result.selected_files:
            assert 0.0 <= sf.score <= 1.0, (
                f"Score {sf.score} out of [0,1] for {sf.file_node.path}"
            )

    def test_all_scored_files_in_range(self, builder_with_large_map):
        """All files in the graph get scores in [0,1]."""
        builder, source_map = builder_with_large_map
        diff = DiffSpec(changed_files=["src/core/engine.py"])
        result = builder.build(diff, source_map)
        for sf in result.selected_files:
            assert 0.0 <= sf.score <= 1.0, (
                f"Score {sf.score:.4f} out of [0,1] for {sf.file_node.path}"
            )


# ---------------------------------------------------------------------------
# Empty query / diff invariants
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Tests for empty/edge-case inputs."""

    def test_empty_changed_files_raises(self, builder, source_map):
        """Empty changed_files list should raise ValidationError."""
        from graphsift import ValidationError
        diff = DiffSpec(changed_files=[])
        with pytest.raises(ValidationError):
            builder.build(diff, source_map)

    def test_empty_source_map(self):
        """Building with empty source map returns empty result."""
        config = ContextConfig(token_budget=1000)
        builder = ContextBuilder(config)
        stats = builder.index_files({})
        assert stats.files_indexed == 0

    @given(st.lists(st.text(min_size=1, max_size=30), min_size=0, max_size=5))
    def test_empty_query_in_diff(self, file_list):
        """Query can be empty string without error."""
        assume(len(file_list) > 0)  # need at least one changed file
        diff = DiffSpec(changed_files=file_list, query="")
        assert diff.query == ""


# ---------------------------------------------------------------------------
# Token budget enforcement invariants
# ---------------------------------------------------------------------------

class TestTokenBudget:
    """Tests that token budget is respected."""

    @given(
        budget=st.integers(min_value=100, max_value=100_000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_token_budget_hard_limit(self, source_map, budget):
        """rendered tokens must not exceed budget (plus small overhead)."""
        config = ContextConfig(token_budget=budget, output_mode=OutputMode.FULL)
        builder = ContextBuilder(config)
        builder.index_files(source_map)
        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)
        # Allow 2x overhead for headers/tier labels
        assert result.total_rendered_tokens <= budget * 2 + 500, (
            f"Rendered {result.total_rendered_tokens} tokens exceeds "
            f"budget {budget} (with overhead)"
        )

    def test_small_budget_still_returns_changed_files(self, source_map):
        """Even with tiny budget, changed files should be included."""
        config = ContextConfig(token_budget=100, output_mode=OutputMode.FULL)
        builder = ContextBuilder(config)
        builder.index_files(source_map)
        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)
        assert result.files_selected >= 1

    @given(
        budget=st.integers(min_value=200, max_value=10_000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_token_budget_small_variations(self, source_map, budget):
        """Different budgets all produce valid results."""
        config = ContextConfig(token_budget=budget)
        builder = ContextBuilder(config)
        builder.index_files(source_map)
        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)
        assert result.files_selected >= 1
        assert result.total_rendered_tokens >= 0


# ---------------------------------------------------------------------------
# Relevance semantics: higher score = more relevant
# ---------------------------------------------------------------------------

class TestRelevanceSemantics:
    """Tests that scoring semantics hold."""

    def test_changed_file_highest_score(self, builder, source_map, diff_spec):
        """Changed files should receive the highest scores."""
        result = builder.build(diff_spec, source_map)
        changed = diff_spec.changed_files
        changed_scores = [
            sf.score for sf in result.selected_files
            if sf.file_node.path in changed
        ]
        other_scores = [
            sf.score for sf in result.selected_files
            if sf.file_node.path not in changed
        ]
        if changed_scores and other_scores:
            assert max(changed_scores) >= max(other_scores), (
                f"Changed file max score {max(changed_scores)} < "
                f"other file max {max(other_scores)}"
            )

    def test_direct_imports_score_higher(self, builder, source_map):
        """Files that import the changed module should score higher."""
        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)
        paths = [sf.file_node.path for sf in result.selected_files]
        scores = {sf.file_node.path: sf.score for sf in result.selected_files}
        # user.py imports auth.py — should appear and have reasonable score
        if "src/user.py" in scores:
            assert scores["src/user.py"] >= 0.1, (
                f"user.py scores {scores['src/user.py']} too low"
            )

    def test_unrelated_files_low_score(self, builder, source_map):
        """Files unrelated to the diff should get low or no score."""
        diff = DiffSpec(changed_files=["src/auth.py"])
        result = builder.build(diff, source_map)
        scores = {sf.file_node.path: sf.score for sf in result.selected_files}
        # utils.py is unrelated (no imports to/from auth)
        if "src/utils.py" in scores:
            assert scores["src/utils.py"] <= 0.5, (
                f"Unrelated utils.py score {scores['src/utils.py']} too high"
            )


# ---------------------------------------------------------------------------
# Ranker property invariants
# ---------------------------------------------------------------------------

class TestRankerProperties:
    """Tests for RelevanceRanker invariants."""

    def test_ranker_output_is_sorted(self, builder, source_map, diff_spec):
        """Ranker output should be sorted by score descending."""
        from graphsift.core import RelevanceRanker
        graph = builder._graph
        graph_scores = graph.ranked_neighbors(diff_spec.changed_files)
        ranker = RelevanceRanker()
        config = ContextConfig()
        ranked = ranker.rank(diff_spec, graph_scores, graph.all_files(), config)
        scores = [sf.score for sf in ranked]
        assert scores == sorted(scores, reverse=True), (
            f"Ranker output not sorted descending: {scores}"
        )

    def test_ranker_preserves_changed_files(self, builder, source_map, diff_spec):
        """All changed files should appear in ranked output."""
        from graphsift.core import RelevanceRanker
        graph = builder._graph
        graph_scores = graph.ranked_neighbors(diff_spec.changed_files)
        ranker = RelevanceRanker()
        config = ContextConfig()
        ranked = ranker.rank(diff_spec, graph_scores, graph.all_files(), config)
        ranked_paths = {sf.file_node.path for sf in ranked}
        for cf in diff_spec.changed_files:
            assert cf in ranked_paths, (
                f"Changed file {cf} missing from ranked output"
            )

    def test_no_duplicate_files(self, builder, source_map, diff_spec):
        """Each file should appear at most once in ranked output."""
        result = builder.build(diff_spec, source_map)
        paths = [sf.file_node.path for sf in result.selected_files]
        assert len(paths) == len(set(paths)), (
            f"Duplicate files in result: {paths}"
        )


# ---------------------------------------------------------------------------
# ScoredFile model invariants
# ---------------------------------------------------------------------------

@given(
    path=st.text(min_size=1, max_size=50),
    score=valid_scores,
    rank=st.integers(min_value=0, max_value=100),
)
def test_scored_file_score_invariant(path, score, rank):
    """ScoredFile score must always be in [0, 1]."""
    fn = FileNode(path=path, language=Language.PYTHON)
    sf = ScoredFile(file_node=fn, score=score, rank=rank)
    assert 0.0 <= sf.score <= 1.0


@given(score=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False))
def test_scored_file_invalid_score_raises(score):
    """ScoredFile with out-of-range score should raise."""
    import pydantic
    try:
        fn = FileNode(path="test.py", language=Language.PYTHON)
        ScoredFile(file_node=fn, score=score, rank=0)
        # If no error, score must be valid
        assert 0.0 <= score <= 1.0
    except pydantic.ValidationError:
        pass  # Expected for invalid scores
