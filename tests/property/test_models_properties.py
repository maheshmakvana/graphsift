"""Property-based tests for Pydantic models using hypothesis.

Verifies:
  - Pydantic model serialization roundtrips
  - Field validations (ge, le, regex patterns)
  - Enum invariants
"""

import json
from hypothesis import given, assume
from hypothesis import strategies as st
import pytest

from graphsift.models import (
    ContextConfig,
    ContextResult,
    DiffSpec,
    EdgeKind,
    FileNode,
    GraphEdge,
    GraphNode,
    IndexStats,
    Language,
    NodeKind,
    OutputMode,
    ScoredFile,
    DepthTier,
    TierLevel,
    CycleInfo,
    CycleReport,
    DeadCodeInfo,
    DeadCodeReport,
    FixSuggestion,
    FixSeverity,
    FixReport,
)


# ---------------------------------------------------------------------------
# Strategies for model fields
# ---------------------------------------------------------------------------

# Language enum strategy
language_strategy = st.sampled_from(list(Language))

# NodeKind enum strategy
node_kind_strategy = st.sampled_from(list(NodeKind))

# EdgeKind enum strategy
edge_kind_strategy = st.sampled_from(list(EdgeKind))

# OutputMode enum strategy
output_mode_strategy = st.sampled_from(list(OutputMode))

# DepthTier enum strategy
depth_tier_strategy = st.sampled_from(list(DepthTier))

# FixSeverity enum strategy
fix_severity_strategy = st.sampled_from(list(FixSeverity))

# Non-empty text
non_empty_text = st.text(min_size=1, max_size=100)

# File path-like strings
file_path_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P"), whitelist_characters="./_-"),
    min_size=1,
    max_size=100,
)


# ---------------------------------------------------------------------------
# Model serialization roundtrips
# ---------------------------------------------------------------------------

class TestSerializationRoundtrips:
    """Every model should survive JSON serialization roundtrip."""

    def test_file_node_roundtrip(self):
        """FileNode serialize → deserialize → identical."""
        fn = FileNode(
            path="src/main.py",
            language=Language.PYTHON,
            size_bytes=1024,
            line_count=50,
            sha256="abc123",
            token_estimate=500,
        )
        data = fn.model_dump()
        fn2 = FileNode.model_validate(data)
        assert fn == fn2

    def test_graph_node_roundtrip(self):
        """GraphNode serialize → deserialize → identical."""
        gn = GraphNode(
            node_id="src/main.py::MyClass",
            file_path="src/main.py",
            kind=NodeKind.CLASS,
            name="MyClass",
            qualified_name="MyClass",
            line_start=1,
            line_end=50,
            language=Language.PYTHON,
            signature="class MyClass:",
            decorators=["@dataclass"],
        )
        data = gn.model_dump()
        gn2 = GraphNode.model_validate(data)
        assert gn == gn2

    def test_graph_edge_roundtrip(self):
        """GraphEdge serialize → deserialize → identical."""
        ge = GraphEdge(
            source_id="a.py::foo",
            target_id="b.py::bar",
            kind=EdgeKind.CALLS,
            weight=1.5,
        )
        data = ge.model_dump()
        ge2 = GraphEdge.model_validate(data)
        assert ge == ge2

    def test_scored_file_roundtrip(self):
        """ScoredFile serialize → deserialize → identical."""
        fn = FileNode(path="test.py", language=Language.PYTHON)
        sf = ScoredFile(file_node=fn, score=0.75, rank=1)
        data = sf.model_dump()
        sf2 = ScoredFile.model_validate(data)
        assert sf == sf2

    def test_diff_spec_roundtrip(self):
        """DiffSpec serialize → deserialize → identical."""
        ds = DiffSpec(
            changed_files=["src/main.py", "src/utils.py"],
            diff_text="--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-foo\n+bar\n",
            commit_message="fix: update foo to bar",
            query="What changed?",
        )
        data = ds.model_dump()
        ds2 = DiffSpec.model_validate(data)
        assert ds == ds2

    def test_context_config_roundtrip(self):
        """ContextConfig serialize → deserialize → identical."""
        cc = ContextConfig(
            token_budget=50_000,
            max_depth=3,
            output_mode=OutputMode.SMART,
        )
        data = cc.model_dump()
        cc2 = ContextConfig.model_validate(data)
        assert cc == cc2

    def test_context_config_defaults(self):
        """ContextConfig should have sensible defaults."""
        cc = ContextConfig()
        assert cc.token_budget == 80_000
        assert cc.max_depth == 4
        assert cc.min_score == 0.1
        assert cc.output_mode == OutputMode.SMART
        assert cc.hot_threshold == 0.8
        assert cc.warm_threshold == 0.25

    def test_index_stats_roundtrip(self):
        """IndexStats serialize → deserialize → identical."""
        stats = IndexStats(
            files_indexed=10,
            files_skipped=2,
            symbols_extracted=45,
            edges_created=30,
            duration_ms=150.5,
            languages={"python": 8, "javascript": 2},
        )
        data = stats.model_dump()
        stats2 = IndexStats.model_validate(data)
        assert stats == stats2

    def test_context_result_roundtrip(self):
        """ContextResult serialize → deserialize (simplified)."""
        ds = DiffSpec(changed_files=["src/main.py"])
        fn = FileNode(path="src/main.py", language=Language.PYTHON)
        sf = ScoredFile(file_node=fn, score=1.0, rank=0)
        cr = ContextResult(
            diff_spec=ds,
            selected_files=[sf],
            rendered_context="# src/main.py\nprint('hello')",
            cache_breakpoints=0,
            total_original_tokens=1000,
            total_rendered_tokens=200,
            reduction_ratio=0.8,
            files_scanned=5,
            files_selected=1,
        )
        data = cr.model_dump()
        cr2 = ContextResult.model_validate(data)
        assert cr.diff_spec == cr2.diff_spec
        assert len(cr.selected_files) == len(cr2.selected_files)


# ---------------------------------------------------------------------------
# Field validation invariants
# ---------------------------------------------------------------------------

class TestFieldValidations:
    """Tests that Pydantic field constraints are enforced."""

    def test_context_config_token_budget_min(self):
        """token_budget must be >= 100."""
        with pytest.raises(Exception):
            ContextConfig(token_budget=50)

    def test_context_config_max_depth_bounds(self):
        """max_depth must be between 1 and 10."""
        with pytest.raises(Exception):
            ContextConfig(max_depth=0)
        with pytest.raises(Exception):
            ContextConfig(max_depth=11)
        # Valid values
        cc = ContextConfig(max_depth=5)
        assert cc.max_depth == 5

    def test_context_config_hot_threshold_range(self):
        """hot_threshold must be in [0, 1]."""
        with pytest.raises(Exception):
            ContextConfig(hot_threshold=-0.1)
        with pytest.raises(Exception):
            ContextConfig(hot_threshold=1.1)
        cc = ContextConfig(hot_threshold=0.5)
        assert cc.hot_threshold == 0.5

    def test_context_config_warm_threshold_range(self):
        """warm_threshold must be in [0, 1]."""
        with pytest.raises(Exception):
            ContextConfig(warm_threshold=-0.1)
        with pytest.raises(Exception):
            ContextConfig(warm_threshold=1.1)

    def test_context_config_trimming_context_lines_bounds(self):
        """trimming_context_lines must be in [0, 100]."""
        with pytest.raises(Exception):
            ContextConfig(trimming_context_lines=-1)
        with pytest.raises(Exception):
            ContextConfig(trimming_context_lines=101)

    def test_context_config_cache_ttl_bounds(self):
        """cache_ttl_days must be in [1, 365]."""
        with pytest.raises(Exception):
            ContextConfig(cache_ttl_days=0)
        with pytest.raises(Exception):
            ContextConfig(cache_ttl_days=366)

    def test_graph_edge_weight_range(self):
        """Edge weight must be in [0, 10]."""
        with pytest.raises(Exception):
            GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS, weight=-0.1)
        with pytest.raises(Exception):
            GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS, weight=10.1)
        ge = GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS, weight=5.0)
        assert ge.weight == 5.0

    def test_scored_file_score_range(self):
        """ScoredFile score must be in [0, 1]."""
        fn = FileNode(path="test.py", language=Language.PYTHON)
        with pytest.raises(Exception):
            ScoredFile(file_node=fn, score=-0.01, rank=0)
        with pytest.raises(Exception):
            ScoredFile(file_node=fn, score=1.01, rank=0)
        sf = ScoredFile(file_node=fn, score=0.5, rank=0)
        assert sf.score == 0.5


# ---------------------------------------------------------------------------
# Enum uniqueness / invariants
# ---------------------------------------------------------------------------

class TestEnumInvariants:
    """Tests that enum values have expected properties."""

    def test_all_languages_have_names(self):
        """Every Language enum member has a non-empty value."""
        for lang in Language:
            assert lang.value, f"Language {lang} has empty value"

    def test_all_node_kinds_have_names(self):
        """Every NodeKind enum member has a non-empty value."""
        for nk in NodeKind:
            assert nk.value, f"NodeKind {nk} has empty value"

    def test_all_edge_kinds_have_names(self):
        """Every EdgeKind enum member has a non-empty value."""
        for ek in EdgeKind:
            assert ek.value, f"EdgeKind {ek} has empty value"

    def test_output_mode_values(self):
        """OutputMode should have expected values."""
        assert OutputMode.FULL.value == "full"
        assert OutputMode.SIGNATURES.value == "signatures"
        assert OutputMode.COMPRESSED.value == "compressed"
        assert OutputMode.SMART.value == "smart"

    def test_depth_tier_values(self):
        """DepthTier should have expected values."""
        assert DepthTier.PLANNING.value == "planning"
        assert DepthTier.EXPLORATION.value == "exploration"
        assert DepthTier.EXECUTION.value == "execution"


# ---------------------------------------------------------------------------
# Hypothesis-driven model generation
# ---------------------------------------------------------------------------

@given(
    path=file_path_strategy,
    language=language_strategy,
    size_bytes=st.integers(min_value=0, max_value=10_000_000),
    line_count=st.integers(min_value=0, max_value=100_000),
)
def test_file_node_arbitrary_fields(path, language, size_bytes, line_count):
    """FileNode can be constructed with arbitrary valid fields."""
    fn = FileNode(
        path=path,
        language=language,
        size_bytes=size_bytes,
        line_count=line_count,
    )
    assert fn.path == path
    assert fn.language == language
    assert fn.size_bytes == size_bytes
    assert fn.line_count == line_count


@given(
    node_id=non_empty_text,
    file_path=file_path_strategy,
    kind=node_kind_strategy,
    name=non_empty_text,
    qualified_name=non_empty_text,
)
def test_graph_node_arbitrary_fields(node_id, file_path, kind, name, qualified_name):
    """GraphNode can be constructed with arbitrary valid fields."""
    gn = GraphNode(
        node_id=node_id,
        file_path=file_path,
        kind=kind,
        name=name,
        qualified_name=qualified_name,
    )
    assert gn.node_id == node_id
    assert gn.kind == kind


@given(
    source_id=non_empty_text,
    target_id=non_empty_text,
    kind=edge_kind_strategy,
    weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_graph_edge_arbitrary_fields(source_id, target_id, kind, weight):
    """GraphEdge can be constructed with arbitrary valid fields."""
    assume(source_id != target_id)  # No self-loops
    ge = GraphEdge(source_id=source_id, target_id=target_id, kind=kind, weight=weight)
    assert ge.source_id == source_id
    assert ge.target_id == target_id
    assert ge.kind == kind
    assert ge.weight == weight


# ---------------------------------------------------------------------------
# DiffSpec invariants
# ---------------------------------------------------------------------------

@given(
    changed_files=st.lists(file_path_strategy, min_size=1, max_size=10),
    diff_text=st.text(max_size=200),
    commit_message=st.text(max_size=100),
    query=st.text(max_size=200),
)
def test_diff_spec_arbitrary(changed_files, diff_text, commit_message, query):
    """DiffSpec can be constructed with arbitrary values."""
    ds = DiffSpec(
        changed_files=changed_files,
        diff_text=diff_text,
        commit_message=commit_message,
        query=query,
    )
    assert len(ds.changed_files) == len(changed_files)
    assert ds.diff_text == diff_text
