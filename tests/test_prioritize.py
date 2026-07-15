"""Tests for the priority scoring engine (graphsift.prioritize)."""

from __future__ import annotations

import pytest

from graphsift.prioritize import PriorityScorer, ScoredFinding, PrioritizedResult


class TestPriorityScorer:
    """Tests for the multi-signal priority scorer."""

    def test_score_dead_code_empty(self):
        """Empty dead code list should return empty result."""
        scorer = PriorityScorer()
        result = scorer.score_dead_code([])
        assert result.total == 0
        assert result.entries == []
        assert result.summary == "No dead code found."

    def test_score_dead_code_returns_ranked(self):
        """Dead code entries should be scored and sorted."""
        entries = [
            {"node_id": "mod1::func_a", "file_path": "src/mod1.py",
             "name": "func_a", "kind": "function", "line_start": 10, "line_end": 25},
            {"node_id": "mod2::ClassB", "file_path": "src/mod2.py",
             "name": "ClassB", "kind": "class", "line_start": 1, "line_end": 200},
            {"node_id": "mod3::tiny", "file_path": "src/mod3.py",
             "name": "tiny", "kind": "function", "line_start": 5, "line_end": 7},
        ]
        scorer = PriorityScorer()
        result = scorer.score_dead_code(entries)
        assert result.total == 3
        assert len(result.entries) == 3  # below cutoff
        # Should be sorted by score descending
        scores = [e.score for e in result.entries]
        assert scores == sorted(scores, reverse=True)
        assert "tiers" in result.__dict__
        assert isinstance(result.summary, str)

    def test_scored_finding_to_dict(self):
        """ScoredFinding should serialize properly."""
        entry = {"node_id": "x::y", "file_path": "x.py", "name": "y", "kind": "function"}
        sf = ScoredFinding(entry=entry, score=0.85, tier="critical", signals={"impact": 0.9})
        d = sf.to_dict()
        assert d["_priority_score"] == 0.85
        assert d["_priority_tier"] == "critical"
        assert d["name"] == "y"

    def test_prioritized_result_to_dict(self):
        """PrioritizedResult should serialize properly."""
        entries = [
            ScoredFinding({"name": "a"}, 0.9, "critical", {}),
            ScoredFinding({"name": "b"}, 0.5, "medium", {}),
        ]
        result = PrioritizedResult(
            entries=entries,
            tiers={"critical": 1, "medium": 1},
            total=2,
            summary="2 findings",
            truncated=False,
        )
        d = result.to_dict()
        assert d["total"] == 2
        assert d["tiers"]["critical"] == 1
        assert len(d["entries"]) == 2
        assert d["entries"][0]["_priority_tier"] == "critical"

    def test_truncation_with_many_entries(self):
        """Large result sets should be truncated to show critical/high first."""
        entries = []
        for i in range(100):
            # Large functions (120+ lines) produce lower scores (medium/low)
            entries.append({
                "node_id": f"m::f{i}", "file_path": "x.py",
                "name": f"func{i}", "kind": "function",
                "line_start": i, "line_end": i + 120,
            })
        scorer = PriorityScorer()
        result = scorer.score_dead_code(entries)
        assert result.total == 100
        # With 120-line functions, scores should be low enough to trigger truncation
        assert len(result.entries) <= 55, (
            f"Expected truncation to ~50, got {len(result.entries)}"
        )

    def test_tier_thresholds(self):
        """Tier computation should match expected ranges."""
        scorer = PriorityScorer()
        assert scorer._tier_for(0.90) == "critical"
        assert scorer._tier_for(0.80) == "critical"
        assert scorer._tier_for(0.79) == "high"
        assert scorer._tier_for(0.60) == "high"
        assert scorer._tier_for(0.59) == "medium"
        assert scorer._tier_for(0.35) == "medium"
        assert scorer._tier_for(0.34) == "low"
        assert scorer._tier_for(0.0) == "low"

    def test_is_test_file(self):
        """Test file detection heuristic."""
        scorer = PriorityScorer()
        assert scorer._is_test_file("tests/test_auth.py") is True
        assert scorer._is_test_file("src/test_utils.py") is True
        assert scorer._is_test_file("src/utils.py") is False

    def test_effort_inverse_small(self):
        """Small functions should score highly on effort_inverse."""
        scorer = PriorityScorer()
        _, signals = scorer._score_entry(
            {"name": "tiny", "line_start": 10, "line_end": 12},
            "dead_code",
        )
        assert signals["effort_inverse"] == 1.0

    def test_effort_inverse_large(self):
        """Large functions should score low on effort_inverse."""
        scorer = PriorityScorer()
        _, signals = scorer._score_entry(
            {"name": "huge", "line_start": 10, "line_end": 400},
            "dead_code",
        )
        assert signals["effort_inverse"] < 0.5

    def test_score_fix_suggestions_empty(self):
        """Empty suggestions should return empty result."""
        scorer = PriorityScorer()
        result = scorer.score_fix_suggestions([])
        assert result.total == 0
        assert result.summary == "No fix suggestions."

    def test_confidence_from_severity(self):
        """Severity should map to reasonable confidence."""
        scorer = PriorityScorer()
        _, sig1 = scorer._score_entry(
            {"name": "err", "severity": "error"}, "import"
        )
        _, sig2 = scorer._score_entry(
            {"name": "warn", "severity": "warning"}, "import"
        )
        _, sig3 = scorer._score_entry(
            {"name": "info", "severity": "info"}, "import"
        )
        assert sig1["confidence"] >= sig2["confidence"]
        assert sig2["confidence"] >= sig3["confidence"]

    def test_confidence_direct_value(self):
        """Explicit confidence value should be used."""
        scorer = PriorityScorer()
        _, sig = scorer._score_entry(
            {"name": "x", "confidence": 0.75}, "dead_code"
        )
        assert sig["confidence"] == 0.75

    def test_todo_freshness_boost(self):
        """Files with TODO markers should get a freshness boost."""
        scorer = PriorityScorer(source_map={
            "src/busy.py": "def foo():\n    # TODO: fix this\n    pass",
        })
        _, sig1 = scorer._score_entry(
            {"name": "x", "file_path": "src/busy.py"}, "dead_code"
        )
        _, sig2 = scorer._score_entry(
            {"name": "y", "file_path": "src/clean.py"}, "dead_code"
        )
        assert sig1["freshness"] >= sig2["freshness"]

    def test_private_symbol_lower_risk(self):
        """Private symbols (_name) should have higher risk_inverse (safer)."""
        scorer = PriorityScorer()
        _, sig1 = scorer._score_entry(
            {"name": "_internal"}, "dead_code"
        )
        _, sig2 = scorer._score_entry(
            {"name": "public_api"}, "dead_code"
        )
        assert sig1["risk_inverse"] >= sig2["risk_inverse"]
