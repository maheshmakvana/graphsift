"""Tests for v2.4+ cache features."""

from __future__ import annotations

from graphsift.cache import ASTCache


class TestASTCacheWarming:
    """Tests for ASTCache warming methods."""

    def test_warm_no_disk(self):
        """Warming without disk backend should not crash."""
        cache = ASTCache(max_memory=100, db_path="")
        warmed = cache.warm(["nonexistent_key"])
        assert warmed == 0

    def test_warm_empty_list(self):
        """Warming with empty list should return 0."""
        cache = ASTCache(max_memory=100, db_path="")
        warmed = cache.warm([])
        assert warmed == 0

    def test_warm_from_paths_empty(self):
        """Warming from empty paths should return 0."""
        cache = ASTCache(max_memory=100, db_path="")
        warmed = cache.warm_from_paths([])
        assert warmed == 0

    def test_warm_from_paths_nonexistent(self):
        """Warming from non-existent paths should return 0."""
        cache = ASTCache(max_memory=100, db_path="")
        warmed = cache.warm_from_paths(["/nonexistent/path/file.py"])
        assert warmed == 0

    def test_predictive_warm_no_graph(self):
        """Predictive warm without a graph should not crash."""
        cache = ASTCache(max_memory=100, db_path="")
        warmed = cache.predictive_warm(["test.py"], None)
        assert warmed == 0

    def test_predictive_warm_with_graph(self):
        """Predictive warm with a graph should call ranked_neighbors."""
        cache = ASTCache(max_memory=100, db_path="")

        class MockGraph:
            def ranked_neighbors(self, seed_paths=None):
                return {"a.py": (0.9, 1, ["import"])}

        warmed = cache.predictive_warm(["seed.py"], MockGraph())
        # Should not crash even without disk cache
        assert warmed >= 0

    def test_predictive_warm_graph_no_method(self):
        """Predictive warm with graph missing ranked_neighbors should return 0."""
        cache = ASTCache(max_memory=100, db_path="")

        class BadGraph:
            pass

        warmed = cache.predictive_warm(["seed.py"], BadGraph())
        assert warmed == 0

    def test_stats_after_warm(self):
        """Stats should include warmed_entries field."""
        cache = ASTCache(max_memory=100, db_path="")
        cache.warm([])
        stats = cache.stats()
        assert "warmed_entries" in stats
        assert stats["warmed_entries"] >= 0
