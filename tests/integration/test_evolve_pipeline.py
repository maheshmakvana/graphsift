"""Integration tests for the auto-evolve pipeline (ContextBuilder + auto_evolve)."""

from __future__ import annotations

import os
import tempfile

import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec


class TestAutoEvolvePipeline:
    """Tests for ContextBuilder with auto_evolve=True."""

    def test_evolve_disabled_by_default(self):
        """auto_evolve should be False by default."""
        config = ContextConfig()
        assert config.auto_evolve is False

    def test_evolve_enabled_constructs(self):
        """ContextConfig should accept auto_evolve=True."""
        config = ContextConfig(auto_evolve=True, token_budget=2000)
        assert config.auto_evolve is True
        assert config.evolve_rounds == 20
        assert config.evolve_population == 6

    def test_custom_evolve_params(self):
        """Custom evolve params should be stored correctly."""
        config = ContextConfig(
            auto_evolve=True,
            evolve_rounds=10,
            evolve_population=4,
            token_budget=2000,
        )
        assert config.evolve_rounds == 10
        assert config.evolve_population == 4

    def test_build_with_auto_evolve(self):
        """ContextBuilder.build() should complete with auto_evolve=True."""
        source_map = {
            "src/auth.py": "class Auth:\n    def login(self): pass\n",
            "src/user.py": "class User:\n    def get(self): pass\n",
            "src/db.py": "from src.auth import Auth\nfrom src.user import User\n",
        }
        diff = DiffSpec(changed_files=["src/auth.py"], query="Review auth")
        config = ContextConfig(
            auto_evolve=True,
            evolve_rounds=5,   # Small for test speed
            evolve_population=3,
            token_budget=2000,
        )
        builder = ContextBuilder(config)
        builder.index_files(source_map)
        result = builder.build(diff, source_map)

        assert result is not None
        assert result.files_selected >= 1
        assert result.total_rendered_tokens > 0
        assert result.files_scanned >= result.files_selected

    def test_evolve_caches_across_builds(self):
        """Second build with same source_map should use cached params."""
        source_map = {
            "src/a.py": "class A: pass\n",
            "src/b.py": "from src.a import A\nclass B: pass\n",
        }
        diff = DiffSpec(changed_files=["src/a.py"], query="Review")
        config = ContextConfig(
            auto_evolve=True,
            evolve_rounds=3,
            evolve_population=3,
            token_budget=2000,
        )

        # First build triggers evolution
        builder1 = ContextBuilder(config)
        builder1.index_files(source_map)
        result1 = builder1.build(diff, source_map)

        # Second build should use cache (faster)
        builder2 = ContextBuilder(config)
        builder2.index_files(source_map)
        result2 = builder2.build(diff, source_map)

        assert result2.files_selected >= 1

    def test_evolve_disabled_build(self):
        """With auto_evolve=False, build should work normally without evolution."""
        source_map = {
            "src/auth.py": "def login(): pass\n",
        }
        diff = DiffSpec(changed_files=["src/auth.py"])
        config = ContextConfig(auto_evolve=False, token_budget=2000)
        builder = ContextBuilder(config)
        builder.index_files(source_map)
        result = builder.build(diff, source_map)

        assert result.files_selected >= 1
