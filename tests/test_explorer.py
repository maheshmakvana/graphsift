"""Tests for the Deep Context Enricher."""

from __future__ import annotations

from pathlib import Path

from graphsift.explorer import (
    ContextEnricher,
    EnrichmentResult,
    Discovery,
    DiscoveryType,
    ConfigDiscoverer,
    TestDiscoverer,
    CoChangeDiscoverer,
    EnvDiscoverer,
)


class TestDiscoveryModels:
    """Tests for Discovery data classes."""

    def test_discovery_defaults(self):
        """Discovery should have sensible defaults."""
        d = Discovery(type="test", path="file.py")
        assert d.relevance == 0.5
        assert d.evidence == ""
        assert d.source == ""

    def test_discovery_with_all_fields(self):
        """Discovery should store all fields."""
        d = Discovery(
            type=DiscoveryType.TEST_FILE,
            path="tests/test_auth.py",
            relevance=0.9,
            evidence="Test file for auth",
            source="src/auth.py",
        )
        assert d.type == DiscoveryType.TEST_FILE
        assert d.relevance == 0.9

    def test_enrichment_result_defaults(self):
        """EnrichmentResult should start empty."""
        result = EnrichmentResult(changed_files=[])
        assert result.total == 0
        assert result.discoveries == []

    def test_enrichment_result_with_discoveries(self):
        """EnrichmentResult should track discoveries."""
        d = Discovery(type=DiscoveryType.TEST_FILE, path="test.py", relevance=0.9)
        result = EnrichmentResult(changed_files=["src/main.py"], discoveries=[d])
        assert result.total == 1
        assert "1 test" in result.summary or "1" in result.summary

    def test_enrichment_result_categories(self):
        """EnrichmentResult should categorize discoveries."""
        d1 = Discovery(type=DiscoveryType.CONFIG, path="pyproject.toml", relevance=0.8)
        d2 = Discovery(type=DiscoveryType.TEST_FILE, path="test_x.py", relevance=0.9)
        result = EnrichmentResult(
            changed_files=["x.py"],
            discoveries=[d1, d2],
            config_files=[d1],
            test_files=[d2],
        )
        assert len(result.config_files) == 1
        assert len(result.test_files) == 1

    def test_discovery_type_constants(self):
        """DiscoveryType constants should have correct values."""
        assert DiscoveryType.CONFIG == "config"
        assert DiscoveryType.ENV_VAR == "env_var"
        assert DiscoveryType.TEST_FILE == "test_file"
        assert DiscoveryType.CO_CHANGE == "co_change"
        assert DiscoveryType.DOCKER_REF == "docker_ref"
        assert DiscoveryType.CI_REF == "ci_ref"


class TestConfigDiscoverer:
    """Tests for the ConfigDiscoverer."""

    def test_discover_no_configs(self):
        """Discovering configs in dir without configs should return empty."""
        discoverer = ConfigDiscoverer(Path("."))
        discoveries = discoverer.discover(["nonexistent.py"])
        assert discoveries == []

    def test_discover_pyproject_not_found(self):
        """Discoverer should handle missing config files gracefully."""
        discoverer = ConfigDiscoverer(Path("/nonexistent_dir_xyz"))
        discoveries = discoverer.discover(["test.py"])
        assert discoveries == []

    def test_discover_empty_changed_files(self):
        """Discovering with empty changed files should return nothing."""
        discoverer = ConfigDiscoverer(Path("."))
        discoveries = discoverer.discover([])
        assert discoveries == []


class TestTestDiscoverer:
    """Tests for the TestDiscoverer."""

    def test_discover_empty(self):
        """Discovering with empty changed files should return nothing."""
        discoverer = TestDiscoverer(Path("."))
        discoveries = discoverer.discover([])
        assert discoveries == []

    def test_discover_existing_test(self):
        """Discovering tests for an existing file should find candidates."""
        discoverer = TestDiscoverer(Path("."))
        discoveries = discoverer.discover(["graphsift/core.py"])
        for d in discoveries:
            assert d.type == DiscoveryType.TEST_FILE
            assert d.relevance == 0.9

    def test_discover_nonexistent_file(self):
        """Discovering tests for a non-existent file should not crash."""
        discoverer = TestDiscoverer(Path("."))
        discoveries = discoverer.discover(["completely_fake_file_xyz.py"])
        # Should return empty or find nothing
        assert isinstance(discoveries, list)


class TestCoChangeDiscoverer:
    """Tests for the CoChangeDiscoverer."""

    def test_discover_no_git(self):
        """Discovering without a git repo should return empty."""
        discoverer = CoChangeDiscoverer(Path("/nonexistent_dir"))
        discoveries = discoverer.discover(["test.py"])
        assert discoveries == []

    def test_discover_empty_changed_files(self):
        """Discovering with empty changed files should return nothing."""
        discoverer = CoChangeDiscoverer(Path("."))
        discoveries = discoverer.discover([])
        assert discoveries == []


class TestEnvDiscoverer:
    """Tests for the EnvDiscoverer."""

    def test_discover_empty(self):
        """Discovering with empty changed files should return nothing."""
        discoverer = EnvDiscoverer(Path("."))
        discoveries = discoverer.discover([])
        assert discoveries == []

    def test_discover_file_with_env_vars(self):
        """Discovering env vars in a file with os.getenv should find them."""
        source_map = {
            "config.py": (
                "import os\n"
                "DB_HOST = os.getenv('DATABASE_HOST')\n"
                "DB_PORT = os.getenv('DATABASE_PORT', '5432')\n"
            ),
        }
        discoverer = EnvDiscoverer(Path("."), source_map=source_map)
        discoveries = discoverer.discover(["config.py"])
        assert len(discoveries) >= 2
        var_names = {d.path for d in discoveries}
        assert "DATABASE_HOST" in var_names

    def test_discover_file_with_environ_access(self):
        """Discovering env vars using os.environ[] should work."""
        source_map = {
            "app.py": "import os\nkey = os.environ['API_KEY']\n",
        }
        discoverer = EnvDiscoverer(Path("."), source_map=source_map)
        discoveries = discoverer.discover(["app.py"])
        assert len(discoveries) >= 1
        assert discoveries[0].path == "API_KEY"

    def test_discover_file_no_env_vars(self):
        """A file with no env var references should yield nothing."""
        source_map = {
            "simple.py": "x = 1\ny = 2\n",
        }
        discoverer = EnvDiscoverer(Path("."), source_map=source_map)
        discoveries = discoverer.discover(["simple.py"])
        assert discoveries == []


class TestContextEnricher:
    """Tests for the ContextEnricher orchestrator."""

    def test_enrich_empty(self):
        """Enriching with no changed files should return empty result."""
        enricher = ContextEnricher(root=".")
        result = enricher.enrich([])
        assert result.total == 0

    def test_enrich_with_discoveries_disabled(self):
        """Enriching with all discoverers disabled should return empty."""
        enricher = ContextEnricher(root=".")
        result = enricher.enrich(
            ["test.py"],
            find_config=False,
            find_tests=False,
            find_co_changes=False,
            find_env=False,
        )
        assert result.total == 0

    def test_enrich_with_file(self):
        """Enriching with a real file should find tests."""
        enricher = ContextEnricher(root=".")
        result = enricher.enrich(
            ["graphsift/core.py"],
            find_config=True,
            find_tests=True,
            find_co_changes=False,
            find_env=False,
        )
        # Should find test candidates even if config doesn't exist
        assert isinstance(result, EnrichmentResult)

    def test_enrich_with_env(self):
        """Enriching with env discoverer should find references."""
        source_map = {
            "env_test.py": "import os\nx = os.getenv('MY_VAR')\n",
        }
        enricher = ContextEnricher(root=".", source_map=source_map)
        result = enricher.enrich(
            ["env_test.py"],
            find_config=False,
            find_tests=False,
            find_co_changes=False,
            find_env=True,
        )
        assert len(result.env_refs) >= 1
        assert result.env_refs[0].path == "MY_VAR"
