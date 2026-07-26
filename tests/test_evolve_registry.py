"""Tests for the EvolveRegistry persistence layer."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from graphsift.evolve_registry import EvolveRegistry


class TestEvolveRegistry:
    """Tests for EvolveRegistry."""

    def test_default_path(self):
        """Default path should be .graphsift/evolve_registry.json."""
        registry = EvolveRegistry()
        assert registry.path.endswith(".graphsift/evolve_registry.json") or \
               registry.path.endswith(".graphsift\\evolve_registry.json")

    def test_get_missing_fingerprint(self):
        """Getting a non-existent fingerprint should return None."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            result = registry.get("nonexistent")
            assert result is None

    def test_set_and_get(self):
        """Setting then getting should return the same params."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            params = {"bm25_weight": 0.6, "graph_weight": 0.9}
            registry.set("abc123", "full", params, 0.5778)

            result = registry.get("abc123", "full")
            assert result is not None
            assert result["bm25_weight"] == 0.6
            assert result["graph_weight"] == 0.9

    def test_set_and_get_without_space_type(self):
        """get() should default to 'full' space_type."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            registry.set("abc123", "full", {"x": 1.0}, 0.5)
            result = registry.get("abc123")  # defaults to "full"
            assert result is not None

    def test_list_entries(self):
        """list_entries should return all entries with metadata."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            registry.set("fp1", "full", {"a": 1}, 0.9)
            registry.set("fp2", "ranker", {"b": 2}, 0.8)

            entries = registry.list_entries()
            assert len(entries) == 2
            fingerprints = {e["fingerprint"] for e in entries}
            assert "fp1" in fingerprints
            assert "fp2" in fingerprints

    def test_list_entries_empty(self):
        """list_entries on empty registry should return empty list."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            assert registry.list_entries() == []

    def test_clear(self):
        """Clear should remove all entries."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            registry.set("fp1", "full", {"a": 1}, 0.9)
            registry.clear()
            assert registry.list_entries() == []
            assert registry.get("fp1") is None

    def test_persistence_across_instances(self):
        """Data should persist across registry instances."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            r1 = EvolveRegistry(path)
            r1.set("fp1", "full", {"x": 0.5}, 0.9)

            r2 = EvolveRegistry(path)
            result = r2.get("fp1", "full")
            assert result is not None
            assert result["x"] == 0.5

    def test_corrupted_file(self):
        """Corrupted JSON file should return empty state without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            with open(path, "w") as f:
                f.write("this is not json")
            registry = EvolveRegistry(path)
            assert registry.list_entries() == []
            assert registry.get("anything") is None

    def test_overwrite_existing(self):
        """Setting same fingerprint should overwrite previous value."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            registry.set("fp1", "full", {"a": 1}, 0.5)
            registry.set("fp1", "full", {"a": 2}, 0.9)
            result = registry.get("fp1", "full")
            assert result["a"] == 2

    def test_multiple_space_types(self):
        """Multiple space types for same fingerprint should coexist."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            registry = EvolveRegistry(path)
            registry.set("fp1", "ranker", {"bm25": 0.6}, 0.8)
            registry.set("fp1", "config", {"budget": 500}, 0.7)

            ranker = registry.get("fp1", "ranker")
            config = registry.get("fp1", "config")
            assert ranker["bm25"] == 0.6
            assert config["budget"] == 500
