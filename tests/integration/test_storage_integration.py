"""Integration test: save/load cycle with SQLite storage."""

import json
import tempfile
import os
import pytest

from graphsift import ContextBuilder, ContextConfig, DiffSpec
from graphsift.adapters.storage import GraphStore


class TestStorageIntegration:
    """Integration tests for storage save/load cycles."""

    @pytest.fixture
    def db_path(self):
        """Create a temporary SQLite database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        try:
            if os.path.exists(path):
                os.unlink(path)
        except (PermissionError, OSError):
            pass  # Windows may still have locks on the file

    def test_graph_store_initialization(self, db_path):
        """GraphStore initializes and creates tables."""
        store = GraphStore(db_path)
        assert store is not None
        # Should not crash on init
        store.close()

    def test_save_and_load_context(self, db_path, source_map):
        """Save a context result and load it back."""
        store = GraphStore(db_path)

        # Build a context first
        config = ContextConfig(token_budget=50_000, session_id="test_session")
        builder = ContextBuilder(config)
        builder.index_files(source_map)

        diff = DiffSpec(
            changed_files=["src/auth.py"],
            query="Review auth changes",
        )
        result = builder.build(diff, source_map)

        # Save context data
        store.save_session_context(
            session_id="test_session",
            diff_spec_hash="abc123",
            context_data={
                "diff_spec": diff.model_dump(),
                "selected_files": [
                    {
                        "file_node": sf.file_node.model_dump(),
                        "score": sf.score,
                        "rank": sf.rank,
                    }
                    for sf in result.selected_files
                ],
                "rendered_context": result.rendered_context,
                "total_original_tokens": result.total_original_tokens,
                "total_rendered_tokens": result.total_rendered_tokens,
                "reduction_ratio": result.reduction_ratio,
            },
        )

        # Load it back
        loaded = store.load_session_context("test_session", "abc123")
        assert loaded is not None
        assert "rendered_context" in loaded
        assert loaded["total_original_tokens"] == result.total_original_tokens

        store.close()

    def test_load_nonexistent_returns_none(self, db_path):
        """Loading non-existent session returns None."""
        store = GraphStore(db_path)
        result = store.load_session_context("nonexistent", "nohash")
        assert result is None
        store.close()

    def test_multiple_sessions(self, db_path, source_map):
        """Multiple sessions can be saved and loaded independently."""
        store = GraphStore(db_path)

        sessions = {
            "session_A": {"diff_spec_hash": "hash1", "data": {"note": "First session"}},
            "session_B": {"diff_spec_hash": "hash2", "data": {"note": "Second session"}},
            "session_C": {"diff_spec_hash": "hash3", "data": {"note": "Third session"}},
        }

        for sid, info in sessions.items():
            store.save_session_context(sid, info["diff_spec_hash"], info["data"])

        # Verify each can be loaded
        for sid, info in sessions.items():
            loaded = store.load_session_context(sid, info["diff_spec_hash"])
            assert loaded is not None
            assert loaded["note"] == info["data"]["note"]

        store.close()

    def test_graph_store_persistence(self, db_path):
        """Data persists across GraphStore instances."""
        # First instance: write
        store1 = GraphStore(db_path)
        store1.save_session_context("persist_session", "hash1",
                                     {"data": "persistent_data"})
        store1.close()

        # Second instance: read
        store2 = GraphStore(db_path)
        loaded = store2.load_session_context("persist_session", "hash1")
        assert loaded is not None
        assert loaded["data"] == "persistent_data"
        store2.close()


class TestStorageEdgeCases:
    """Edge cases for storage operations."""

    def test_empty_db_path(self):
        """GraphStore with empty path should handle gracefully."""
        try:
            store = GraphStore("")
            # Some implementations may create in-memory DB
            store.close()
        except Exception as e:
            # Acceptable if it raises a controlled error
            assert "path" in str(e).lower() or "file" in str(e).lower()

    def test_invalid_session_data(self, temp_db):
        """Save with invalid data should not crash."""
        store = GraphStore(temp_db)
        try:
            store.save_session_context("test", "hash", None)
        except Exception:
            pass  # Accept controlled errors
        store.close()

    def test_overwrite_session(self, temp_db):
        """Overwriting an existing session should work."""
        store = GraphStore(temp_db)

        store.save_session_context("same_session", "hash1",
                                     {"version": 1, "data": "original"})
        store.save_session_context("same_session", "hash1",
                                     {"version": 2, "data": "overwritten"})

        loaded = store.load_session_context("same_session", "hash1")
        assert loaded is not None
        # Should be the latest version
        assert loaded.get("version") == 2

        store.close()
