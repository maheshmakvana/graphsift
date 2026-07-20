"""Tests for incremental builds: SHA cache, purge, dot-dir exclusion."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from graphsift.sha_cache import (
    compute_sha,
    has_changed,
    load_sha_cache,
    save_sha_cache,
    sha_cache_path,
)
from graphsift.adapters.storage import GraphStore
from graphsift.adapters.filesystem import load_source_map


# ==============================================================================
# SHA Cache Tests
# ==============================================================================


class TestShaCache:
    """SHA-256 cache for incremental builds."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.root = str(self.tmpdir)

    def teardown_method(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_compute_sha_consistent(self):
        """Same content should produce the same SHA."""
        content = "def foo(): pass\n"
        assert compute_sha(content) == compute_sha(content)

    def test_compute_sha_different(self):
        """Different content should produce different SHAs."""
        assert compute_sha("hello") != compute_sha("world")

    def test_compute_sha_empty(self):
        """Empty string should produce a valid SHA."""
        sha = compute_sha("")
        assert len(sha) == 64  # SHA-256 hex digest is 64 chars
        assert sha.isalnum()

    def test_save_and_load_cache(self):
        """Saving then loading should return the same data."""
        cache = {"file1.py": "abc123", "file2.ts": "def456"}
        save_sha_cache(self.root, cache)
        loaded = load_sha_cache(self.root)
        assert loaded == cache

    def test_load_missing_cache(self):
        """Loading from a path with no cache should return empty dict."""
        cache = load_sha_cache(self.root)
        assert cache == {}

    def test_load_corrupt_cache(self):
        """Corrupt JSON should return empty dict without crashing."""
        path = Path(sha_cache_path(self.root))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not-json{{")
        cache = load_sha_cache(self.root)
        assert cache == {}

    def test_has_changed_new_file(self):
        """A file not in the cache should be reported as changed."""
        assert has_changed("new.py", "content", {})

    def test_has_changed_unchanged(self):
        """A file with matching SHA should be reported as unchanged."""
        content = "print('hello')"
        sha = compute_sha(content)
        assert not has_changed("main.py", content, {"main.py": sha})

    def test_has_changed_modified(self):
        """A file whose content changed should be reported as changed."""
        sha = compute_sha("old content")
        assert has_changed("main.py", "new content", {"main.py": sha})

    def test_sha_cache_path_consistent(self):
        """Same root should produce the same cache path."""
        p1 = sha_cache_path(self.root)
        p2 = sha_cache_path(self.root)
        assert p1 == p2

    def test_sha_cache_path_contains_filename(self):
        """Cache path should end with sha_cache.json."""
        path = sha_cache_path(self.root)
        assert path.endswith("sha_cache.json")


# ==============================================================================
# Purge Stale Files Tests
# ==============================================================================


class TestPurgeStaleFiles:
    """GraphStore.purge_stale_files() — remove files no longer in source map."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.store = GraphStore(self.db_path)

    def teardown_method(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _add_file(self, path: str, content: str = "dummy") -> None:
        """Helper to manually insert a file record into the DB."""
        conn = self.store._pool.acquire()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO files (path, language, size_bytes, sha256) VALUES (?, 'python', ?, ?)",
                (path, len(content), "abc123"),
            )
            conn.commit()
        finally:
            self.store._pool.release(conn)

    def _add_node(self, node_id: str, file_path: str) -> None:
        """Helper to manually insert a node record."""
        conn = self.store._pool.acquire()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO nodes (node_id, file_path, kind, name, qualified_name) VALUES (?, ?, 'function', ?, ?)",
                (node_id, file_path, node_id.split("::")[-1], node_id),
            )
            conn.commit()
        finally:
            self.store._pool.release(conn)

    def _count_files(self) -> int:
        return self.store._execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def _count_nodes(self) -> int:
        return self.store._execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def test_purge_no_stale(self):
        """When all DB files are in valid_paths, nothing should be purged."""
        self._add_file("src/main.py")
        self._add_file("src/utils.py")
        n = self.store.purge_stale_files({"src/main.py", "src/utils.py"})
        assert n == 0
        assert self._count_files() == 2

    def test_purge_some_stale(self):
        """Files in DB but not in valid_paths should be purged."""
        self._add_file("src/main.py")
        self._add_file("src/old.py")
        n = self.store.purge_stale_files({"src/main.py"})
        assert n == 1
        assert self._count_files() == 1

    def test_purge_nodes_with_stale_files(self):
        """Nodes belonging to purged files should also be removed."""
        self._add_file("src/main.py")
        self._add_file("src/old.py")
        self._add_node("src/old.py::func1", "src/old.py")
        self._add_node("src/old.py::func2", "src/old.py")
        n = self.store.purge_stale_files({"src/main.py"})
        assert n == 1
        assert self._count_files() == 1
        assert self._count_nodes() == 0

    def test_purge_all_stale(self):
        """Purging all files should leave an empty DB."""
        self._add_file("a.py")
        self._add_file("b.py")
        self._add_node("a.py::f1", "a.py")
        n = self.store.purge_stale_files(set())
        assert n == 2
        assert self._count_files() == 0
        assert self._count_nodes() == 0

    def test_purge_idempotent(self):
        """Running purge twice should be safe (no-op on second run)."""
        self._add_file("keep.py")
        self._add_file("remove.py")
        n1 = self.store.purge_stale_files({"keep.py"})
        assert n1 == 1
        n2 = self.store.purge_stale_files({"keep.py"})
        assert n2 == 0
        assert self._count_files() == 1

    def test_purge_empty_db(self):
        """Purging an empty DB should return 0."""
        n = self.store.purge_stale_files({"a.py"})
        assert n == 0

    def test_purge_preserves_valid_edges(self):
        """Edges referencing valid files should survive purge."""
        self._add_file("a.py")
        self._add_file("b.py")
        self._add_node("a.py::mod", "a.py")
        self._add_node("b.py::mod", "b.py")
        self.store._execute(
            "INSERT INTO edges (source_id, target_id, kind) VALUES ('a.py::mod', 'b.py::mod', 'imports')",
        )
        n = self.store.purge_stale_files({"a.py", "b.py"})
        assert n == 0
        edge_count = self.store._execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        assert edge_count == 1


# ==============================================================================
# Dot-Dir Exclusion Tests
# ==============================================================================


class TestDotDirExclusion:
    """load_source_map should skip dot-directories automatically."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.root = Path(self.tmpdir)

    def teardown_method(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_file(self, rel_path: str) -> Path:
        """Create a file under the temp root and return its path."""
        fp = self.root / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text("x = 1\n")
        return fp

    def _has_path(self, paths: list[str], suffix: str) -> bool:
        """Check if any path ends with suffix (OS-agnostic)."""
        return any(p.endswith(suffix.replace("/", "\\")) or p.endswith(suffix) for p in paths)

    def test_hidden_dir_excluded(self):
        """Files inside .next/ should not appear in the source map."""
        self._create_file("src/main.py")
        self._create_file(".next/server/chunks/index.js")
        sm = load_source_map(str(self.root), extensions={".py", ".js"})
        paths = list(sm.keys())
        assert self._has_path(paths, "src/main.py")
        assert not self._has_path(paths, ".next/index.js")

    def test_multiple_hidden_dirs_excluded(self):
        """Multiple dot-directories should all be excluded."""
        self._create_file("src/app.py")
        self._create_file(".git/HEAD")
        self._create_file(".venv/lib/site.py")
        self._create_file(".next/build/stats.js")
        self._create_file(".cache/webpack/bundle.js")
        sm = load_source_map(str(self.root), extensions={".py", ".js"})
        paths = list(sm.keys())
        assert self._has_path(paths, "src/app.py")
        assert not self._has_path(paths, ".git/HEAD")
        assert not self._has_path(paths, ".venv/site.py")
        assert not self._has_path(paths, ".next/stats.js")
        assert not self._has_path(paths, ".cache/bundle.js")

    def test_mixed_dot_and_normal(self):
        """Normal subdirectories should work alongside dot-dirs."""
        self._create_file(".config/settings.py")
        self._create_file("config/settings.py")
        self._create_file("src/main.py")
        sm = load_source_map(str(self.root), extensions={".py"})
        paths = list(sm.keys())
        assert self._has_path(paths, "config/settings.py")
        assert self._has_path(paths, "src/main.py")
        assert not self._has_path(paths, ".config/settings.py")

    def test_exclude_dirs_still_works(self):
        """Explicit exclude_dirs should still be honored alongside dot-dir skip."""
        self._create_file("src/main.py")
        self._create_file("node_modules/express/index.js")
        self._create_file(".next/out.js")
        sm = load_source_map(
            str(self.root),
            extensions={".py", ".js"},
            exclude_dirs={"node_modules"},
        )
        paths = list(sm.keys())
        assert self._has_path(paths, "src/main.py")
        assert not self._has_path(paths, "node_modules/index.js")
        assert not self._has_path(paths, ".next/out.js")


# ==============================================================================
# Integration: Purge during build flow
# ==============================================================================


class TestBuildPersistence:
    """Verify that saved SHA cache survives a build cycle."""

    def setup_method(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.root = str(self.tmpdir)
        # Create a source file
        src = Path(self.tmpdir) / "src" / "main.py"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("x = 1\n")

    def teardown_method(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sha_cache_round_trip(self):
        """SHA cache should survive save/load cycle."""
        content = "x = 1\n"
        cache = {"src/main.py": compute_sha(content)}
        save_sha_cache(self.root, cache)
        loaded = load_sha_cache(self.root)
        assert loaded == cache

    def test_purge_then_rebuild(self):
        """After purge, rebuilding should re-add files."""
        db_path = os.path.join(self.tmpdir, "test.db")
        store = GraphStore(db_path)

        # Add a file then purge it
        conn = store._pool.acquire()
        try:
            conn.execute("INSERT INTO files (path, language, size_bytes) VALUES (?, 'python', 10)", ("old.py",))
            conn.commit()
        finally:
            store._pool.release(conn)
        n = store.purge_stale_files(set())
        assert n == 1
        assert store._execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0

        # Add it back — should work cleanly
        conn = store._pool.acquire()
        try:
            conn.execute("INSERT INTO files (path, language, size_bytes) VALUES (?, 'python', 10)", ("new.py",))
            conn.commit()
        finally:
            store._pool.release(conn)
        assert store._execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
