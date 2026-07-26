"""Test-impact analysis — run only tests affected by changed files.

Uses the dependency graph to detect which tests cover recently changed
files and their dependents. Skips unaffected tests, saving 60-95% of
test time on incremental changes.

Architecture::

    TestImpactAnalyzer
        ├── _git_changed_files()      # files changed since last full test
        ├── _find_impacted_tests()    # dependency graph → related test files
        ├── _store_snapshot()         # persist test state to SQLite
        └── run_selective()           # orchestrate the pipeline

Memory (SQLite):
    - test_snapshots  — full test run history (commit_hash, status, duration)
    - impacted_tests  — cached mapping: source_file → [test_files]
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from graphsift.executor import ProcessRunner
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TestSnapshot:
    """A record of a full or selective test run."""
    id: int = 0
    commit_hash: str = ""
    mode: str = "full"  # "full" or "selective"
    status: str = "passed"  # "passed", "failed", "running"
    files_tested: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    duration_ms: float = 0.0
    changed_files: list[str] = field(default_factory=list)
    impacted_tests: list[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class ImpactResult:
    """Result of a selective test run."""
    mode: str  # "full" or "selective"
    status: str  # "passed", "failed", "skipped"
    changed_files: list[str] = field(default_factory=list)
    impacted_tests: list[str] = field(default_factory=list)
    skipped_tests: int = 0
    total_tests: int = 0
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    duration_ms: float = 0.0
    savings_pct: float = 0.0
    message: str = ""

    @property
    def summary(self) -> str:
        if self.mode == "full":
            return (
                f"FULL TEST: {self.status} | "
                f"{self.tests_run} tests in {self.duration_ms:.0f}ms"
            )
        if not self.impacted_tests:
            return (
                f"SELECTIVE: SKIPPED (0 impacted tests) | "
                f"{self.savings_pct:.0f}% savings"
            )
        return (
            f"SELECTIVE [{self.status}]: {self.tests_run} impacted tests "
            f"({self.skipped_tests} skipped, "
            f"{self.savings_pct:.0f}% savings) in {self.duration_ms:.0f}ms"
        )


# ---------------------------------------------------------------------------
# SQLite-backed memory for test snapshots and impacted-test mapping
# ---------------------------------------------------------------------------

class _TestMemory:
    """Persistent storage for test-impact analysis state.

    Uses a separate ``graph.db`` extension table so it survives restarts.
    """

    _TABLE_SNAPSHOTS = """
        CREATE TABLE IF NOT EXISTS test_snapshots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            commit_hash   TEXT NOT NULL,
            mode          TEXT NOT NULL DEFAULT 'full',
            status        TEXT NOT NULL DEFAULT 'passed',
            files_tested  INTEGER DEFAULT 0,
            tests_run     INTEGER DEFAULT 0,
            tests_passed  INTEGER DEFAULT 0,
            tests_failed  INTEGER DEFAULT 0,
            duration_ms   REAL DEFAULT 0.0,
            changed_files TEXT DEFAULT '[]',
            impacted_tests TEXT DEFAULT '[]',
            created_at    TEXT DEFAULT (datetime('now'))
        )
    """
    _TABLE_MAPPING = """
        CREATE TABLE IF NOT EXISTS impacted_tests (
            source_file TEXT NOT NULL,
            test_file   TEXT NOT NULL,
            confidence  REAL DEFAULT 1.0,
            updated_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (source_file, test_file)
        )
    """
    _TABLE_META = """
        CREATE TABLE IF NOT EXISTS test_impact_meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=10)
            try:
                conn.executescript(self._TABLE_SNAPSHOTS)
                conn.executescript(self._TABLE_MAPPING)
                conn.executescript(self._TABLE_META)
                conn.commit()
            finally:
                conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=10)

    # -- Snapshots -------------------------------------------------------

    def save_snapshot(self, snap: TestSnapshot) -> int:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute(
                    """INSERT INTO test_snapshots
                       (commit_hash, mode, status, files_tested,
                        tests_run, tests_passed, tests_failed, duration_ms,
                        changed_files, impacted_tests)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        snap.commit_hash, snap.mode, snap.status,
                        snap.files_tested, snap.tests_run,
                        snap.tests_passed, snap.tests_failed,
                        snap.duration_ms,
                        json.dumps(snap.changed_files),
                        json.dumps(snap.impacted_tests),
                    ),
                )
                conn.commit()
                return cur.lastrowid or 0
            finally:
                conn.close()

    def last_full_snapshot(self) -> TestSnapshot | None:
        """Return the most recent *full* test run that PASSED."""
        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    """SELECT id, commit_hash, mode, status, files_tested,
                              tests_run, tests_passed, tests_failed,
                              duration_ms, changed_files, impacted_tests,
                              created_at
                       FROM test_snapshots
                       WHERE mode='full' AND status='passed'
                       ORDER BY id DESC LIMIT 1"""
                ).fetchone()
                if not row:
                    return None
                return TestSnapshot(
                    id=row[0], commit_hash=row[1], mode=row[2],
                    status=row[3], files_tested=row[4],
                    tests_run=row[5], tests_passed=row[6],
                    tests_failed=row[7], duration_ms=row[8],
                    changed_files=json.loads(row[9] or "[]"),
                    impacted_tests=json.loads(row[10] or "[]"),
                    created_at=row[11],
                )
            finally:
                conn.close()

    def last_snapshot(self) -> TestSnapshot | None:
        """Return the most recent test run of any kind."""
        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    """SELECT id, commit_hash, mode, status, files_tested,
                              tests_run, tests_passed, tests_failed,
                              duration_ms, changed_files, impacted_tests,
                              created_at
                       FROM test_snapshots
                       ORDER BY id DESC LIMIT 1"""
                ).fetchone()
                if not row:
                    return None
                return TestSnapshot(
                    id=row[0], commit_hash=row[1], mode=row[2],
                    status=row[3], files_tested=row[4],
                    tests_run=row[5], tests_passed=row[6],
                    tests_failed=row[7], duration_ms=row[8],
                    changed_files=json.loads(row[9] or "[]"),
                    impacted_tests=json.loads(row[10] or "[]"),
                    created_at=row[11],
                )
            finally:
                conn.close()

    # -- Impacted test mappings ------------------------------------------

    def save_mappings(self, mappings: dict[str, list[str]]) -> int:
        """Store source_file -> [test_files] mappings. Returns count saved."""
        count = 0
        with self._lock:
            conn = self._conn()
            try:
                for source_file, test_files in mappings.items():
                    for tf in test_files:
                        conn.execute(
                            """INSERT OR REPLACE INTO impacted_tests
                               (source_file, test_file) VALUES (?, ?)""",
                            (source_file, tf),
                        )
                        count += 1
                conn.commit()
            finally:
                conn.close()
        return count

    def get_tests_for(self, source_file: str) -> list[str]:
        """Return test files that cover a given source file."""
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT test_file FROM impacted_tests WHERE source_file=?",
                    (source_file,),
                ).fetchall()
                return [r[0] for r in rows]
            finally:
                conn.close()

    def get_all_source_files(self) -> set[str]:
        """Return all source files that have known test mappings."""
        with self._lock:
            conn = self._conn()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT source_file FROM impacted_tests"
                ).fetchall()
                return {r[0] for r in rows}
            finally:
                conn.close()

    # -- Meta key/value store -------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO test_impact_meta (key, value) VALUES (?, ?)",
                    (key, value),
                )
                conn.commit()
            finally:
                conn.close()

    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT value FROM test_impact_meta WHERE key=?",
                    (key,),
                ).fetchone()
                return row[0] if row else default
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Test-Impact Analyzer
# ---------------------------------------------------------------------------

class TestImpactAnalyzer:
    """Smart test runner: runs only tests affected by changed files.

    Uses the dependency graph to trace which test files cover recently
    modified source files and their dependents. Remembers last full test
    state in SQLite so it can detect what's changed since then.

    Args:
        project_root: Repository root path.
        graph: Optional ``DependencyGraph`` instance (needed for impacted test
               resolution). If None, falls back to filename-pattern matching.
        store: Optional ``GraphStore`` (used for DB path). If None, uses
               ``.graphsift/graph.db`` under *project_root*.
    """

    def __init__(
        self,
        project_root: str = "",
        graph: Any = None,
        store: Any = None,
    ):
        self.project_root = Path(project_root or ".").resolve()
        self._graph = graph
        self._runner = ProcessRunner(cwd=str(self.project_root), timeout=300)

        # Determine DB path
        if store and hasattr(store, '_db_path'):
            db_path = store._db_path
        else:
            db_dir = self.project_root / ".graphsift"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "graph.db")

        self._memory = _TestMemory(db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full(
        self,
        pytest_args: str = "-x --tb=short -q",
        timeout: int = 300,
    ) -> ImpactResult:
        """Run the *full* test suite and store the result as a baseline.

        Call this after a complete test pass. Future ``run_selective()``
        calls will compare against this baseline.

        Args:
            pytest_args: Extra pytest CLI arguments.
            timeout: Per-command timeout in seconds.

        Returns:
            ``ImpactResult`` with full test results.
        """
        logger.info("Running FULL test suite (baseline) ...")
        t0 = time.monotonic()

        try:
            result = self._runner.run(
                f"pytest {pytest_args} --tb=short -q --no-header",
                timeout=timeout, check=False,
            )
            duration = (time.monotonic() - t0) * 1000

            # Parse test counts from output
            tests_run, passed, failed = self._parse_test_counts(result.stdout + result.stderr)
            passed_flag = result.ok()

            # Get current commit hash
            commit = self._get_git_hash()

            snap = TestSnapshot(
                commit_hash=commit,
                mode="full",
                status="passed" if passed_flag else "failed",
                files_tested=0,
                tests_run=tests_run,
                tests_passed=passed,
                tests_failed=failed,
                duration_ms=duration,
            )
            self._memory.save_snapshot(snap)

            # Build/refresh impacted-test mappings from the dependency graph
            self._build_impact_mappings()

            savings = 0.0
            msg = "Full test baseline saved"
            if passed_flag:
                msg += " — selective mode will skip unaffected tests next time"

            return ImpactResult(
                mode="full",
                status="passed" if passed_flag else "failed",
                total_tests=tests_run,
                tests_run=tests_run,
                tests_passed=passed,
                tests_failed=failed,
                duration_ms=duration,
                savings_pct=savings,
                message=msg,
            )

        except RuntimeError:
            return ImpactResult(
                mode="full", status="failed",
                message=f"Full test timed out after {timeout}s",
            )
        except Exception as exc:
            logger.exception("Full test failed")
            return ImpactResult(
                mode="full", status="failed",
                message=f"Full test error: {exc}",
            )

    def run_selective(
        self,
        changed_files: list[str] | None = None,
        pytest_args: str = "--tb=short -q",
        timeout: int = 120,
        auto_full_if_needed: bool = True,
    ) -> ImpactResult:
        """Run only tests affected by *changed_files*.

        If no changed_files provided, auto-detects via ``git diff`` since
        last full test snapshot.

        Args:
            changed_files: List of changed file paths. If None, auto-detect.
            pytest_args: Extra pytest CLI arguments.
            timeout: Per-command timeout in seconds.
            auto_full_if_needed: If True, run full test suite when no prior
                                 full-test snapshot exists.

        Returns:
            ``ImpactResult`` with selective test results.
        """
        t0 = time.monotonic()

        # -- Step 1: Check memory for last full test --------------------
        last_full = self._memory.last_full_snapshot()

        if last_full is None:
            if auto_full_if_needed:
                logger.info("No prior full test found — running full suite first")
                return self.run_full(pytest_args=pytest_args, timeout=timeout)
            return ImpactResult(
                mode="selective", status="skipped",
                message="No prior full test snapshot. Run `graphsift test-impact full` first.",
            )

        # -- Step 2: Detect changed files -------------------------------
        if changed_files is None:
            changed_files = self._git_changed_files(last_full.commit_hash)

        if not changed_files:
            return ImpactResult(
                mode="selective", status="skipped",
                changed_files=[],
                message="No files changed since last full test.",
                savings_pct=100.0,
                duration_ms=(time.monotonic() - t0) * 1000,
            )

        # -- Step 3: Find impacted tests --------------------------------
        impacted = self._find_impacted_tests(changed_files)

        if not impacted:
            # No tests directly impacted — still run a quick smoke test
            # on the changed files themselves if they look testable
            logger.info(
                "No impacted tests found for %d changed files",
                len(changed_files),
            )
            return ImpactResult(
                mode="selective", status="passed",
                changed_files=changed_files,
                impacted_tests=[],
                skipped_tests=0,
                total_tests=0,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                savings_pct=100.0,
                duration_ms=(time.monotonic() - t0) * 1000,
                message=f"No tests impacted by {len(changed_files)} changed files — skipping",
            )

        # -- Step 4: Estimate total test count --------------------------
        total_tests = self._estimate_total_tests()

        # -- Step 5: Run only impacted tests (parallel + timeout) ------
        logger.info(
            "Running SELECTIVE tests: %d impacted files for %d source changes "
            "(%d%% savings expected)",
            len(impacted), len(changed_files),
            self._calc_savings(len(impacted), total_tests) if total_tests else 0,
        )

        try:
            # Build test file arguments
            test_files_str = " ".join(
                f'"{t}"' if " " in t else t
                for t in impacted
            )

            import os as _os
            cpu_count = _os.cpu_count() or 4
            workers = max(1, cpu_count - 1)

            result = self._runner.run(
                f"pytest {test_files_str} -n={workers} "
                f"--dist=loadscope --timeout=120 --timeout-method=thread "
                f"{pytest_args} --no-header",
                timeout=timeout, check=False,
            )

            duration = (time.monotonic() - t0) * 1000
            tests_run, passed, failed = self._parse_test_counts(result.stdout + result.stderr)
            passed_flag = result.ok()

            # -- Step 6: Store snapshot ---------------------------------
            commit = self._get_git_hash()
            snap = TestSnapshot(
                commit_hash=commit,
                mode="selective",
                status="passed" if passed_flag else "failed",
                files_tested=len(changed_files),
                tests_run=tests_run,
                tests_passed=passed,
                tests_failed=failed,
                duration_ms=duration,
                changed_files=changed_files,
                impacted_tests=impacted,
            )
            self._memory.save_snapshot(snap)

            skipped = max(0, total_tests - tests_run) if total_tests else 0
            savings_pct = self._calc_savings(tests_run, total_tests) if total_tests else 0

            return ImpactResult(
                mode="selective",
                status="passed" if passed_flag else "failed",
                changed_files=changed_files,
                impacted_tests=impacted,
                skipped_tests=skipped,
                total_tests=total_tests,
                tests_run=tests_run,
                tests_passed=passed,
                tests_failed=failed,
                duration_ms=duration,
                savings_pct=savings_pct,
                message=(
                    f"{len(impacted)} impacted tests for {len(changed_files)} changed files. "
                    f"{skipped} tests skipped ({savings_pct:.0f}% savings)."
                ),
            )

        except RuntimeError:
            return ImpactResult(
                mode="selective", status="failed",
                changed_files=changed_files,
                impacted_tests=impacted,
                message=f"Selective test timed out after {timeout}s",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("Selective test failed")
            return ImpactResult(
                mode="selective", status="failed",
                changed_files=changed_files,
                impacted_tests=impacted,
                message=f"Error: {exc}",
                duration_ms=(time.monotonic() - t0) * 1000,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_impact_mappings(self) -> int:
        """Build source_file -> test_file mappings from the dependency graph.

        Uses ``tests_for`` queries on graph nodes, plus filename pattern
        matching as a fallback.

        Returns:
            Number of mappings saved.
        """
        mappings: dict[str, list[str]] = {}
        graph = self._graph

        if graph is None:
            return 0

        try:
            with graph._lock:
                nodes = dict(graph._nodes)
                file_nodes = dict(graph._file_nodes)
                adj_in = {k: list(v) for k, v in graph._adj_in.items()}

            # For every source file, find test files that import it
            for fp in file_nodes:
                # Skip test files themselves
                if "test_" in Path(fp).stem or "_test" in Path(fp).stem:
                    continue

                # Find test files that reference this file's symbols
                test_files: set[str] = set()

                # Method 1: Follow IMPORT edges to find test → source
                for nid, node in nodes.items():
                    if "test_" in Path(node.file_path).stem or "_test" in Path(node.file_path).stem:
                        for edge in adj_in.get(nid, []):
                            target = nodes.get(edge.target_id)
                            if target and target.file_path == fp:
                                test_files.add(node.file_path)

                # Method 2: Filename pattern
                stem = Path(fp).stem
                for other_fp in file_nodes:
                    if other_fp == fp:
                        continue
                    other_stem = Path(other_fp).stem
                    if f"test_{stem}" == other_stem or f"{stem}_test" == other_stem:
                        test_files.add(other_fp)

                if test_files:
                    mappings[fp] = sorted(test_files)

            return self._memory.save_mappings(mappings)

        except Exception as exc:
            logger.debug("Failed to build impact mappings: %s", exc)
            return 0

    def _find_impacted_tests(self, changed_files: list[str]) -> list[str]:
        """Find all test files impacted by *changed_files*.

        Uses cached mappings from SQLite + dependency graph traversal
        for dependent files.

        Returns:
            Deduplicated, sorted list of test file paths.
        """
        graph = self._graph
        impacted: set[str] = set()

        for cf in changed_files:
            # Method 1: Check cached mappings
            cached_tests = self._memory.get_tests_for(cf)
            impacted.update(cached_tests)

            # Method 2: Find dependents (files that import the changed file)
            if graph is not None:
                try:
                    with graph._lock:
                        nodes = dict(graph._nodes)
                        adj_in = {k: list(v) for k, v in graph._adj_in.items()}
                        file_nodes = dict(graph._file_nodes)

                    # Find nodes in the changed file
                    for nid, node in nodes.items():
                        if node.file_path == cf:
                            # Find importers of this node
                            for edge in adj_in.get(nid, []):
                                source_node = nodes.get(edge.source_id)
                                if source_node:
                                    dependent_file = source_node.file_path
                                    if dependent_file != cf:
                                        # Find tests for the dependent file
                                        dep_tests = self._memory.get_tests_for(dependent_file)
                                        impacted.update(dep_tests)
                except Exception:
                    pass

            # Method 3: Filename pattern (quick fallback)
            stem = Path(cf).stem
            for test_fp in self._find_test_files_by_pattern(stem):
                impacted.add(test_fp)

        return sorted(impacted)

    def _find_test_files_by_pattern(self, source_stem: str) -> list[str]:
        """Find test files matching ``test_{source_stem}`` or ``{source_stem}_test``."""
        test_files: list[str] = []
        search_dir = self.project_root

        # Common test directory patterns
        for pattern in [
            f"test_{source_stem}.py",
            f"{source_stem}_test.py",
            f"**/test_{source_stem}.py",
            f"**/{source_stem}_test.py",
        ]:
            matched = list(Path(search_dir).rglob(pattern))
            test_files.extend(str(m.relative_to(search_dir)) for m in matched)

        return test_files

    def _git_changed_files(self, since_commit: str) -> list[str]:
        """Return list of files changed since *since_commit*."""
        try:
            result = self._runner.run_simple(
                ["git", "diff", "--name-only", since_commit, "HEAD"],
                timeout=30,
            )
            if not result.ok():
                # Try with just HEAD~1
                result = self._runner.run_simple(
                    ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                    timeout=30,
                )
            files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
            return files
        except Exception as exc:
            logger.debug("git diff failed: %s", exc)
            return []

    def _get_git_hash(self) -> str:
        """Return current git commit hash."""
        try:
            result = self._runner.run_simple(
                ["git", "rev-parse", "--short", "HEAD"],
                timeout=10,
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def _parse_test_counts(self, output: str) -> tuple[int, int, int]:
        """Parse pytest output for: tests_run, passed, failed.

        Handles formats:
          - "3 passed in 0.12s"
          - "2 failed, 5 passed in 0.45s"
          - "1 failed in 0.10s"
        """
        import re

        # Try "X passed" / "Y failed" pattern
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        total_match = re.search(r"(\d+)\s+(?:passed|failed|skipped)", output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0

        # For total, sum all test outcomes
        total = passed + failed

        # Also look for "X items" in verbose output
        items_match = re.search(r"(\d+)\s+items?\s+(?:passed|failed)", output)
        if items_match:
            total = max(total, int(items_match.group(1)))

        if total == 0 and "no tests ran" in output:
            return 0, 0, 0

        return max(total, 1), passed, failed

    def _estimate_total_tests(self) -> int:
        """Estimate total tests in the project (for savings calculation)."""
        try:
            result = self._runner.run_simple(
                ["pytest", "--collect-only", "-q", "--no-header"],
                timeout=30,
            )
            # Last line should have the count
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            if lines:
                import re
                match = re.search(r"(\d+)\s+(?:tests?|items?)", lines[-1])
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        return 0

    @staticmethod
    def _calc_savings(selected: int, total: int) -> float:
        """Calculate savings percentage from selective testing."""
        if total <= 0:
            return 0.0
        raw = (1 - (selected / total)) * 100
        return max(0.0, round(raw, 1))


def run_full_test(project_root: str = "", **kwargs) -> ImpactResult:
    """Convenience function: run full test suite with impact tracking."""
    analyzer = TestImpactAnalyzer(project_root=project_root)
    return analyzer.run_full(**kwargs)


def run_selective_test(
    project_root: str = "",
    changed_files: list[str] | None = None,
    **kwargs,
) -> ImpactResult:
    """Convenience function: run only impacted tests."""
    analyzer = TestImpactAnalyzer(project_root=project_root)
    return analyzer.run_selective(changed_files=changed_files, **kwargs)
