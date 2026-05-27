"""Git-history-aware temporal code graph for GraphSift.

Tracks symbol and file changes across git history, providing bi-temporal
queries (point-in-time, range-diff) and recency-boosted relevance scoring.

Architecture:
  TemporalGraph  — wraps DependencyGraph with git-log awareness via subprocess
  SymbolVersion  — a single commit-level event for one symbol
  FileVersion    — a single commit-level event for one file
  CommitInfo     — parsed commit metadata
  TemporalStats  — summary of index_history() run
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphsift.core import DependencyGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class SymbolVersion:
    """One commit-level change event for a single symbol."""

    qualified_name: str
    commit_hash: str
    timestamp: datetime
    action: str  # 'added' | 'modified' | 'deleted' | 'renamed'
    filepath: str
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class FileVersion:
    """One commit-level change event for a single file."""

    filepath: str
    commit_hash: str
    timestamp: datetime
    action: str  # 'added' | 'modified' | 'deleted' | 'renamed'
    previous_path: str | None = None


@dataclass
class CommitInfo:
    """Parsed metadata for a single git commit."""

    hash: str
    author: str
    timestamp: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0


@dataclass
class TemporalStats:
    """Summary statistics returned by ``TemporalGraph.index_history()``."""

    commits_indexed: int = 0
    symbols_tracked: int = 0
    files_tracked: int = 0
    oldest_commit: str = ""
    newest_commit: str = ""
    time_span_days: int = 0
    renames_detected: int = 0


# ---------------------------------------------------------------------------
# TemporalGraph
# ---------------------------------------------------------------------------


class TemporalGraph:
    """Git-history-aware dependency graph with bi-temporal symbol tracking.

    Wraps an optional ``DependencyGraph`` to correlate current parsed symbols
    with their git-history provenance. All git I/O is done via ``subprocess``
    (no ``GitPython`` dependency).

    Args:
        repo_path: Absolute or relative path to the git repository root.
        graph: Optional pre-built ``DependencyGraph`` to correlate with history.
    """

    def __init__(self, repo_path: str, graph: Any = None) -> None:
        self.repo_path = Path(repo_path)
        self._graph = graph  # optional DependencyGraph instance
        self._symbol_history: dict[str, list[SymbolVersion]] = {}
        self._file_history: dict[str, list[FileVersion]] = {}
        self._commits: list[CommitInfo] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_history(
        self, max_commits: int = 500, branch: str = "HEAD"
    ) -> TemporalStats:
        """Parse git log to build temporal symbol tracking.

        Uses ``git log --name-status --find-renames`` to track file changes,
        then correlates with graph symbols for temporal validity windows.

        Args:
            max_commits: Maximum number of commits to index.
            branch: Branch ref to walk (default ``HEAD``).

        Returns:
            TemporalStats summarising the indexing run.
        """
        commits: list[CommitInfo] = []
        file_hist: dict[str, list[FileVersion]] = {}

        try:
            result = subprocess.run(
                [
                    "git", "-C", str(self.repo_path),
                    "log", f"-{max_commits}", branch,
                    "--name-status", "--find-renames",
                    "--format=%H|%an|%aI|%s",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning("git log failed: %s", result.stderr)
                return TemporalStats()

            current_commit: CommitInfo | None = None
            for line in result.stdout.strip().split("\n"):
                if "|" in line and len(line.split("|")) >= 4:
                    parts = line.split("|", 3)
                    current_commit = CommitInfo(
                        hash=parts[0],
                        author=parts[1],
                        timestamp=datetime.fromisoformat(
                            parts[2].replace("Z", "+00:00")
                        ),
                        message=parts[3],
                        files_changed=[],
                        insertions=0,
                        deletions=0,
                    )
                    commits.append(current_commit)
                elif current_commit and line and line[0] in ("A", "M", "D", "R"):
                    action_char = line[0]
                    path_raw = line[2:]
                    current_commit.files_changed.append(path_raw)

                    action = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}.get(
                        action_char, "modified"
                    )
                    fv = FileVersion(
                        filepath=path_raw,
                        commit_hash=current_commit.hash,
                        timestamp=current_commit.timestamp,
                        action=action,
                    )
                    if action_char == "R" and "\t" in path_raw:
                        old, new = path_raw.split("\t", 1)
                        fv.filepath = new
                        fv.previous_path = old
                    file_hist.setdefault(fv.filepath, []).append(fv)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("git not available: %s", exc)
            return TemporalStats()

        with self._lock:
            self._commits = commits
            self._file_history = file_hist

            # Correlate with current graph symbols
            if self._graph is not None:
                nodes = getattr(self._graph, "_nodes", {})
                for node in nodes.values():
                    fp = getattr(node, "file_path", "")
                    if fp in file_hist and hasattr(node, "qualified_name"):
                        self._symbol_history[node.qualified_name] = [
                            SymbolVersion(
                                qualified_name=node.qualified_name,
                                commit_hash=fv.commit_hash,
                                timestamp=fv.timestamp,
                                action=fv.action,
                                filepath=fp,
                                line_start=getattr(node, "line_start", None),
                                line_end=getattr(node, "line_end", None),
                            )
                            for fv in file_hist[fp]
                        ]

        return TemporalStats(
            commits_indexed=len(commits),
            symbols_tracked=len(self._symbol_history),
            files_tracked=len(file_hist),
            oldest_commit=commits[-1].hash if commits else "",
            newest_commit=commits[0].hash if commits else "",
            time_span_days=(
                (commits[0].timestamp - commits[-1].timestamp).days
                if len(commits) > 1
                else 0
            ),
            renames_detected=sum(
                1
                for fv_list in file_hist.values()
                for fv in fv_list
                if fv.previous_path is not None
            ),
        )

    # ------------------------------------------------------------------
    # Temporal queries
    # ------------------------------------------------------------------

    def symbols_at(self, commit_hash: str) -> list[str]:
        """Return all symbol qualified-names that existed at a given commit.

        ``commit_hash`` may be any ref that appears in the indexed history.
        """
        result: list[str] = []
        commit_map = {c.hash: c for c in self._commits}

        target = commit_map.get(commit_hash)
        if target is None:
            return result

        for name, versions in self._symbol_history.items():
            for v in versions:
                if v.commit_hash == commit_hash:
                    if v.action != "deleted":
                        result.append(name)
                    break
                # Check whether this version's timestamp is at/before target
                if v.timestamp <= target.timestamp and v.action != "deleted":
                    result.append(name)
                    break

        return result

    def introduced_in(self, symbol_name: str) -> str | None:
        """Return the commit hash where *symbol_name* was first introduced."""
        versions = self._symbol_history.get(symbol_name, [])
        if versions:
            return versions[-1].commit_hash  # oldest first
        return None

    def last_modified(self, symbol_name: str) -> str | None:
        """Return the most recent commit hash that touched *symbol_name*."""
        versions = self._symbol_history.get(symbol_name, [])
        if versions:
            return versions[0].commit_hash  # newest first
        return None

    def age_boost(self, filepath: str, current_commit: str = "HEAD") -> float:
        """Recency boost for relevance ranking (0-1, newer = higher).

        Uses a logarithmic decay over 30-day windows so very old files
        still get a small baseline boost.

        Args:
            filepath: File path relative to repo root.
            current_commit: Ignored for now (uses wall-clock time).

        Returns:
            Float in ``(0, 1]``; 1.0 = modified today.
        """
        versions = self._file_history.get(filepath, [])
        if not versions:
            return 0.5  # unknown age → neutral
        newest = versions[0].timestamp
        days_since = (datetime.now().astimezone() - newest).days
        return 1.0 / (1.0 + math.log(1.0 + max(days_since, 0) / 30.0))

    # ------------------------------------------------------------------
    # Diff / file queries
    # ------------------------------------------------------------------

    def diff_between(self, from_commit: str, to_commit: str) -> list[str]:
        """Return files changed between two commits (``git diff --name-only``)."""
        try:
            result = subprocess.run(
                [
                    "git", "-C", str(self.repo_path),
                    "diff", "--name-only", from_commit, to_commit,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return [f for f in result.stdout.strip().split("\n") if f]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return []

    def history_of(self, symbol_name: str) -> list[SymbolVersion]:
        """Return the full change history for *symbol_name*.

        Returns an empty list if the symbol is not tracked.
        """
        return self._symbol_history.get(symbol_name, [])

    def file_history_of(self, filepath: str) -> list[FileVersion]:
        """Return the full change history for *filepath*.

        Returns an empty list if the file has no tracked history.
        """
        return self._file_history.get(filepath, [])

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def commits(self) -> list[CommitInfo]:
        """All indexed commits (newest first)."""
        return list(self._commits)

    @property
    def symbol_count(self) -> int:
        """Number of symbols with tracked history."""
        return len(self._symbol_history)

    @property
    def file_count(self) -> int:
        """Number of files with tracked history."""
        return len(self._file_history)
