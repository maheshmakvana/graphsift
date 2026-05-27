"""SQLite persistence adapter for graphsift.

Schema versioned via a migrations table — same pattern as code-review-graph.

Migration history:
  v1  — nodes table (core symbol records)
  v2  — edges table (directed dependency edges)
  v3  — files table (file-level metadata)
  v4  — community_id column on nodes + communities table
  v5  — nodes_fts FTS5 full-text search virtual table
  v6  — summary tables: community_summaries, flow_snapshots, risk_index
  v7  — graph_meta key/value store (embed_version, token mode settings)

Usage (caller-supplied path)::

    from graphsift.adapters.storage import GraphStore

    store = GraphStore("/path/to/repo/.graphsift/graph.db")
    store.save_nodes(nodes)
    store.save_edges(edges)
    store.save_files(files)
    nodes = store.load_nodes()
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ..exceptions import GraphError
from ..models import (
    EdgeKind,
    FileNode,
    GraphEdge,
    GraphNode,
    Language,
    NodeKind,
)

logger = logging.getLogger(__name__)

_CURRENT_VERSION = 8


# ---------------------------------------------------------------------------
# Migration definitions
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "created nodes table",
        [
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id       TEXT PRIMARY KEY,
                file_path     TEXT NOT NULL,
                kind          TEXT NOT NULL,
                name          TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                line_start    INTEGER DEFAULT 0,
                line_end      INTEGER DEFAULT 0,
                language      TEXT DEFAULT 'unknown',
                signature     TEXT DEFAULT '',
                decorators    TEXT DEFAULT '[]',
                is_async      INTEGER DEFAULT 0,
                is_dynamic    INTEGER DEFAULT 0,
                metadata      TEXT DEFAULT '{}'
            )
            """,
        ],
    ),
    (
        2,
        "created edges table",
        [
            """
            CREATE TABLE IF NOT EXISTS edges (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                kind      TEXT NOT NULL,
                weight    REAL DEFAULT 1.0,
                metadata  TEXT DEFAULT '{}',
                UNIQUE(source_id, target_id, kind)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)",
        ],
    ),
    (
        3,
        "created files table",
        [
            """
            CREATE TABLE IF NOT EXISTS files (
                path          TEXT PRIMARY KEY,
                language      TEXT NOT NULL,
                size_bytes    INTEGER DEFAULT 0,
                line_count    INTEGER DEFAULT 0,
                sha256        TEXT DEFAULT '',
                token_estimate INTEGER DEFAULT 0,
                imports       TEXT DEFAULT '[]',
                dynamic_imports TEXT DEFAULT '[]',
                metadata      TEXT DEFAULT '{}'
            )
            """,
        ],
    ),
    (
        4,
        "added 'community_id' column to nodes; created communities table",
        [
            "ALTER TABLE nodes ADD COLUMN community_id INTEGER",
            """
            CREATE TABLE IF NOT EXISTS communities (
                community_id  INTEGER PRIMARY KEY,
                label         TEXT DEFAULT '',
                node_count    INTEGER DEFAULT 0,
                metadata      TEXT DEFAULT '{}'
            )
            """,
        ],
    ),
    (
        5,
        "created nodes_fts FTS5 virtual table",
        [
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
            USING fts5(
                node_id,
                name,
                qualified_name,
                signature,
                file_path,
                content='nodes',
                content_rowid='rowid'
            )
            """,
            # Populate FTS from existing nodes
            "INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')",
        ],
    ),
    (
        6,
        "created summary tables (community_summaries, flow_snapshots, risk_index)",
        [
            """
            CREATE TABLE IF NOT EXISTS community_summaries (
                community_id  INTEGER PRIMARY KEY,
                summary_text  TEXT DEFAULT '',
                key_symbols   TEXT DEFAULT '[]',
                created_at    TEXT DEFAULT (datetime('now')),
                metadata      TEXT DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flow_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_name     TEXT NOT NULL,
                entry_point   TEXT NOT NULL,
                nodes_json    TEXT DEFAULT '[]',
                edges_json    TEXT DEFAULT '[]',
                created_at    TEXT DEFAULT (datetime('now')),
                metadata      TEXT DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_index (
                file_path     TEXT PRIMARY KEY,
                risk_score    REAL DEFAULT 0.0,
                reasons       TEXT DEFAULT '[]',
                computed_at   TEXT DEFAULT (datetime('now')),
                metadata      TEXT DEFAULT '{}'
            )
            """,
        ],
    ),
    (
        7,
        "created graph_meta key/value store for embed_version and token-mode settings",
        [
            """
            CREATE TABLE IF NOT EXISTS graph_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL DEFAULT ''
            )
            """,
        ],
    ),
    (
        8,
        "created session_memory and context_feedback tables for cross-session persistence",
        [
            """
            CREATE TABLE IF NOT EXISTS session_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                diff_spec_hash TEXT NOT NULL,
                context_json TEXT NOT NULL,
                files_selected INTEGER,
                tokens_rendered INTEGER,
                reduction_ratio REAL,
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed_at TEXT DEFAULT (datetime('now')),
                access_count INTEGER DEFAULT 1,
                UNIQUE(session_id, diff_spec_hash)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_session_memory_session
            ON session_memory(session_id, created_at)
            """,
            """
            CREATE TABLE IF NOT EXISTS context_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id INTEGER REFERENCES session_memory(id),
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_context_feedback_context
            ON context_feedback(context_id)
            """,
        ],
    ),
]


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------


class GraphStore:
    """Persistent SQLite store for a graphsift dependency graph.

    The caller supplies the database path. The store runs all pending
    migrations automatically on first open and logs each step just like
    ``code-review-graph`` does.

    Args:
        db_path: Absolute path to the SQLite database file. Parent directory
            must exist (or be created by the caller).
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._run_migrations()

    def __repr__(self) -> str:
        return f"GraphStore({self._db_path!r})"

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def _schema_version(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()
            return row[0] or 0
        except sqlite3.OperationalError:
            return 0

    def _run_migrations(self) -> None:
        with self._lock:
            # Bootstrap migrations table
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version    INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            self._conn.commit()

            current = self._schema_version()
            if current >= _CURRENT_VERSION:
                logger.info("graphsift: schema already at version %d", current)
                return

            for version, description, statements in _MIGRATIONS:
                if version <= current:
                    continue
                logger.info("INFO: Running migration v%d", version)
                try:
                    for sql in statements:
                        self._conn.execute(sql.strip())
                    self._conn.execute(
                        "INSERT INTO schema_migrations(version, description) VALUES(?, ?)",
                        (version, description),
                    )
                    self._conn.commit()
                    logger.info("INFO: Migration v%d: %s", version, description)
                except sqlite3.OperationalError as exc:
                    # FTS5 may not be available in all SQLite builds — skip gracefully
                    if version == 5 and "no such module: fts5" in str(exc).lower():
                        logger.warning(
                            "graphsift: FTS5 not available in this SQLite build — skipping v5 migration"
                        )
                        self._conn.execute(
                            "INSERT INTO schema_migrations(version, description) VALUES(?, ?)",
                            (version, f"{description} [skipped: no fts5]"),
                        )
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                        raise GraphError(
                            f"Migration v{version} failed: {exc}"
                        ) from exc

            logger.info(
                "INFO: Migrations complete, now at schema version %d", _CURRENT_VERSION
            )

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def save_nodes(self, nodes: list[GraphNode]) -> None:
        """Upsert a list of GraphNode records into the database.

        Args:
            nodes: Nodes to persist.
        """
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO nodes
                    (node_id, file_path, kind, name, qualified_name,
                     line_start, line_end, language, signature, decorators,
                     is_async, is_dynamic, community_id, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(node_id) DO UPDATE SET
                    file_path=excluded.file_path,
                    kind=excluded.kind,
                    name=excluded.name,
                    qualified_name=excluded.qualified_name,
                    line_start=excluded.line_start,
                    line_end=excluded.line_end,
                    language=excluded.language,
                    signature=excluded.signature,
                    decorators=excluded.decorators,
                    is_async=excluded.is_async,
                    is_dynamic=excluded.is_dynamic,
                    community_id=excluded.community_id,
                    metadata=excluded.metadata
                """,
                [
                    (
                        n.node_id,
                        n.file_path,
                        n.kind.value,
                        n.name,
                        n.qualified_name,
                        n.line_start,
                        n.line_end,
                        n.language.value,
                        n.signature,
                        json.dumps(n.decorators),
                        int(n.is_async),
                        int(n.is_dynamic),
                        n.community_id,
                        json.dumps(n.metadata),
                    )
                    for n in nodes
                ],
            )
            self._conn.commit()

    def load_nodes(self) -> list[GraphNode]:
        """Load all nodes from the database.

        Returns:
            List of GraphNode instances.
        """
        with self._lock:
            rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        result: list[GraphNode] = []
        for row in rows:
            try:
                result.append(
                    GraphNode(
                        node_id=row["node_id"],
                        file_path=row["file_path"],
                        kind=NodeKind(row["kind"]),
                        name=row["name"],
                        qualified_name=row["qualified_name"],
                        line_start=row["line_start"],
                        line_end=row["line_end"],
                        language=Language(row["language"]),
                        signature=row["signature"] or "",
                        decorators=json.loads(row["decorators"] or "[]"),
                        is_async=bool(row["is_async"]),
                        is_dynamic=bool(row["is_dynamic"]),
                        community_id=row["community_id"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("graphsift: skipping malformed node row: %s", exc)
        return result

    def search_nodes(self, query: str, limit: int = 20) -> list[GraphNode]:
        """Full-text search over node names/signatures using FTS5 (if available).

        Falls back to LIKE search when FTS5 is unavailable.

        Args:
            query: Search term.
            limit: Max results.

        Returns:
            Matching GraphNode instances.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT n.* FROM nodes n
                    JOIN nodes_fts f ON n.node_id = f.node_id
                    WHERE nodes_fts MATCH ?
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                # FTS5 not available — fall back to LIKE
                like = f"%{query}%"
                rows = self._conn.execute(
                    "SELECT * FROM nodes WHERE name LIKE ? OR qualified_name LIKE ? LIMIT ?",
                    (like, like, limit),
                ).fetchall()

        result: list[GraphNode] = []
        for row in rows:
            try:
                result.append(
                    GraphNode(
                        node_id=row["node_id"],
                        file_path=row["file_path"],
                        kind=NodeKind(row["kind"]),
                        name=row["name"],
                        qualified_name=row["qualified_name"],
                        line_start=row["line_start"],
                        line_end=row["line_end"],
                        language=Language(row["language"]),
                        signature=row["signature"] or "",
                        decorators=json.loads(row["decorators"] or "[]"),
                        is_async=bool(row["is_async"]),
                        is_dynamic=bool(row["is_dynamic"]),
                        community_id=row["community_id"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("graphsift: skipping malformed search row: %s", exc)
        return result

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def save_edges(self, edges: list[GraphEdge]) -> None:
        """Upsert a list of GraphEdge records.

        Args:
            edges: Edges to persist.
        """
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO edges (source_id, target_id, kind, weight, metadata)
                VALUES (?,?,?,?,?)
                ON CONFLICT(source_id, target_id, kind) DO UPDATE SET
                    weight=excluded.weight,
                    metadata=excluded.metadata
                """,
                [
                    (
                        e.source_id,
                        e.target_id,
                        e.kind.value,
                        e.weight,
                        json.dumps(e.metadata),
                    )
                    for e in edges
                ],
            )
            self._conn.commit()

    def load_edges(self) -> list[GraphEdge]:
        """Load all edges from the database.

        Returns:
            List of GraphEdge instances.
        """
        with self._lock:
            rows = self._conn.execute("SELECT * FROM edges").fetchall()
        result: list[GraphEdge] = []
        for row in rows:
            try:
                result.append(
                    GraphEdge(
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        kind=EdgeKind(row["kind"]),
                        weight=row["weight"],
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("graphsift: skipping malformed edge row: %s", exc)
        return result

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def save_files(self, files: list[FileNode]) -> None:
        """Upsert a list of FileNode records.

        Args:
            files: Files to persist.
        """
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO files
                    (path, language, size_bytes, line_count, sha256,
                     token_estimate, imports, dynamic_imports, metadata)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                    language=excluded.language,
                    size_bytes=excluded.size_bytes,
                    line_count=excluded.line_count,
                    sha256=excluded.sha256,
                    token_estimate=excluded.token_estimate,
                    imports=excluded.imports,
                    dynamic_imports=excluded.dynamic_imports,
                    metadata=excluded.metadata
                """,
                [
                    (
                        f.path,
                        f.language.value,
                        f.size_bytes,
                        f.line_count,
                        f.sha256,
                        f.token_estimate,
                        json.dumps(f.imports),
                        json.dumps(f.dynamic_imports),
                        json.dumps(f.metadata),
                    )
                    for f in files
                ],
            )
            self._conn.commit()

    def load_files(self) -> list[FileNode]:
        """Load all files from the database.

        Returns:
            List of FileNode instances (symbols list is empty — use load_nodes for symbols).
        """
        with self._lock:
            rows = self._conn.execute("SELECT * FROM files").fetchall()
        result: list[FileNode] = []
        for row in rows:
            try:
                result.append(
                    FileNode(
                        path=row["path"],
                        language=Language(row["language"]),
                        size_bytes=row["size_bytes"],
                        line_count=row["line_count"],
                        sha256=row["sha256"] or "",
                        token_estimate=row["token_estimate"],
                        imports=json.loads(row["imports"] or "[]"),
                        dynamic_imports=json.loads(row["dynamic_imports"] or "[]"),
                        metadata=json.loads(row["metadata"] or "{}"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("graphsift: skipping malformed file row: %s", exc)
        return result

    # ------------------------------------------------------------------
    # Communities
    # ------------------------------------------------------------------

    def save_community(self, community_id: int, label: str = "", node_count: int = 0, metadata: dict[str, Any] | None = None) -> None:
        """Upsert a community record.

        Args:
            community_id: Integer cluster ID.
            label: Human-readable label for the community.
            node_count: Number of nodes in this community.
            metadata: Additional metadata dict.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO communities (community_id, label, node_count, metadata)
                VALUES (?,?,?,?)
                ON CONFLICT(community_id) DO UPDATE SET
                    label=excluded.label,
                    node_count=excluded.node_count,
                    metadata=excluded.metadata
                """,
                (community_id, label, node_count, json.dumps(metadata or {})),
            )
            self._conn.commit()

    def assign_community(self, node_id: str, community_id: int) -> None:
        """Set the community_id for a node.

        Args:
            node_id: Node to update.
            community_id: Community cluster ID to assign.
        """
        with self._lock:
            self._conn.execute(
                "UPDATE nodes SET community_id=? WHERE node_id=?",
                (community_id, node_id),
            )
            self._conn.commit()

    def load_communities(self) -> list[dict[str, Any]]:
        """Load all community records.

        Returns:
            List of dicts with community_id, label, node_count, metadata.
        """
        with self._lock:
            rows = self._conn.execute("SELECT * FROM communities").fetchall()
        return [
            {
                "community_id": row["community_id"],
                "label": row["label"],
                "node_count": row["node_count"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Risk index
    # ------------------------------------------------------------------

    def upsert_risk(self, file_path: str, risk_score: float, reasons: list[str], metadata: dict[str, Any] | None = None) -> None:
        """Upsert a risk score for a file.

        Args:
            file_path: Path to the file.
            risk_score: Risk score between 0.0 and 1.0.
            reasons: Human-readable reasons for the score.
            metadata: Additional metadata.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO risk_index (file_path, risk_score, reasons, computed_at, metadata)
                VALUES (?, ?, ?, datetime('now'), ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    risk_score=excluded.risk_score,
                    reasons=excluded.reasons,
                    computed_at=excluded.computed_at,
                    metadata=excluded.metadata
                """,
                (file_path, risk_score, json.dumps(reasons), json.dumps(metadata or {})),
            )
            self._conn.commit()

    def load_risk_index(self, min_score: float = 0.0) -> list[dict[str, Any]]:
        """Load risk index entries above a minimum score.

        Args:
            min_score: Minimum risk score to include (default 0 = all).

        Returns:
            List of dicts sorted by risk_score descending.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM risk_index WHERE risk_score >= ? ORDER BY risk_score DESC",
                (min_score,),
            ).fetchall()
        return [
            {
                "file_path": row["file_path"],
                "risk_score": row["risk_score"],
                "reasons": json.loads(row["reasons"] or "[]"),
                "computed_at": row["computed_at"],
                "metadata": json.loads(row["metadata"] or "{}"),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Community summaries
    # ------------------------------------------------------------------

    def save_community_summary(self, community_id: int, summary_text: str, key_symbols: list[str], metadata: dict[str, Any] | None = None) -> None:
        """Upsert a community summary.

        Args:
            community_id: Community to summarize.
            summary_text: Human-readable description of this community.
            key_symbols: Most important symbols in the community.
            metadata: Additional metadata.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO community_summaries
                    (community_id, summary_text, key_symbols, created_at, metadata)
                VALUES (?, ?, ?, datetime('now'), ?)
                ON CONFLICT(community_id) DO UPDATE SET
                    summary_text=excluded.summary_text,
                    key_symbols=excluded.key_symbols,
                    created_at=excluded.created_at,
                    metadata=excluded.metadata
                """,
                (community_id, summary_text, json.dumps(key_symbols), json.dumps(metadata or {})),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Flow snapshots
    # ------------------------------------------------------------------

    def save_flow_snapshot(self, flow_name: str, entry_point: str, nodes: list[str], edges: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> int:
        """Persist a flow snapshot (execution path through the graph).

        Args:
            flow_name: Name/label for this flow.
            entry_point: Entry-point node_id or file path.
            nodes: List of node_ids in the flow.
            edges: List of edge dicts in the flow.
            metadata: Additional metadata.

        Returns:
            Row ID of the inserted snapshot.
        """
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO flow_snapshots
                    (flow_name, entry_point, nodes_json, edges_json, created_at, metadata)
                VALUES (?, ?, ?, ?, datetime('now'), ?)
                """,
                (flow_name, entry_point, json.dumps(nodes), json.dumps(edges), json.dumps(metadata or {})),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    # ------------------------------------------------------------------
    # Session memory — cross-session context reuse
    # ------------------------------------------------------------------

    def save_session_context(
        self,
        session_id: str,
        diff_spec_hash: str,
        context_data: dict,
    ) -> None:
        """Save a context selection result for future reuse across sessions.

        Args:
            session_id: Session identifier (e.g. repo root hash).
            diff_spec_hash: SHA-256 of (sorted changed_files + query + commit_message).
            context_data: Dict with rendered_context, selected_file_paths, token stats.
        """
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_memory
                    (session_id, diff_spec_hash, context_json,
                     files_selected, tokens_rendered, reduction_ratio)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, diff_spec_hash) DO UPDATE SET
                    context_json=excluded.context_json,
                    files_selected=excluded.files_selected,
                    tokens_rendered=excluded.tokens_rendered,
                    reduction_ratio=excluded.reduction_ratio,
                    last_accessed_at=datetime('now'),
                    access_count=access_count + 1
                """,
                (
                    session_id,
                    diff_spec_hash,
                    json.dumps(context_data),
                    context_data.get("files_selected"),
                    context_data.get("total_rendered_tokens"),
                    context_data.get("reduction_ratio"),
                ),
            )
            self._conn.commit()

    def load_session_context(
        self,
        session_id: str,
        diff_spec_hash: str,
        max_age_days: int = 7,
    ) -> dict | None:
        """Load a previously cached context selection.

        Args:
            session_id: Session identifier.
            diff_spec_hash: SHA-256 of the diff specification.
            max_age_days: Maximum age in days before the entry is considered expired.

        Returns:
            The stored context_data dict, or None if missing or expired.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT *,
                       julianday('now') - julianday(created_at) AS age_days
                FROM session_memory
                WHERE session_id=? AND diff_spec_hash=?
                """,
                (session_id, diff_spec_hash),
            ).fetchone()
            if row is None:
                return None
            age = row["age_days"] if row["age_days"] is not None else 0.0
            if age > max_age_days:
                logger.info(
                    "Session memory entry expired (age=%.1f days, max=%d days)",
                    age, max_age_days,
                )
                return None
            # Bump access count and last_accessed_at
            self._conn.execute(
                """
                UPDATE session_memory
                SET access_count = access_count + 1,
                    last_accessed_at = datetime('now')
                WHERE id = ?
                """,
                (row["id"],),
            )
            self._conn.commit()
            data = json.loads(row["context_json"])
            data["_cached_at"] = row["created_at"]
            data["_access_count"] = row["access_count"] + 1
            data["_memory_id"] = row["id"]
            return data

    def list_recent_sessions(self, limit: int = 10) -> list[dict]:
        """List recent sessions with their metadata.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with session metadata (no context_json body).
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_id, diff_spec_hash,
                       files_selected, tokens_rendered, reduction_ratio,
                       created_at, last_accessed_at, access_count
                FROM session_memory
                ORDER BY last_accessed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "diff_spec_hash": r["diff_spec_hash"],
                "files_selected": r["files_selected"],
                "tokens_rendered": r["tokens_rendered"],
                "reduction_ratio": r["reduction_ratio"],
                "created_at": r["created_at"],
                "last_accessed_at": r["last_accessed_at"],
                "access_count": r["access_count"],
            }
            for r in rows
        ]

    def save_review_feedback(
        self,
        context_id: int,
        rating: int,
        notes: str = "",
    ) -> None:
        """Save user feedback on context quality (1-5 rating).

        Args:
            context_id: ID from the session_memory table.
            rating: Quality rating (1-5, 5 = best).
            notes: Optional free-text notes.
        """
        if rating < 1 or rating > 5:
            raise ValueError("rating must be between 1 and 5")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO context_feedback (context_id, rating, notes)
                VALUES (?, ?, ?)
                """,
                (context_id, rating, notes),
            )
            self._conn.commit()

    def get_context_quality_stats(self) -> dict:
        """Return aggregate stats on context quality from feedback.

        Returns:
            Dict with count, average_rating, rating_distribution, and
            recent_feedback summary.
        """
        with self._lock:
            # Overall stats
            stats_row = self._conn.execute(
                """
                SELECT COUNT(*) AS count,
                       ROUND(AVG(rating), 2) AS avg_rating,
                       MIN(rating) AS min_rating,
                       MAX(rating) AS max_rating
                FROM context_feedback
                """,
            ).fetchone()
            # Rating distribution
            dist_rows = self._conn.execute(
                """
                SELECT rating, COUNT(*) AS cnt
                FROM context_feedback
                GROUP BY rating
                ORDER BY rating
                """,
            ).fetchall()
            # Recent feedback with context info
            recent = self._conn.execute(
                """
                SELECT f.id, f.rating, f.notes, f.created_at,
                       s.session_id, s.diff_spec_hash
                FROM context_feedback f
                LEFT JOIN session_memory s ON f.context_id = s.id
                ORDER BY f.created_at DESC
                LIMIT 20
                """,
            ).fetchall()

        count = stats_row["count"] if stats_row else 0
        return {
            "total_feedback": count,
            "average_rating": stats_row["avg_rating"] if stats_row else 0.0,
            "min_rating": stats_row["min_rating"] if stats_row else 0,
            "max_rating": stats_row["max_rating"] if stats_row else 0,
            "rating_distribution": {
                str(r["rating"]): r["cnt"] for r in dist_rows
            },
            "recent_feedback": [
                {
                    "id": r["id"],
                    "rating": r["rating"],
                    "notes": r["notes"],
                    "created_at": r["created_at"],
                    "session_id": r["session_id"],
                    "diff_spec_hash": r["diff_spec_hash"],
                }
                for r in recent
            ],
        }

    # ------------------------------------------------------------------
    # Stats / schema
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return row counts across all tables.

        Returns:
            Dict with counts for nodes, edges, files, communities, risk_index,
            community_summaries, flow_snapshots, and the current schema version.
        """
        with self._lock:
            def _count(table: str) -> int:
                try:
                    return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.OperationalError:
                    return 0

            return {
                "schema_version": self._schema_version(),
                "nodes": _count("nodes"),
                "edges": _count("edges"),
                "files": _count("files"),
                "communities": _count("communities"),
                "risk_index": _count("risk_index"),
                "community_summaries": _count("community_summaries"),
                "flow_snapshots": _count("flow_snapshots"),
                "session_memory": _count("session_memory"),
                "context_feedback": _count("context_feedback"),
                "db_path": self._db_path,
            }

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
