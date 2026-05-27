"""Agent Memory Layer -- persists agent conversation context as a knowledge graph.

Stores "facts" the agent learns about the codebase across sessions, supporting
TTL-based expiry, hybrid search (BM25 + TF-IDF), cross-session consolidation,
and compact session summarization for context injection.

Usage::

    from graphsift.memory import AgentMemory

    mem = AgentMemory(db_path="/path/to/.graphsift/memory.db")
    fact_id = mem.remember("The auth module uses JWT tokens")
    results = mem.recall("authentication tokens")
    for fact in results:
        print(fact.content)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .exceptions import graphsiftError

logger = logging.getLogger(__name__)

# Regex for camelCase-aware tokenisation -- mirrors hybrid_search.py so that
# TF vectors stored during remember() align with query vectors in recall().
_TOKEN_RE = re.compile(
    r"[a-zA-Z][a-z]*|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+"
)

_CURRENT_VERSION = 1

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MemoryFact(BaseModel):
    """A single fact stored in agent memory.

    Each fact represents a piece of knowledge the agent has learned about the
    codebase -- a design decision, a discovered pattern, a known issue, etc.

    Facts support TTL-based expiry (via *valid_until*) and soft-deletion
    (via *invalid_at*).  The *access_count* and *last_accessed* fields track
    usage for summarization priority.
    """

    fact_id: str
    content: str
    session_id: str
    linked_symbols: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    valid_until: datetime | None = None
    invalid_at: datetime | None = None
    access_count: int = 0
    last_accessed: datetime

    def is_expired(self) -> bool:
        """Return True when the fact's TTL has elapsed."""
        if self.valid_until is None:
            return False
        return datetime.now(timezone.utc) > self.valid_until

    def is_deleted(self) -> bool:
        """Return True when the fact has been soft-deleted."""
        return self.invalid_at is not None

    def __repr__(self) -> str:
        status = ""
        if self.is_deleted():
            status = " [deleted]"
        elif self.is_expired():
            status = " [expired]"
        return (
            f"MemoryFact({self.fact_id!r},"
            f" {self.content[:60]!r}{status})"
        )


class SessionInfo(BaseModel):
    """Summary information about a memory session.

    Returned by :meth:`AgentMemory.list_sessions` to give an overview of all
    tracked sessions without loading every fact.
    """

    session_id: str
    fact_count: int
    created_at: datetime
    last_accessed: datetime
    summary: str = ""


# ---------------------------------------------------------------------------
# SQLite schema + migrations
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "created memory_facts and memory_fact_edges tables",
        [
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                fact_id        TEXT PRIMARY KEY,
                content        TEXT NOT NULL,
                session_id     TEXT NOT NULL,
                linked_symbols TEXT NOT NULL DEFAULT '[]',
                context        TEXT NOT NULL DEFAULT '{}',
                created_at     TEXT NOT NULL DEFAULT (datetime('now')),
                valid_until    TEXT,
                invalid_at     TEXT,
                access_count   INTEGER NOT NULL DEFAULT 0,
                last_accessed  TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_mem_facts_session
            ON memory_facts(session_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_mem_facts_valid
            ON memory_facts(valid_until)
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_fact_edges (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id   TEXT NOT NULL REFERENCES memory_facts(fact_id),
                symbol    TEXT NOT NULL,
                kind      TEXT NOT NULL DEFAULT 'references',
                UNIQUE(fact_id, symbol)
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_mem_edges_fact
            ON memory_fact_edges(fact_id)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_mem_edges_symbol
            ON memory_fact_edges(symbol)
            """,
        ],
    ),
]

# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------


class AgentMemory:
    """Session-scoped knowledge graph for agent conversations.

    Stores **facts** the agent learns about the codebase across sessions.
    Each fact lives in a **session** (a conversation scope) and may be linked
    to code symbols via REFERENCES edges.

    Facts can optionally have a **TTL**, after which they are automatically
    excluded from recall results.  Facts are **soft-deleted** on forget so
    they can be recovered if needed.

    Recall uses a hybrid BM25 + TF-IDF cosine similarity scorer that mirrors
    the :class:`~graphsift.hybrid_search.HybridSearcher` algorithm.

    Args:
        db_path: Path to the SQLite database file.  If not provided defaults
            to ``.graphsift/memory.db`` under the current working directory.
        session_id: Session identifier.  If not provided a new UUID hex
            string is generated on construction.
    """

    def __init__(
        self,
        db_path: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._session_id = session_id or uuid.uuid4().hex

        if db_path is None:
            db_path = str(Path.cwd() / ".graphsift" / "memory.db")

        self._db_path = str(db_path)
        self._lock = threading.RLock()
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._run_migrations()

    def __repr__(self) -> str:
        return (
            f"AgentMemory({self._db_path!r}, session={self._session_id!r})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Session identifier used by this instance."""
        return self._session_id

    # ------------------------------------------------------------------
    # Schema migrations
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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at  TEXT DEFAULT (datetime('now'))
                )
                """
            )
            self._conn.commit()

            current = self._schema_version()
            if current >= _CURRENT_VERSION:
                logger.debug(
                    "memory: schema already at version %d", current
                )
                return

            for version, description, statements in _MIGRATIONS:
                if version <= current:
                    continue
                logger.info("memory: running migration v%d", version)
                try:
                    for sql in statements:
                        self._conn.execute(sql.strip())
                    self._conn.execute(
                        """
                        INSERT INTO schema_migrations(version, description)
                        VALUES (?, ?)
                        """,
                        (version, description),
                    )
                    self._conn.commit()
                    logger.info(
                        "memory: migration v%d: %s", version, description
                    )
                except sqlite3.OperationalError as exc:
                    self._conn.rollback()
                    raise graphsiftError(
                        f"AgentMemory migration v{version} failed: {exc}"
                    ) from exc

            logger.info(
                "memory: migrations complete, schema version %d",
                _CURRENT_VERSION,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remember(
        self,
        fact: str,
        context: dict[str, Any] | None = None,
        linked_symbols: list[str] | None = None,
        ttl_minutes: int | None = None,
    ) -> str:
        """Store a fact with optional linkage to code symbols.

        The fact is associated with the current session.  A sparse TF vector
        is computed from the fact content (and linked symbols) and stored in
        the fact's context metadata so that subsequent calls to :meth:`recall`
        can use TF-IDF cosine similarity alongside BM25.

        Args:
            fact: Fact content (free text).
            context: Arbitrary metadata dict (e.g. file paths, line numbers).
            linked_symbols: Code symbol names (``GraphNode.node_id`` or
                qualified names) this fact relates to.
            ttl_minutes: Time-to-live in minutes.  The fact is automatically
                excluded from recall results after this duration.

        Returns:
            The assigned ``fact_id`` (uuid4 hex string).
        """
        fact_id = uuid.uuid4().hex
        now = self._now_str()

        valid_until: str | None = None
        if ttl_minutes is not None:
            valid_until = (
                datetime.now(timezone.utc)
                + timedelta(minutes=ttl_minutes)
            ).strftime("%Y-%m-%d %H:%M:%S")

        # Build a sparse TF vector and stash it in context for hybrid search.
        ctx = dict(context or {})
        tf_vec = self._build_tf_vector(fact)
        if linked_symbols:
            for sym in linked_symbols:
                sym_vec = self._build_tf_vector(sym)
                for token, weight in sym_vec.items():
                    tf_vec[token] = (
                        tf_vec.get(token, 0.0) + weight * 0.5
                    )
        if tf_vec:
            ctx["_tf_vec"] = tf_vec

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_facts
                    (fact_id, content, session_id, linked_symbols,
                     context, created_at, last_accessed, valid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_id,
                    fact,
                    self._session_id,
                    json.dumps(linked_symbols or []),
                    json.dumps(ctx),
                    now,
                    now,
                    valid_until,
                ),
            )

            if linked_symbols:
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO memory_fact_edges
                        (fact_id, symbol, kind)
                    VALUES (?, ?, 'references')
                    """,
                    [(fact_id, sym) for sym in linked_symbols],
                )

            self._conn.commit()

        logger.debug(
            "remembered fact %s in session %s (ttl=%s)",
            fact_id[:8],
            self._session_id,
            f"{ttl_minutes}m" if ttl_minutes else "none",
        )
        return fact_id

    def recall(
        self,
        query: str,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[MemoryFact]:
        """Recall facts by natural language query.

        Uses a hybrid BM25 + TF-IDF scorer that mirrors
        :class:`~graphsift.hybrid_search.HybridSearcher`.  Facts with a
        stored ``_tf_vec`` vector in their context metadata get the full
        hybrid score; facts without one fall back to BM25-only.

        Expired and soft-deleted facts are never returned.

        Args:
            query: Free-text search query.
            top_k: Maximum number of facts to return, sorted descending by
                relevance score.
            session_id: If provided, limit search to this session.  Otherwise
                search across all sessions.

        Returns:
            List of :class:`MemoryFact` instances.
        """
        if not query:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_vec = self._build_query_vector(query)
        candidates = self._load_active_facts(session_id)
        if not candidates:
            return []

        scored: list[tuple[MemoryFact, float]] = []
        for fact in candidates:
            score = self._hybrid_score(fact, query_tokens, query_vec)
            if score > 0.0:
                scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        if results:
            self._bump_access([f.fact_id for f, _ in results])

        return [fact for fact, _ in results]

    def get_session(self, session_id: str) -> list[MemoryFact]:
        """Get all active facts from a session.

        Only returns non-deleted, non-expired facts, ordered by creation time.

        Args:
            session_id: Session identifier.

        Returns:
            List of :class:`MemoryFact` instances.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memory_facts
                WHERE session_id = ?
                  AND invalid_at IS NULL
                  AND (valid_until IS NULL
                       OR valid_until > datetime('now'))
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()

        return [self._row_to_fact(r) for r in rows]

    def forget(self, fact_id: str) -> None:
        """Soft-delete a fact.

        Sets the fact's ``invalid_at`` timestamp to now so that it is excluded
        from future recall and session queries.  The row remains in the
        database and can be inspected for debugging.

        Args:
            fact_id: Fact to forget.
        """
        now = self._now_str()
        with self._lock:
            self._conn.execute(
                "UPDATE memory_facts SET invalid_at = ? WHERE fact_id = ?",
                (now, fact_id),
            )
            self._conn.commit()
        logger.debug("forgot fact %s", fact_id[:8])

    def summarize(self, session_id: str, max_tokens: int = 500) -> str:
        """Summarize a session into compact form for context injection.

        Builds a condensed multi-line string of the most-accessed facts in the
        session, capped at approximately *max_tokens* (rough 4-char-per-token
        heuristic).  Facts with linked symbols include a bracketed symbol list.

        Args:
            session_id: Session to summarize.
            max_tokens: Approximate upper bound for the summary length in
                LLM tokens.

        Returns:
            Compact string representation of the session.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memory_facts
                WHERE session_id = ?
                  AND invalid_at IS NULL
                  AND (valid_until IS NULL
                       OR valid_until > datetime('now'))
                ORDER BY access_count DESC, created_at DESC
                """,
                (session_id,),
            ).fetchall()

        if not rows:
            return f"Session {session_id!r}: no active facts."

        facts = [self._row_to_fact(r) for r in rows]
        max_chars = max_tokens * 4
        lines: list[str] = []
        char_count = 0

        for fact in facts:
            symbols_str = ""
            if fact.linked_symbols:
                symbols_str = f" [{', '.join(fact.linked_symbols)}]"
            line = f"- {fact.content}{symbols_str}"
            line_len = len(line) + 1  # +1 for newline

            if char_count + line_len > max_chars and lines:
                break
            lines.append(line)
            char_count += line_len

        return (
            f"Session {session_id!r}"
            f" ({len(facts)} active facts, showing {len(lines)}):\n"
            + "\n".join(lines)
        )

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions with fact counts and access recency.

        Returns:
            List of :class:`SessionInfo` objects sorted by last access
            descending.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    session_id,
                    COUNT(*)                                     AS fact_count,
                    MIN(created_at)                              AS created_at,
                    MAX(last_accessed)                           AS last_accessed,
                    COUNT(CASE WHEN invalid_at IS NULL
                           AND (valid_until IS NULL
                                OR valid_until > datetime('now'))
                          THEN 1 END)                            AS active_count
                FROM memory_facts
                GROUP BY session_id
                ORDER BY last_accessed DESC
                """
            ).fetchall()

        results: list[SessionInfo] = []
        for row in rows:
            try:
                results.append(
                    SessionInfo(
                        session_id=row["session_id"],
                        fact_count=row["fact_count"],
                        created_at=self._parse_dt(row["created_at"]),
                        last_accessed=self._parse_dt(
                            row["last_accessed"]
                        ),
                        summary=f"{row['active_count']} active facts",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "memory: skipping malformed session row: %s", exc
                )

        return results

    def consolidate(self, session_ids: list[str]) -> int:
        """Merge facts from old sessions into the current session.

        Deduplicates facts by content similarity (token-set Jaccard > 0.7),
        consolidates linked symbols and context metadata, then soft-deletes
        the originals.

        This is useful for reducing fragmentation after multiple short
        sessions have accumulated similar or overlapping facts.

        Args:
            session_ids: Session IDs whose active facts should be merged
                into the current session.

        Returns:
            Number of merged fact clusters (each cluster becomes one fact).
        """
        if not session_ids:
            return 0

        all_facts: list[MemoryFact] = []
        for sid in session_ids:
            all_facts.extend(self.get_session(sid))

        if not all_facts:
            return 0

        # Greedy clustering by Jaccard similarity on token sets.
        consumed: set[str] = set()
        grouped: list[list[MemoryFact]] = []

        for i, a in enumerate(all_facts):
            if a.fact_id in consumed:
                continue
            cluster: list[MemoryFact] = [a]
            consumed.add(a.fact_id)
            a_tokens = set(self._tokenize(a.content))

            for b in all_facts[i + 1:]:
                if b.fact_id in consumed:
                    continue
                b_tokens = set(self._tokenize(b.content))
                jaccard = self._jaccard_similarity(a_tokens, b_tokens)
                if jaccard > 0.7:
                    cluster.append(b)
                    consumed.add(b.fact_id)

            if len(cluster) > 1:
                grouped.append(cluster)

        # Merge each cluster into a single fact in the current session.
        now = self._now_str()
        merged_ids: list[str] = []
        merged_count = 0

        for cluster in grouped:
            seen: set[str] = set()
            merged_content_lines: list[str] = []
            merged_symbols: dict[str, float] = {}
            merged_ctx: dict[str, Any] = {}

            for f in cluster:
                if f.content not in seen:
                    seen.add(f.content)
                    merged_content_lines.append(f.content)
                for sym in f.linked_symbols:
                    merged_symbols[sym] = 1.0
                merged_ctx.update(f.context)

            merged_content = " | ".join(merged_content_lines)
            merged_symbols_list = list(merged_symbols.keys())

            # Strip internal _tf_vec -- will be rebuilt by remember().
            merged_ctx.pop("_tf_vec", None)

            self.remember(
                fact=merged_content,
                context=merged_ctx if merged_ctx else None,
                linked_symbols=(
                    merged_symbols_list if merged_symbols_list else None
                ),
            )
            merged_ids.extend(f.fact_id for f in cluster)
            merged_count += 1

        # Soft-delete originals
        if merged_ids:
            with self._lock:
                for fid in merged_ids:
                    self._conn.execute(
                        """
                        UPDATE memory_facts
                        SET invalid_at = ?
                        WHERE fact_id = ?
                        """,
                        (now, fid),
                    )
                self._conn.commit()

        logger.info(
            "consolidated %d cluster(s) from %d session(s) "
            "(%d original facts merged)",
            merged_count,
            len(session_ids),
            len(all_facts),
        )
        return merged_count

    # ------------------------------------------------------------------
    # Hybrid search internals
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenise text into lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _build_tf_vector(text: str) -> dict[str, float]:
        """Build a sparse term-frequency vector from *text*.

        Uses camelCase-aware tokenisation (the same ``_TOKEN_RE`` as
        :class:`~graphsift.hybrid_search.HybridSearcher`) so that stored
        vectors align with query vectors built by :meth:`_build_query_vector`.
        """
        tokens = _TOKEN_RE.findall(text)
        tokens = [t.lower() for t in tokens if len(t) > 1]
        if not tokens:
            return {}
        total = len(tokens)
        tf: dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        return {t: c / total for t, c in tf.items()}

    @staticmethod
    def _build_query_vector(text: str) -> dict[str, float]:
        """Build a sparse query vector (TF) from free-text *text*.

        Mirrors :meth:`HybridSearcher._build_query_vector`.
        """
        tokens = _TOKEN_RE.findall(text)
        tokens = [t.lower() for t in tokens if len(t) > 1]
        if not tokens:
            return {}
        total = len(tokens)
        tf: dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        return {t: c / total for t, c in tf.items()}

    def _hybrid_score(
        self,
        fact: MemoryFact,
        query_tokens: list[str],
        query_vec: dict[str, float],
    ) -> float:
        """Compute a hybrid BM25 + TF-IDF score for *fact* against *query*.

        BM25 is computed over the concatenation of fact content and linked
        symbols.  If the fact has a stored ``_tf_vec`` in its context, cosine
        similarity against the query vector is blended in using a 0.3 / 0.7
        weight split (matching :class:`~graphsift.hybrid_search.HybridSearcher`
        default alpha).

        Args:
            fact: The fact to score.
            query_tokens: Tokenised query terms for BM25.
            query_vec: Sparse query TF vector for cosine similarity.

        Returns:
            Score in ``[0, 1]``.
        """
        # Concatenate content + linked symbols for BM25
        doc_text = fact.content
        if fact.linked_symbols:
            doc_text += " " + " ".join(fact.linked_symbols)

        doc_terms = self._tokenize(doc_text)
        doc_freq: dict[str, int] = defaultdict(int)
        for t in doc_terms:
            doc_freq[t] += 1

        k1, b = 1.5, 0.75
        avg_dl = 20.0
        dl = len(doc_terms)
        bm25 = 0.0
        for term in query_tokens:
            if term not in doc_freq:
                continue
            tf = doc_freq[term]
            # Simplified IDF (same as HybridSearcher)
            idf = math.log(1 + 1.0 / (0.5 + 0.5))
            tf_norm = (
                tf * (k1 + 1)
            ) / (tf + k1 * (1 - b + b * dl / avg_dl))
            bm25 += idf * tf_norm

        bm25 = min(1.0, bm25 / max(len(query_tokens), 1))

        # Cosine similarity with stored TF vector
        stored_tf: dict[str, float] | None = fact.context.get("_tf_vec")
        alpha = 0.3

        if stored_tf and query_vec:
            vec_score = self._sparse_cosine(query_vec, stored_tf)
            return alpha * bm25 + (1.0 - alpha) * vec_score

        return bm25

    @staticmethod
    def _sparse_cosine(
        a: dict[str, float],
        b: dict[str, float],
    ) -> float:
        """Cosine similarity between two sparse TF dictionaries.

        Mirrors :meth:`HybridSearcher.sparse_cosine`.
        """
        if not a or not b:
            return 0.0

        if len(a) > len(b):
            a, b = b, a

        dot = 0.0
        for key, val in a.items():
            if key in b:
                dot += val * b[key]

        if dot == 0.0:
            return 0.0

        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    @staticmethod
    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not a and not b:
            return 1.0
        union = len(a | b)
        if union == 0:
            return 0.0
        return len(a & b) / union

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _load_active_facts(
        self,
        session_id: str | None = None,
    ) -> list[MemoryFact]:
        """Load non-deleted, non-expired facts, optionally filtered by session."""
        with self._lock:
            if session_id:
                rows = self._conn.execute(
                    """
                    SELECT * FROM memory_facts
                    WHERE session_id = ?
                      AND invalid_at IS NULL
                      AND (valid_until IS NULL
                           OR valid_until > datetime('now'))
                    ORDER BY created_at DESC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM memory_facts
                    WHERE invalid_at IS NULL
                      AND (valid_until IS NULL
                           OR valid_until > datetime('now'))
                    ORDER BY created_at DESC
                    """
                ).fetchall()

        return [self._row_to_fact(r) for r in rows]

    def _bump_access(self, fact_ids: list[str]) -> None:
        """Increment access count and update last_accessed for given facts."""
        now = self._now_str()
        with self._lock:
            for fid in fact_ids:
                self._conn.execute(
                    """
                    UPDATE memory_facts
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE fact_id = ?
                    """,
                    (now, fid),
                )
            self._conn.commit()

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> MemoryFact:
        """Convert a raw SQLite row into a :class:`MemoryFact` instance."""
        return MemoryFact(
            fact_id=row["fact_id"],
            content=row["content"],
            session_id=row["session_id"],
            linked_symbols=json.loads(row["linked_symbols"] or "[]"),
            context=json.loads(row["context"] or "{}"),
            created_at=AgentMemory._parse_dt(row["created_at"]),
            valid_until=AgentMemory._parse_dt_opt(row["valid_until"]),
            invalid_at=AgentMemory._parse_dt_opt(row["invalid_at"]),
            access_count=row["access_count"],
            last_accessed=AgentMemory._parse_dt(row["last_accessed"]),
        )

    @staticmethod
    def _now_str() -> str:
        """Current UTC timestamp in SQLite datetime('now') format."""
        return datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        """Parse a ``YYYY-MM-DD HH:MM:SS`` string into an aware UTC datetime."""
        try:
            return datetime.strptime(
                value, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logger.warning(
                "memory: could not parse datetime %r, using now()", value
            )
            return datetime.now(timezone.utc)

    @staticmethod
    def _parse_dt_opt(value: str | None) -> datetime | None:
        """Parse an optional SQLite datetime string."""
        if value is None:
            return None
        try:
            return datetime.strptime(
                value, "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> AgentMemory:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
