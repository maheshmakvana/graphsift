"""Code-anchored agent memory for GraphSift.

Every memory links to code symbols, with SQLite persistence, TTL-based
expiry, importance decay, and recall via graph proximity to changed symbols.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_TYPES = {
    "decision": {"ttl_days": 365, "description": "Architectural or design decision"},
    "gotcha": {"ttl_days": 180, "description": "Non-obvious pitfall or edge case"},
    "todo": {"ttl_days": 30, "description": "Pending task or fix"},
    "note": {"ttl_days": 90, "description": "General observation"},
    "bug": {"ttl_days": 30, "description": "Known bug or issue"},
    "insight": {"ttl_days": 180, "description": "Deep understanding gained"},
}


class CodeMemory:
    """Code-anchored agent memory. Every memory links to code symbols."""

    def __init__(self, db_path: str = None, graph=None) -> None:
        self._db_path = db_path or str(Path.home() / ".graphsift" / "code_memory.db")
        self._graph = graph
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS code_memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'note',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
                    access_count INTEGER NOT NULL DEFAULT 0,
                    valid_until TEXT,
                    invalid_at TEXT,
                    session_id TEXT,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS memory_symbols (
                    memory_id TEXT NOT NULL,
                    symbol_name TEXT NOT NULL,
                    PRIMARY KEY (memory_id, symbol_name),
                    FOREIGN KEY (memory_id) REFERENCES code_memories(memory_id)
                );
                CREATE TABLE IF NOT EXISTS memory_files (
                    memory_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    PRIMARY KEY (memory_id, filepath),
                    FOREIGN KEY (memory_id) REFERENCES code_memories(memory_id)
                );
                CREATE TABLE IF NOT EXISTS memory_tags (
                    memory_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (memory_id, tag),
                    FOREIGN KEY (memory_id) REFERENCES code_memories(memory_id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_type ON code_memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_session ON code_memories(session_id);
                CREATE INDEX IF NOT EXISTS idx_memories_valid ON code_memories(valid_until, invalid_at);
            """)

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        linked_symbols: list[str] = None,
        linked_files: list[str] = None,
        memory_type: str = "note",
        importance: float = 0.5,
        ttl_days: int = None,
        tags: list[str] = None,
        session_id: str = None,
    ) -> str:
        """Store a memory linked to code symbols and files. Returns memory_id."""
        if ttl_days is None:
            ttl_days = MEMORY_TYPES.get(memory_type, {}).get("ttl_days", 90)
        memory_id = hashlib.sha256(f"{content}{time.time()}{linked_symbols}".encode()).hexdigest()[:16]
        valid_until = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "INSERT INTO code_memories (memory_id, content, memory_type, importance, valid_until, session_id, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (memory_id, content, memory_type, importance, valid_until, session_id, json.dumps({})),
                )
                for sym in linked_symbols or []:
                    conn.execute("INSERT OR IGNORE INTO memory_symbols (memory_id, symbol_name) VALUES (?, ?)", (memory_id, sym))
                for fp in linked_files or []:
                    conn.execute("INSERT OR IGNORE INTO memory_files (memory_id, filepath) VALUES (?, ?)", (memory_id, fp))
                for tag in tags or []:
                    conn.execute("INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)", (memory_id, tag))
        logger.info("Stored memory %s (type=%s, symbols=%d, files=%d)", memory_id, memory_type, len(linked_symbols or []), len(linked_files or []))
        return memory_id

    # ------------------------------------------------------------------
    # Recall API
    # ------------------------------------------------------------------

    def recall_for_diff(self, changed_files: list[str], changed_symbols: list[str] = None,
                        query: str = None, top_k: int = 10) -> list[CodeMemoryEntry]:
        """Get memories relevant to a diff. Ranks by symbol match > file match > graph proximity."""
        results: list[tuple[CodeMemoryEntry, float]] = []
        changed_symbols = changed_symbols or []
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM code_memories WHERE invalid_at IS NULL AND valid_until > datetime('now') ORDER BY importance DESC"
                ).fetchall()
                for row in rows:
                    entry = self._row_to_entry(row, conn)
                    score = 0.0
                    sym_matches = sum(1 for s in entry.linked_symbols if s in changed_symbols)
                    if sym_matches > 0:
                        score += entry.importance * 1.0 * sym_matches
                    file_matches = sum(1 for f in entry.linked_files if f in changed_files)
                    if file_matches > 0:
                        score += entry.importance * 0.8 * file_matches
                    if self._graph and entry.linked_symbols:
                        for sym in entry.linked_symbols:
                            for changed_sym in changed_symbols:
                                dist = self._graph_distance(sym, changed_sym)
                                if dist == 1:
                                    score += entry.importance * 0.5
                                elif dist == 2:
                                    score += entry.importance * 0.3
                    if query:
                        query_terms = set(query.lower().split())
                        content_terms = set(entry.content.lower().split())
                        overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
                        score += entry.importance * 0.2 * overlap
                    if score > 0:
                        results.append((entry, score))
        results.sort(key=lambda x: x[1], reverse=True)
        for entry, _ in results[:top_k]:
            self._touch(entry.memory_id)
        return [e for e, _ in results[:top_k]]

    def recall(self, query: str, top_k: int = 10) -> list[CodeMemoryEntry]:
        """Text-based recall using keyword overlap on memory content, boosted by importance and recency."""
        query_terms = set(query.lower().split())
        results: list[tuple[CodeMemoryEntry, float]] = []
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM code_memories WHERE invalid_at IS NULL AND valid_until > datetime('now')"
                ).fetchall()
                for row in rows:
                    entry = self._row_to_entry(row, conn)
                    content_terms = set(entry.content.lower().split())
                    overlap = len(query_terms & content_terms) / max(len(query_terms), 1)
                    if overlap > 0:
                        days_since = (datetime.utcnow() - entry.last_accessed).days
                        recency = 1.0 / (1.0 + max(days_since, 0) * 0.1)
                        score = overlap * entry.importance * recency
                        results.append((entry, score))
        results.sort(key=lambda x: x[1], reverse=True)
        for entry, _ in results[:top_k]:
            self._touch(entry.memory_id)
        return [e for e, _ in results[:top_k]]

    def recall_for_symbol(self, symbol_name: str) -> list[CodeMemoryEntry]:
        """Get all active memories linked to a specific symbol."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT cm.* FROM code_memories cm
                       INNER JOIN memory_symbols ms ON cm.memory_id = ms.memory_id
                       WHERE ms.symbol_name = ? AND cm.invalid_at IS NULL AND cm.valid_until > datetime('now')
                       ORDER BY cm.importance DESC""",
                    (symbol_name,),
                ).fetchall()
                return [self._row_to_entry(row, conn) for row in rows]

    def recall_for_file(self, filepath: str) -> list[CodeMemoryEntry]:
        """Get all active memories linked to a specific file."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT cm.* FROM code_memories cm
                       INNER JOIN memory_files mf ON cm.memory_id = mf.memory_id
                       WHERE mf.filepath = ? AND cm.invalid_at IS NULL AND cm.valid_until > datetime('now')
                       ORDER BY cm.importance DESC""",
                    (filepath,),
                ).fetchall()
                return [self._row_to_entry(row, conn) for row in rows]

    # ------------------------------------------------------------------
    # Maintenance API
    # ------------------------------------------------------------------

    def forget(self, memory_id: str) -> bool:
        """Soft-delete a memory by setting invalid_at."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.execute(
                    "UPDATE code_memories SET invalid_at = datetime('now') WHERE memory_id = ? AND invalid_at IS NULL",
                    (memory_id,),
                )
                return cur.rowcount > 0

    def decay(self) -> int:
        """Apply decay: soft-delete expired memories, reduce importance of survivors by 2%."""
        count = 0
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                cur = conn.execute(
                    "UPDATE code_memories SET invalid_at = datetime('now') WHERE invalid_at IS NULL AND valid_until < datetime('now')",
                )
                count = cur.rowcount
                conn.execute(
                    "UPDATE code_memories SET importance = MAX(0.1, importance * 0.98) WHERE invalid_at IS NULL",
                )
        return count

    def summarize_for_context(self, changed_files: list[str], max_tokens: int = 300) -> str:
        """One-paragraph summary of relevant memories for context injection."""
        entries = self.recall_for_diff(changed_files, top_k=10)
        if not entries:
            return ""
        lines = ["## Relevant Past Context"]
        char_count = len(lines[0])
        for e in entries:
            line = f"- [{e.memory_type}] {e.content[:120]}"
            if char_count + len(line) > max_tokens * 4:
                break
            lines.append(line)
            char_count += len(line)
        return "\n".join(lines)

    @property
    def stats(self) -> CodeMemoryStats:
        """Current store statistics."""
        with self._lock:
            with sqlite3.connect(self._db_path) as conn:
                total = conn.execute("SELECT COUNT(*) FROM code_memories").fetchone()[0]
                active = conn.execute(
                    "SELECT COUNT(*) FROM code_memories WHERE invalid_at IS NULL AND valid_until > datetime('now')"
                ).fetchone()[0]
                expired = total - active
                by_type = {
                    row[0]: row[1]
                    for row in conn.execute(
                        "SELECT memory_type, COUNT(*) FROM code_memories WHERE invalid_at IS NULL GROUP BY memory_type"
                    ).fetchall()
                }
                avg_imp = conn.execute("SELECT AVG(importance) FROM code_memories WHERE invalid_at IS NULL").fetchone()[0] or 0
                total_acc = conn.execute("SELECT SUM(access_count) FROM code_memories").fetchone()[0] or 0
                return CodeMemoryStats(
                    total_memories=total, active_memories=active, expired_memories=expired,
                    by_type=by_type, avg_importance=round(avg_imp, 3), total_accesses=total_acc,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_entry(self, row: sqlite3.Row, conn: sqlite3.Connection) -> CodeMemoryEntry:
        memory_id = row["memory_id"]
        symbols = [r[0] for r in conn.execute("SELECT symbol_name FROM memory_symbols WHERE memory_id = ?", (memory_id,)).fetchall()]
        files = [r[0] for r in conn.execute("SELECT filepath FROM memory_files WHERE memory_id = ?", (memory_id,)).fetchall()]
        tags = [r[0] for r in conn.execute("SELECT tag FROM memory_tags WHERE memory_id = ?", (memory_id,)).fetchall()]
        return CodeMemoryEntry(
            memory_id=memory_id, content=row["content"], memory_type=row["memory_type"],
            importance=row["importance"], created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]), access_count=row["access_count"],
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else datetime.utcnow(),
            invalid_at=datetime.fromisoformat(row["invalid_at"]) if row["invalid_at"] else None,
            session_id=row["session_id"], linked_symbols=symbols, linked_files=files, tags=tags,
        )

    def _touch(self, memory_id: str) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    "UPDATE code_memories SET last_accessed = datetime('now'), access_count = access_count + 1 WHERE memory_id = ?",
                    (memory_id,),
                )
        except Exception:
            pass

    def _graph_distance(self, sym1: str, sym2: str) -> int:
        """Approximate graph distance between two symbols. Returns 999 if unreachable."""
        if sym1 == sym2:
            return 0
        if not self._graph:
            return 999
        try:
            node1 = self._graph.nodes.get(sym1)
            if node1 is None:
                return 999
            for edge_list in self._graph.edges.values():
                for edge in edge_list:
                    if hasattr(edge, "source") and hasattr(edge, "target"):
                        if edge.source == sym1 and edge.target == sym2:
                            return 1
            return 999
        except Exception:
            return 999


@dataclass
class CodeMemoryEntry:
    """A single code-anchored memory entry."""

    memory_id: str
    content: str
    memory_type: str
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    valid_until: datetime
    invalid_at: datetime | None = None
    session_id: str | None = None
    linked_symbols: list[str] = field(default_factory=list)
    linked_files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.invalid_at is None and self.valid_until > datetime.utcnow()


@dataclass
class CodeMemoryStats:
    """Aggregate statistics for a CodeMemory store."""

    total_memories: int = 0
    active_memories: int = 0
    expired_memories: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    avg_importance: float = 0.0
    total_accesses: int = 0
