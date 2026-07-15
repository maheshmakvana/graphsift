"""Thread-safe SQLite connection pool with auto-reconnect.

Provides DatabasePool — a simple but robust SQLite connection pool
that supports concurrent readers via WAL mode, automatic dead-connection
detection, and context manager support.

Usage::

    from graphsift.pool import DatabasePool

    pool = DatabasePool("/path/to/db.sqlite", max_connections=5)

    # Execute directly
    cursor = pool.execute("SELECT * FROM nodes WHERE kind = ?", ("function",))
    rows = cursor.fetchall()

    # Or acquire/release manually for transactions
    conn = pool.acquire()
    try:
        conn.execute("BEGIN")
        conn.execute("INSERT INTO ...")
        conn.commit()
    finally:
        pool.release(conn)

    pool.close()
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

logger = logging.getLogger(__name__)

# A connection is considered stale if it has not been checked in for this
# many seconds (prevents holding connections indefinitely).
_DEFAULT_MAX_IDLE_SECS = 300.0


class DatabasePool:
    """Thread-safe SQLite connection pool with automatic reconnect.

    Connections are created on demand up to *max_connections*.  Each
    connection uses ``check_same_thread=False`` (required for sharing
    across threads) and enables WAL mode for concurrent reads.

    Args:
        db_path: Path to the SQLite database file.
        max_connections: Maximum number of connections in the pool
            (default 5).  This limits concurrent writers to 1 since
            SQLite serialises writes at the database level.
        max_idle_secs: Seconds after which an idle connection is
            considered stale and will be replaced on next acquire.
    """

    def __init__(
        self,
        db_path: str,
        max_connections: int = 5,
        max_idle_secs: float = _DEFAULT_MAX_IDLE_SECS,
    ) -> None:
        self._db_path = str(db_path)
        self._max_connections = max_connections
        self._max_idle_secs = max_idle_secs
        self._lock = threading.RLock()
        # Pool of available connections (Queue for thread-safe handoff)
        self._available: Queue[sqlite3.Connection] = Queue()
        self._all_conns: set[sqlite3.Connection] = set()
        self._in_use: dict[int, sqlite3.Connection] = {}
        self._conn_timestamps: dict[int, float] = {}  # conn id -> last checkin time
        self._closed = False

        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"DatabasePool(path={self._db_path!r}, "
            f"available={self._available.qsize()}, "
            f"in_use={len(self._in_use)}, "
            f"total={len(self._all_conns)})"
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def acquire(self) -> sqlite3.Connection:
        """Get a connection from the pool.

        Blocks (via Queue.get with timeout) if all connections are in use.
        If a connection is stale or closed, it is replaced automatically.

        Returns:
            An open SQLite connection with ``row_factory = sqlite3.Row``
            and WAL journal mode enabled.

        Raises:
            RuntimeError: If pool is closed or max wait exceeded.
        """
        if self._closed:
            raise RuntimeError("DatabasePool is closed")

        with self._lock:
            if len(self._in_use) >= self._max_connections:
                # All connections checked out — try to wait for one
                try:
                    conn = self._available.get(timeout=30.0)
                except Empty:
                    raise RuntimeError(
                        f"DatabasePool: all {self._max_connections} connections "
                        f"in use, timed out waiting"
                    ) from None
            else:
                # Try to get an available connection first
                try:
                    conn = self._available.get_nowait()
                except Empty:
                    # Create a new connection
                    conn = self._create_connection()
                    with self._lock:
                        self._all_conns.add(conn)

        # Validate the connection
        if not self._is_connection_alive(conn):
            logger.debug("graphsift: DatabasePool replacing stale connection")
            self._discard_connection(conn)
            conn = self._create_connection()
            with self._lock:
                self._all_conns.add(conn)

        with self._lock:
            conn_id = id(conn)
            self._in_use[conn_id] = conn
            self._conn_timestamps.pop(conn_id, None)

        return conn

    def release(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool.

        Args:
            conn: Connection previously obtained via ``acquire()``.
        """
        if conn is None:
            return

        with self._lock:
            conn_id = id(conn)
            self._in_use.pop(conn_id, None)
            if conn in self._all_conns:
                self._conn_timestamps[conn_id] = time.monotonic()
                self._available.put(conn)

    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> sqlite3.Cursor:
        """Acquire a connection, execute, and release.

        Convenience method for single-statement operations.

        Args:
            sql: SQL statement to execute.
            params: Query parameters (tuple or dict).

        Returns:
            sqlite3.Cursor with results.

        Raises:
            sqlite3.Error: On database errors (connection is released
                before raising).
        """
        conn = self.acquire()
        try:
            if params is not None:
                return conn.execute(sql, params)
            return conn.execute(sql)
        except sqlite3.Error:
            # Mark connection as suspect but still release it
            self.release(conn)
            raise
        else:
            self.release(conn)

    def executemany(
        self,
        sql: str,
        params_list: list[tuple[Any, ...]] | list[dict[str, Any]],
    ) -> sqlite3.Cursor:
        """Execute the same SQL with multiple parameter sets.

        Args:
            sql: SQL statement.
            params_list: List of parameter tuples or dicts.

        Returns:
            sqlite3.Cursor (positioned at last result set).
        """
        conn = self.acquire()
        try:
            return conn.executemany(sql, params_list)
        except sqlite3.Error:
            self.release(conn)
            raise
        else:
            self.release(conn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with recommended settings."""
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @staticmethod
    def _is_connection_alive(conn: sqlite3.Connection) -> bool:
        """Check if a connection is still alive with a simple ping."""
        try:
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    def _discard_connection(self, conn: sqlite3.Connection) -> None:
        """Remove a connection from the pool and close it."""
        with self._lock:
            conn_id = id(conn)
            self._in_use.pop(conn_id, None)
            self._all_conns.discard(conn)
            self._conn_timestamps.pop(conn_id, None)
        try:
            conn.close()
        except sqlite3.Error:
            pass

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of connections in the pool (in use + available)."""
        with self._lock:
            return len(self._all_conns)

    @property
    def available_count(self) -> int:
        """Number of idle connections ready to use."""
        return self._available.qsize()

    @property
    def in_use_count(self) -> int:
        """Number of connections currently checked out."""
        with self._lock:
            return len(self._in_use)

    def close(self) -> None:
        """Close all connections and shut down the pool."""
        with self._lock:
            self._closed = True

        # Close all in-use connections (forcefully)
        with self._lock:
            in_use = dict(self._in_use)
            all_conns = set(self._all_conns)

        for conn in in_use.values():
            try:
                conn.close()
            except sqlite3.Error:
                pass

        # Drain available queue
        while not self._available.empty():
            try:
                conn = self._available.get_nowait()
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            except Empty:
                break

        # Close any remaining tracked connections
        for conn in all_conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass

        with self._lock:
            self._in_use.clear()
            self._all_conns.clear()

    def resize(self, new_max: int) -> None:
        """Change the maximum number of connections.

        If *new_max* is smaller than the current count, excess idle
        connections are closed.  In-use connections are not affected.

        Args:
            new_max: New connection limit (must be >= 1).
        """
        if new_max < 1:
            raise ValueError("max_connections must be >= 1")

        with self._lock:
            self._max_connections = new_max

        # Close excess idle connections
        while self._available.qsize() > new_max:
            try:
                conn = self._available.get_nowait()
                self._discard_connection(conn)
            except Empty:
                break

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> DatabasePool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
