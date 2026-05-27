"""Async task manager for long-running MCP operations (SEP-1686).

Provides:
  - TaskManager       — generic async task lifecycle via background threads
  - Pre-built tasks   — index_repo, analyze_diff, build_graph
  - ToolRegistry      — progressive tool loading via named categories

Usage::

    manager = TaskManager(max_concurrent=3)
    task_id = manager.create("index_repo", metadata={"root": "/path"})
    manager.start(task_id, task_index_repo, "/path", store)

    status = manager.status(task_id)
    # Task(name='index_repo', state=TaskState.RUNNING, progress=0.5, ...)
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ===========================================================================
# Task State & Data
# ===========================================================================


class TaskState(str, Enum):
    """MCP task lifecycle states per SEP-1686."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a single async MCP task.

    Attributes:
        task_id: Unique identifier.
        name: Human-readable task name.
        state: Current lifecycle state.
        progress: Float 0.0–1.0.
        progress_message: Human-readable progress description.
        result: Return value of the task function (set on COMPLETED).
        error: Error message (set on FAILED).
        created_at: ``time.monotonic()`` timestamp of creation.
        started_at: Timestamp when the task began running.
        completed_at: Timestamp when the task reached a terminal state.
        metadata: Arbitrary key-value data provided at creation time.
    """

    task_id: str
    name: str
    state: TaskState = TaskState.PENDING
    progress: float = 0.0
    progress_message: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Task({self.task_id[:8]}..., name={self.name!r}, "
            f"state={self.state.value}, progress={self.progress:.1%})"
        )

    @property
    def duration_ms(self) -> float | None:
        """Wall-clock duration in milliseconds, or None if not yet completed."""
        if self.completed_at is not None and self.started_at is not None:
            return (self.completed_at - self.started_at) * 1000
        if self.started_at is not None:
            return (time.monotonic() - self.started_at) * 1000
        return None

    @property
    def is_terminal(self) -> bool:
        """True if the task is in a terminal state."""
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        )


# ===========================================================================
# Task Manager
# ===========================================================================


class TaskManager:
    """Manages async task lifecycle for MCP operations.

    Thread-safe.  Creates tasks, runs them in background daemon threads,
    supports progress reporting, cooperative cancellation, and periodic
    cleanup of stale entries.

    Args:
        max_concurrent: Maximum number of tasks running simultaneously.
            Raise to allow more parallel indexing operations.
    """

    def __init__(self, max_concurrent: int = 3) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.RLock()
        self._max_concurrent = max_concurrent
        self._running_count = 0
        self._cancel_flags: dict[str, threading.Event] = {}

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"TaskManager(tasks={len(self._tasks)}, "
                f"running={self._running_count}/{self._max_concurrent})"
            )

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new task in PENDING state and return its task_id.

        Args:
            name: Human-readable task name (e.g. ``"index_repo"``).
            metadata: Optional key-value metadata attached to the task.

        Returns:
            Unique 32-char hex task ID.
        """
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = Task(
                task_id=task_id,
                name=name,
                metadata=metadata or {},
            )
        logger.info(
            "graphsift: task created",
            extra={"task_id": task_id, "name": name},
        )
        return task_id

    def start(
        self,
        task_id: str,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Execute *fn* in a background daemon thread.

        The function *fn* may accept a ``progress_callback`` keyword argument
        — a callable ``(percent: float, message: str) -> None`` — to report
        progress.  If *fn* does not accept it the callback is silently omitted.

        Args:
            task_id: Task ID from :meth:`create`.
            fn: Callable to execute in the background.
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            True if the task was started, False if it was already running /
            pending / cancelled, or the concurrency limit is reached.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                logger.warning(
                    "graphsift: start called for unknown task",
                    extra={"task_id": task_id},
                )
                return False
            if task.state != TaskState.PENDING:
                logger.warning(
                    "graphsift: start called for non-pending task",
                    extra={"task_id": task_id, "state": task.state.value},
                )
                return False
            if self._running_count >= self._max_concurrent:
                logger.warning(
                    "graphsift: concurrency limit reached",
                    extra={"running": self._running_count, "max": self._max_concurrent},
                )
                return False

            task.state = TaskState.RUNNING
            task.started_at = time.monotonic()
            self._running_count += 1

        # Cancel flag for cooperative cancellation
        self._cancel_flags[task_id] = threading.Event()

        # Build progress callback closure
        def _progress(pct: float, msg: str) -> None:
            with self._lock:
                t = self._tasks.get(task_id)
                if t is None:
                    return
                t.progress = max(0.0, min(1.0, pct))
                t.progress_message = msg

        t = threading.Thread(
            target=self._run_task,
            args=(task_id, fn),
            kwargs={
                "progress_callback": _progress,
                "args": args,
                "kwargs": kwargs,
            },
            daemon=True,
        )
        t.start()
        return True

    def _run_task(
        self,
        task_id: str,
        fn: Callable[..., Any],
        progress_callback: Callable[[float, str], None],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Internal: execute *fn*, capture result/error, update task state."""
        try:
            # Only pass progress_callback if fn accepts it
            sig = inspect.signature(fn)
            has_progress = "progress_callback" in sig.parameters

            if has_progress:
                result = fn(*args, **kwargs, progress_callback=progress_callback)
            else:
                result = fn(*args, **kwargs)

            # Final state check (race: task may have been cancelled mid-flight)
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    return
                if self._cancel_flags.get(task_id, threading.Event()).is_set():
                    task.state = TaskState.CANCELLED
                    task.completed_at = time.monotonic()
                    logger.info(
                        "graphsift: task cancelled",
                        extra={"task_id": task_id},
                    )
                else:
                    task.state = TaskState.COMPLETED
                    task.result = result
                    task.progress = 1.0
                    task.completed_at = time.monotonic()
                    logger.info(
                        "graphsift: task completed",
                        extra={"task_id": task_id},
                    )
        except Exception as exc:
            logger.exception(
                "graphsift: task failed",
                extra={"task_id": task_id},
            )
            with self._lock:
                task = self._tasks.get(task_id)
                if task is not None:
                    task.state = TaskState.FAILED
                    task.error = str(exc)
                    task.completed_at = time.monotonic()
        finally:
            with self._lock:
                self._running_count = max(0, self._running_count - 1)
            self._cancel_flags.pop(task_id, None)

    # ------------------------------------------------------------------
    # Status & query
    # ------------------------------------------------------------------

    def status(self, task_id: str) -> Task | None:
        """Return the current task snapshot, or None if unknown.

        Returns:
            Task dataclass with latest state, progress, result or error.
        """
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Request cooperative cancellation of a running task.

        Sets the task state to CANCELLED immediately and signals the
        cancel event.  The task function is responsible for checking
        ``progress_callback`` or its own state — cancellation is
        **cooperative**, not forced.

        Args:
            task_id: Task ID to cancel.

        Returns:
            True if cancellation was requested, False if the task was
            not running or unknown.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.state != TaskState.RUNNING:
                return False
            task.state = TaskState.CANCELLED
            task.completed_at = time.monotonic()

        event = self._cancel_flags.get(task_id)
        if event is not None:
            event.set()

        logger.info("graphsift: cancel requested", extra={"task_id": task_id})
        return True

    def list_tasks(self, state: TaskState | None = None) -> list[Task]:
        """List all tasks, optionally filtered by state.

        Args:
            state: If set, only return tasks in this state.

        Returns:
            List of Task objects, newest first.
        """
        with self._lock:
            tasks = list(self._tasks.values())
        if state is not None:
            tasks = [t for t in tasks if t.state == state]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    def wait(self, task_id: str, timeout: float | None = None) -> Task:
        """Block until the task reaches a terminal state.

        Args:
            task_id: Task ID to wait on.
            timeout: Maximum seconds to wait.  None = wait indefinitely.

        Returns:
            Final Task state.

        Raises:
            TimeoutError: If *timeout* is exceeded.
            LookupError: If *task_id* is unknown.
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        while True:
            with self._lock:
                task = self._tasks.get(task_id)
                if task is None:
                    raise LookupError(f"Unknown task: {task_id}")
                if task.is_terminal:
                    return task

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Task {task_id} did not complete within {timeout}s",
                )
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self, max_age_seconds: float = 3600.0) -> int:
        """Remove terminal tasks older than *max_age_seconds*.

        Args:
            max_age_seconds: Remove tasks whose ``completed_at`` timestamp
                is older than this (default 1 hour).

        Returns:
            Number of tasks removed.
        """
        now = time.monotonic()
        removed = 0
        with self._lock:
            to_remove = [
                tid
                for tid, task in self._tasks.items()
                if task.is_terminal
                and task.completed_at is not None
                and (now - task.completed_at) > max_age_seconds
            ]
            for tid in to_remove:
                del self._tasks[tid]
                removed += 1
        if removed:
            logger.info(
                "graphsift: task cleanup removed %d tasks",
                removed,
            )
        return removed


# ===========================================================================
# Pre-built task functions
# ===========================================================================


def _noop_progress(pct: float, msg: str) -> None:
    """No-op progress callback for direct (non-TaskManager) usage."""
    del pct, msg


def task_index_repo(
    root_path: str,
    store: Any,
    config: dict[str, Any] | None = None,
    progress_callback: Callable[[float, str], None] = _noop_progress,
) -> dict[str, Any]:
    """Async task: index a repository.  Reports progress as files are parsed.

    Args:
        root_path: Root directory of the repository.
        store: ``GraphStore`` instance for SQLite persistence.
        config: Optional dict with keys:
            - extensions: set of file extensions to include.
            - exclude_dirs: set of directory names to skip.
            - progress_interval: report progress every N files.
        progress_callback: Called with ``(percent, message)`` during indexing.

    Returns:
        Dict with ``status``, ``files_indexed``, ``symbols_extracted``,
        ``edges_created``, ``duration_ms``.
    """
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    cfg = config or {}
    extensions = cfg.get("extensions") or {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    }
    exclude_dirs = cfg.get("exclude_dirs") or {
        "venv", ".venv", "node_modules", ".git", "__pycache__",
        "dist", "build", ".mypy_cache", ".pytest_cache",
    }
    progress_interval = int(cfg.get("progress_interval", 200))

    progress_callback(0.0, "Loading source map...")
    source_map = load_source_map(
        root_path,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
    )
    total_files = len(source_map)
    progress_callback(0.05, f"Source map loaded: {total_files} files")

    builder = ContextBuilder(ContextConfig())
    parsed_count = 0
    all_nodes: list[Any] = []
    all_file_nodes: list[Any] = []

    for path in source_map:
        source = source_map[path]
        try:
            builder.index_file(path, source)
        except Exception:  # noqa: BLE001
            logger.debug("task_index_repo: skipped %s", path)

        parsed_count += 1
        if progress_interval > 0 and parsed_count % progress_interval == 0:
            pct = 0.05 + 0.85 * (parsed_count / max(total_files, 1))
            progress_callback(pct, f"Indexing: {parsed_count}/{total_files} files")

    progress_callback(0.90, "Indexing complete. Building edges...")

    with builder._lock:
        stats = builder.index_files(source_map)
        graph = getattr(builder, "_graph", None)

    # Persist to SQLite
    if graph is not None and store is not None:
        try:
            from graphsift.models import GraphNode, NodeKind as _NodeKind

            for file_node in graph.all_files():
                all_file_nodes.append(file_node)
                for sym in file_node.symbols:
                    if hasattr(sym, "node_id"):
                        all_nodes.append(sym)
                    else:
                        all_nodes.append(
                            GraphNode(
                                node_id=f"{file_node.path}::{sym}",
                                file_path=file_node.path,
                                kind=_NodeKind.FUNCTION,
                                name=str(sym),
                                qualified_name=str(sym),
                                language=file_node.language,
                            ),
                        )
            store.save_nodes(all_nodes)
            store.save_files(all_file_nodes)
            progress_callback(0.98, f"Persisted {len(all_nodes)} nodes")
        except Exception as exc:  # noqa: BLE001
            logger.warning("task_index_repo: SQLite persist failed: %s", exc)

    progress_callback(1.0, "Indexing complete")
    return {
        "status": "indexed",
        "root": root_path,
        "files_indexed": stats.files_indexed,
        "files_skipped": stats.files_skipped,
        "symbols_extracted": stats.symbols_extracted,
        "edges_created": stats.edges_created,
        "duration_ms": stats.duration_ms,
    }


def task_analyze_diff(
    root_path: str,
    diff_spec: dict[str, Any],
    store: Any,
    config: dict[str, Any] | None = None,
    progress_callback: Callable[[float, str], None] = _noop_progress,
) -> dict[str, Any]:
    """Async task: analyze a diff with full evidence tracing.

    Loads the built graph, runs ranked neighbor traversal, and computes
    blast radius with per-file evidence (reasons).

    Args:
        root_path: Repository root.
        diff_spec: Dict with ``changed_files`` list, optional ``query``,
            ``diff_text``, ``commit_message``.
        store: ``GraphStore`` for cross-session memory.
        config: Optional dict with ``token_budget``, ``max_depth``, etc.
        progress_callback: Called with ``(percent, message)``.

    Returns:
        Dict with ``rendered_context``, ``files_selected``, ``affected_files``,
        and token savings stats.
    """
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig, DiffSpec

    progress_callback(0.0, "Loading graph...")

    cfg = config or {}
    builder = ContextBuilder(
        ContextConfig(
            token_budget=int(cfg.get("token_budget", 60_000)),
            max_depth=int(cfg.get("max_depth", 4)),
            session_id=cfg.get("session_id", ""),
        ),
        store=store,
    )

    graph = getattr(builder, "_graph", None)
    progress_callback(0.1, "Graph loaded")

    diff = DiffSpec(
        changed_files=diff_spec.get("changed_files", []),
        query=diff_spec.get("query", ""),
        diff_text=diff_spec.get("diff_text", ""),
        commit_message=diff_spec.get("commit_message", ""),
    )

    if not diff.changed_files:
        return {
            "error": "No changed files specified.",
            "rendered_context": "",
            "files_selected": 0,
        }

    progress_callback(0.2, "Running graph traversal...")

    if graph is not None:
        scores = graph.ranked_neighbors(
            seed_paths=diff.changed_files,
            include_dynamic=True,
        )
        all_files = graph.all_files()
        ranked = builder._ranker.rank(
            diff,
            scores,
            all_files,
            builder._config,
        )
        progress_callback(0.5, "Relevance ranking complete")

        selected, context, orig_tokens, rendered_tokens = (
            builder._selector.select_and_render(ranked, {}, diff)
        )
        progress_callback(0.8, "Context rendered")

        reduction = 1.0 - (rendered_tokens / max(orig_tokens, 1))

        affected = [
            {"path": p, "score": round(s[0], 3), "depth": s[1], "reasons": s[2][:3]}
            for p, s in sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:50]
        ]

        progress_callback(1.0, "Analysis complete")
        return {
            "status": "analyzed",
            "rendered_context": context,
            "files_selected": len(selected),
            "files_scanned": len(all_files),
            "total_original_tokens": orig_tokens,
            "total_rendered_tokens": rendered_tokens,
            "token_savings_pct": round(reduction * 100, 1),
            "affected_files": affected,
            "total_affected": len(affected),
        }

    return {
        "error": "Graph not built yet. Run index_repo or build_graph first.",
        "rendered_context": "",
        "files_selected": 0,
    }


def task_build_graph(
    root_path: str,
    store: Any,
    config: dict[str, Any] | None = None,
    progress_callback: Callable[[float, str], None] = _noop_progress,
) -> dict[str, Any]:
    """Async task: rebuild dependency graph for a repo in a single pass.

    Indexes all source files, builds import/inheritance/decorator edges,
    persists to SQLite, and runs post-processing (flow detection, community
    detection, risk scoring, FTS rebuild).

    Args:
        root_path: Repository root.
        store: ``GraphStore`` instance for persistence.
        config: Optional dict with extensions, exclude_dirs, postprocess flags.
        progress_callback: Called with ``(percent, message)``.

    Returns:
        Dict with full index + postprocess results.
    """
    from graphsift.adapters.filesystem import load_source_map
    from graphsift.core import ContextBuilder
    from graphsift.models import ContextConfig

    cfg = config or {}
    extensions = cfg.get("extensions") or {
        ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    }
    exclude_dirs = cfg.get("exclude_dirs") or {
        "venv", ".venv", "node_modules", ".git", "__pycache__",
        "dist", "build", ".mypy_cache", ".pytest_cache",
    }

    progress_callback(0.0, "Loading source map...")
    source_map = load_source_map(
        root_path,
        extensions=extensions,
        exclude_dirs=exclude_dirs,
    )
    total_files = len(source_map)
    progress_callback(0.02, f"Source map loaded: {total_files} files")

    builder = ContextBuilder(ContextConfig())
    parsed_count = 0

    for path in source_map:
        source = source_map[path]
        try:
            builder.index_file(path, source)
        except Exception:  # noqa: BLE001
            logger.debug("task_build_graph: skipped %s", path)
        parsed_count += 1
        if parsed_count % max(1, total_files // 20) == 0:
            pct = 0.02 + 0.60 * (parsed_count / max(total_files, 1))
            progress_callback(pct, f"Indexing: {parsed_count}/{total_files} files")

    progress_callback(0.65, "Building edges...")

    with builder._lock:
        stats = builder.index_files(source_map)
        graph = getattr(builder, "_graph", None)

    if graph is None:
        return {"error": "Graph construction failed.", "status": "failed"}

    # Persist to SQLite
    progress_callback(0.72, "Persisting to SQLite...")
    if store is not None:
        try:
            from graphsift.models import GraphNode, NodeKind as _NodeKind

            all_nodes: list[Any] = []
            all_file_nodes: list[Any] = []
            for file_node in graph.all_files():
                all_file_nodes.append(file_node)
                for sym in file_node.symbols:
                    if hasattr(sym, "node_id"):
                        all_nodes.append(sym)
                    else:
                        all_nodes.append(
                            GraphNode(
                                node_id=f"{file_node.path}::{sym}",
                                file_path=file_node.path,
                                kind=_NodeKind.FUNCTION,
                                name=str(sym),
                                qualified_name=str(sym),
                                language=file_node.language,
                            ),
                        )
            store.save_nodes(all_nodes)
            store.save_files(all_file_nodes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("task_build_graph: SQLite persist failed: %s", exc)

    # Postprocessing
    do_flows = cfg.get("flows", True)
    do_communities = cfg.get("communities", True)
    do_fts = cfg.get("fts", True)
    do_risk = cfg.get("risk", True)

    postprocess_results: dict[str, Any] = {}
    if do_flows or do_communities or do_fts or do_risk:
        progress_callback(0.80, "Running postprocessing...")
        try:
            from graphsift.adapters.postprocess import Postprocessor

            pp = Postprocessor()
            pp_result = pp.run(
                graph,
                store,
                source_map,
                flows=do_flows,
                communities=do_communities,
                risk=do_risk,
                fts=do_fts,
            )
            postprocess_results = pp_result
            progress_callback(0.95, "Postprocessing complete")
        except Exception as exc:  # noqa: BLE001
            logger.warning("task_build_graph: postprocess failed: %s", exc)
            postprocess_results = {"error": str(exc)}

    progress_callback(1.0, "Graph build complete")
    return {
        "status": "built",
        "root": root_path,
        "files_indexed": stats.files_indexed,
        "files_skipped": stats.files_skipped,
        "symbols_extracted": stats.symbols_extracted,
        "edges_created": stats.edges_created,
        "duration_ms": stats.duration_ms,
        "postprocess": postprocess_results,
    }


# ===========================================================================
# Tool Registry with Progressive Disclosure
# ===========================================================================


@dataclass
class ToolDef:
    """Definition of a single MCP tool.

    Attributes:
        tool_id: Unique tool name (e.g. ``"build_graph"``).
        category: Category name (e.g. ``"indexing"``).
        short_description: One-line description (used in progressive listing).
        full_schema: Complete ``inputSchema`` JSON Schema dict.
        handler: Callable accepting a params dict and returning a result dict.
    """

    tool_id: str
    category: str
    short_description: str
    full_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]

    def __repr__(self) -> str:
        return f"ToolDef({self.tool_id!r}, category={self.category!r})"


@dataclass
class ToolCategory:
    """Metadata for a tool category (progressive disclosure).

    Attributes:
        name: Category name (e.g. ``"indexing"``).
        description: One-liner describing the category.
        tool_count: Number of tools in the category.
        tool_ids: Ordered list of tool IDs.
    """

    name: str
    description: str
    tool_count: int = 0
    tool_ids: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"ToolCategory(name={self.name!r}, tools={self.tool_count})"


class ToolRegistry:
    """MCP tool registry with progressive disclosure loading.

    Tools are organised into named categories.  A lightweight
    :meth:`get_categories` call returns category names and counts only
    (low token cost).  Full schemas are loaded per-category on demand
    via :meth:`get_tools_for_category`.  Non-progressive clients can
    use :meth:`get_all_tools`.

    Usage::

        registry = ToolRegistry()

        registry.register(
            "build_graph", "indexing",
            "Index all source files and build the dependency graph.",
            {"type": "object", "properties": {...}},
            my_handler,
        )

        # Progressive client:
        cats = registry.get_categories()
        # [{"name": "indexing", "description": "...", "tool_count": 1}]

        tools = registry.get_tools_for_category("indexing")
        # [{"name": "build_graph", "description": "...", "inputSchema": {...}}]

        # Non-progressive client:
        all_tools = registry.get_all_tools()
    """

    def __init__(self) -> None:
        self._categories: dict[str, ToolCategory] = {}
        self._tools: dict[str, ToolDef] = {}
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"ToolRegistry(categories={len(self._categories)}, "
                f"tools={len(self._tools)})"
            )

    def register(
        self,
        tool_id: str,
        category: str,
        description: str,
        full_schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        category_description: str = "",
    ) -> None:
        """Register a tool under a named category.

        If the category does not exist yet it is created with an optional
        *category_description* (a sensible default is inferred for known
        category names).

        Args:
            tool_id: Unique tool name.
            category: Category name.
            description: Short one-line tool description.
            full_schema: Complete ``inputSchema`` JSON Schema dict.
            handler: Callable accepting ``params dict -> result dict``.
            category_description: One-liner for the category itself.
                Only used when the category is first created.
        """
        with self._lock:
            if category not in self._categories:
                self._categories[category] = ToolCategory(
                    name=category,
                    description=category_description
                    or self._default_cat_desc(category),
                    tool_ids=[],
                )

            self._tools[tool_id] = ToolDef(
                tool_id=tool_id,
                category=category,
                short_description=description,
                full_schema=full_schema,
                handler=handler,
            )

            cat = self._categories[category]
            cat.tool_count += 1
            if tool_id not in cat.tool_ids:
                cat.tool_ids.append(tool_id)

    def get_categories(self) -> list[dict[str, Any]]:
        """Return lightweight category list (names only, minimal tokens).

        Returns:
            List of dicts with ``name``, ``description``, and ``tool_count``.
        """
        with self._lock:
            return [
                {
                    "name": c.name,
                    "description": c.description,
                    "tool_count": c.tool_count,
                }
                for c in sorted(
                    self._categories.values(),
                    key=lambda x: x.name,
                )
            ]

    def get_tools_for_category(self, category: str) -> list[dict[str, Any]]:
        """Return full tool descriptors for *category* (loaded on demand).

        Args:
            category: Category name.

        Returns:
            List of tool descriptors with ``name``, ``description``,
            and ``inputSchema``.

        Raises:
            KeyError: If *category* does not exist.
        """
        with self._lock:
            if category not in self._categories:
                raise KeyError(f"Unknown category: {category}")

            tools: list[dict[str, Any]] = []
            for tid in self._categories[category].tool_ids:
                tool = self._tools.get(tid)
                if tool is None:
                    continue
                tools.append({
                    "name": tool.tool_id,
                    "description": tool.short_description,
                    "inputSchema": tool.full_schema,
                })
            return tools

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Return all tools with full schemas (for non-progressive clients).

        Returns:
            List of tool descriptors in the same format as
            :meth:`get_tools_for_category` but across all categories.
        """
        with self._lock:
            tools: list[dict[str, Any]] = []
            for tool in self._tools.values():
                tools.append({
                    "name": tool.tool_id,
                    "description": tool.short_description,
                    "inputSchema": tool.full_schema,
                })
            return tools

    def get_handler(
        self,
        tool_id: str,
    ) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
        """Return the handler for a tool by ID, or None if not found.

        Args:
            tool_id: Tool name / ID.

        Returns:
            Handler callable, or None.
        """
        with self._lock:
            tool = self._tools.get(tool_id)
            return tool.handler if tool is not None else None

    def tool_count(self) -> int:
        """Return total number of registered tools."""
        with self._lock:
            return len(self._tools)

    def category_count(self) -> int:
        """Return total number of categories."""
        with self._lock:
            return len(self._categories)

    def unregister(self, tool_id: str) -> bool:
        """Remove a tool from the registry.

        Args:
            tool_id: Tool ID to remove.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        with self._lock:
            tool = self._tools.pop(tool_id, None)
            if tool is None:
                return False
            cat = self._categories.get(tool.category)
            if cat is not None:
                cat.tool_count = max(0, cat.tool_count - 1)
                if tool_id in cat.tool_ids:
                    cat.tool_ids.remove(tool_id)
            return True

    @staticmethod
    def _default_cat_desc(category: str) -> str:
        """Return a sensible one-line description for a known category name."""
        descriptions: dict[str, str] = {
            "indexing": "Repository indexing, graph building, incremental updates",
            "analysis": "Impact analysis, dependency search, architecture queries",
            "review": "Diff review, context building, relevance ranking",
            "graph": "Graph queries, path finding, neighborhood exploration",
            "analytics": "Token savings, usage stats, cost reports",
            "admin": "Cache management, cleanup, configuration",
        }
        return descriptions.get(category, f"{category} tools")
