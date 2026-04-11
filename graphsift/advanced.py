"""Advanced capabilities for graphsift.

8 mandate categories:
1. Smart Cache       — LRU+TTL graph cache, .memoize(), .stats(), thread-safe
2. Pipeline          — staged analysis pipeline, .arun(), audit log, .with_retry()
3. Validator         — declarative diff/context validation DSL
4. Async Batch       — async_batch_index() + sync batch_index(), bounded semaphore
5. Rate Limiter      — token-bucket for LLM API calls, per-key, .stats()
6. Streaming         — async generator yielding ScoredFile chunks
7. Diff Engine       — structural diff of two ContextResults, .summary(), .to_json()
8. Circuit Breaker   — protects LLM API calls from cascading failures
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Generic, TypeVar

from .core import ContextBuilder, ContextResult, DiffSpec, FileNode
from .exceptions import (
    graphsiftError,
    ConfigurationError,
    GraphError,
    ValidationError,
)
from .models import ContextConfig, IndexStats, ScoredFile

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ===========================================================================
# 1. Smart Cache — LRU + TTL, .memoize(), .stats(), thread-safe
# ===========================================================================


@dataclass
class _CacheEntry(Generic[T]):
    value: T
    expires_at: float
    hits: int = 0


class GraphCache(Generic[T]):
    """Thread-safe LRU+TTL cache for graphsift graph results.

    Caches ContextResult objects so repeated queries on the same diff
    don't re-traverse the graph.

    Args:
        maxsize: Max entries before LRU eviction.
        ttl: TTL in seconds. None = no expiry.

    Example::

        cache: GraphCache[ContextResult] = GraphCache(maxsize=64, ttl=300)

        @cache.memoize
        def build_context(diff_key: str) -> ContextResult:
            ...
    """

    def __init__(self, maxsize: int = 128, ttl: float | None = 600.0) -> None:
        if maxsize < 1:
            raise ConfigurationError("maxsize must be >= 1.")
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, _CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __repr__(self) -> str:
        return f"GraphCache(maxsize={self._maxsize}, ttl={self._ttl}, size={len(self._store)})"

    def _expired(self, entry: _CacheEntry[T]) -> bool:
        return self._ttl is not None and time.monotonic() > entry.expires_at

    def get(self, key: str) -> T | None:
        """Retrieve cached value or None."""
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            entry = self._store[key]
            if self._expired(entry):
                del self._store[key]
                self._misses += 1
                self._evictions += 1
                return None
            self._store.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T) -> None:
        """Store value in cache."""
        exp = (time.monotonic() + self._ttl) if self._ttl else float("inf")
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = _CacheEntry(value=value, expires_at=exp)
                return
            if len(self._store) >= self._maxsize:
                self._store.popitem(last=False)
                self._evictions += 1
            self._store[key] = _CacheEntry(value=value, expires_at=exp)

    def invalidate(self, key: str) -> bool:
        """Remove a key. Returns True if it existed."""
        with self._lock:
            return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """Evict all entries."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "size": len(self._store),
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }

    def memoize(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator: cache return value keyed by hashed args."""

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            raw = json.dumps(
                {"a": [repr(a) for a in args], "k": kwargs}, sort_keys=True
            )
            key = hashlib.sha256(raw.encode()).hexdigest()
            cached = self.get(key)
            if cached is not None:
                return cached
            result = fn(*args, **kwargs)
            self.set(key, result)
            return result

        return wrapper


# ===========================================================================
# 2. Pipeline — staged analysis, .arun(), audit log, .with_retry()
# ===========================================================================


@dataclass
class _StepAudit:
    step_name: str
    input_files: int
    output_files: int
    duration_ms: float
    error: str | None = None


class AnalysisPipeline:
    """Staged graphsift analysis pipeline with audit log and retry.

    Each step is a ``Callable[[ContextResult], ContextResult]`` transformation.

    Example::

        pipeline = (
            AnalysisPipeline(builder)
            .add_step("filter_tests", lambda r: filter_tests(r))
            .add_step("rerank", lambda r: rerank_by_complexity(r))
            .with_retry(n=2, backoff=0.3)
        )
        result, audit = pipeline.run(diff_spec, source_map)
    """

    def __init__(self, builder: ContextBuilder) -> None:
        self._builder = builder
        self._steps: list[tuple[str, Callable[[ContextResult], ContextResult]]] = []
        self._retries = 0
        self._backoff = 0.5
        self._audit: list[_StepAudit] = []

    def add_step(
        self, name: str, fn: Callable[[ContextResult], ContextResult]
    ) -> "AnalysisPipeline":
        """Add a named analysis step."""
        self._steps.append((name, fn))
        return self

    def with_retry(self, n: int = 2, backoff: float = 0.5) -> "AnalysisPipeline":
        """Configure per-step retry with exponential backoff."""
        if n < 0:
            raise ConfigurationError("Retry count must be >= 0.")
        self._retries = n
        self._backoff = backoff
        return self

    def audit_log(self) -> list[dict[str, Any]]:
        """Return per-step audit records."""
        return [
            {
                "step": a.step_name,
                "input_files": a.input_files,
                "output_files": a.output_files,
                "duration_ms": round(a.duration_ms, 2),
                "error": a.error,
            }
            for a in self._audit
        ]

    def run(
        self, diff_spec: DiffSpec, source_map: dict[str, str]
    ) -> tuple[ContextResult, list[dict[str, Any]]]:
        """Execute pipeline synchronously."""
        self._audit = []
        result = self._builder.build(diff_spec, source_map)
        for name, fn in self._steps:
            result = self._run_step(name, fn, result)
        return result, self.audit_log()

    async def arun(
        self, diff_spec: DiffSpec, source_map: dict[str, str]
    ) -> tuple[ContextResult, list[dict[str, Any]]]:
        """Execute pipeline asynchronously."""
        self._audit = []
        result = await asyncio.to_thread(self._builder.build, diff_spec, source_map)
        for name, fn in self._steps:
            result = await asyncio.to_thread(self._run_step, name, fn, result)
        return result, self.audit_log()

    def _run_step(
        self,
        name: str,
        fn: Callable[[ContextResult], ContextResult],
        ctx: ContextResult,
    ) -> ContextResult:
        attempt = 0
        delay = self._backoff
        last_exc: Exception | None = None
        t0 = time.monotonic()

        while attempt <= self._retries:
            try:
                out = fn(ctx)
                dur = (time.monotonic() - t0) * 1000
                self._audit.append(
                    _StepAudit(
                        step_name=name,
                        input_files=ctx.files_selected,
                        output_files=out.files_selected,
                        duration_ms=dur,
                    )
                )
                return out
            except Exception as exc:
                last_exc = exc
                attempt += 1
                if attempt <= self._retries:
                    time.sleep(delay)
                    delay *= 2

        dur = (time.monotonic() - t0) * 1000
        self._audit.append(
            _StepAudit(
                step_name=name,
                input_files=ctx.files_selected,
                output_files=0,
                duration_ms=dur,
                error=str(last_exc),
            )
        )
        raise GraphError(f"Step '{name}' failed after {self._retries + 1} attempts.") from last_exc


# ===========================================================================
# 3. Validator — declarative DSL, field-level errors, sync+async
# ===========================================================================


@dataclass
class _Rule:
    name: str
    check: Callable[[DiffSpec], bool]
    message: str
    field: str = "diff"


class DiffValidator:
    """Declarative validator for DiffSpec inputs.

    Example::

        validator = (
            DiffValidator()
            .require_changed_files()
            .require_max_files(50)
            .add_rule("no_binary", lambda d: not any(p.endswith(".bin") for p in d.changed_files), "Binary files not supported")
        )
        errors = validator.validate(diff_spec)
    """

    def __init__(self) -> None:
        self._rules: list[_Rule] = []

    def add_rule(
        self,
        name: str,
        check: Callable[[DiffSpec], bool],
        message: str,
        field: str = "diff",
    ) -> "DiffValidator":
        """Add a custom rule."""
        self._rules.append(_Rule(name=name, check=check, message=message, field=field))
        return self

    def require_changed_files(self) -> "DiffValidator":
        """Require at least one changed file."""
        return self.add_rule(
            "has_changed_files",
            lambda d: len(d.changed_files) > 0,
            "DiffSpec must have at least one changed file.",
        )

    def require_max_files(self, n: int) -> "DiffValidator":
        """Require no more than n changed files."""
        return self.add_rule(
            "max_files",
            lambda d: len(d.changed_files) <= n,
            f"DiffSpec must not have more than {n} changed files.",
        )

    def require_extensions(self, allowed: set[str]) -> "DiffValidator":
        """Require all changed files to have allowed extensions."""
        return self.add_rule(
            "allowed_extensions",
            lambda d: all(
                any(p.endswith(ext) for ext in allowed)
                for p in d.changed_files
            ),
            f"All changed files must have extensions: {allowed}",
        )

    def require_no_secrets_in_query(self) -> "DiffValidator":
        """Reject queries containing API key patterns."""
        import re  # noqa: PLC0415
        pat = re.compile(r"\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})\b")
        return self.add_rule(
            "no_secrets",
            lambda d: not pat.search(d.query),
            "Query appears to contain API keys.",
        )

    def validate(self, diff: DiffSpec) -> dict[str, list[str]]:
        """Run all rules. Returns field-level errors dict (empty = valid)."""
        errors: dict[str, list[str]] = {}
        for rule in self._rules:
            try:
                if not rule.check(diff):
                    errors.setdefault(rule.field, []).append(rule.message)
            except Exception as exc:
                errors.setdefault(rule.field, []).append(f"Rule '{rule.name}' raised: {exc}")
        return errors

    async def avalidate(self, diff: DiffSpec) -> dict[str, list[str]]:
        """Async version of validate."""
        return await asyncio.to_thread(self.validate, diff)

    def validate_or_raise(self, diff: DiffSpec) -> None:
        """Validate and raise ValidationError if any rule fails."""
        errors = self.validate(diff)
        if errors:
            raise ValidationError(f"DiffSpec validation failed: {errors}")


# ===========================================================================
# 4. Async Batch — async_batch_index() + batch_index(), bounded semaphore
# ===========================================================================


async def async_batch_index(
    builder: ContextBuilder,
    source_maps: list[dict[str, str]],
    *,
    concurrency: int = 4,
) -> list[IndexStats | Exception]:
    """Index multiple source maps concurrently.

    Each source_map is a separate batch (e.g. different repos or modules).
    Per-item errors are isolated.

    Args:
        builder: ContextBuilder to index into.
        source_maps: List of source maps to index.
        concurrency: Max concurrent indexing tasks.

    Returns:
        List of IndexStats or Exception per source_map.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _index_one(sm: dict[str, str]) -> IndexStats | Exception:
        async with sem:
            try:
                return await asyncio.to_thread(builder.index_files, sm)
            except Exception as exc:
                logger.warning(
                    "graphsift: batch index item failed",
                    extra={"error": str(exc), "files": len(sm)},
                )
                return exc

    return list(await asyncio.gather(*[_index_one(sm) for sm in source_maps]))


def batch_index(
    builder: ContextBuilder,
    source_maps: list[dict[str, str]],
    *,
    concurrency: int = 4,
) -> list[IndexStats | Exception]:
    """Synchronous batch indexing.

    Args:
        builder: ContextBuilder.
        source_maps: List of source maps.
        concurrency: Max concurrent tasks.

    Returns:
        List of IndexStats or Exception per source_map.
    """
    return asyncio.run(async_batch_index(builder, source_maps, concurrency=concurrency))


async def async_batch_build(
    builder: ContextBuilder,
    diff_specs: list[DiffSpec],
    source_map: dict[str, str],
    *,
    concurrency: int = 4,
) -> list[ContextResult | Exception]:
    """Build context for multiple diffs concurrently.

    Args:
        builder: Pre-indexed ContextBuilder.
        diff_specs: List of DiffSpec objects.
        source_map: Shared source map.
        concurrency: Max concurrent tasks.

    Returns:
        List of ContextResult or Exception per diff.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _build_one(diff: DiffSpec) -> ContextResult | Exception:
        async with sem:
            try:
                return await asyncio.to_thread(builder.build, diff, source_map)
            except Exception as exc:
                logger.warning(
                    "graphsift: batch build item failed",
                    extra={"error": str(exc)},
                )
                return exc

    return list(await asyncio.gather(*[_build_one(d) for d in diff_specs]))


# ===========================================================================
# 5. Rate Limiter — token bucket, per-key, sync+async context manager
# ===========================================================================


class RateLimiter:
    """Token-bucket rate limiter for LLM API calls.

    Args:
        rate: Tokens replenished per second.
        capacity: Bucket max capacity.
        key: Optional per-key label.

    Example::

        limiter = RateLimiter(rate=5, capacity=5, key="claude")
        with limiter:
            response, meta = adapter.review(...)

        async with limiter:
            response, meta = await async_review(...)
    """

    def __init__(self, rate: float = 5.0, capacity: float = 5.0, key: str = "default") -> None:
        if rate <= 0 or capacity <= 0:
            raise ConfigurationError("rate and capacity must be > 0.")
        self._rate = rate
        self._capacity = capacity
        self._key = key
        self._tokens = capacity
        self._last = time.monotonic()
        self._lock = threading.RLock()
        self._acquired = 0
        self._waited_ms = 0.0

    def __repr__(self) -> str:
        return f"RateLimiter(key={self._key!r}, rate={self._rate}/s)"

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
        self._last = now

    def acquire(self, n: float = 1.0) -> None:
        """Block until n tokens available."""
        t0 = time.monotonic()
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    self._acquired += 1
                    self._waited_ms += (time.monotonic() - t0) * 1000
                    return
            time.sleep(1.0 / self._rate)

    async def aacquire(self, n: float = 1.0) -> None:
        """Async acquire."""
        t0 = time.monotonic()
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    self._acquired += 1
                    self._waited_ms += (time.monotonic() - t0) * 1000
                    return
            await asyncio.sleep(1.0 / self._rate)

    def stats(self) -> dict[str, Any]:
        """Return rate limiter stats."""
        with self._lock:
            self._refill()
            return {
                "key": self._key,
                "total_acquired": self._acquired,
                "avg_wait_ms": round(self._waited_ms / max(self._acquired, 1), 2),
                "available_tokens": round(self._tokens, 2),
            }

    def __enter__(self) -> "RateLimiter":
        self.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    async def __aenter__(self) -> "RateLimiter":
        await self.aacquire()
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass


_key_limiters: dict[str, RateLimiter] = {}
_key_lock = threading.RLock()


def get_rate_limiter(key: str, rate: float = 5.0, capacity: float = 5.0) -> RateLimiter:
    """Get or create a per-key RateLimiter singleton."""
    with _key_lock:
        if key not in _key_limiters:
            _key_limiters[key] = RateLimiter(rate=rate, capacity=capacity, key=key)
        return _key_limiters[key]


# ===========================================================================
# 6. Streaming — async generator yielding ScoredFile batches
# ===========================================================================


async def async_stream_context(
    builder: ContextBuilder,
    diff_spec: DiffSpec,
    source_map: dict[str, str],
    *,
    batch_size: int = 3,
) -> AsyncGenerator[list[ScoredFile], None]:
    """Stream context results in batches of ranked files, highest-score first.

    Allows callers to start processing the most relevant files before
    all files are analysed.

    Args:
        builder: Pre-indexed ContextBuilder.
        diff_spec: Diff specification.
        source_map: Source map.
        batch_size: Number of ScoredFiles per yielded batch.

    Yields:
        Batches of ScoredFile (most relevant first).

    Raises:
        GraphError: If graph traversal fails.
    """
    result = await asyncio.to_thread(builder.build, diff_spec, source_map)
    ranked = result.selected_files
    i = 0
    while i < len(ranked):
        try:
            yield ranked[i : i + batch_size]
        except asyncio.CancelledError:
            raise
        i += batch_size


def stream_context(
    builder: ContextBuilder,
    diff_spec: DiffSpec,
    source_map: dict[str, str],
    *,
    batch_size: int = 3,
) -> Generator[list[ScoredFile], None, None]:
    """Sync streaming of ranked ScoredFile batches.

    Args:
        builder: Pre-indexed ContextBuilder.
        diff_spec: Diff specification.
        source_map: Source map.
        batch_size: Files per batch.

    Yields:
        Batches of ScoredFile.
    """
    result = builder.build(diff_spec, source_map)
    ranked = result.selected_files
    for i in range(0, len(ranked), batch_size):
        yield ranked[i : i + batch_size]


# ===========================================================================
# 7. Diff Engine — compare two ContextResults, .summary(), .to_json()
# ===========================================================================


class ContextDiff:
    """Structural diff between two ContextResult objects.

    Use this to compare context selection before and after a code change,
    or between two different ContextConfig settings.

    Args:
        before: ContextResult from the first run.
        after: ContextResult from the second run.

    Example::

        diff = ContextDiff(result_v1, result_v2)
        print(diff.summary())
        data = diff.to_json()
    """

    def __init__(self, before: ContextResult, after: ContextResult) -> None:
        self._before = before
        self._after = after

    def __repr__(self) -> str:
        return f"ContextDiff(before={self._before.files_selected}, after={self._after.files_selected})"

    @property
    def files_added(self) -> list[str]:
        """Files selected in after but not in before."""
        b = {sf.file_node.path for sf in self._before.selected_files}
        a = {sf.file_node.path for sf in self._after.selected_files}
        return sorted(a - b)

    @property
    def files_removed(self) -> list[str]:
        """Files selected in before but not in after."""
        b = {sf.file_node.path for sf in self._before.selected_files}
        a = {sf.file_node.path for sf in self._after.selected_files}
        return sorted(b - a)

    @property
    def token_delta(self) -> int:
        """Token change: positive = after uses more tokens."""
        return self._after.total_rendered_tokens - self._before.total_rendered_tokens

    @property
    def reduction_delta(self) -> float:
        """Reduction ratio improvement: positive = after is more compressed."""
        return self._after.reduction_ratio - self._before.reduction_ratio

    def score_changes(self) -> list[dict[str, Any]]:
        """Files whose score changed between before and after."""
        b_scores = {sf.file_node.path: sf.score for sf in self._before.selected_files}
        a_scores = {sf.file_node.path: sf.score for sf in self._after.selected_files}
        changes = []
        for path in b_scores.keys() & a_scores.keys():
            delta = a_scores[path] - b_scores[path]
            if abs(delta) > 0.01:
                changes.append({"path": path, "before": b_scores[path], "after": a_scores[path], "delta": round(delta, 4)})
        return sorted(changes, key=lambda x: abs(x["delta"]), reverse=True)

    def summary(self) -> str:
        """Human-readable diff summary."""
        lines = [
            "Context Diff Summary",
            f"  Files: {self._before.files_selected} → {self._after.files_selected} "
            f"({'↑' if self._after.files_selected > self._before.files_selected else '↓'}"
            f"{abs(self._after.files_selected - self._before.files_selected)})",
            f"  Tokens: {self._before.total_rendered_tokens:,} → {self._after.total_rendered_tokens:,} "
            f"(delta {self.token_delta:+,})",
            f"  Reduction: {self._before.reduction_ratio:.1%} → {self._after.reduction_ratio:.1%} "
            f"(delta {self.reduction_delta:+.1%})",
        ]
        if self.files_added:
            lines.append(f"  Added: {', '.join(self.files_added[:5])}")
        if self.files_removed:
            lines.append(f"  Removed: {', '.join(self.files_removed[:5])}")
        score_ch = self.score_changes()
        if score_ch:
            lines.append(f"  Score changes: {len(score_ch)} files")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Machine-readable diff data."""
        return {
            "files_before": self._before.files_selected,
            "files_after": self._after.files_selected,
            "tokens_before": self._before.total_rendered_tokens,
            "tokens_after": self._after.total_rendered_tokens,
            "token_delta": self.token_delta,
            "reduction_before": self._before.reduction_ratio,
            "reduction_after": self._after.reduction_ratio,
            "reduction_delta": round(self.reduction_delta, 4),
            "files_added": self.files_added,
            "files_removed": self.files_removed,
            "score_changes": self.score_changes()[:10],
        }


# ===========================================================================
# 8. Circuit Breaker — closed/open/half-open, auto-reset, .protect()
# ===========================================================================


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker protecting LLM API calls from cascading failures.

    Opens after failure_threshold consecutive failures, auto-resets
    after reset_timeout seconds via HALF_OPEN probe.

    Args:
        failure_threshold: Failures before opening.
        reset_timeout: Seconds before OPEN→HALF_OPEN transition.
        expected_exception: Exception type(s) counted as failures.

    Example::

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

        @cb.protect
        def call_claude(prompt: str) -> str:
            ...
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    ) -> None:
        if failure_threshold < 1:
            raise ConfigurationError("failure_threshold must be >= 1.")
        self._threshold = failure_threshold
        self._timeout = reset_timeout
        self._expected = expected_exception
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._last_opened: float | None = None
        self._lock = threading.RLock()
        self._total = 0
        self._rejected = 0

    def __repr__(self) -> str:
        return f"CircuitBreaker(state={self._state.value}, {self._failures}/{self._threshold})"

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if (
            self._state == CircuitState.OPEN
            and self._last_opened is not None
            and time.monotonic() - self._last_opened >= self._timeout
        ):
            self._state = CircuitState.HALF_OPEN

    def _on_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == CircuitState.HALF_OPEN or self._failures >= self._threshold:
                self._state = CircuitState.OPEN
                self._last_opened = time.monotonic()

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute fn through the circuit breaker."""
        with self._lock:
            self._maybe_half_open()
            if self._state == CircuitState.OPEN:
                self._rejected += 1
                raise graphsiftError(
                    f"Circuit OPEN (failures={self._failures}). Retry in {self._timeout}s."
                )
            self._total += 1
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except self._expected:
            self._on_failure()
            raise

    def protect(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator: wrap function with circuit breaker."""

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(fn, *args, **kwargs)

        return wrapper

    def stats(self) -> dict[str, Any]:
        """Return circuit breaker stats."""
        with self._lock:
            self._maybe_half_open()
            return {
                "state": self._state.value,
                "failures": self._failures,
                "threshold": self._threshold,
                "total_calls": self._total,
                "rejected_calls": self._rejected,
            }

    def reset(self) -> None:
        """Manually reset to CLOSED."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._last_opened = None


# ===========================================================================
# 9. Retry Strategy — exponential backoff with jitter, per-exception routing
# ===========================================================================


@dataclass
class _RetryAttempt:
    attempt: int
    exception: Exception
    delay_ms: float
    exc_type: str


class RetryStrategy:
    """Exponential backoff with jitter and per-exception routing.

    Supports deadline (max wall-clock seconds), per-exception class routing
    (retry only specific errors, fail fast on others), and per-attempt audit log.

    Args:
        max_attempts: Maximum total attempts (including first).
        base_delay: Initial retry delay in seconds.
        max_delay: Cap on computed delay (before jitter).
        jitter: If True, multiply computed delay by random(0.5, 1.5).
        deadline: Optional max wall-clock seconds from first call.
        retry_on: Exception types to retry; None = retry on all exceptions.

    Example::

        strategy = RetryStrategy(max_attempts=4, base_delay=0.5, retry_on=(TimeoutError,))

        @strategy.retry
        def call_api(prompt: str) -> str:
            ...

        result = strategy.call(call_api, prompt)
        print(strategy.audit_log())
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        jitter: bool = True,
        deadline: float | None = None,
        retry_on: tuple[type[Exception], ...] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ConfigurationError("max_attempts must be >= 1.")
        if base_delay <= 0:
            raise ConfigurationError("base_delay must be > 0.")
        self._max_attempts = max_attempts
        self._base = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._deadline = deadline
        self._retry_on = retry_on
        self._log: list[_RetryAttempt] = []
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return (
            f"RetryStrategy(max={self._max_attempts}, base={self._base}s, "
            f"jitter={self._jitter})"
        )

    def _compute_delay(self, attempt: int) -> float:
        import random  # noqa: PLC0415

        delay = min(self._base * (2 ** attempt), self._max_delay)
        if self._jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay

    def _should_retry(self, exc: Exception) -> bool:
        if self._retry_on is None:
            return True
        return isinstance(exc, self._retry_on)

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute fn with retry strategy.

        Args:
            fn: Callable to invoke.
            *args: Positional arguments for fn.
            **kwargs: Keyword arguments for fn.

        Returns:
            Return value of fn on success.

        Raises:
            The last exception raised after all retries exhausted.
            graphsiftError: If deadline exceeded.
        """
        import random  # noqa: PLC0415
        import time  # noqa: PLC0415

        with self._lock:
            self._log = []

        t_start = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(self._max_attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._should_retry(exc):
                    raise

                delay = self._compute_delay(attempt)
                with self._lock:
                    self._log.append(_RetryAttempt(
                        attempt=attempt + 1,
                        exception=exc,
                        delay_ms=round(delay * 1000, 1),
                        exc_type=type(exc).__name__,
                    ))

                if attempt + 1 >= self._max_attempts:
                    break

                if self._deadline is not None:
                    elapsed = time.monotonic() - t_start
                    if elapsed + delay > self._deadline:
                        raise graphsiftError(
                            f"RetryStrategy: deadline {self._deadline}s exceeded after {elapsed:.1f}s."
                        ) from exc

                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    async def acall(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Async version of call — wraps blocking fn via asyncio.to_thread.

        Args:
            fn: Callable to invoke (sync or async).
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Return value of fn on success.
        """
        import asyncio as _asyncio  # noqa: PLC0415
        import time  # noqa: PLC0415

        with self._lock:
            self._log = []

        t_start = time.monotonic()
        last_exc: Exception | None = None

        for attempt in range(self._max_attempts):
            try:
                if _asyncio.iscoroutinefunction(fn):
                    return await fn(*args, **kwargs)
                return await _asyncio.to_thread(fn, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._should_retry(exc):
                    raise

                delay = self._compute_delay(attempt)
                with self._lock:
                    self._log.append(_RetryAttempt(
                        attempt=attempt + 1,
                        exception=exc,
                        delay_ms=round(delay * 1000, 1),
                        exc_type=type(exc).__name__,
                    ))

                if attempt + 1 >= self._max_attempts:
                    break

                if self._deadline is not None:
                    elapsed = time.monotonic() - t_start
                    if elapsed + delay > self._deadline:
                        raise graphsiftError(
                            f"RetryStrategy: deadline exceeded."
                        ) from exc

                await _asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]

    def retry(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Decorator: wrap function with this retry strategy."""

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.call(fn, *args, **kwargs)

        return wrapper

    def audit_log(self) -> list[dict[str, Any]]:
        """Return per-attempt audit records (attempt, exc type, delay ms)."""
        with self._lock:
            return [
                {
                    "attempt": r.attempt,
                    "exc_type": r.exc_type,
                    "error": str(r.exception),
                    "delay_ms": r.delay_ms,
                }
                for r in self._log
            ]


# ===========================================================================
# 10. Schema Evolution — versioned model migration, compatibility checks
# ===========================================================================


@dataclass
class _Migration:
    from_version: int
    to_version: int
    transform: Callable[[dict[str, Any]], dict[str, Any]]
    description: str


class SchemaEvolution:
    """Versioned schema migration registry for graphsift data payloads.

    Enables safe rolling upgrades: serialised ContextResult / DiffSpec dicts
    from older versions are migrated forward to the current schema before
    being deserialised. Provides compatibility checks and a migration audit log.

    Args:
        current_version: The schema version this instance targets.

    Example::

        evo = SchemaEvolution(current_version=3)

        @evo.migration(from_version=1, to_version=2, description="add diff_text field")
        def v1_to_v2(data: dict) -> dict:
            data.setdefault("diff_text", "")
            return data

        migrated, audit = evo.migrate(old_payload, from_version=1)
        evo.check_compatibility(other_payload)  # raises if incompatible
    """

    def __init__(self, current_version: int = 1) -> None:
        if current_version < 1:
            raise ConfigurationError("current_version must be >= 1.")
        self._current = current_version
        self._migrations: list[_Migration] = []
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        return f"SchemaEvolution(current_version={self._current}, migrations={len(self._migrations)})"

    def migration(
        self,
        from_version: int,
        to_version: int,
        description: str = "",
    ) -> Callable[[Callable[[dict[str, Any]], dict[str, Any]]], Callable[[dict[str, Any]], dict[str, Any]]]:
        """Decorator: register a migration function.

        Args:
            from_version: Source schema version.
            to_version: Target schema version.
            description: Human-readable description of what changed.

        Returns:
            Decorator that registers fn as a migration.
        """

        def decorator(
            fn: Callable[[dict[str, Any]], dict[str, Any]],
        ) -> Callable[[dict[str, Any]], dict[str, Any]]:
            with self._lock:
                self._migrations.append(_Migration(
                    from_version=from_version,
                    to_version=to_version,
                    transform=fn,
                    description=description or fn.__name__,
                ))
                self._migrations.sort(key=lambda m: (m.from_version, m.to_version))
            return fn

        return decorator

    def register(
        self,
        from_version: int,
        to_version: int,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        description: str = "",
    ) -> None:
        """Register a migration without using the decorator syntax.

        Args:
            from_version: Source schema version.
            to_version: Target schema version.
            fn: Transform function ``dict → dict``.
            description: Human-readable description.
        """
        with self._lock:
            self._migrations.append(_Migration(
                from_version=from_version,
                to_version=to_version,
                transform=fn,
                description=description or fn.__name__,
            ))
            self._migrations.sort(key=lambda m: (m.from_version, m.to_version))

    def migrate(
        self,
        data: dict[str, Any],
        from_version: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Migrate data from from_version to current_version.

        Applies migrations in order, chaining each output into the next.

        Args:
            data: Raw serialised payload dict.
            from_version: Schema version of the input data.

        Returns:
            Tuple of (migrated_data, audit_log).

        Raises:
            ConfigurationError: If no migration path exists.
            ValidationError: If migration raises.
        """
        audit: list[dict[str, Any]] = []
        current = dict(data)  # copy
        version = from_version

        with self._lock:
            migrations = list(self._migrations)

        while version < self._current:
            step = next(
                (m for m in migrations if m.from_version == version),
                None,
            )
            if step is None:
                raise ConfigurationError(
                    f"No migration from version {version} to {version + 1}. "
                    f"Cannot reach current version {self._current}."
                )
            try:
                current = step.transform(current)
                audit.append({
                    "from": step.from_version,
                    "to": step.to_version,
                    "description": step.description,
                    "status": "ok",
                })
                version = step.to_version
            except Exception as exc:
                audit.append({
                    "from": step.from_version,
                    "to": step.to_version,
                    "description": step.description,
                    "status": "error",
                    "error": str(exc),
                })
                raise ValidationError(
                    f"Migration v{step.from_version}→v{step.to_version} failed: {exc}"
                ) from exc

        current["__schema_version__"] = self._current
        return current, audit

    def check_compatibility(self, data: dict[str, Any]) -> bool:
        """Return True if data is at current_version, False if it needs migration.

        Args:
            data: Payload dict, optionally containing ``__schema_version__`` key.

        Returns:
            True if schema_version matches current_version.
        """
        return data.get("__schema_version__", 1) == self._current

    def migration_path(self, from_version: int) -> list[str]:
        """Return human-readable list of migration steps from from_version.

        Args:
            from_version: Starting schema version.

        Returns:
            List of step descriptions.
        """
        path: list[str] = []
        version = from_version
        with self._lock:
            migrations = list(self._migrations)
        while version < self._current:
            step = next((m for m in migrations if m.from_version == version), None)
            if step is None:
                break
            path.append(f"v{step.from_version}→v{step.to_version}: {step.description}")
            version = step.to_version
        return path
