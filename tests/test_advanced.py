"""Tests for graphsift advanced capabilities."""

import asyncio
import time

import pytest

from graphsift import (
    AnalysisPipeline,
    CircuitBreaker,
    CircuitState,
    graphsiftError,
    ContextBuilder,
    ContextConfig,
    ContextDiff,
    DiffSpec,
    DiffValidator,
    GraphCache,
    RateLimiter,
    ValidationError,
    async_batch_build,
    async_stream_context,
    batch_index,
    get_rate_limiter,
    stream_context,
)


# ---------------------------------------------------------------------------
# 1. GraphCache
# ---------------------------------------------------------------------------


def test_graph_cache_set_get():
    cache: GraphCache[int] = GraphCache(maxsize=10)
    cache.set("k", 42)
    assert cache.get("k") == 42


def test_graph_cache_miss_returns_none():
    cache: GraphCache[int] = GraphCache()
    assert cache.get("missing") is None


def test_graph_cache_lru_eviction():
    cache: GraphCache[int] = GraphCache(maxsize=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.set("d", 4)
    assert cache.get("a") is None
    assert cache.get("d") == 4


def test_graph_cache_ttl_expiry():
    cache: GraphCache[int] = GraphCache(maxsize=10, ttl=0.05)
    cache.set("k", 99)
    time.sleep(0.1)
    assert cache.get("k") is None


def test_graph_cache_stats():
    cache: GraphCache[str] = GraphCache(maxsize=10)
    cache.set("x", "val")
    cache.get("x")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_graph_cache_memoize(builder, source_map, diff_spec):
    cache: GraphCache = GraphCache(maxsize=32)
    calls = {"n": 0}

    @cache.memoize
    def expensive(key: str) -> int:
        calls["n"] += 1
        return len(key)

    expensive("hello")
    expensive("hello")
    assert calls["n"] == 1


def test_graph_cache_invalidate():
    cache: GraphCache[int] = GraphCache()
    cache.set("k", 1)
    assert cache.invalidate("k")
    assert cache.get("k") is None


def test_graph_cache_repr():
    cache: GraphCache[int] = GraphCache(maxsize=50)
    assert "GraphCache" in repr(cache)


# ---------------------------------------------------------------------------
# 2. AnalysisPipeline
# ---------------------------------------------------------------------------


def test_pipeline_basic_run(builder, source_map, diff_spec):
    pipeline = AnalysisPipeline(builder).add_step("identity", lambda r: r)
    result, audit = pipeline.run(diff_spec, source_map)
    assert result.files_selected >= 1
    assert len(audit) == 1
    assert audit[0]["step"] == "identity"


def test_pipeline_multi_step(builder, source_map, diff_spec):
    pipeline = (
        AnalysisPipeline(builder)
        .add_step("step1", lambda r: r)
        .add_step("step2", lambda r: r)
    )
    result, audit = pipeline.run(diff_spec, source_map)
    assert len(audit) == 2


def test_pipeline_arun(builder, source_map, diff_spec):
    pipeline = AnalysisPipeline(builder).add_step("identity", lambda r: r)

    async def run():
        return await pipeline.arun(diff_spec, source_map)

    result, audit = asyncio.run(run())
    assert result.files_selected >= 1


def test_pipeline_retry_on_flaky(builder, source_map, diff_spec):
    attempts = {"n": 0}

    def flaky(r):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return r

    pipeline = AnalysisPipeline(builder).add_step("flaky", flaky).with_retry(n=2, backoff=0.01)
    result, audit = pipeline.run(diff_spec, source_map)
    assert result.files_selected >= 1


def test_pipeline_audit_captures_error(builder, source_map, diff_spec):
    def fail(r):
        raise RuntimeError("always fails")

    pipeline = AnalysisPipeline(builder).add_step("fail", fail).with_retry(n=1, backoff=0.01)
    from graphsift import GraphError

    with pytest.raises(GraphError):
        pipeline.run(diff_spec, source_map)
    audit = pipeline.audit_log()
    assert audit[-1]["error"] is not None


# ---------------------------------------------------------------------------
# 3. DiffValidator
# ---------------------------------------------------------------------------


def test_validator_passes_valid(diff_spec):
    v = DiffValidator().require_changed_files().require_max_files(10)
    assert v.validate(diff_spec) == {}


def test_validator_requires_changed_files():
    v = DiffValidator().require_changed_files()
    bad = DiffSpec(changed_files=[])
    errors = v.validate(bad)
    assert "diff" in errors


def test_validator_max_files():
    v = DiffValidator().require_max_files(1)
    bad = DiffSpec(changed_files=["a.py", "b.py"])
    errors = v.validate(bad)
    assert "diff" in errors


def test_validator_extensions():
    v = DiffValidator().require_extensions({".py"})
    good = DiffSpec(changed_files=["a.py"])
    bad = DiffSpec(changed_files=["a.js"])
    assert v.validate(good) == {}
    assert "diff" in v.validate(bad)


def test_validator_no_secrets_in_query():
    v = DiffValidator().require_no_secrets_in_query()
    clean = DiffSpec(changed_files=["a.py"], query="review this code")
    dirty = DiffSpec(changed_files=["a.py"], query="use sk-abcdefghijklmnopqrstuvwxyz123456")
    assert v.validate(clean) == {}
    assert "diff" in v.validate(dirty)


def test_validator_raises(diff_spec):
    v = DiffValidator().require_max_files(0)
    with pytest.raises(ValidationError):
        v.validate_or_raise(diff_spec)


def test_validator_avalidate(diff_spec):
    v = DiffValidator().require_changed_files()

    async def run():
        return await v.avalidate(diff_spec)

    result = asyncio.run(run())
    assert result == {}


# ---------------------------------------------------------------------------
# 4. Async Batch
# ---------------------------------------------------------------------------


def test_batch_index_sync(source_map):
    builder = ContextBuilder()
    results = batch_index(builder, [source_map], concurrency=1)
    assert len(results) == 1
    from graphsift import IndexStats
    assert isinstance(results[0], IndexStats)


def test_batch_index_isolates_errors(source_map):
    builder = ContextBuilder()
    # Pass an invalid source map (non-dict) wrapped in list
    results = batch_index(builder, [source_map, {}], concurrency=2)
    assert len(results) == 2


def test_async_batch_build(builder, source_map, diff_spec):
    async def run():
        return await async_batch_build(builder, [diff_spec, diff_spec], source_map)

    results = asyncio.run(run())
    assert len(results) == 2
    from graphsift import ContextResult
    for r in results:
        assert isinstance(r, ContextResult)


# ---------------------------------------------------------------------------
# 5. RateLimiter
# ---------------------------------------------------------------------------


def test_rate_limiter_acquire():
    rl = RateLimiter(rate=100, capacity=10)
    rl.acquire(1)
    assert rl.stats()["total_acquired"] == 1


def test_rate_limiter_context_manager():
    rl = RateLimiter(rate=100, capacity=10)
    with rl:
        pass
    assert rl.stats()["total_acquired"] == 1


def test_rate_limiter_async_context_manager():
    rl = RateLimiter(rate=100, capacity=10)

    async def run():
        async with rl:
            pass

    asyncio.run(run())
    assert rl.stats()["total_acquired"] == 1


def test_rate_limiter_stats_keys():
    rl = RateLimiter(rate=10, capacity=10, key="test")
    rl.acquire()
    stats = rl.stats()
    assert "key" in stats
    assert "total_acquired" in stats
    assert "avg_wait_ms" in stats
    assert "available_tokens" in stats


def test_get_rate_limiter_singleton():
    a = get_rate_limiter("repo-1")
    b = get_rate_limiter("repo-1")
    assert a is b


def test_rate_limiter_repr():
    rl = RateLimiter(rate=5, capacity=5, key="x")
    assert "RateLimiter" in repr(rl)


# ---------------------------------------------------------------------------
# 6. Streaming
# ---------------------------------------------------------------------------


def test_stream_context_yields_batches(builder, source_map, diff_spec):
    batches = list(stream_context(builder, diff_spec, source_map, batch_size=2))
    assert len(batches) >= 1
    for batch in batches:
        assert isinstance(batch, list)
        for sf in batch:
            from graphsift import ScoredFile
            assert isinstance(sf, ScoredFile)


def test_stream_context_all_selected(builder, source_map, diff_spec):
    all_files = []
    for batch in stream_context(builder, diff_spec, source_map, batch_size=1):
        all_files.extend(batch)
    result = builder.build(diff_spec, source_map)
    assert len(all_files) == result.files_selected


def test_async_stream_context(builder, source_map, diff_spec):
    async def run():
        results = []
        async for batch in async_stream_context(builder, diff_spec, source_map, batch_size=2):
            results.extend(batch)
        return results

    files = asyncio.run(run())
    assert len(files) >= 1


# ---------------------------------------------------------------------------
# 7. ContextDiff
# ---------------------------------------------------------------------------


def test_context_diff_summary(builder, source_map, diff_spec):
    r1 = builder.build(diff_spec, source_map)
    r2 = builder.build(diff_spec, source_map)
    diff = ContextDiff(r1, r2)
    summary = diff.summary()
    assert "Context Diff Summary" in summary


def test_context_diff_to_json(builder, source_map, diff_spec):
    r1 = builder.build(diff_spec, source_map)
    r2 = builder.build(diff_spec, source_map)
    diff = ContextDiff(r1, r2)
    data = diff.to_json()
    assert "files_before" in data
    assert "files_after" in data
    assert "token_delta" in data
    assert "files_added" in data
    assert "files_removed" in data


def test_context_diff_files_added(source_map, diff_spec):
    b1 = ContextBuilder(ContextConfig(token_budget=50_000))
    b1.index_files(source_map)
    r1 = b1.build(diff_spec, source_map)

    b2 = ContextBuilder(ContextConfig(token_budget=50_000, min_score=0.0))
    b2.index_files(source_map)
    r2 = b2.build(diff_spec, source_map)

    diff = ContextDiff(r1, r2)
    assert isinstance(diff.files_added, list)
    assert isinstance(diff.files_removed, list)


def test_context_diff_repr(builder, source_map, diff_spec):
    r = builder.build(diff_spec, source_map)
    diff = ContextDiff(r, r)
    assert "ContextDiff" in repr(diff)


# ---------------------------------------------------------------------------
# 8. CircuitBreaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_closed_by_default():
    cb = CircuitBreaker(failure_threshold=3)
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_opens_after_failures():
    cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)

    def fail():
        raise ValueError("boom")

    for _ in range(3):
        try:
            cb.call(fail)
        except ValueError:
            pass
    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_rejects_when_open():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=60)

    def fail():
        raise RuntimeError("x")

    try:
        cb.call(fail)
    except RuntimeError:
        pass

    with pytest.raises(graphsiftError):
        cb.call(lambda: "ok")


def test_circuit_breaker_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)

    def fail():
        raise RuntimeError()

    try:
        cb.call(fail)
    except RuntimeError:
        pass

    time.sleep(0.1)
    assert cb.state == CircuitState.HALF_OPEN


def test_circuit_breaker_closes_on_success():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)

    def fail():
        raise RuntimeError()

    try:
        cb.call(fail)
    except RuntimeError:
        pass

    time.sleep(0.1)
    result = cb.call(lambda: 42)
    assert result == 42
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_protect_decorator():
    cb = CircuitBreaker(failure_threshold=5)

    @cb.protect
    def double(x: int) -> int:
        return x * 2

    assert double(5) == 10


def test_circuit_breaker_stats():
    cb = CircuitBreaker()
    cb.call(lambda: None)
    stats = cb.stats()
    assert "state" in stats
    assert "failures" in stats
    assert "total_calls" in stats
    assert stats["total_calls"] == 1


def test_circuit_breaker_reset():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=60)

    def fail():
        raise RuntimeError()

    try:
        cb.call(fail)
    except RuntimeError:
        pass

    cb.reset()
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_repr():
    cb = CircuitBreaker()
    assert "CircuitBreaker" in repr(cb)


# ---------------------------------------------------------------------------
# 9. RetryStrategy
# ---------------------------------------------------------------------------

from graphsift import RetryStrategy  # noqa: E402


def test_retry_strategy_succeeds_first_try():
    strategy = RetryStrategy(max_attempts=3, base_delay=0.01)
    result = strategy.call(lambda: 42)
    assert result == 42
    assert strategy.audit_log() == []


def test_retry_strategy_retries_and_succeeds():
    attempts = [0]

    def flaky():
        attempts[0] += 1
        if attempts[0] < 3:
            raise ValueError("not yet")
        return "ok"

    strategy = RetryStrategy(max_attempts=4, base_delay=0.01, jitter=False)
    result = strategy.call(flaky)
    assert result == "ok"
    assert attempts[0] == 3
    assert len(strategy.audit_log()) == 2  # 2 failures before success


def test_retry_strategy_exhausts_and_raises():
    strategy = RetryStrategy(max_attempts=2, base_delay=0.01, jitter=False)
    with pytest.raises(RuntimeError):
        strategy.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))


def test_retry_strategy_no_retry_on_unmatched():
    strategy = RetryStrategy(max_attempts=3, base_delay=0.01, retry_on=(TimeoutError,))
    with pytest.raises(ValueError):
        strategy.call(lambda: (_ for _ in ()).throw(ValueError("wrong type")))
    assert strategy.audit_log() == []  # not retried


def test_retry_strategy_decorator():
    calls = [0]

    strategy = RetryStrategy(max_attempts=3, base_delay=0.01, jitter=False)

    @strategy.retry
    def fragile():
        calls[0] += 1
        if calls[0] < 2:
            raise OSError("retry me")
        return "done"

    assert fragile() == "done"
    assert calls[0] == 2


def test_retry_strategy_repr():
    s = RetryStrategy(max_attempts=5, base_delay=1.0)
    assert "RetryStrategy" in repr(s)


@pytest.mark.asyncio
async def test_retry_strategy_acall():
    calls = [0]

    async def async_fn():
        calls[0] += 1
        if calls[0] < 2:
            raise OSError("retry")
        return "async_ok"

    strategy = RetryStrategy(max_attempts=3, base_delay=0.01, jitter=False)
    result = await strategy.acall(async_fn)
    assert result == "async_ok"


# ---------------------------------------------------------------------------
# 10. SchemaEvolution
# ---------------------------------------------------------------------------

from graphsift import SchemaEvolution  # noqa: E402


def test_schema_evolution_migrate_single_step():
    evo = SchemaEvolution(current_version=2)

    @evo.migration(from_version=1, to_version=2, description="add field")
    def v1_to_v2(data):
        data["new_field"] = "default"
        return data

    payload = {"existing": "value"}
    migrated, audit = evo.migrate(payload, from_version=1)
    assert migrated["new_field"] == "default"
    assert migrated["__schema_version__"] == 2
    assert len(audit) == 1
    assert audit[0]["status"] == "ok"


def test_schema_evolution_migrate_chain():
    evo = SchemaEvolution(current_version=3)

    @evo.migration(from_version=1, to_version=2, description="v1→v2")
    def v1_to_v2(data):
        data["v2"] = True
        return data

    @evo.migration(from_version=2, to_version=3, description="v2→v3")
    def v2_to_v3(data):
        data["v3"] = True
        return data

    migrated, audit = evo.migrate({}, from_version=1)
    assert migrated["v2"] is True
    assert migrated["v3"] is True
    assert len(audit) == 2


def test_schema_evolution_check_compatibility():
    evo = SchemaEvolution(current_version=2)
    assert evo.check_compatibility({"__schema_version__": 2}) is True
    assert evo.check_compatibility({"__schema_version__": 1}) is False
    assert evo.check_compatibility({}) is False  # defaults to 1


def test_schema_evolution_missing_path_raises():
    evo = SchemaEvolution(current_version=3)
    # No migrations registered → can't reach v3
    with pytest.raises(Exception):
        evo.migrate({}, from_version=1)


def test_schema_evolution_migration_path():
    evo = SchemaEvolution(current_version=3)
    evo.register(1, 2, lambda d: d, "first step")
    evo.register(2, 3, lambda d: d, "second step")
    path = evo.migration_path(from_version=1)
    assert len(path) == 2
    assert "first step" in path[0]


def test_schema_evolution_repr():
    evo = SchemaEvolution(current_version=2)
    assert "SchemaEvolution" in repr(evo)
