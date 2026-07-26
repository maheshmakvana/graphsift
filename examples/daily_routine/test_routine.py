"""Test script for the Developer Daily Routine application.

Exercises the full daily routine through the FastAPI endpoints and measures:
- Token usage per operation
- Duration per operation
- Total tokens for full routine
- graphsift compression savings

Usage:
    pytest test_routine.py -xvs          # run via pytest
    python test_routine.py                # run standalone
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

# Ensure the daily_routine package is importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from graphsift_adapter import (
    analyze_code,
    build_context,
    compress_output,
    estimate_tokens_precise,
    get_tracker,
    reset_tracker,
    ultra_compress_output,
)

BASE_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _print_sep(title: str = "") -> None:
    line = "=" * 72
    if title:
        print(f"\n  {line}")
        print(f"  {title}")
        print(f"  {line}")
    else:
        print(f"  {line}")


# ===========================================================================
# API Integration Tests (require running server)
# ===========================================================================


def test_seed_tasks():
    """Test that we can seed default tasks."""
    if httpx is None:
        print("  [SKIP] httpx not installed — skipping API test")
        return
    try:
        resp = httpx.post(f"{BASE_URL}/api/tasks", json={
            "name": "__seed_defaults__",
            "description": "",
            "task_type": "custom",
        }, timeout=10)
        assert resp.status_code in (200, 201)
        print("  [PASS] Seeded default tasks")
    except httpx.ConnectError:
        print("  [SKIP] Server not running (start with: uvicorn main:app --port 8000)")


def test_list_tasks():
    """Test listing tasks."""
    if httpx is None:
        return
    try:
        resp = httpx.get(f"{BASE_URL}/api/tasks", timeout=10)
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)
        print(f"  [PASS] Listed {len(tasks)} tasks")
    except httpx.ConnectError:
        pass


def test_run_all_tasks():
    """Test running all tasks and gathering report."""
    if httpx is None:
        return
    try:
        resp = httpx.post(f"{BASE_URL}/api/tasks/run-all", timeout=60)
        assert resp.status_code == 200
        tasks = resp.json()
        done = [t for t in tasks if t["status"] == "done"]
        any_failed = any(t["status"] == "failed" for t in tasks)
        assert not any_failed, f"Some tasks failed: {[t for t in tasks if t['status'] == 'failed']}"
        total_tokens = sum(t.get("tokens_used", 0) for t in tasks)
        print(f"  [PASS] Ran {len(done)} tasks, {total_tokens} tokens used")
    except httpx.ConnectError:
        pass


def test_daily_report():
    """Test the daily report endpoint."""
    if httpx is None:
        return
    try:
        resp = httpx.get(f"{BASE_URL}/api/report", timeout=10)
        assert resp.status_code == 200
        report = resp.json()
        assert "total_tasks" in report
        assert "graphsift_metrics" in report
        print(f"  [PASS] Report: {report['total_tasks']} tasks, "
              f"{report['completion_rate']}% complete, "
              f"{report['total_tokens']} tokens")
    except httpx.ConnectError:
        pass


# ===========================================================================
# Direct graphsift_adapter unit tests (no server needed)
# ===========================================================================


def test_compress_output():
    """Test CLI output compression."""
    reset_tracker()
    raw = (
        "========================================\n"
        "Test Run: 42 passed, 3 failed in 12.3s\n"
        "========================================\n"
        "FAILED tests/test_auth.py::test_login\n"
        "FAILED tests/test_api.py::test_rate_limit\n"
        "FAILED tests/test_db.py::test_migration\n"
    )
    result = compress_output(raw, cmd_type="pytest")
    assert isinstance(result, str)
    assert len(result) > 0
    metrics = get_tracker().calls[-1]
    print(f"  [PASS] compress_output: {len(raw)}c -> {len(result)}c, "
          f"saved {metrics.tokens_saved} tokens, {metrics.duration_ms:.1f}ms")
    assert metrics.tokens_saved >= 0


def test_ultra_compress_output():
    """Test multi-pass ultra compression."""
    reset_tracker()
    raw = (
        "commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0\n"
        "Author: Dev <dev@example.com>\n"
        "Date:   Thu Jul 16 09:00:00 2026 +0000\n\n"
        "    feat(auth): implement OAuth2 refresh token rotation\n"
    ) * 20  # 20 commits
    result = ultra_compress_output(raw, cmd_type="git_log")
    assert isinstance(result, str)
    assert len(result) > 0
    metrics = get_tracker().calls[-1]
    ratio = (1 - len(result) / max(len(raw), 1)) * 100
    print(f"  [PASS] ultra_compress: {len(raw)}c -> {len(result)}c "
          f"({ratio:.0f}% reduction), {metrics.duration_ms:.1f}ms")


def test_estimate_tokens():
    """Test token estimation."""
    reset_tracker()
    text = "Hello, world! This is a test sentence to estimate tokens."
    tokens = estimate_tokens_precise(text)
    assert tokens > 0
    metrics = get_tracker().calls[-1]
    print(f"  [PASS] estimate_tokens: '{text[:30]}...' -> {tokens} tokens, "
          f"{metrics.duration_ms:.1f}ms")


def test_build_context():
    """Test context building from sample files."""
    reset_tracker()
    sample_files = {
        "src/hello.py": (
            "def greet(name: str) -> str:\n"
            '    """Return a greeting for the given name."""\n'
            "    return f'Hello, {name}!'\n"
        ),
        "src/main.py": (
            "from hello import greet\n"
            "def main():\n"
            '    print(greet("World"))\n'
        ),
    }
    context = build_context(sample_files, query="Review this code")
    assert isinstance(context, str)
    assert len(context) > 0
    metrics = get_tracker().calls[-1]
    print(f"  [PASS] build_context: {len(context)} chars output, "
          f"{metrics.duration_ms:.1f}ms, saved {metrics.tokens_saved} tokens")


def test_analyze_code():
    """Test code analysis with FixSuggester."""
    reset_tracker()
    source = (
        "import os\n"
        "import sys\n"
        "from typing import List\n"
        "\n"
        "def process(data):\n"
        "    result = []\n"
        "    for i, item in enumerate(data):\n"
        "        result.append(item.strip())\n"
        "    return result\n"
        "\n"
        "class Manager:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "    def run(self):\n"
        "        pass\n"
    )
    suggestions = analyze_code("src/example.py", source)
    metrics = get_tracker().calls[-1]
    print(f"  [PASS] analyze_code: {len(suggestions)} suggestions, "
          f"{metrics.duration_ms:.1f}ms, {metrics.tokens_saved} tokens saved")
    assert isinstance(suggestions, list)


# ===========================================================================
# Full Routine Benchmark
# ===========================================================================


def test_full_routine_benchmark():
    """Run the complete daily routine and report overall metrics."""
    _print_sep("Full Daily Routine Benchmark")
    reset_tracker()

    operations = {
        "1. Code Review": lambda: build_context(
            {
                "src/auth/login.py": "def login(u: str, p: str) -> bool: ...",
                "src/auth/session.py": "def create_session(uid: int) -> str: ...",
                "src/api/routes.py": "from auth import login\n@router.post('/login')\ndef handle_login(): ...",
            },
            query="Review authentication flow for security issues",
        ),
        "2. CI Failure Triage": lambda: compress_output(
            "===== CI RUN 8472 =====\n"
            "tests/test_auth.py::test_login PASSED\n"
            "tests/test_auth.py::test_session_expiry FAILED\n"
            "tests/test_auth.py::test_concurrent FAILED\n"
            "245 passed, 2 failed in 182.3s\n",
            cmd_type="pytest",
        ),
        "3. Dependency Scan": lambda: analyze_code(
            "src/db/manager.py",
            "from typing import Dict, List\n"
            "def load_config(path):\n"
            "    with open(path) as f:\n"
            "        return json.load(f)\n"
            "class DB:\n"
            "    def __init__(self, cs: str):\n"
            "        self.pool = None\n"
            "    def query(self, sql): pass\n",
        ),
        "4. Changelog Draft": lambda: ultra_compress_output(
            "commit a1b2c3d4\n"
            "Author: Dev\n"
            "Date:   Thu Jul 16 09:00:00 2026\n\n"
            "    feat(auth): implement refresh token rotation\n\n"
            "commit b2c3d4e5\n"
            "Author: Dev\n"
            "Date:   Thu Jul 16 14:00:00 2026\n\n"
            "    fix(api): correct rate limiting headers\n\n",
            cmd_type="git_log",
        ),
        "5. Issue Triage": lambda: estimate_tokens_precise(
            "BUG: Login fails on null token\n"
            "FEAT: Add CSV export\n"
            "BUG: Memory leak in websocket\n"
            "CHORE: Update Python to 3.12\n"
            "SEC: Dependency CVE-2026-1234\n"
        ),
    }

    print()
    for name, fn in operations.items():
        t0 = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"    {name:<30s} {elapsed:>8.1f}ms")

    _print_sep("Results")
    tracker = get_tracker()
    print(tracker.summary_table())

    # Derived stats
    savings_pct = (
        tracker.total_tokens_saved / max(tracker.total_tokens, 1) * 100
    )

    _print_sep("Summary")
    print(f"    Operations executed : {len(tracker.calls)}")
    print(f"    Total duration      : {tracker.total_duration_ms:.1f}ms")
    print(f"    Total tokens        : {tracker.total_tokens:,}")
    print(f"    Total tokens saved  : {tracker.total_tokens_saved:,}")
    print(f"    Savings rate        : {savings_pct:.1f}%")
    print(f"    Avg per operation   : {tracker.total_duration_ms / max(len(tracker.calls), 1):.1f}ms")
    print()

    # Assertions for the test
    assert len(tracker.calls) == len(operations)
    assert tracker.total_tokens > 0
    assert tracker.total_duration_ms > 0


# ===========================================================================
# Main — standalone runner
# ===========================================================================

def run_all() -> int:
    """Run all tests and return exit code (0 = all passed)."""
    _print_sep("Developer Daily Routine — Test Suite")
    failed = 0
    total = 0

    # Unit tests (adapter level, no server needed)
    unit_tests = [
        ("compress_output", test_compress_output),
        ("ultra_compress_output", test_ultra_compress_output),
        ("estimate_tokens", test_estimate_tokens),
        ("build_context", test_build_context),
        ("analyze_code", test_analyze_code),
    ]

    print("\n  Unit Tests (graphsift_adapter):")
    for name, fn in unit_tests:
        total += 1
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    # Benchmark
    total += 1
    try:
        test_full_routine_benchmark()
    except Exception as e:
        print(f"  [FAIL] Full Routine Benchmark: {e}")
        failed += 1

    # API tests (optional — need running server)
    print("\n  API Integration Tests (optional, server required):")
    for name, fn in [
        ("seed_tasks", test_seed_tasks),
        ("list_tasks", test_list_tasks),
        ("run_all_tasks", test_run_all_tasks),
        ("daily_report", test_daily_report),
    ]:
        total += 1
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    _print_sep()
    print(f"\n  {'ALL PASSED' if failed == 0 else f'{failed}/{total} FAILED'}")
    print()
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(run_all())
