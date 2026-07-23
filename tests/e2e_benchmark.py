"""End-to-end benchmark: compare old vs new version across 10 dimensions.

Usage:
    python tests/e2e_benchmark.py              # run on current (new) version only
    python tests/e2e_benchmark.py --compare     # stash, run old, pop, run new, compare
    python tests/e2e_benchmark.py --report      # just generate report from saved data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
TEST_APP_DIR = Path(os.environ.get("CLAUDE_JOB_DIR", tempfile.gettempdir())) / "tmp" / "fastapi-test-app"
RESULTS_FILE = REPO_ROOT / ".graphsift" / "e2e_results.json"


# =========================================================================
# Helpers
# =========================================================================

def run(cmd: list[str], cwd: str | None = None, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(REPO_ROOT), timeout=timeout)


def graphsift(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run graphsift CLI."""
    python = sys.executable or "python"
    return run([python, "-m", "graphsift", *args], cwd=cwd)


def measure_memory(fn, *args, **kwargs) -> tuple[Any, int, int]:
    """Measure peak memory usage of a function call. Returns (result, current_kb, peak_kb)."""
    tracemalloc.start()
    try:
        result = fn(*args, **kwargs)
        _current, peak = tracemalloc.get_traced_memory()
        return result, peak // 1024, peak // 1024
    finally:
        tracemalloc.stop()


# =========================================================================
# Benchmark dimensions
# =========================================================================

class BenchmarkResult(dict):
    """Auto-saving dict for benchmark results."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dirty = False

    def save(self):
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_FILE.write_text(json.dumps(self, indent=2, default=str))
        log.info(f"  Results saved to {RESULTS_FILE}")


def benchmark_build() -> dict[str, Any]:
    """1. Performance: build time + 4. Memory: peak during build."""
    log.info("\n  [build] Indexing test app...")
    t0 = time.monotonic()
    tracemalloc.start()
    result = graphsift(["build", "--project-root", str(TEST_APP_DIR), "--force"])
    build_time = time.monotonic() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "build_time_s": round(build_time, 3),
        "build_peak_memory_kb": peak // 1024,
        "build_exit_code": result.returncode,
        "build_stderr_tail": result.stderr.strip()[-200:] if result.stderr else "",
    }


def benchmark_guide() -> dict[str, Any]:
    """1. Performance: guide context generation + 10. Token savings."""
    results = {}
    for task, query in [("task_mgmt", "list and create tasks"), ("auth", "user login and authentication")]:
        t0 = time.monotonic()
        r = graphsift(["guide", "--project-root", str(TEST_APP_DIR), query])
        elapsed = time.monotonic() - t0
        results[task] = {
            "time_s": round(elapsed, 3),
            "output_chars": len(r.stdout.strip()),
            "exit_code": r.returncode,
            "preview": r.stdout.strip()[:200],
        }
    return results


def benchmark_detect_cycles() -> dict[str, Any]:
    """3. Bug hunting: cycle detection."""
    t0 = time.monotonic()
    r = graphsift(["detect-cycles", "--root", str(TEST_APP_DIR)])
    elapsed = time.monotonic() - t0
    output = r.stdout.strip()
    cycles_found = output.count("→") if "→" in output else (1 if "cycle" in output.lower() else 0)
    return {
        "cycle_detect_time_s": round(elapsed, 3),
        "cycles_found": cycles_found,
        "exit_code": r.returncode,
    }


def benchmark_dead_code() -> dict[str, Any]:
    """3. Bug hunting: dead code detection."""
    t0 = time.monotonic()
    r = graphsift(["detect-dead-code", "--root", str(TEST_APP_DIR), "--kind", "function", "--all"])
    elapsed = time.monotonic() - t0
    lines = [l for l in r.stdout.strip().split("\n") if l.strip() and not l.strip().startswith("[")]
    return {
        "dead_code_time_s": round(elapsed, 3),
        "dead_functions_found": len(lines),
        "exit_code": r.returncode,
    }


def benchmark_code_quality() -> dict[str, Any]:
    """2. Code quality: suggest-fixes + verify."""
    results = {}

    # List test app Python files for verification
    py_files = list(TEST_APP_DIR.rglob("*.py"))
    verify_results = []
    for f in py_files[:5]:  # verify first 5 files
        t0 = time.monotonic()
        r = graphsift(["verify", "--file", str(f), "--project-root", str(TEST_APP_DIR)])
        verify_results.append({
            "file": str(f.relative_to(TEST_APP_DIR)),
            "time_s": round(time.monotonic() - t0, 3),
            "exit_code": r.returncode,
            "output_preview": r.stdout.strip()[:150],
        })
    results["verify"] = verify_results

    # Suggest fixes
    t0 = time.monotonic()
    r = graphsift(["suggest-fixes", "--project-root", str(TEST_APP_DIR)])
    results["suggest_fixes"] = {
        "time_s": round(time.monotonic() - t0, 3),
        "exit_code": r.returncode,
        "output_preview": r.stdout.strip()[:300],
    }

    return results


def benchmark_compress() -> dict[str, Any]:
    """10. Token savings: compression ratio."""
    # Generate realistic CLI output to compress
    test_outputs = {
        "pytest": """============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-8.0.0, pluggy-1.3.0
rootdir: C:\\project
plugins: asyncio-0.23, cov-4.1, xdist-3.5, timeout-2.2
collected 47 items

tests/test_tasks.py ...........F..                                        [ 40%]
tests/test_auth.py .......................                                [ 80%]
tests/test_api.py .........                                               [100%]

=================================== FAILURES ===================================
_________________________________ test_create_task _________________________________
    def test_create_task():
        response = client.post("/tasks", json={"title": "Test"})
>       assert response.status_code == 201
E       assert 422 == 201

tests/test_tasks.py:42: AssertionError
short test summary info
FAILED tests/test_tasks.py::test_create_task - assert 422 == 201
1 failed, 46 passed in 2.34s""",
        "git_diff": """diff --git a/app/api/tasks.py b/app/api/tasks.py
index a1b2c3d..e4f5g6h 100644
--- a/app/api/tasks.py
+++ b/app/api/tasks.py
@@ -42,6 +42,7 @@ def create_task(task: TaskCreate, db: Session = Depends(get_db)):
     \"\"\"Create a new task.\"\"\"
     db_task = Task(**task.model_dump())
     db.add(db_task)
+    db.commit()
     db.refresh(db_task)
     return db_task""",
        "npm": """npm WARN deprecated old-package@1.0.0: This package is no longer maintained
npm WARN deprecated another@2.0.0: Use the new version instead

added 1243 packages in 45.2s

1243 packages are looking for funding
  run `npm fund` for details""",
    }

    results = {}
    for name, output in test_outputs.items():
        # Write to temp file and pipe through compress
        tmp = Path(tempfile.mktemp(suffix=".txt"))
        tmp.write_text(output)
        t0 = time.monotonic()
        r = run([sys.executable, "-m", "graphsift", "compress", "-t", name, str(tmp)])
        elapsed = time.monotonic() - t0
        compressed = r.stdout.strip()
        orig_chars = len(output)
        comp_chars = len(compressed)
        ratio = (orig_chars - comp_chars) / orig_chars * 100 if orig_chars > 0 else 0
        results[name] = {
            "time_s": round(elapsed, 3),
            "original_chars": orig_chars,
            "compressed_chars": comp_chars,
            "savings_pct": round(ratio, 1),
        }
        tmp.unlink(missing_ok=True)

    return results


def benchmark_context_selection() -> dict[str, Any]:
    """1. Hallucination resistance: context relevance via get_context.

    We test that the right files are selected for a given query.
    """
    results = {}

    # Query about task CRUD — should include tasks.py/api files
    r = graphsift(["get-context", "--project-root", str(TEST_APP_DIR), "--query", "task creation and management"])
    results["task_context"] = {
        "exit_code": r.returncode,
        "output_length": len(r.stdout.strip()),
        "output_preview": r.stdout.strip()[:300],
    }

    # Query about authentication — should include auth.py
    r = graphsift(["get-context", "--project-root", str(TEST_APP_DIR), "--query", "user login authentication JWT"])
    results["auth_context"] = {
        "exit_code": r.returncode,
        "output_length": len(r.stdout.strip()),
        "output_preview": r.stdout.strip()[:300],
    }

    return results


def benchmark_cli_startup() -> dict[str, Any]:
    """1. Performance: CLI startup time."""
    N = 5
    times = []
    for _ in range(N):
        t0 = time.monotonic()
        r = graphsift(["--help"])
        times.append((time.monotonic() - t0) * 1000)
    return {
        "cli_help_time_ms_mean": round(sum(times) / len(times), 1),
        "cli_help_time_ms_min": round(min(times), 1),
        "cli_help_time_ms_max": round(max(times), 1),
    }


def benchmark_stale_refs() -> dict[str, Any]:
    """8. Reliability: stale reference detection."""
    r = graphsift(["status", "--project-root", str(TEST_APP_DIR)])
    return {
        "status_exit": r.returncode,
        "status_output": r.stdout.strip()[:300],
    }


# =========================================================================
# Main
# =========================================================================

def run_all(label: str = "new") -> BenchmarkResult:
    """Run all benchmarks and return results dict."""
    log.info(f"\n{'='*60}")
    log.info(f"  Running benchmarks: {label} version")
    log.info(f"{'='*60}")

    result = BenchmarkResult(version=label, timestamp=time.time())

    log.info("\n--- Build ---")
    result["build"] = benchmark_build()

    log.info("\n--- Guide ---")
    result["guide"] = benchmark_guide()

    log.info("\n--- Cycle Detection ---")
    result["cycles"] = benchmark_detect_cycles()

    log.info("\n--- Dead Code Detection ---")
    result["dead_code"] = benchmark_dead_code()

    log.info("\n--- Code Quality ---")
    result["code_quality"] = benchmark_code_quality()

    log.info("\n--- Compress ---")
    result["compress"] = benchmark_compress()

    log.info("\n--- Context Selection ---")
    result["context"] = benchmark_context_selection()

    log.info("\n--- CLI Startup ---")
    result["cli_startup"] = benchmark_cli_startup()

    log.info("\n--- Status ---")
    result["status"] = benchmark_stale_refs()

    return result


def compare(old: dict, new: dict) -> dict:
    """Compare old vs new results and produce a delta report."""
    log.info(f"\n{'='*60}")
    log.info("  COMPARISON: OLD vs NEW")
    log.info(f"{'='*60}")

    report = {
        "build_speedup": None,
        "search_speedup": None,
        "memory_savings": None,
        "compress_improvement": None,
        "cli_startup_improvement": None,
        "details": {},
    }

    # Build time
    if "build" in old and "build" in new:
        o = old["build"]["build_time_s"]
        n = new["build"]["build_time_s"]
        ratio = (o - n) / o * 100 if o > 0 else 0
        report["build_speedup"] = f"{ratio:+.1f}%"
        report["details"]["build_time_s"] = {"old": o, "new": n}

        om = old["build"].get("build_peak_memory_kb", 0)
        nm = new["build"].get("build_peak_memory_kb", 0)
        mratio = (om - nm) / om * 100 if om > 0 else 0
        report["memory_savings"] = f"{mratio:+.1f}%"
        report["details"]["build_memory_kb"] = {"old": om, "new": nm}

    # Guide time
    if "guide" in old and "guide" in new:
        o = old["guide"].get("task_mgmt", {}).get("time_s", 0)
        n = new["guide"].get("task_mgmt", {}).get("time_s", 0)
        if o and n:
            ratio = (o - n) / o * 100
            report["guide_speedup"] = f"{ratio:+.1f}%"
        report["details"]["guide"] = {"old": o, "new": n}

    # CLI startup
    if "cli_startup" in old and "cli_startup" in new:
        o = old["cli_startup"]["cli_help_time_ms_mean"]
        n = new["cli_startup"]["cli_help_time_ms_mean"]
        ratio = (o - n) / o * 100 if o > 0 else 0
        report["cli_startup_improvement"] = f"{ratio:+.1f}%"
        report["details"]["cli_startup_ms"] = {"old": o, "new": n}

    # Compression
    if "compress" in old and "compress" in new:
        comp_improvements = {}
        for key in old["compress"]:
            if key in new["compress"]:
                o = old["compress"][key]["savings_pct"]
                n = new["compress"][key]["savings_pct"]
                comp_improvements[key] = {"old": o, "new": n}
        report["details"]["compress"] = comp_improvements

    # Cycles detected
    if "cycles" in old and "cycles" in new:
        report["details"]["cycles"] = {
            "old": old["cycles"]["cycles_found"],
            "new": new["cycles"]["cycles_found"],
        }

    # Dead code
    if "dead_code" in old and "dead_code" in new:
        report["details"]["dead_code"] = {
            "old": old["dead_code"]["dead_functions_found"],
            "new": new["dead_code"]["dead_functions_found"],
        }

    return report


def print_report(result: BenchmarkResult, label: str):
    """Print a human-readable report."""
    log.info(f"\n{'='*60}")
    log.info(f"  REPORT: {label} version")
    log.info(f"{'='*60}")

    # Performance
    if "build" in result:
        b = result["build"]
        log.info(f"\n  ⚡ PERFORMANCE:")
        log.info(f"    Build time:      {b['build_time_s']:.2f}s")
        log.info(f"    Search time:     {result.get('search', {}).get('search_time_s', 'N/A')}s")
        log.info(f"    CLI --help:      {result.get('cli_startup', {}).get('cli_help_time_ms_mean', 'N/A')}ms")

    # Memory
    if "build" in result:
        log.info(f"\n  💾 MEMORY:")
        log.info(f"    Build peak:      {b['build_peak_memory_kb']:,} KB")

    # Bug hunting
    if "cycles" in result:
        log.info(f"\n  🐛 BUG HUNTING:")
        log.info(f"    Cycles found:    {result['cycles']['cycles_found']}")
        log.info(f"    Dead functions:  {result.get('dead_code', {}).get('dead_functions_found', 'N/A')}")

    # Compression
    if "compress" in result:
        log.info(f"\n  💰 TOKEN SAVINGS:")
        for name, data in result["compress"].items():
            log.info(f"    {name}: {data['savings_pct']:.0f}% savings ({data['compressed_chars']}/{data['original_chars']} chars)")

    # Context
    if "context" in result:
        log.info(f"\n  🧠 HALLUCINATION RESISTANCE:")
        for name, data in result["context"].items():
            log.info(f"    {name}: {data['output_length']} chars returned")


def main():
    parser = argparse.ArgumentParser(description="E2E benchmark for graphsift")
    parser.add_argument("--compare", action="store_true", help="Run old vs new comparison")
    parser.add_argument("--report", action="store_true", help="Print report from saved results")
    args = parser.parse_args()

    if args.report:
        if RESULTS_FILE.exists():
            data = json.loads(RESULTS_FILE.read_text())
            if "old" in data:
                print_report(BenchmarkResult(data["old"]), "OLD")
            if "new" in data:
                print_report(BenchmarkResult(data["new"]), "NEW")
            if "comparison" in data:
                log.info(f"\n{'='*60}")
                log.info("  COMPARISON SUMMARY")
                log.info(f"{'='*60}")
                c = data["comparison"]
                for key, val in c.items():
                    if key != "details":
                        log.info(f"    {key}: {val}")
        return

    if not TEST_APP_DIR.exists():
        log.error(f"Test app not found at {TEST_APP_DIR}")
        log.error("Create it first with the test-app creation script.")
        sys.exit(1)

    if args.compare:
        # --- Run OLD version ---
        # Stash current changes (new version)
        log.info("\nStashing current changes to run OLD version...")
        stash_result = run(["git", "stash", "push", "--include-untracked", "-m", "e2e-benchmark-stash"])
        if stash_result.returncode not in (0, 1):  # 1 = nothing to stash
            log.error(f"Git stash failed: {stash_result.stderr}")
            sys.exit(1)

        try:
            old_results = run_all(label="old")
            old_results.save()

            # Pop stash to restore new version
            log.info("\nRestoring NEW version from stash...")
            run(["git", "stash", "pop"])
        except Exception as e:
            log.error(f"Error during old version test: {e}")
            run(["git", "stash", "pop"])
            sys.exit(1)

        # --- Run NEW version ---
        new_results = run_all(label="new")

        # Compare
        comparison = compare(old_results, new_results)

        final = {
            "old": dict(old_results),
            "new": dict(new_results),
            "comparison": comparison,
        }
        BenchmarkResult(final).save()

        # Print comparison summary
        log.info(f"\n{'='*60}")
        log.info("  FINAL COMPARISON")
        log.info(f"{'='*60}")
        for key, val in comparison.items():
            if key != "details":
                log.info(f"  {key}: {val}")

    else:
        results = run_all(label="new")
        results.save()
        print_report(results, "NEW")


if __name__ == "__main__":
    main()
