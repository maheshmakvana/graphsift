"""Developer Daily Routine — FastAPI backend.

Simulates a developer's daily workflow tasks with token-aware metrics
powered by graphsift. Each task uses graphsift adapters for compression,
context building, and code analysis.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graphsift_adapter import (
    analyze_code,
    build_context,
    compress_output,
    estimate_tokens_precise,
    get_tracker,
    reset_tracker,
    ultra_compress_output,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Developer Daily Routine",
    version="1.0.0",
    description="Daily developer workflow simulator powered by graphsift",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML dashboard from the root
_HERE = Path(__file__).resolve().parent


@app.get("/")
async def serve_dashboard() -> FileResponse:
    """Serve the HTML dashboard."""
    return FileResponse(str(_HERE / "index.html"))

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class TaskType(str, Enum):
    CODE_REVIEW = "code_review"
    PR_CHECKS = "pr_checks"
    DEPENDENCY_SCAN = "dependency_scan"
    CHANGELOG = "changelog"
    ISSUE_TRIAGE = "issue_triage"
    CUSTOM = "custom"


class TaskCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    task_type: TaskType = TaskType.CUSTOM


class TaskOut(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    status: TaskStatus
    tokens_used: int = 0
    duration_ms: float = 0.0
    result: str = ""
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class ReportOut(BaseModel):
    total_tasks: int
    completed: int
    failed: int
    pending: int
    total_tokens: int
    total_duration_ms: float
    avg_duration_ms: float
    completion_rate: float
    tasks: list[TaskOut]
    graphsift_metrics: dict[str, Any]


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_tasks: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_to_out(task: dict[str, Any]) -> dict[str, Any]:
    """Convert internal task dict to response-safe dict."""
    return {k: v for k, v in task.items()}


# ---------------------------------------------------------------------------
# Simulation functions (each uses graphsift adapter)
# ---------------------------------------------------------------------------


def _code_review_sim(task_id: str) -> dict[str, Any]:
    """Simulate code review using graphsift context builder."""
    sample_files = {
        "src/auth/login.py": (
            "def authenticate_user(username: str, password: str) -> bool:\n"
            '    """Authenticate a user against the database."""\n'
            "    user = db.query(User).filter(User.username == username).first()\n"
            "    if not user:\n"
            "        return False\n"
            "    return verify_password(password, user.password_hash)\n"
        ),
        "src/auth/session.py": (
            "from datetime import timedelta\n"
            "def create_session(user_id: int) -> str:\n"
            '    """Create a new session token for the user."""\n'
            "    token = secrets.token_urlsafe(32)\n"
            "    redis.setex(f'session:{token}', timedelta(hours=24), user_id)\n"
            "    return token\n"
        ),
    }
    context = build_context(
        sample_files, query="Review authentication code for security issues"
    )
    issues_found = random.randint(2, 6)
    return {
        "result": (
            f"Code review complete. Files analyzed: {len(sample_files)}. "
            f"Issues found: {issues_found}."
        ),
        "tokens_used": estimate_tokens_precise(context),
        "duration_ms": random.uniform(800, 2500),
        "context_preview": (
            context[:300] + "..." if len(context) > 300 else context
        ),
    }


def _pr_checks_sim(task_id: str) -> dict[str, Any]:
    """Simulate CI log triage using graphsift compression."""
    raw_log = (
        "========================================\n"
        "CI Pipeline Run #8472 -- 2026-07-16 08:42 UTC\n"
        "========================================\n"
        "[2026-07-16T08:42:01Z] Running pytest on branch: feature/auth-refactor\n"
        "[2026-07-16T08:42:03Z] Collected 247 items\n"
        "[2026-07-16T08:42:05Z] tests/test_auth.py::test_login_ok PASSED               [  0%]\n"
        "[2026-07-16T08:42:06Z] tests/test_auth.py::test_login_fail PASSED             [  1%]\n"
        "[2026-07-16T08:42:08Z] tests/test_auth.py::test_session_expiry FAILED         [  2%]\n"
        "[2026-07-16T08:42:08Z] tests/test_auth.py::test_token_refresh PASSED          [  3%]\n"
        "[2026-07-16T08:42:10Z] tests/test_api.py::test_rate_limit PASSED              [  4%]\n"
        "[2026-07-16T08:42:12Z] tests/test_auth.py::test_session_expiry FAILED         [  5%]\n"
        "[2026-07-16T08:42:15Z] tests/test_auth.py::test_concurrent_sessions FAILED    [  6%]\n"
        "[2026-07-16T08:42:18Z] tests/test_db.py::test_migration PASSED                [  7%]\n"
        "[2026-07-16T08:42:20Z] tests/test_auth.py::test_2fa PASSED                    [  8%]\n"
        "--- snip 230 lines ---\n"
        "[2026-07-16T08:45:01Z] === short test summary info ===\n"
        "[2026-07-16T08:45:01Z] FAILED tests/test_auth.py::test_session_expiry\n"
        "[2026-07-16T08:45:01Z] FAILED tests/test_auth.py::test_concurrent_sessions\n"
        "[2026-07-16T08:45:01Z] 245 passed, 2 failed in 182.3s\n"
    )
    compressed = compress_output(raw_log, cmd_type="pytest")
    return {
        "result": (
            f"CI triage complete. Compressed {len(raw_log)} chars "
            f"to {len(compressed)} chars."
        ),
        "tokens_used": estimate_tokens_precise(compressed),
        "duration_ms": random.uniform(300, 1000),
        "compressed_preview": (
            compressed[:300] + "..." if len(compressed) > 300 else compressed
        ),
    }


def _dependency_scan_sim(task_id: str) -> dict[str, Any]:
    """Simulate dependency scan using graphsift code analysis."""
    sample_source = (
        "import os\n"
        "import sys\n"
        "import json\n"
        "from typing import Dict, List, Optional\n"
        "from datetime import datetime\n"
        "\n"
        "def load_config(path: str) -> Dict:\n"
        "    with open(path) as f:\n"
        "        return json.load(f)\n"
        "\n"
        "def process_items(items: List[str]) -> List[str]:\n"
        "    result = []\n"
        "    for i, item in enumerate(items):\n"
        "        result.append(item.strip().lower())\n"
        "    return result\n"
        "\n"
        "class DatabaseManager:\n"
        "    def __init__(self, connection_string: str):\n"
        "        self.conn = self._create_pool(connection_string)\n"
        "\n"
        "    def _create_pool(self, cs: str):\n"
        "        return {'pool': cs, 'size': 10}\n"
        "\n"
        "    def query(self, sql: str, params: tuple = None):\n"
        "        if params is None:\n"
        "            params = ()\n"
        "        return self.conn['pool'] + str(params)\n"
    )
    suggestions = analyze_code("src/db/manager.py", sample_source)
    return {
        "result": (
            f"Dependency scan complete. {len(suggestions)} findings "
            f"detected via graph analysis."
        ),
        "tokens_used": estimate_tokens_precise(str(suggestions)),
        "duration_ms": random.uniform(500, 2000),
        "suggestions": suggestions[:5] if suggestions else [],
    }


def _changelog_sim(task_id: str) -> dict[str, Any]:
    """Simulate changelog generation using ultra compression."""
    raw_commits = (
        "commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0\n"
        "Author: Dev <dev@example.com>\n"
        "Date:   Thu Jul 16 09:00:00 2026 +0000\n\n"
        "    feat(auth): implement OAuth2 refresh token rotation\n\n"
        "    - Add refresh token rotation with explicit revocation\n"
        "    - Update session manager to handle concurrent token usage\n\n"
        "commit b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1\n"
        "Author: Dev <dev@example.com>\n"
        "Date:   Thu Jul 16 10:30:00 2026 +0000\n\n"
        "    fix(api): correct rate limiting headers for 429 responses\n\n"
        "    - Retry-After header now reflects actual reset time\n\n"
        "commit c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2\n"
        "Author: Dev <dev@example.com>\n"
        "Date:   Thu Jul 16 14:15:00 2026 +0000\n\n"
        "    chore(deps): bump requests from 2.31.0 to 2.32.0\n\n"
        "    - Security patch for CVE-2026-1234\n\n"
        "commit d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3\n"
        "Author: Dev <dev@example.com>\n"
        "Date:   Thu Jul 16 16:45:00 2026 +0000\n\n"
        "    docs: update API reference with new endpoint examples\n\n"
        "    - Added OpenAPI 3.1 examples for auth endpoints\n"
    )
    compressed = ultra_compress_output(raw_commits, cmd_type="git_log")
    return {
        "result": (
            f"Changelog draft generated. Compressed to {len(compressed)} chars."
        ),
        "tokens_used": estimate_tokens_precise(compressed),
        "duration_ms": random.uniform(200, 800),
        "changelog_preview": (
            compressed[:400] + "..." if len(compressed) > 400 else compressed
        ),
    }


def _issue_triage_sim(task_id: str) -> dict[str, Any]:
    """Simulate issue triage using token estimation."""
    issues = [
        "BUG: Login page returns 500 on invalid token format",
        "FEAT: Add export-to-CSV feature for dashboard",
        "BUG: Memory leak in websocket connection handler",
        "CHORE: Update CI pipeline to use Python 3.12",
        "FEAT: Implement dark mode toggle",
        "BUG: Rate limiter not firing for authenticated endpoints",
        "SEC: Dependency requests@2.31.0 has CVE-2026-1234",
        "FEAT: Add webhook notification system",
    ]
    token_est = estimate_tokens_precise("\n".join(issues))
    return {
        "result": (
            f"Issue triage complete. {len(issues)} issues processed, "
            f"~{token_est} tokens estimated."
        ),
        "tokens_used": token_est,
        "duration_ms": random.uniform(100, 500),
        "issues": issues,
    }


def _custom_sim(task_id: str) -> dict[str, Any]:
    """Simulate a custom/generic task."""
    time.sleep(0.05)
    sample_text = (
        "Task executed successfully.\n"
        "  - Operation: custom processing\n"
        "  - Status: completed\n"
        "  - Resources: normal\n"
    )
    compressed = compress_output(sample_text)
    return {
        "result": compressed,
        "tokens_used": estimate_tokens_precise(compressed),
        "duration_ms": random.uniform(50, 300),
    }


# ---------------------------------------------------------------------------
# Task template registry (maps TaskType -> metadata + simulation function)
# ---------------------------------------------------------------------------

TASK_TEMPLATES: dict[TaskType, dict[str, Any]] = {
    TaskType.CODE_REVIEW: {
        "name": "Code Review -- Yesterday's PRs",
        "description": (
            "Review outstanding pull requests using graphsift context "
            "builder for targeted diff analysis."
        ),
        "graphsift_op": "build_context",
        "simulate": _code_review_sim,
    },
    TaskType.PR_CHECKS: {
        "name": "CI Failure Triage",
        "description": (
            "Triage CI pipeline failures, compress logs with graphsift "
            "to identify root cause."
        ),
        "graphsift_op": "compress_output",
        "simulate": _pr_checks_sim,
    },
    TaskType.DEPENDENCY_SCAN: {
        "name": "Dependency Vulnerability Scan",
        "description": (
            "Scan dependencies for known vulnerabilities using graphsift "
            "dependency graph analysis."
        ),
        "graphsift_op": "analyze_code",
        "simulate": _dependency_scan_sim,
    },
    TaskType.CHANGELOG: {
        "name": "Changelog Draft Generation",
        "description": (
            "Draft changelog from recent commits, compress with graphsift "
            "for concise release notes."
        ),
        "graphsift_op": "ultra_compress",
        "simulate": _changelog_sim,
    },
    TaskType.ISSUE_TRIAGE: {
        "name": "Issue Triage Queue",
        "description": (
            "Process issue triage queue, classify and prioritize using "
            "graphsift analytics."
        ),
        "graphsift_op": "estimate_tokens",
        "simulate": _issue_triage_sim,
    },
}

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/tasks", response_model=list[TaskOut])
async def list_tasks() -> list[dict[str, Any]]:
    """List all tasks, ordered by creation time."""
    tasks = sorted(_tasks.values(), key=lambda t: t["created_at"])
    return [_task_to_out(t) for t in tasks]


@app.post("/api/tasks", response_model=None, status_code=201)
async def create_task(body: TaskCreate) -> JSONResponse:
    """Create a new task. Use name='__seed_defaults__' to seed defaults."""
    if body.name == "__seed_defaults__":
        await _seed_default_tasks()
        return JSONResponse(
            content={"message": f"Seeded {len(_tasks)} default tasks."},
            status_code=201,
        )

    task_id = str(uuid.uuid4())
    now = _now()
    task: dict[str, Any] = {
        "id": task_id,
        "name": body.name,
        "description": body.description,
        "task_type": body.task_type.value,
        "status": TaskStatus.PENDING,
        "tokens_used": 0,
        "duration_ms": 0.0,
        "result": "",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
    }
    _tasks[task_id] = task
    return JSONResponse(content=_task_to_out(task), status_code=201)


@app.post("/api/tasks/{task_id}/run", response_model=TaskOut)
async def run_task(task_id: str) -> dict[str, Any]:
    """Simulate running a single task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task["status"] = TaskStatus.RUNNING
    task["started_at"] = _now()

    try:
        ttype = TaskType(task["task_type"])
        template = TASK_TEMPLATES.get(ttype)
        if template and ttype != TaskType.CUSTOM:
            sim_result = template["simulate"](task_id)
        else:
            sim_result = _custom_sim(task_id)

        task["result"] = sim_result.get("result", "Task completed.")
        task["tokens_used"] = sim_result.get("tokens_used", 0)
        task["duration_ms"] = sim_result.get("duration_ms", random.uniform(100, 500))
        task["status"] = TaskStatus.DONE
        task["completed_at"] = _now()
    except Exception as exc:
        task["status"] = TaskStatus.FAILED
        task["result"] = f"Error: {exc}"
        task["completed_at"] = _now()
        task["duration_ms"] = 0.0

    return _task_to_out(task)


@app.post("/api/tasks/run-all", response_model=list[TaskOut])
async def run_all_tasks() -> list[dict[str, Any]]:
    """Run all pending tasks sequentially."""
    reset_tracker()
    results = []
    for task_id in list(_tasks.keys()):
        task = _tasks[task_id]
        if task["status"] == TaskStatus.PENDING:
            r = await run_task(task_id)
            results.append(r)
    return [_task_to_out(_tasks[tid]) for tid in _tasks]


@app.get("/api/report", response_model=ReportOut)
async def daily_report() -> dict[str, Any]:
    """Generate the daily summary report with graphsift metrics."""
    tracker = get_tracker()
    tasks_list = [_task_to_out(t) for t in _tasks.values()]

    completed = sum(1 for t in tasks_list if t["status"] == TaskStatus.DONE)
    failed = sum(1 for t in tasks_list if t["status"] == TaskStatus.FAILED)
    pending = sum(1 for t in tasks_list if t["status"] == TaskStatus.PENDING)
    total = len(tasks_list)

    total_dur = sum(t["duration_ms"] for t in tasks_list)
    total_tok = sum(t["tokens_used"] for t in tasks_list)

    return {
        "total_tasks": total,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "total_tokens": total_tok,
        "total_duration_ms": round(total_dur, 1),
        "avg_duration_ms": round(total_dur / max(completed, 1), 1),
        "completion_rate": round(completed / max(total, 1) * 100, 1),
        "tasks": tasks_list,
        "graphsift_metrics": _graphsift_metrics(),
    }


@app.post("/api/reset", status_code=204)
async def reset_tasks() -> JSONResponse:
    """Reset all tasks and tracker data."""
    _tasks.clear()
    reset_tracker()
    return JSONResponse(content=None, status_code=204)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _seed_default_tasks() -> None:
    """Seed the default daily routine tasks."""
    _tasks.clear()
    reset_tracker()
    seeds = [
        (
            "Code Review -- Yesterday's PRs",
            "Review outstanding pull requests using graphsift",
            TaskType.CODE_REVIEW,
        ),
        (
            "CI Failure Triage",
            "Triage CI pipeline failures with compressed logs",
            TaskType.PR_CHECKS,
        ),
        (
            "Dependency Vulnerability Scan",
            "Scan dependencies for known vulnerabilities",
            TaskType.DEPENDENCY_SCAN,
        ),
        (
            "Changelog Draft Generation",
            "Draft changelog from recent commits",
            TaskType.CHANGELOG,
        ),
        (
            "Issue Triage Queue",
            "Process and prioritize open issues",
            TaskType.ISSUE_TRIAGE,
        ),
    ]
    for name, desc, ttype in seeds:
        tid = str(uuid.uuid4())
        _tasks[tid] = {
            "id": tid,
            "name": name,
            "description": desc,
            "task_type": ttype.value,
            "status": TaskStatus.PENDING,
            "tokens_used": 0,
            "duration_ms": 0.0,
            "result": "",
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
        }


def _graphsift_metrics() -> dict[str, Any]:
    """Aggregate graphsift adapter metrics."""
    tracker = get_tracker()
    return {
        "total_calls": len(tracker.calls),
        "total_duration_ms": round(tracker.total_duration_ms, 1),
        "total_tokens_processed": tracker.total_tokens,
        "total_tokens_saved": tracker.total_tokens_saved,
        "calls": [
            {
                "operation": c.operation,
                "duration_ms": round(c.duration_ms, 1),
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "tokens_saved": c.tokens_saved,
            }
            for c in tracker.calls
        ],
    }


# ---------------------------------------------------------------------------
# Startup hook — seed default tasks
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    """Seed default tasks on first startup."""
    await _seed_default_tasks()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
