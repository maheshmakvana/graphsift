"""Reproducible compression benchmarks for graphsift.
Usage: python benchmark/run_benchmarks.py
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphsift.compress import (
    compress_pytest, compress_cargo, compress_go_test, compress_jest,
    compress_eslint, compress_git_status, compress_git_diff, compress_git_log,
    compress_grep, compress_npm, compress_docker, compress_kubectl,
    compress_aws, compress_json_output, compress_make, compress_pip,
    compress_log, compress_cat, compress_generic,
)

# Realistic sample outputs — edit to test your own
SAMPLES: dict[str, str] = {}

# --- Test runner outputs ---
SAMPLES["pytest"] = (
    "============================= test session starts ==================\n"
    "platform linux -- Python 3.11.6, pytest-7.4.3\n"
    "collected 45 items\n\n"
    "tests/test_auth.py::test_login_success PASSED               [  2%]\n"
    "tests/test_auth.py::test_login_failure PASSED               [  4%]\n"
    "tests/test_auth.py::test_token_refresh PASSED               [  8%]\n"
    # ... 41 more passing ...
    "tests/test_auth.py::test_mfa_enroll FAILED                  [ 20%]\n"
    "tests/test_auth.py::test_brute_force_detection FAILED       [ 31%]\n"
    "tests/test_api.py::test_auth_required FAILED                [ 62%]\n"
    "tests/test_models.py::test_user_creation PASSED             [ 68%]\n"
    "tests/test_utils.py::test_validate_email PASSED             [100%]\n\n"
    "=================================== FAILURES ========================\n"
    "____________ test_mfa_enroll ______________\n"
    "tests/test_auth.py:78:\n"
    ">   assert response.status_code == 200\n"
    "E   assert 500 == 200\n"
    "____________ test_brute_force_detection ____\n"
    "tests/test_auth.py:201:\n"
    ">   assert detector.triggered is True\n"
    "E   assert False is True\n"
    "======================== 3 failed, 42 passed in 12.34s ==============\n"
)

SAMPLES["jest"] = (
    "PASS  src/auth/__tests__/login.test.ts\n"
    "  ✓ should authenticate with valid credentials (25 ms)\n"
    "  ✓ should reject invalid password (12 ms)\n"
    "PASS  src/auth/__tests__/session.test.ts\n"
    "  ✓ should create a new session (10 ms)\n"
    "FAIL  src/auth/__tests__/mfa.test.ts\n"
    "  ● MFA Verification › should verify valid TOTP code\n"
    "    Expected: true, Received: false (mfa.test.ts:36:20)\n"
    "Test Suites: 1 failed, 2 passed, 3 total\n"
    "Tests: 1 failed, 5 passed, 6 total\n"
)

SAMPLES["go_test"] = (
    "=== RUN   TestAuthenticate\n"
    "--- PASS: TestAuthenticate (0.12s)\n"
    "=== RUN   TestRefreshSession\n"
    "--- FAIL: TestRefreshSession (0.05s)\n"
    "    manager_test.go:78: expected status 200, got 500\n"
    "=== RUN   TestBruteForceDetection\n"
    "--- FAIL: TestBruteForceDetection (0.02s)\n"
    "panic: runtime error: nil pointer dereference\n"
    "FAIL\tmyapp/internal/auth\t0.432s\n"
)

# --- Git outputs ---
SAMPLES["git_diff"] = (
    "diff --git a/src/auth/manager.py b/src/auth/manager.py\n"
    "index a1b2c3d..e4f5g6h 100644\n"
    "--- a/src/auth/manager.py\n"
    "+++ b/src/auth/manager.py\n"
    "@@ -12,8 +12,8 @@\n"
    "-from .models import User, Session\n"
    "+from .models import User, Session, RefreshToken\n"
    "@@ -34,7 +34,9 @@ class AuthManager:\n"
    "     def authenticate(self, email, password, mfa_code=None):\n"
    "+        if user.mfa_enabled:\n"
    "+            self._verify_mfa(user, mfa_code)\n"
    "         return session\n"
    "+    def refresh_session(self, refresh_token):\n"
    "+        return self._refresh(refresh_token)\n"
)

SAMPLES["git_status"] = (
    "On branch master\n"
    "Your branch is up to date with 'origin/master'.\n"
    "Changes not staged for commit:\n"
    "\tmodified:   src/auth/manager.py\n"
    "\tmodified:   src/auth/models.py\n"
    "\tmodified:   src/api/middleware.py\n"
    "\tmodified:   tests/test_auth.py\n"
    "Untracked files:\n"
    "\tsrc/auth/providers/oauth2.py\n"
    "\tsrc/auth/mfa.py\n"
    "\tmigrations/004_add_mfa.sql\n"
)

SAMPLES["git_log"] = (
    "commit e4f5g6h7i8j9k0l1 (HEAD -> master)\n"
    "Author: Jane Developer <jane@example.com>\n"
    "Date:   Mon Jan 15 14:32:10 2026 -0500\n"
    "    feat: add MFA support and refresh token rotation\n"
    "commit a1b2c3d4e5f6g7h8i9\n"
    "Author: Jane Developer <jane@example.com>\n"
    "Date:   Fri Jan 12 09:15:45 2026 -0500\n"
    "    fix: resolve session timeout race condition\n"
    "commit b2c3d4e5f6g7h8i9j0\n"
    "Author: Bob Reviewer <bob@example.com>\n"
    "Date:   Wed Jan 10 16:47:22 2026 -0500\n"
    "    refactor: extract auth providers into plugin architecture\n"
)

# --- Container outputs ---
SAMPLES["docker"] = (
    "CONTAINER ID   IMAGE                 NAMES\n"
    "a1b2c3d4e5f6   myapp/api:v2.3.1      api-prod-1\n"
    "b2c3d4e5f6g7   myapp/api:v2.3.1      api-prod-2\n"
    "c3d4e5f6g7h8   myapp/worker:v2.3.1   worker-1\n"
    "d4e5f6g7h8i9   redis:7.2-alpine      redis-cache\n"
    "e5f6g7h8i9j0   postgres:16-alpine    postgres-db\n"
    "f6g7h8i9j0k1   nginx:1.25-alpine     nginx-lb\n"
    "g7h8i9j0k1l2   grafana/grafana:10.2  grafana\n"
    "h8i9j0k1l2m3   prom/prometheus:v2.48 prometheus\n"
    "i9j0k1l2m3n4   elasticsearch:8.11    elasticsearch\n"
    "j0k1l2m3n4o5   rabbitmq:3.12         rabbitmq\n"
)

SAMPLES["kubectl"] = (
    "NAME                                          READY   STATUS    RESTARTS   AGE\n"
    "pod/api-deployment-7d8f9c6b5-abc12            1/1     Running   0          2d\n"
    "pod/api-deployment-7d8f9c6b5-def34            1/1     Running   0          2d\n"
    "pod/worker-deployment-6c7d8e9f0-jkl78         1/1     Running   0          5d\n"
    "pod/redis-master-5b6c7d8e9-pqr12              1/1     Running   0          7d\n"
    "pod/postgres-statefulset-0                    1/1     Running   0          7d\n"
    "service/api-service      ClusterIP   10.96.100.1     8000/TCP         30d\n"
    "service/nginx-lb         LoadBalancer 10.96.100.5   80:30080/TCP     30d\n"
    "deployment.apps/api-deployment             3/3     3            3           30d\n"
)

# --- Package manager outputs ---
SAMPLES["npm"] = (
    "npm WARN deprecated uuid@3.4.0: Please upgrade\n"
    "npm ERR! code ERESOLVE\n"
    "npm ERR! ERESOLVE could not resolve\n"
    "npm ERR! Found: react@18.2.0\n"
    "npm ERR! Could not resolve dependency:\n"
    "npm ERR! peer react@\"^17.0.0\" from @deprecated/lib@3.1.0\n"
    "added 423 packages, removed 87 packages, changed 156 packages in 45s\n"
)

SAMPLES["pip"] = (
    "Collecting fastapi>=0.104.0\n"
    "  Downloading fastapi-0.109.0-py3-none-any.whl (92 kB)\n"
    "Collecting sqlalchemy>=2.0.23\n"
    "  Downloading sqlalchemy-2.0.25.whl (3.1 MB)\n"
    "ERROR: Could not find a version that satisfies the requirement internal-auth-lib==2.1.0\n"
    "ERROR: No matching distribution found for internal-auth-lib==2.1.0\n"
)

SAMPLES["cargo"] = (
    "   Compiling myapp v0.1.0\n"
    "error[E0308]: mismatched types\n"
    "  --> src/auth/manager.rs:45:9\n"
    "   = note: expected `User`, found `Option<User>`\n"
    "error[E0596]: cannot borrow `self.sessions` as mutable\n"
    "  --> src/auth/manager.rs:67:9\n"
    "warning: unused import: `std::collections::HashMap`\n"
    "error: could not compile `myapp` due to 2 errors\n"
)

# --- Search / lint outputs ---
SAMPLES["grep"] = (
    "src/auth/manager.py:34:    def authenticate(self, email, password):\n"
    "src/auth/manager.py:59:        if not verify_password(password, user.hashed_password):\n"
    "src/auth/manager.py:62:        if user.mfa_enabled and not self._verify_mfa(user, mfa_code):\n"
    "src/auth/manager.py:78:    def refresh_session(self, refresh_token):\n"
    "src/auth/manager.py:90:    def _record_failed_attempt(self, user_id):\n"
    "src/auth/models.py:27:    hashed_password: str\n"
    "src/auth/models.py:29:    mfa_enabled: bool = False\n"
    "src/auth/providers/base.py:45:    def verify_password(self, hashed, plain):\n"
    "tests/test_auth.py:34:    def test_login_success(self):\n"
    "tests/test_auth.py:78:    def test_mfa_enroll(self):\n"
)

SAMPLES["eslint"] = (
    "/workspace/src/auth/manager.py\n"
    "  12:1   error    'Optional' imported but never used\n"
    "  34:5   warning  Function 'authenticate' has too many parameters\n"
    "  62:9   error    'mfa_code' is not defined\n"
    "/workspace/src/auth/models.py\n"
    "  27:3   warning  Missing trailing comma\n"
    "/workspace/tests/test_auth.py\n"
    "  78:5   error    Test fixture not found\n"
    "✖ 6 problems (4 errors, 2 warnings)\n"
)

# --- Infrastructure outputs ---
SAMPLES["log"] = (
    "2024-01-15T14:30:00.001Z [INFO] app.server: Starting HTTP server\n"
    "2024-01-15T14:30:00.125Z [INFO] app.database: Connection pool established\n"
    "2024-01-15T14:32:15.300Z [WARN] app.auth: Failed login user_id=abc123\n"
    "2024-01-15T14:33:45.000Z [ERROR] app.api: Unhandled exception: division by zero\n"
    "2024-01-15T14:35:00.000Z [ERROR] app.auth: Brute force threshold exceeded user_id=xyz456\n"
    "2024-01-15T14:35:10.100Z [FATAL] app.database: Connection pool exhausted (20/20)\n"
)

SAMPLES["make"] = (
    "make[1]: Entering directory '/home/runner/myapp'\n"
    "[ 25%] Building CXX object src/core/CMakeFiles/core.dir/engine.cpp.o\n"
    "[ 50%] Building CXX object src/core/CMakeFiles/core.dir/parser.cpp.o\n"
    "[ 75%] Building CXX object src/core/CMakeFiles/core.dir/graph.cpp.o\n"
    "/workspace/src/core/graph.cpp:345:20: error: 'stack' was not declared\n"
    "  345 |     std::vector<int> result; stack<int> stk;\n"
    "make[1]: *** [CMakeFiles/core.dir/graph.cpp.o] Error 1\n"
    "make: *** [Makefile:84: all] Error 2\n"
)

SAMPLES["aws"] = (
    '{"Reservations":[{"Groups":[],"Instances":[{'
    '"InstanceId":"i-0abcd1234efgh5678","InstanceType":"t3.xlarge",'
    '"State":{"Code":16,"Name":"running"},'
    '"PrivateIpAddress":"10.0.1.45","PublicIpAddress":"34.201.50.10",'
    '"Tags":[{"Key":"Name","Value":"api-prod-1"},{"Key":"Environment","Value":"production"}]'
    '}],"OwnerId":"123456789012"}]}'
)

# --- Pass-through outputs ---
SAMPLES["cat"] = (
    "#!/usr/bin/env python3\n"
    "\"\"\"Enterprise authentication service.\"\"\"\n"
    "from __future__ import annotations\n"
    "import hashlib, secrets, time\n"
    "from datetime import datetime, timedelta\n"
    "from typing import Optional, Dict, Any\n"
    "import jwt, redis\n"
    "from pydantic import BaseModel, Field\n\n"
    "class User(BaseModel):\n"
    "    id: str\n"
    "    email: str\n"
    "    hashed_password: str\n"
    "    mfa_enabled: bool = False\n\n"
    "class AuthManager:\n"
    "    def authenticate(self, email, password, mfa_code=None):\n"
    "        pass\n"
    "    def refresh_session(self, refresh_token):\n"
    "        pass\n"
)

SAMPLES["json_output"] = (
    '{"name":"myapp","version":"2.3.1",'
    '"scripts":{"build":"tsc","test":"vitest","lint":"eslint"},'
    '"dependencies":{"express":"^4.18.2","jsonwebtoken":"^9.0.2","redis":"^4.6.12"}}'
)

# ---------------------------------------------------------------------------
COMPRESSORS = {
    "pytest": compress_pytest, "cargo": compress_cargo, "go_test": compress_go_test,
    "jest": compress_jest, "eslint": compress_eslint, "git_status": compress_git_status,
    "git_diff": compress_git_diff, "git_log": compress_git_log, "grep": compress_grep,
    "npm": compress_npm, "docker": compress_docker, "kubectl": compress_kubectl,
    "aws": compress_aws, "json_output": compress_json_output, "make": compress_make,
    "pip": compress_pip, "log": compress_log, "cat": compress_cat,
    "generic": compress_generic,
}


def estimate_tokens(text: str) -> int:
    """4 chars ~= 1 token (cl100k_base approximation)."""
    return max(1, len(text) // 4)


def run():
    results = []
    total_orig = total_comp = 0
    for name in SAMPLES:
        if name not in COMPRESSORS:
            continue
        original = SAMPLES[name].strip()
        compressed = COMPRESSORS[name](original).strip()
        orig_tok = estimate_tokens(original)
        comp_tok = estimate_tokens(compressed)
        pct = round((1 - comp_tok / max(orig_tok, 1)) * 100, 1)
        total_orig += orig_tok
        total_comp += comp_tok
        results.append((name, pct, len(original), len(compressed), orig_tok, comp_tok))

    results.sort(key=lambda r: -r[1])

    print(f"{'Compressor':<16} {'Orig ch':>8} {'Comp ch':>8} {'Orig tk':>7} {'Comp tk':>7} {'Save %':>7}")
    print("-" * 63)
    for name, pct, oc, cc, ot, ct in results:
        print(f"{name:<16} {oc:>8} {cc:>8} {ot:>7} {ct:>7} {pct:>6.1f}%")

    avg = round((1 - total_comp / max(total_orig, 1)) * 100, 1)
    print("-" * 63)
    print(f"{'WEIGHTED AVG':<16} {'':>8} {'':>8} {total_orig:>7} {total_comp:>7} {avg:>6.1f}%")
    print(f"\n{len(results)} compressors tested. Total: {total_orig:,} -> {total_comp:,} tokens ({avg}% savings)")


if __name__ == "__main__":
    run()
