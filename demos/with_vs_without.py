"""graphsift: WITH vs WITHOUT — Full Comparison Across All Developer Scenarios.

Tests every major daily-dev command and shows:
  - Tokens WITHOUT graphsift (raw)
  - Tokens WITH graphsift (compressed)
  - % savings
  - What signals survive in each case
  - Projected monthly cost difference
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphsift import compress, estimate_tokens, __version__

SEP = "  " + "-" * 65

# ===========================================================================
# ALL DAILY-DEV SCENARIOS — real-world commands developers run every day
# ===========================================================================
SCENARIOS = {}

SCENARIOS["1. Bug Diagnosis"] = {
    "cmd": "pytest",
    "scenario": "Developer debugging 3 test failures",
    "raw": """============================= test session starts =============================
platform linux -- Python 3.12.0, pytest-8.0.0
rootdir: /home/ci/app
collected 47 items

tests/test_auth.py::test_login PASSED
tests/test_auth.py::test_login_fails PASSED
tests/test_db.py::test_connect FAILED
tests/test_db.py::test_query PASSED
tests/test_db.py::test_migrate FAILED
tests/test_db.py::test_rollback FAILED

=================================== FAILURES ===================================
_______________________________ test_connect ___________________________________

    def test_connect():
        db = get_test_db()
>       result = db.connect("postgresql://bad:url@localhost:9999/nonexistent")
E       psycopg2.OperationalError: could not connect to server: Connection refused
E           Is the server running on host "localhost" (127.0.0.1) and accepting
E           TCP/IP connections on port 9999?

tests/test_db.py:15: OperationalError
_______________________________ test_migrate __________________________________

    def test_migrate():
        db = get_test_db()
        db.execute("CREATE TABLE test (id INT)")
>       db.execute("ALTER TABLE test ADD COLUMN name VARCHAR(255)")
E       psycopg2.errors.DuplicateColumn: column "name" already exists in table "test"

tests/test_db.py:45: DuplicateColumn
_______________________________ test_rollback _________________________________

    def test_rollback():
        db = get_test_db()
        db.execute("BEGIN")
>       db.execute("INVALID SQL STATEMENT")
E       psycopg2.errors.SyntaxError: syntax error at or near "INVALID"

tests/test_db.py:62: SyntaxError
=========================== 3 failed, 4 passed in 0.35s ===========================
""",
}

SCENARIOS["2. Security Audit"] = {
    "cmd": "npm",
    "scenario": "Security team reviewing npm audit",
    "raw": """npm WARN EBADENGINE Unsupported engine {
npm WARN EBADENGINE   package: 'sharp@0.33.2',
npm WARN EBADENGINE   current: { node: 'v18.16.0' }
}
added 1423 packages in 15.2s

found 12 vulnerabilities (3 critical, 5 high, 4 moderate)

3 critical:
  CVE-2026-1234: Remote code execution in express <4.19.0
  CVE-2026-5678: Prototype pollution in lodash <=4.17.20
  CVE-2026-9012: Path traversal in serve-static <1.16.0

5 high:
  CVE-2026-3456: XSS in marked <4.3.0
  CVE-2026-7890: Command injection in shelljs <0.9.0
  CVE-2026-2345: CSRF in cookie-parser <1.4.7
  CVE-2026-6789: DoS in body-parser <1.20.3
  CVE-2026-0123: Insecure randomness in uuid <9.0.0

4 moderate:
  CVE-2026-4567: Info exposure in morgan <1.10.0
  CVE-2026-2346: Timing attack in bcrypt <5.1.1
""",
}

SCENARIOS["3. Code Review"] = {
    "cmd": "git_diff",
    "scenario": "Reviewing a PR with 2 file changes",
    "raw": """diff --git a/src/auth/manager.py b/src/auth/manager.py
index abc123..def456 100644
--- a/src/auth/manager.py
+++ b/src/auth/manager.py
@@ -42,6 +42,9 @@ def login(self, username, password):
         user = self.user_repo.find_by_username(username)
+        if not user:
+            raise UserNotFoundError(username)
         if not verify_password(password, user.password_hash):
             raise InvalidPasswordError()
+        return AuthSession(user.id, user.role)

@@ -89,3 +92,8 @@ def logout(self, session_id):
         self.sessions.pop(session_id, None)
+        audit.log_event("logout", session_id)
+
+    def refresh_session(self, session_id: str) -> AuthSession:
+        session = self.sessions.get(session_id)
+        if session is None:
+            raise SessionExpiredError(session_id)
+        session.refresh()
+        return session

diff --git a/tests/test_auth.py b/tests/test_auth.py
new file mode 100644
index 000000..abc123
--- /dev/null
+++ b/tests/test_auth.py
@@ -0,0 +1,24 @@
+import pytest
+from src.auth.manager import AuthManager, SessionExpiredError
+
+
+def test_login_success():
+    auth = AuthManager()
+    token = auth.login("admin", "correct-password")
+    assert token is not None
+
+
+def test_login_invalid_user():
+    auth = AuthManager()
+    with pytest.raises(UserNotFoundError):
+        auth.login("nonexistent", "password")
""",
}

SCENARIOS["4. CI/CD Pipeline"] = {
    "cmd": "go_test",
    "scenario": "CI build failing with test failures",
    "raw": """go build ./...
# my-project/src/api
src/api/handler.go:42:12: undefined: processRequest
src/api/handler.go:45:5: undefined: processRequest
FAIL    my-project/src/api [build failed]
ok      my-project/src/auth 0.312s
ok      my-project/src/db 0.245s
--- FAIL: TestProcessRequest (0.01s)
    handler_test.go:25: expected status 200, got 500
--- FAIL: TestMiddleware (0.02s)
    middleware_test.go:38: expected header X-Request-Id to be set
FAIL
FAIL    my-project/src/api 0.456s
ok      my-project/src/worker 0.189s
""",
}

SCENARIOS["5. Lint Report"] = {
    "cmd": "eslint",
    "scenario": "Running linter before commit",
    "raw": """C:\\project\\src\\app.tsx
  5:12  error    'unusedVar' is assigned a value but never used  @typescript-eslint/no-unused-vars
  8:3   warning  Unexpected console statement                    no-console
 12:20  error    Missing return type on function                 @typescript-eslint/explicit-function-return-type
 15:5   warning  'x' is never reassigned, use 'const'            prefer-const
 18:10  error    React Hook useMemo has a missing dependency     react-hooks/exhaustive-deps
 22:1   error    Using 'any' type is not allowed                 @typescript-eslint/no-explicit-any

C:\\project\\src\\utils.ts
  3:1   error    'deprecated_fn' is deprecated                   deprecation/deprecated

C:\\project\\src\\components\\Button.tsx
  4:5   error    Missing accessibility label                     jsx-a11y/control-has-associated-label

7 errors, 4 warnings
""",
}

SCENARIOS["6. git log"] = {
    "cmd": "git_log",
    "scenario": "Reviewing commit history for release notes",
    "raw": """commit 7a3f9b2e1c5d8f4a6b0c3e2d1f4a5b6c7d8e9f0a
Author: Alice <alice@example.com>
Date:   Mon Jul 14 14:23:45 2026 -0400

    fix: resolve race condition in session manager

commit 9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c
Author: Bob <bob@example.com>
Date:   Mon Jul 14 11:15:22 2026 -0400

    feat: add rate limiting middleware

commit 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b
Author: Charlie <charlie@example.com>
Date:   Sun Jul 13 18:30:10 2026 -0400

    refactor: extract auth validation to shared module

commit 0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e
Author: Alice <alice@example.com>
Date:   Sun Jul 13 10:05:33 2026 -0400

    tests: add integration tests for login flow

commit 5e4d3c2b1a0f9e8d7c6b5a4f3e2d1c0b9a8f7e6d
Author: Bob <bob@example.com>
Date:   Sat Jul 12 16:45:00 2026 -0400

    docs: update API reference for v2 endpoints
""",
}

SCENARIOS["7. git status"] = {
    "cmd": "git_status",
    "scenario": "Checking working tree before commit",
    "raw": """On branch feature/rate-limiting
Your branch is ahead of 'origin/feature/rate-limiting' by 3 commits.

Changes to be committed:
        modified:   src/middleware/rate_limit.py
        modified:   src/config.py

Changes not staged for commit:
        modified:   src/api/routes.py
        modified:   tests/test_rate_limit.py

Untracked files:
        docs/rate-limiting.md
        scripts/bench_rate_limit.py
""",
}

SCENARIOS["8. Log Analysis"] = {
    "cmd": "log",
    "scenario": "Root-causing production incident from logs",
    "raw": """2026-07-15 09:15:22,345 [INFO] Server started on port 8080
2026-07-15 09:15:23,890 [WARNING] Connection pool at 80% capacity (40/50 connections)
2026-07-15 09:15:24,012 [INFO] Health check OK
2026-07-15 09:15:24,234 [ERROR] Cache connection failed: Connection refused to redis:6379
2026-07-15 09:15:24,345 [CRITICAL] Service degraded -- cache unavailable, falling back to direct DB
2026-07-15 09:15:24,567 [WARNING] Rate limit approaching: 950/1000 requests this minute
2026-07-15 09:15:25,890 [INFO] Request served: GET /api/users 200 45ms
2026-07-15 09:15:26,123 [ERROR] Request timeout on /api/users/batch: deadline exceeded
2026-07-15 09:15:26,345 [WARNING] Retry attempt 1/3 for /api/users/batch
2026-07-15 09:15:27,890 [INFO] Request served: POST /api/orders 201 120ms
2026-07-15 09:15:28,456 [WARNING] Memory usage at 85% of limit
2026-07-15 09:15:29,000 [ERROR] Database pool exhausted: no available connections
""",
}

SCENARIOS["9. Build Output"] = {
    "cmd": "make",
    "scenario": "Developer building C project",
    "raw": """gcc -Wall -O2 -c src/main.c -o build/main.o
gcc -Wall -O2 -c src/utils.c -o build/utils.o
gcc -Wall -O2 -c src/parser.c -o build/parser.o
gcc -Wall -O2 -c src/config.c -o build/config.o
gcc build/main.o build/utils.o build/parser.o build/config.o -o build/app -lm
make: Nothing to be done for 'all'.
""",
}

SCENARIOS["10. Rust Build"] = {
    "cmd": "cargo",
    "scenario": "Rust developer compiling with errors",
    "raw": """   Compiling serde v1.0.200
   Compiling serde_json v1.0.120
   Compiling clap v4.5.8
   Compiling tokio v1.38.0
   Compiling reqwest v0.12.4
   Compiling my-crate v0.1.0
error[E0308]: mismatched types
  --> src/main.rs:42:20
   |
42 |     let result: String = compute_value();
   |                    ^^^^^^ expected `String`, found `i32`

warning: unused import `std::collections::HashMap`
 --> src/utils.rs:1:5
  |
1 | use std::collections::HashMap;

error: could not compile `my-crate` (bin "my-crate") due to 1 previous error
""",
}

SCENARIOS["11. Docker ps"] = {
    "cmd": "docker",
    "scenario": "Developer checking running containers",
    "raw": """CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                    NAMES
a1b2c3d4e5f6   nginx:latest   "/docker-entrypoint."   10 minutes ago   Up 10 minutes   0.0.0.0:80->80/tcp      web-prod
b2c3d4e5f6a7   redis:7        "docker-entrypoint."   15 minutes ago   Up 15 minutes   0.0.0.0:6379->6379/tcp  redis-cache
c3d4e5f6a7b8   postgres:15    "docker-entrypoint."   20 minutes ago   Up 20 minutes   0.0.0.0:5432->5432/tcp  pg-main
d4e5f6a7b8c9   mysql:8        "docker-entrypoint."   1 hour ago       Up 1 hour       0.0.0.0:3306->3306/tcp  mysql-backup
e5f6a7b8c9d0   mongo:6        "docker-entrypoint."   2 hours ago      Up 2 hours      0.0.0.0:27017->27017/tcp mongo-db
""",
}

SCENARIOS["12. Kubectl"] = {
    "cmd": "kubectl",
    "scenario": "SRE checking pod status",
    "raw": """NAME                                READY   STATUS    RESTARTS   AGE
pod/backend-api-7d5f8c8b4f-abcde    1/1     Running   0          5d
pod/frontend-6d9f7c8b4f-fghij       1/1     Running   1          3d
pod/cache-8c5f6d8b4f-klmno          1/1     Running   0          7d
pod/worker-batch-9d3f2c8b4f-pqrst   0/1     CrashLoopBackOff   3    1h
pod/db-migrate-4f7e6d8b4f-uvwxy     0/1     Error              2    30m
NAME                                TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
service/backend                     ClusterIP      10.100.200.1     <none>        8080/TCP       5d
service/frontend                    LoadBalancer   10.100.200.2     203.0.113.10  80:32000/TCP   3d
""",
}

SCENARIOS["13. Terraform Plan"] = {
    "cmd": "terraform",
    "scenario": "Infra engineer reviewing plan",
    "raw": """Terraform will perform the following actions:

  # aws_instance.web_server will be updated in-place
  ~ resource "aws_instance" "web_server" {
        id                           = "i-0abcd1234efgh5678"
      ~ instance_type                = "t3.medium" -> "t3.large"
        tags                         = {}
    }

  # aws_s3_bucket.logs will be destroyed
  - resource "aws_s3_bucket" "logs" {
      - bucket                      = "app-logs-2024"
    }

  # aws_lambda_function.processor will be created
  + resource "aws_lambda_function" "processor" {
      + function_name               = "data-processor-v2"
      + handler                     = "index.handler"
      + runtime                     = "nodejs18.x"
    }

Plan: 1 to add, 1 to change, 1 to destroy.
""",
}

SCENARIOS["14. Go Test PASS"] = {
    "cmd": "go_test",
    "scenario": "All tests passing, checking coverage",
    "raw": """ok  	my-project/src/auth	0.245s
ok  	my-project/src/db	0.312s
ok  	my-project/src/api	0.189s
?   	my-project/src/cmd	[no test files]
ok  	my-project/src/worker	0.156s
ok  	my-project/src/cache	0.423s
""",
}

SCENARIOS["15. pip install"] = {
    "cmd": "pip",
    "scenario": "Dev setting up project dependencies",
    "raw": """Requirement already satisfied: pydantic in c:\\python312\\lib\\site-packages
Collecting rich>=12.0
  Downloading rich-13.7.0-py3-none-any.whl (239 kB)
     -- 100% | 239 kB 2.1 MB/s
Collecting pyyaml>=6.0
  Downloading pyyaml-6.0.2-cp312-cp312-win_amd64.whl (150 kB)
     -- 100% | 150 kB 3.0 MB/s
Installing collected packages: pyyaml, rich
Successfully installed pyyaml-6.0.2 rich-13.7.0
""",
}

# ===========================================================================
# RUN ALL TESTS
# ===========================================================================
print()
print(f"  {'='*70}")
print(f"  graphsift v{__version__}: WITH vs WITHOUT Comparison")
print(f"  {'='*70}")
print(f"  Testing {len(SCENARIOS)} real-world developer scenarios")
print()

all_raw = 0
all_comp = 0

for sid in sorted(SCENARIOS):
    s = SCENARIOS[sid]
    raw_text = s["raw"].strip()
    cmd = s["cmd"]
    scenario = s["scenario"]

    raw_tok = estimate_tokens(raw_text)
    compressed = compress(raw_text, command=cmd)
    comp_tok = estimate_tokens(compressed)
    pct = (1 - comp_tok / raw_tok) * 100

    all_raw += raw_tok
    all_comp += comp_tok

    print(f"  [{sid}] {scenario}")
    print(f"        Command: {cmd}")
    print(f"        WITHOUT: {raw_tok:>5} tokens")
    print(f"        WITH:    {comp_tok:>5} tokens")
    print(f"        SAVED:   {pct:>5.1f}%")
    print(f"        Without: {raw_text[:60]!r:.60}...")
    print(f"        With:    {compressed[:60]!r:.60}...")
    print()

# ===========================================================================
# SUMMARY TOTALS
# ===========================================================================
print(f"  {'='*70}")
print(f"  GRAND TOTALS — All {len(SCENARIOS)} Scenarios")
print(f"  {'='*70}")
print(f"  WITHOUT graphsift (raw):  {all_raw:>6} tokens")
print(f"  WITH graphsift:           {all_comp:>6} tokens")
print(f"  TOTAL SAVED:              {all_raw - all_comp:>6} tokens ({(1-all_comp/all_raw)*100:.1f}%)")
print()
print(f"  At Claude Opus pricing ($15/M input tokens):")
print(f"    WITHOUT: ${all_raw * 15 / 1_000_000:.4f} per run")
print(f"    WITH:    ${all_comp * 15 / 1_000_000:.4f} per run")
print(f"    SAVINGS: ${(all_raw - all_comp) * 15 / 1_000_000:.4f} per run")
print()
print(f"  At GPT-4o pricing ($10/M input tokens):")
print(f"    WITHOUT: ${all_raw * 10 / 1_000_000:.4f} per run")
print(f"    WITH:    ${all_comp * 10 / 1_000_000:.4f} per run")
print(f"    SAVINGS: ${(all_raw - all_comp) * 10 / 1_000_000:.4f} per run")
print()

# Monthly projections
runs_per_day = 10
work_days = 22
monthly_runs = runs_per_day * work_days
print(f"  Monthly Projection ({runs_per_day} runs/day × {work_days} days):")
print(f"    WITHOUT: {all_raw * monthly_runs:,} tokens = ${all_raw * monthly_runs * 15 / 1_000_000:.2f}/mo (Claude)")
print(f"    WITH:    {all_comp * monthly_runs:,} tokens = ${all_comp * monthly_runs * 15 / 1_000_000:.2f}/mo (Claude)")
print(f"    SAVINGS: ${(all_raw - all_comp) * monthly_runs * 15 / 1_000_000:.2f}/mo")

print()
print(f"  {'='*70}")
print(f"  SIDEBAR: What graphsift DOESN'T cost you")
print(f"  {'='*70}")
print(f"  - Zero API calls from library code (no telemetry)")
print(f"  - Zero data exfiltration (all processing local)")
print(f"  - Zero accounts or signup required")
print(f"  - Zero npm/Docker dependencies")
print(f"  - Compression latency: <1ms per command")
print(f"  - Security layer: PathValidator + CommandSanitizer + DataScrubber")
print()
