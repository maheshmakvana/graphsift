"""graphsift CLI/UI Demo — Simulates real CLI output piped through `graphsift compress`.

Shows before/after for each command type in a visual way.
All ASCII-safe for Windows terminals.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphsift import compress, estimate_tokens, __version__
from graphsift.compress import detect_type, COMPRESSORS


def fmt(v):
    return f"{v:.1f}%"

def run_demo():
    ver = __version__
    print()
    print("=" * 65)
    print(f"  graphsift CLI Compression Demo v{ver}")
    print("=" * 65)

    demo_cases = []

    pytest_text = """============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-8.0.0, pluggy-1.5.0
rootdir: C:\\project\\app
collected 47 items

tests/test_auth.py::test_login PASSED
tests/test_auth.py::test_login_fails PASSED
tests/test_auth.py::test_logout PASSED
tests/test_db.py::test_connect FAILED
tests/test_db.py::test_query PASSED
tests/test_db.py::test_disconnect FAILED

=================================== FAILURES ===================================
_________________________________ test_connect _________________________________

    def test_connect():
>       assert db.connect("bad_url") == True
E       assert False == True

tests/test_db.py:12: AssertionError
_______________________________ test_disconnect ________________________________

    def test_disconnect():
>       db.disconnect()
E       AttributeError: 'NoneType' object has no attribute 'disconnect'

tests/test_db.py:18: AttributeError
=========================== 2 failed, 4 passed in 0.35s ===========================
"""
    demo_cases.append(("pytest", pytest_text))

    git_text = """diff --git a/src/auth/manager.py b/src/auth/manager.py
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
"""
    demo_cases.append(("git_diff", git_text))

    grep_text = """src/auth/manager.py:12:from typing import Optional
src/auth/manager.py:15:class AuthManager:
src/auth/manager.py:42:    def login(self, username, password):
src/auth/manager.py:45:        user = self.user_repo.find_by_username(username)
src/auth/manager.py:46:        if not user:
src/auth/manager.py:47:            raise UserNotFoundError(username)
src/models/user.py:22:class UserNotFoundError(Exception):
src/models/user.py:25:def verify_password(plain, hashed):
src/utils/validators.py:1:import re
"""
    demo_cases.append(("grep", grep_text))

    npm_text = """npm ERR! code ERESOLVE
npm ERR! While resolving: my-app@1.0.0
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^17.0.0" from react-dom@17.0.2
npm ERR! Fix with: npm install --force
npm ERR! Found: react@18.2.0
npm ERR!   react@"^18.2.0" from my-app@1.0.0
added 1 package, removed 0 packages, audited 1423 packages in 3.2s
found 12 vulnerabilities (3 critical, 5 high, 4 moderate)
"""
    demo_cases.append(("npm", npm_text))

    eslint_text = """C:\\project\\src\\app.tsx
  5:12  error    'unusedVar' is assigned a value but never used
  8:3   warning  Unexpected console statement
 12:20  error    Missing return type on function
 15:5   warning  'x' is never reassigned, use 'const'
 18:10  error    React Hook has a missing dependency

C:\\project\\src\\utils.ts
  3:1   error    'deprecated_fn' is deprecated

4 problems (3 errors, 1 warning)
"""
    demo_cases.append(("eslint", eslint_text))

    kubectl_text = """NAME                                READY   STATUS    RESTARTS   AGE
pod/backend-api-7d5f8c8b4f-abcde    1/1     Running   0          5d
pod/frontend-6d9f7c8b4f-fghij       1/1     Running   1          3d
pod/cache-8c5f6d8b4f-klmno          1/1     Running   0          7d
pod/worker-batch-9d3f2c8b4f-pqrst   0/1     CrashLoopBackOff   3    1h
pod/db-migrate-4f7e6d8b4f-uvwxy     0/1     Error              2    30m
"""
    demo_cases.append(("kubectl", kubectl_text))

    docker_text = """CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                    NAMES
a1b2c3d4e5f6   nginx:latest   "/docker-entrypoint."   10 minutes ago   Up 10 minutes   0.0.0.0:80->80/tcp      web-prod
b2c3d4e5f6a7   redis:7        "docker-entrypoint."   15 minutes ago   Up 15 minutes   0.0.0.0:6379->6379/tcp  redis-cache
c3d4e5f6a7b8   postgres:15    "docker-entrypoint."   20 minutes ago   Up 20 minutes   0.0.0.0:5432->5432/tcp  pg-main
"""
    demo_cases.append(("docker", docker_text))

    log_text = """2026-07-15 09:15:22,345 [INFO] Starting application server...
2026-07-15 09:15:22,456 [DEBUG] Loading configuration from /etc/app/config.yaml
2026-07-15 09:15:22,567 [INFO] Connecting to database
2026-07-15 09:15:23,890 [WARNING] Connection pool at 80% capacity
2026-07-15 09:15:24,012 [INFO] Database connection successful
2026-07-15 09:15:24,234 [ERROR] Failed to initialize cache: Connection refused
2026-07-15 09:15:24,345 [CRITICAL] Service degraded -- cache unavailable
2026-07-15 09:15:24,567 [WARNING] Rate limit approaching: 950/1000 requests
2026-07-15 09:15:25,890 [INFO] Health check endpoint registered
2026-07-15 09:15:26,123 [ERROR] Request timeout on /api/users/batch
"""
    demo_cases.append(("log", log_text))

    total_raw = 0
    total_comp = 0
    results = []

    print(f"\n  {'='*60}")
    print(f"  {'Command':<12} {'Raw Tok':>7} {'->':>2} {'Comp Tok':>7} {'Saved':>7}  Bar")
    print(f"  {'-'*12} {'-'*7} {'-'*2} {'-'*7} {'-'*7}  {'-'*20}")

    for cmd, sample in demo_cases:
        compressed = compress(sample, command=cmd)
        rt = estimate_tokens(sample)
        ct = estimate_tokens(compressed)
        pct = (1 - ct / rt) * 100 if rt else 0
        total_raw += rt
        total_comp += ct
        results.append((cmd, rt, ct, pct))

        bar_len = int(pct / 8)
        bar = "#" * bar_len + "-" * (12 - bar_len)
        print(f"  {cmd:<12} {rt:>7} -> {ct:>7} {fmt(pct):>7}  [{bar}]")

    overall = (1 - total_comp / total_raw) * 100 if total_raw else 0
    print(f"  {'-'*12} {'-'*7} {'-'*2} {'-'*7} {'-'*7}  {'-'*20}")
    print(f"  {'TOTAL':<12} {total_raw:>7} -> {total_comp:>7} {fmt(overall):>7}")
    print()

    # Ultra mode comparison
    print(f"  --- Ultra Mode Comparison ---")
    for cmd, sample in demo_cases[:4]:
        normal = compress(sample, command=cmd)
        ultra = compress(sample, command=cmd, ultra=True)
        n_tok = estimate_tokens(normal)
        u_tok = estimate_tokens(ultra)
        extra = (1 - u_tok / n_tok) * 100
        print(f"  {cmd:<12}: normal={n_tok:>4} tok, ultra={u_tok:>4} tok (extra {fmt(extra)} savings)")

    print()
    print(f"  --- Auto-Detect Accuracy ---")
    correct_count = 0
    for cmd, sample in demo_cases:
        detected = detect_type(sample)
        correct = "OK" if detected == cmd else "MISDETECT"
        if detected == cmd:
            correct_count += 1
        print(f"  {cmd:<12}: detected as {detected:<10} [{correct}]")

    print()
    print(f"  --- Summary ---")
    print(f"  graphsift v{ver} -- {len(demo_cases)} command types demoed")
    print(f"  Total raw tokens:  {total_raw}")
    print(f"  Total compressed:  {total_comp} ({fmt(overall)} saved)")
    print(f"  Auto-detect:       {correct_count}/{len(demo_cases)} correct")
    print(f"  Compressors avail: {len(COMPRESSORS)}")
    print()
    print(f"  Usage: <command> | graphsift compress")
    print(f"         <command> | graphsift compress --ultra")
    print()

if __name__ == "__main__":
    run_demo()
