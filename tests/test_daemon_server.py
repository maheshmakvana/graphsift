"""Tests for the persistent TCP daemon server and its client (daemon.py).

Covers the request handler directly (pure, no network), plus an end-to-end
round trip through the detached server. The daemon uses an isolated info
file (``GRAPHSIFT_DAEMON_FILE``) so parallel pytest workers never collide.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphsift.daemon_server import _handle_request, _Server  # noqa: E402


# ---------------------------------------------------------------------------
# Handler unit tests (pure — no network, no process spawn)
# ---------------------------------------------------------------------------


class TestHandleRequest:
    def _srv(self) -> _Server:
        return _Server()

    def test_exec(self):
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": srv.token, "cmd": "exec", "code": "print(40+2)", "cwd": ""},
                srv.token,
            )
            assert r["ok"] is True
            assert "42" in r["stdout"]
            assert r["exit_code"] == 0
        finally:
            srv.server_close()

    def test_bad_token_rejected(self):
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": "WRONG", "cmd": "exec", "code": "print(1)"},
                srv.token,
            )
            assert r["ok"] is False
            assert "unauthorized" in r["stderr"]
        finally:
            srv.server_close()

    def test_error_returns_traceback(self):
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": srv.token, "cmd": "exec", "code": "raise ValueError('boom')"},
                srv.token,
            )
            assert r["ok"] is False
            assert "ValueError" in r["stderr"]
            assert r["exit_code"] == 1
        finally:
            srv.server_close()

    def test_system_exit_code(self):
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": srv.token, "cmd": "exec", "code": "import sys; sys.exit(3)"},
                srv.token,
            )
            assert r["ok"] is True
            assert r["exit_code"] == 3
        finally:
            srv.server_close()

    def test_sleep(self):
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": srv.token, "cmd": "sleep", "duration": 0.01},
                srv.token,
            )
            assert r["ok"] is True
        finally:
            srv.server_close()

    def test_result_caching(self):
        srv = self._srv()
        try:
            payload = {"token": srv.token, "cmd": "exec", "code": "print('cache-me')", "cwd": ""}
            first = _handle_request(payload, srv.token)
            second = _handle_request(payload, srv.token)
            assert first["ok"] is True
            assert second["cached"] is True
        finally:
            srv.server_close()

    def test_script_path(self):
        srv = self._srv()
        try:
            tmp = Path(tempfile.gettempdir()) / "gs_script_test.py"
            tmp.write_text("print('script-ran')\n", encoding="utf-8")
            r = _handle_request(
                {"token": srv.token, "cmd": "exec", "path": str(tmp), "cwd": ""},
                srv.token,
            )
            assert r["ok"] is True
            assert "script-ran" in r["stdout"]
            tmp.unlink(missing_ok=True)
        finally:
            srv.server_close()

    def test_module(self):
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": srv.token, "cmd": "exec", "module": "this", "cwd": ""},
                srv.token,
            )
            assert r["ok"] is True
            assert "Beautiful" in r["stdout"]  # zen of python
        finally:
            srv.server_close()

    def test_unrunnable_module_reports_error(self):
        """`json` has no __main__ — must error like `python -m json`."""
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": srv.token, "cmd": "exec", "module": "json", "cwd": ""},
                srv.token,
            )
            assert r["ok"] is False
            assert "cannot be directly executed" in r["stderr"]
        finally:
            srv.server_close()

    def test_unknown_command(self):
        srv = self._srv()
        try:
            r = _handle_request(
                {"token": srv.token, "cmd": "frobnicate"},
                srv.token,
            )
            assert r["ok"] is False
        finally:
            srv.server_close()

    def test_ping(self):
        srv = self._srv()
        try:
            r = _handle_request({"token": srv.token, "cmd": "ping"}, srv.token)
            assert r["ok"] is True
            assert r["stdout"].strip() == "pong"
        finally:
            srv.server_close()


# ---------------------------------------------------------------------------
# End-to-end round trip through the detached server (daemon.py client)
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_daemon(monkeypatch, tmp_path):
    """Start a daemon on an isolated info file; stop it afterwards."""
    info_file = tmp_path / "daemon.json"
    monkeypatch.setenv("GRAPHSIFT_DAEMON_FILE", str(info_file))
    monkeypatch.setenv("GRAPHSIFT_DAEMON_IDLE_TTL", "0")

    from graphsift import daemon

    st = daemon.start()
    if st.get("status") == "failed":
        pytest.skip(f"daemon could not start: {st.get('error')}")

    yield daemon
    daemon.stop()


@pytest.mark.skipif(
    os.environ.get("GRAPHSIFT_NO_DAEMON") == "1",
    reason="daemon disabled via env",
)
class TestDaemonClientRoundTrip:
    def test_start_status_exec_cache_stop(self, isolated_daemon):
        daemon = isolated_daemon
        assert daemon.status().get("status") == "running"

        r = daemon.exec_code("print(6*7)", ".")
        assert r["ok"] is True
        assert "42" in r["stdout"]
        assert r["exit_code"] == 0

        r2 = daemon.exec_code("print(6*7)", ".")
        assert r2["cached"] is True

        r3 = daemon.exec_code("raise RuntimeError('x')", ".")
        assert r3["ok"] is False
        assert "RuntimeError" in r3["stderr"]

        assert daemon.sleep(0.05)["ok"] is True

        stats = daemon.cache_stats()
        assert stats["ok"] is True

    def test_info_file_written_and_cleaned(self, isolated_daemon):
        daemon = isolated_daemon
        info_file = Path(os.environ["GRAPHSIFT_DAEMON_FILE"])
        assert info_file.exists()
        daemon.stop()
        assert not info_file.exists() or daemon.status().get("status") == "stopped"


# ---------------------------------------------------------------------------
# Native launcher (Windows / csc only)
# ---------------------------------------------------------------------------


def _csc_available() -> bool:
    if sys.platform != "win32":
        return False
    for cand in (
        r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
        r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe",
    ):
        if Path(cand).exists():
            return True
    return False


@pytest.mark.skipif(
    not _csc_available(),
    reason="requires Windows .NET Framework csc.exe",
)
class TestNativeLauncher:
    def test_build_and_sleep(self, tmp_path, monkeypatch):
        from graphsift import launcher

        # Point the launcher at an isolated info file via env (inherited).
        info_file = tmp_path / "daemon.json"
        monkeypatch.setenv("GRAPHSIFT_DAEMON_FILE", str(info_file))

        exe = launcher.build_launcher(force=True)
        assert exe is not None, "csc build should succeed"
        assert Path(exe).exists()

        # --sleep needs no daemon at all.
        proc = subprocess.run(
            [exe, "--sleep", "0.01"],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0

    def test_full_round_trip(self, tmp_path, monkeypatch):
        from graphsift import daemon, launcher

        info_file = tmp_path / "daemon.json"
        monkeypatch.setenv("GRAPHSIFT_DAEMON_FILE", str(info_file))
        monkeypatch.setenv("GRAPHSIFT_DAEMON_IDLE_TTL", "0")

        exe = launcher.ensure_launcher()
        assert exe is not None

        st = daemon.start()
        assert st.get("status") in ("started", "already_running")

        # Write code to a temp file (as the hook would) and run the launcher.
        code_file = tmp_path / "code.py"
        code_file.write_text("print('launcher-42')\n", encoding="utf-8")
        proc = subprocess.run(
            [exe, "--codefile", str(code_file), "--cwd", str(tmp_path)],
            capture_output=True, text=True, timeout=30,
        )
        assert "launcher-42" in proc.stdout, proc.stdout + proc.stderr

        # Failure propagation.
        code_file.write_text("import sys; sys.exit(7)\n", encoding="utf-8")
        proc = subprocess.run(
            [exe, "--codefile", str(code_file), "--cwd", str(tmp_path)],
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 7, proc.stdout + proc.stderr

        daemon.stop()
