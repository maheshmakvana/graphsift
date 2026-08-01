"""Persistent TCP daemon server for graphsift.

A standalone process that stays alive across commands and executes Python
code sent by the native launcher (or the hook / CLI) over a localhost TCP
socket. Unlike the legacy per-process daemon, this server survives its
parent process, so module imports, ``sys.path`` and the result cache
persist for the whole session — that is what makes repeat commands fast.

Security model
--------------
- Binds ``127.0.0.1`` only (loopback) — never exposed to the network.
- Requires a per-start random token on every request. The token and port
  are written to ``~/.graphsift/daemon.json`` (created with user-only
  permissions) so the launcher and hook can authenticate.
- Request size is capped; connection/read timeouts are enforced.
- Code is executed with normal Python builtins (a documented change from
  the legacy restricted-exec daemon). The daemon is not exposed beyond
  loopback and requires the token, which is the same trust model as a
  local REPL server. Set ``GRAPHSIFT_NO_DAEMON=1`` to refuse to run.

Protocol (one JSON object per line over TCP)
--------------------------------------------
Request::

    {"token": str, "cmd": "exec"|"sleep"|"cache_clear"|"cache_stats",
     "code"?: str, "cwd"?: str, "path"?: str, "duration"?: float}

Response::

    {"ok": bool, "stdout": str, "stderr": str, "exit_code": int,
     "cached": bool, "detail"?: str}

Run directly::

    python -m graphsift.daemon_server
"""

from __future__ import annotations

import io
import json
import os
import secrets
import socket
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path

_DAEMON_DIR = Path.home() / ".graphsift"
_HOST = "127.0.0.1"
_REQUEST_MAX_BYTES = 1_000_000  # 1 MB per request line
_SOCKET_TIMEOUT = 30.0          # seconds per read
_IDLE_TTL = float(os.environ.get("GRAPHSIFT_DAEMON_IDLE_TTL", "7200"))  # 2h
_CACHE_TTL = 300.0
_CACHE_MAX = 256


def _info_path() -> Path:
    """Path to the daemon info file (overridable for tests)."""
    env = os.environ.get("GRAPHSIFT_DAEMON_FILE")
    if env:
        return Path(env)
    return _DAEMON_DIR / "daemon.json"

# Result cache: {sha256(code|cwd): {"result": ..., "ts": ...}}
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.RLock()
_LAST_ACTIVITY = time.monotonic()


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------


def _exec_code(code: str, cwd: str = "", path: str = "", module: str = "") -> dict:
    """Execute *code* (or a script file / module) and return a result dict.

    Runs with normal Python builtins, captures stdout/stderr, chdir's into
    *cwd*, and caches successful results (5-minute TTL).
    """
    cache_key = _cache_key(code, cwd, path, module)
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached and (time.monotonic() - cached["ts"]) < _CACHE_TTL:
            result = dict(cached["result"])
            result["cached"] = True
            return result

    old_cwd = os.getcwd()
    old_path = list(sys.path)
    try:
        if cwd:
            try:
                os.chdir(cwd)
            except OSError:
                pass
            sys.path.insert(0, cwd)

        if path:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                code = fh.read()

        if module:
            import runpy

            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                try:
                    runpy.run_module(module, run_name="__main__")
                    result = _result(
                        True, sys.stdout.getvalue(), sys.stderr.getvalue(), 0
                    )
                except SystemExit as exc:
                    code_ = int(exc.code) if isinstance(exc.code, int) else 0
                    result = _result(
                        True,
                        sys.stdout.getvalue(),
                        sys.stderr.getvalue(),
                        code_,
                    )
            except Exception:  # noqa: BLE001
                result = _result(
                    False,
                    sys.stdout.getvalue(),
                    sys.stderr.getvalue() + traceback.format_exc(),
                    1,
                )
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        else:
            # exec path — covers both --code and --script (script content
            # was loaded into *code* above).
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                try:
                    exec(code, {"__name__": "__main__"}, {})
                    result = _result(
                        True, sys.stdout.getvalue(), sys.stderr.getvalue(), 0
                    )
                except SystemExit as exc:
                    code_ = int(exc.code) if isinstance(exc.code, int) else 0
                    result = _result(
                        True,
                        sys.stdout.getvalue(),
                        sys.stderr.getvalue(),
                        code_,
                    )
            except Exception:  # noqa: BLE001
                result = _result(
                    False,
                    sys.stdout.getvalue(),
                    sys.stderr.getvalue() + traceback.format_exc(),
                    1,
                )
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        if result.get("ok") and result.get("stdout"):
            with _CACHE_LOCK:
                _CACHE[cache_key] = {"result": result, "ts": time.monotonic()}
                while len(_CACHE) > _CACHE_MAX:
                    try:
                        oldest = min(_CACHE.keys(), key=lambda k: _CACHE[k]["ts"])
                        del _CACHE[oldest]
                    except (ValueError, KeyError):
                        break
        return result
    finally:
        try:
            os.chdir(old_cwd)
        except OSError:
            pass
        sys.path = old_path


def _cache_key(code: str, cwd: str, path: str, module: str) -> str:
    import hashlib

    return hashlib.sha256(f"{path}|{module}|{code}|{cwd}".encode()).hexdigest()


def _result(ok: bool, stdout: str, stderr: str, exit_code: int) -> dict:
    return {
        "ok": bool(ok),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": int(exit_code),
        "cached": False,
    }


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------


def _handle_request(req: dict, server_token: str) -> dict:
    """Dispatch one authenticated request. Never raises."""
    if req.get("token") != server_token:
        return _result(False, "", "unauthorized (bad token)", 1)
    cmd = req.get("cmd", "")
    if cmd == "exec":
        return _exec_code(
            req.get("code", ""),
            req.get("cwd", ""),
            req.get("path", ""),
            req.get("module", ""),
        )
    if cmd == "sleep":
        duration = min(float(req.get("duration", 1)), 30.0)
        time.sleep(duration)
        return _result(True, "", "", 0)
    if cmd == "cache_clear":
        with _CACHE_LOCK:
            _CACHE.clear()
        return _result(True, "cache cleared", "", 0)
    if cmd == "cache_stats":
        with _CACHE_LOCK:
            n = len(_CACHE)
        return _result(True, f"{n} entries", "", 0)
    if cmd == "ping":
        return _result(True, "pong", "", 0)
    return _result(False, "", f"unknown command: {cmd}", 1)


class _Handler(socketserver.BaseRequestHandler):
    """One request per connection; validate token, dispatch, respond."""

    def handle(self) -> None:
        global _LAST_ACTIVITY
        conn = self.request
        conn.settimeout(_SOCKET_TIMEOUT)
        try:
            raw = conn.recv(_REQUEST_MAX_BYTES)
        except Exception:
            return
        if not raw:
            return
        _LAST_ACTIVITY = time.monotonic()
        try:
            req = json.loads(raw.decode("utf-8", errors="replace").strip())
        except (json.JSONDecodeError, ValueError):
            return
        resp = _handle_request(req, self.server.token)
        try:
            conn.sendall(json.dumps(resp).encode("utf-8") + b"\n")
        except Exception:
            pass
        # Graceful shutdown on demand (used by `daemon stop`).
        if req.get("cmd") == "shutdown" and resp.get("ok"):
            threading.Thread(target=self.server.shutdown, daemon=True).start()


class _Server(socketserver.ThreadingTCPServer):
    """Threaded TCP server bound to loopback with a dynamic port."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        self.token = secrets.token_hex(32)
        super().__init__((_HOST, 0), _Handler)
        self._port = self.server_address[1]
        self._pid = os.getpid()


def _graphsift_version() -> str:
    """The installed graphsift version (without importing the heavy package)."""
    try:
        from graphsift import _version  # noqa: PLC0415

        return getattr(_version, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_info_file(port: int, token: str, pid: int) -> None:
    info_file = _info_path()
    info_file.parent.mkdir(parents=True, exist_ok=True)
    import datetime as _dt

    payload = {
        "port": int(port),
        "token": token,
        "pid": int(pid),
        "python": sys.executable,
        "version": _graphsift_version(),
        "started": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    tmp = info_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, info_file)


def serve(forever: bool = True) -> int:
    """Start the server, write the info file, and serve until idle-shutdown.

    Returns the process exit code.
    """
    if os.environ.get("GRAPHSIFT_NO_DAEMON"):
        print("graphsift daemon refused: GRAPHSIFT_NO_DAEMON=1", file=sys.stderr)
        return 1

    server = _Server()
    _write_info_file(server._port, server.token, server._pid)
    _LAST_ACTIVITY = time.monotonic()

    if not forever:
        # Used only for startup self-test.
        return 0

    # Idle watchdog: graceful shutdown after _IDLE_TTL seconds of no requests.
    def _watchdog() -> None:
        while True:
            time.sleep(60)
            if (time.monotonic() - _LAST_ACTIVITY) > _IDLE_TTL:
                try:
                    server.shutdown()
                except Exception:
                    pass
                return

    if _IDLE_TTL > 0:
        threading.Thread(target=_watchdog, daemon=True).start()

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        # Best-effort cleanup of the info file — but only if it still
        # points at THIS server (a newer server may have overwritten it).
        try:
            info_file = _info_path()
            if info_file.exists():
                data = json.loads(info_file.read_text(encoding="utf-8"))
                if int(data.get("port", -1)) == server._port:
                    info_file.unlink()
        except (OSError, ValueError, TypeError):
            pass
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="graphsift persistent daemon server")
    parser.add_argument("--no-idle-ttl", action="store_true",
                        help="disable idle auto-shutdown")
    args = parser.parse_args()
    global _IDLE_TTL
    if args.no_idle_ttl:
        _IDLE_TTL = 0
    return serve()


if __name__ == "__main__":
    sys.exit(main())
