"""Persistent daemon client for graphsift.

The daemon worker now lives in :mod:`graphsift.daemon_server` — a detached
TCP server bound to ``127.0.0.1`` that **survives its parent process**. This
module is the *client*: it spawns the server on demand, locates it via
``~/.graphsift/daemon.json`` (port + auth token), and talks to it over a
short-lived TCP connection per request.

Public API is unchanged from the legacy per-process daemon::

    start() / stop() / status()
    exec_code(code, cwd="") / sleep(duration) / cache_clear() / cache_stats()

Features preserved from the original design:
  - **Result caching**: identical commands return cached results (server-side).
  - **Sleep handling**: ``sleep N`` handled server-side (no Python exec).
  - **Thread safety**: all public functions hold ``_LOCK``.
  - **Env switches**: ``GRAPHSIFT_NO_DAEMON=1`` refuses to start the server.
  - **CLI**: ``python -m graphsift.daemon start|stop|status|exec|sleep|
    cache-clear|cache-stats``.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DAEMON_DIR = Path.home() / ".graphsift"
_HOST = "127.0.0.1"
_CONNECT_TIMEOUT = float(os.environ.get("GRAPHSIFT_DAEMON_CONNECT_TIMEOUT", "10"))
_READ_TIMEOUT = float(os.environ.get("GRAPHSIFT_DAEMON_TIMEOUT", "30"))
_LOCK = threading.RLock()

_NO_DAEMON_ERROR = {"status": "failed", "error": "GRAPHSIFT_NO_DAEMON=1 is set"}


def _current_version() -> str:
    """The installed graphsift version."""
    try:
        from graphsift._version import __version__ as _v  # noqa: PLC0415

        return str(_v)
    except Exception:  # noqa: BLE001
        return "unknown"


def _version_ok(info: dict | None) -> bool:
    """True if the recorded server version matches the installed version."""
    if not info:
        return False
    recorded = info.get("version")
    if recorded is None:
        return False  # pre-version info file (old server) — treat as stale
    return recorded == _current_version()


# ---------------------------------------------------------------------------
# Info-file helpers
# ---------------------------------------------------------------------------


def _info_path() -> Path:
    """Path to the daemon info file (overridable for tests)."""
    env = os.environ.get("GRAPHSIFT_DAEMON_FILE")
    if env:
        return Path(env)
    return _DAEMON_DIR / "daemon.json"


def _read_info() -> dict | None:
    """Read the daemon info file (port + token + pid) if present."""
    try:
        info_file = _info_path()
        if info_file.exists():
            data = json.loads(info_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("port"):
                return data
    except (OSError, ValueError, TypeError):
        pass
    return None


def _cleanup_info_file() -> None:
    try:
        info_file = _info_path()
        if info_file.exists():
            info_file.unlink()
    except OSError:
        pass


def _ping(port: int | None, timeout: float = 1.0) -> bool:
    """Return True if a server responds on *port* (raw connect probe).

    Retries briefly so a single transient connect failure never causes an
    unnecessary respawn (which would churn the info file and bounce
    requests between servers).
    """
    if not port:
        return False
    for attempt in range(3):
        try:
            with socket.create_connection((_HOST, int(port)), timeout=timeout):
                return True
        except OSError:
            if attempt < 2:
                time.sleep(0.1)
    return False


def _acquire_start_lock(timeout: float = 5.0):
    """Serialize server spawns across processes (O_EXCL lock file)."""
    lock_path = _info_path().with_suffix(".start.lock")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            time.sleep(0.05)
        except OSError:
            return None  # cannot create the lock — proceed anyway
    return None


def _release_start_lock(lock_path) -> None:
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def start() -> dict:
    """Spawn the persistent server (if not already reachable) and return status.

    Uses a cross-process lock so concurrent ``start()`` calls (auto-config
    on import, SessionStart hook, CLI) spawn at most one server.

    Returns:
        dict with keys: status ("started"/"already_running"/"failed"),
        pid, port, error.
    """
    with _LOCK:
        info = _read_info()
        if info and _ping(info.get("port")) and _version_ok(info):
            _sweep_orphans()
            return {
                "status": "already_running",
                "pid": info.get("pid"),
                "port": info.get("port"),
            }

        if os.environ.get("GRAPHSIFT_NO_DAEMON"):
            return dict(_NO_DAEMON_ERROR)

        lock = _acquire_start_lock()
        try:
            # Re-check after acquiring the lock — another process may have
            # finished starting the server while we waited.
            info = _read_info()
            if info and _ping(info.get("port")) and _version_ok(info):
                return {
                    "status": "already_running",
                    "pid": info.get("pid"),
                    "port": info.get("port"),
                }
            if os.environ.get("GRAPHSIFT_NO_DAEMON"):
                return dict(_NO_DAEMON_ERROR)

            # A stale-version (or orphaned) server is responding — shut it down.
            if info and _ping(info.get("port")):
                _shutdown_server(info)

            # Clean up leftover daemon processes from older versions / crashes.
            _cleanup_orphans()

            try:
                kwargs: dict = {}
                if os.name == "nt":
                    kwargs["creationflags"] = (
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS
                    )
                else:
                    kwargs["start_new_session"] = True
                # The server must NOT run graphsift's import-time auto-config:
                # that would call daemon.start() recursively and spawn a
                # cascade of servers until one writes the info file.
                env = os.environ.copy()
                env["GRAPHSIFT_NO_AUTOCONFIGURE"] = "1"
                subprocess.Popen(
                    [sys.executable, "-m", "graphsift.daemon_server"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    env=env,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("daemon start failed: %s", exc)
                return {"status": "failed", "error": str(exc)}

            deadline = time.monotonic() + _CONNECT_TIMEOUT
            while time.monotonic() < deadline:
                info = _read_info()
                if info and _ping(info.get("port")):
                    return {
                        "status": "started",
                        "pid": info.get("pid"),
                        "port": info.get("port"),
                    }
                time.sleep(0.05)
            return {
                "status": "failed",
                "error": "daemon did not become reachable",
            }
        finally:
            _release_start_lock(lock)


def status() -> dict:
    """Check if the persistent server is running (never spawns)."""
    info = _read_info()
    if info and _ping(info.get("port"), timeout=1.0):
        return {"status": "running", "pid": info.get("pid"), "port": info.get("port")}
    return {"status": "stopped"}


def stop() -> dict:
    """Gracefully shut down the server and remove the info file."""
    with _LOCK:
        info = _read_info()
        _shutdown_server(info)
        _cleanup_orphans()
        _cleanup_info_file()
        return {"ok": True}


def _shutdown_server(info: dict | None) -> None:
    """Ask the server described by *info* to shut down, then wait briefly."""
    if not info or not _ping(info.get("port")):
        return
    try:
        _request(
            {"token": info.get("token", ""), "cmd": "shutdown"},
            timeout=5.0,
        )
    except Exception:  # noqa: BLE001
        pass
    # Give it a moment to remove the info file / close.
    for _ in range(20):
        if not _ping(info.get("port"), timeout=0.3):
            break
        time.sleep(0.1)


_last_sweep = 0.0


def _sweep_orphans() -> None:
    """Throttled orphan sweep on the healthy-server fast path."""
    global _last_sweep
    now = time.monotonic()
    if now - _last_sweep < 60.0:
        return
    _last_sweep = now
    _cleanup_orphans()


def _cleanup_orphans() -> None:
    """Kill leftover graphsift daemon-server processes.

    Handles automatic cleanup when the user upgrades: an old-version server
    (or one orphaned by a crash) that is no longer the registered server is
    terminated so only one daemon exists.
    """
    current_pid = None
    info = _read_info()
    if info:
        current_pid = info.get("pid")

    try:
        if os.name == "nt":
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                    "Where-Object { $_.CommandLine -like '*graphsift.daemon_server*' } | "
                    "ForEach-Object { $_.ProcessId }",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            for token in out.stdout.split():
                if not token.strip().isdigit():
                    continue
                pid = int(token)
                if pid == current_pid:
                    continue
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:  # noqa: BLE001
                    pass
        else:
            out = subprocess.run(
                ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10
            )
            for line in out.stdout.splitlines():
                if "graphsift.daemon_server" not in line:
                    continue
                try:
                    pid = int(line.split()[0])
                except ValueError:
                    continue
                if pid == current_pid:
                    continue
                try:
                    os.kill(pid, 9)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# RPC
# ---------------------------------------------------------------------------


def _request(payload: dict, timeout: float | None = None) -> dict:
    """Send one request to the server and return its JSON response.

    Never raises — returns an error dict on any transport failure.
    """
    info = _read_info()
    if not info:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "daemon not running (no info file)",
            "exit_code": 1,
        }
    try:
        with socket.create_connection(
            (_HOST, int(info["port"])),
            timeout=min(5.0, timeout or _CONNECT_TIMEOUT),
        ) as sock:
            sock.settimeout(timeout or _READ_TIMEOUT)
            sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            if not buf:
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": "no response from daemon",
                    "exit_code": 1,
                }
            return json.loads(buf.decode("utf-8", errors="replace").strip())
    except (OSError, socket.timeout, ValueError) as exc:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"daemon unreachable: {exc}",
            "exit_code": 1,
        }


def _ensure_running() -> bool:
    """Return True if a *current-version* server is reachable, else (re)start."""
    if status().get("status") == "running":
        info = _read_info()
        if _version_ok(info):
            return True
        # Running server is a stale version — start() will replace it.
    return start().get("status") in ("started", "already_running")


def _is_transport_error(resp: dict) -> bool:
    """True if *resp* is a stale/broken server result (restart-worthy).

    Includes client-side transport failures AND a token rejection — the
    latter means the server on the info-file port is a leftover from an
    older run whose token no longer matches the info file.
    """
    if not isinstance(resp, dict) or "ok" not in resp:
        return True
    marker = (resp.get("stderr") or "").lower()
    return (
        "daemon unreachable" in marker
        or "no response from daemon" in marker
        or "daemon returned invalid json" in marker
        or "unauthorized (bad token)" in marker
    )


def _rpc(payload: dict, timeout: float | None = None) -> dict:
    """Send one request, restarting the server on stale/broken connections.

    A server left over from an older graphsift version (or a crashed one)
    may accept the TCP connection but never reply. In that case we stop it,
    spawn a fresh server, and retry once.
    """
    resp = _request(payload, timeout)
    if _is_transport_error(resp):
        stop()
        start()
        resp = _request(payload, timeout)
    return resp


def _token() -> str:
    info = _read_info()
    return (info or {}).get("token", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def exec_code(
    code: str,
    cwd: str = "",
    path: str = "",
    module: str = "",
) -> dict:
    """Send *code* (or a script file / module) to the persistent server.

    Returns a dict with keys: ok, stdout, stderr, exit_code, cached.
    """
    with _LOCK:
        if not _ensure_running():
            return {
                "ok": False,
                "stdout": "",
                "stderr": "failed to start daemon",
                "exit_code": 1,
            }
        return _rpc({
            "token": _token(),
            "cmd": "exec",
            "code": code,
            "cwd": cwd,
            "path": path,
            "module": module,
        })


def sleep(duration: float = 1.0) -> dict:
    """Ask the server to sleep *duration* seconds (capped at 30)."""
    with _LOCK:
        if not _ensure_running():
            return {"ok": False, "stdout": "", "stderr": "failed to start daemon"}
        return _rpc(
            {"token": _token(), "cmd": "sleep", "duration": duration},
            timeout=max(30.0, min(duration, 30.0) + 5.0),
        )


def cache_clear() -> dict:
    """Clear the server-side result cache."""
    with _LOCK:
        if status().get("status") != "running":
            return {"ok": True}
        return _rpc({"token": _token(), "cmd": "cache_clear"})


def cache_stats() -> dict:
    """Return cache stats from the server."""
    with _LOCK:
        if status().get("status") != "running":
            return {"ok": True, "in_process": 0, "daemon": "stopped"}
        resp = _rpc({"token": _token(), "cmd": "cache_stats"})
        return {
            "ok": resp.get("ok", False),
            "in_process": 0,  # cache now lives entirely server-side
            "daemon": resp.get("stdout", "unknown"),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="graphsift persistent daemon")
    parser.add_argument(
        "action",
        choices=["start", "stop", "status", "exec", "sleep", "cache-clear", "cache-stats"],
    )
    parser.add_argument("code", nargs="?", default="", help="Code to execute")
    parser.add_argument("--cwd", default="", help="Working directory")
    parser.add_argument("--duration", type=float, default=1.0, help="Sleep duration")
    args = parser.parse_args()

    if args.action == "start":
        result = start()
    elif args.action == "stop":
        result = stop()
    elif args.action == "status":
        result = status()
    elif args.action == "exec":
        result = exec_code(args.code, args.cwd)
    elif args.action == "sleep":
        result = sleep(args.duration)
    elif args.action == "cache-clear":
        result = cache_clear()
    elif args.action == "cache-stats":
        result = cache_stats()
    else:  # pragma: no cover
        result = {"error": f"Unknown action: {args.action}"}

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
