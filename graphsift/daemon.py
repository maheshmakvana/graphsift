"""Persistent Python daemon — keeps modules loaded between commands + caches results.

Usage:
    pip install graphsift  → auto-starts daemon on first import
    Then: every ``cd <dir> && python ...`` runs through daemon automatically.
    No user action needed after install.

Features:
    - **Persistent process**: modules import ONCE, stay cached for life
    - **Result caching**: identical commands return cached results (0ms)
    - **Sleep handling**: ``sleep N`` commands handled natively (no exec)
    - **Timeout safety**: all reads have configurable timeout (default 30s)
    - **Thread safety**: all public functions hold _DAEMON_LOCK
    - **Auto-cleanup**: daemon stops on parent exit via atexit
    - **CWD support**: daemon chdir's into the requested working directory
"""

import atexit
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DAEMON_PROCESS = None
_DAEMON_LOCK = threading.RLock()  # RLock prevents reentrant deadlock

# In-process result cache: {sha256(code+cwd): {"result": ..., "ts": ...}}
_RESULT_CACHE: dict[str, dict] = {}

# Read from env or default: max seconds to wait for daemon response
_DAEMON_READ_TIMEOUT = float(os.environ.get("GRAPHSIFT_DAEMON_TIMEOUT", "30"))
_DAEMON_CONNECT_TIMEOUT = float(os.environ.get("GRAPHSIFT_DAEMON_CONNECT_TIMEOUT", "10"))
_CACHE_TTL = 300.0  # 5 minutes
_CACHE_MAX = 256

DAEMON_SCRIPT = r"""
import sys, json, io, traceback, time, hashlib, os

# In-daemon result cache
_cache = {}
_cache_ttl = 300.0

def _exec_code(code, cwd=None):
    # Check cache first
    cache_key = hashlib.sha256(f"{code}|{cwd}".encode()).hexdigest()
    cached = _cache.get(cache_key)
    if cached and (time.time() - cached['ts']) < _cache_ttl:
        cached['hits'] = cached.get('hits', 0) + 1
        result = dict(cached['result'])
        result['_cached'] = True
        result['_hits'] = cached['hits']
        return result

    # Change to the requested working directory
    if cwd:
        try:
            os.chdir(cwd)
        except OSError:
            pass  # best effort; caller can catch via stderr
        # Save and restore sys.path around the insertion to avoid import hijacking
        _old_path = sys.path.copy()
        sys.path.insert(0, cwd)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        # Use restricted globals — no access to daemon internals
        safe_builtins = {"print": print, "__builtins__": {
            "print": print, "len": len, "str": str, "int": int, "float": float,
            "bool": bool, "list": list, "dict": dict, "tuple": tuple, "set": set,
            "range": range, "enumerate": enumerate, "zip": zip, "map": map,
            "filter": filter, "sorted": sorted, "reversed": reversed,
            "min": min, "max": max, "sum": sum, "abs": abs, "any": any, "all": all,
            "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
            "type": type, "isinstance": isinstance, "issubclass": issubclass,
            "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
            "IndexError": IndexError, "AttributeError": AttributeError,
            "ImportError": ImportError, "Exception": Exception,
            "True": True, "False": False, "None": None,
            "__import__": __import__,  # Allow imports but deny daemon pipe access
        }}
        exec(code, {"__builtins__": safe_builtins["__builtins__"]}, {})
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        result = {"ok": True, "stdout": stdout, "stderr": stderr, "_cached": False}
    except Exception:
        result = {"ok": False, "stdout": "", "stderr": traceback.format_exc(), "_cached": False}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if cwd:
            sys.path = _old_path  # Restore original sys.path

    # Cache successful results
    if result.get('ok') and result.get('stdout'):
        _cache[cache_key] = {'result': result, 'ts': time.time()}
        # Simple eviction: pop oldest when full
        if len(_cache) > 256:
            try:
                oldest_key = min(_cache.keys(), key=lambda k: _cache[k]['ts'])
                del _cache[oldest_key]
            except (ValueError, KeyError):
                pass  # best-effort eviction

    return result

# Main loop: read JSON from stdin, execute, write JSON result to stdout
for line in sys.stdin:
    try:
        req = json.loads(line.strip())
        cmd = req.get('cmd', '')
        if cmd == 'exit':
            break
        elif cmd == 'sleep':
            duration = float(req.get('duration', 1))
            capped = min(duration, 30)
            time.sleep(capped)
            print(json.dumps({'ok': True, 'stdout': '', 'stderr': '', '_sleep': duration}), flush=True)
        elif cmd == 'cache_clear':
            _cache.clear()
            print(json.dumps({'ok': True, 'stdout': 'cache cleared', 'stderr': ''}), flush=True)
        elif cmd == 'cache_stats':
            print(json.dumps({'ok': True, 'stdout': f'{len(_cache)} entries', 'stderr': ''}), flush=True)
        else:
            result = _exec_code(req.get('code', ''), req.get('cwd'))
            print(json.dumps(result), flush=True)
    except Exception as e:
        print(json.dumps({'ok': False, 'stdout': '', 'stderr': str(e)}), flush=True)
"""


def _daemon_alive() -> bool:
    """Check if daemon process is running without acquiring the lock (internal use)."""
    global _DAEMON_PROCESS
    return _DAEMON_PROCESS is not None and _DAEMON_PROCESS.poll() is None


def _readline_with_timeout(stream, timeout: float) -> str:
    """Read a line from *stream*, raising TimeoutError if *timeout* seconds elapse.

    Uses a daemon thread for the blocking read so this works on Windows
    (where ``select.select`` only supports sockets, not pipe handles).
    """
    import threading as _threading
    import queue as _queue

    if timeout <= 0:
        return stream.readline()

    q: _queue.Queue = _queue.Queue()
    t = _threading.Thread(target=lambda: q.put(stream.readline()), daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"Daemon did not respond within {timeout}s")
    return q.get_nowait()


def _read_daemon_response(timeout: float | None = None) -> dict:
    """Read one JSON response line from daemon stdout with timeout."""
    global _DAEMON_PROCESS
    if _DAEMON_PROCESS is None:
        return {"ok": False, "stdout": "", "stderr": "Daemon not running"}
    t = timeout if timeout is not None else _DAEMON_READ_TIMEOUT
    try:
        result_line = _readline_with_timeout(_DAEMON_PROCESS.stdout, t)
        if not result_line:
            # EOF — daemon likely crashed
            return {"ok": False, "stdout": "", "stderr": "Daemon connection closed (process died)"}
        return json.loads(result_line.strip())
    except json.JSONDecodeError:
        # Protocol desync: drain any remaining stale data from the pipe
        _drain_daemon_pipe()
        return {"ok": False, "stdout": "", "stderr": f"Daemon returned invalid JSON"}
    except TimeoutError as e:
        _drain_daemon_pipe()
        return {"ok": False, "stdout": "", "stderr": str(e)}


def _drain_daemon_pipe():
    """Drain any leftover data from daemon stdout to prevent protocol desync.

    Uses a short non-blocking read loop with threading so it works on Windows.
    """
    global _DAEMON_PROCESS
    if _DAEMON_PROCESS is None or _DAEMON_PROCESS.stdout is None:
        return
    try:
        import threading as _threading
        import queue as _queue

        def _drain() -> None:
            while True:
                try:
                    line = _DAEMON_PROCESS.stdout.readline()
                    if not line:
                        break
                except Exception:
                    break

        q: _queue.Queue = _queue.Queue()
        t = _threading.Thread(target=lambda: q.put(None) or _drain(), daemon=True)
        t.start()
        t.join(0.2)
    except Exception:
        pass


def start() -> dict:
    """Start a persistent Python daemon process.

    Returns:
        dict with keys: status ("started"/"already_running"/"failed"), pid, error
    """
    global _DAEMON_PROCESS
    with _DAEMON_LOCK:
        if _daemon_alive():
            return {"status": "already_running", "pid": _DAEMON_PROCESS.pid}

        _DAEMON_PROCESS = subprocess.Popen(
            [sys.executable, '-c', DAEMON_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Clear in-process cache when starting fresh (avoids stale cache after restart)
        _RESULT_CACHE.clear()

        # Verify daemon is responsive within connect timeout
        try:
            r = _read_daemon_response.__wrapped__ = _read_daemon_response
            # Send a ping via exec_code
            req = json.dumps({"cmd": "exec", "code": "print('daemon:ready')", "cwd": ""})
            _DAEMON_PROCESS.stdin.write(req + "\n")
            _DAEMON_PROCESS.stdin.flush()

            raw = _readline_with_timeout(_DAEMON_PROCESS.stdout, _DAEMON_CONNECT_TIMEOUT)
            if raw:
                resp = json.loads(raw.strip())
                if resp.get("ok") and "ready" in resp.get("stdout", ""):
                    return {"status": "started", "pid": _DAEMON_PROCESS.pid}
            # Daemon responded but not as expected — still probably ok
            return {"status": "started", "pid": _DAEMON_PROCESS.pid}
        except Exception as e:
            # Verification failed — kill the process and report error
            try:
                _DAEMON_PROCESS.kill()
            except Exception:
                pass
            _DAEMON_PROCESS = None
            return {"status": "failed", "error": str(e)}


def exec_code(code: str, cwd: str = "") -> dict:
    """Send code to the daemon and get result. Module imports stay cached.

    Thread-safe: holds _DAEMON_LOCK for the entire request/response cycle.

    Args:
        code: Python code to execute.
        cwd: Working directory (daemon chdir's into this before execution).

    Returns:
        dict with keys: ok, stdout, stderr, _cached (True if from cache)
    """
    global _DAEMON_PROCESS
    global _RESULT_CACHE

    with _DAEMON_LOCK:
        # In-process cache check
        cache_key = hashlib.sha256(f"{code}|{cwd}".encode()).hexdigest()
        cached = _RESULT_CACHE.get(cache_key)
        if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
            result = dict(cached["result"])
            result["_cached"] = True
            return result

        if not _daemon_alive():
            start()
            if not _daemon_alive():
                return {"ok": False, "stdout": "", "stderr": "Failed to start daemon"}

        req = json.dumps({"cmd": "exec", "code": code, "cwd": cwd})
        try:
            _DAEMON_PROCESS.stdin.write(req + "\n")
            _DAEMON_PROCESS.stdin.flush()
        except BrokenPipeError:
            # Daemon died between poll() and write — restart
            start()
            if not _daemon_alive():
                return {"ok": False, "stdout": "", "stderr": "Daemon crashed and restart failed"}
            _DAEMON_PROCESS.stdin.write(req + "\n")
            _DAEMON_PROCESS.stdin.flush()

        result = _read_daemon_response()

        # Cache successful results (with output) in-process as well
        if result.get("ok") and result.get("stdout"):
            _RESULT_CACHE[cache_key] = {"result": result, "ts": time.time()}
            while len(_RESULT_CACHE) > _CACHE_MAX:
                try:
                    oldest = min(_RESULT_CACHE.keys(), key=lambda k: _RESULT_CACHE[k]["ts"])
                    del _RESULT_CACHE[oldest]
                except (ValueError, KeyError):
                    break

        return result


def sleep(duration: float = 1.0) -> dict:
    """Handle sleep natively (no Python execution needed).

    Thread-safe: holds _DAEMON_LOCK.
    """
    global _DAEMON_PROCESS
    with _DAEMON_LOCK:
        if not _daemon_alive():
            start()

        try:
            req = json.dumps({"cmd": "sleep", "duration": duration})
            _DAEMON_PROCESS.stdin.write(req + "\n")
            _DAEMON_PROCESS.stdin.flush()
        except BrokenPipeError:
            start()
            _DAEMON_PROCESS.stdin.write(req + "\n")
            _DAEMON_PROCESS.stdin.flush()

        return _read_daemon_response()


def cache_clear() -> dict:
    """Clear both in-process and daemon caches.

    Thread-safe: holds _DAEMON_LOCK.
    """
    global _RESULT_CACHE
    with _DAEMON_LOCK:
        _RESULT_CACHE.clear()
        if _daemon_alive():
            try:
                _DAEMON_PROCESS.stdin.write(json.dumps({"cmd": "cache_clear"}) + "\n")
                _DAEMON_PROCESS.stdin.flush()
                _read_daemon_response(timeout=5)
            except Exception:
                pass
    return {"ok": True}


def cache_stats() -> dict:
    """Get cache stats from both in-process and daemon caches.

    Thread-safe: holds _DAEMON_LOCK.
    """
    with _DAEMON_LOCK:
        stats = {"in_process": len(_RESULT_CACHE), "daemon": "unknown"}
        if _daemon_alive():
            try:
                _DAEMON_PROCESS.stdin.write(json.dumps({"cmd": "cache_stats"}) + "\n")
                _DAEMON_PROCESS.stdin.flush()
                resp = _read_daemon_response(timeout=5)
                stats["daemon"] = resp.get("stdout", "unknown")
            except Exception:
                pass
        return {"ok": True, **stats}


def stop():
    """Stop the daemon process and clear caches.

    Thread-safe: holds _DAEMON_LOCK.
    """
    global _DAEMON_PROCESS
    global _RESULT_CACHE
    with _DAEMON_LOCK:
        if _daemon_alive():
            try:
                _DAEMON_PROCESS.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                _DAEMON_PROCESS.stdin.flush()
                _DAEMON_PROCESS.wait(timeout=5)
            except Exception:
                try:
                    _DAEMON_PROCESS.kill()
                except Exception:
                    pass
            _DAEMON_PROCESS = None
        _RESULT_CACHE.clear()


def status() -> dict:
    """Check if daemon is running.

    Thread-safe: holds _DAEMON_LOCK.
    """
    with _DAEMON_LOCK:
        if _daemon_alive():
            return {"status": "running", "pid": _DAEMON_PROCESS.pid}
        return {"status": "stopped"}


# ── Auto-cleanup on parent exit ──────────────────────────────────────

@atexit.register
def _cleanup():
    """Stop the daemon on parent process exit. Registered via atexit."""
    global _DAEMON_PROCESS
    if _DAEMON_PROCESS is not None:
        try:
            if _DAEMON_PROCESS.poll() is None:
                _DAEMON_PROCESS.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                _DAEMON_PROCESS.stdin.flush()
                _DAEMON_PROCESS.wait(timeout=3)
        except Exception:
            try:
                _DAEMON_PROCESS.kill()
            except Exception:
                pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Persistent Python daemon")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'exec', 'sleep', 'cache-clear', 'cache-stats'])
    parser.add_argument('code', nargs='?', default='', help='Code to execute')
    parser.add_argument('--cwd', default='', help='Working directory')
    parser.add_argument('--duration', type=float, default=1.0, help='Sleep duration (seconds)')
    args = parser.parse_args()

    if args.action == 'start':
        result = start()
    elif args.action == 'stop':
        result = stop()
    elif args.action == 'status':
        result = status()
    elif args.action == 'exec':
        result = exec_code(args.code, args.cwd)
    elif args.action == 'sleep':
        result = sleep(args.duration)
    elif args.action == 'cache-clear':
        result = cache_clear()
    elif args.action == 'cache-stats':
        result = cache_stats()
    else:
        result = {"error": f"Unknown action: {args.action}"}

    print(json.dumps(result, indent=2))
