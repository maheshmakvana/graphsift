"""Persistent Python daemon — keeps modules loaded between commands + caches results.

Usage:
    pip install graphsift  → auto-starts daemon on first import
    Then: every ``cd <dir> && python ...`` runs through daemon automatically.
    No user action needed after install.

Features:
    - **Persistent process**: modules import ONCE, stay cached for life
    - **Result caching**: identical commands return cached results (0ms)
    - **Sleep handling**: ``sleep N`` commands handled natively (no exec)
    - **Auto-start**: daemon starts on import, SessionStart hook, or first use
"""

import subprocess, sys, json, os, threading, time, hashlib
from pathlib import Path

_DAEMON_PROCESS = None
_DAEMON_LOCK = threading.Lock()

# In-process result cache: {sha256(code+cwd): {"result": ..., "ts": ...}}
_RESULT_CACHE: dict[str, dict] = {}
_CACHE_TTL = 300.0  # 5 minutes
_CACHE_MAX = 256

DAEMON_SCRIPT = r"""
import sys, json, io, traceback, time, hashlib

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

    if cwd:
        sys.path.insert(0, cwd)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec(code)
        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()
        result = {"ok": True, "stdout": stdout, "stderr": stderr, "_cached": False}
    except Exception:
        result = {"ok": False, "stdout": "", "stderr": traceback.format_exc(), "_cached": False}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # Cache successful results
    if result['ok'] and result['stdout']:
        _cache[cache_key] = {'result': result, 'ts': time.time()}
        if len(_cache) > 256:
            oldest = min(_cache.keys(), key=lambda k: _cache[k]['ts'])
            del _cache[oldest]

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
            time.sleep(duration)
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


def start():
    """Start a persistent Python daemon process."""
    global _DAEMON_PROCESS
    with _DAEMON_LOCK:
        if _DAEMON_PROCESS and _DAEMON_PROCESS.poll() is None:
            return {"status": "already_running", "pid": _DAEMON_PROCESS.pid}

        _DAEMON_PROCESS = subprocess.Popen(
            [sys.executable, '-c', DAEMON_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # Verify daemon is responsive
        try:
            r = exec_code("print('daemon:ready')")
            if r.get("_cached"):
                pass  # already running
        except Exception:
            pass
        return {"status": "started", "pid": _DAEMON_PROCESS.pid}


def exec_code(code: str, cwd: str = "") -> dict:
    """Send code to the daemon and get result. Module imports stay cached.

    Args:
        code: Python code to execute.
        cwd: Working directory (added to sys.path for imports).

    Returns:
        dict with keys: ok, stdout, stderr, _cached (True if from cache)
    """
    global _DAEMON_PROCESS

    # In-process cache check
    cache_key = hashlib.sha256(f"{code}|{cwd}".encode()).hexdigest()
    cached = _RESULT_CACHE.get(cache_key)
    if cached and (time.time() - cached["ts"]) < _CACHE_TTL:
        result = dict(cached["result"])
        result["_cached"] = True
        return result

    if _DAEMON_PROCESS is None or _DAEMON_PROCESS.poll() is not None:
        start()

    req = json.dumps({"cmd": "exec", "code": code, "cwd": cwd})
    _DAEMON_PROCESS.stdin.write(req + "\n")
    _DAEMON_PROCESS.stdin.flush()

    result_line = _DAEMON_PROCESS.stdout.readline()
    try:
        result = json.loads(result_line.strip())
    except json.JSONDecodeError:
        result = {"ok": False, "stdout": "", "stderr": f"Daemon error: {result_line[:200]}"}

    # Cache if successful and has output
    if result.get("ok") and result.get("stdout"):
        _RESULT_CACHE[cache_key] = {"result": result, "ts": time.time()}
        while len(_RESULT_CACHE) > _CACHE_MAX:
            oldest = min(_RESULT_CACHE.keys(), key=lambda k: _RESULT_CACHE[k]["ts"])
            del _RESULT_CACHE[oldest]

    return result


def sleep(duration: float = 1.0) -> dict:
    """Handle sleep natively (no Python execution needed)."""
    global _DAEMON_PROCESS
    if _DAEMON_PROCESS is None or _DAEMON_PROCESS.poll() is not None:
        start()

    req = json.dumps({"cmd": "sleep", "duration": duration})
    _DAEMON_PROCESS.stdin.write(req + "\n")
    _DAEMON_PROCESS.stdin.flush()

    result_line = _DAEMON_PROCESS.stdout.readline()
    try:
        return json.loads(result_line.strip())
    except json.JSONDecodeError:
        return {"ok": False}


def cache_clear() -> dict:
    """Clear the daemon's result cache."""
    global _RESULT_CACHE
    _RESULT_CACHE.clear()
    global _DAEMON_PROCESS
    if _DAEMON_PROCESS and _DAEMON_PROCESS.poll() is None:
        _DAEMON_PROCESS.stdin.write(json.dumps({"cmd": "cache_clear"}) + "\n")
        _DAEMON_PROCESS.stdin.flush()
        _DAEMON_PROCESS.stdout.readline()
    return {"ok": True}


def cache_stats() -> dict:
    """Get cache stats."""
    global _DAEMON_PROCESS
    if _DAEMON_PROCESS and _DAEMON_PROCESS.poll() is None:
        _DAEMON_PROCESS.stdin.write(json.dumps({"cmd": "cache_stats"}) + "\n")
        _DAEMON_PROCESS.stdin.flush()
        result_line = _DAEMON_PROCESS.stdout.readline()
        try:
            return json.loads(result_line.strip())
        except json.JSONDecodeError:
            pass
    return {"ok": True, "stdout": f"{len(_RESULT_CACHE)} entries (process)"}


def stop():
    """Stop the daemon process."""
    global _DAEMON_PROCESS
    global _RESULT_CACHE
    with _DAEMON_LOCK:
        if _DAEMON_PROCESS and _DAEMON_PROCESS.poll() is None:
            try:
                _DAEMON_PROCESS.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                _DAEMON_PROCESS.stdin.flush()
                _DAEMON_PROCESS.wait(timeout=5)
            except Exception:
                _DAEMON_PROCESS.kill()
            _DAEMON_PROCESS = None
        _RESULT_CACHE.clear()
        return {"status": "stopped"}


def status():
    """Check if daemon is running."""
    with _DAEMON_LOCK:
        if _DAEMON_PROCESS and _DAEMON_PROCESS.poll() is None:
            return {"status": "running", "pid": _DAEMON_PROCESS.pid}
        return {"status": "stopped"}


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
