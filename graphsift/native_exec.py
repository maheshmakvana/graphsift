#!/usr/bin/env python3
"""Run Python code via direct subprocess (NO shell) and return output.
Pre-approved in settings.json = no permission prompt, no classifier delay."""

import subprocess, sys, json, os, argparse
from pathlib import Path

def run_python_file(path: str, cwd: str = "", max_lines: int = 200, timeout: int = 60):
    """Run a Python file via direct subprocess — no shell involved."""
    resolved_cwd = cwd or os.getcwd()
    script = Path(path)
    if not script.is_absolute():
        script = Path(resolved_cwd) / script
    if not script.exists():
        return {"exit_code": -1, "stdout": "", "stderr": f"File not found: {script}"}

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, cwd=resolved_cwd, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        stdout_lines = proc.stdout.splitlines()[:max_lines]
        return {
            "exit_code": proc.returncode,
            "stdout": "\n".join(stdout_lines),
            "stderr": proc.stderr[:2000],
            "execution_ms": "in-process (no shell overhead)",
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}

def run_python_code(code: str, cwd: str = "", max_lines: int = 200):
    """Run inline Python code via direct subprocess — no shell involved."""
    resolved_cwd = cwd or os.getcwd()
    try:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, cwd=resolved_cwd, timeout=30,
            encoding="utf-8", errors="replace",
        )
        stdout_lines = proc.stdout.splitlines()[:max_lines]
        return {
            "exit_code": proc.returncode,
            "stdout": "\n".join(stdout_lines),
            "stderr": proc.stderr[:2000],
            "execution_ms": "in-process (no shell overhead)",
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "Timeout after 30s"}

def main():
    parser = argparse.ArgumentParser(description="Run Python without shell")
    parser.add_argument('command', choices=['file', 'code'])
    parser.add_argument('value', help='Script path or code string')
    parser.add_argument('--cwd', default='', help='Working directory')
    parser.add_argument('--max-lines', type=int, default=200)
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()

    if args.command == 'file':
        result = run_python_file(args.value, args.cwd, args.max_lines, args.timeout)
    else:
        result = run_python_code(args.value, args.cwd, args.max_lines)

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
