"""Python fallback launcher for the graphsift persistent daemon.

Used when the native launcher cannot be built (no ``csc.exe`` on Windows,
non-Windows platforms without a compiler). Guarantees *correct* execution
for commands rewritten by the smart-execution hook, though it does not get
the native shim's ~50ms startup.

Invoked by the hook as::

    python -m graphsift.launcher_fallback --codefile <path> [--cwd <dir>]
    python -m graphsift.launcher_fallback --script <path> [--cwd <dir>]
    python -m graphsift.launcher_fallback --module <name> [--cwd <dir>]
    python -m graphsift.launcher_fallback --sleep <seconds>
"""

from __future__ import annotations

import argparse
import os
import sys


def _main() -> int:
    parser = argparse.ArgumentParser(description="graphsift launcher fallback")
    parser.add_argument("--codefile", default="")
    parser.add_argument("--script", default="")
    parser.add_argument("--module", default="")
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--cwd", default="")
    args = parser.parse_args()

    if args.sleep is not None:
        import time

        time.sleep(min(args.sleep, 30.0))
        return 0

    from graphsift.daemon import exec_code

    code = ""
    codefile_used = args.codefile
    if args.codefile:
        try:
            with open(args.codefile, "r", encoding="utf-8", errors="replace") as fh:
                code = fh.read()
        except OSError:
            codefile_used = ""

    result = exec_code(
        code,
        args.cwd,
        path=args.script,
        module=args.module,
    )

    out = result.get("stdout", "") or ""
    err = result.get("stderr", "") or ""
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    sys.stdout.flush()
    sys.stderr.flush()

    # Best-effort cleanup of the temp code file.
    if codefile_used:
        try:
            os.unlink(codefile_used)
        except OSError:
            pass

    return int(result.get("exit_code", 1))


if __name__ == "__main__":
    sys.exit(_main())
