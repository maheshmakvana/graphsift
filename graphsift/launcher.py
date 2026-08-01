"""Native launcher for the graphsift persistent daemon.

The launcher is a small compiled shim that connects to the daemon's
localhost TCP server and executes the requested Python code in *~50ms* —
far faster than spawning a fresh interpreter. It is built lazily on first
use so ``pip install graphsift`` stays zero-config:

  - **Windows**: compiled from C# with the .NET Framework ``csc.exe`` that
    ships with Windows (no toolchain to install).
  - **Everywhere else / no compiler**: a Python fallback launcher
    (``graphsift.launcher_fallback``) guarantees correct execution, just
    not the same speed.

The hook rewrites ``cd X && python -c "..."`` commands to invoke this
launcher, passing the code via a temp ``--codefile`` (avoiding shell
quoting) and letting the launcher read ``~/.graphsift/daemon.json`` for
the port + auth token.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

_LAUNCHER_DIR = Path.home() / ".graphsift" / "bin"
_EXE = "graphsift_launcher.exe" if os.name == "nt" else "graphsift_launcher"
_LAUNCHER_PATH = _LAUNCHER_DIR / _EXE
_FAIL_MARKER = _LAUNCHER_DIR / ".launcher_failed"
_VERSION_FILE = _LAUNCHER_DIR / "graphsift_launcher.version"


def _current_version() -> str:
    try:
        from graphsift._version import __version__ as _v  # noqa: PLC0415

        return str(_v)
    except Exception:  # noqa: BLE001
        return "unknown"

# ---------------------------------------------------------------------------
# C# source (Windows). Uses System.Web.Extensions' JavaScriptSerializer so no
# NuGet / external deps are needed under the stock .NET Framework 4.x csc.
# ---------------------------------------------------------------------------

_C_SHARP_SOURCE = r"""
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;

class GraphsiftLauncher
{
    static string _InfoPath;

    static int Main(string[] args)
    {
        string codefile = null, script = null, moduleName = null, cwd = null;
        double sleepSec = -1;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--codefile" && i + 1 < args.Length) codefile = args[++i];
            else if (args[i] == "--script" && i + 1 < args.Length) script = args[++i];
            else if (args[i] == "--module" && i + 1 < args.Length) moduleName = args[++i];
            else if (args[i] == "--sleep" && i + 1 < args.Length) double.TryParse(args[++i], out sleepSec);
            else if (args[i] == "--cwd" && i + 1 < args.Length) cwd = args[++i];
        }

        string infoEnv = Environment.GetEnvironmentVariable("GRAPHSIFT_DAEMON_FILE");
        if (!string.IsNullOrEmpty(infoEnv))
        {
            _InfoPath = infoEnv;
        }
        else
        {
            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            _InfoPath = Path.Combine(home, ".graphsift", "daemon.json");
        }

        // Sleep is handled natively — no daemon required.
        if (sleepSec >= 0)
        {
            Thread.Sleep((int)(Math.Min(sleepSec, 30.0) * 1000));
            return 0;
        }

        int port = 0; string token = null, python = "python";
        try
        {
            var ser = new JavaScriptSerializer();
            var info = ser.Deserialize<Dictionary<string, object>>(File.ReadAllText(_InfoPath));
            if (info != null && info.ContainsKey("port")) port = Convert.ToInt32(info["port"]);
            if (info != null && info.ContainsKey("token")) token = Convert.ToString(info["token"]);
            if (info != null && info.ContainsKey("python")) python = Convert.ToString(info["python"]);
        }
        catch (Exception) { }

        if (port > 0 && token != null)
        {
            int exitCode;
            if (TryDaemonExec(port, token, codefile, script, moduleName, cwd, out exitCode))
            {
                CleanupCodefile(codefile);
                return exitCode;
            }
        }

        int code = RunPythonFallback(python, codefile, script, moduleName, cwd);
        CleanupCodefile(codefile);
        return code;
    }

    static void CleanupCodefile(string codefile)
    {
        try { if (codefile != null && File.Exists(codefile)) File.Delete(codefile); } catch (Exception) { }
    }

    static bool TryDaemonExec(int port, string token, string codefile, string script,
                              string moduleName, string cwd, out int exitCode)
    {
        exitCode = 1;
        try
        {
            string code = "";
            if (codefile != null && File.Exists(codefile)) code = File.ReadAllText(codefile);
            string cmd = script != null ? "script" : (moduleName != null ? "module" : "exec");
            string pathVal = script != null ? script : "";

            var ser = new JavaScriptSerializer();
            var req = new Dictionary<string, object> {
                { "token", token },
                { "cmd", cmd },
                { "code", code },
                { "cwd", cwd == null ? "" : cwd },
                { "path", pathVal },
                { "module", moduleName == null ? "" : moduleName }
            };
            string reqJson = ser.Serialize(req);

            using (var client = new TcpClient())
            {
                client.Connect("127.0.0.1", port);
                client.ReceiveTimeout = 30000;
                var stream = client.GetStream();
                byte[] bytes = Encoding.UTF8.GetBytes(reqJson + "\n");
                stream.Write(bytes, 0, bytes.Length);
                stream.Flush();

                var sb = new StringBuilder();
                var buf = new byte[4096];
                while (true)
                {
                    int n = stream.Read(buf, 0, buf.Length);
                    if (n <= 0) break;
                    sb.Append(Encoding.UTF8.GetString(buf, 0, n));
                    if (sb.ToString().Contains("\n")) break;
                }
                string respJson = sb.ToString().TrimEnd('\r', '\n');
                if (respJson.Length == 0) return false;

                var resp = ser.Deserialize<Dictionary<string, object>>(respJson);
                if (resp == null) return false;
                string stdout = resp.ContainsKey("stdout") ? Convert.ToString(resp["stdout"]) : "";
                string stderr = resp.ContainsKey("stderr") ? Convert.ToString(resp["stderr"]) : "";
                if (resp.ContainsKey("exit_code")) exitCode = Convert.ToInt32(resp["exit_code"]);
                if (stdout.Length > 0) Console.Write(stdout);
                if (stderr.Length > 0) Console.Error.Write(stderr);
                return true;
            }
        }
        catch (Exception) { return false; }
    }

    static int RunPythonFallback(string python, string codefile, string script,
                                 string moduleName, string cwd)
    {
        try
        {
            var psi = new ProcessStartInfo();
            psi.FileName = python;
            psi.UseShellExecute = false;
            if (cwd != null && cwd.Length > 0) psi.WorkingDirectory = cwd;
            if (script != null) psi.Arguments = Quote(script);
            else if (moduleName != null) psi.Arguments = "-m " + Quote(moduleName);
            else psi.Arguments = Quote(codefile != null ? codefile : "");
            using (var p = Process.Start(psi))
            {
                p.WaitForExit();
                return p.ExitCode;
            }
        }
        catch (Exception) { return 1; }
    }

    static string Quote(string s)
    {
        return "\"" + s.Replace("\"", "\\\"") + "\"";
    }
}
"""


# ---------------------------------------------------------------------------
# Build + lookup
# ---------------------------------------------------------------------------


def _find_csc() -> str | None:
    """Locate the .NET Framework C# compiler on Windows."""
    if os.name != "nt":
        return None
    candidates = (
        r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe",
        r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe",
    )
    for cand in candidates:
        if Path(cand).exists():
            return cand
    found = shutil.which("csc.exe") or shutil.which("csc")
    return found


def _launcher_is_current() -> bool:
    """True if the built launcher matches the installed graphsift version."""
    if not _LAUNCHER_PATH.exists():
        return False
    try:
        ver = _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return ver == _current_version()


def build_launcher(force: bool = False) -> str | None:
    """Compile the native launcher into ``~/.graphsift/bin/``.

    Rebuilds automatically when the installed graphsift version changes, so
    upgraded users get the new binary. If a rebuild fails and an older
    binary exists, the old one is kept (same JSON protocol, still works).

    Returns the launcher path on success (or existing binary as fallback),
    None if no binary is possible (e.g. no compiler and none built).
    """
    if _LAUNCHER_PATH.exists() and _launcher_is_current() and not force:
        return str(_LAUNCHER_PATH)
    # Hard failure with no binary at all — don't retry every command.
    if not _LAUNCHER_PATH.exists() and _FAIL_MARKER.exists() and not force:
        return None
    # Rebuilding a stale binary: clear the failure marker so retries happen.
    if _LAUNCHER_PATH.exists():
        try:
            _FAIL_MARKER.unlink(missing_ok=True)
        except OSError:
            pass

    csc = _find_csc()
    if csc is None:
        # No compiler — keep any existing (possibly older) binary.
        return str(_LAUNCHER_PATH) if _LAUNCHER_PATH.exists() else None

    try:
        _LAUNCHER_DIR.mkdir(parents=True, exist_ok=True)
        cs_path = _LAUNCHER_DIR / "graphsift_launcher.cs"
        cs_path.write_text(_C_SHARP_SOURCE, encoding="utf-8")
        proc = subprocess.run(
            [
                csc,
                "/nologo",
                "/optimize+",
                "/target:exe",
                f"/out:{_LAUNCHER_PATH}",
                "/r:System.Web.Extensions.dll",
                str(cs_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0 or not _LAUNCHER_PATH.exists():
            _FAIL_MARKER.write_text(
                proc.stdout + "\n" + proc.stderr, encoding="utf-8"
            )
            return str(_LAUNCHER_PATH) if _LAUNCHER_PATH.exists() else None
        _VERSION_FILE.write_text(_current_version(), encoding="utf-8")
        try:
            _FAIL_MARKER.unlink(missing_ok=True)
        except OSError:
            pass
        return str(_LAUNCHER_PATH)
    except Exception:  # noqa: BLE001
        try:
            _FAIL_MARKER.write_text("build failed", encoding="utf-8")
        except OSError:
            pass
        return str(_LAUNCHER_PATH) if _LAUNCHER_PATH.exists() else None


def launcher_path() -> str | None:
    """Return the native launcher path if a binary exists, else None."""
    if _LAUNCHER_PATH.exists():
        return str(_LAUNCHER_PATH)
    return None


def ensure_launcher() -> str | None:
    """Return a current-version launcher path, building it on first use.

    Returns the native launcher path when available, else None (callers fall
    back to the Python launcher).
    """
    if _LAUNCHER_PATH.exists() and _launcher_is_current():
        return str(_LAUNCHER_PATH)
    return build_launcher()


# Lazy build in a background thread (so first import is never blocked).
def ensure_launcher_async() -> None:
    """Trigger a background launcher (re)build if it isn't current yet."""
    if not _launcher_is_current():
        threading.Thread(target=build_launcher, daemon=True).start()


# ---------------------------------------------------------------------------
# Command builder shared by the hook
# ---------------------------------------------------------------------------


def build_launcher_command(
    *,
    codefile: str = "",
    script: str = "",
    module: str = "",
    sleep_seconds: float | None = None,
    cwd: str = "",
    shell: str = "bash",
) -> str | None:
    """Build the shell command that invokes the launcher for a request.

    Returns the command string, or None if no launcher (native or Python
    fallback) is available.
    """
    launcher = ensure_launcher()
    if launcher is None:
        py = sys.executable.replace("\\", "/")
        launcher = f'{py} -m graphsift.launcher_fallback'
        prefix = ""  # `python -m` needs no call operator
    else:
        launcher = launcher.replace("\\", "/")
        # PowerShell needs the call operator for a quoted exe path.
        prefix = "& " if shell == "powershell" else ""

    parts = [f'"{launcher}"']
    if codefile:
        parts.append(f'--codefile "{codefile.replace(chr(92), "/")}"')
    if script:
        parts.append(f'--script "{script.replace(chr(92), "/")}"')
    if module:
        parts.append(f'--module "{module}"')
    if sleep_seconds is not None:
        parts.append(f"--sleep {sleep_seconds}")
    if cwd:
        parts.append(f'--cwd "{cwd.replace(chr(92), "/")}"')

    return prefix + " ".join(parts)
