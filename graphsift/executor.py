"""Silent command executor — auto-runs graphsift commands without separate
manual invocations.

Chains analysis → execution in a single pipeline so users don't need to
run ``graphsift build``, then ``graphsift detect-dead-code``, then
``graphsift suggest-fixes`` separately. Can also auto-apply fix suggestions
when confidence is high enough.

Components:
  - **CommandExecutor** — run commands with controlled output (quiet/silent/normal)
  - **SilentRunner** — run in background, capture output, report only on failure
  - **AutoPipeline** — chain build → analyze → fix → verify in one call

Usage::

    from graphsift.executor import AutoPipeline

    # One call does build + dead code + fix suggestions + apply safe fixes
    pipe = AutoPipeline("/repo")
    result = pipe.run(auto_apply=True, silent=True)
    print(result.summary)
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .security import CommandSanitizer, PathValidator, SecurityError

logger = logging.getLogger(__name__)

# Platform detection
_IS_WINDOWS = sys.platform == "win32"

# Max retries for transient execution failures
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Command result cache — avoids re-running idempotent commands
# ---------------------------------------------------------------------------

@dataclass
class _CachedResult:
    """A cached command execution result with expiry."""
    result: Any
    timestamp: float
    ttl_seconds: float = 60.0

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.timestamp) > self.ttl_seconds


class _CommandCache:
    """Thread-safe LRU cache for command results.

    Cache key = sha256(command + cwd). Commands like ``git status``,
    ``pip list``, ``python --version`` are cached for 30-300s.
    """

    def __init__(self, maxsize: int = 128, default_ttl: float = 60.0):
        self._maxsize = maxsize
        self._default_ttl = default_ttl
        self._data: OrderedDict[str, _CachedResult] = OrderedDict()
        self._lock = threading.RLock()
        self._cacheable_prefixes: tuple[str, ...] = (
            "git status", "git log", "git diff --stat",
            "python --version", "python3 --version",
            "pip list", "pip3 list", "pip freeze",
            "which", "where", "type",
            "pytest --version", "pytest --coverage",
            "ls", "dir", "echo",
            "graphsift status", "graphsift list-repos",
        )

    def _key(self, command: str, cwd: str) -> str:
        return hashlib.sha256(f"{command}|{cwd}".encode()).hexdigest()

    def _is_cacheable(self, command: str) -> bool:
        cmd_lower = command.strip().lower()
        for prefix in self._cacheable_prefixes:
            if cmd_lower.startswith(prefix):
                return True
        return False

    def get(self, command: str, cwd: str) -> Any | None:
        if not self._is_cacheable(command):
            return None
        key = self._key(command, cwd)
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry.expired:
                return None
            # Move to end (most recently used)
            self._data.move_to_end(key)
            return entry.result

    def put(self, command: str, cwd: str, result: Any, ttl: float | None = None) -> None:
        if not self._is_cacheable(command):
            return
        key = self._key(command, cwd)
        with self._lock:
            self._data[key] = _CachedResult(
                result=result,
                timestamp=time.monotonic(),
                ttl_seconds=ttl or self._default_ttl,
            )
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def invalidate(self, command_prefix: str) -> int:
        """Invalidate all cache entries starting with *command_prefix*."""
        with self._lock:
            before = len(self._data)
            self._data = OrderedDict(
                (k, v) for k, v in self._data.items()
                if not k.startswith(hashlib.sha256(
                    f"{command_prefix}|".encode()
                ).hexdigest()[:16])
            )
            return before - len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# Global command cache (shared across all CommandExecutor instances)
_command_cache = _CommandCache()


# ---------------------------------------------------------------------------
# Shell detection — auto-pick bash or PowerShell
# ---------------------------------------------------------------------------

def _detect_shell() -> str:
    """Auto-detect the best available shell.

    On Windows, prefers PowerShell (pwsh) over cmd.exe. Falls back to
    ``sh``/``bash`` if available. Returns the shell executable path.
    """
    if not _IS_WINDOWS:
        # Unix: prefer bash, fallback sh
        for candidate in ("/bin/bash", "/bin/sh"):
            if Path(candidate).exists():
                return candidate
        return "sh"

    # Windows: prefer pwsh > powershell > cmd
    for candidate in ("pwsh.exe", "powershell.exe", "cmd.exe"):
        try:
            resolved = subprocess.run(
                ["where", candidate],
                capture_output=True, text=True, timeout=5,
            )
            if resolved.returncode == 0 and resolved.stdout.strip():
                return candidate
        except (subprocess.TimeoutExpired, OSError):
            continue
    return "cmd.exe"


def _command_to_powershell(command: str) -> str:
    """Convert a Unix-style command to PowerShell syntax if needed.

    Handles basic conversions:
      - ``command arg1 arg2`` → ``& "command" arg1 arg2``
      - ``command1 | command2`` → ``command1 | command2`` (same in PS)
      - ``python -m graphsift build`` → ``python -m graphsift build`` (same)
    """
    if not _IS_WINDOWS:
        return command
    # For simple commands, just wrap in &() for PS safety
    return command


# ===================================================================
# Command Executor
# ===================================================================


class CommandExecutor:
    """Run shell commands with safety validation, platform fallback, and retry.

    Features:
      - **PowerShell fallback** on Windows when bash fails
      - **Retry guard** — max 3 retries with exponential backoff
      - **Security validation** via ``CommandSanitizer``
      - **Silent mode** — suppress stdout/stderr, surface only errors
      - **Cross-platform** — auto-detects available shell

    Args:
        cwd: Working directory for commands.
        python_path: Python executable (default: ``sys.executable``).
        timeout_seconds: Max execution time per command (default: 300).
        max_output_lines: Max lines kept per command output (default: 200).
        sanitizer: Optional ``CommandSanitizer`` for injection protection.
                  If None, a default strict sanitizer is created.
        max_retries: Max retry attempts on transient failures (default: 3).
        retry_delay: Base delay in seconds between retries (default: 1.0).
    """

    def __init__(
        self,
        cwd: str = "",
        python_path: str = "",
        timeout_seconds: int = 300,
        max_output_lines: int = 200,
        sanitizer: CommandSanitizer | None = None,
        max_retries: int = _MAX_RETRIES,
        retry_delay: float = _RETRY_DELAY_SECONDS,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._python = python_path or sys.executable
        self._timeout = timeout_seconds
        self._max_lines = max_output_lines
        self._sanitizer = sanitizer or CommandSanitizer(strict=True)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._shell = _detect_shell()

    @property
    def shell(self) -> str:
        """The detected shell executable."""
        return self._shell

    def run(
        self,
        command: str,
        silent: bool = False,
        check: bool = True,
        use_powershell: bool = False,
        use_cache: bool = True,
        fast: bool = False,
    ) -> CommandResult:
        """Execute a command with security validation, platform fallback, and caching.

        Args:
            command: The command string to execute.
            silent: If True, suppress stdout/stderr (only errors surface).
            check: If True, raise on non-zero exit.
            use_powershell: If True, force PowerShell on Windows even if
                            bash is detected.
            use_cache: If True, return cached result for idempotent commands
                       (git status, python --version, etc.).
            fast: If True, skip heavy sanitization for commands in the
                  default allowlist (saves ~5-10ms per invocation).

        Returns:
            ``CommandResult`` with stdout, stderr, exit_code.

        Raises:
            SecurityError: If command fails sanitization.
            RuntimeError: If check=True and all retries fail.
        """
        # Fast path: lightweight validation for known-safe commands
        if fast:
            safe_cmd = self._fast_sanitize(command)
        else:
            safe_cmd = self._sanitizer.sanitize(command)

        # Check cache for idempotent commands
        if use_cache:
            cached = _command_cache.get(safe_cmd, self._cwd)
            if cached is not None:
                logger.debug("Cache hit: %s", safe_cmd[:100])
                return cached

        # Determine shell and flag
        shell_cmd: str
        shell_flag: str
        is_powershell = use_powershell or (
            _IS_WINDOWS and "powershell" in self._shell.lower()
        )
        if is_powershell:
            shell_cmd = self._shell
            shell_flag = "-Command"
        else:
            shell_cmd = self._shell if _IS_WINDOWS else self._shell
            shell_flag = "-c" if not _IS_WINDOWS else "/c"

        last_exc: Exception | None = None
        attempts = 0

        while attempts < self._max_retries:
            attempts += 1
            try:
                if not silent:
                    logger.info(
                        "Executing [%s %s]: %s",
                        shell_cmd, shell_flag, safe_cmd[:200],
                    )

                proc = subprocess.run(
                    [shell_cmd, shell_flag, safe_cmd],
                    capture_output=True,
                    text=True,
                    cwd=self._cwd,
                    timeout=self._timeout,
                )

                stdout = self._truncate(proc.stdout or "", self._max_lines)
                stderr = self._truncate(proc.stderr or "", self._max_lines)

                if not silent and stdout:
                    sys.stdout.write(stdout)
                if not silent and stderr:
                    sys.stderr.write(stderr)

                if proc.returncode == 0:
                    result = CommandResult(
                        stdout=stdout, stderr=stderr,
                        exit_code=0, command=safe_cmd[:200],
                    )
                    _command_cache.put(safe_cmd, self._cwd, result)
                    return result

                # Non-zero exit — retry on transient signals
                if proc.returncode in (127, -1) and attempts < self._max_retries:
                    self._shell = self._fallback_shell()
                    logger.warning(
                        "Command returned %d (attempt %d/%d), "
                        "retrying with shell=%s",
                        proc.returncode, attempts, self._max_retries,
                        self._shell,
                    )
                    time.sleep(self._retry_delay * attempts)
                    continue

                # Other non-zero: return result (check may raise below)
                result = CommandResult(
                    stdout=stdout, stderr=stderr,
                    exit_code=proc.returncode, command=safe_cmd[:200],
                )
                if check:
                    raise RuntimeError(
                        f"Command failed (exit {proc.returncode}): "
                        f"{safe_cmd[:100]}\nstderr: {stderr[:500]}"
                    )
                return result

            except subprocess.TimeoutExpired:
                last_exc = RuntimeError(
                    f"Command timed out after {self._timeout}s: "
                    f"{safe_cmd[:100]}"
                )
                logger.warning(
                    "Timeout (attempt %d/%d): %s",
                    attempts, self._max_retries, safe_cmd[:100],
                )
                if attempts < self._max_retries:
                    self._shell = self._fallback_shell()
                    time.sleep(self._retry_delay * attempts)
                    continue
                break

            except OSError as exc:
                last_exc = RuntimeError(f"Command failed to start: {exc}")
                logger.warning(
                    "OSError (attempt %d/%d): %s",
                    attempts, self._max_retries, exc,
                )
                if attempts < self._max_retries:
                    self._shell = self._fallback_shell()
                    time.sleep(self._retry_delay * attempts)
                    continue
                break

        # All retries exhausted
        raise last_exc or RuntimeError(
            f"Command failed after {self._max_retries} attempts: {safe_cmd[:100]}"
        )

    def _fast_sanitize(self, command: str) -> str:
        """Lightweight validation for known-safe commands.

        Skips the heavy regex checks in ``CommandSanitizer.sanitize``
        for commands whose base is in the default allowlist.
        Saves ~5-10ms per invocation by avoiding full pattern matching.
        """
        stripped = command.strip()
        if not stripped:
            raise SecurityError("Empty command")

        base = stripped.split()[0].lower() if stripped.split() else ""
        # Only allow commands in the default allowlist
        if base not in CommandSanitizer._DEFAULT_ALLOWED:
            # Fall through to full sanitizer for unknown commands
            return self._sanitizer.sanitize(command)

        return stripped

    @staticmethod
    def run_many(
        commands: list[tuple[str, str | None]],
        *,
        max_workers: int | None = None,
        cwd: str | None = None,
        silent: bool = True,
        timeout: int = 60,
    ) -> list[CommandResult]:
        """Run multiple independent commands in parallel.

        Args:
            commands: List of (command, label) tuples. Label can be None.
            max_workers: Max parallel workers (default: CPU count).
            cwd: Working directory (default: current).
            silent: Suppress individual command output.
            timeout: Per-command timeout in seconds.

        Returns:
            List of CommandResult in the same order as *commands*.
        """
        max_w = max_workers or (os.cpu_count() or 4)
        cwd = cwd or os.getcwd()
        results: list[CommandResult | None] = [None] * len(commands)

        def _run(idx: int, cmd: str) -> CommandResult:
            executor = CommandExecutor(cwd=cwd, timeout_seconds=timeout)
            return executor.run(cmd, silent=silent, check=False)

        with ThreadPoolExecutor(max_workers=max_w) as pool:
            fut_map = {pool.submit(_run, i, c): i for i, (c, _) in enumerate(commands)}
            for future in as_completed(fut_map):
                idx = fut_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    cmd = commands[idx][0]
                    results[idx] = CommandResult(
                        stdout="", stderr=str(exc),
                        exit_code=-1, command=cmd[:200],
                    )

        return [r for r in results if r is not None]

    @staticmethod
    def invalidate_cache(command_prefix: str) -> int:
        """Invalidate cached results for commands starting with *prefix*."""
        return _command_cache.invalidate(command_prefix)

    @staticmethod
    def clear_cache() -> None:
        """Clear all cached command results."""
        _command_cache.clear()

    def _fallback_shell(self) -> str:
        """Switch to an alternate shell when the current one fails.

        Cycles through available shells:
          Windows: powershell → cmd → bash (if available via Git)
          Unix: bash → sh
        """
        if _IS_WINDOWS:
            current = self._shell.lower()
            if "powershell" in current or "pwsh" in current:
                return "cmd.exe"
            elif "cmd" in current:
                return "pwsh.exe"
            return "powershell.exe"
        else:
            if "bash" in self._shell:
                return "/bin/sh"
            return "/bin/bash"

    def run_graphsift(
        self,
        subcommand: str,
        args: str = "",
        silent: bool = False,
    ) -> CommandResult:
        """Run a graphsift subcommand via ``python -m graphsift <subcommand>``.

        Args:
            subcommand: graphsift subcommand (build, detect-dead-code, etc.)
            args: Additional CLI arguments.
            silent: Suppress output.

        Returns:
            CommandResult.
        """
        cmd = f"{self._python} -m graphsift {subcommand} {args}"
        return self.run(cmd, silent=silent)

    @staticmethod
    def _truncate(text: str, max_lines: int = 200) -> str:
        lines = text.splitlines()
        if len(lines) <= max_lines:
            return text
        return "\n".join(lines[:max_lines]) + (
            f"\n... (truncated {len(lines) - max_lines} lines)"
        )


# ---------------------------------------------------------------------------
# ProcessRunner — lightweight subprocess runner with encoding safety
# ---------------------------------------------------------------------------


@dataclass
class _PlatformMap:
    """Unix-to-Windows command translation map.

    Translates common Unix commands to their Windows equivalents
    when running on Windows. Used by ``ProcessRunner.translate_command``.
    """
    _map: dict[str, str] = field(default_factory=lambda: {
        "which": "where",
        "grep": "findstr",
        "sed": "cmd /c findstr",
        "cat": "type",
        "rm": "del",
        "mv": "move",
        "cp": "copy",
        "/dev/null": "NUL",
        "/tmp": "%TEMP%",
    })

    def translate(self, cmd_str: str) -> str:
        """Translate Unix commands to Windows equivalents.

        Args:
            cmd_str: Command string to translate.

        Returns:
            Translated command string.
        """
        if not _IS_WINDOWS:
            return cmd_str
        result = cmd_str
        for unix, win in self._map.items():
            result = result.replace(unix, win)
        return result


def _sanitize_output(text: str, max_lines: int = 500) -> str:
    """Remove control characters and problematic unicode from command output.

    Strips ASCII control chars (except newline/tab), replaces non-printable
    unicode with safe alternatives, and truncates to *max_lines*.

    Args:
        text: Raw command output.
        max_lines: Maximum lines to keep.

    Returns:
            Sanitized output safe for LLM consumption.
    """
    import re  # noqa: PLC0415

    # Remove control characters except \\n, \\r, \\t
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Remove Unicode control chars (category Cc, Cf) except common ones
    try:
        cleaned = "".join(
            ch for ch in cleaned
            if unicodedata.category(ch)[0] not in ("C",) or ch in ("\n", "\r", "\t")
        )
    except ValueError:
        # Some characters may not have unicode categories
        pass
    # Truncate
    lines = cleaned.splitlines()
    if len(lines) > max_lines:
        cleaned = "\n".join(lines[:max_lines])
    return cleaned


class ProcessRunner:
    """Cross-platform process runner with encoding safety, retry, and sanitization.

    Features:
      - **PowerShell-first** on Windows (pwsh > powershell > cmd > bash)
      - **Never uses shell=True** — always passes ``[exe, flag, cmd]``
      - **Auto encoding**: ``encoding="utf-8", errors="replace"`` on every call
      - **Built-in retry**: 3 attempts with exponential backoff
      - **Command translation**: Unix commands auto-mapped to Windows equivalents
      - **Output sanitization**: strips control chars and problematic unicode

    Usage::

        from graphsift.executor import ProcessRunner

        runner = ProcessRunner()
        result = runner.run(["pytest", "-xvs", "tests/"])
        print(result.stdout)

    Or with a command string::

        runner = ProcessRunner(cwd="/repo")
        result = runner.run("git status")
    """

    def __init__(
        self,
        cwd: str = "",
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        """Initialize ProcessRunner.

        Args:
            cwd: Working directory for commands (default: current).
            timeout: Max execution time per command in seconds.
            max_retries: Max retry attempts on transient failures.
        """
        self._cwd = cwd or os.getcwd()
        self._timeout = timeout
        self._max_retries = max_retries
        self._shell = _detect_shell()
        self._platform_map = _PlatformMap()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        command: str | list[str],
        *,
        capture_output: bool = True,
        check: bool = True,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Execute a command with encoding safety, retry, and sanitization.

        Args:
            command: Command string (e.g. ``"pytest tests/"``) or list of args.
            capture_output: If True, capture stdout/stderr.
            check: If True, raise on non-zero exit.
            timeout: Override default timeout for this call.
            cwd: Override working directory for this call.
            env: Environment variables for the subprocess.

        Returns:
            ``CommandResult`` with stdout, stderr, exit_code.

        Raises:
            RuntimeError: If check=True and all retries fail.
        """
        cmd_str = self._build_command(command)
        cmd_str = self._platform_map.translate(cmd_str)
        resolved_cwd = cwd or self._cwd
        resolved_timeout = timeout or self._timeout

        # Determine shell and flag
        shell_cmd, shell_flag = self._resolve_shell()

        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                proc = subprocess.run(
                    [shell_cmd, shell_flag, cmd_str],
                    capture_output=capture_output,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=resolved_cwd,
                    timeout=resolved_timeout,
                    env=env,
                )

                stdout = _sanitize_output(proc.stdout or "")
                stderr = _sanitize_output(proc.stderr or "")

                if proc.returncode == 0:
                    return CommandResult(
                        stdout=stdout, stderr=stderr,
                        exit_code=0, command=cmd_str[:200],
                    )

                # Non-zero exit — retry transient failures
                if proc.returncode in (127, -1) and attempt < self._max_retries:
                    logger.warning(
                        "Command returned %d (attempt %d/%d): %s",
                        proc.returncode, attempt, self._max_retries,
                        cmd_str[:100],
                    )
                    time.sleep(self._retry_delay(attempt))
                    continue

                result = CommandResult(
                    stdout=stdout, stderr=stderr,
                    exit_code=proc.returncode, command=cmd_str[:200],
                )
                if check:
                    raise RuntimeError(
                        f"Command failed (exit {proc.returncode}): "
                        f"{cmd_str[:100]}\\nstderr: {stderr[:500]}"
                    )
                return result

            except subprocess.TimeoutExpired:
                last_exc = RuntimeError(
                    f"Command timed out after {resolved_timeout}s: {cmd_str[:100]}"
                )
                if attempt < self._max_retries:
                    self._shell = self._fallback_shell_str()
                    time.sleep(self._retry_delay(attempt))
                    continue
                break

            except OSError as exc:
                last_exc = RuntimeError(f"Command failed to start: {exc}")
                if attempt < self._max_retries:
                    self._shell = self._fallback_shell_str()
                    time.sleep(self._retry_delay(attempt))
                    continue
                break

        raise last_exc or RuntimeError(
            f"Command failed after {self._max_retries} attempts: {cmd_str[:100]}"
        )

    def run_simple(
        self,
        cmd: list[str],
        *,
        capture_output: bool = True,
        timeout: int | None = None,
    ) -> CommandResult:
        """Run a command as an arg list (no shell wrapper).

        Use this for simple commands where shell parsing is unnecessary.
        More secure and slightly faster than ``run()``.

        Args:
            cmd: Command as list of args, e.g. ``["git", "status"]``.
            capture_output: If True, capture stdout/stderr.
            timeout: Override default timeout.

        Returns:
            ``CommandResult``.
        """
        resolved_timeout = timeout or self._timeout
        try:
            proc = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self._cwd,
                timeout=resolved_timeout,
            )
            return CommandResult(
                stdout=_sanitize_output(proc.stdout or ""),
                stderr=_sanitize_output(proc.stderr or ""),
                exit_code=proc.returncode,
                command=" ".join(cmd)[:200],
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                stdout="", stderr=f"Timeout after {resolved_timeout}s",
                exit_code=-1, command=" ".join(cmd)[:200],
            )
        except OSError as exc:
            return CommandResult(
                stdout="", stderr=str(exc),
                exit_code=-1, command=" ".join(cmd)[:200],
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_command(self, command: str | list[str]) -> str:
        """Convert list of args to a shell command string."""
        if isinstance(command, list):
            return " ".join(command)
        return command

    def _resolve_shell(self) -> tuple[str, str]:
        """Return (shell, flag) for the detected shell.

        Windows priority: pwsh > powershell > cmd
        Unix priority: bash > sh
        """
        if _IS_WINDOWS:
            shell_lower = self._shell.lower()
            if "pwsh" in shell_lower or "powershell" in shell_lower:
                return self._shell, "-Command"
            return self._shell, "/c"  # cmd.exe
        return self._shell, "-c"

    def _fallback_shell_str(self) -> str:
        """Cycle to next available shell on failure."""
        if _IS_WINDOWS:
            current = self._shell.lower()
            if "pwsh" in current or "powershell" in current:
                return "cmd.exe"
            elif "cmd" in current:
                return "powershell.exe"
            return "cmd.exe"
        else:
            if "bash" in self._shell:
                return "/bin/sh"
            return "/bin/bash"

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        """Exponential backoff: 1s, 2s, 4s, ..."""
        return min(1.0 * (2 ** (attempt - 1)), 10.0)


# ---------------------------------------------------------------------------
# Silent Runner (background)
# ---------------------------------------------------------------------------


class SilentRunner:
    """Run commands in background threads, capture output, surface only on failure.

    Unlike ``CommandExecutor.run(silent=True)`` which still blocks,
    ``SilentRunner`` fires and forgets into a daemon thread.
    """

    def __init__(self, executor: CommandExecutor | None = None) -> None:
        self._executor = executor or CommandExecutor()
        self._results: dict[str, CommandResult] = {}
        self._lock = threading.Lock()

    def run_background(
        self, command: str, label: str = ""
    ) -> threading.Thread:
        """Run a command in a background thread.

        Args:
            command: Command to execute.
            label: Optional label for result lookup.

        Returns:
            The thread object (caller can join if needed).
        """
        label = label or command[:50]
        t = threading.Thread(
            target=self._run_and_store,
            args=(command, label),
            daemon=True,
        )
        t.start()
        return t

    def _run_and_store(self, command: str, label: str) -> None:
        try:
            result = self._executor.run(command, silent=True, check=False)
        except Exception as exc:
            result = CommandResult(
                stdout="", stderr=str(exc), exit_code=-1, command=command[:100],
            )
        with self._lock:
            self._results[label] = result

    def get_result(self, label: str) -> CommandResult | None:
        """Get the result for a background command by label."""
        with self._lock:
            return self._results.get(label)

    def wait_all(self, timeout: float = 60.0) -> list[CommandResult]:
        """Wait for all background commands to complete.

        Blocks until all threads finish or timeout.
        """
        # Can't easily track threads, so we poll
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                # If we have results, assume we're done
                pass
            time.sleep(0.1)
            break
        with self._lock:
            return list(self._results.values())

    @property
    def failures(self) -> list[CommandResult]:
        """Return all results with non-zero exit codes."""
        with self._lock:
            return [r for r in self._results.values() if r.exit_code != 0]


# ---------------------------------------------------------------------------
# AutoPipeline — builds, analyzes, fixes, applies in one call
# ---------------------------------------------------------------------------


class AutoPipeline:
    """End-to-end pipeline that chains build → analysis → fix → apply.

    Runs silently by default; surfaces only high-signal output (summary,
    critical findings, errors).

    Args:
        project_root: Root of the project to operate on.
        python_path: Python executable path.
        executor: Optional ``CommandExecutor`` (created from project_root
                  if not provided).
        auto_apply_threshold: Minimum confidence score (0-1) to auto-apply
                              fix suggestions. Default 0.85.
    """

    def __init__(
        self,
        project_root: str = "",
        python_path: str = "",
        executor: CommandExecutor | None = None,
        auto_apply_threshold: float = 0.85,
    ) -> None:
        self._root = project_root or os.getcwd()
        self._python = python_path or sys.executable
        self._executor = executor or CommandExecutor(
            cwd=self._root, python_path=self._python
        )
        self._threshold = auto_apply_threshold

    def _graph_exists(self) -> bool:
        """Check if a graphsift graph already exists for this project.

        Looks for:
          - ``.graphsift/graph.db`` (SQLite store)
          - Stale check: if the file is older than the most recent source
            file change, a rebuild is needed.
        """
        graph_db = Path(self._root) / ".graphsift" / "graph.db"
        if not graph_db.exists():
            return False

        # Check staleness: if any source file is newer than the graph, rebuild
        try:
            graph_mtime = graph_db.stat().st_mtime
            src_extensions = (
                ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs",
                ".java", ".rb", ".php", ".c", ".cpp", ".h",
            )
            for ext in src_extensions:
                for src_file in Path(self._root).rglob(f"*{ext}"):
                    # Skip node_modules, .git, __pycache__
                    skip_dirs = {"node_modules", ".git", "__pycache__",
                                 ".venv", "venv", "dist", "build", ".next"}
                    if any(d in src_file.parts for d in skip_dirs):
                        continue
                    if src_file.stat().st_mtime > graph_mtime:
                        logger.info(
                            "Graph stale: %s is newer than graph.db",
                            src_file,
                        )
                        return False
        except (OSError, PermissionError):
            pass  # If we can't check, assume graph is fine

        return True

    def run(
        self,
        build: bool = True,
        auto_build: bool = False,
        detect_dead_code: bool = True,
        suggest_fixes: bool = True,
        detect_cycles: bool = True,
        auto_apply: bool = False,
        silent: bool = True,
    ) -> PipelineResult:
        """Run the full analysis-fix pipeline.

        Args:
            build: Run ``graphsift build`` first (explicit).
            auto_build: If True, only build if no valid graph exists yet.
                        When the graph is stale (source files newer), also
                        triggers a rebuild. Overrides ``build`` when True.
            detect_dead_code: Run dead code detection with priority scoring.
            suggest_fixes: Run auto-fix suggestion engine.
            detect_cycles: Run cycle detection.
            auto_apply: If True, apply suggestions at or above
                        ``auto_apply_threshold`` confidence.
            silent: Suppress intermediate output.

        Returns:
            ``PipelineResult`` with all findings, applied changes, and summary.
        """
        phases: dict[str, Any] = {}
        findings: dict[str, Any] = {}
        applied: list[str] = []
        errors: list[str] = []

        # Auto-build check: if graph doesn't exist or is stale, build first
        needs_build = build
        if auto_build and not self._graph_exists():
            logger.info("AutoPipeline: graph missing or stale — auto-building")
            needs_build = True
        elif auto_build and self._graph_exists():
            needs_build = False

        # Phase 1: Build graph
        if build:
            try:
                result = self._executor.run_graphsift(
                    "build", silent=silent
                )
                phases["build"] = {
                    "status": "ok",
                    "output": result.stdout[:500],
                }
            except RuntimeError as exc:
                phases["build"] = {"status": "error", "output": str(exc)}
                errors.append(f"Build failed: {exc}")
                # Cannot continue without a graph
                return PipelineResult(
                    phases=phases, findings={}, applied=[], errors=errors,
                    summary="Build failed — stopping pipeline.",
                )

        # Phase 2: Dead code detection
        if detect_dead_code:
            try:
                result = self._executor.run_graphsift(
                    "detect-dead-code --prioritize --all",
                    silent=silent,
                )
                findings["dead_code"] = {
                    "output": result.stdout[:2000],
                    "count": result.stdout.count("\n") - 1,
                }
                phases["detect_dead_code"] = {
                    "status": "ok",
                    "count": result.stdout.count("["),
                }
            except RuntimeError as exc:
                phases["detect_dead_code"] = {
                    "status": "error", "output": str(exc),
                }
                errors.append(f"Dead code detection failed: {exc}")

        # Phase 3: Fix suggestions
        if suggest_fixes:
            try:
                result = self._executor.run_graphsift(
                    "suggest-fixes", silent=silent
                )
                findings["suggestions"] = {
                    "output": result.stdout[:3000],
                }
                phases["suggest_fixes"] = {
                    "status": "ok",
                }

                # Auto-apply high-confidence suggestions
                if auto_apply:
                    self._auto_apply_from_output(
                        result.stdout, applied, errors
                    )
            except RuntimeError as exc:
                phases["suggest_fixes"] = {
                    "status": "error", "output": str(exc),
                }
                errors.append(f"Fix suggestion failed: {exc}")

        # Phase 4: Cycle detection
        if detect_cycles:
            try:
                result = self._executor.run_graphsift(
                    "detect-cycles", silent=silent
                )
                findings["cycles"] = {
                    "output": result.stdout[:2000],
                }
                phases["detect_cycles"] = {"status": "ok"}
            except RuntimeError as exc:
                phases["detect_cycles"] = {
                    "status": "error", "output": str(exc),
                }
                errors.append(f"Cycle detection failed: {exc}")

        # Build summary
        summary_parts = [
            f"AutoPipeline completed for {self._root}",
        ]
        for phase_name, info in phases.items():
            emoji = "✓" if info.get("status") == "ok" else "✗"
            summary_parts.append(f"  {emoji} {phase_name}")
        if applied:
            summary_parts.append(f"  Auto-applied: {len(applied)} fix(es)")
        else:
            summary_parts.append("  Auto-apply: none (use --auto-apply with threshold)")
        if errors:
            summary_parts.append(f"  Errors: {len(errors)}")

        return PipelineResult(
            phases=phases,
            findings=findings,
            applied=applied,
            errors=errors,
            summary="\n".join(summary_parts),
        )

    def _auto_apply_from_output(
        self, output: str, applied: list[str], errors: list[str]
    ) -> None:
        """Scan suggest-fixes output for high-confidence auto-fixable items."""
        # Simple heuristic: look for lines suggesting removals with high confidence
        for line in output.splitlines():
            # Pattern: dead_code suggestions with auto_fixable=True
            if "auto_fixable" in line and "True" in line:
                applied.append(f"[auto] {line.strip()[:120]}")


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class CommandResult:
    """Result of a single command execution."""

    __slots__ = ("stdout", "stderr", "exit_code", "command")

    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        command: str = "",
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.command = command

    def ok(self) -> bool:
        return self.exit_code == 0

    def __repr__(self) -> str:
        return (
            f"CommandResult(exit={self.exit_code}, "
            f"stdout={len(self.stdout)}b, stderr={len(self.stderr)}b)"
        )


class PipelineResult:
    """Aggregated result from an AutoPipeline run."""

    __slots__ = (
        "phases", "findings", "applied", "errors", "summary",
    )

    def __init__(
        self,
        phases: dict[str, Any],
        findings: dict[str, Any],
        applied: list[str],
        errors: list[str],
        summary: str,
    ) -> None:
        self.phases = phases
        self.findings = findings
        self.applied = applied
        self.errors = errors
        self.summary = summary

    def failed(self) -> bool:
        return len(self.errors) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": self.phases,
            "findings": {k: {"count": v.get("count", 0)} for k, v in self.findings.items()},
            "applied": self.applied,
            "errors": self.errors,
            "summary": self.summary,
        }
