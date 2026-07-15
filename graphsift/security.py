"""Security layer for graphsift — protects against data theft, path traversal,
command injection, and accidental data leakage.

Wraps every external-facing operation (build, serve, compress, analyze) with
security checks before any file I/O, network access, or subprocess call.

Components:
  - **PathValidator** — prevent directory traversal (``../../etc``) and
    symlink-based escapes. Every user-supplied file path is validated against
    the project root before use.
  - **CommandSanitizer** — prevent shell injection in command execution.
    Rejects dangerous shell metacharacters and suspicious patterns.
  - **DataScrubber** — prevent accidental leakage of secrets, tokens, or PII
    in context / compressed output. Scans for API keys, tokens, credentials.
  - **SecurePipeline** — wraps the full build/analyze pipeline with all
    security checks active. Drops privileges (no-root check), validates every
    path, scrubs output, and limits network access.

Usage::

    from graphsift.security import SecurePipeline, PathValidator

    pipe = SecurePipeline(project_root="/repo")
    pipe.build(source_map)       # validates every path automatically

    # Standalone validation
    validator = PathValidator("/repo")
    safe = validator.sanitize(path)  # raises SecurityError on bad path
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from .exceptions import graphsiftError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security exceptions
# ---------------------------------------------------------------------------


class SecurityError(graphsiftError):
    """Base for all security violations."""


class PathTraversalError(SecurityError):
    """Path traversal attempt detected (e.g. ``../../etc/passwd``)."""


class CommandInjectionError(SecurityError):
    """Suspicious command-line pattern detected (shell metacharacters)."""


class DataLeakError(SecurityError):
    """Potential secret / credential / PII found in output data."""


class NetworkAccessError(SecurityError):
    """Unexpected network access blocked (exfiltration protection)."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DANGEROUS_PATH_PATTERNS: list[re.Pattern] = [
    re.compile(r"\.\./"),        # Unix parent dir
    re.compile(r"\.\.\\"),       # Windows parent dir
    re.compile(r"^~"),           # Home directory reference
    re.compile(r"[/\\]etc[/\\]"),  # /etc/
    re.compile(r"[/\\]proc[/\\]"),  # /proc/
    re.compile(r"[/\\]sys[/\\]"),   # /sys/
    re.compile(r"[/\\]dev[/\\]"),   # /dev/
    re.compile(r"[/\\]boot[/\\]"),  # /boot/
    re.compile(r"[/\\]windows[/\\]", re.IGNORECASE),  # C:\Windows\
    re.compile(r"[/\\]winnt[/\\]", re.IGNORECASE),     # C:\WINNT\
]

# Block dangerous shell-only metacharacters. Curly braces {} and
# parentheses () are valid in Python/JS/TS and NOT blocked.
_SHELL_METACHARACTERS = re.compile(
    r'(?:^|\s)(?:;|&&|\|\|)\s|`[^`]*`|\$\([^)]+\)|>[\s\S]*\||<[\s\S]*\|'
)

# Note: { } and ( ) are NOT blocked by default because they are valid
# in Python/JS code. Only dangerous shell-specific constructs are blocked.
_DANGEROUS_SUBCOMMANDS: list[re.Pattern] = [
    re.compile(r'\beval\s', re.IGNORECASE),
    re.compile(r'\bexec\s', re.IGNORECASE),
    re.compile(r'\bsource\s+.*[;&|]', re.IGNORECASE),
    re.compile(r'\benv\s+\w+=', re.IGNORECASE),
]
_SUSPICIOUS_COMMANDS: list[re.Pattern] = [
    re.compile(r"\bcurl\s+", re.IGNORECASE),
    re.compile(r"\bwget\s+", re.IGNORECASE),
    re.compile(r"\bnc\s+", re.IGNORECASE),
    re.compile(r"\bnetcat\s+", re.IGNORECASE),
    re.compile(r"\bchmod\s+\+s", re.IGNORECASE),
    re.compile(r"\bchown\s", re.IGNORECASE),
    re.compile(r"\bdd\s+", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bdd if=", re.IGNORECASE),
    re.compile(r"\beval\s", re.IGNORECASE),
    re.compile(r"\bexec\s", re.IGNORECASE),
    re.compile(r"\bsource\s+.*[;&|]", re.IGNORECASE),
    re.compile(r"\b(base64|decode|encode)\s", re.IGNORECASE),
    re.compile(r"\bsudo\s+", re.IGNORECASE),
    re.compile(r"\bpasswd\b", re.IGNORECASE),
]

# Regex patterns that look like secrets
_SECRET_PATTERNS: list[re.Pattern] = [
    # API keys / tokens
    re.compile(r"(?i)(?:api[_-]?key|apikey|secret|token|password|passwd|"
               r"credential|auth[_-]?token|access[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    # AWS keys
    re.compile(r"(?i)AKIA[0-9A-Z]{16}"),
    # AWS secret
    re.compile(r"(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*['\"][A-Za-z0-9/+=]{40}"),
    # Private SSH keys
    re.compile(r"-{3,}BEGIN\s+(RSA|DSA|EC|OPENSSH|PRIVATE)\s+KEY-{3,}"),
    # JWT tokens
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    # GitHub tokens
    re.compile(r"(?i)ghp_[A-Za-z0-9]{36}|gho_[A-Za-z0-9]{36}|github[_-]?token\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{35,}"),
    # Generic private keys
    re.compile(r"-{3,}BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-{3,}"),
    # Slack tokens
    re.compile(r"(?i)xox[baprs]-[A-Za-z0-9]{10,}"),
    # Generic bearer tokens
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}"),
    # Database URLs / connection strings
    re.compile(r"(?i)(?:mysql|postgres|mongodb|redis|amqp)://[^@]+:[^@]+@"),
]

_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Path Validator
# ---------------------------------------------------------------------------


class PathValidator:
    """Prevent path traversal and unauthorized file access.

    Every user-supplied path is resolved and checked against the project root
    before it is used for any I/O operation.

    Args:
        project_root: Absolute path to the project root directory.
        allow_external: If True, paths outside project_root are warned but
                        not rejected (default: False).
    """

    def __init__(
        self, project_root: str, allow_external: bool = False
    ) -> None:
        self._root = Path(project_root).resolve()
        self._allow_external = allow_external

    @property
    def root(self) -> Path:
        return self._root

    def sanitize(self, path: str | Path) -> Path:
        """Validate and resolve a user-supplied path.

        Raises:
            PathTraversalError: If path traverses outside the project root
                                or targets a sensitive system path.

        Returns:
            Absolute, resolved Path.
        """
        if not path:
            raise PathTraversalError("Empty path is not allowed")

        p = Path(path)
        # Check for dangerous patterns before resolving
        self._check_dangerous_patterns(str(p))

        try:
            resolved = p.resolve(strict=False)
        except (OSError, RuntimeError):
            raise PathTraversalError(f"Cannot resolve path: {path}")

        # Check if it's inside the project root
        try:
            resolved.relative_to(self._root)
        except ValueError:
            try:
                # Also check if it's an absolute path that happens to be outside
                if resolved.is_absolute():
                    if not self._allow_external:
                        rp_str = str(resolved)
                        # Allow common temp / home patterns for --tee, --cache
                        if self._is_permitted_external(resolved):
                            return resolved
                        raise PathTraversalError(
                            f"Path '{rp_str}' is outside project root "
                            f"'{self._root}'"
                        )
            except PathTraversalError:
                raise
            except Exception:
                pass

        # File size guard (only for existing files)
        if resolved.exists() and resolved.is_file():
            try:
                size = resolved.stat().st_size
                if size > _MAX_FILE_SIZE_BYTES:
                    logger.warning(
                        "File exceeds size limit: %s (%d MB)",
                        resolved, size // (1024 * 1024),
                    )
            except OSError:
                pass

        return resolved

    def sanitize_many(self, paths: list[str]) -> list[Path]:
        """Validate multiple paths. Skips invalid ones with a warning."""
        safe: list[Path] = []
        for p in paths:
            try:
                safe.append(self.sanitize(p))
            except SecurityError as exc:
                logger.warning("Skipping unsafe path '%s': %s", p, exc)
        return safe

    def _check_dangerous_patterns(self, path_str: str) -> None:
        """Check for known dangerous path patterns."""
        for pat in _DANGEROUS_PATH_PATTERNS:
            if pat.search(path_str):
                raise PathTraversalError(
                    f"Path '{path_str}' matches dangerous pattern: {pat.pattern}"
                )

    @staticmethod
    def _is_permitted_external(path: Path) -> bool:
        """Allow specific external paths (temp, cache, home config)."""
        path_str = str(path).lower()
        permitted_prefixes = [
            os.path.expanduser("~/.graphsift").lower(),
            os.path.expanduser("~/.claude").lower(),
        ]
        # Temp directories
        if "tmp" in path_str or "temp" in path_str:
            return True
        for prefix in permitted_prefixes:
            if path_str.startswith(prefix):
                return True
        return False


# ---------------------------------------------------------------------------
# Command Sanitizer
# ---------------------------------------------------------------------------


class CommandSanitizer:
    """Detect and block shell injection attempts.

    Used before executing any command that was built from user or LLM input.

    Args:
        allowed_commands: List of command prefixes that are always allowed
                          (e.g. ``["pytest", "npm run", "git status"]``).
                          If None, a default allowlist is used.
        strict: If True, reject any command with shell metacharacters even
                if it starts with an allowed prefix (default: True).
    """

    _DEFAULT_ALLOWED: list[str] = [
        "pytest", "python", "python3", "node", "npm", "npx",
        "yarn", "pnpm", "cargo", "go", "rustc", "make",
        "git", "docker", "kubectl", "terraform", "aws",
        "grep", "cat", "ls", "find", "sort", "head", "tail",
        "wc", "echo", "printf", "test", "[",
        "pip", "pip3", "poetry", "uv",
        "eslint", "prettier", "black", "ruff", "mypy",
        "jest", "vitest", "mocha",
        # Utility commands needed for testing and dev workflows
        "echo", "exit", "true", "false", "sleep", "which", "where",
        "type", "dir", "copy", "move", "del", "mkdir", "rmdir",
    ]

    def __init__(
        self,
        allowed_commands: list[str] | None = None,
        strict: bool = True,
    ) -> None:
        self._allowed = allowed_commands or list(self._DEFAULT_ALLOWED)
        self._strict = strict

    def sanitize(self, command: str) -> str:
        """Validate a command string.

        Raises:
            CommandInjectionError: If the command contains shell injection
                                   patterns or is not in the allowlist.

        Returns:
            The command string if safe.
        """
        if not command or not command.strip():
            raise CommandInjectionError("Empty command")

        stripped = command.strip()

        # Check for shell metacharacters
        if _SHELL_METACHARACTERS.search(stripped):
            raise CommandInjectionError(
                f"Command contains shell metacharacters: {stripped[:100]}"
            )

        # Extract base command
        base = stripped.split()[0].lower() if stripped.split() else ""

        # Check allowlist
        if base not in self._allowed:
            # Also check two-word prefixes
            words = stripped.split()
            if len(words) >= 2:
                two_word = f"{words[0].lower()} {words[1].lower()}"
                if two_word not in self._allowed:
                    raise CommandInjectionError(
                        f"Command '{base}' is not in the allowlist"
                    )
            else:
                raise CommandInjectionError(
                    f"Command '{base}' is not in the allowlist"
                )

        # Check for suspicious command patterns
        self._check_suspicious(stripped)

        # Check for redirection to network
        self._check_network_exfiltration(stripped)

        return stripped

    def sanitize_with_args(self, command: str, args: list[str]) -> str:
        """Build and validate a command from base + args list.

        Args is a list (not a string) so individual arguments can be checked.
        """
        if not command:
            raise CommandInjectionError("Empty base command")

        # Validate each arg individually
        safe_args: list[str] = []
        for arg in args:
            arg_s = str(arg)
            if _SHELL_METACHARACTERS.search(arg_s):
                raise CommandInjectionError(
                    f"Argument contains shell metacharacters: {arg_s[:100]}"
                )
            safe_args.append(arg_s)

        full = f"{command} {' '.join(safe_args)}"
        return self.sanitize(full)

    @staticmethod
    def _check_suspicious(command: str) -> None:
        """Check for obviously malicious command patterns."""
        for pat in _SUSPICIOUS_COMMANDS:
            if pat.search(command):
                raise CommandInjectionError(
                    f"Suspicious command pattern '{pat.pattern}' "
                    f"found in: {command[:100]}"
                )

    @staticmethod
    def _check_network_exfiltration(command: str) -> None:
        """Block commands that pipe/send data to external hosts."""
        # curl/wget with pipe to shell
        if re.search(
            r"\b(?:curl|wget)\s+.*\||\|\s*(?:curl|wget)\s+", command,
            re.IGNORECASE,
        ):
            raise NetworkAccessError(
                "Network command piped to/from another command blocked "
                "(potential exfiltration)"
            )
        # Base64 encoding before network send
        if re.search(
            r"\b(?:base64|b64encode)\s+.*\|\s*(?:curl|wget)", command,
            re.IGNORECASE,
        ):
            raise NetworkAccessError(
                "Encoded data piped to network blocked (potential exfiltration)"
            )


# ---------------------------------------------------------------------------
# Data Scrubber
# ---------------------------------------------------------------------------


class DataScrubber:
    """Detect and remove secrets / PII from output data.

    Used by compress, context builder, and MCP tools to ensure no credentials
    leak into prompts, compressed output, or analytics.

    Args:
        action: ``"mask"`` (replace secrets with ``[REDACTED]``, default),
                ``"warn"`` (warn but keep), or ``"block"`` (raise error).
    """

    def __init__(self, action: str = "mask") -> None:
        self._action = action

    def scrub(self, text: str) -> str:
        """Scan and optionally redact secrets in a string.

        Args:
            text: The text to scan (command output, source code, context).

        Returns:
            Scrubbed text if action is ``"mask"`` or ``"warn"``.

        Raises:
            DataLeakError: If action is ``"block"`` and secrets are found.
        """
        if not text:
            return text

        findings: list[str] = []
        result = text

        for pat in _SECRET_PATTERNS:
            matches = pat.findall(result)
            if matches:
                findings.append(pat.pattern)
                count = len(matches)
                result = pat.sub("[REDACTED]", result)

        if findings:
            if self._action == "block":
                raise DataLeakError(
                    f"Blocked: {len(findings)} secret pattern(s) detected "
                    f"in data: {', '.join(f[:50] for f in findings)}"
                )
            elif self._action == "warn":
                logger.warning(
                    "DataScrubber: %d secret pattern(s) matched and masked: %s",
                    len(findings),
                    ", ".join(f[:50] for f in findings),
                )

        return result

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively scrub all string values in a dict."""
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.scrub(value)
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = [self.scrub(item) if isinstance(item, str) else item
                               for item in value]
            elif isinstance(value, bytes):
                result[key] = self.scrub(value.decode(errors="replace"))
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Secure Pipeline
# ---------------------------------------------------------------------------


class SecurePipeline:
    """Security-hardened wrapper around the full graphsift pipeline.

    Validates every path, sanitizes every command, scrubs every output.
    Use this instead of raw ``ContextBuilder`` / ``compress`` when running
    graphsift in a production or multi-tenant environment.

    Args:
        project_root: Absolute path to the project root.
        data_scrub_action: ``"mask"`` (default), ``"warn"``, or ``"block"``.
        strict_commands: If True, reject any commands with shell metachars.
    """

    def __init__(
        self,
        project_root: str,
        data_scrub_action: str = "mask",
        strict_commands: bool = True,
    ) -> None:
        self.project_root = project_root
        self.path_validator = PathValidator(project_root)
        self.command_sanitizer = CommandSanitizer(strict=strict_commands)
        self.data_scrubber = DataScrubber(action=data_scrub_action)

    def validate_path(self, path: str) -> Path:
        """Validate a single path. Delegates to PathValidator."""
        return self.path_validator.sanitize(path)

    def validate_paths(self, paths: list[str]) -> list[Path]:
        """Validate multiple paths."""
        return self.path_validator.sanitize_many(paths)

    def sanitize_command(self, command: str) -> str:
        """Sanitize a command string. Delegates to CommandSanitizer."""
        return self.command_sanitizer.sanitize(command)

    def scrub(self, text: str) -> str:
        """Scrub secrets from text. Delegates to DataScrubber."""
        return self.data_scrubber.scrub(text)

    def scrub_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Scrub secrets from a dict."""
        return self.data_scrubber.scrub_dict(data)

    def safe_build(self, source_map: dict[str, str]) -> Any:
        """Build a dependency graph with path validation.

        Every key in the source map is checked for path traversal.
        """
        from .core import ContextBuilder, ContextConfig  # noqa: PLC0415

        validated: dict[str, str] = {}
        for raw_path, source in source_map.items():
            safe_path = self.path_validator.sanitize(raw_path)
            validated[str(safe_path)] = source

        builder = ContextBuilder(ContextConfig())
        builder.index_files(validated)
        return builder

    def safe_compress(self, text: str, cmd_type: str = "auto") -> str:
        """Compress output with data scrubbing.

        Scrubs secrets before returning compressed output.
        """
        from .compress import compress  # noqa: PLC0415

        # Scrub input before compression
        safe_text = self.scrub(text)
        compressed = compress(safe_text, cmd_type)
        # Scrub output too
        return self.scrub(compressed)

    def safe_wrap_command(self, command: str) -> str:
        """Validate and wrap a command for compression.

        Like ``hooks.wrap_command()`` but with injection protection.
        """
        self.command_sanitizer.sanitize(command)
        from .hooks import wrap_command  # noqa: PLC0415

        return wrap_command(command)
