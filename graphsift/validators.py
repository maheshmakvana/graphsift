"""Validators for graphsift — file, config, graph, security, and diff-spec validation.

Each validator is a self-contained class with a ``validate()`` method that
returns a :class:`ValidationReport` containing any warnings or errors found.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------


class ValidationIssue:
    """A single validation issue (warning or error)."""

    def __init__(
        self,
        message: str,
        severity: str = "warning",
        code: str | None = None,
        field: str | None = None,
    ) -> None:
        self.message = message
        self.severity = severity  # "info", "warning", "error"
        self.code = code
        self.field = field

    def __repr__(self) -> str:
        return (
            f"ValidationIssue({self.severity.upper()}: "
            f"{self.message})"
        )


class ValidationReport:
    """Result of a validation run."""

    def __init__(self) -> None:
        self._issues: list[ValidationIssue] = []

    def add(
        self,
        message: str,
        severity: str = "warning",
        code: str | None = None,
        field: str | None = None,
    ) -> None:
        self._issues.append(
            ValidationIssue(message, severity, code, field)
        )

    @property
    def issues(self) -> list[ValidationIssue]:
        return list(self._issues)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self._issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self._issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self._issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self._issues if i.severity == "warning")

    def raise_if_errors(self) -> None:
        """Raise a :class:`ValidationError` if any errors were found."""
        if self.has_errors:
            messages = [
                f"[{i.severity.upper()}] {i.message}" for i in self._issues
            ]
            raise ValidationError("\n".join(messages))

    def __repr__(self) -> str:
        return (
            f"ValidationReport("
            f"errors={self.error_count}, "
            f"warnings={self.warning_count})"
        )


# ---------------------------------------------------------------------------
# File Validator
# ---------------------------------------------------------------------------

# Common source file extensions
_SOURCE_EXTENSIONS: set[str] = {
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx",
    ".go", ".rs", ".java", ".cpp", ".cxx", ".cc", ".c", ".h",
    ".rb", ".php", ".sh", ".bash", ".zsh", ".tf", ".tfvars", ".hcl",
    ".yaml", ".yml", ".json", ".md", ".txt", ".css", ".scss", ".html",
}

_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_FILE_SIZE_WARN_BYTES = 500 * 1024   # 500 KB


class FileValidator:
    """Validate file paths, extensions, and content size limits.

    Args:
        allowed_extensions: Set of allowed file extensions. If None, a
            default set of source code extensions is used.
        max_size_bytes: Maximum allowed file size in bytes (default 10 MB).
        warn_size_bytes: Warning threshold for file size (default 500 KB).
    """

    def __init__(
        self,
        allowed_extensions: set[str] | None = None,
        max_size_bytes: int = _MAX_FILE_SIZE_BYTES,
        warn_size_bytes: int = _MAX_FILE_SIZE_WARN_BYTES,
    ) -> None:
        self._allowed_extensions = (
            allowed_extensions or _SOURCE_EXTENSIONS
        )
        self._max_size = max_size_bytes
        self._warn_size = warn_size_bytes

    def validate_path(self, path: str) -> ValidationReport:
        """Validate a single file path.

        Checks:
        - Path is not empty
        - File extension is recognized
        - No path traversal patterns
        - File exists (size check if applicable)

        Args:
            path: File path to validate.

        Returns:
            ValidationReport with any issues found.
        """
        report = ValidationReport()

        if not path or not path.strip():
            report.add("File path is empty", severity="error", code="EMPTY_PATH")
            return report

        p = Path(path)

        # Check for path traversal
        path_str = str(p).replace("\\", "/")
        if ".." in path_str.split("/"):
            report.add(
                f"Path '{path}' contains parent directory reference '..'",
                severity="error",
                code="PATH_TRAVERSAL",
            )

        # Check extension
        suffix = p.suffix.lower()
        if suffix and suffix not in self._allowed_extensions:
            report.add(
                f"File extension '{suffix}' is not in allowed list: "
                f"{sorted(self._allowed_extensions)[:10]}...",
                severity="warning",
                code="UNKNOWN_EXTENSION",
                field="path",
            )

        # Check file exists and size
        if p.exists() and p.is_file():
            try:
                size = p.stat().st_size
                if size > self._max_size:
                    report.add(
                        f"File exceeds maximum size "
                        f"({size:,} bytes > {self._max_size:,} bytes)",
                        severity="error",
                        code="FILE_TOO_LARGE",
                        field="path",
                    )
                elif size > self._warn_size:
                    report.add(
                        f"File is large ({size:,} bytes), "
                        f"may consume significant tokens",
                        severity="warning",
                        code="FILE_LARGE",
                        field="path",
                    )
            except OSError as exc:
                report.add(
                    f"Cannot stat file: {exc}",
                    severity="warning",
                    code="FILE_STAT_ERROR",
                    field="path",
                )

        return report

    def validate_paths(self, paths: list[str]) -> ValidationReport:
        """Validate multiple file paths.

        Args:
            paths: List of file paths to validate.

        Returns:
            Combined ValidationReport.
        """
        report = ValidationReport()
        for path in paths:
            sub = self.validate_path(path)
            for issue in sub.issues:
                report.add(
                    issue.message,
                    severity=issue.severity,
                    code=issue.code,
                    field=issue.field or path,
                )
        return report

    def validate_content_size(self, content: str, path: str = "") -> ValidationReport:
        """Validate the size of file content (in characters/bytes).

        Args:
            content: File content as string.
            path: Optional file path for context in error messages.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        size = len(content.encode("utf-8"))

        if size > self._max_size:
            label = path or "content"
            report.add(
                f"{label} exceeds maximum size "
                f"({size:,} bytes > {self._max_size:,} bytes)",
                severity="error",
                code="CONTENT_TOO_LARGE",
            )
        elif size > self._warn_size:
            label = path or "content"
            report.add(
                f"{label} is large ({size:,} bytes)",
                severity="warning",
                code="CONTENT_LARGE",
            )

        return report


# ---------------------------------------------------------------------------
# Config Validator
# ---------------------------------------------------------------------------


class ConfigValidator:
    """Validate :class:`~graphsift.models.ContextConfig` for consistency.

    Checks:
    - hot_threshold > warm_threshold (tier ordering)
    - token_budget within reasonable range
    - max_depth within bounds
    - compression_ratio consistent with compress_low_score
    - smart_threshold consistent with hot/warm thresholds
    """

    def validate(self, config: Any) -> ValidationReport:
        """Validate a ContextConfig instance or dict.

        Args:
            config: A ContextConfig instance or dict with config fields.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        # Extract fields
        if hasattr(config, "model_dump"):
            cfg = config.model_dump()
        elif isinstance(config, dict):
            cfg = config
        else:
            report.add(
                "Config must be a ContextConfig instance or dict",
                severity="error",
                code="INVALID_CONFIG_TYPE",
            )
            return report

        # hot_threshold > warm_threshold
        hot = cfg.get("hot_threshold", 0.8)
        warm = cfg.get("warm_threshold", 0.25)
        if hot <= warm:
            report.add(
                f"hot_threshold ({hot}) must be greater than "
                f"warm_threshold ({warm})",
                severity="error",
                code="THRESHOLD_ORDER",
                field="hot_threshold",
            )

        # smart_threshold should be between warm and hot
        smart = cfg.get("smart_threshold", 0.5)
        if not (warm <= smart <= hot):
            report.add(
                f"smart_threshold ({smart}) should be between "
                f"warm_threshold ({warm}) and hot_threshold ({hot})",
                severity="warning",
                code="SMART_THRESHOLD_RANGE",
                field="smart_threshold",
            )

        # token_budget
        budget = cfg.get("token_budget", 80_000)
        if budget < 100:
            report.add(
                f"token_budget ({budget}) is too low (minimum 100)",
                severity="error",
                code="BUDGET_TOO_LOW",
                field="token_budget",
            )
        elif budget > 1_000_000:
            report.add(
                f"token_budget ({budget:,}) is very large; "
                f"most LLM context windows are 200K or less",
                severity="warning",
                code="BUDGET_VERY_LARGE",
                field="token_budget",
            )

        # max_depth
        depth = cfg.get("max_depth", 4)
        if depth < 1:
            report.add(
                f"max_depth ({depth}) must be at least 1",
                severity="error",
                code="DEPTH_TOO_LOW",
                field="max_depth",
            )
        elif depth > 10:
            report.add(
                f"max_depth ({depth}) is very high; "
                f"this may include many unrelated files",
                severity="warning",
                code="DEPTH_VERY_HIGH",
                field="max_depth",
            )

        # compression_ratio consistency
        compress_low = cfg.get("compress_low_score", True)
        comp_ratio = cfg.get("compression_ratio", 0.35)
        if compress_low and (comp_ratio < 0.1 or comp_ratio > 1.0):
            report.add(
                f"compression_ratio ({comp_ratio}) should be "
                f"between 0.1 and 1.0 when compress_low_score is enabled",
                severity="warning",
                code="COMPRESSION_RATIO_RANGE",
                field="compression_ratio",
            )

        # min_score
        min_score = cfg.get("min_score", 0.1)
        if not (0.0 <= min_score <= 1.0):
            report.add(
                f"min_score ({min_score}) must be between 0.0 and 1.0",
                severity="error",
                code="MIN_SCORE_RANGE",
                field="min_score",
            )

        # cache_ttl_days
        ttl = cfg.get("cache_ttl_days", 7)
        if ttl < 1 or ttl > 365:
            report.add(
                f"cache_ttl_days ({ttl}) must be between 1 and 365",
                severity="error",
                code="TTL_RANGE",
                field="cache_ttl_days",
            )

        # trimming_context_lines
        trim_lines = cfg.get("trimming_context_lines", 10)
        if trim_lines < 0 or trim_lines > 100:
            report.add(
                f"trimming_context_lines ({trim_lines}) must be "
                f"between 0 and 100",
                severity="error",
                code="TRIM_LINES_RANGE",
                field="trimming_context_lines",
            )

        return report


# ---------------------------------------------------------------------------
# Graph Validator
# ---------------------------------------------------------------------------


class GraphValidator:
    """Validate graph integrity: no orphan edges, no self-loops.

    Args:
        allow_self_loops: If True, self-loop edges are not flagged.
        allow_orphan_edges: If True, edges referencing missing nodes are
            not flagged.
    """

    def __init__(
        self,
        allow_self_loops: bool = False,
        allow_orphan_edges: bool = False,
    ) -> None:
        self._allow_self_loops = allow_self_loops
        self._allow_orphan_edges = allow_orphan_edges

    def validate(
        self,
        nodes: list[Any],
        edges: list[Any],
    ) -> ValidationReport:
        """Validate graph structure.

        Args:
            nodes: List of node objects (must have ``node_id`` attribute).
            edges: List of edge objects (must have ``source_id``, ``target_id``).

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        # Build node ID set
        node_ids: set[str] = set()
        for i, node in enumerate(nodes):
            nid = getattr(node, "node_id", None) or (
                node.get("node_id") if isinstance(node, dict) else None
            )
            if nid is None:
                report.add(
                    f"Node at index {i} has no node_id",
                    severity="error",
                    code="NODE_MISSING_ID",
                )
            else:
                if nid in node_ids:
                    report.add(
                        f"Duplicate node_id: {nid}",
                        severity="warning",
                        code="DUPLICATE_NODE",
                        field=nid,
                    )
                node_ids.add(nid)

        # Validate edges
        for j, edge in enumerate(edges):
            source = getattr(edge, "source_id", None) or (
                edge.get("source_id") if isinstance(edge, dict) else None
            )
            target = getattr(edge, "target_id", None) or (
                edge.get("target_id") if isinstance(edge, dict) else None
            )

            if source is None or target is None:
                report.add(
                    f"Edge at index {j} missing source_id or target_id",
                    severity="error",
                    code="EDGE_MISSING_IDS",
                )
                continue

            # Self-loop check
            if source == target and not self._allow_self_loops:
                report.add(
                    f"Self-loop edge: {source} -> {target}",
                    severity="warning",
                    code="SELF_LOOP",
                    field=source,
                )

            # Orphan edge check
            if not self._allow_orphan_edges:
                if source not in node_ids:
                    report.add(
                        f"Orphan edge: source '{source}' not found in nodes",
                        severity="error",
                        code="ORPHAN_SOURCE",
                        field=source,
                    )
                if target not in node_ids:
                    report.add(
                        f"Orphan edge: target '{target}' not found in nodes",
                        severity="error",
                        code="ORPHAN_TARGET",
                        field=target,
                    )

        return report


# ---------------------------------------------------------------------------
# Security Validator
# ---------------------------------------------------------------------------

# Path traversal patterns
_TRAVERSAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\.\.[/\\]"),        # Unix and Windows parent dir
    re.compile(r"^~[/\\]"),          # Home directory reference
    re.compile(r"[/\\]etc[/\\]"),    # /etc/
    re.compile(r"[/\\]proc[/\\]"),   # /proc/
    re.compile(r"[/\\]sys[/\\]"),    # /sys/
    re.compile(r"[/\\]windows[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]winnt[/\\]", re.IGNORECASE),
    re.compile(r"[/\\]boot[/\\]"),   # /boot/
    re.compile(r"[/\\]dev[/\\]"),    # /dev/
]


class SecurityValidator:
    """Validate paths against traversal attacks and token budget limits.

    Args:
        project_root: The allowed project root directory.
    """

    def __init__(self, project_root: str | None = None) -> None:
        self._root = (
            Path(project_root).resolve()
            if project_root
            else None
        )

    def validate_path(self, path: str) -> ValidationReport:
        """Validate a single path against traversal attacks.

        Args:
            path: File path to validate.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        if not path or not path.strip():
            report.add(
                "Path is empty",
                severity="error",
                code="EMPTY_PATH",
            )
            return report

        # Check dangerous patterns
        path_str = str(path).replace("\\", "/")
        for pat in _TRAVERSAL_PATTERNS:
            if pat.search(path_str):
                report.add(
                    f"Path '{path}' matches traversal pattern: {pat.pattern}",
                    severity="error",
                    code="PATH_TRAVERSAL",
                )
                break

        # Check against project root
        if self._root is not None:
            try:
                resolved = Path(path).resolve(strict=False)
                resolved.relative_to(self._root)
            except ValueError:
                report.add(
                    f"Path '{path}' is outside project root '{self._root}'",
                    severity="error",
                    code="PATH_OUTSIDE_ROOT",
                )
            except (OSError, RuntimeError):
                report.add(
                    f"Cannot resolve path: {path}",
                    severity="warning",
                    code="PATH_RESOLVE_ERROR",
                )

        return report

    def validate_paths(self, paths: list[str]) -> ValidationReport:
        """Validate multiple paths.

        Args:
            paths: List of paths to validate.

        Returns:
            Combined ValidationReport.
        """
        report = ValidationReport()
        for path in paths:
            sub = self.validate_path(path)
            for issue in sub.issues:
                report.add(
                    issue.message,
                    severity=issue.severity,
                    code=issue.code,
                    field=path,
                )
        return report

    @staticmethod
    def validate_token_budget(
        token_budget: int,
        max_recommended: int = 200_000,
    ) -> ValidationReport:
        """Validate a token budget value.

        Args:
            token_budget: The token budget to validate.
            max_recommended: Maximum recommended budget.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        if token_budget < 100:
            report.add(
                f"Token budget ({token_budget}) is too low (minimum 100)",
                severity="error",
                code="BUDGET_TOO_LOW",
            )
        elif token_budget < 1_000:
            report.add(
                f"Token budget ({token_budget}) is very small; "
                f"unlikely to produce useful context",
                severity="warning",
                code="BUDGET_VERY_SMALL",
            )
        elif token_budget > max_recommended:
            report.add(
                f"Token budget ({token_budget:,}) exceeds recommended "
                f"maximum ({max_recommended:,})",
                severity="warning",
                code="BUDGET_EXCEEDS_RECOMMENDED",
            )

        return report


# ---------------------------------------------------------------------------
# DiffSpec Validator
# ---------------------------------------------------------------------------


class DiffSpecValidator:
    """Validate :class:`~graphsift.models.DiffSpec`.

    Checks:
    - changed_files is not empty
    - each changed_file path is valid
    - diff format is recognizable (if provided)
    - commit_message is not excessively long
    """

    @staticmethod
    def validate_changed_files(
        changed_files: list[str],
    ) -> ValidationReport:
        """Validate the changed_files list.

        Args:
            changed_files: List of changed file paths.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        if not changed_files:
            report.add(
                "changed_files is empty; at least one file is required",
                severity="error",
                code="NO_CHANGED_FILES",
            )
            return report

        for path in changed_files:
            if not path or not path.strip():
                report.add(
                    "Empty path in changed_files",
                    severity="error",
                    code="EMPTY_CHANGED_FILE",
                )

        # Check for duplicates
        seen: set[str] = set()
        for path in changed_files:
            if path in seen:
                report.add(
                    f"Duplicate changed_file: '{path}'",
                    severity="warning",
                    code="DUPLICATE_CHANGED_FILE",
                    field=path,
                )
            seen.add(path)

        return report

    @staticmethod
    def validate_diff_format(diff_text: str) -> ValidationReport:
        """Validate the diff format.

        Checks for recognizable unified diff markers (``---``, ``+++``, ``@@``).

        Args:
            diff_text: The diff text to validate.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        if not diff_text or not diff_text.strip():
            return report

        lines = diff_text.splitlines()
        has_marker = False

        for line in lines[:20]:  # Check first 20 lines
            if line.startswith("--- ") or line.startswith("+++ "):
                has_marker = True
                break

        # Check for @@ hunk headers
        has_hunk = any(line.startswith("@@") for line in lines[:50])

        if not has_marker and not has_hunk:
            report.add(
                "Diff text does not appear to be a standard unified diff "
                "(missing ---/+++ markers or @@ hunk headers)",
                severity="warning",
                code="UNRECOGNIZED_DIFF_FORMAT",
            )

        # Check for truncated diff
        if len(lines) > 5000:
            report.add(
                f"Diff text is very long ({len(lines)} lines); "
                f"may exceed token budget",
                severity="warning",
                code="DIFF_TOO_LONG",
            )

        return report

    @staticmethod
    def validate_commit_message(commit_message: str) -> ValidationReport:
        """Validate the commit message length.

        Args:
            commit_message: The commit message to validate.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        if len(commit_message) > 10_000:
            report.add(
                f"Commit message is very long "
                f"({len(commit_message)} characters)",
                severity="warning",
                code="COMMIT_MESSAGE_LONG",
            )

        return report

    def validate(self, diff_spec: Any) -> ValidationReport:
        """Validate a complete DiffSpec.

        Args:
            diff_spec: A DiffSpec instance or dict.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()

        if hasattr(diff_spec, "model_dump"):
            spec = diff_spec.model_dump()
        elif isinstance(diff_spec, dict):
            spec = diff_spec
        else:
            report.add(
                "DiffSpec must be a DiffSpec instance or dict",
                severity="error",
                code="INVALID_DIFFSPEC_TYPE",
            )
            return report

        # Validate components
        changed = spec.get("changed_files", [])
        sub = self.validate_changed_files(changed)
        for iss in sub.issues:
            report.add(iss.message, iss.severity, iss.code, iss.field)

        diff_text = spec.get("diff_text", "")
        sub = self.validate_diff_format(diff_text)
        for iss in sub.issues:
            report.add(iss.message, iss.severity, iss.code, iss.field)

        commit_msg = spec.get("commit_message", "")
        sub = self.validate_commit_message(commit_msg)
        for iss in sub.issues:
            report.add(iss.message, iss.severity, iss.code, iss.field)

        return report


# ---------------------------------------------------------------------------
# Composite Validator
# ---------------------------------------------------------------------------


class CompositeValidator:
    """Runs multiple validators and aggregates results.

    Usage::

        from graphsift.validators import (
            CompositeValidator,
            ConfigValidator,
            DiffSpecValidator,
        )

        validator = CompositeValidator(
            ConfigValidator(),
            DiffSpecValidator(),
        )
        report = validator.validate(config, diff_spec)
        report.raise_if_errors()
    """

    def __init__(self, *validators: Any) -> None:
        self._validators = list(validators)

    def add(self, validator: Any) -> None:
        """Add a validator to the chain."""
        self._validators.append(validator)

    def validate(self, *args: Any, **kwargs: Any) -> ValidationReport:
        """Run all validators in sequence.

        Each validator's ``validate()`` method is called with the provided
        arguments. Results are aggregated.

        Returns:
            Combined ValidationReport.
        """
        report = ValidationReport()
        for validator in self._validators:
            try:
                sub = validator.validate(*args, **kwargs)
                for issue in sub.issues:
                    report.add(
                        issue.message,
                        severity=issue.severity,
                        code=issue.code,
                        field=issue.field,
                    )
            except Exception as exc:
                report.add(
                    f"Validator {type(validator).__name__} raised: {exc}",
                    severity="error",
                    code="VALIDATOR_ERROR",
                )
        return report
