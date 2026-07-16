"""graphsift adapter for the Developer Daily Routine application.

Wraps graphsift's key functions with metrics tracking so we can benchmark
WITH vs WITHOUT loop-engineering. Every call records timing and token usage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from graphsift import (
    ContextBuilder,
    ContextConfig,
    DependencyGraph,
    DiffSpec,
    FixSuggester,
    compress,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class CallMetrics:
    """Metrics recorded for a single adapter call."""

    operation: str
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_saved: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        saved = f", saved={self.tokens_saved}" if self.tokens_saved else ""
        return (
            f"[{self.operation}] {self.duration_ms:.1f}ms "
            f"in={self.input_tokens} out={self.output_tokens}{saved}"
        )


class MetricsTracker:
    """Collects CallMetrics across adapter operations."""

    def __init__(self) -> None:
        self.calls: list[CallMetrics] = []

    def record(
        self,
        operation: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tokens_saved: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> CallMetrics:
        m = CallMetrics(
            operation=operation,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_saved=tokens_saved,
            extra=extra or {},
        )
        self.calls.append(m)
        return m

    @property
    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls)

    @property
    def total_duration_ms(self) -> float:
        return sum(c.duration_ms for c in self.calls)

    @property
    def total_tokens_saved(self) -> int:
        return sum(c.tokens_saved for c in self.calls)

    def summary_table(self) -> str:
        """Return a human-readable summary of all metrics."""
        lines = [
            "  Operation                         Duration    Input   Output   Saved",
            "  " + "-" * 75,
        ]
        for c in self.calls:
            lines.append(
                f"  {c.operation:<35s} {c.duration_ms:>8.1f}ms "
                f"{c.input_tokens:>6d} {c.output_tokens:>7d} {c.tokens_saved:>6d}"
            )
        lines.append("  " + "-" * 75)
        lines.append(
            f"  {'TOTAL':<35s} {self.total_duration_ms:>8.1f}ms "
            f"{self.total_tokens:>6d}  {'':>6s} {self.total_tokens_saved:>6d}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared tracker instance
# ---------------------------------------------------------------------------

_tracker = MetricsTracker()


def get_tracker() -> MetricsTracker:
    """Return the global metrics tracker instance."""
    return _tracker


def reset_tracker() -> None:
    """Reset all collected metrics."""
    _tracker.calls.clear()


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------


def compress_output(text: str, cmd_type: str = "auto") -> str:
    """Compress CLI output using graphsift's compressor.

    Args:
        text: Raw output text to compress.
        cmd_type: Command type hint (``"auto"`` = auto-detect).

    Returns:
        Compressed text.
    """
    start = time.perf_counter()
    raw_tokens = estimate_tokens(text)
    result = compress(text, command=cmd_type)
    compressed_tokens = estimate_tokens(result)
    elapsed = (time.perf_counter() - start) * 1000
    _tracker.record(
        operation="compress_output",
        duration_ms=elapsed,
        input_tokens=raw_tokens,
        output_tokens=compressed_tokens,
        tokens_saved=raw_tokens - compressed_tokens,
        extra={"cmd_type": cmd_type},
    )
    return result


def ultra_compress_output(
    text: str,
    cmd_type: str = "auto",
) -> str:
    """Compress output with ultra mode for maximum token savings.

    Uses graphsift's built-in ``ultra=True`` flag which truncates
    the result to 30 non-blank lines maximum.

    Args:
        text: Raw output text to compress.
        cmd_type: Command type hint (``"auto"`` = auto-detect).

    Returns:
        Ultra-compressed text.
    """
    start = time.perf_counter()
    raw_tokens = estimate_tokens(text)
    result = compress(text, command=cmd_type, ultra=True)
    compressed_tokens = estimate_tokens(result)
    elapsed = (time.perf_counter() - start) * 1000
    _tracker.record(
        operation="ultra_compress",
        duration_ms=elapsed,
        input_tokens=raw_tokens,
        output_tokens=compressed_tokens,
        tokens_saved=raw_tokens - compressed_tokens,
        extra={"cmd_type": cmd_type, "mode": "ultra"},
    )
    return result


def estimate_tokens_precise(text: str) -> int:
    """Estimate token count for a text string.

    Delegates to both ``graphsift.estimate_tokens`` (fast heuristic) and
    ``graphsift.analytics.estimate_tokens_precise`` (tiktoken when available)
    for comparison.

    Args:
        text: Input text.

    Returns:
        Estimated token count (fast heuristic).
    """
    from graphsift.analytics import estimate_tokens_precise as precise_token_count

    start = time.perf_counter()
    fast = estimate_tokens(text)
    precise = precise_token_count(text)
    elapsed = (time.perf_counter() - start) * 1000
    _tracker.record(
        operation="estimate_tokens",
        duration_ms=elapsed,
        input_tokens=len(text),
        output_tokens=fast,
        extra={"precise_tokens": precise, "char_len": len(text)},
    )
    return fast


def build_context(files: dict[str, str], query: str = "") -> str:
    """Build a context snippet from a set of files using graphsift ContextBuilder.

    Args:
        files: Mapping of file path -> source text.
        query: Optional query / instruction for the diff spec.

    Returns:
        Rendered context string.
    """
    start = time.perf_counter()
    raw_tokens = sum(estimate_tokens(s) for s in files.values())

    config = ContextConfig(token_budget=16_000)
    builder = ContextBuilder(config)
    for path, source in files.items():
        builder.index_file(path, source)

    diff = DiffSpec(changed_files=list(files.keys()), query=query)
    graph = DependencyGraph()
    result = builder.build(diff, source_map=files)

    output = result.rendered_context if hasattr(result, "rendered_context") else str(result)
    out_tokens = estimate_tokens(output)
    elapsed = (time.perf_counter() - start) * 1000

    _tracker.record(
        operation="build_context",
        duration_ms=elapsed,
        input_tokens=raw_tokens,
        output_tokens=out_tokens,
        tokens_saved=raw_tokens - out_tokens,
        extra={"num_files": len(files), "selected": getattr(result, "selected", 0)},
    )
    return output


def analyze_code(filepath: str, source: str) -> list[dict[str, Any]]:
    """Run graphsift FixSuggester analysis on a source file.

    Args:
        filepath: Path of the file to analyze.
        source: Source code text.

    Returns:
        List of fix suggestions as dicts.
    """
    start = time.perf_counter()
    input_tokens = estimate_tokens(source)

    graph = DependencyGraph()
    parser = _get_parser(filepath)
    if parser:
        file_node = parser.parse_file(filepath, source)
        graph.add_file(file_node)

    suggester = FixSuggester(graph, source_map={filepath: source})
    report = suggester.analyze(changed_files=[filepath])
    suggestions = report.suggestions if report else []

    elapsed = (time.perf_counter() - start) * 1000
    output = [
        {
            "id": s.suggestion_id,
            "category": s.category,
            "severity": str(s.severity),
            "title": s.title,
            "description": s.description,
            "line_start": s.line_start,
        }
        for s in suggestions
    ]
    _tracker.record(
        operation="analyze_code",
        duration_ms=elapsed,
        input_tokens=input_tokens,
        output_tokens=estimate_tokens(str(output)),
        extra={"filepath": filepath, "suggestions": len(output)},
    )
    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_parser(filepath: str) -> Any:
    """Get the appropriate parser for a file path."""
    from graphsift import get_parser, detect_language

    lang = detect_language(filepath)
    try:
        return get_parser(lang)
    except Exception:
        return None
