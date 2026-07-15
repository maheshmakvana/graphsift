"""Token savings analytics for graphsift — pure Python, no external deps.

Equivalent of rtk "gain" — tracks tokens saved per call, daily aggregates,
per-command breakdowns, and compression ratios. Data stored at
``~/.graphsift/analytics.json``.

Includes optional tiktoken integration for precise token counting
(lazy-loaded, requires ``tiktoken>=0.7.0``).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

_ANALYTICS_PATH = Path.home() / ".graphsift" / "analytics.json"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(project_root: str | None = None) -> dict[str, Any]:
    """Load analytics JSON, returning default structure if missing."""
    _ = project_root  # kept for API consistency; data is global
    if _ANALYTICS_PATH.exists():
        try:
            return json.loads(_ANALYTICS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "total_calls": 0,
        "total_tokens_saved": 0,
        "daily": {},
        "by_command": {},
        "sessions": [],
        "total_original_chars": 0,
        "total_compressed_chars": 0,
    }


def _save(data: dict[str, Any]) -> None:
    _ANALYTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ANALYTICS_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Tiktoken integration (lazy-loaded)
# ---------------------------------------------------------------------------

_ENCODING_CACHE: dict[str, object] = {}


def _get_encoding(model: str = "claude") -> object | None:
    """Get tiktoken encoding, returning None if tiktoken is unavailable.

    Args:
        model: ``"claude"`` or ``"gpt"`` — maps to cl100k_base encoding.

    Returns:
        tiktoken encoding object, or None.
    """
    if model in _ENCODING_CACHE:
        return _ENCODING_CACHE[model]
    try:
        import tiktoken  # noqa: PLC0415
        enc = tiktoken.get_encoding("cl100k_base")
        _ENCODING_CACHE[model] = enc
        return enc
    except ImportError:
        logger.debug("tiktoken not available — falling back to heuristic")
        return None


def estimate_tokens_precise(text: str, model: str = "claude") -> int:
    """Count tokens precisely using tiktoken (when available).

    Falls back to ``len(text) // 4`` heuristic when tiktoken is not
    installed.

    Args:
        text: Text to count tokens for.
        model: ``"claude"`` or ``"gpt"`` (both use cl100k_base).

    Returns:
        Token count.
    """
    enc = _get_encoding(model)
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))  # type: ignore[arg-type]
        except Exception:
            pass
    return max(1, len(text) // 4)


def estimate_cost(tokens: int, model: str = "claude", direction: str = "input") -> float:
    """Estimate the cost of *tokens* in USD.

    Args:
        tokens: Number of tokens.
        model: Model family (``"claude"`` or ``"gpt"``).
        direction: ``"input"`` or ``"output"``.

    Returns:
        Estimated cost in USD.
    """
    rates = {
        "claude": {"input": 3.0, "output": 15.0},   # $/M tokens (Sonnet)
        "gpt": {"input": 2.5, "output": 10.0},       # $/M tokens (GPT-4o)
    }
    rate_map = rates.get(model, rates["claude"])
    rate = rate_map.get(direction, rate_map["input"])
    return tokens * rate / 1_000_000


def token_report(data: dict[str, Any] | None = None, format: str = "text") -> str:
    """Generate a detailed token savings report with cost estimates.

    Args:
        data: Analytics data dict. If None, loads from disk.
        format: ``"text"`` or ``"json"``.

    Returns:
        Formatted report string.
    """
    if data is None:
        data = _load()

    tc = data["total_calls"]
    ts = data["total_tokens_saved"]
    avg = ts // max(tc, 1)
    est_cost_saved = estimate_cost(ts)
    orig_c = data["total_original_chars"]
    comp_c = data["total_compressed_chars"]
    ratio = (1 - comp_c / max(orig_c, 1)) * 100 if orig_c > 0 else 0.0

    cmds_sorted = sorted(
        data.get("by_command", {}).items(),
        key=lambda x: -x[1]["tokens_saved"],
    )

    if format == "json":
        return json.dumps({
            "total_calls": tc,
            "total_tokens_saved": ts,
            "avg_tokens_per_call": avg,
            "est_cost_saved_usd": round(est_cost_saved, 4),
            "compression_ratio_pct": round(ratio, 1),
            "by_command": [
                {"command": c, **s} for c, s in cmds_sorted
            ],
        }, indent=2, default=str)

    lines = [
        "graphsift Token Report",
        "=" * 50,
        f"  Total calls            : {tc}",
        f"  Tokens saved           : {ts:,}",
        f"  Avg tokens/call        : {avg:,}",
        f"  Est. cost saved        : ${est_cost_saved:.4f}",
        f"  Compression ratio      : {ratio:.1f}%"
        f"  ({comp_c:,} chars vs {orig_c:,})",
        "",
        "By command:",
    ]
    for cmd, s in cmds_sorted:
        cmd_cost = estimate_cost(s["tokens_saved"])
        lines.append(
            f"  {cmd:<20} {s['calls']:>4} calls  "
            f"{s['tokens_saved']:>10,} tokens  "
            f"${cmd_cost:.4f}"
        )
    return "\n".join(lines)


def summary_line(project_root: str | None = None) -> str:
    """Return a one-liner with cumulative token savings and estimated cost saved.

    Args:
        project_root: Ignored; kept for API consistency.

    Returns:
        A single-line string summarizing token savings and estimated cost,
        or a hint to run build/compress first if no analytics data exists yet.
    """
    if not _ANALYTICS_PATH.exists():
        return 'Run `graphsift build` or `graphsift compress` first to generate analytics.'
    data = _load(project_root)
    ts = data["total_tokens_saved"]
    est_cost = ts * 15 / 1_000_000
    return f"Tokens saved: {ts:,} | Est. cost saved: ${est_cost:.4f}"


# Lazy alias — resolves at call time so gain() is always defined
def token_report(*args: Any, **kwargs: Any) -> str:
    """Alias for gain()."""
    return gain(*args, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_call(
    tokens_saved: int = 50000,
    command_type: str = "generic",
    project_root: str | None = None,
    original_chars: int = 0,
    compressed_chars: int = 0,
) -> None:
    """Record a token-saving event.

    Args:
        tokens_saved: Number of tokens this call saved.
        command_type: Category (e.g. ``"review"``, ``"build"``).
        project_root: Ignored; kept for API consistency.
        original_chars: Source character count before compression.
        compressed_chars: Character count after compression.
    """
    data = _load(project_root)
    data["total_calls"] += 1
    data["total_tokens_saved"] += tokens_saved
    data["total_original_chars"] += original_chars
    data["total_compressed_chars"] += compressed_chars

    today = date.today().isoformat()
    daily = data.setdefault("daily", {})
    day = daily.setdefault(today, {"calls": 0, "tokens_saved": 0, "original_chars": 0, "compressed_chars": 0})
    day["calls"] += 1
    day["tokens_saved"] += tokens_saved
    day["original_chars"] += original_chars
    day["compressed_chars"] += compressed_chars

    by_cmd = data.setdefault("by_command", {})
    cmd = by_cmd.setdefault(command_type, {"calls": 0, "tokens_saved": 0, "original_chars": 0, "compressed_chars": 0})
    cmd["calls"] += 1
    cmd["tokens_saved"] += tokens_saved
    cmd["original_chars"] += original_chars
    cmd["compressed_chars"] += compressed_chars

    sessions = data.setdefault("sessions", [])
    sessions.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tokens_saved": tokens_saved,
        "command_type": command_type,
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
    })

    _save(data)


def gain(project_root: str | None = None, format: str = "text") -> str:
    """Show token savings summary.

    Args:
        project_root: Ignored; kept for API consistency.
        format: ``"text"`` for human-readable, ``"json"`` for JSON string.

    Returns:
        Formatted summary string.
    """
    data = _load(project_root)
    tc = data["total_calls"]
    ts = data["total_tokens_saved"]
    avg = ts // max(tc, 1)
    est_cost = ts * 15 / 1_000_000  # $15 per million tokens (Opus)
    orig_c = data["total_original_chars"]
    comp_c = data["total_compressed_chars"]
    ratio = (1 - comp_c / max(orig_c, 1)) * 100 if orig_c > 0 else 0.0

    days_sorted = sorted(data.get("daily", {}).items(), reverse=True)[:7]
    cmds_sorted = sorted(data.get("by_command", {}).items(), key=lambda x: -x[1]["tokens_saved"])

    result: dict[str, Any] = {
        "total_calls": tc,
        "total_tokens_saved": ts,
        "avg_tokens_per_call": avg,
        "est_cost_saved": round(est_cost, 4),
        "compression_ratio_pct": round(ratio, 1),
        "last_7_days": [
            {"date": d, **s} for d, s in days_sorted
        ],
        "by_command": [
            {"command": c, **s} for c, s in cmds_sorted
        ],
    }

    if format == "json":
        return json.dumps(result, indent=2, default=str)

    lines: list[str] = [
        "Token Savings Report (graphsift)",
        "=" * 40,
        f"  Total calls            : {tc}",
        f"  Tokens saved           : {ts:,}",
        f"  Avg tokens/call        : {avg:,}",
        f"  Est. cost saved        : ${est_cost:.2f}  (Opus $15/M tokens)",
    ]
    if orig_c > 0:
        lines.append(f"  Compression ratio      : {ratio:.1f}%  ({comp_c:,} chars vs {orig_c:,})")

    if days_sorted:
        lines.append("")
        lines.append("Last 7 days:")
        lines.append(f"  {'Date':<14} {'Calls':>6} {'Tokens':>10} {'Comp%':>7}")
        lines.append("  " + "-" * 39)
        for d, s in days_sorted:
            day_orig = s.get("original_chars", 0)
            day_comp = s.get("compressed_chars", 0)
            cr = (1 - day_comp / max(day_orig, 1)) * 100 if day_orig > 0 else 0
            lines.append(f"  {d:<14} {s['calls']:>6} {s['tokens_saved']:>10,} {cr:>6.1f}%")

    if cmds_sorted:
        lines.append("")
        lines.append("By command:")
        for c, s in cmds_sorted:
            lines.append(f"  {c:<20} {s['calls']} calls, {s['tokens_saved']:,} tokens")

    return "\n".join(lines)


def history(project_root: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Return the last N session records.

    Args:
        project_root: Ignored; kept for API consistency.
        limit: Maximum number of records to return.

    Returns:
        Dict with ``"records"`` key containing the list of session entries.
    """
    data = _load(project_root)
    sessions = data.get("sessions", [])
    return {"records": sessions[-limit:]}


def discover(project_root: str | None = None) -> dict[str, Any]:
    """Analyze usage patterns and surface missed opportunities.

    Args:
        project_root: Ignored; kept for API consistency.

    Returns:
        Dict with keys ``recent_avg_calls``, ``older_avg_calls``, ``trend``,
        and ``opportunities`` (list of suggestion strings).
    """
    data = _load(project_root)
    daily = data.get("daily", {})
    cmd_stats = data.get("by_command", {})
    opportunities: list[str] = []

    days_sorted = sorted(daily.items())
    recent = days_sorted[-7:]
    older = days_sorted[-14:-7] if len(days_sorted) >= 14 else []

    recent_avg = sum(s["calls"] for _, s in recent) / max(len(recent), 1)
    older_avg = sum(s["calls"] for _, s in older) / max(len(older), 1)

    if recent and older:
        if recent_avg > older_avg * 1.25:
            trend = "up"
            opportunities.append("Usage is trending up — consider reviewing your token budget to ensure you are not overspending.")
        elif recent_avg < older_avg * 0.75:
            trend = "down"
        else:
            trend = "flat"
    else:
        trend = "flat"

    if recent_avg < 1:
        opportunities.append("Low daily usage — graphsift saves tokens on every context build. Try using it for code reviews and impact analysis.")

    total_ts = data["total_tokens_saved"]
    total_calls = data["total_calls"]
    avg_ts = total_ts // max(total_calls, 1)
    if total_calls > 0 and avg_ts < 10000:
        opportunities.append("Average savings per call is low (<10k tokens). Your repos may be small, or you could benefit from increasing token_budget in ContextConfig.")

    used_commands = set(cmd_stats.keys())
    all_commands = {"build", "review", "impact", "update", "watch"}
    unused = all_commands - used_commands
    if unused:
        opportunities.append(f"Unused command types: {', '.join(sorted(unused))}. Try them for additional token savings.")

    return {
        "recent_avg_calls": round(recent_avg, 1),
        "older_avg_calls": round(older_avg, 1),
        "trend": trend,
        "opportunities": opportunities,
    }


def reset(project_root: str | None = None) -> None:
    """Reset all analytics data.

    Args:
        project_root: Ignored; kept for API consistency.
    """
    _ = project_root
    if _ANALYTICS_PATH.exists():
        _ANALYTICS_PATH.unlink()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(prog="graphsift.analytics", description="Token savings analytics")
    sub = parser.add_subparsers(dest="subcommand")

    gain_p = sub.add_parser("gain", help="Show token savings summary")
    gain_p.add_argument("--json", action="store_true", help="Output as JSON")
    gain_p.add_argument("--history", type=int, nargs="?", const=20, default=0, metavar="N",
                        help="Show last N session records instead of summary")

    sub.add_parser("discover", help="Analyze usage patterns and find opportunities")
    sub.add_parser("reset", help="Reset all analytics data")

    args = parser.parse_args()

    if args.subcommand == "gain":
        if args.history:
            result = history(limit=args.history)
            print(json.dumps(result, indent=2, default=str))
        else:
            print(gain(format="json" if args.json else "text"))

    elif args.subcommand == "discover":
        result = discover()
        print(json.dumps(result, indent=2, default=str))

    elif args.subcommand == "reset":
        reset()
        print("Analytics data reset.")

    else:
        # Default: print gain
        print(gain())


if __name__ == "__main__":
    _cli()
