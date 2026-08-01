#!/usr/bin/env python3
"""Comparison benchmark: unguarded AI vs guarded AI on trading-strategy claims.

Simulates the exact failure the user hit — an LLM generating option strategies
that quotes a spectacular *backtested* profit (e.g. Rs.44,00,000) as if it were
a live/proven result, when the real-time proven P&L is a fraction of that
(e.g. Rs.4,00,000). The guard is then applied and the two sides compared.

Modes:
  --simulate   (default)  Use the built-in scenario that mirrors the real
                          44L -> 4L collapse. Deterministic, no API key.
  --live       Call a real LLM for both unguarded and guarded generations.
               Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY). The guarded
               generation uses ``graphsift guard prompt`` as the system prompt.

Outputs:
  - A markdown comparison report (stdout)
  - Optional --out report.md to save

Usage:
  python scripts/guard_benchmark.py --simulate
  python scripts/guard_benchmark.py --live --out report.md
  python scripts/guard_benchmark.py --text "custom AI strategy text here"

Exit code: 0 if the guard reduces hallucination vs unguarded, else 1.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows console is often cp1252 — force UTF-8 for stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from graphsift.guard import JsonBacktestProvider, StrategyGuard  # noqa: E402

DEFAULT_REFERENCE = Path(__file__).resolve().parent.parent / "graphsift" / "guard_data" / "angelbot_reference.json"

# ---------------------------------------------------------------------------
# Built-in scenario — mirrors the real hallucination the user described.
# ---------------------------------------------------------------------------

HALLUCINATED_STRATEGY = """STRATEGY: 2X WINNERS Iron Condor on BANKNIFTY — GUARANTEED PROFIT

This is a sure-shot, 100% safe strategy with 99% win rate and zero risk.
Buy BANKNIFTY 54500 CE and 51000 PE, sell 55000 CE and 50500 PE.
Net premium collected: Rs.44,00,000 profit in one month.
Annualized ROI of 8,500%. Backtested on 828 trades — this strategy never fails.
No loss possible. High accuracy, low risk, guaranteed returns.
"""

GROUNDED_STRATEGY = """STRATEGY: Iron Condor on BANKNIFTY — Realistic Assessment

Reference (real-time proven, paper trading): net P&L ~Rs.4,00,000 cumulative.
Live win rate ~59%, ROI ~200% (not the backtested 1313-2208%).
The 2X WINNERS backtest showed Rs.44,15,988 over 58 days, but that is a
SIMULATION — live results have been far lower. Risk: losses on trend days
(e.g. -22,581 on 2026-07-23). Do not expect guaranteed returns.
"""


# ---------------------------------------------------------------------------
# LLM generation (live mode)
# ---------------------------------------------------------------------------

def _call_llm(system: str, user: str) -> str:
    """Call an LLM via env API key. Supports Anthropic or OpenAI."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"),
            max_tokens=1500,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    if os.environ.get("OPENAI_API_KEY"):
        import openai

        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
    raise RuntimeError("--live requires ANTHROPIC_API_KEY or OPENAI_API_KEY in env")


# ---------------------------------------------------------------------------
# Audit helpers
# ---------------------------------------------------------------------------

def audit(guard: StrategyGuard, text: str) -> dict:
    report = guard.audit(text)
    data = report.to_dict()
    data["summary"] = report.summary()
    return data


def metric_summary(data: dict) -> str:
    return (
        f"score={data['hallucination_score']:.0f}/100 "
        f"(verified={data['verified']}, synthetic={data['synthetic']}, "
        f"contradicted={data['contradicted']}, unverifiable={data['unverifiable']})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--simulate", action="store_true", help="Use built-in scenario (default)")
    ap.add_argument("--live", action="store_true", help="Call a real LLM (needs API key)")
    ap.add_argument("--text", default="", help="Audit a single custom strategy text")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="Reference JSON path")
    ap.add_argument("--out", default="", help="Save markdown report to file")
    args = ap.parse_args()

    provider = JsonBacktestProvider(args.reference)
    guard = StrategyGuard(provider=provider)
    ref = provider.get_reference_stats()

    rows: list[dict] = []

    if args.text:
        rows.append({"label": "custom AI text", "text": args.text, "mode": "unguarded"})
    elif args.live:
        request = "Generate an option trading strategy for BANKNIFTY with expected profit and win rate."
        print("Calling LLM (unguarded)...")
        unguarded = _call_llm("You are a helpful trading analyst.", request)
        print("Calling LLM (guarded)...")
        guarded = _call_llm(
            guard.build_grounding_prompt(strategy_request=request),
            request,
        )
        rows.append({"label": "unguarded LLM", "text": unguarded, "mode": "unguarded"})
        rows.append({"label": "guarded LLM", "text": guarded, "mode": "guarded"})
    else:
        rows.append({"label": "unguarded AI (as delivered)", "text": HALLUCINATED_STRATEGY, "mode": "unguarded"})
        rows.append({"label": "guarded AI (after grounding prompt)", "text": GROUNDED_STRATEGY, "mode": "guarded"})

    results = []
    for row in rows:
        data = audit(guard, row["text"])
        results.append({**row, "audit": data})

    # ------------------------------------------------------------------
    # Render report
    # ------------------------------------------------------------------
    def _fmt_num(v, suffix=""):
        """Format a number robustly (int, float, or numeric string)."""
        try:
            if v is None:
                return "?"
            return f"{float(v):,.0f}{suffix}"
        except (TypeError, ValueError):
            return str(v) + suffix

    lines: list[str] = []
    lines.append("# Strategy-Guard Comparison Benchmark")
    lines.append("")
    lines.append(f"**Reference (real-time proven):** {ref.get('name', 'Angelbot')}")
    lines.append(
        f"- Live net P&L: **Rs.{_fmt_num(ref.get('net_pnl'))}** "
        f"| win rate {_fmt_num(ref.get('win_rate_pct'), '%')} "
        f"| ROI {_fmt_num(ref.get('roi_pct'), '%')}"
    )
    bt = ref.get("backtest_pnl"); bt2 = ref.get("backtest_2x_pnl")
    if bt:
        lines.append(
            f"- Backtest (NOT live-proven): ALL-23 Rs.{_fmt_num(bt)} "
            f"| 2X WINNERS Rs.{_fmt_num(bt2)}"
        )
    lines.append("")

    lines.append("| Side | Hallucination score | Verified | Synthetic(backtest) | Contradicted | Unverifiable |")
    lines.append("|---|---|---|---|---|---|")
    unguarded_audit = None
    guarded_audit = None
    for r in results:
        a = r["audit"]
        lines.append(
            f"| {r['label']} | {a['hallucination_score']:.0f}/100 | {a['verified']} "
            f"| {a['synthetic']} | {a['contradicted']} | {a['unverifiable']} |"
        )
        if r["mode"] == "unguarded":
            unguarded_audit = a
        else:
            guarded_audit = a

    if unguarded_audit and guarded_audit:
        improvement = unguarded_audit["hallucination_score"] - guarded_audit["hallucination_score"]
        lines.append("")
        lines.append("## Improvement")
        lines.append(
            f"- Hallucination score: **{unguarded_audit['hallucination_score']:.0f} → "
            f"{guarded_audit['hallucination_score']:.0f}** ({improvement:+.0f} pts)"
        )
        lines.append(
            f"- Verified claims: {unguarded_audit['verified']} → {guarded_audit['verified']} "
            f"({guarded_audit['verified'] - unguarded_audit['verified']:+d})"
        )
        lines.append(
            f"- Contradicted/fabricated claims: {unguarded_audit['contradicted']} → "
            f"{guarded_audit['contradicted']} "
            f"({guarded_audit['contradicted'] - unguarded_audit['contradicted']:+d})"
        )
        lines.append(
            f"- Risk level: {unguarded_audit['risk_level']} → {guarded_audit['risk_level']}"
        )
        lines.append("")

    lines.append("## Unguarded output (the hallucination)")
    lines.append("")
    lines.append("```")
    lines.append(next(r["text"] for r in results if r["mode"] == "unguarded"))
    lines.append("```")
    lines.append("")
    lines.append("## Claim-by-claim audit (unguarded)")
    lines.append("")
    for r in results:
        if r["mode"] != "unguarded":
            continue
        for c in r["audit"]["claims"]:
            val = f"{c['value']:,.0f}{c['unit']}" if c.get("value") is not None else "-"
            lines.append(f"- `[{c['status']}]` **{c['type']}** `{val}` `{c['raw']}` — {c.get('reason','')}")

    report = "\n".join(lines) + "\n"

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    sys.stdout.write(report)

    if args.live:
        print("\n## Raw outputs")
        for r in results:
            print(f"\n### {r['label']}\n")
            print(r["text"])

    # Exit 0 if guard helps (or no comparison available), else 1.
    if unguarded_audit and guarded_audit:
        return 0 if guarded_audit["hallucination_score"] < unguarded_audit["hallucination_score"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
