"""Trading-strategy hallucination guard.

Detects, classifies, and neutralizes fabricated claims in AI-generated
trading/options strategy text — the "44 lakh profit" problem.

Why this exists
---------------
LLMs happily produce confident strategy write-ups with invented profit
figures, win rates, and ROI ("This Iron Condor guarantees Rs.44,00,000 in
a month!"). When asked for a real-time / proven example, the number
collapses to a fraction of the claim. This module catches that collapse
*before* the strategy reaches a user or an order router.

What it does
------------
1. **Extract** — parse strategy text into structured ``StrategyClaim``
   objects (profit, win rate, ROI, signals, price levels, guarantees).
2. **Verify** — cross-check every claim against a pluggable ``MarketDataProvider``
   (a real backtest reference, a live feed adapter, or the built-in demo
   reference). Each claim becomes ``verified`` | ``contradicted`` | ``unverifiable``.
3. **Score** — compute a ``hallucination_score`` (0-100) from how many claims
   are contradicted / unverifiable / guarantee-language.
4. **Enforce** — rewrite the text to strip or flag fabricated numbers, in
   the same MARK / STRIP / REPORT / ENFORCE spirit as ``evidence_check``.
5. **Ground** — build a hardened system prompt (``build_grounding_prompt``)
   that forces the next strategy-generation call to cite real data and label
   uncertainty instead of inventing returns.

No external dependencies. Pure Python, type-hinted.

Typical usage::

    from graphsift.guard import StrategyGuard, JsonBacktestProvider

    guard = StrategyGuard(provider=JsonBacktestProvider("reference.json"))
    report = guard.audit(ai_strategy_text)
    print(report.hallucination_score, report.contradicted_claims)

    cleaned = guard.enforce(ai_strategy_text, mode="strip")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Claim model
# ---------------------------------------------------------------------------


class ClaimType:
    """The kind of factual claim found in strategy text."""

    PROFIT = "profit"            # "Rs.44,00,000 profit"
    WIN_RATE = "win_rate"        # "99% win rate"
    ROI = "roi"                  # "1313% ROI"
    SIGNAL = "signal"            # "Buy BANKNIFTY 54000 CE"
    PRICE = "price"              # "NIFTY at 24500"
    GUARANTEE = "guarantee"      # "guaranteed", "sure shot", "100% safe"
    DURATION = "duration"        # "in one month", "over 60 days"
    BACKTEST = "backtest"        # "backtested 44 lakh profit"
    GENERIC = "generic"          # anything else numeric


class ClaimStatus:
    """Verification result for a single claim.

    The critical distinction this guard enforces:

        VERIFIED     — matches the REAL-TIME PROVEN reference figure
                       (live/paper results actually observed)
        SYNTHETIC    — provable as a BACKTEST/simulation output, but NOT
                       real-time proven. Backtests can look spectacular
                       (44 lakh) and collapse to a fraction live (4 lakh).
        CONTRADICTED — contradicts the proven reference (implausible vs real)
        UNVERIFIABLE — no way to check against real data
    """

    VERIFIED = "verified"
    SYNTHETIC = "synthetic"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


@dataclass
class StrategyClaim:
    """A single factual claim extracted from strategy text."""

    raw: str
    type: str = ClaimType.GENERIC
    value: Optional[float] = None
    unit: str = ""
    status: str = ClaimStatus.UNVERIFIABLE
    reason: str = ""
    source: str = ""

    @property
    def label(self) -> str:
        """Short human-readable label, e.g. 'profit 4400000'."""
        if self.value is not None:
            return f"{self.type} {self.value:,.0f}{self.unit}"
        return self.type

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "type": self.type,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "reason": self.reason,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Reference data / market-data provider
# ---------------------------------------------------------------------------

# Real reference drawn from Angelbot (the "real application"):
#
#   * LIVE/PAPER (real-time proven)  — what the bot actually produced in
#     paper trading. This is the ONLY baseline claims are verified against.
#     As of 2026-07-31: ~Rs.4,00,000 cumulative net P&L. Backtested figures
#     are NOT real-time proven and must not be presented as live results.
#
#   * BACKTEST (simulated, NOT proven) — ALL-23 60-day BANKNIFTY backtest
#     (58 days, May 7 → Jul 29 2026) showed Rs.26,26,665; the 2X WINNERS
#     variant showed Rs.44,15,988. These are the numbers an AI love to quote
#     as "guaranteed profit" — exactly the hallucination this guard catches.
#
# The 26L / 44L figures are kept in `backtest_pnl` so the guard can label a
# claim as SYNTHETIC (true backtest output, not real-time proven) rather than
# VERIFIED. Anything above `net_pnl` with no separate live source is
# contradicted.
_DEFAULT_REFERENCE: dict[str, Any] = {
    "name": "Angelbot paper-trading P&L (baseline)",
    "mode": "paper",
    # PROVENANCE HONESTY:
    #  - net_pnl (Rs.4,00,000) is the user-reported cumulative live/paper P&L
    #    as of 2026-07-31. It is NOT derived from a backtest file and should be
    #    re-verified against Angelbot's live tracker when it updates.
    #  - win_rate_pct / total_trades / avg_trade are the 6-day realised figures
    #    (Jul 22-29 backtest run) — a RECENT SAMPLE, treated as the live proxy.
    #    They are NOT the 60-day numbers. Do not relabel them as verified live.
    #  - roi_pct is a conservative live proxy (real returns run far below the
    #    backtest 1,313-2,208%). Treat as an estimate, not a measured fact.
    "net_pnl": 400000,
    "win_rate_pct": 59.3,
    "roi_pct": 200.0,
    "capital": 200000,
    "total_trades": 59,
    "avg_trade": 2166,
    # Simulated backtests — real output, but NOT real-time proven. EXACTLY
    # matched to Angelbot source files (2026-07-29):
    "backtest_pnl": 2626665,        # ALL-23 60-day backtest (Rs.26,26,665)
    "backtest_2x_pnl": 4415988,     # 2X WINNERS backtest (Rs.44,15,988)
    "backtest_win_rate_pct": 64.0,
    "backtest_roi_pct": 1313.3,
    "note": "PROVENANCE: net_pnl = user-reported cumulative live/paper P&L "
            "as of 2026-07-31 (re-verify against live tracker). "
            "win_rate/total_trades/avg_trade = 6-day realised sample "
            "(Jul 22-29 backtest run), treated as live proxy. "
            "roi_pct = conservative estimate. backtest_* = EXACT matches to "
            "Angelbot source files (2026-07-29). Live/paper = real-time "
            "proven. Backtest = simulated, never present as live result. If "
            "AI claims a figure matching a backtest but frames it as "
            "live/proven, it is SYNTHETIC at best and CONTRADICTED if above "
            "the live net_pnl.",
}

# Backtest-framing words → a profit claim using these is a backtest, not live.
_BACKTEST_WORDS = re.compile(
    r"\b(backtest|back-tested|simulation|simulated|paper[- ]trad|"
    r"historical|forward[- ]test|strategy tester|replay|hypothetical|"
    r"assumption|assuming|estimated|projected|expected)\b",
    re.IGNORECASE,
)

_LIVE_WORDS = re.compile(
    r"\b(live|real[- ]?time|real[- ]?account|paper trading result|"
    r"actual (traded|result|pnl)|executed|broker)\b",
    re.IGNORECASE,
)


def _nearest_framing(text: str, start: int, end: int) -> tuple[bool, bool]:
    """Return (is_backtest, is_live) based on the framing word NEAREST the number.

    Scans out to ±80 chars and keeps the closest match. If a backtest word
    and a live word are both near, the closer one wins. Prefers preceding
    words (e.g. "Backtest showed Rs.44,15,988").
    """
    num_center = (start + end) / 2
    best: tuple[str, float] | None = None  # (kind, dist)

    for kind, pat in (("backtest", _BACKTEST_WORDS), ("live", _LIVE_WORDS)):
        for m in pat.finditer(text):
            kw_center = (m.start() + m.end()) / 2
            dist = abs(kw_center - num_center)
            if m.end() <= start + 3:
                dist -= 20
            if dist > 80:
                continue
            if best is None or dist < best[1]:
                best = (kind, dist)

    if best is None:
        return False, False
    return best[0] == "backtest", best[0] == "live"


class MarketDataProvider:
    """Pluggable source of real market data for claim verification.

    Subclass and override to point at a live feed (Angel One / yfinance /
    broker API). The guard never fabricates: if a provider cannot answer,
    the claim stays ``unverifiable``.
    """

    reference: dict[str, Any] = {}

    def get_reference_stats(self) -> dict[str, Any]:
        """Return the real backtest/market summary the guard compares to."""
        return dict(self.reference)

    def verify_profit(self, value: float, *, backtest: bool = False) -> tuple[str, str]:
        """Return (status, reason) for a profit amount (in INR).

        Args:
            value: The profit amount claimed.
            backtest: True if the claim is explicitly framed as a backtest /
                simulation / assumption (e.g. "backtested 44 lakh profit").
                Backtest numbers get status SYNTHETIC — true as a simulation,
                but NOT real-time proven.
        """
        ref = self.reference
        net = ref.get("net_pnl")
        if net is None:
            return ClaimStatus.UNVERIFIABLE, "no reference net P&L available"
        if backtest:
            backtest_pnl = ref.get("backtest_pnl")
            backtest_2x = ref.get("backtest_2x_pnl")
            # Match against the known backtest figures → SYNTHETIC.
            for bt_name, bt_value in (("ALL-23 backtest", backtest_pnl),
                                      ("2X WINNERS backtest", backtest_2x)):
                if bt_value and abs(value - bt_value) <= max(1.0, bt_value * 0.05):
                    return (
                        ClaimStatus.SYNTHETIC,
                        f"matches {bt_name} figure (Rs.{bt_value:,.0f}) — a SIMULATION, "
                        f"not real-time proven; live result was only ~Rs.{net:,.0f}",
                    )
            if value > net:
                return (
                    ClaimStatus.CONTRADICTED,
                    f"backtest-framed profit of {value:,.0f} exceeds real-time proven "
                    f"net P&L ({net:,.0f}) and matches no known backtest output",
                )
            return ClaimStatus.UNVERIFIABLE, "backtest-framed claim with no matching reference backtest"
        # Non-backtest claim: compare against real-time proven figure.
        if value < net * 0.01:
            return (
                ClaimStatus.UNVERIFIABLE,
                f"value {value:,.0f} is too small to be a cumulative profit claim "
                f"(reference net P&L {net:,.0f}) — likely a per-lot premium/price, "
                "requires live feed confirmation",
            )
        if abs(value - net) <= net * 0.25:
            return ClaimStatus.VERIFIED, f"within range of real-time proven net P&L ({net:,.0f})"
        if value > net:
            return (
                ClaimStatus.CONTRADICTED,
                f"claims {value:,.0f} profit vs real-time proven {net:,.0f} "
                f"({(value / net - 1) * 100:.0f}% higher) — likely a backtest or "
                "fabricated figure presented as live result",
            )
        return ClaimStatus.VERIFIED, "below real-time proven net P&L (conservative)"

    def verify_win_rate(self, value: float) -> tuple[str, str]:
        """Return (status, reason) for a win-rate percentage."""
        ref = self.reference
        wr = ref.get("win_rate_pct")
        if wr is None:
            return ClaimStatus.UNVERIFIABLE, "no reference win rate available"
        if value > 90:
            return (
                ClaimStatus.CONTRADICTED,
                f"{value:.0f}% win rate is implausible vs reference {wr:.0f}% — "
                "near-perfect rates are an overfitting/hallucination marker",
            )
        if abs(value - wr) <= 15:
            return ClaimStatus.VERIFIED, f"within range of reference win rate ({wr:.0f}%)"
        return ClaimStatus.UNVERIFIABLE, f"differs from reference win rate ({wr:.0f}%)"

    def verify_roi(self, value: float, *, backtest: bool = False) -> tuple[str, str]:
        """Return (status, reason) for an ROI percentage.

        A backtest-framed ROI that matches the reference backtest ROI is
        SYNTHETIC (true simulation, not live-proven). Otherwise compared
        against the real-time proven ROI.
        """
        ref = self.reference
        roi = ref.get("roi_pct")
        if roi is None:
            return ClaimStatus.UNVERIFIABLE, "no reference ROI available"
        if backtest:
            bt_roi = ref.get("backtest_roi_pct")
            if bt_roi and abs(value - bt_roi) <= max(5.0, bt_roi * 0.1):
                return (
                    ClaimStatus.SYNTHETIC,
                    f"matches reference backtest ROI ({bt_roi:.0f}%) — a SIMULATION, "
                    f"not real-time proven (live ~{roi:.0f}%)",
                )
            if value > roi:
                return ClaimStatus.SYNTHETIC, "backtest-framed ROI — simulated, not live-proven"
            return ClaimStatus.UNVERIFIABLE, "backtest-framed ROI with no matching reference"
        if value > roi * 1.5:
            return (
                ClaimStatus.CONTRADICTED,
                f"ROI {value:.0f}% far exceeds real-time proven {roi:.0f}% — "
                "likely a backtest figure presented as live, or fabricated",
            )
        if abs(value - roi) <= roi * 0.5:
            return ClaimStatus.VERIFIED, f"within range of real-time proven ROI ({roi:.0f}%)"
        return ClaimStatus.UNVERIFIABLE, f"differs from real-time proven ROI ({roi:.0f}%)"

    def verify_signal(self, raw: str) -> tuple[str, str]:
        """Signals can only be verified against a live feed — never here."""
        return ClaimStatus.UNVERIFIABLE, "signal requires live feed verification; not provable from reference"

    def verify_price(self, raw: str) -> tuple[str, str]:
        return ClaimStatus.UNVERIFIABLE, "price level requires live feed verification"


def _coerce_numeric(data: dict[str, Any]) -> dict[str, Any]:
    """Convert stringified numbers in a reference dict to floats.

    Many user-authored reference JSONs quote numbers (``"net_pnl": "400000"``).
    Coerce the known numeric keys so ``:,.0f`` formatting and range checks work
    regardless of source typing.
    """
    numeric_keys = {
        "net_pnl", "win_rate_pct", "roi_pct", "capital", "total_trades",
        "avg_trade", "backtest_pnl", "backtest_2x_pnl", "backtest_win_rate_pct",
        "backtest_roi_pct",
    }
    out = dict(data)
    for key in numeric_keys:
        val = out.get(key)
        if isinstance(val, str):
            try:
                out[key] = float(val.replace(",", ""))
            except ValueError:
                pass
    return out


class JsonBacktestProvider(MarketDataProvider):
    """Load reference backtest stats from a JSON file.

    Expected shape mirrors :data:`_DEFAULT_REFERENCE`::

        {"net_pnl": 2626665, "win_rate_pct": 64.0, "roi_pct": 1313.3, ...}

    Numeric values may be JSON numbers *or* strings (``"2626665"``) — they are
    coerced to ``float`` on load so downstream formatting never crashes. If the
    file is missing or empty, the provider falls back to the built-in demo
    reference.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.reference = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path is not None and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data:
                    return _coerce_numeric(data)
            except Exception:
                pass
        return dict(_DEFAULT_REFERENCE)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        ref = self.reference
        net = ref.get("net_pnl")
        wr = ref.get("win_rate_pct")
        net_s = f"{net:,.0f}" if isinstance(net, (int, float)) else "?"
        wr_s = f"{wr:.0f}%" if isinstance(wr, (int, float)) else "?"
        return f"JsonBacktestProvider(net_pnl={net_s}, win_rate={wr_s})"


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# Indian number + currency patterns
_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")
# Currency with optional lakh/crore unit, OR a bare lakh/crore figure
# (e.g. "Rs.44,00,000", "44 lakh", "1.5 crore", "₹ 2,50,000").
_CURRENCY_RE = re.compile(
    r"(?:(?:rs\.?|inr|₹|rupees?)\s*([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore)?)"
    r"|(\d+(?:\.\d+)?)\s*(lakh|lac|crore)\b",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Guarantee / hype language — strong hallucination marker
_GUARANTEE_PATTERNS = [
    re.compile(r"\bguarantee[a-z]*\b", re.IGNORECASE),
    re.compile(r"\bsure[- ]shot\b", re.IGNORECASE),
    re.compile(r"\brisk[- ]?free\b", re.IGNORECASE),
    re.compile(r"\b100%\s*(safe|profit|return|win)\b", re.IGNORECASE),
    re.compile(r"\bno loss\b", re.IGNORECASE),
    re.compile(r"\bnever (fails?|loses?)\b", re.IGNORECASE),
    re.compile(r"\bzero (risk|loss)\b", re.IGNORECASE),
    re.compile(r"\blow[- ]risk\b", re.IGNORECASE),
    re.compile(r"\bhigh[- ](probability|accuracy)\b", re.IGNORECASE),
]

# Claim-type keywords
_PROFIT_KW = re.compile(r"\b(profit|pnl|p&l|returns?|gains?|gained|earned|earn|earnings|revenue|premium collected|made)\b", re.IGNORECASE)
_WIN_RATE_KW = re.compile(r"\bwin[- ]?rate|hit[- ]?rate|accuracy|success[- ]?rate\b", re.IGNORECASE)
_ROI_KW = re.compile(r"\broi\b|\bannualized? return\b", re.IGNORECASE)
_SIGNAL_KW = re.compile(
    r"\b(buy|sell|short|long|exit|entry)\b.*?\b(call|put|ce|pe)\b|"
    r"\b(call|put|ce|pe|futures?)\b.*?\b(buy|sell|short|long)\b",
    re.IGNORECASE,
)
_DURATION_KW = re.compile(r"\b(in|over|within|for)\s+(a\s+)?(\d+)\s*(day|week|month|year)s?\b", re.IGNORECASE)

# Negation words that flip a guarantee/hype phrase ("do NOT expect guaranteed
# returns", "no guarantee of profit") — such phrases are honest, not hype.
_NEGATION_RE = re.compile(r"\b(not|no|never|don't|dont|doesn't|wont|won't|cannot|can't|avoid|against)\b", re.IGNORECASE)


def _is_negated(text: str, pos: int) -> bool:
    """True if a negation word appears in the SAME clause before *pos*.

    The window stops at sentence boundaries (`.`, `;`, `!`, `?`, newline) so
    a "never" in a *previous* sentence does not cancel a guarantee in this
    one (e.g. "This strategy never fails. 100% safe." — "100% safe" is NOT
    negated by the earlier "never").
    """
    clause = text[max(0, pos - 60):pos]
    boundary = max(
        clause.rfind(". "), clause.rfind("; "), clause.rfind("! "),
        clause.rfind("? "), clause.rfind(".\n"), clause.rfind("\n"),
    )
    if boundary != -1:
        clause = clause[boundary + 1:]
    return bool(_NEGATION_RE.search(clause))

# Option strike pattern: "54000 CE", "BANKNIFTY 54000PE", "NIFTY 24500"
# Symbol must be UPPERCASE (tickers like BANKNIFTY/NIFTY/FINNIFTY) so English
# words like "and 52000 PE" / "sell 54500 CE" are NOT mistaken for strikes.
_STRIKE_RE = re.compile(r"\b([A-Z]{2,10})\s*(\d{4,6})\s*(CE|PE|CALL|PUT|FUT)\b")


def _parse_indian_number(text: str) -> Optional[float]:
    """Parse '12,34,567', '44 lakh', '1.5 crore' into a float."""
    m = _CURRENCY_RE.search(text)
    if not m:
        m2 = _NUMBER_RE.search(text)
        if not m2:
            return None
        return float(m2.group(0).replace(",", ""))
    # Two shapes: "Rs.44,00,000 [lakh?]" (groups 1,2) or "44 lakh" (groups 3,4).
    value_str = m.group(1) if m.group(1) is not None else m.group(3)
    unit = (m.group(2) or m.group(4) or "").lower()
    value = float(value_str.replace(",", ""))
    if unit in ("lakh", "lac"):
        return value * 1e5
    if unit == "crore":
        return value * 1e7
    return value


def _classify(text: str) -> str:
    """Classify a chunk of strategy text into a ClaimType.

    Window-based fallback: order matters. Numeric/signal types are checked
    BEFORE guarantee language so a window like "backtested Rs.44,00,000
    profit — GUARANTEED" classifies as a profit claim (guarantee words are
    captured separately by the whole-text scan). Guarantee only applies when
    nothing numeric matched.

    Prefer :func:`_classify_at` when you know the number's position — it uses
    the *nearest* keyword and avoids cross-talk between numbers in the same
    window (e.g. a profit figure next to "win rate 59%").
    """
    if _SIGNAL_KW.search(text) and _STRIKE_RE.search(text):
        return ClaimType.SIGNAL
    if _WIN_RATE_KW.search(text) and _PERCENT_RE.search(text):
        return ClaimType.WIN_RATE
    if _ROI_KW.search(text) and _PERCENT_RE.search(text):
        return ClaimType.ROI
    if _PROFIT_KW.search(text) and (_CURRENCY_RE.search(text) or _NUMBER_RE.search(text)):
        return ClaimType.PROFIT
    if _STRIKE_RE.search(text):
        return ClaimType.PRICE
    if _DURATION_KW.search(text):
        return ClaimType.DURATION
    if any(p.search(text) for p in _GUARANTEE_PATTERNS):
        return ClaimType.GUARANTEE
    if _PERCENT_RE.search(text) or _CURRENCY_RE.search(text) or _NUMBER_RE.search(text):
        return ClaimType.GENERIC
    return ClaimType.GENERIC


# Keyword → ClaimType, in priority order. Each value is a regex searched
# around the matched number.
_NEAR_KEYWORDS: list[tuple[str, re.Pattern]] = [
    (ClaimType.PROFIT, _PROFIT_KW),
    (ClaimType.WIN_RATE, _WIN_RATE_KW),
    (ClaimType.ROI, _ROI_KW),
    (ClaimType.SIGNAL, _STRIKE_RE),
    (ClaimType.DURATION, _DURATION_KW),
]


def _classify_at(text: str, start: int, end: int, *, is_percent: bool = False) -> str:
    """Classify the number at ``text[start:end]`` by its NEAREST keyword.

    Searches outward from the number for the closest keyword, so a currency
    amount (e.g. "Rs.44,00,000") is not mislabeled by a different metric
    ("win rate 59%") merely sharing the same 60-char window.
    """
    # Priority 1: the number IS a currency amount. Prefer PROFIT (or the
    # nearest metric keyword), and treat backtest-framed currency figures as
    # PROFIT so they can be verified as synthetic-backtest rather than junk.
    if not is_percent and _CURRENCY_RE.search(text[max(0, start - 60):end + 60]):
        # Backtest-framed currency amount (e.g. "Backtest showed Rs.44,15,988")
        # → PROFIT, beating duration/etc. so the synthetic-backtest label applies.
        if _BACKTEST_WORDS.search(text[max(0, start - 40):end + 20]):
            return ClaimType.PROFIT
        for ctype, pat in _NEAR_KEYWORDS:
            if ctype == ClaimType.SIGNAL:
                # only if strike pattern directly adjacent
                if pat.search(text[max(0, start - 15):end + 15]):
                    return ctype
                continue
            if pat.search(text[max(0, start - 50):end + 50]):
                return ctype
        return ClaimType.PROFIT

    # Priority 2: find the nearest keyword match to the number.
    # Prefer a keyword that PRECEDES the number (metric labels usually come
    # before their value: "win rate 59%", "ROI about 200%"), so a trailing
    # keyword from the next metric ("..., ROI about 200%") doesn't steal it.
    num_center = (start + end) / 2
    best_ctype = ClaimType.GENERIC
    best_dist = 1 << 30
    for ctype, pat in _NEAR_KEYWORDS:
        for m in pat.finditer(text):
            kw_center = (m.start() + m.end()) / 2
            preceding = m.end() <= start + 3   # keyword ends before/at the number
            dist = abs(kw_center - num_center)
            if preceding:
                dist -= 25   # strong preference for preceding labels
            if dist < best_dist and dist <= 60:
                if ctype == ClaimType.PROFIT and is_percent:
                    continue
                best_ctype = ctype
                best_dist = dist

    if best_ctype != ClaimType.GENERIC:
        return best_ctype

    # Priority 3: pure strike / duration / generic.
    if _STRIKE_RE.search(text):
        return ClaimType.PRICE
    if _DURATION_KW.search(text):
        return ClaimType.DURATION
    if any(p.search(text) for p in _GUARANTEE_PATTERNS):
        return ClaimType.GUARANTEE
    return ClaimType.GENERIC


def extract_claims(text: str) -> list[StrategyClaim]:
    """Extract structured claims from strategy text.

    Pure string analysis — no data source needed. Verification happens
    separately via a :class:`MarketDataProvider`.
    """
    if not text:
        return []
    claims: list[StrategyClaim] = []
    seen: set[str] = set()

    # Whole-text guarantee scan (skip negated phrases like "not guaranteed")
    for pat in _GUARANTEE_PATTERNS:
        for m in pat.finditer(text):
            if _is_negated(text, m.start()):
                continue
            frag = m.group(0).strip()
            if frag.lower() in seen:
                continue
            seen.add(frag.lower())
            claims.append(StrategyClaim(raw=frag, type=ClaimType.GUARANTEE, reason="guarantee/hype language"))

    # Currency amounts (Rs. 44,00,000 / 44 lakh / 1.5 crore)
    for m in _CURRENCY_RE.finditer(text):
        raw = m.group(0).strip()
        value = _parse_indian_number(raw)
        # Dedup key: use the resolved numeric value so both bare-lakh forms
        # ("44 lakh", "2 crore") produce distinct keys (the regex group is
        # None for the bare form, which would collide all bare-lakh claims).
        key = f"amt:{value:.2f}" if value is not None else raw.lower()
        if key in seen:
            continue
        seen.add(key)
        # classify by NEAREST keyword, not the whole window
        ctype = _classify_at(text, m.start(), m.end())
        claim = StrategyClaim(raw=raw, type=ctype, value=value, unit="INR")
        # Backtest vs live framing — use the NEAREST framing word so a later
        # "Live paper trading..." sentence doesn't un-flag a backtest claim.
        backtest, live = _nearest_framing(text, m.start(), m.end())
        if backtest and not live:
            claim._backtest = True
            claim.type = ClaimType.BACKTEST if ctype == ClaimType.GENERIC else ctype
        claims.append(claim)

    # Bare profit numbers without a currency marker ("profit 5000",
    # "made 15000", "gained 2,50,000"). Only when a profit keyword is near
    # AND no currency marker is adjacent (avoid double-counting
    # "Rs.44,00,000"). Candidate numbers come from _NUMBER_RE, which handles
    # both plain and Indian-grouped integers.
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).strip()
        # Skip decimals and floats — profit claims are whole rupees.
        if "." in raw:
            continue
        digits = raw.replace(",", "")
        if not digits or len(digits) < 3:
            continue
        # Skip if a currency marker (Rs./₹/lakh/crore) touches this number —
        # handled by the currency pass already.
        win = text[max(0, m.start() - 25):m.end() + 25]
        if _CURRENCY_RE.search(text[max(0, m.start() - 2):m.end() + 2]):
            continue
        if not _PROFIT_KW.search(win):
            continue
        value = float(digits)
        key = f"amt:{value:.2f}"
        if key in seen:
            continue
        seen.add(key)
        ctype = _classify_at(text, m.start(), m.end())
        claim = StrategyClaim(raw=raw, type=ctype, value=value, unit="INR")
        backtest, live = _nearest_framing(text, m.start(), m.end())
        if backtest and not live:
            claim._backtest = True
            claim.type = ClaimType.BACKTEST if ctype == ClaimType.GENERIC else ctype
        claims.append(claim)

    # Percentages (win rate / ROI / generic)
    for m in _PERCENT_RE.finditer(text):
        # Skip percentages that are part of guarantee phrases ("100% safe",
        # "100% profit", "100% win") — those are already caught as guarantees.
        if any(
            p.search(text[max(0, m.start() - 5):m.end() + 30])
            and not _is_negated(text, m.start())
            for p in _GUARANTEE_PATTERNS
        ):
            continue
        ctype = _classify_at(text, m.start(), m.end(), is_percent=True)
        if ctype not in (ClaimType.WIN_RATE, ClaimType.ROI, ClaimType.GUARANTEE):
            continue  # avoid duplicate generic percentages
        value = float(m.group(1))
        raw = m.group(0)
        if ctype == ClaimType.WIN_RATE:
            claims.append(StrategyClaim(raw=raw, type=ClaimType.WIN_RATE, value=value, unit="%"))
        elif ctype == ClaimType.ROI:
            bt, live = _nearest_framing(text, m.start(), m.end())
            claim = StrategyClaim(raw=raw, type=ClaimType.ROI, value=value, unit="%")
            if bt and not live:
                claim._backtest = True
            claims.append(claim)

    # Option strikes / signals
    for m in _STRIKE_RE.finditer(text):
        raw = m.group(0).strip()
        if raw.lower() in seen:
            continue
        seen.add(raw.lower())
        ctype = _classify_at(text, m.start(), m.end())
        claims.append(StrategyClaim(raw=raw, type=ctype, reason="option strike/signal"))

    # De-dup near-identical profit claims
    deduped: list[StrategyClaim] = []
    for c in claims:
        if any(
            c.type == d.type and c.value is not None and d.value is not None
            and abs(c.value - d.value) < max(1.0, d.value * 0.01)
            for d in deduped
        ):
            continue
        deduped.append(c)
    return deduped


# ---------------------------------------------------------------------------
# Guard report
# ---------------------------------------------------------------------------


@dataclass
class GuardReport:
    """Result of auditing strategy text for hallucinated claims."""

    text: str
    claims: list[StrategyClaim] = field(default_factory=list)
    provider: Optional[MarketDataProvider] = None

    @property
    def verified_claims(self) -> list[StrategyClaim]:
        return [c for c in self.claims if c.status == ClaimStatus.VERIFIED]

    @property
    def synthetic_claims(self) -> list[StrategyClaim]:
        """Backtest/simulation numbers presented as results — real output, not proven live."""
        return [c for c in self.claims if c.status == ClaimStatus.SYNTHETIC]

    @property
    def contradicted_claims(self) -> list[StrategyClaim]:
        return [c for c in self.claims if c.status == ClaimStatus.CONTRADICTED]

    @property
    def unverifiable_claims(self) -> list[StrategyClaim]:
        return [c for c in self.claims if c.status == ClaimStatus.UNVERIFIABLE]

    @property
    def guarantee_claims(self) -> list[StrategyClaim]:
        return [c for c in self.claims if c.type == ClaimType.GUARANTEE]

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def hallucination_score(self) -> float:
        """0-100 risk score. Higher = more likely the text is fabricated.

        - Guarantee/hype language: 30 each
        - Contradicted claims: 20 each
        - Unverifiable claims: 10 each
        - Synthetic (backtest-as-live) claims: 15 each
        - Verified claims subtract 15 each (floor 0)
        """
        score = 0.0
        score += 30.0 * len(self.guarantee_claims)
        score += 20.0 * len(self.contradicted_claims)
        score += 15.0 * len(self.synthetic_claims)
        score += 10.0 * len(self.unverifiable_claims)
        score -= 15.0 * len(self.verified_claims)
        return max(0.0, min(100.0, score))

    @property
    def risk_level(self) -> str:
        s = self.hallucination_score
        if s >= 50:
            return "HIGH — treat as fabricated until independently verified"
        if s >= 25:
            return "MEDIUM — contains claims that need real-data confirmation"
        return "LOW — claims largely consistent with real reference data"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "verified": len(self.verified_claims),
            "synthetic": len(self.synthetic_claims),
            "contradicted": len(self.contradicted_claims),
            "unverifiable": len(self.unverifiable_claims),
            "guarantee_claims": len(self.guarantee_claims),
            "hallucination_score": round(self.hallucination_score, 1),
            "risk_level": self.risk_level,
            "claims": [c.to_dict() for c in self.claims],
        }

    def summary(self) -> str:
        return (
            f"GuardReport: {self.total_claims} claims | "
            f"{len(self.verified_claims)} verified, "
            f"{len(self.synthetic_claims)} synthetic(backtest), "
            f"{len(self.contradicted_claims)} contradicted, "
            f"{len(self.unverifiable_claims)} unverifiable | "
            f"hallucination_score={self.hallucination_score:.1f}"
        )


# ---------------------------------------------------------------------------
# StrategyGuard
# ---------------------------------------------------------------------------


class StrategyGuard:
    """Audit and enforce AI-generated trading strategy text.

    Args:
        provider: A :class:`MarketDataProvider` holding real reference data.
            Defaults to ``JsonBacktestProvider()`` (built-in demo reference).
        claim_extractor: Callable(text) -> list[StrategyClaim]. Defaults to
            :func:`extract_claims`.
    """

    def __init__(
        self,
        provider: Optional[MarketDataProvider] = None,
        claim_extractor: Any = extract_claims,
    ) -> None:
        self.provider = provider or JsonBacktestProvider()
        self._extract = claim_extractor

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(self, text: str) -> GuardReport:
        """Extract and verify every claim in *text*."""
        claims = self._extract(text)
        for c in claims:
            self._verify(c)
        return GuardReport(text=text, claims=claims, provider=self.provider)

    def _verify(self, claim: StrategyClaim) -> None:
        """Fill in claim.status / claim.reason / claim.source."""
        # Detect backtest framing from the raw claim AND its neighbourhood.
        # We store a shadow attribute on the claim during extraction.
        backtest = getattr(claim, "_backtest", False)
        if claim.type == ClaimType.GUARANTEE:
            claim.status = ClaimStatus.CONTRADICTED
            claim.reason = "guarantee/hype language — impossible to verify"
            claim.source = "n/a"
            return
        if claim.type == ClaimType.SIGNAL:
            claim.status, claim.reason = self.provider.verify_signal(claim.raw)
            claim.source = "live feed required"
            return
        if claim.type == ClaimType.PRICE:
            claim.status, claim.reason = self.provider.verify_price(claim.raw)
            claim.source = "live feed required"
            return
        if claim.value is None:
            claim.status = ClaimStatus.UNVERIFIABLE
            claim.reason = "no numeric value to verify"
            return
        if claim.type in (ClaimType.PROFIT, ClaimType.BACKTEST):
            claim.status, claim.reason = self.provider.verify_profit(claim.value, backtest=backtest)
            claim.source = "reference (live) / backtest"
            return
        if claim.type == ClaimType.WIN_RATE:
            claim.status, claim.reason = self.provider.verify_win_rate(claim.value)
            claim.source = "reference (live)"
            return
        if claim.type == ClaimType.ROI:
            claim.status, claim.reason = self.provider.verify_roi(claim.value, backtest=backtest)
            claim.source = "reference (live) / backtest"
            return
        claim.status = ClaimStatus.UNVERIFIABLE
        claim.reason = "generic numeric claim — no verification rule"
        claim.source = "n/a"

    # ------------------------------------------------------------------
    # Enforce
    # ------------------------------------------------------------------

    def enforce(self, text: str, mode: str = "mark") -> tuple[str, GuardReport]:
        """Rewrite *text* to neutralize unverifiable/fabricated claims.

        Modes (mirror ``evidence_check.EnforceMode``):
            - ``mark``   — append ``[UNVERIFIED]`` / ``[CONTRADICTED]`` after risky claims
            - ``strip``  — remove risky claim text from the output
            - ``report`` — leave text unchanged, return the audit report
            - ``enforce``— strip risky claims AND return the report

        Returns ``(rewritten_text, report)``.
        """
        report = self.audit(text)
        if mode == "report":
            return text, report

        risky = report.contradicted_claims + report.unverifiable_claims
        risky = [c for c in risky if c.type != ClaimType.GENERIC]

        if mode == "mark":
            result = text
            for c in risky:
                marker = "[CONTRADICTED]" if c.status == ClaimStatus.CONTRADICTED else "[UNVERIFIED]"
                result = result.replace(c.raw, f"{c.raw} {marker}", 1)
            return result, report

        if mode in ("strip", "enforce"):
            result = text
            for c in risky:
                result = result.replace(c.raw, "", 1)
            result = re.sub(r"  +", " ", result)
            return result, report

        raise ValueError(f"unknown mode: {mode!r} (use mark/strip/report/enforce)")

    # ------------------------------------------------------------------
    # Prompt grounding
    # ------------------------------------------------------------------

    def build_grounding_prompt(
        self,
        strategy_request: str = "",
        include_reference: bool = True,
    ) -> str:
        """Build a hardened system prompt that forces grounded strategy output.

        The returned prompt is meant to be prepended to (or used as the system
        prompt for) the strategy-generation call. It tells the model to:
          - only quote real, sourced figures
          - label every number [VERIFIED-REAL] or [UNKNOWN]
          - never guarantee returns
          - provide best/expected/worst scenarios when it must estimate
        """
        ref = self.provider.get_reference_stats()
        ref_block = ""
        if include_reference and ref:
            ref_block = (
                "\n\n## REAL REFERENCE DATA (the only trusted baseline)\n"
                + json.dumps(ref, indent=2)
                + "\n"
            )

        return (
            "You are generating trading/options strategy analysis. "
            "Anti-hallucination rules are NON-NEGOTIABLE.\n"
            "\n"
            "1. **No invented numbers.** Every profit, win-rate, ROI, or price "
            "figure MUST be traceable to the REAL REFERENCE DATA below or to a "
            "live market feed you actually queried. If you do not have a number, "
            "write `[UNKNOWN]` — never fabricate.\n"
            "2. **No guarantees.** Never say 'guaranteed', 'sure shot', "
            "'100% win', 'risk-free', 'no loss'. Trading returns are not "
            "guaranteed. If the user asks for guarantees, refuse and explain why.\n"
            "3. **Source every number.** Format: `Rs.X [VERIFIED-REAL — from "
            "reference backtest]` or `Rs.X [UNKNOWN — requires live "
            "confirmation]`.\n"
            "4. **Scenario honesty.** When estimating, give best / expected / "
            "worst cases with the assumptions listed, and mark estimates "
            "[UNKNOWN]. Never present an estimate as a proven result.\n"
            "5. **Contradiction check.** If a requested figure exceeds the REAL "
            "REFERENCE DATA, say so and explain the discrepancy instead of "
            "inflating it.\n"
            "6. **Self-check before output.** Re-read your reply. Every number "
            "must have a source tag. If any number is unsourced, fix it to "
            "[UNKNOWN] before sending.\n"
            f"{ref_block}"
            "\n"
            f"## User's strategy request\n{strategy_request}".rstrip()
        )


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def guard_strategy(
    text: str,
    reference_path: str | Path | None = None,
    mode: str = "report",
) -> GuardReport | tuple[str, GuardReport]:
    """One-call guard: audit or enforce AI strategy text.

    ``mode="report"`` (default) returns a :class:`GuardReport`.
    Other modes (mark/strip/enforce) return ``(rewritten_text, report)``.
    """
    guard = StrategyGuard(provider=JsonBacktestProvider(reference_path))
    if mode == "report":
        return guard.audit(text)
    return guard.enforce(text, mode=mode)


__all__ = [
    "ClaimStatus",
    "ClaimType",
    "GuardReport",
    "JsonBacktestProvider",
    "MarketDataProvider",
    "StrategyClaim",
    "StrategyGuard",
    "extract_claims",
    "guard_strategy",
]
