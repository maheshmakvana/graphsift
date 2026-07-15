"""Priority scoring engine for graphsift findings.

When analysis tools (dead code detection, issue finder, auto-fix) return too
many results, PriorityScorer ranks them so users know what to tackle first.

Signals used per finding:
  1. **Impact**    — How many other symbols/files reference this one.
  2. **Risk**      — Would removing it break tests, public API, or exports?
  3. **Confidence** — How sure are we that this finding is real.
  4. **Effort**     — Estimated lines of code affected (lower effort = higher
                     priority for quick wins).
  5. **Freshness**  — Recently changed code gets priority (git-aware).

Composite score = f(impact, confidence, freshness, 1/effort, 1/risk).

Tiers:
  - critical  (≥0.80) — Must fix, high blast radius.
  - high      (≥0.60) — Real issue, worth fixing soon.
  - medium    (≥0.35) — Valid but lower impact.
  - low       (<0.35) — Nice-to-have; consider deferring.

Usage::

    scorer = PriorityScorer(graph)
    ranked = scorer.score_dead_code(dead_entries, source_map=source_map)
    # ranked is a PrioritizedResult with .entries (sorted) and .summary

    # Or for any generic finding list:
    ranked = scorer.score_findings(findings)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from .models import NodeKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TIER_THRESHOLDS: list[tuple[str, float]] = [
    ("critical", 0.80),
    ("high", 0.60),
    ("medium", 0.35),
    ("low", 0.0),
]

_WEIGHTS = {
    "impact": 0.35,
    "confidence": 0.25,
    "freshness": 0.15,
    "effort_inverse": 0.15,
    "risk_inverse": 0.10,
}


# ---------------------------------------------------------------------------
# PriorityScorer
# ---------------------------------------------------------------------------


class PriorityScorer:
    """Multi-signal ranking for code-analysis findings.

    Args:
        graph: Optional DependencyGraph instance. Without it, graph-aware
               signals (impact count, freshness) fall back to defaults.
        source_map: Optional dict of file_path -> source text. Used to
                    compute token/line estimates.
    """

    def __init__(
        self,
        graph: Any | None = None,
        source_map: dict[str, str] | None = None,
    ) -> None:
        self._graph = graph
        self._source_map = source_map or {}

        # Pre-computed caches (built lazily)
        self._incoming_edge_count: dict[str, int] | None = None
        self._exported_symbols: set[str] | None = None
        self._test_files: set[str] | None = None

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score_dead_code(
        self,
        dead_entries: list[dict[str, Any]],
    ) -> PrioritizedResult:
        """Score and rank dead-code findings.

        Args:
            dead_entries: Raw output from ``DependencyGraph.find_dead_code()``
                          or ``RefactorEngine.find_dead_code()``.

        Returns:
            ``PrioritizedResult`` with tiers, sorted entries, and summary.
        """
        if not dead_entries:
            return PrioritizedResult(
                entries=[],
                tiers={},
                total=0,
                summary="No dead code found.",
            )

        scored: list[ScoredFinding] = []
        for entry in dead_entries:
            score, signals = self._score_entry(entry, finding_type="dead_code")
            scored.append(ScoredFinding(
                entry=entry,
                score=round(score, 3),
                tier=self._tier_for(score),
                signals=signals,
            ))

        # Sort by score descending
        scored.sort(key=lambda s: (-s.score, s.entry.get("name", "")))

        # Group into tiers
        tiers: dict[str, int] = defaultdict(int)
        for s in scored:
            tiers[s.tier] += 1

        total = len(scored)
        summary_lines = [
            f"Found {total} dead code symbol(s) — prioritized by impact, risk, and confidence.",
        ]
        if tiers:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(tiers.items()))
            summary_lines.append(f"Tiers: {parts}")

        cutoff = self._compute_cutoff(scored)
        if cutoff < total:
            summary_lines.append(
                f"Showing top {cutoff} of {total} — use --all to see everything."
            )

        return PrioritizedResult(
            entries=scored[:cutoff],
            all_entries=scored if cutoff >= total else scored,
            tiers=dict(tiers),
            total=total,
            summary="\n".join(summary_lines),
            truncated=cutoff < total,
            truncated_count=total - cutoff if cutoff < total else 0,
        )

    def score_fix_suggestions(
        self,
        suggestions: list[Any],
    ) -> PrioritizedResult:
        """Score and rank ``FixSuggestion`` objects.

        Args:
            suggestions: List of ``FixSuggestion`` instances (from
                         ``FixSuggester.analyze()``).

        Returns:
            ``PrioritizedResult`` with tiers, sorted entries, and summary.
        """
        if not suggestions:
            return PrioritizedResult(
                entries=[],
                tiers={},
                total=0,
                summary="No fix suggestions.",
            )

        scored: list[ScoredFinding] = []
        for s in suggestions:
            entry = {
                "file_path": s.file_path,
                "line_start": s.line_start,
                "line_end": s.line_end,
                "name": s.title,
                "kind": s.category,
                "reason": s.description,
                "confidence": s.confidence,
                "severity": s.severity.value,
                "auto_fixable": s.auto_fixable,
            }
            score, signals = self._score_entry(entry, finding_type=s.category)
            # Boost auto-fixable suggestions
            if s.auto_fixable:
                score = min(1.0, score + 0.10)
                signals["auto_fixable_bonus"] = 0.10
            scored.append(ScoredFinding(
                entry=entry,
                score=round(score, 3),
                tier=self._tier_for(score),
                signals=signals,
            ))

        scored.sort(key=lambda s: (-s.score, s.entry.get("name", "")))

        tiers: dict[str, int] = defaultdict(int)
        for s in scored:
            tiers[s.tier] += 1

        total = len(scored)
        cutoff = self._compute_cutoff(scored)

        summary_lines = [
            f"Found {total} fix suggestion(s) — prioritized by impact, confidence, and fixability.",
        ]
        if tiers:
            parts = ", ".join(
                f"{k}={v}" for k, v in sorted(tiers.items())
            )
            summary_lines.append(f"Tiers: {parts}")
        if cutoff < total:
            summary_lines.append(
                f"Showing top {cutoff} of {total} — use --all to see everything."
            )

        return PrioritizedResult(
            entries=scored[:cutoff],
            all_entries=scored if cutoff >= total else scored,
            tiers=dict(tiers),
            total=total,
            summary="\n".join(summary_lines),
            truncated=cutoff < total,
            truncated_count=total - cutoff if cutoff < total else 0,
        )

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _score_entry(
        self,
        entry: dict[str, Any],
        finding_type: str = "dead_code",
    ) -> tuple[float, dict[str, float]]:
        """Compute composite score (0-1) for a single finding entry.

        Returns (score, signal_breakdown).
        """
        signals: dict[str, float] = {}

        # 1. Impact — how many incoming edges reference this symbol
        impact = self._compute_impact(entry)
        signals["impact"] = impact

        # 2. Confidence — how sure the detector was
        confidence = self._compute_confidence(entry, finding_type)
        signals["confidence"] = confidence

        # 3. Freshness — git recency if available
        freshness = self._compute_freshness(entry)
        signals["freshness"] = freshness

        # 4. Effort inverse — smaller changes score higher
        effort_inv = self._compute_effort_inverse(entry)
        signals["effort_inverse"] = effort_inv

        # 5. Risk inverse — less risky to fix = higher priority
        risk_inv = self._compute_risk_inverse(entry)
        signals["risk_inverse"] = risk_inv

        # Composite
        score = (
            _WEIGHTS["impact"] * impact
            + _WEIGHTS["confidence"] * confidence
            + _WEIGHTS["freshness"] * freshness
            + _WEIGHTS["effort_inverse"] * effort_inv
            + _WEIGHTS["risk_inverse"] * risk_inv
        )

        return min(1.0, max(0.0, score)), signals

    def _compute_impact(self, entry: dict[str, Any]) -> float:
        """Estimate blast radius from incoming edges in the graph.

        Symbols with more callers/references are higher impact because
        removing them affects more code.
        """
        node_id = entry.get("node_id", "")
        if not node_id or self._graph is None:
            return 0.5  # neutral without graph

        # Lazily build incoming-edge counts
        if self._incoming_edge_count is None:
            self._incoming_edge_count = defaultdict(int)
            try:
                for edge in self._graph._edges:
                    self._incoming_edge_count[edge.target_id] += 1
            except Exception:
                return 0.5

        count = self._incoming_edge_count.get(node_id, 0)
        if count == 0:
            return 0.3  # truly dead — no callers
        elif count < 2:
            return 0.5
        elif count < 5:
            return 0.7
        elif count < 20:
            return 0.85
        else:
            return 1.0  # widely referenced

    def _compute_confidence(
        self, entry: dict[str, Any], finding_type: str
    ) -> float:
        """Map the finding's confidence/severity to a 0-1 scale."""
        # Use explicit confidence if present
        conf = entry.get("confidence")
        if conf is not None:
            try:
                return min(1.0, max(0.0, float(conf)))
            except (TypeError, ValueError):
                pass

        # Fall back to severity for fix suggestions
        severity = entry.get("severity", "")
        if severity == "error":
            return 0.90
        elif severity == "warning":
            return 0.65
        elif severity == "info":
            return 0.35

        # Type-specific defaults
        if finding_type == "dead_code":
            return 0.85  # graph-based dead detection is fairly reliable
        elif finding_type == "import":
            return 0.70
        elif finding_type == "type":
            return 0.80
        elif finding_type == "structure":
            return 0.60
        elif finding_type == "cycle":
            return 0.55
        return 0.65

    def _compute_freshness(self, entry: dict[str, Any]) -> float:
        """Score recency: recently modified code is higher priority.

        Without git access, returns a neutral 0.5.
        """
        file_path = entry.get("file_path", "")
        if not file_path:
            return 0.5

        # Currently stateless — in a future iteration this could check
        # git log --oneline -1 -- <file> and parse recency.
        # For now: files with 'TODO', 'FIXME', or 'HACK' nearby get a
        # slight boost as "already on the radar".
        source = self._source_map.get(file_path, "")
        if source and any(marker in source for marker in ("TODO", "FIXME", "HACK")):
            return 0.6

        return 0.5

    def _compute_effort_inverse(self, entry: dict[str, Any]) -> float:
        """Smaller changes score higher (quick wins first).

        Estimates effort from line count of the symbol.
        """
        line_start = entry.get("line_start", 0) or 0
        line_end = entry.get("line_end", 0) or line_start
        line_count = max(1, line_end - line_start)

        # Under 5 lines: effortless
        if line_count <= 5:
            return 1.0
        elif line_count <= 20:
            return 0.8
        elif line_count <= 50:
            return 0.6
        elif line_count <= 100:
            return 0.4
        elif line_count <= 300:
            return 0.2
        else:
            return 0.1

    def _compute_risk_inverse(self, entry: dict[str, Any]) -> float:
        """Lower risk of removal = higher priority.

        Heuristics:
          - Symbol is in a test file → lower risk (test-only code).
          - Symbol name starts with ``_`` → lower risk (private/internal).
          - Symbol appears in many exports → higher risk (public API).
        """
        file_path = entry.get("file_path", "")
        name = entry.get("name", "")
        kind = entry.get("kind", "")

        risk = 0.5  # neutral baseline

        # Test file → safer to remove
        if self._is_test_file(file_path):
            risk -= 0.2

        # Private convention → safer to remove
        if name.startswith("_") or name == "__module__":
            risk -= 0.15

        # Public export → riskier to remove
        if self._is_exported(name, file_path):
            risk += 0.2

        # Risk inverse: higher score = lower risk
        return 1.0 - risk

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_test_file(self, file_path: str) -> bool:
        """Heuristic: does the path look like a test file?"""
        lower = file_path.lower()
        if "/test" in lower or "\\test" in lower:
            return True
        if lower.startswith("test") or "/test_" in lower or "\\test_" in lower:
            return True
        return False

    def _is_exported(self, name: str, file_path: str) -> bool:
        """Check if a symbol is part of a public API surface.

        Uses package-level ``__init__`` / ``index`` re-exports when
        the graph is available.
        """
        # Module nodes aren't "exported" in the API sense
        if not name or name == "__module__":
            return False

        if self._graph is None:
            return False

        # Lazily build exported symbol set
        if self._exported_symbols is None:
            self._exported_symbols = set()
            try:
                for node in self._graph._nodes.values():
                    # Python __all__ or re-exported in __init__
                    if node.file_path and node.file_path.endswith(
                        ("__init__.py", "index.ts", "index.js", "index.tsx")
                    ):
                        self._exported_symbols.add(node.name)
            except Exception:
                pass

        return name in self._exported_symbols

    @staticmethod
    def _tier_for(score: float) -> str:
        """Map a composite score to a tier label."""
        for tier_name, threshold in _TIER_THRESHOLDS:
            if score >= threshold:
                return tier_name
        return "low"

    @staticmethod
    def _compute_cutoff(
        scored: list[ScoredFinding], max_entries: int = 50
    ) -> int:
        """Determine how many entries to show before truncating.

        Adaptive cutoff: always show all critical + high, then fill to
        ``max_entries`` with medium, then stop.
        """
        if len(scored) <= max_entries:
            return len(scored)

        # Count how many are critical or high
        important = sum(
            1 for s in scored if s.tier in ("critical", "high")
        )
        # Always show all critical+high, then top medium up to max_entries
        if important >= max_entries:
            return important

        # Count medium entries up to the limit
        count = important
        for s in scored:
            if s.tier == "medium":
                count += 1
                if count >= max_entries:
                    break

        return max(max_entries, count)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ScoredFinding:
    """A single finding with its composite priority score and tier."""

    __slots__ = ("entry", "score", "tier", "signals")

    def __init__(
        self,
        entry: dict[str, Any],
        score: float,
        tier: str,
        signals: dict[str, float],
    ) -> None:
        self.entry = entry
        self.score = score
        self.tier = tier
        self.signals = signals

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.entry,
            "_priority_score": self.score,
            "_priority_tier": self.tier,
            "_signals": self.signals,
        }


class PrioritizedResult:
    """Ranked output from the priority scorer.

    Attributes:
        entries: Top-K scored findings (preview, always sorted by score).
        all_entries: Complete sorted list if truncation was applied.
        tiers: Count of findings per tier {"critical": 3, "high": 5, ...}.
        total: Total number of findings scored.
        summary: Human-readable summary line(s).
        truncated: True if the result was cut for size.
        truncated_count: Number of entries hidden.
    """

    def __init__(
        self,
        entries: list[ScoredFinding],
        tiers: dict[str, int],
        total: int,
        summary: str,
        all_entries: list[ScoredFinding] | None = None,
        truncated: bool = False,
        truncated_count: int = 0,
    ) -> None:
        self.entries = entries
        self.all_entries = all_entries or entries
        self.tiers = tiers
        self.total = total
        self.summary = summary
        self.truncated = truncated
        self.truncated_count = truncated_count

    def to_dict(self, include_all: bool = False) -> dict[str, Any]:
        """Serialize to a plain dict for JSON/MCP responses."""
        data: dict[str, Any] = {
            "total": self.total,
            "summary": self.summary,
            "tiers": self.tiers,
            "truncated": self.truncated,
            "truncated_count": self.truncated_count,
            "entries": [e.to_dict() for e in self.entries],
        }
        if include_all and self.all_entries is not self.entries:
            data["entries_all"] = [e.to_dict() for e in self.all_entries]
        return data
