"""Evidence and audit trail for context selection decisions.

Records WHY each file was selected or excluded during the graphsift context
building pipeline.  Addresses the 2026 RAGA paper's emphasis on
evidence-anchored verification — every selection decision is traceable
back to its scoring signals, graph paths, and budget constraints.

Architecture::

    EvidenceTracer          — wraps ContextBuilder, records per-file rationale
    EvidenceResult          — enhanced ContextResult with evidence for every file
    FileEvidence            — why a specific file was included or excluded
    ScoreBreakdown          — decomposition of the relevance score into signals
    ConnectionEvidence      — how a file connects to changed files via the graph

Usage::

    from graphsift.evidence import EvidenceTracer
    from graphsift import ContextBuilder, ContextConfig, DiffSpec

    builder = ContextBuilder(ContextConfig(token_budget=50_000))
    builder.index_files(source_map)

    tracer = EvidenceTracer(builder)
    evidence = tracer.build_with_evidence(
        DiffSpec(changed_files=["src/auth.py"]),
        source_map,
    )

    print(evidence.summary())
    # Evidence Summary: 12 selected / 143 total · 3 HOT · 5 WARM · 4 COLD
    # Tokens: 11,200 / 80,000 · saved 94%

    why = evidence.find_why_not("src/deprecated.py")
    # "Score 0.120 below warm threshold 0.250"
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .core import (
    ContextBuilder,
    ContextSelector,
    RelevanceRanker,
    estimate_tokens,
)
from .exceptions import ValidationError
from .models import (
    ContextConfig,
    ContextResult,
    DiffSpec,
    EdgeKind,
    OutputMode,
    ScoredFile,
    TierLevel,
)

logger = logging.getLogger(__name__)

# Max number of excluded files to include in evidence output
_MAX_EXCLUDED = 50

# ---------------------------------------------------------------------------
# Evidence data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreBreakdown:
    """Decomposition of a file's relevance score into individual signals.

    Attributes:
        bm25_score: BM25 keyword overlap score (0–1) against query tokens.
        graph_distance_score: Graph traversal score (0–1) from changed files.
        tfidf_similarity: TF-IDF cosine similarity (0–1).  Not computed in
            the current ranker; reserved for future expansion.
        recency_boost: Recency multiplier for files modified recently.
            Not computed in the current ranker; reserved for future expansion.
        final_score: The combined final relevance score after all weights,
            bonuses, and penalties.
        weights_used: The weight values applied in scoring (e.g.
            ``{"bm25": 0.3, "graph": 0.7}``).
    """

    bm25_score: float = 0.0
    graph_distance_score: float = 0.0
    tfidf_similarity: float = 0.0
    recency_boost: float = 0.0
    final_score: float = 0.0
    weights_used: dict[str, float] = field(default_factory=lambda: {"bm25": 0.3, "graph": 0.7})


@dataclass(frozen=True)
class ConnectionEvidence:
    """How a file is connected to a changed file in the dependency graph.

    Attributes:
        from_file: The changed (seed) file path.
        to_file: The connected file path.
        edge_types: Edge kind labels along the connection path
            (e.g. ``["IMPORTS", "CALLS"]``).
        path: List of file paths forming the graph connection from
            ``from_file`` to ``to_file``.  Empty for direct connections
            (same file).
        path_length: Number of hops in the graph path (0 = direct).
        contribution_to_score: The score contribution from this connection
            path, typically ``decay ** path_length``.
    """

    from_file: str = ""
    to_file: str = ""
    edge_types: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    path_length: int = 0
    contribution_to_score: float = 0.0


@dataclass(frozen=True)
class FileEvidence:
    """Evidence record for a single file in the selection pipeline.

    Attributes:
        filepath: Absolute or repo-relative file path.
        selected: Whether this file was included in the final context.
        tier: Selection tier — ``"hot"``, ``"warm"``, or ``"cold"``.
        relevance_score: Final combined relevance score (0–1).
        score_breakdown: Decomposed scoring signals.
        inclusion_reasons: Human-readable list of why the file was
            selected or excluded.
        connected_to_changed: Graph-path evidence showing how this
            file connects to each changed file.
        token_contribution: Number of tokens this file consumed in
            the rendered context (0 if excluded).
    """

    filepath: str = ""
    selected: bool = False
    tier: str = "cold"
    relevance_score: float = 0.0
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    inclusion_reasons: list[str] = field(default_factory=list)
    connected_to_changed: list[ConnectionEvidence] = field(default_factory=list)
    token_contribution: int = 0


# ---------------------------------------------------------------------------
# EvidenceResult — enhanced ContextResult with per-file rationale
# ---------------------------------------------------------------------------


@dataclass
class EvidenceResult:
    """Enhanced context selection result with per-file evidence and rationale.

    Attributes:
        selected: Evidence records for all included files.
        excluded: Evidence records for top-N excluded files (with reasons).
        total_files: Total number of files scanned.
        total_tokens: Total original tokens across all files.
        token_budget: Hard token budget limit.
        tokens_saved: Tokens saved (original minus rendered).
        rendered_context: LLM-ready context with inline evidence comments
            prefixed to each file block.
        _context_result: The underlying ContextResult from the build pipeline.
            Used internally for metadata access; not serialised.
    """

    selected: list[FileEvidence] = field(default_factory=list)
    excluded: list[FileEvidence] = field(default_factory=list)
    total_files: int = 0
    total_tokens: int = 0
    token_budget: int = 0
    tokens_saved: int = 0
    rendered_context: str = ""
    _context_result: ContextResult | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def files_selected(self) -> int:
        """Number of files selected for inclusion."""
        return len(self.selected)

    @property
    def files_excluded(self) -> int:
        """Number of files excluded (capped at ``_MAX_EXCLUDED``)."""
        return len(self.excluded)

    @property
    def total_rendered_tokens(self) -> int:
        """Tokens consumed by the rendered context."""
        return estimate_tokens(self.rendered_context)

    @property
    def hot_files(self) -> list[FileEvidence]:
        """Files classified as HOT tier (score >= 0.8)."""
        return [fe for fe in self.selected if fe.tier == "hot"]

    @property
    def warm_files(self) -> list[FileEvidence]:
        """Files classified as WARM tier (score >= 0.25, < 0.8)."""
        return [fe for fe in self.selected if fe.tier == "warm"]

    @property
    def cold_files(self) -> list[FileEvidence]:
        """Files classified as COLD tier (score < 0.25)."""
        return [fe for fe in self.selected if fe.tier == "cold"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of selection decisions.

        Example::

            Evidence Summary: 12 selected / 143 total · 3 HOT · 5 WARM · 4 COLD
            Tokens: 11,200 / 80,000 · saved 94%
            Top reasons:
            . directly changed (3)
            . test coverage bonus (2)
            . caller via IMPORTS (depth 1) (5)
        """
        lines: list[str] = []

        selected_count = len(self.selected)
        excluded_count = len(self.excluded)
        reduction = 1.0 - (self.total_rendered_tokens / max(self.total_tokens, 1))
        lines.append(
            f"Evidence Summary: {selected_count} selected / {self.total_files} total "
            f"({excluded_count} excluded) · "
            f"{len(self.hot_files)} HOT · {len(self.warm_files)} WARM · "
            f"{len(self.cold_files)} COLD"
        )

        # Token stats
        lines.append(
            f"Tokens: {self.total_rendered_tokens:,} / {self.token_budget:,} · "
            f"saved {reduction:.0%}"
        )

        # Top reasons from selected files
        reason_counts: dict[str, int] = defaultdict(int)
        for fe in self.selected:
            for reason in fe.inclusion_reasons:
                clean = reason.split("(")[0].strip()
                reason_counts[clean] += 1
        if reason_counts:
            lines.append("Top reasons:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  . {reason} ({count})")

        # Top exclusion reasons
        excl_reason_counts: dict[str, int] = defaultdict(int)
        for fe in self.excluded:
            if fe.inclusion_reasons:
                excl_reason_counts[fe.inclusion_reasons[0]] += 1
        if excl_reason_counts:
            lines.append("Top exclusion reasons:")
            for reason, count in sorted(excl_reason_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  . {reason} ({count})")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Export the evidence result to JSON for debugging/analytics.

        Args:
            indent: JSON indentation level (default 2).

        Returns:
            JSON string with all evidence fields.
        """
        data = {
            "summary": {
                "total_files": self.total_files,
                "total_tokens": self.total_tokens,
                "token_budget": self.token_budget,
                "tokens_saved": self.tokens_saved,
                "files_selected": self.files_selected,
                "files_excluded": self.files_excluded,
                "total_rendered_tokens": self.total_rendered_tokens,
                "reduction_ratio": round(
                    1.0 - (self.total_rendered_tokens / max(self.total_tokens, 1)), 4
                ),
                "hot_count": len(self.hot_files),
                "warm_count": len(self.warm_files),
                "cold_count": len(self.cold_files),
            },
            "selected": [asdict(fe) for fe in self.selected],
            "excluded": [asdict(fe) for fe in self.excluded],
        }
        return json.dumps(data, indent=indent, default=str)

    def find_why_not(self, filepath: str) -> str:
        """Explain why a given file was NOT included in the context.

        Args:
            filepath: File path to investigate.

        Returns:
            Human-readable explanation for why the file was excluded.
            If the file was actually included, returns its selection summary.
            If the file was not found in either list, explains that.
        """
        for fe in self.selected:
            if fe.filepath == filepath:
                return (
                    f"File WAS included | score={fe.relevance_score:.3f} "
                    f"| tier={fe.tier} "
                    f"| {fe.token_contribution:,} tokens "
                    f"| reasons: {'; '.join(fe.inclusion_reasons[:3])}"
                )

        for fe in self.excluded:
            if fe.filepath == filepath:
                reasons = fe.inclusion_reasons or ["No specific reason recorded"]
                return " | ".join(reasons)

        return (
            f"File {filepath!r} was not scored. "
            "Possible reasons: not in the source map, "
            "language not supported, or below the minimum score threshold "
            "before ranking."
        )


# ---------------------------------------------------------------------------
# EvidenceTracer — wraps ContextBuilder, records per-file rationale
# ---------------------------------------------------------------------------


class EvidenceTracer:
    """Wraps a :class:`ContextBuilder` to record selection rationale for every file.

    Uses composition, not inheritance — ``EvidenceTracer`` delegates to the
    underlying ``ContextBuilder`` and intercepts the pipeline at each stage
    to collect scoring, graph, and budget evidence.

    Internal access notes:
        Reads ``builder._graph``, ``builder._ranker``, ``builder._selector``,
        and ``builder._config`` to reconstruct per-file score breakdowns and
        graph-path connections.  These are single-underscore attributes — this
        is standard within-package access in Python.

    Args:
        builder: An initialised :class:`ContextBuilder` with files already
            indexed (or to be indexed before calling ``build_with_evidence``).
    """

    def __init__(self, builder: ContextBuilder) -> None:
        self._builder = builder
        self._config: ContextConfig = builder._config
        self._ranker: RelevanceRanker = builder._ranker
        self._selector: ContextSelector = builder._selector
        self._graph = builder._graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_with_evidence(
        self,
        diff: DiffSpec,
        source_map: dict[str, str],
    ) -> EvidenceResult:
        """Build context with per-file evidence collection.

        Runs the full graphsift pipeline (graph traversal -> relevance
        ranking -> budget-aware selection -> rendering) while recording
        score breakdowns, graph-path connections, and exclusion reasons
        for every file.

        Args:
            diff: Diff specification (changed files, query, commit message).
            source_map: Dict mapping file path to source text.

        Returns:
            :class:`EvidenceResult` with per-file evidence.

        Raises:
            ValidationError: If ``diff`` has no changed files.
        """
        if not diff.changed_files:
            raise ValidationError("DiffSpec must have at least one changed_file.")

        # ---- 1. Build context normally (via ContextBuilder.build) ----
        result: ContextResult = self._builder.build(diff, source_map)

        # ---- 2. Run graph traversal for evidence (always from current graph state) ----
        graph_scores: dict[str, tuple[float, int, list[str]]] = (
            self._graph.ranked_neighbors(
                diff.changed_files,
                include_dynamic=self._config.include_dynamic,
            )
        )
        all_files = self._graph.all_files()
        ranked: list[ScoredFile] = self._ranker.rank(
            diff, graph_scores, all_files, self._config,
        )

        # ---- 3. Determine which files were selected vs excluded ----
        selected_paths: set[str] = {sf.file_node.path for sf in result.selected_files}
        changed_set: set[str] = set(diff.changed_files)

        # ---- 4. Build FileEvidence for selected files ----
        selected_evidence: list[FileEvidence] = []
        for sf in result.selected_files:
            fe = self._build_file_evidence(sf, graph_scores, diff, changed_set, source_map)
            selected_evidence.append(fe)

        # ---- 5. Build FileEvidence for excluded files ----
        excluded_evidence: list[FileEvidence] = self._build_excluded_evidence(
            ranked, selected_paths, graph_scores, diff, changed_set, source_map,
        )

        # ---- 6. Generate rendered context with inline evidence comments ----
        rendered_with_evidence = self._render_with_evidence(
            result.rendered_context, selected_evidence,
        )

        # ---- 7. Assemble EvidenceResult ----
        orig_tokens = result.total_original_tokens
        rendered_tokens = result.total_rendered_tokens

        evidence_result = EvidenceResult(
            selected=selected_evidence,
            excluded=excluded_evidence,
            total_files=len(all_files),
            total_tokens=orig_tokens,
            token_budget=self._config.token_budget,
            tokens_saved=orig_tokens - rendered_tokens,
            rendered_context=rendered_with_evidence,
            _context_result=result,
        )

        return evidence_result

    def explain(
        self,
        filepath: str,
        diff: DiffSpec,
        source_map: dict[str, str],
    ) -> FileEvidence:
        """Get a detailed evidence record for a single file.

        This is a convenience method that runs the full evidence pipeline
        and extracts the record for the specified file.  If you already
        have an ``EvidenceResult``, use ``find_why_not()`` for a string
        explanation instead.

        Args:
            filepath: File path to explain.
            diff: Diff specification.
            source_map: Dict mapping file path to source text.

        Returns:
            :class:`FileEvidence` for the given file, whether it was
            selected or excluded.  If the file was not found in either
            list, returns a minimal record with a descriptive reason.
        """
        result = self.build_with_evidence(diff, source_map)

        for fe in result.selected:
            if fe.filepath == filepath:
                return fe

        for fe in result.excluded:
            if fe.filepath == filepath:
                return fe

        # Not found — build a minimal record
        return FileEvidence(
            filepath=filepath,
            selected=False,
            tier="cold",
            relevance_score=0.0,
            inclusion_reasons=[
                "File was not found in the scored result set. "
                "It may not be in the index, may have been below the "
                "minimum score threshold, or the language may not be supported."
            ],
        )

    # ------------------------------------------------------------------
    # Internal: build FileEvidence for selected files
    # ------------------------------------------------------------------

    def _build_file_evidence(
        self,
        sf: ScoredFile,
        graph_scores: dict[str, tuple[float, int, list[str]]],
        diff: DiffSpec,
        changed_set: set[str],
        source_map: dict[str, str],
    ) -> FileEvidence:
        """Construct a ``FileEvidence`` from a selected ``ScoredFile``."""
        filepath = sf.file_node.path

        # --- Score breakdown ---
        g_score, _depth, _reasons = graph_scores.get(filepath, (0.0, 99, []))
        if filepath in changed_set:
            g_score = 1.0

        query_text = (
            diff.diff_text + " " + diff.commit_message + " " + diff.query
        )
        query_tokens = RelevanceRanker._tokenize(query_text)
        bm25_score = RelevanceRanker._bm25_score(sf.file_node, query_tokens)

        bm25_w = self._ranker._bm25_w
        graph_w = self._ranker._graph_w

        breakdown = ScoreBreakdown(
            bm25_score=round(bm25_score, 4),
            graph_distance_score=round(g_score, 4),
            tfidf_similarity=0.0,
            recency_boost=0.0,
            final_score=sf.score,
            weights_used={"bm25": bm25_w, "graph": graph_w},
        )

        # --- Tier ---
        tier = self._determine_tier(sf.score)

        # --- Clean inclusion reasons ---
        reasons = [
            r for r in sf.reasons
            if not r.endswith(")") or "(" not in r
        ]
        if not reasons:
            reasons = list(sf.reasons[:2])

        # --- Graph-path connections ---
        connections = self._trace_connections(filepath, changed_set)

        # --- Token contribution ---
        source = source_map.get(filepath, "")
        tokens = estimate_tokens(source) if source else sf.file_node.token_estimate

        return FileEvidence(
            filepath=filepath,
            selected=True,
            tier=tier,
            relevance_score=sf.score,
            score_breakdown=breakdown,
            inclusion_reasons=reasons,
            connected_to_changed=connections,
            token_contribution=tokens,
        )

    # ------------------------------------------------------------------
    # Internal: build FileEvidence for excluded files
    # ------------------------------------------------------------------

    def _build_excluded_evidence(
        self,
        ranked: list[ScoredFile],
        selected_paths: set[str],
        graph_scores: dict[str, tuple[float, int, list[str]]],
        diff: DiffSpec,
        changed_set: set[str],
        source_map: dict[str, str],
    ) -> list[FileEvidence]:
        """Build evidence records for top-N excluded files."""
        excluded: list[FileEvidence] = []

        for sf in ranked:
            path = sf.file_node.path
            if path in selected_paths:
                continue
            if len(excluded) >= _MAX_EXCLUDED:
                break

            reason = self._classify_exclusion(sf, changed_set)
            g_score, _depth, _reasons = graph_scores.get(path, (0.0, 99, []))
            if path in changed_set:
                g_score = 1.0

            query_text = (
                diff.diff_text + " " + diff.commit_message + " " + diff.query
            )
            query_tokens = RelevanceRanker._tokenize(query_text)
            bm25_score = RelevanceRanker._bm25_score(sf.file_node, query_tokens)

            bm25_w = self._ranker._bm25_w
            graph_w = self._ranker._graph_w

            breakdown = ScoreBreakdown(
                bm25_score=round(bm25_score, 4),
                graph_distance_score=round(g_score, 4),
                tfidf_similarity=0.0,
                recency_boost=0.0,
                final_score=sf.score,
                weights_used={"bm25": bm25_w, "graph": graph_w},
            )

            tier = self._determine_tier(sf.score)
            connections = self._trace_connections(path, changed_set)
            source = source_map.get(path, "")
            tokens = estimate_tokens(source) if source else sf.file_node.token_estimate

            excluded.append(FileEvidence(
                filepath=path,
                selected=False,
                tier=tier,
                relevance_score=sf.score,
                score_breakdown=breakdown,
                inclusion_reasons=[reason],
                connected_to_changed=connections,
                token_contribution=tokens,
            ))

        return excluded

    # ------------------------------------------------------------------
    # Internal: graph-path tracing
    # ------------------------------------------------------------------

    def _trace_connections(
        self,
        filepath: str,
        changed_files: set[str],
    ) -> list[ConnectionEvidence]:
        """Trace graph connections from changed files to *filepath*.

        For each changed file, find the shortest graph path to the given
        file and return a ``ConnectionEvidence`` describing it.
        """
        connections: list[ConnectionEvidence] = []
        for cf in sorted(changed_files):
            if cf == filepath:
                connections.append(ConnectionEvidence(
                    from_file=cf,
                    to_file=filepath,
                    edge_types=["DIRECT"],
                    path=[],
                    path_length=0,
                    contribution_to_score=1.0,
                ))
            else:
                path_info = self._find_graph_path(cf, filepath)
                if path_info is not None:
                    node_path, edge_types = path_info
                    file_paths = self._resolve_paths(node_path)
                    hop_count = len(node_path) - 1
                    decay = getattr(self._graph, "_decay", 0.7)
                    contribution = decay ** hop_count
                    connections.append(ConnectionEvidence(
                        from_file=cf,
                        to_file=filepath,
                        edge_types=edge_types,
                        path=file_paths,
                        path_length=hop_count,
                        contribution_to_score=round(contribution, 4),
                    ))
        return connections

    def _find_graph_path(
        self,
        from_file: str,
        to_file: str,
        max_hops: int = 8,
    ) -> tuple[list[str], list[str]] | None:
        """Find the shortest graph path from *from_file* to *to_file*.

        Uses BFS on module-level nodes in the dependency graph.  Returns a tuple
        of ``(node_id_path, edge_kind_labels)`` or ``None`` if no path
        exists within ``max_hops``.

        Args:
            from_file: Source file path.
            to_file: Target file path.
            max_hops: Maximum path length.  Default 8.

        Returns:
            ``(node_id_path, edge_kind_strings)`` if connected,
            ``None`` otherwise.
        """
        from_id = f"{from_file}::__module__"
        to_id = f"{to_file}::__module__"

        graph = self._graph

        if from_id not in graph._nodes or to_id not in graph._nodes:
            return None

        queue: deque[tuple[str, list[str], list[str]]] = deque()
        queue.append((from_id, [from_id], []))
        visited: set[str] = {from_id}

        while queue:
            node_id, path, edge_types = queue.popleft()
            current_depth = len(path) - 1

            if current_depth >= max_hops:
                continue

            for edge in graph._adj_out.get(node_id, []):
                nid = edge.target_id
                if nid == to_id:
                    return (
                        path + [nid],
                        edge_types + [edge.kind.value],
                    )
                if nid not in visited:
                    visited.add(nid)
                    queue.append((
                        nid,
                        path + [nid],
                        edge_types + [edge.kind.value],
                    ))

            for edge in graph._adj_in.get(node_id, []):
                nid = edge.source_id
                if nid == to_id:
                    return (
                        path + [nid],
                        edge_types + [edge.kind.value],
                    )
                if nid not in visited:
                    visited.add(nid)
                    queue.append((
                        nid,
                        path + [nid],
                        edge_types + [edge.kind.value],
                    ))

        return None

    @staticmethod
    def _resolve_paths(node_ids: list[str]) -> list[str]:
        """Convert a list of node IDs back to file paths.

        Node IDs follow the pattern ``file_path::symbol``, so extracting
        the file path means splitting on the last ``::`` separator.
        """
        paths: list[str] = []
        for nid in node_ids:
            idx = nid.rfind("::")
            if idx != -1:
                fp = nid[:idx]
                if not paths or paths[-1] != fp:
                    paths.append(fp)
            else:
                if not paths or paths[-1] != nid:
                    paths.append(nid)
        return paths

    # ------------------------------------------------------------------
    # Internal: tier determination and exclusion classification
    # ------------------------------------------------------------------

    def _determine_tier(self, score: float) -> str:
        """Map a relevance score to a tier label.

        Uses the same thresholds as ``ContextSelector._render_file()``.
        """
        hot_threshold = getattr(self._config, "hot_threshold", 0.8)
        warm_threshold = getattr(self._config, "warm_threshold", 0.25)
        if score >= hot_threshold:
            return "hot"
        if score >= warm_threshold:
            return "warm"
        return "cold"

    def _classify_exclusion(
        self,
        sf: ScoredFile,
        changed_set: set[str],
    ) -> str:
        """Determine the dominant reason a ranked file was not selected.

        Checks are ordered from most specific to most general:
        1. Changed file that was not selected (should be rare)
        2. Below warm threshold
        3. Test file (when tests excluded)
        4. Below min score
        5. Budget exhaustion (catch-all)
        """
        config = self._config
        path = sf.file_node.path

        if path in changed_set:
            return "Changed file not selected — budget exhausted at priority inclusion stage"

        if sf.score < config.warm_threshold:
            return (
                f"Score {sf.score:.3f} below warm threshold "
                f"{config.warm_threshold} (COLD tier)"
            )

        if not config.include_tests:
            if RelevanceRanker._is_test(path):
                return "Test files excluded by configuration (include_tests=False)"

        if sf.score < config.min_score:
            return (
                f"Score {sf.score:.3f} below minimum score threshold "
                f"{config.min_score}"
            )

        return (
            f"Token budget exhausted before file could be selected "
            f"(score={sf.score:.3f})"
        )

    # ------------------------------------------------------------------
    # Internal: evidence comment rendering
    # ------------------------------------------------------------------

    def _render_with_evidence(
        self,
        original_context: str,
        selected_evidence: list[FileEvidence],
    ) -> str:
        """Insert inline evidence HTML comments into the rendered context.

        Each file section preceded by ``## path [TIER]`` gets an evidence
        comment block directly above it:

        .. code-block:: html

            <!-- INCLUDED: src/auth.py | score=0.92 | tier=hot |
                 BM25=0.88 graph=0.95 |
                 connected to changed: src/auth.py (direct) |
                 1,200 tokens -->
            ## src/auth.py [HOT]

        Args:
            original_context: The standard rendered LLM context.
            selected_evidence: Evidence records for selected files.

        Returns:
            Context string with inline evidence comments inserted.
        """
        evidence_map: dict[str, FileEvidence] = {
            fe.filepath: fe for fe in selected_evidence
        }

        lines = original_context.splitlines()
        result: list[str] = []
        tier_suffixes = ("[HOT]", "[WARM]", "[COLD]")

        for line in lines:
            if line.startswith("## "):
                stripped = line.strip()
                matched_tier = None
                for suffix in tier_suffixes:
                    if stripped.endswith(suffix):
                        matched_tier = suffix
                        break

                if matched_tier is not None:
                    filepath_part = stripped[3:-len(matched_tier)].rstrip()
                    filepath_part = filepath_part.rstrip("[").rstrip()
                    fe = evidence_map.get(filepath_part)
                    if fe is not None:
                        comment = self._build_evidence_comment(fe)
                        result.append(comment)

            result.append(line)

        return "\n".join(result)

    def _build_evidence_comment(self, fe: FileEvidence) -> str:
        """Build an HTML-comment evidence string for a ``FileEvidence``."""
        parts: list[str] = [
            f"INCLUDED: {fe.filepath}",
            f"score={fe.relevance_score:.3f}",
            f"tier={fe.tier}",
        ]

        sb = fe.score_breakdown
        parts.append(f"BM25={sb.bm25_score:.3f} graph={sb.graph_distance_score:.3f}")

        if fe.connected_to_changed:
            conn_parts: list[str] = []
            for conn in fe.connected_to_changed:
                if conn.path_length == 0:
                    conn_parts.append(f"{Path(conn.from_file).name} (direct)")
                else:
                    edge_str = "/".join(conn.edge_types[:3])
                    conn_parts.append(
                        f"{Path(conn.from_file).name} via {edge_str}"
                    )
            if conn_parts:
                parts.append("connected to changed: " + ", ".join(conn_parts[:3]))

        parts.append(f"{fe.token_contribution:,} tokens")

        return "<!-- " + " | ".join(parts) + " -->"
