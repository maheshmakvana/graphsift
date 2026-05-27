"""Typed-path retrieval over the dependency graph.

PRISM-style typed traversal where different edge types receive different
weights based on the query intent.  Instead of generic graph-distance decay
from changed files, this provides intent-aware scoring.

Usage::

    from graphsift.typed_retrieval import TypedRetriever, QueryIntent

    retriever = TypedRetriever(graph, intent=QueryIntent.SECURITY_REVIEW)
    scores = retriever.score(["src/auth.py"])
    # -> {"src/db.py": 0.84, "src/middleware.py": 0.62, ...}

    paths = retriever.find_paths("src/auth.py", "src/db.py")
    nb = retriever.neighborhood("src/auth.py")
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .core import DependencyGraph
from .models import EdgeKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query intent enum
# ---------------------------------------------------------------------------


class QueryIntent(str, Enum):
    """Intent of the retrieval query, controlling per-edge-type weights.

    Each intent biases the traversal toward edge kinds that are most
    informative for that type of analysis:

    * **SECURITY_REVIEW** — emphasises CALLS, IMPORTS, and DECORATES edges
      to trace data-flow and authentication boundaries.
    * **REFACTOR_IMPACT** — emphasises CALLS, IMPORTS, and INHERITS edges
      to surface ripple effects of structural changes.
    * **TEST_IMPACT** — strongly weights TEST_COVERS edges to highlight
      testing surface area.
    * **DEPENDENCY_UPDATE** — weights IMPORTS and DYNAMIC_IMPORT edges
      to capture the full dependency footprint.
    * **ARCHITECTURE_REVIEW** — weights INHERITS and DECORATES edges
      to expose class hierarchy and cross-cutting concerns.
    * **GENERAL** — all edge kinds weighted equally.
    """

    SECURITY_REVIEW = "security_review"
    REFACTOR_IMPACT = "refactor_impact"
    TEST_IMPACT = "test_impact"
    DEPENDENCY_UPDATE = "dependency_update"
    ARCHITECTURE_REVIEW = "architecture_review"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Intent weights: per-EdgeKind multipliers for each QueryIntent
# ---------------------------------------------------------------------------

_INTENT_WEIGHTS: dict[QueryIntent, dict[EdgeKind, float]] = {
    QueryIntent.SECURITY_REVIEW: {
        EdgeKind.CALLS: 2.0,
        EdgeKind.IMPORTS: 1.5,
        EdgeKind.INHERITS: 0.8,
        EdgeKind.DECORATES: 1.5,
        EdgeKind.REFERENCES: 0.8,
        EdgeKind.TEST_COVERS: 0.8,
        EdgeKind.DYNAMIC_IMPORT: 0.8,
    },
    QueryIntent.REFACTOR_IMPACT: {
        EdgeKind.CALLS: 1.8,
        EdgeKind.IMPORTS: 1.5,
        EdgeKind.INHERITS: 1.5,
        EdgeKind.DECORATES: 1.0,
        EdgeKind.REFERENCES: 1.0,
        EdgeKind.TEST_COVERS: 1.0,
        EdgeKind.DYNAMIC_IMPORT: 1.0,
    },
    QueryIntent.TEST_IMPACT: {
        EdgeKind.CALLS: 1.2,
        EdgeKind.IMPORTS: 0.7,
        EdgeKind.INHERITS: 0.7,
        EdgeKind.DECORATES: 0.7,
        EdgeKind.REFERENCES: 0.7,
        EdgeKind.TEST_COVERS: 3.0,
        EdgeKind.DYNAMIC_IMPORT: 0.7,
    },
    QueryIntent.DEPENDENCY_UPDATE: {
        EdgeKind.CALLS: 0.8,
        EdgeKind.IMPORTS: 2.5,
        EdgeKind.INHERITS: 0.8,
        EdgeKind.DECORATES: 0.8,
        EdgeKind.REFERENCES: 0.8,
        EdgeKind.TEST_COVERS: 0.8,
        EdgeKind.DYNAMIC_IMPORT: 2.5,
    },
    QueryIntent.ARCHITECTURE_REVIEW: {
        EdgeKind.CALLS: 1.3,
        EdgeKind.IMPORTS: 0.9,
        EdgeKind.INHERITS: 2.0,
        EdgeKind.DECORATES: 2.0,
        EdgeKind.REFERENCES: 0.9,
        EdgeKind.TEST_COVERS: 0.9,
        EdgeKind.DYNAMIC_IMPORT: 0.9,
    },
    QueryIntent.GENERAL: {
        EdgeKind.CALLS: 1.0,
        EdgeKind.IMPORTS: 1.0,
        EdgeKind.INHERITS: 1.0,
        EdgeKind.DECORATES: 1.0,
        EdgeKind.REFERENCES: 1.0,
        EdgeKind.TEST_COVERS: 1.0,
        EdgeKind.DYNAMIC_IMPORT: 1.0,
    },
}


def get_intent_weights(intent: QueryIntent) -> dict[EdgeKind, float]:
    """Return a copy of the per-edge-kind weight map for *intent*.

    Args:
        intent: The retrieval intent.

    Returns:
        Dict mapping each :class:`EdgeKind` to its weight multiplier.
    """
    return dict(_INTENT_WEIGHTS[intent])


# ---------------------------------------------------------------------------
# Supporting data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedPath:
    """A single typed path between two files in the dependency graph.

    Attributes:
        nodes: File paths along the path (source to target), in order.
        edges: Edge kinds along the path, one fewer than *nodes*.
        cumulative_score: Product of edge weights times hop decay along
            the path.  Higher = more relevant given the intent.
        path_description: Human-readable description of the path, e.g.
            ``"auth.py --[calls]--> middleware.py --[imports]--> db.py"``.
    """

    nodes: list[str]
    edges: list[EdgeKind]
    cumulative_score: float
    path_description: str

    def __repr__(self) -> str:
        return (
            f"TypedPath(nodes={len(self.nodes)}, "
            f"score={self.cumulative_score:.4f})"
        )


@dataclass(frozen=True)
class TypedNeighborhood:
    """Typed one-hop neighborhood of a file node.

    Each field holds a sorted list of file paths connected to the seed file
    via the corresponding edge kind.

    Attributes:
        calls: Files that this file directly calls.
        imports: Files that this file directly imports.
        inherits: Files that this file inherits from.
        decorated_by: Files that provide decorators used by this file.
        test_covered_by: Test files that cover this file.
        references: Files that reference symbols in this file.
    """

    calls: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    inherits: list[str] = field(default_factory=list)
    decorated_by: list[str] = field(default_factory=list)
    test_covered_by: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        total = (
            len(self.calls)
            + len(self.imports)
            + len(self.inherits)
            + len(self.decorated_by)
            + len(self.test_covered_by)
            + len(self.references)
        )
        return f"TypedNeighborhood({total} neighbors)"

    @property
    def all(self) -> list[str]:
        """All unique neighbor file paths across all edge kinds."""
        seen: set[str] = set()
        result: list[str] = []
        for bucket in (
            self.calls,
            self.imports,
            self.inherits,
            self.decorated_by,
            self.test_covered_by,
            self.references,
        ):
            for fp in bucket:
                if fp not in seen:
                    seen.add(fp)
                    result.append(fp)
        return result


# ---------------------------------------------------------------------------
# TypedRetriever
# ---------------------------------------------------------------------------


class TypedRetriever:
    """PRISM-style typed-path retrieval over the dependency graph.

    Instead of generic graph-distance decay from changed files (which treats
    all edges uniformly), this performs typed-path traversal where each edge
    kind receives a weight multiplier based on the query intent.

    The core algorithm is a two-directional BFS (outgoing **and** incoming
    edges) from seed files, applying:

    1. A **hop decay** factor (default 0.7) per step.
    2. An **edge-type weight** from the chosen intent.

    Final scores are normalised to ``[0, 1]``.

    Args:
        graph: The :class:`DependencyGraph` to traverse.
        intent: The retrieval intent controlling per-edge-type weights.
        decay: Score multiplier per hop (0.7 = 30% decay each level).
            Must be in ``(0, 1]``.
        max_hops: Maximum BFS depth during scoring and traversal.
            Must be >= 1.

    Raises:
        ValueError: If *decay* or *max_hops* are out of range.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        intent: QueryIntent = QueryIntent.GENERAL,
        decay: float = 0.7,
        max_hops: int = 4,
    ) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")

        self._graph = graph
        self._intent = intent
        self._decay = decay
        self._max_hops = max_hops
        self._intent_weights = get_intent_weights(intent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        changed_files: list[str],
        max_hops: int | None = None,
    ) -> dict[str, float]:
        """Score files by typed multi-hop traversal from *changed_files*.

        Runs a bidirectional BFS (outgoing and incoming edges) from each
        changed file, accumulating type-weighted scores along every
        discovered path.  Scores are normalised to ``[0, 1]``.

        The scoring formula for each reachable file is::

            score(f) = sum over all discovered paths P from seeds to f of
                product over each edge e in P of (decay * intent_weight(e.kind))

        Visited nodes are tracked by identity so each node is expanded
        once (the first time it is reached), but alternative paths that
        reach an *already-visited* node still contribute their score
        to that node without re-expanding it.  This prevents exponential
        blowup while still approximating the full sum-over-paths semantics.

        Args:
            changed_files: Seed file paths (the changed / modified files).
            max_hops: Override the instance-level ``max_hops`` for this
                call.  ``None`` falls back to the instance default.

        Returns:
            Dict mapping ``file_path`` to normalised relevance score
            (``0.0`` = irrelevant, ``1.0`` = most relevant).  An empty
            dict is returned when no seeds are found in the graph.
        """
        max_hops = max_hops if max_hops is not None else self._max_hops
        changed_set = set(changed_files)

        # accumulated raw scores per file path
        scores: dict[str, float] = defaultdict(float)
        # visited set tracks node IDs that have been *queued for expansion*
        visited: set[str] = set()

        for seed in changed_files:
            seed_id = self._module_id(seed)
            if seed_id not in self._graph._nodes:
                continue
            if seed_id in visited:
                continue
            visited.add(seed_id)

            queue: deque[tuple[str, float, int]] = deque()
            queue.append((seed_id, 1.0, 0))

            while queue:
                node_id, cum_score, depth = queue.popleft()

                if depth >= max_hops:
                    continue

                # --- Outgoing edges: what this file depends on ---
                for edge in self._graph._adj_out.get(node_id, []):
                    w = self._intent_weights.get(edge.kind, 1.0)
                    new_score = cum_score * self._decay * w
                    self._enqueue_or_accumulate(
                        edge.target_id, new_score, depth + 1,
                        visited, queue, scores, changed_set,
                    )

                # --- Incoming edges: what depends on this file ---
                for edge in self._graph._adj_in.get(node_id, []):
                    w = self._intent_weights.get(edge.kind, 1.0)
                    new_score = cum_score * self._decay * w
                    self._enqueue_or_accumulate(
                        edge.source_id, new_score, depth + 1,
                        visited, queue, scores, changed_set,
                    )

        # Normalise to [0, 1]
        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score <= 0:
            return {}

        return {
            path: round(s / max_score, 4)
            for path, s in scores.items()
        }

    def find_paths(
        self,
        from_file: str,
        to_file: str,
        max_hops: int | None = None,
    ) -> list[TypedPath]:
        """Find all typed paths between *from_file* and *to_file*.

        Enumerates all paths up to given hop limit, ranked by
        type-weighted cumulative score descending.  Because this
        performs an exhaustive enumeration, it is best suited for
        targeted queries between a small number of files.

        Paths are returned in descending order of ``cumulative_score``
        so the most relevant connection appears first.

        Args:
            from_file: Source file path.
            to_file: Target file path.
            max_hops: Maximum path length.  ``None`` uses the instance
                default.

        Returns:
            List of :class:`TypedPath` objects sorted by score descending.
            Empty list if either file is not in the graph or no path
            exists within the hop limit.
        """
        max_hops = max_hops if max_hops is not None else self._max_hops
        target_id = self._module_id(to_file)
        source_id = self._module_id(from_file)

        if (
            source_id not in self._graph._nodes
            or target_id not in self._graph._nodes
        ):
            return []

        found: list[TypedPath] = []

        # BFS state: (node_id, cumulative_score, path_nodes, path_edges)
        queue: deque[tuple[str, float, list[str], list[EdgeKind]]] = deque()
        queue.append((source_id, 1.0, [from_file], []))

        while queue:
            node_id, cum_score, path_nodes, path_edges = queue.popleft()
            hop_count = len(path_nodes) - 1

            if hop_count > max_hops:
                continue

            if node_id == target_id and path_nodes:
                desc = self._describe_path(path_nodes, path_edges, from_file, to_file)
                found.append(TypedPath(
                    nodes=list(path_nodes),
                    edges=list(path_edges),
                    cumulative_score=round(cum_score, 4),
                    path_description=desc,
                ))
                # Continue searching — there may be longer / alternative paths
                if hop_count >= max_hops:
                    continue

            # Outgoing edges
            for edge in self._graph._adj_out.get(node_id, []):
                tgt_file = self._file_for_node(edge.target_id)
                if tgt_file is None:
                    continue
                w = self._intent_weights.get(edge.kind, 1.0)
                new_score = cum_score * self._decay * w
                queue.append((
                    edge.target_id,
                    new_score,
                    path_nodes + [tgt_file],
                    path_edges + [edge.kind],
                ))

            # Incoming edges
            for edge in self._graph._adj_in.get(node_id, []):
                src_file = self._file_for_node(edge.source_id)
                if src_file is None:
                    continue
                w = self._intent_weights.get(edge.kind, 1.0)
                new_score = cum_score * self._decay * w
                queue.append((
                    edge.source_id,
                    new_score,
                    path_nodes + [src_file],
                    path_edges + [edge.kind],
                ))

        found.sort(key=lambda p: p.cumulative_score, reverse=True)
        return found

    def reachable(
        self,
        from_files: list[str],
        edge_types: list[EdgeKind] | None = None,
        max_hops: int | None = None,
    ) -> set[str]:
        """Find all files reachable via specific edge types within N hops.

        Unlike :meth:`score` which returns weighted scores, this returns
        a flat set of reachable file paths — useful for blast-radius
        analysis or when you need just the set of affected files without
        scoring.

        Args:
            from_files: Seed file paths to start traversal from.
            edge_types: Allowed edge kinds.  ``None`` means all edge
                kinds are traversable.
            max_hops: Maximum traversal depth.  ``None`` uses the
                instance default.

        Returns:
            Set of reachable file paths (empty if no seeds are in the
            graph or no connections are found).
        """
        max_hops = max_hops if max_hops is not None else self._max_hops
        allowed = set(edge_types) if edge_types is not None else None

        reachable_files: set[str] = set()
        visited: set[str] = set()

        for seed in from_files:
            seed_id = self._module_id(seed)
            if seed_id not in self._graph._nodes:
                continue
            if seed_id in visited:
                continue
            visited.add(seed_id)

            queue: deque[tuple[str, int]] = deque()
            queue.append((seed_id, 0))

            while queue:
                node_id, depth = queue.popleft()

                if depth > 0:
                    fp = self._file_for_node(node_id)
                    if fp is not None:
                        reachable_files.add(fp)

                if depth >= max_hops:
                    continue

                # Outgoing edges
                for edge in self._graph._adj_out.get(node_id, []):
                    if allowed is not None and edge.kind not in allowed:
                        continue
                    if edge.target_id not in visited:
                        visited.add(edge.target_id)
                        queue.append((edge.target_id, depth + 1))

                # Incoming edges
                for edge in self._graph._adj_in.get(node_id, []):
                    if allowed is not None and edge.kind not in allowed:
                        continue
                    if edge.source_id not in visited:
                        visited.add(edge.source_id)
                        queue.append((edge.source_id, depth + 1))

        return reachable_files

    def neighborhood(
        self,
        filepath: str,
        radius: int = 1,
    ) -> TypedNeighborhood:
        """Get the typed neighborhood of a file.

        Returns what the file calls, imports, inherits from, is decorated
        by, is tested by, and references — grouped by edge kind.

        This is useful for quickly understanding a file's role in the
        dependency graph without needing to inspect raw edge lists.

        Args:
            filepath: File path to inspect.
            radius: Number of hops to expand.  Default ``1`` (direct
                neighbors only).

        Returns:
            :class:`TypedNeighborhood` with neighbor paths grouped by
            edge kind.  An empty neighborhood is returned if the file
            is not in the graph.
        """
        node_id = self._module_id(filepath)
        if node_id not in self._graph._nodes:
            return TypedNeighborhood()

        calls: set[str] = set()
        imports: set[str] = set()
        inherits: set[str] = set()
        decorated_by: set[str] = set()
        test_covered_by: set[str] = set()
        references: set[str] = set()

        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque()
        queue.append((node_id, 0))

        while queue:
            current_id, depth = queue.popleft()

            if depth >= radius:
                continue

            for edge in self._graph._adj_out.get(current_id, []):
                neighbor_file = self._file_for_node(edge.target_id)
                if neighbor_file is not None and neighbor_file != filepath:
                    self._classify_edge(
                        edge.kind, neighbor_file,
                        calls, imports, inherits,
                        decorated_by, test_covered_by, references,
                    )
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, depth + 1))

            for edge in self._graph._adj_in.get(current_id, []):
                neighbor_file = self._file_for_node(edge.source_id)
                if neighbor_file is not None and neighbor_file != filepath:
                    self._classify_edge(
                        edge.kind, neighbor_file,
                        calls, imports, inherits,
                        decorated_by, test_covered_by, references,
                    )
                if edge.source_id not in visited:
                    visited.add(edge.source_id)
                    queue.append((edge.source_id, depth + 1))

        return TypedNeighborhood(
            calls=sorted(calls),
            imports=sorted(imports),
            inherits=sorted(inherits),
            decorated_by=sorted(decorated_by),
            test_covered_by=sorted(test_covered_by),
            references=sorted(references),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def intent(self) -> QueryIntent:
        """The current retrieval intent."""
        return self._intent

    @property
    def decay(self) -> float:
        """Score multiplier per hop."""
        return self._decay

    @property
    def max_hops(self) -> int:
        """Maximum traversal depth."""
        return self._max_hops

    def __repr__(self) -> str:
        return (
            f"TypedRetriever(intent={self._intent.value}, "
            f"decay={self._decay}, max_hops={self._max_hops})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _module_id(file_path: str) -> str:
        """Build the module-level node ID for *file_path*.

        Matches the convention used by :class:`DependencyGraph` where
        each file has a ``path::__module__`` sentinel node.
        """
        return f"{file_path}::__module__"

    @staticmethod
    def _file_for_node(node_id: str) -> str | None:
        """Extract the file path from a ``file_path::symbol`` node ID.

        Splits on the last ``::`` separator to handle both module nodes
        (``path::__module__``) and symbol nodes (``path::ClassName``).

        Returns ``None`` if the node ID does not contain ``::``.
        """
        idx = node_id.rfind("::")
        if idx != -1:
            return node_id[:idx]
        return None

    @staticmethod
    def _enqueue_or_accumulate(
        neighbor_id: str,
        new_score: float,
        new_depth: int,
        visited: set[str],
        queue: deque[tuple[str, float, int]],
        scores: dict[str, float],
        changed_set: set[str],
    ) -> None:
        """Enqueue *neighbor_id* for expansion or accumulate its score.

        If the neighbor has not been visited yet, it is queued for
        expansion at the new depth.  Regardless of visit status, the
        score for the neighbor's file is accumulated (this allows
        alternative paths to contribute even when we avoid re-expansion).
        """
        neighbor_file = TypedRetriever._file_for_node(neighbor_id)

        # Accumulate score contribution regardless of visited status
        if neighbor_file is not None and neighbor_file not in changed_set:
            scores[neighbor_file] += new_score

        # Only enqueue for expansion once
        if neighbor_id not in visited:
            visited.add(neighbor_id)
            queue.append((neighbor_id, new_score, new_depth))

    @staticmethod
    def _classify_edge(
        kind: EdgeKind,
        neighbor: str,
        calls: set[str],
        imports: set[str],
        inherits: set[str],
        decorated_by: set[str],
        test_covered_by: set[str],
        references: set[str],
    ) -> None:
        """Route *neighbor* into the correct typed bucket by edge kind."""
        if kind == EdgeKind.CALLS:
            calls.add(neighbor)
        elif kind == EdgeKind.IMPORTS:
            imports.add(neighbor)
        elif kind == EdgeKind.INHERITS:
            inherits.add(neighbor)
        elif kind == EdgeKind.DECORATES:
            decorated_by.add(neighbor)
        elif kind == EdgeKind.TEST_COVERS:
            test_covered_by.add(neighbor)
        elif kind == EdgeKind.REFERENCES:
            references.add(neighbor)

    @staticmethod
    def _describe_path(
        nodes: list[str],
        edges: list[EdgeKind],
        from_file: str,
        to_file: str,
    ) -> str:
        """Build a human-readable description of a typed path.

        Produces output like::

            auth.py --[calls]--> middleware.py --[imports]--> db.py

        Args:
            nodes: File paths in the path (including source and target).
            edges: Edge kinds connecting consecutive nodes.
            from_file: Source file path (for compact display).
            to_file: Target file path (for compact display).

        Returns:
            Compact string representation of the path.
        """
        if not nodes or not edges:
            return f"{Path(from_file).name} -> {Path(to_file).name} (no edges)"

        parts: list[str] = []
        for i, edge_kind in enumerate(edges):
            src_name = Path(nodes[i]).name
            tgt_name = Path(nodes[i + 1]).name
            parts.append(f"{src_name} --[{edge_kind.value}]--> {tgt_name}")

        return " ".join(parts)
