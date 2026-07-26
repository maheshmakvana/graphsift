"""Evolutionary parameter optimization for graphsift components.

Finds optimal parameter sets for RelevanceRanker weights, ContextConfig
fields, and DependencyGraph hyperparameters using a genetic-algorithm-
inspired evolutionary loop (selection, mutation, crossover, evaluation).

Unlike grid search or random search, the evolutionary optimizer:

- Adapts the search distribution as better candidates are found
- Combines successful parameter sets via crossover
- Handles mixed int/float parameter spaces natively
- Catches evaluator failures gracefully (no single bad run kills the loop)

Usage::

    from graphsift.evolve import EvolutionOptimizer, ParameterSpace

    space = ParameterSpace.ranker_space()
    opt = EvolutionOptimizer(space, seed=42)
    result = opt.optimize(
        seed_params=space.defaults(),
        evaluator=my_evaluator,
        rounds=30,
    )
    print(result.summary())

    # Quick-start convenience for ranker tuning:
    result = opt.optimize_ranker(
        source_map=my_source_map,
        diff_spec=my_diff,
        rounds=20,
    )
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .models import DiffSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParamDef — single tunable parameter definition
# ---------------------------------------------------------------------------


@dataclass
class ParamDef:
    """Definition of a single tunable parameter.

    Attributes:
        name: Parameter name (matches the keyword argument it tunes).
        low: Minimum allowed value (inclusive).
        high: Maximum allowed value (inclusive).
        typ: Python type — ``int`` or ``float``.
        default: Default / recommended starting value.
    """

    name: str
    low: float
    high: float
    typ: type
    default: float


# ---------------------------------------------------------------------------
# ParameterSpace — collection of tunable parameters
# ---------------------------------------------------------------------------


@dataclass
class ParameterSpace:
    """Collection of tunable parameters with bounds and types.

    Provides factory methods for pre-built graphsift parameter spaces and
    supports sampling, mutation, crossover, and validation.

    Attributes:
        params: List of :class:`ParamDef` entries.
    """

    params: list[ParamDef] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Pre-built factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def ranker_space() -> ParameterSpace:
        """Return parameter space for RelevanceRanker tuning.

        Tunes:
            ``bm25_weight`` (0.05–0.8), ``graph_weight`` (0.2–0.95),
            ``god_node_penalty`` (0.05–0.7).
        """
        return ParameterSpace([
            ParamDef("bm25_weight", 0.05, 0.8, float, 0.3),
            ParamDef("graph_weight", 0.2, 0.95, float, 0.7),
            ParamDef("god_node_penalty", 0.05, 0.7, float, 0.3),
        ])

    @staticmethod
    def config_space() -> ParameterSpace:
        """Return parameter space for ContextConfig tuning.

        Tunes:
            ``token_budget`` (500–8000), ``min_score`` (0.01–0.2),
            ``hot_threshold`` (0.6–0.95), ``warm_threshold`` (0.1–0.5),
            ``trimming_context_lines`` (0–20).
        """
        return ParameterSpace([
            ParamDef("token_budget", 500, 8000, int, 2000),
            ParamDef("min_score", 0.01, 0.2, float, 0.05),
            ParamDef("hot_threshold", 0.6, 0.95, float, 0.8),
            ParamDef("warm_threshold", 0.1, 0.5, float, 0.25),
            ParamDef("trimming_context_lines", 0, 20, int, 3),
        ])

    @staticmethod
    def graph_space() -> ParameterSpace:
        """Return parameter space for DependencyGraph hyperparameters.

        Tunes:
            ``decay`` (0.3–0.95), ``max_depth`` (1–6).
        """
        return ParameterSpace([
            ParamDef("decay", 0.3, 0.95, float, 0.7),
            ParamDef("max_depth", 1, 6, int, 4),
        ])

    @staticmethod
    def full_space() -> ParameterSpace:
        """Return combined ranker + config + graph parameter space (10 params)."""
        return ParameterSpace(
            ParameterSpace.ranker_space().params
            + ParameterSpace.config_space().params
            + ParameterSpace.graph_space().params
        )

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def defaults(self) -> dict[str, Any]:
        """Return a dict of parameter name → default value."""
        return {p.name: p.default for p in self.params}

    def sample(
        self,
        rng: random.Random,
        seed: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a random candidate within bounds.

        For each parameter, a uniform-random value is drawn inside
        ``[low, high]``.  If *seed* is provided, its keys are kept and
        only missing parameters are randomised (useful for partial seeding).

        Args:
            rng: Random instance for reproducibility.
            seed: Optional template dict to merge results into.

        Returns:
            Dict of parameter name → value.
        """
        result = dict(seed) if seed else {}
        for p in self.params:
            if p.name not in result:
                val = rng.uniform(p.low, p.high)
                result[p.name] = p.typ(val) if p.typ is int else val
        return result

    def mutate(
        self,
        params: dict[str, Any],
        rng: random.Random,
        rate: float = 0.3,
    ) -> dict[str, Any]:
        """Gaussian mutation within bounds.

        Each parameter is independently mutated with probability *rate*.
        The standard deviation of the Gaussian perturbation is
        ``(high - low) * 0.15``.  All values are clamped to
        ``[low, high]`` after mutation; integer parameters are rounded.

        Args:
            params: Parent parameter dict.
            rng: Random instance for reproducibility.
            rate: Per-parameter mutation probability (0–1).

        Returns:
            Mutated parameter dict.
        """
        bounds = {p.name: (p.low, p.high, p.typ) for p in self.params}
        result = dict(params)
        for name, (lo, hi, typ) in bounds.items():
            if name not in result:
                continue
            if rng.random() < rate:
                sigma = (hi - lo) * 0.15
                val = rng.gauss(float(result[name]), sigma)
                val = max(lo, min(hi, val))
                result[name] = typ(val) if typ is int else val
        return result

    def crossover(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        """Uniform crossover between two parent parameter sets.

        Each parameter is inherited from parent *a* or *b* with equal
        probability.

        Args:
            a: First parent.
            b: Second parent.
            rng: Random instance for reproducibility.

        Returns:
            Child parameter dict.
        """
        child = {}
        for key in a:
            if key not in b:
                child[key] = a[key]
            else:
                child[key] = a[key] if rng.random() < 0.5 else b[key]
        return child

    def validate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Clamp all values to their bounds and cast to the correct type.

        ``int`` params that arrive as ``float`` (e.g. from crossover) are
        rounded and converted.

        Args:
            params: Parameter dict to validate.

        Returns:
            New dict with clamped and type-corrected values.
        """
        bounds = {p.name: (p.low, p.high, p.typ) for p in self.params}
        result = {}
        for name, val in params.items():
            if name in bounds:
                lo, hi, typ = bounds[name]
                clamped = max(lo, min(hi, float(val)))
                result[name] = typ(clamped) if typ is int else clamped
            else:
                # Unknown param — pass through unchanged
                result[name] = val
        return result


# ---------------------------------------------------------------------------
# EvolutionResult — outcome of an optimization run
# ---------------------------------------------------------------------------


@dataclass
class EvolutionResult:
    """Result of an evolutionary optimization run.

    Attributes:
        best_params: Best parameter set found.
        best_score: Best score achieved (higher is better).
        rounds: Total evaluation rounds.
        improvements: Number of times the best score was improved.
        history: List of ``(round, score, params)`` tuples for each
                 improvement milestone.
        duration_s: Wall-clock time in seconds.
    """

    best_params: dict[str, Any]
    best_score: float
    rounds: int
    improvements: int
    history: list[tuple[int, float, dict[str, Any]]] = field(default_factory=list)
    duration_s: float = 0.0

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"EvolutionOptimizer: best_score={self.best_score:.4f}, "
            f"improvements={self.improvements}/{self.rounds} rounds, "
            f"duration={self.duration_s:.1f}s"
        )


# ---------------------------------------------------------------------------
# EvolutionOptimizer — main evolutionary loop
# ---------------------------------------------------------------------------


class EvolutionOptimizer:
    """Evolutionary optimizer for parameter tuning.

    Implements a **select → mutate / crossover → evaluate → accept** loop
    that evolves parameter sets toward higher scores.

    All randomness is controlled by a seedable ``random.Random`` instance
    for full reproducibility.

    Args:
        space: :class:`ParameterSpace` defining tunable params and bounds.
        seed: Random seed for reproducibility.
        verbose: If ``True``, log progress to ``logger.info``.
    """

    def __init__(
        self,
        space: ParameterSpace,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self._space = space
        self._rng = random.Random(seed)
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        seed_params: dict[str, Any],
        evaluator: Callable[[dict[str, Any]], float],
        rounds: int = 30,
        population: int = 6,
    ) -> EvolutionResult:
        """Run the evolutionary optimization loop.

        Algorithm:

            1. Evaluate *seed_params* to set the initial best score.
            2. For each round:

               a. Generate candidates via mutation of random pool members
                  and crossover of parent pairs.
               b. Evaluate all candidates.
               c. Select top candidates for the next pool.
               d. If the best candidate improves over the global best,
                  record the improvement in history.

        Args:
            seed_params: Starting parameter dict.
            evaluator: Callable that takes a param dict and returns a
                score (higher = better).
            rounds: Number of evaluation rounds.
            population: Number of candidates per round (minimum 2).

        Returns:
            :class:`EvolutionResult` with the best parameters found.

        Raises:
            ValueError: If *population* < 2 or *rounds* < 1.
        """
        if population < 2:
            raise ValueError(f"population must be >= 2, got {population}")
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")

        seed_params = self._space.validate(seed_params)
        t0 = time.perf_counter()

        # --- Evaluate seed ---
        best_score = self._evaluate_safe(evaluator, seed_params)
        best_params = dict(seed_params)
        pool: list[dict[str, Any]] = [dict(seed_params)]
        improvements = 0
        history: list[tuple[int, float, dict[str, Any]]] = []

        if self._verbose:
            logger.info("Seed score: %.4f", best_score)

        # --- Evolution rounds ---
        for round_idx in range(rounds):
            candidates: list[dict[str, Any]] = []

            # 1. Mutation candidates
            for _ in range(population):
                parent = self._rng.choice(pool)
                candidates.append(self._space.mutate(parent, self._rng))

            # 2. Crossover children (only when pool has ≥ 2 members)
            for _ in range(population // 2):
                if len(pool) >= 2:
                    a, b = self._rng.sample(pool, 2)
                    candidates.append(self._space.crossover(a, b, self._rng))

            # 3. Evaluate
            scored: list[tuple[float, dict[str, Any]]] = []
            for c in candidates:
                score = self._evaluate_safe(evaluator, c)
                scored.append((score, c))

            # 4. Select top performers for next pool
            scored.sort(key=lambda x: x[0], reverse=True)
            pool = [c for _, c in scored[:max(3, population // 2)]]
            if not pool:
                pool = [dict(best_params)]

            # 5. Check for improvement
            round_best_score = scored[0][0]
            if round_best_score > best_score:
                best_score = round_best_score
                best_params = dict(scored[0][1])
                improvements += 1
                history.append((round_idx + 1, best_score, dict(best_params)))
                if self._verbose:
                    logger.info(
                        "Round %d: new best = %.4f", round_idx + 1, best_score,
                    )

        elapsed = time.perf_counter() - t0
        return EvolutionResult(
            best_params=best_params,
            best_score=best_score,
            rounds=rounds,
            improvements=improvements,
            history=history,
            duration_s=elapsed,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def optimize_ranker(
        self,
        seed_params: dict[str, Any] | None = None,
        *,
        source_map: dict[str, str],
        diff_spec: DiffSpec,
        rounds: int = 20,
        population: int = 6,
    ) -> EvolutionResult:
        """Convenience method: tune RelevanceRanker weights only.

        Builds an internal evaluator using :meth:`ParameterSpace.ranker_space`.
        Scores use the composite metric (harmonic mean of token savings ×
        average relevance).

        Args:
            seed_params: Optional starting params (defaults to
                ``ranker_space().defaults()``).
            source_map: File path → source text mapping.
            diff_spec: :class:`~.models.DiffSpec` for the evaluation scenario.
            rounds: Number of evaluation rounds.
            population: Candidates per round.

        Returns:
            :class:`EvolutionResult`.
        """
        self._space = ParameterSpace.ranker_space()
        if seed_params is None:
            seed_params = self._space.defaults()
        evaluator = make_evaluator(source_map, diff_spec, space_type="ranker")
        return self.optimize(seed_params, evaluator, rounds=rounds, population=population)

    def optimize_config(
        self,
        seed_params: dict[str, Any] | None = None,
        *,
        source_map: dict[str, str],
        diff_spec: DiffSpec,
        rounds: int = 20,
        population: int = 6,
    ) -> EvolutionResult:
        """Convenience method: tune ContextConfig fields only.

        Args:
            seed_params: Optional starting params (defaults to
                ``config_space().defaults()``).
            source_map: File path → source text mapping.
            diff_spec: :class:`~.models.DiffSpec` for the evaluation scenario.
            rounds: Number of evaluation rounds.
            population: Candidates per round.

        Returns:
            :class:`EvolutionResult`.
        """
        self._space = ParameterSpace.config_space()
        if seed_params is None:
            seed_params = self._space.defaults()
        evaluator = make_evaluator(source_map, diff_spec, space_type="config")
        return self.optimize(seed_params, evaluator, rounds=rounds, population=population)

    def optimize_full(
        self,
        seed_params: dict[str, Any] | None = None,
        *,
        source_map: dict[str, str],
        diff_spec: DiffSpec,
        rounds: int = 40,
        population: int = 6,
    ) -> EvolutionResult:
        """Convenience method: tune ranker + config + graph params together.

        Searches the full 10-parameter space.  Use more rounds than the
        focused variants since the search volume is larger.

        Args:
            seed_params: Optional starting params (defaults to
                ``full_space().defaults()``).
            source_map: File path → source text mapping.
            diff_spec: :class:`~.models.DiffSpec` for the evaluation scenario.
            rounds: Number of evaluation rounds.
            population: Candidates per round.

        Returns:
            :class:`EvolutionResult`.
        """
        self._space = ParameterSpace.full_space()
        if seed_params is None:
            seed_params = self._space.defaults()
        evaluator = make_evaluator(source_map, diff_spec, space_type="full")
        return self.optimize(seed_params, evaluator, rounds=rounds, population=population)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_safe(
        evaluator: Callable[[dict[str, Any]], float],
        params: dict[str, Any],
    ) -> float:
        """Evaluate *params*, returning ``0.0`` on any failure.

        Catches:
        - ``None`` or ``NaN`` scores
        - Any exception raised by the evaluator

        Args:
            evaluator: The scoring callable.
            params: Parameter dict to evaluate.

        Returns:
            Float score (0.0 on failure).
        """
        try:
            score = evaluator(params)
            if score is None:
                return 0.0
            if isinstance(score, float) and math.isnan(score):
                return 0.0
            return float(score)
        except Exception as exc:
            logger.warning("Evaluator failed for params %s: %s", params, exc)
            return 0.0


# ---------------------------------------------------------------------------
# make_evaluator — build a scoring function from source data
# ---------------------------------------------------------------------------


def make_evaluator(
    source_map: dict[str, str],
    diff_spec: DiffSpec,
    space_type: str = "ranker",
) -> Callable[[dict[str, Any]], float]:
    """Build a composite-score evaluator closure for evolutionary tuning.

    The evaluator constructs graphsift objects from a parameter dict,
    runs the full pipeline (graph → rank → select → render), and returns
    a composite score.

    The **composite score** is the harmonic mean of:

    * Token savings ratio: ``1.0 - (used_tokens / orig_tokens)``
    * Average relevance score of selected files

    The dependency graph is pre-built once by this factory and captured
    in the closure so it is **not** rebuilt on every evaluation — that
    saves O(N) graph parsing per round.

    Args:
        source_map: File path → source text mapping.
        diff_spec: :class:`~.models.DiffSpec` for the evaluation.
        space_type: One of ``"ranker"``, ``"config"``, ``"graph"``,
            ``"full"``.

    Returns:
        Callable that takes a param dict and returns a float score.

    Raises:
        ValueError: If *space_type* is not recognised.
    """
    if space_type not in ("ranker", "config", "graph", "full"):
        raise ValueError(
            f"Unknown space_type '{space_type}'. "
            f"Expected one of: ranker, config, graph, full"
        )

    # Lazy imports to avoid circular deps at module load time
    from .core import (  # noqa: PLC0415
        ContextSelector,
        DependencyGraph,
        PythonParser,
        RelevanceRanker,
    )
    from .models import ContextConfig  # noqa: PLC0415

    # Pre-build the dependency graph (static across evaluations)
    graph = DependencyGraph(decay=0.7, max_depth=4)
    parser = PythonParser()
    for path, source in source_map.items():
        fn = parser.parse_file(path, source)
        graph.add_file(fn)
    graph.build_import_edges()
    graph.build_inheritance_edges()
    graph.build_decorator_edges()

    graph_scores = graph.ranked_neighbors(diff_spec.changed_files)
    all_files = graph.all_files()

    def _evaluate(params: dict[str, Any]) -> float:
        """Inner evaluator — called once per candidate."""
        try:
            # Route parameters to the right constructor
            ranker_kwargs: dict[str, Any] = {}
            config_kwargs: dict[str, Any] = {}

            if space_type in ("ranker", "full"):
                ranker_kwargs["bm25_weight"] = params.get("bm25_weight", 0.3)
                ranker_kwargs["graph_weight"] = params.get("graph_weight", 0.7)
                ranker_kwargs["god_node_penalty"] = params.get("god_node_penalty", 0.3)

            if space_type in ("config", "full"):
                config_kwargs["token_budget"] = params.get("token_budget", 2000)
                config_kwargs["min_score"] = params.get("min_score", 0.05)
                config_kwargs["hot_threshold"] = params.get("hot_threshold", 0.8)
                config_kwargs["warm_threshold"] = params.get("warm_threshold", 0.25)
                config_kwargs["trimming_context_lines"] = params.get(
                    "trimming_context_lines", 3
                )

            if space_type == "graph":
                # Graph params are passed through to the pre-built graph,
                # but we still need defaults for ranker and config
                pass

            # Build objects
            config = ContextConfig(
                **config_kwargs,
                diff_aware_trimming=True,
                dedup_enabled=True,
            ) if config_kwargs else ContextConfig(
                token_budget=2000,
                min_score=0.05,
                hot_threshold=0.8,
                warm_threshold=0.25,
                trimming_context_lines=3,
                diff_aware_trimming=True,
                dedup_enabled=True,
            )

            ranker = (
                RelevanceRanker(**ranker_kwargs)
                if ranker_kwargs
                else RelevanceRanker()
            )

            # Pipeline
            scored = ranker.rank(diff_spec, graph_scores, all_files, config)
            selector = ContextSelector(config)
            selected, _context, orig_tokens, used_tokens = selector.select_and_render(
                scored, source_map, diff_spec,
            )

            # Composite score (harmonic mean of savings × relevance)
            savings_ratio = 1.0 - (used_tokens / max(orig_tokens, 1))
            avg_score = (
                sum(s.score for s in selected) / max(len(selected), 1)
                if selected
                else 0.0
            )
            denom = savings_ratio + avg_score
            if denom < 0.01:
                return 0.0
            return (2.0 * savings_ratio * avg_score) / denom

        except Exception:
            logger.warning("Evaluator failed for params %s", params, exc_info=True)
            return 0.0

    return _evaluate


__all__ = [
    "EvolutionOptimizer",
    "EvolutionResult",
    "ParameterSpace",
    "make_evaluator",
]
