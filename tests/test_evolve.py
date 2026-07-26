"""Tests for the evolutionary parameter optimizer (graphsift.evolve)."""

from __future__ import annotations

import math
import random

import pytest

from graphsift.evolve import (
    EvolutionOptimizer,
    EvolutionResult,
    ParameterSpace,
    ParamDef,
)
from graphsift.models import DiffSpec


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _simple_evaluator(params: dict) -> float:
    """Quadratic function with known optimum at x=4, score=20.

    f(x) = -(x - 4)^2 + 20, bounds [-10, 10].
    """
    x = params.get("x", 0.0)
    return -((x - 4) ** 2) + 20.0


_SIMPLE_SPACE = ParameterSpace([ParamDef("x", -10.0, 10.0, float, 0.0)])


# ---------------------------------------------------------------------------
# TestParameterSpace
# ---------------------------------------------------------------------------


class TestParameterSpace:
    """Tests for the ParameterSpace collection and its methods."""

    def test_ranker_space_returns_three_params(self):
        """ranker_space() should return 3 parameters with correct names."""
        space = ParameterSpace.ranker_space()
        assert len(space.params) == 3
        names = {p.name for p in space.params}
        assert names == {"bm25_weight", "graph_weight", "god_node_penalty"}

    def test_ranker_space_has_correct_defaults(self):
        """ranker_space() should have the documented defaults."""
        space = ParameterSpace.ranker_space()
        by_name = {p.name: p for p in space.params}
        assert by_name["bm25_weight"] == ParamDef(
            "bm25_weight", 0.05, 0.8, float, 0.3
        )
        assert by_name["graph_weight"] == ParamDef(
            "graph_weight", 0.2, 0.95, float, 0.7
        )
        assert by_name["god_node_penalty"] == ParamDef(
            "god_node_penalty", 0.05, 0.7, float, 0.3
        )

    def test_config_space_returns_five_params(self):
        """config_space() should return 5 parameters with the expected names."""
        space = ParameterSpace.config_space()
        assert len(space.params) == 5
        names = {p.name for p in space.params}
        assert names == {
            "token_budget",
            "min_score",
            "hot_threshold",
            "warm_threshold",
            "trimming_context_lines",
        }

    def test_config_space_mixed_types(self):
        """config_space() should include int-typed params for budget and lines."""
        space = ParameterSpace.config_space()
        by_name = {p.name: p for p in space.params}
        assert by_name["token_budget"].typ is int
        assert by_name["trimming_context_lines"].typ is int
        assert by_name["min_score"].typ is float

    def test_graph_space_returns_two_params(self):
        """graph_space() should return 2 parameters with correct names."""
        space = ParameterSpace.graph_space()
        assert len(space.params) == 2
        names = {p.name for p in space.params}
        assert names == {"decay", "max_depth"}

    def test_graph_space_int_and_float(self):
        """graph_space() should have float decay and int max_depth."""
        space = ParameterSpace.graph_space()
        by_name = {p.name: p for p in space.params}
        assert by_name["decay"].typ is float
        assert by_name["max_depth"].typ is int

    def test_full_space_combines_all_sub_spaces(self):
        """full_space() should combine ranker + config + graph (10 params)."""
        space = ParameterSpace.full_space()
        assert len(space.params) == 10

    def test_defaults_returns_expected_values(self):
        """defaults() should return a dict of name -> default for each param."""
        space = ParameterSpace.ranker_space()
        assert space.defaults() == {
            "bm25_weight": 0.3,
            "graph_weight": 0.7,
            "god_node_penalty": 0.3,
        }

    def test_defaults_empty_space(self):
        """defaults() on an empty space should return an empty dict."""
        space = ParameterSpace([])
        assert space.defaults() == {}

    def test_sample_stays_within_bounds(self):
        """sample() should produce values inside [low, high] for all params."""
        space = ParameterSpace.ranker_space()
        rng = random.Random(42)
        for _ in range(100):
            candidate = space.sample(rng)
            for p in space.params:
                assert p.low <= candidate[p.name] <= p.high

    def test_sample_returns_correct_types(self):
        """sample() should return int for int-typed params and float for float."""
        space = ParameterSpace.config_space()
        rng = random.Random(1)
        candidate = space.sample(rng)
        assert isinstance(candidate["token_budget"], int)
        assert isinstance(candidate["trimming_context_lines"], int)
        assert isinstance(candidate["min_score"], float)

    def test_sample_preserves_seed_keys(self):
        """sample() with seed= should keep seed keys and fill missing ones."""
        space = ParameterSpace.ranker_space()
        rng = random.Random(1)
        candidate = space.sample(rng, seed={"bm25_weight": 0.5})
        assert candidate["bm25_weight"] == 0.5
        # Unseeded params are still present
        assert "graph_weight" in candidate
        assert "god_node_penalty" in candidate

    def test_sample_empty_space(self):
        """sample() on an empty space should return an empty dict."""
        space = ParameterSpace([])
        rng = random.Random(42)
        assert space.sample(rng) == {}
        assert space.sample(rng, seed={"a": 1}) == {"a": 1}

    def test_sample_reproducible(self):
        """sample() with the same seed and rng should produce identical results."""
        space = ParameterSpace.ranker_space()
        a = space.sample(random.Random(99))
        b = space.sample(random.Random(99))
        assert a == b

    def test_mutate_changes_values_within_bounds(self):
        """mutate() with rate=1.0 should change values but keep them in bounds."""
        space = ParameterSpace.ranker_space()
        rng = random.Random(42)
        original = space.defaults()
        mutated = space.mutate(original, rng, rate=1.0)
        for p in space.params:
            assert p.low <= mutated[p.name] <= p.high
        # High probability that at least one value changed with rate=1.0
        assert mutated != original

    def test_mutate_rate_zero_returns_identical(self):
        """mutate() with rate=0.0 should return the exact same params."""
        space = ParameterSpace.ranker_space()
        rng = random.Random(42)
        original = {"bm25_weight": 0.3, "graph_weight": 0.7, "god_node_penalty": 0.3}
        assert space.mutate(original, rng, rate=0.0) == original

    def test_mutate_empty_space(self):
        """mutate() on an empty space should return the input unchanged."""
        space = ParameterSpace([])
        rng = random.Random(42)
        assert space.mutate({"a": 1}, rng) == {"a": 1}

    def test_crossover_combines_both_parents(self):
        """crossover() child should contain values from both parents."""
        space = ParameterSpace.ranker_space()
        rng = random.Random(42)
        a = {"bm25_weight": 0.1, "graph_weight": 0.9, "god_node_penalty": 0.1}
        b = {"bm25_weight": 0.5, "graph_weight": 0.3, "god_node_penalty": 0.6}
        child = space.crossover(a, b, rng)
        for key in a:
            assert child[key] in (a[key], b[key])

    def test_crossover_identical_parents(self):
        """crossover() with identical parents should produce the same child."""
        space = ParameterSpace.ranker_space()
        rng = random.Random(42)
        parent = {"bm25_weight": 0.5, "graph_weight": 0.5, "god_node_penalty": 0.5}
        assert space.crossover(parent, parent, rng) == parent

    def test_crossover_missing_key_uses_first_parent(self):
        """crossover() falls back to parent 'a' for keys absent in parent 'b'."""
        space = ParameterSpace([])  # crossover doesn't use space.params
        rng = random.Random(42)
        child = space.crossover({"x": 1, "y": 2}, {"x": 10}, rng)
        assert child["x"] in (1, 10)
        assert child["y"] == 2  # from 'a' since 'b' lacks 'y'

    def test_validate_clamps_above_high(self):
        """validate() should clamp values above high to the high bound."""
        space = ParameterSpace.ranker_space()
        params = {"bm25_weight": 99.0, "graph_weight": 0.5, "god_node_penalty": 0.3}
        validated = space.validate(params)
        assert validated["bm25_weight"] == 0.8

    def test_validate_clamps_below_low(self):
        """validate() should clamp values below low to the low bound."""
        space = ParameterSpace.ranker_space()
        params = {"bm25_weight": -10.0, "graph_weight": 0.5, "god_node_penalty": 0.3}
        validated = space.validate(params)
        assert validated["bm25_weight"] == 0.05

    def test_validate_casts_int_types(self):
        """validate() should round float to int for int-typed params."""
        space = ParameterSpace.config_space()
        params = {
            "token_budget": 1234.56,
            "min_score": 0.1,
            "hot_threshold": 0.8,
            "warm_threshold": 0.25,
            "trimming_context_lines": 5,
        }
        validated = space.validate(params)
        assert isinstance(validated["token_budget"], int)
        assert validated["token_budget"] == 1234

    def test_validate_passes_unknown_keys(self):
        """validate() should forward unknown parameter keys unchanged."""
        space = ParameterSpace.ranker_space()
        params = {
            "bm25_weight": 0.3,
            "graph_weight": 0.7,
            "god_node_penalty": 0.3,
            "unknown_key": "some_value",
        }
        validated = space.validate(params)
        assert validated["unknown_key"] == "some_value"

    def test_validate_empty_space(self):
        """validate() on an empty space should pass all keys through."""
        space = ParameterSpace([])
        assert space.validate({"x": 1, "y": 2.5}) == {"x": 1, "y": 2.5}

    def test_single_param_space_works(self):
        """A single-param ParameterSpace should support all operations."""
        space = ParameterSpace([ParamDef("x", 0.0, 10.0, float, 5.0)])
        assert space.defaults() == {"x": 5.0}
        rng = random.Random(42)
        s = space.sample(rng)
        assert 0.0 <= s["x"] <= 10.0
        m = space.mutate({"x": 5.0}, rng, rate=1.0)
        assert 0.0 <= m["x"] <= 10.0
        child = space.crossover({"x": 1.0}, {"x": 9.0}, rng)
        assert child["x"] in (1.0, 9.0)
        v = space.validate({"x": 100.0})
        assert v["x"] == 10.0
        v = space.validate({"x": -1.0})
        assert v["x"] == 0.0

    def test_reproducible_sequence(self):
        """Same seed should produce identical sequences across multiple calls."""
        space = ParameterSpace.ranker_space()
        rng_a = random.Random(123)
        rng_b = random.Random(123)
        for _ in range(20):
            assert space.sample(rng_a) == space.sample(rng_b)


# ---------------------------------------------------------------------------
# TestEvolutionResult
# ---------------------------------------------------------------------------


class TestEvolutionResult:
    """Tests for the EvolutionResult dataclass."""

    def test_construction(self):
        """EvolutionResult should store all constructor args."""
        result = EvolutionResult(
            best_params={"x": 0.5},
            best_score=0.95,
            rounds=10,
            improvements=3,
            history=[(2, 0.8, {"x": 0.3}), (5, 0.9, {"x": 0.4})],
            duration_s=1.23,
        )
        assert result.best_params == {"x": 0.5}
        assert result.best_score == 0.95
        assert result.rounds == 10
        assert result.improvements == 3
        assert len(result.history) == 2
        assert result.duration_s == 1.23

    def test_summary_contains_score(self):
        """summary() should include the best_score value."""
        result = EvolutionResult(
            best_params={"x": 1.0},
            best_score=0.8765,
            rounds=20,
            improvements=4,
            duration_s=3.5,
        )
        summary = result.summary()
        assert "0.8765" in summary
        assert "4/20" in summary
        assert "3.5" in summary

    def test_summary_rounds_score(self):
        """summary() should round best_score to 4 decimal places."""
        result = EvolutionResult(
            best_params={"x": 1.0},
            best_score=0.123456789,
            rounds=5,
            improvements=1,
            duration_s=0.5,
        )
        assert "0.1235" in result.summary()

    def test_empty_history(self):
        """EvolutionResult with no history should work correctly."""
        result = EvolutionResult(
            best_params={"x": 0.5},
            best_score=0.8,
            rounds=5,
            improvements=0,
        )
        assert result.history == []
        assert "0/5" in result.summary()

    def test_history_with_data(self):
        """History should store (round, score, params) tuples."""
        result = EvolutionResult(
            best_params={"x": 0.9},
            best_score=0.9,
            rounds=10,
            improvements=2,
            history=[
                (3, 0.6, {"x": 0.5}),
                (7, 0.9, {"x": 0.9}),
            ],
            duration_s=2.0,
        )
        assert result.history[0] == (3, 0.6, {"x": 0.5})
        assert result.history[1] == (7, 0.9, {"x": 0.9})

    def test_duration_defaults_to_zero(self):
        """duration_s should default to 0.0."""
        result = EvolutionResult(
            best_params={"x": 0.5},
            best_score=0.8,
            rounds=5,
            improvements=0,
        )
        assert result.duration_s == 0.0


# ---------------------------------------------------------------------------
# TestEvolutionOptimizer
# ---------------------------------------------------------------------------


class TestEvolutionOptimizer:
    """Tests for the evolutionary optimizer main loop."""

    def test_constructor(self):
        """Constructor should store the space and seed."""
        space = _SIMPLE_SPACE
        opt = EvolutionOptimizer(space, seed=42, verbose=False)
        assert opt._space is space
        assert opt._rng is not None

    def test_optimize_raises_on_small_population(self):
        """population < 2 should raise ValueError."""
        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        with pytest.raises(ValueError, match="population"):
            opt.optimize({"x": 0.0}, _simple_evaluator, rounds=5, population=1)

    def test_optimize_raises_on_zero_rounds(self):
        """rounds < 1 should raise ValueError."""
        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        with pytest.raises(ValueError, match="rounds"):
            opt.optimize({"x": 0.0}, _simple_evaluator, rounds=0, population=3)

    def test_optimize_raises_on_negative_rounds(self):
        """Negative rounds should raise ValueError."""
        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        with pytest.raises(ValueError, match="rounds"):
            opt.optimize({"x": 0.0}, _simple_evaluator, rounds=-1, population=3)

    def test_optimize_improves_from_suboptimal_seed(self):
        """Optimizer should find a better score than the suboptimal seed."""
        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        seed_params = {"x": -5.0}  # f(-5) = -61, optimum at x=4 gives f(4)=20
        result = opt.optimize(
            seed_params, _simple_evaluator, rounds=15, population=5,
        )
        seed_score = _simple_evaluator(seed_params)  # -61
        assert result.best_score > seed_score, (
            f"Expected improvement over {seed_score}, got {result.best_score}"
        )

    def test_optimize_approaches_optimum(self):
        """Optimizer with enough rounds should get close to the known optimum.

        f(x) = -(x-4)^2 + 20 has optimum score=20 at x=4. With rounds=20,
        population=6, the optimizer should reach at least 75% of optimum.
        """
        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        result = opt.optimize(
            {"x": -5.0}, _simple_evaluator, rounds=20, population=6,
        )
        # 75% of optimum 20 = 15
        assert result.best_score > 15.0, (
            f"Expected score > 15.0, got {result.best_score}"
        )

    def test_history_tracks_improvements(self):
        """optimize() should record improvements in history."""
        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        result = opt.optimize(
            {"x": -5.0}, _simple_evaluator, rounds=15, population=5,
        )
        assert result.improvements > 0
        assert len(result.history) == result.improvements
        if result.history:
            round_num, score, params = result.history[0]
            assert isinstance(round_num, int)
            assert isinstance(score, float)
            assert "x" in params

    def test_evaluate_safe_handles_exception(self):
        """_evaluate_safe should return 0.0 when evaluator raises."""
        def broken(params):
            raise RuntimeError("evaluator crashed")

        score = EvolutionOptimizer._evaluate_safe(broken, {})
        assert score == 0.0

    def test_evaluate_safe_handles_none(self):
        """_evaluate_safe should return 0.0 when evaluator returns None."""
        def none_evaluator(params):
            return None

        score = EvolutionOptimizer._evaluate_safe(none_evaluator, {})
        assert score == 0.0

    def test_evaluate_safe_handles_nan(self):
        """_evaluate_safe should return 0.0 when evaluator returns NaN."""
        def nan_evaluator(params):
            return float("nan")

        score = EvolutionOptimizer._evaluate_safe(nan_evaluator, {})
        assert score == 0.0

    def test_evaluate_safe_passes_through_valid(self):
        """_evaluate_safe should return the score for a valid evaluator."""
        def valid(params):
            return 42.5

        score = EvolutionOptimizer._evaluate_safe(valid, {})
        assert score == 42.5

    def test_optimizer_handles_flaky_evaluator(self):
        """Optimizer should not crash when evaluator occasionally fails."""
        call_count = [0]

        def flaky(params):
            call_count[0] += 1
            if call_count[0] % 3 == 0:  # every 3rd call fails
                raise ValueError("transient failure")
            return _simple_evaluator(params)

        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        # Should complete without raising
        result = opt.optimize(
            {"x": -5.0}, flaky, rounds=10, population=4,
        )
        assert isinstance(result, EvolutionResult)
        assert result.best_score > _simple_evaluator({"x": -5.0})

    def test_reproducible_results(self):
        """Same seed should produce identical results."""
        seed = 42
        opt_a = EvolutionOptimizer(_SIMPLE_SPACE, seed=seed)
        opt_b = EvolutionOptimizer(_SIMPLE_SPACE, seed=seed)

        result_a = opt_a.optimize(
            {"x": -5.0}, _simple_evaluator, rounds=10, population=5,
        )
        result_b = opt_b.optimize(
            {"x": -5.0}, _simple_evaluator, rounds=10, population=5,
        )

        assert result_a.best_score == result_b.best_score
        assert result_a.best_params == result_b.best_params
        assert result_a.improvements == result_b.improvements
        assert result_a.history == result_b.history

    def test_different_seed_different_results(self):
        """Different seeds should (likely) produce different trajectories."""
        opt_a = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        opt_b = EvolutionOptimizer(_SIMPLE_SPACE, seed=999)

        result_a = opt_a.optimize(
            {"x": -5.0}, _simple_evaluator, rounds=10, population=5,
        )
        result_b = opt_b.optimize(
            {"x": -5.0}, _simple_evaluator, rounds=10, population=5,
        )

        # Very unlikely that two seeds produce identical trajectories
        # with evolutionary search
        different = (
            result_a.best_params != result_b.best_params
            or result_a.history != result_b.history
        )
        assert different, (
            "Two different seeds unexpectedly produced identical results"
        )

    def test_returns_evolution_result_type(self):
        """optimize() should return an EvolutionResult instance."""
        opt = EvolutionOptimizer(_SIMPLE_SPACE, seed=42)
        result = opt.optimize(
            {"x": 0.0}, _simple_evaluator, rounds=5, population=3,
        )
        assert isinstance(result, EvolutionResult)
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_score, float)
        assert isinstance(result.rounds, int)
        assert isinstance(result.improvements, int)
        assert isinstance(result.duration_s, float)

    def test_optimize_validates_seed_params(self):
        """optimize() should validate/clamp out-of-bounds seed params."""
        space = ParameterSpace([ParamDef("x", 0.0, 10.0, int, 5)])
        opt = EvolutionOptimizer(space, seed=42)

        # x=100 is out of bounds, should be clamped to 10
        result = opt.optimize(
            {"x": 100}, _simple_evaluator, rounds=5, population=3,
        )
        # The evaluator uses float internally, but the param should be clamped
        # Note: _simple_evaluator accesses params["x"] as float, but
        # validate converts int to int. The evaluator still works.
        assert result.best_params["x"] <= 10

    # ------------------------------------------------------------------
    # End-to-end integration with inline source_map
    # ------------------------------------------------------------------

    def _make_test_source_map(self) -> dict[str, str]:
        """Create a small 3-file Python source map with cross-imports.

        Files:
            user.py       — User class + get_user helper
            auth.py       — login/logout that import from user
            test_auth.py  — tests that import from auth
        """
        return {
            "user.py": (
                "class User:\n"
                "    pass\n"
                "\n"
                "def get_user(user_id):\n"
                "    pass\n"
            ),
            "auth.py": (
                "from user import User, get_user\n"
                "\n"
                "def login(username, password):\n"
                "    pass\n"
                "\n"
                "def logout(username):\n"
                "    pass\n"
            ),
            "test_auth.py": (
                "from auth import login, logout\n"
                "\n"
                "def test_login():\n"
                "    pass\n"
                "\n"
                "def test_logout():\n"
                "    pass\n"
            ),
        }

    def _make_test_diff_spec(self) -> DiffSpec:
        """Create a DiffSpec targeting the auth module."""
        return DiffSpec(
            changed_files=["auth.py"],
            diff_text="Changes to login function",
            commit_message="Add login functionality",
            query="Review auth changes",
        )

    def test_end_to_end_ranker_space(self):
        """End-to-end optimize_ranker with 3-file source map should complete.

        This test exercises the full pipeline: make_evaluator builds a
        DependencyGraph from the source map, the evolutionary loop calls
        the evaluator with different ranker weights, and returns an
        EvolutionResult with tuned parameters.
        """
        source_map = self._make_test_source_map()
        diff_spec = self._make_test_diff_spec()

        opt = EvolutionOptimizer(ParameterSpace.ranker_space(), seed=42)
        result = opt.optimize_ranker(
            source_map=source_map,
            diff_spec=diff_spec,
            rounds=5,
            population=3,
        )

        assert isinstance(result, EvolutionResult)
        assert len(result.best_params) == 3
        assert "bm25_weight" in result.best_params
        assert "graph_weight" in result.best_params
        assert "god_node_penalty" in result.best_params
        assert result.best_score >= 0.0
        assert result.rounds == 5
        assert result.improvements >= 0
        assert result.duration_s > 0.0

    def test_end_to_end_config_space(self):
        """End-to-end optimize_config with 3-file source map should complete."""
        source_map = self._make_test_source_map()
        diff_spec = self._make_test_diff_spec()

        opt = EvolutionOptimizer(ParameterSpace.config_space(), seed=42)
        result = opt.optimize_config(
            source_map=source_map,
            diff_spec=diff_spec,
            rounds=5,
            population=3,
        )

        assert isinstance(result, EvolutionResult)
        assert len(result.best_params) == 5
        assert "token_budget" in result.best_params
        assert "min_score" in result.best_params
        assert result.rounds == 5

    def test_end_to_end_full_space(self):
        """End-to-end optimize_full with 3-file source map should complete."""
        source_map = self._make_test_source_map()
        diff_spec = self._make_test_diff_spec()

        opt = EvolutionOptimizer(ParameterSpace.full_space(), seed=42)
        result = opt.optimize_full(
            source_map=source_map,
            diff_spec=diff_spec,
            rounds=5,
            population=3,
        )

        assert isinstance(result, EvolutionResult)
        assert len(result.best_params) == 10
        assert result.rounds == 5

    def test_end_to_end_ranker_reproducible(self):
        """End-to-end optimize_ranker should be reproducible with same seed."""
        source_map = self._make_test_source_map()
        diff_spec = self._make_test_diff_spec()

        opt_a = EvolutionOptimizer(ParameterSpace.ranker_space(), seed=42)
        opt_b = EvolutionOptimizer(ParameterSpace.ranker_space(), seed=42)

        result_a = opt_a.optimize_ranker(
            source_map=source_map,
            diff_spec=diff_spec,
            rounds=5,
            population=3,
        )
        result_b = opt_b.optimize_ranker(
            source_map=source_map,
            diff_spec=diff_spec,
            rounds=5,
            population=3,
        )

        assert result_a.best_score == result_b.best_score
        assert result_a.best_params == result_b.best_params
        assert result_a.improvements == result_b.improvements
        assert result_a.history == result_b.history

    def test_end_to_end_with_custom_seed_params(self):
        """optimize_ranker should accept custom seed_params."""
        source_map = self._make_test_source_map()
        diff_spec = self._make_test_diff_spec()

        opt = EvolutionOptimizer(ParameterSpace.ranker_space(), seed=42)
        custom_seed = {"bm25_weight": 0.5, "graph_weight": 0.5, "god_node_penalty": 0.5}
        result = opt.optimize_ranker(
            seed_params=custom_seed,
            source_map=source_map,
            diff_spec=diff_spec,
            rounds=5,
            population=3,
        )

        assert isinstance(result, EvolutionResult)
        assert result.best_score >= 0.0
