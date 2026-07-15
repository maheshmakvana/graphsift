"""Tests for v2.4+ model additions."""

from __future__ import annotations

from graphsift.models import BudgetMode, PruningStrategy, ContextConfig


class TestBudgetMode:
    """Tests for the BudgetMode enum."""

    def test_enum_values(self):
        """BudgetMode should have correct values."""
        assert BudgetMode.FIXED.value == "fixed"
        assert BudgetMode.ADAPTIVE.value == "adaptive"
        assert BudgetMode.PER_PHASE.value == "per_phase"

    def test_enum_members(self):
        """BudgetMode should have 3 members."""
        assert len(BudgetMode) == 3

    def test_from_string(self):
        """BudgetMode should be constructable from string."""
        assert BudgetMode("fixed") == BudgetMode.FIXED
        assert BudgetMode("adaptive") == BudgetMode.ADAPTIVE


class TestPruningStrategy:
    """Tests for the PruningStrategy enum."""

    def test_enum_values(self):
        """PruningStrategy should have correct values."""
        assert PruningStrategy.NONE.value == "none"
        assert PruningStrategy.LIGHT.value == "light"
        assert PruningStrategy.BALANCED.value == "balanced"
        assert PruningStrategy.AGGRESSIVE.value == "aggressive"

    def test_enum_members(self):
        """PruningStrategy should have 4 members."""
        assert len(PruningStrategy) == 4

    def test_from_string(self):
        """PruningStrategy should be constructable from string."""
        assert PruningStrategy("none") == PruningStrategy.NONE
        assert PruningStrategy("aggressive") == PruningStrategy.AGGRESSIVE


class TestContextConfigExtensions:
    """Tests for new ContextConfig fields."""

    def test_budget_mode_default(self):
        """ContextConfig should default to FIXED budget mode."""
        config = ContextConfig()
        assert config.budget_mode == BudgetMode.FIXED

    def test_pruning_strategy_default(self):
        """ContextConfig should default to NONE pruning."""
        config = ContextConfig()
        assert config.pruning_strategy == PruningStrategy.NONE

    def test_adaptive_budget_config(self):
        """ContextConfig should accept adaptive budgeting."""
        config = ContextConfig(
            token_budget=50000,
            budget_mode=BudgetMode.ADAPTIVE,
        )
        assert config.budget_mode == BudgetMode.ADAPTIVE
        assert config.centrality_weight == 0.3

    def test_adaptive_budget_custom_weight(self):
        """ContextConfig should accept custom centrality weight."""
        config = ContextConfig(
            token_budget=50000,
            budget_mode=BudgetMode.ADAPTIVE,
            centrality_weight=0.5,
        )
        assert config.centrality_weight == 0.5

    def test_light_pruning_config(self):
        """ContextConfig should accept LIGHT pruning."""
        config = ContextConfig(pruning_strategy=PruningStrategy.LIGHT)
        assert config.pruning_strategy == PruningStrategy.LIGHT

    def test_aggressive_pruning_config(self):
        """ContextConfig should accept AGGRESSIVE pruning."""
        config = ContextConfig(pruning_strategy=PruningStrategy.AGGRESSIVE)
        assert config.pruning_strategy == PruningStrategy.AGGRESSIVE

    def test_overlap_threshold_default(self):
        """ContextConfig should default overlap_threshold to 0.15."""
        config = ContextConfig()
        assert config.overlap_threshold == 0.15

    def test_custom_overlap_threshold(self):
        """ContextConfig should accept custom overlap_threshold."""
        config = ContextConfig(overlap_threshold=0.25)
        assert config.overlap_threshold == 0.25

    def test_all_new_fields_combined(self):
        """ContextConfig should accept all new fields together."""
        config = ContextConfig(
            token_budget=100000,
            budget_mode=BudgetMode.PER_PHASE,
            pruning_strategy=PruningStrategy.BALANCED,
            centrality_weight=0.4,
            overlap_threshold=0.2,
        )
        assert config.budget_mode == BudgetMode.PER_PHASE
        assert config.pruning_strategy == PruningStrategy.BALANCED
        assert config.centrality_weight == 0.4
        assert config.overlap_threshold == 0.2
        assert config.token_budget == 100000

    def test_backward_compatibility(self):
        """ContextConfig should work with only original fields."""
        config = ContextConfig(token_budget=50000)
        assert config.token_budget == 50000
        assert config.budget_mode == BudgetMode.FIXED  # new field with default
        assert config.pruning_strategy == PruningStrategy.NONE  # new field with default
