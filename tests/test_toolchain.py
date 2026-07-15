"""Tests for the ToolChain orchestrator."""

from __future__ import annotations

import pytest

from graphsift.toolchain import ToolChain, ChainState, ChainStep, ChainResult, review_chain, build_chain


class TestToolChain:
    """Tests for the ToolChain orchestrator."""

    def test_empty_chain(self):
        """An empty chain should succeed with no steps."""
        chain = ToolChain("empty", "Empty chain")
        result = chain.run()
        assert result.all_passed
        assert len(result.steps) == 0
        assert "empty" in result.summary

    def test_single_step(self):
        """A single step should execute and succeed."""
        chain = ToolChain("single")
        chain.add_step("step1", lambda: "result", "Single step")
        result = chain.run()
        assert result.all_passed
        assert len(result.steps) == 1
        assert result.steps[0].state == ChainState.SUCCEEDED
        assert result.steps[0].result == "result"
        assert result.total_duration_ms >= 0

    def test_dependency_order(self):
        """Steps with dependencies should execute in order."""
        execution_order = []

        chain = ToolChain("ordered")
        chain.add_step("first", lambda: execution_order.append("first"), "First")
        chain.add_step("second", lambda: execution_order.append("second"), "Second", depends_on=["first"])
        chain.add_step("third", lambda: execution_order.append("third"), "Third", depends_on=["second"])
        result = chain.run()

        assert result.all_passed
        assert execution_order == ["first", "second", "third"]

    def test_parallel_execution(self):
        """Independent steps should execute in parallel (non-blocking)."""
        import threading
        results = {}
        lock = threading.Lock()

        def make_step(name):
            def _step():
                with lock:
                    results[name] = results.get(name, 0) + 1
            return _step

        chain = ToolChain("parallel")
        chain.add_step("a", make_step("a"), "Step A")
        chain.add_step("b", make_step("b"), "Step B")
        chain.add_step("c", make_step("c"), "Step C")
        result = chain.run()

        assert result.all_passed
        assert results == {"a": 1, "b": 1, "c": 1}

    def test_skip_condition(self):
        """A step should be skipped when its skip_if returns True."""
        chain = ToolChain("skip")
        chain.add_step("step1", lambda: "ok", "Step 1")
        chain.add_step("step2", lambda: "unreachable", "Step 2", depends_on=["step1"], skip_if=lambda: True)
        result = chain.run()

        assert result.all_passed
        assert result.steps[1].state == ChainState.SKIPPED
        assert result.steps[1].skipped_reason == "Skip condition met"

    def test_failure_propagation(self):
        """When a step fails, dependents should be skipped."""
        chain = ToolChain("fail")
        chain.add_step("failing", lambda: (_ for _ in ()).throw(RuntimeError("fail")), "Fails")
        chain.add_step("dependent", lambda: "unreachable", "Depends on fail", depends_on=["failing"])
        result = chain.run()

        assert not result.all_passed
        assert result.steps[0].state == ChainState.FAILED
        assert "fail" in result.steps[0].error

    def test_circular_dependency_detection(self):
        """Circular dependencies should raise ValueError."""
        chain = ToolChain("circular")
        chain.add_step("a", lambda: "a", "Step A", depends_on=["b"])
        chain.add_step("b", lambda: "b", "Step B", depends_on=["a"])
        with pytest.raises(ValueError, match="(?i)circular"):
            chain.run()

    def test_unmet_dependency_handling(self):
        """Steps with unmet dependencies should fail."""
        chain = ToolChain("unmet")
        chain.add_step("orphan", lambda: "orphan", "Orphan step", depends_on=["nonexistent"])
        result = chain.run()

        assert not result.all_passed
        assert result.steps[0].state == ChainState.FAILED
        assert "Unmet" in result.steps[0].error

    def test_step_with_error(self):
        """A step that raises should be marked FAILED."""
        chain = ToolChain("error")

        def failing_step():
            raise ValueError("test error")

        chain.add_step("error_step", failing_step, "Error step")
        result = chain.run()

        assert not result.all_passed
        assert result.steps[0].state == ChainState.FAILED
        assert "test error" in result.steps[0].error

    def test_chain_result_properties(self):
        """ChainResult properties should work correctly."""
        chain = ToolChain("props")
        chain.add_step("s1", lambda: "ok", "Step 1")
        chain.add_step("s2", lambda: "ok2", "Step 2", depends_on=["s1"])
        result = chain.run()

        assert "2 passed" in result.summary
        assert result.all_passed

    def test_prebuilt_build_chain(self):
        """Pre-built build_chain should create valid chain."""
        chain = build_chain()
        assert chain.name == "graph-build"
        assert len(chain._steps) == 2

    def test_prebuilt_review_chain(self):
        """Pre-built review_chain should create valid chain."""
        chain = review_chain()
        assert chain.name == "review"
        assert len(chain._steps) >= 3

    def test_double_chaining(self):
        """Chaining add_step should work."""
        chain = ToolChain("chainable")
        chain.add_step("a", lambda: "a", "A").add_step("b", lambda: "b", "B")
        result = chain.run()
        assert result.all_passed
        assert len(result.steps) == 2
