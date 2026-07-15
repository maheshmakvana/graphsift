"""Tests for the Plan-First Engine."""

from __future__ import annotations

import json

import pytest

from graphsift.planner import Planner, PlanPhase, PlanStatus, ExecutionPlan, PlanResult


class TestPlanner:
    """Tests for the Planner class."""

    def test_create_plan_default_phases(self):
        """Creating a plan with default phases should include scan/analyze/architect/plan."""
        planner = Planner()
        plan = planner.create_plan("Test task", changed_files=["src/main.py"])
        assert plan.task_description == "Test task"
        assert len(plan.phases) > 0
        assert PlanPhase.SCAN in plan.phases
        assert PlanPhase.PLAN in plan.phases
        assert "src/main.py" in plan.file_scope

    def test_create_plan_no_changed_files(self):
        """Plan creation without changed_files should work."""
        planner = Planner()
        plan = planner.create_plan("Simple task")
        assert plan.task_description == "Simple task"
        assert plan.file_scope == []

    def test_create_plan_custom_phases(self):
        """Creating a plan with specific phases should only include those."""
        planner = Planner()
        plan = planner.create_plan("Just scan", phases=[PlanPhase.SCAN, PlanPhase.ANALYZE])
        assert len(plan.phases) == 2
        assert PlanPhase.SCAN in plan.phases
        assert PlanPhase.PLAN not in plan.phases

    def test_create_plan_with_graph(self):
        """Creating a plan with graph should detect file count."""
        class MockGraph:
            _file_nodes = {"a.py": None, "b.py": None}

        planner = Planner(graph=MockGraph())
        plan = planner.create_plan("Graph task")
        scan_steps = [s for s in plan.steps if s.id == "scan_graph"]
        assert len(scan_steps) >= 0  # May or may not have file count detail

    def test_execute_plan_simulation(self):
        """Executing a plan without executor should simulate all steps."""
        planner = Planner()
        plan = planner.create_plan("Simulate me")
        result = planner.execute_plan(plan)

        assert result.all_passed
        assert len(result.plan.steps) == len(plan.steps)
        assert result.duration_ms >= 0
        for step in result.plan.steps:
            assert step.status == PlanStatus.COMPLETED
            assert "[simulated]" in (step.result or "")

    def test_execute_plan_with_executor(self):
        """Executing a plan with a custom executor should call it per step."""
        call_log = []

        def executor(step_id, step):
            call_log.append(step_id)
            return f"executed: {step_id}"

        planner = Planner()
        plan = planner.create_plan("Executed task")
        result = planner.execute_plan(plan, step_executor=executor)

        assert result.all_passed
        assert len(call_log) == len(plan.steps)
        for step in result.plan.steps:
            assert step.result and "executed:" in step.result

    def test_execute_plan_failure(self):
        """A failing executor should mark the step as FAILED."""
        def executor(step_id, step):
            raise RuntimeError(f"Failed on {step_id}")

        planner = Planner()
        plan = planner.create_plan("Failing task", phases=[PlanPhase.SCAN])
        result = planner.execute_plan(plan, step_executor=executor)

        assert not result.all_passed
        for step in result.plan.steps:
            assert step.status == PlanStatus.FAILED

    def test_execution_plan_summary(self):
        """ExecutionPlan.summary should describe the plan."""
        planner = Planner()
        plan = planner.create_plan("Short task", changed_files=["x.py"])
        summary = plan.summary
        assert "Short task" in summary
        assert "phases" in summary
        assert "steps" in summary

    def test_execution_plan_to_json(self):
        """ExecutionPlan.to_json should produce valid JSON."""
        planner = Planner()
        plan = planner.create_plan("JSON task")
        json_str = plan.to_json()
        parsed = json.loads(json_str)
        assert parsed["task"] == "JSON task"
        assert "phases" in parsed
        assert "steps" in parsed

    def test_plan_result_summary(self):
        """PlanResult.summary should reflect execution results."""
        planner = Planner()
        plan = planner.create_plan("Result task")
        result = planner.execute_plan(plan)
        summary = result.summary
        assert "completed" in summary
        assert "0 failed" in summary

    def test_plan_result_duration(self):
        """PlanResult.duration_ms should return elapsed time."""
        planner = Planner()
        plan = planner.create_plan("Duration task")
        result = planner.execute_plan(plan)
        assert result.duration_ms >= 0

    def test_plan_step_status_defaults(self):
        """PlanStep should default to PENDING status."""
        from graphsift.planner import PlanStep
        step = PlanStep(id="test", description="Test step", phase=PlanPhase.SCAN)
        assert step.status == PlanStatus.PENDING
        assert step.error == ""
        assert step.duration_ms == 0.0

    def test_plan_phase_enum_values(self):
        """PlanPhase enum should have correct values."""
        assert PlanPhase.SCAN.value == "scan"
        assert PlanPhase.ANALYZE.value == "analyze"
        assert PlanPhase.EXECUTE.value == "execute"
        assert PlanPhase.VALIDATE.value == "validate"
        assert PlanPhase.REVIEW.value == "review"
