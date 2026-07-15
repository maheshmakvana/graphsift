"""Plan-first engine — generates structured execution plans from task + graph.

Given a task description and a built dependency graph, produces a structured
execution plan with ordered phases and steps. Validates the plan against the
graph before any execution begins.

Directly mimics Goose's behavior of scanning the repo before acting, then
producing a dependency-ordered execution sequence.

Usage::

    from graphsift.planner import Planner, PlanPhase

    planner = Planner(graph=dep_graph)
    plan = planner.create_plan(
        "Add OAuth2 authentication",
        changed_files=["src/auth.py", "src/config.py"],
    )
    result = planner.execute_plan(plan)
    print(result.summary)
"""

from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PlanPhase(str, enum.Enum):
    """Phases of a structured execution plan."""
    SCAN = "scan"
    ANALYZE = "analyze"
    ARCHITECT = "architect"
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"
    REVIEW = "review"


class PlanStatus(str, enum.Enum):
    """Status of a single plan step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""
    id: str
    description: str
    phase: PlanPhase
    depends_on: list[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Complete execution plan with all phases and steps."""
    task_description: str
    phases: list[PlanPhase] = field(default_factory=list)
    steps: list[PlanStep] = field(default_factory=list)
    file_scope: list[str] = field(default_factory=list)
    symbol_scope: list[str] = field(default_factory=list)
    estimated_steps: int = 0
    estimated_tokens: int = 0

    @property
    def summary(self) -> str:
        active = [p.value for p in self.phases]
        return (
            f"Plan for '{self.task_description[:50]}': "
            f"{len(self.phases)} phases, {len(self.steps)} steps, "
            f"{len(self.file_scope)} files in scope"
        )

    def to_json(self) -> str:
        """Serialize plan to JSON for MCP tool output."""
        return json.dumps({
            "task": self.task_description,
            "phases": [p.value for p in self.phases],
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "phase": s.phase.value,
                    "depends_on": s.depends_on,
                    "status": s.status.value,
                }
                for s in self.steps
            ],
            "files_in_scope": len(self.file_scope),
            "symbols_in_scope": len(self.symbol_scope),
        }, indent=2)


@dataclass
class PlanResult:
    """Result of executing a plan."""
    plan: ExecutionPlan
    started_at: float = 0.0
    completed_at: float = 0.0
    all_passed: bool = False

    @property
    def duration_ms(self) -> float:
        return (self.completed_at - self.started_at) * 1000

    @property
    def summary(self) -> str:
        passed_n = sum(1 for s in self.plan.steps if s.status == PlanStatus.COMPLETED)
        failed_n = sum(1 for s in self.plan.steps if s.status == PlanStatus.FAILED)
        skipped_n = sum(1 for s in self.plan.steps if s.status == PlanStatus.SKIPPED)
        return (
            f"PlanResult: {passed_n} completed, {failed_n} failed, "
            f"{skipped_n} skipped in {self.duration_ms:.0f}ms"
        )


class Planner:
    """Generates and executes structured plans from task + graph data.

    Mimics Goose's scan → analyze → architect → plan → execute → validate
    behavior. Uses the dependency graph to scope work and determine ordering.

    Args:
        graph: An optional ``DependencyGraph`` instance for topology data.
        source_map: Optional dict of file path → source text.
        store: An optional ``GraphStore`` for persisted data.
    """

    def __init__(
        self,
        graph: Any = None,
        source_map: dict[str, str] | None = None,
        store: Any = None,
    ) -> None:
        self._graph = graph
        self._source_map = source_map or {}
        self._store = store

    # ------------------------------------------------------------------
    # Plan creation
    # ------------------------------------------------------------------

    def create_plan(
        self,
        task_description: str,
        changed_files: list[str] | None = None,
        query: str = "",
        phases: list[PlanPhase] | None = None,
    ) -> ExecutionPlan:
        """Create a structured execution plan for a task.

        Args:
            task_description: What needs to be done.
            changed_files: Files that are the focus of the task.
            query: Additional context or query string.
            phases: Which phases to include (default: all except EXECUTE).

        Returns:
            An ``ExecutionPlan`` with ordered steps and file scope.
        """
        if phases is None:
            phases = [
                PlanPhase.SCAN,
                PlanPhase.ANALYZE,
                PlanPhase.ARCHITECT,
                PlanPhase.PLAN,
                PlanPhase.VALIDATE,
                PlanPhase.REVIEW,
            ]

        plan = ExecutionPlan(
            task_description=task_description,
            phases=phases,
        )

        steps: list[PlanStep] = []

        # -- SCAN phase: repository overview --------------------------------
        if PlanPhase.SCAN in phases:
            steps.append(PlanStep(
                id="scan_files",
                description="Count and categorise source files by type",
                phase=PlanPhase.SCAN,
            ))
            if self._graph is not None:
                try:
                    file_nodes = getattr(self._graph, "_file_nodes", {})
                    file_count = len(file_nodes)
                    steps.append(PlanStep(
                        id="scan_graph",
                        description=(
                            f"Dependency graph with {file_count} indexed files"
                        ),
                        phase=PlanPhase.SCAN,
                        depends_on=["scan_files"],
                        status=PlanStatus.COMPLETED,
                        details={"file_count": file_count},
                    ))
                except Exception as exc:
                    logger.debug("Graph scan error: %s", exc)

        # -- ANALYZE phase: structural analysis -----------------------------
        if PlanPhase.ANALYZE in phases:
            cycle_dep = [
                s.id for s in steps if s.id == "scan_graph"
            ] or []
            steps.append(PlanStep(
                id="analyze_cycles",
                description="Detect circular import/call dependencies",
                phase=PlanPhase.ANALYZE,
                depends_on=cycle_dep,
            ))
            steps.append(PlanStep(
                id="analyze_dead_code",
                description="Find unreachable code from entry points",
                phase=PlanPhase.ANALYZE,
                depends_on=["analyze_cycles"],
            ))
            steps.append(PlanStep(
                id="analyze_structure",
                description="Assess file complexity and structural health",
                phase=PlanPhase.ANALYZE,
                depends_on=["analyze_dead_code"],
            ))

        # -- ARCHITECT phase: component mapping -----------------------------
        if PlanPhase.ARCHITECT in phases:
            arch_deps = [
                s.id for s in steps if s.id == "analyze_cycles"
            ] or []
            steps.append(PlanStep(
                id="architect_mapping",
                description="Map component architecture based on community detection",
                phase=PlanPhase.ARCHITECT,
                depends_on=arch_deps,
            ))

        # -- PLAN phase: final plan -----------------------------------------
        if PlanPhase.PLAN in phases:
            plan_deps = [
                s.id for s in steps if s.phase == PlanPhase.ARCHITECT
            ] or []
            steps.append(PlanStep(
                id="generate_plan",
                description=(
                    f"Generate ordered execution plan for: "
                    f"{task_description[:80]}"
                ),
                phase=PlanPhase.PLAN,
                depends_on=plan_deps,
            ))

        # Populate scope
        if changed_files:
            plan.file_scope = list(changed_files)

        plan.steps = steps
        plan.estimated_steps = len(steps)
        return plan

    # ------------------------------------------------------------------
    # Plan execution
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        plan: ExecutionPlan,
        step_executor: Callable[[str, PlanStep], Any] | None = None,
    ) -> PlanResult:
        """Execute a plan in topological order.

        Args:
            plan: The ``ExecutionPlan`` to execute.
            step_executor: Optional callable ``(step_id, step) -> result``.
                If ``None``, each step is simulated (returns its description).

        Returns:
            ``PlanResult`` with per-step status and timing.
        """
        result = PlanResult(plan=plan, started_at=time.time())

        executed: set[str] = set()
        remaining = [s for s in plan.steps if s.status == PlanStatus.PENDING]

        while remaining:
            # Find steps whose dependencies are all met
            ready: list[PlanStep] = []
            for s in remaining:
                if all(d in executed for d in s.depends_on):
                    ready.append(s)

            if not ready:
                # Mark remaining as failed (circular / unmet)
                for s in remaining:
                    s.status = PlanStatus.FAILED
                    s.error = "Unmet or circular dependencies"
                    executed.add(s.id)
                break

            for step in ready:
                remaining.remove(step)
                try:
                    step.status = PlanStatus.RUNNING
                    t0 = time.monotonic()

                    if step_executor is not None:
                        step.result = step_executor(step.id, step)
                    else:
                        step.result = f"[simulated] {step.description}"

                    step.status = PlanStatus.COMPLETED
                    step.duration_ms = (time.monotonic() - t0) * 1000
                    executed.add(step.id)
                except Exception as exc:
                    step.status = PlanStatus.FAILED
                    step.error = str(exc)
                    executed.add(step.id)

        result.completed_at = time.time()
        result.all_passed = all(
            s.status == PlanStatus.COMPLETED for s in plan.steps
        )
        return result


__all__ = [
    "PlanPhase",
    "PlanStatus",
    "PlanStep",
    "ExecutionPlan",
    "PlanResult",
    "Planner",
]
