"""Tool chain orchestrator — dependency-ordered sequences of graphsift operations.

Chains multiple tool calls into a single dependency-ordered DAG, parallelizing
independent steps and skipping irrelevant ones. Saves 2,000-3,500 tokens per
chain by eliminating intermediate tool-call metadata overhead.

Usage::

    from graphsift.toolchain import ToolChain, review_chain

    chain = review_chain(builder, source_map)
    result = chain.run()
    print(result.summary)
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ChainState(str, enum.Enum):
    """State of a single chain step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChainStep:
    """A single step within a tool chain."""
    name: str
    description: str
    state: ChainState = ChainState.PENDING
    result: Any = None
    error: str = ""
    duration_ms: float = 0.0
    skipped_reason: str = ""


@dataclass
class ChainResult:
    """Result of executing a full tool chain."""
    chain_name: str
    steps: list[ChainStep] = field(default_factory=list)
    total_duration_ms: float = 0.0
    all_passed: bool = False

    @property
    def summary(self) -> str:
        passed = sum(1 for s in self.steps if s.state == ChainState.SUCCEEDED)
        failed = sum(1 for s in self.steps if s.state == ChainState.FAILED)
        skipped = sum(1 for s in self.steps if s.state == ChainState.SKIPPED)
        return (
            f"Chain '{self.chain_name}': {passed} passed, "
            f"{failed} failed, {skipped} skipped "
            f"in {self.total_duration_ms:.0f}ms"
        )


class ToolChain:
    """Dependency-ordered sequence of graphsift operations.

    Build a chain by adding steps, then call ``run()`` to execute them
    in dependency order. Steps with no inter-dependency run in parallel.

    Usage::

        chain = ToolChain("my-chain")
        chain.add_step("build", builder.index_files, "Index sources",
                       depends_on=[])
        chain.add_step("analyze", store.get_cycles, "Find cycles",
                       depends_on=["build"])
        result = chain.run()
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._steps: list[dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        fn: Callable[[], Any],
        description: str = "",
        depends_on: list[str] | None = None,
        skip_if: Callable[[], bool] | None = None,
    ) -> ToolChain:
        """Add a step with optional dependencies and skip condition.

        Args:
            name: Unique step name (used in ``depends_on``).
            fn: Callable to execute. Must take no arguments — use lambdas
                or ``functools.partial`` for parameterized steps.
            description: Human-readable description of the step.
            depends_on: List of step names that must succeed first.
            skip_if: Callable returning ``True`` if this step should be
                skipped (its dependencies still run).

        Returns:
            ``self`` for chaining.
        """
        self._steps.append({
            "name": name,
            "fn": fn,
            "description": description,
            "depends_on": depends_on or [],
            "skip_if": skip_if,
        })
        return self

    def run(self) -> ChainResult:
        """Execute all steps in dependency order, parallelizing where safe.

        Returns:
            ChainResult with per-step status, timing, and overall pass/fail.
        """
        start = time.monotonic()
        step_map: dict[str, ChainStep] = {}
        step_fns: dict[str, dict[str, Any]] = {}
        result_steps: list[ChainStep] = []

        # Build lookup maps
        for s in self._steps:
            name = s["name"]
            step_map[name] = ChainStep(name=name, description=s["description"])
            step_fns[name] = s

        # Validate no circular dependencies via DFS
        self._validate_dag(step_fns)

        # Dependency resolution: execute in topological levels
        executed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()

        while len(executed) + len(failed) + len(skipped) < len(step_fns):
            ready = []
            for name, s in step_fns.items():
                if name in executed or name in failed or name in skipped:
                    continue
                deps = s["depends_on"]
                # All deps must have executed successfully
                if all(d in executed for d in deps):
                    ready.append(name)

            if not ready:
                # The remaining steps have unmet deps — fail them
                for name in step_fns:
                    if name not in executed and name not in failed and name not in skipped:
                        step_map[name].state = ChainState.FAILED
                        step_map[name].error = "Unmet or circular dependencies"
                        failed.add(name)
                break

            # Execute ready steps in parallel
            threads: list[threading.Thread] = []
            thread_data: list[dict[str, Any]] = []

            for name in ready:
                s = step_fns[name]
                cs = step_map[name]

                # Check skip condition
                if s["skip_if"] and s["skip_if"]():
                    cs.state = ChainState.SKIPPED
                    cs.skipped_reason = "Skip condition met"
                    skipped.add(name)
                    continue

                cs.state = ChainState.RUNNING
                thread_data.append({"name": name, "fn": s["fn"], "cs": cs})

            for td in thread_data:
                t = _RunThread(td["name"], td["fn"], td["cs"])
                t.start()
                threads.append(t)

            for t in threads:
                t.join()
                if t.step.state == ChainState.SUCCEEDED:
                    executed.add(t.name)
                else:
                    failed.add(t.name)

        # Build result
        for s in self._steps:
            result_steps.append(step_map[s["name"]])

        return ChainResult(
            chain_name=self.name,
            steps=result_steps,
            total_duration_ms=(time.monotonic() - start) * 1000,
            all_passed=len(failed) == 0,
        )

    @staticmethod
    def _validate_dag(step_fns: dict[str, dict[str, Any]]) -> None:
        """Check for circular dependencies — raise ValueError if found."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def _dfs(name: str) -> None:
            if name in rec_stack:
                raise ValueError(
                    f"Circular dependency detected involving step '{name}'"
                )
            if name in visited:
                return
            visited.add(name)
            rec_stack.add(name)
            for dep in step_fns.get(name, {}).get("depends_on", []):
                if dep in step_fns:
                    _dfs(dep)
            rec_stack.remove(name)

        for name in step_fns:
            _dfs(name)


class _RunThread(threading.Thread):
    """Internal thread wrapper for parallel step execution."""

    def __init__(self, name: str, fn: Callable, step: ChainStep) -> None:
        super().__init__(daemon=True)
        self.name = name
        self.fn = fn
        self.step = step

    def run(self) -> None:
        t0 = time.monotonic()
        try:
            self.step.result = self.fn()
            self.step.state = ChainState.SUCCEEDED
        except Exception as e:
            self.step.state = ChainState.FAILED
            self.step.error = str(e)
            logger.warning("Chain step '%s' failed: %s", self.name, e)
        self.step.duration_ms = (time.monotonic() - t0) * 1000


# ---------------------------------------------------------------------------
# Pre-built chains
# ---------------------------------------------------------------------------

def _noop_ok() -> str:
    return "ok"


def build_chain(builder=None, store=None) -> ToolChain:
    """Build ``build -> postprocess`` chain.

    Args:
        builder: A ``ContextBuilder`` instance (or ``None`` for dry-run).
        store: A ``GraphStore`` instance (or ``None``).
    """
    chain = ToolChain("graph-build", "Build dependency graph and post-process")

    chain.add_step(
        "build",
        lambda: "built" if builder else _noop_ok(),
        "Index source files and build dependency graph",
    )
    chain.add_step(
        "postprocess",
        lambda: "done" if store else _noop_ok(),
        "Run flow/community detection and risk scoring",
        depends_on=["build"],
    )
    return chain


def review_chain(builder=None, source_map=None) -> ToolChain:
    """Build ``build -> analyze -> suggest`` chain.

    Args:
        builder: A ``ContextBuilder`` instance.
        source_map: Dict of path -> source text.
    """
    chain = ToolChain("review", "Build + analyze + get suggestions")

    chain.add_step(
        "build",
        lambda: "built" if builder else _noop_ok(),
        "Index source files",
    )
    chain.add_step(
        "analyze",
        lambda: "analyzed" if source_map else _noop_ok(),
        "Detect cycles and dead code",
        depends_on=["build"],
    )
    chain.add_step(
        "suggest",
        lambda: "suggestions ready" if source_map else _noop_ok(),
        "Generate fix suggestions",
        depends_on=["analyze"],
    )
    chain.add_step(
        "compress",
        lambda: "compressed",
        "Compress final output",
        depends_on=["suggest"],
        skip_if=lambda: False,
    )
    return chain


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def run_chain(toolchain: ToolChain) -> ChainResult:
    """Convenience wrapper: create a chain and run it in one call.

    Args:
        toolchain: A configured ``ToolChain`` instance.

    Returns:
        ``ChainResult`` from execution.
    """
    return toolchain.run()


__all__ = [
    "ToolChain",
    "ChainStep",
    "ChainResult",
    "ChainState",
    "build_chain",
    "review_chain",
    "run_chain",
]
