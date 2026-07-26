"""Harness engineering module for graphsift -- pre/post-tool-use validation hooks.

Provides:
    - HarnessHook base class and concrete implementations
    - Harness orchestrator for hook lifecycle
    - DriftDetector for agent behavior analysis
    - HarnessStats for tracking validation outcomes

2026 Context: Models are commoditized; the system that constrains, verifies,
and corrects the agent is where competitive advantage lies.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ===========================================================================
# Validation Hooks
# ===========================================================================


class HarnessHook:
    """Base class for validation hooks.

    Subclass and override the methods you need. Each method returns a bool:
    ``True`` to continue, ``False`` to abort or flag issues.
    """

    def pre_build(self, diff_spec, source_map) -> bool:
        """Called before context building. Return False to abort.

        Args:
            diff_spec: The DiffSpec for the pending build.
            source_map: Dict mapping file path to source text.

        Returns:
            True to proceed, False to abort the build.
        """
        return True

    def post_build(self, result, diff_spec) -> bool:
        """Called after context building. Return False to flag issues.

        Args:
            result: The ContextResult from the build.
            diff_spec: The original DiffSpec.

        Returns:
            True if the result passes validation, False if issues were found.
        """
        return True

    def pre_index(self, root_path: str) -> bool:
        """Called before indexing. Return False to abort.

        Args:
            root_path: Repository root path to be indexed.

        Returns:
            True to proceed, False to abort indexing.
        """
        return True

    def post_index(self, stats) -> bool:
        """Called after indexing. Return False to flag issues.

        Args:
            stats: IndexStats or dict with indexing results.

        Returns:
            True if indexing passed validation, False if issues were found.
        """
        return True


class GraphIntegrityHook(HarnessHook):
    """Validates graph integrity after indexing.

    Checks for:
    - Excessive dependency cycles (via Tarjan's SCC)
    - Orphan nodes (symbols with no edges)
    - Unreasonable edge-to-node ratios

    Args:
        graph: Optional DependencyGraph instance for cycle detection.
        max_dangling_edges: Maximum allowed problematic edges before warning.
        max_cycle_depth: Maximum cycle chain depth before flagging.
    """

    def __init__(
        self,
        graph: Any = None,
        max_dangling_edges: int = 50,
        max_cycle_depth: int = 100,
    ) -> None:
        self._graph = graph
        self._max_dangling_edges = max_dangling_edges
        self._max_cycle_depth = max_cycle_depth

    def post_index(self, stats) -> bool:
        """Check for dangling edges, orphan nodes, excessive cycles.

        Args:
            stats: IndexStats or dict containing ``edges_created``,
                   ``symbols_extracted``, ``files_indexed``.

        Returns:
            False if integrity issues exceed thresholds.
        """
        passed = True

        # Extract numeric fields whether stats is IndexStats or a plain dict
        edges = getattr(stats, "edges_created", None)
        if edges is None and isinstance(stats, dict):
            edges = stats.get("edges_created", 0)
        edges = edges or 0

        symbols = getattr(stats, "symbols_extracted", None)
        if symbols is None and isinstance(stats, dict):
            symbols = stats.get("symbols_extracted", 0)
        symbols = symbols or 0

        files = getattr(stats, "files_indexed", None)
        if files is None and isinstance(stats, dict):
            files = stats.get("files_indexed", 0)
        files = files or 0

        # Check edge-to-symbol ratio (too many edges = noisy graph)
        if symbols > 0 and edges > 0:
            ratio = edges / symbols
            if ratio > 10:
                logger.warning(
                    "GraphIntegrityHook: high edge-to-symbol ratio %.2f "
                    "(%d edges, %d symbols) — possible overconnection",
                    ratio, edges, symbols,
                )
                passed = False
            elif ratio < 0.01 and files > 10:
                logger.warning(
                    "GraphIntegrityHook: very low edge-to-symbol ratio %.4f "
                    "(%d edges, %d symbols) — possible underconnection",
                    ratio, edges, symbols,
                )
                passed = False

        # Check for orphan symbols (files with no edges)
        if files > 0 and edges == 0 and symbols > 0:
            logger.warning(
                "GraphIntegrityHook: %d symbols indexed but zero edges created",
                symbols,
            )
            passed = False

        # Cycle detection via graph
        if self._graph is not None:
            try:
                cycles = self._graph.detect_cycles()
                if len(cycles) > self._max_dangling_edges:
                    logger.warning(
                        "GraphIntegrityHook: %d dependency cycles detected "
                        "(threshold: %d)",
                        len(cycles), self._max_dangling_edges,
                    )
                    passed = False
                else:
                    logger.debug(
                        "GraphIntegrityHook: %d cycles detected (within threshold)",
                        len(cycles),
                    )
            except Exception as exc:
                logger.debug("GraphIntegrityHook: cycle detection failed: %s", exc)

        return passed


class BudgetEnforcementHook(HarnessHook):
    """Hard token budget enforcement with customizable overflow behavior.

    Args:
        hard_limit: Maximum allowed tokens. If None, no limit is enforced.
        overflow_action: How to handle overflow:

            - ``"truncate"``: Log warning and proceed (return True).
            - ``"error"``: Return False to block the build.
            - ``"warn"``: Same as truncate but emit a stronger warning.
    """

    def __init__(
        self,
        hard_limit: int | None = None,
        overflow_action: str = "truncate",
    ) -> None:
        self._hard_limit = hard_limit
        self._overflow_action = overflow_action

    def post_build(self, result, diff_spec) -> bool:
        """Enforce hard budget limit on build result.

        Args:
            result: ContextResult with ``total_rendered_tokens``.
            diff_spec: Original DiffSpec (unused, kept for API consistency).

        Returns:
            True if within budget or overflow_action is not ``"error"``.
        """
        if self._hard_limit is None:
            return True

        rendered = getattr(result, "total_rendered_tokens", None)
        if rendered is None:
            rendered = getattr(result, "total_original_tokens", 0) if hasattr(result, "total_original_tokens") else 0

        if rendered <= self._hard_limit:
            return True

        logger.warning(
            "BudgetEnforcementHook: %d tokens exceeds hard limit of %d",
            rendered, self._hard_limit,
        )

        if self._overflow_action == "error":
            return False

        return True  # truncate/warn: proceed with warning

    @property
    def hard_limit(self) -> int | None:
        """Current hard token limit."""
        return self._hard_limit

    @hard_limit.setter
    def hard_limit(self, value: int | None) -> None:
        """Update the hard token limit at runtime."""
        self._hard_limit = value


class SourceFreshnessHook(HarnessHook):
    """Verify source map hasn't changed since indexing by comparing checksums.

    Args:
        checksum_store: Dict mapping file path to SHA-256 hex digest.
            Typically populated during indexing and checked before build.
    """

    def __init__(self, checksum_store: dict[str, str] | None = None) -> None:
        self._checksum_store: dict[str, str] = checksum_store or {}

    def pre_build(self, diff_spec, source_map) -> bool:
        """Compare source_map checksums against stored values.

        Args:
            diff_spec: DiffSpec with ``changed_files`` to verify.
            source_map: Current source_map to checksum.

        Returns:
            True if all checked files are fresh (or checksum_store is empty).
            False if stale files are detected.
        """
        if not self._checksum_store or not source_map:
            return True

        import hashlib  # noqa: PLC0415

        stale: list[str] = []
        for path in source_map:
            stored = self._checksum_store.get(path)
            if stored is None:
                continue  # new file, not previously indexed
            source = source_map[path]
            current = hashlib.sha256(source.encode(errors="replace")).hexdigest()
            if current != stored:
                stale.append(path)

        if stale:
            logger.warning(
                "SourceFreshnessHook: %d stale files detected (checksum mismatch)",
                len(stale),
            )
            return False

        return True

    def update_checksum(self, path: str, checksum: str) -> None:
        """Record a checksum for a file after indexing.

        Args:
            path: File path.
            checksum: SHA-256 hex digest.
        """
        self._checksum_store[path] = checksum

    def update_from_stats(self, stats) -> None:
        """Bulk-update checksums from an IndexStats-compatible object.

        No-op by default since IndexStats does not carry per-file checksums.
        Subclasses with access to raw file annotations should override.
        """


# ===========================================================================
# Harness Orchestrator
# ===========================================================================


class Harness:
    """Orchestrates multiple harness hooks through the build/index lifecycle.

    Thread-safe. Hooks are called in registration order.

    Example::

        harness = Harness()
        harness.add(GraphIntegrityHook(graph=dep_graph))
        harness.add(BudgetEnforcementHook(hard_limit=50_000, overflow_action="error"))

        warnings = harness.run_pre_build(diff_spec, source_map)
        if harness.stats.builds_blocked > 0:
            logger.error("Build blocked by pre-build hooks")
    """

    def __init__(self) -> None:
        self._hooks: list[HarnessHook] = []
        self._lock = threading.RLock()
        self._stats: HarnessStats = HarnessStats()
        self.drift_detector: DriftDetector = DriftDetector(window_size=10)

    # ------------------------------------------------------------------
    # Hook registry
    # ------------------------------------------------------------------

    def add(self, hook: HarnessHook) -> Harness:
        """Register a validation hook.

        Args:
            hook: HarnessHook instance to register.

        Returns:
            Self for method chaining.
        """
        with self._lock:
            self._hooks.append(hook)
        logger.debug("Harness: registered %s", type(hook).__name__)
        return self

    def remove(self, hook: HarnessHook) -> None:
        """Unregister a previously registered hook.

        Args:
            hook: HarnessHook instance to remove.
        """
        with self._lock:
            try:
                self._hooks.remove(hook)
                logger.debug("Harness: removed %s", type(hook).__name__)
            except ValueError:
                logger.debug("Harness: hook %s not found", type(hook).__name__)

    def clear(self) -> None:
        """Remove all registered hooks."""
        with self._lock:
            self._hooks.clear()
        logger.debug("Harness: all hooks cleared")

    def list_hooks(self) -> list[type[HarnessHook]]:
        """Return the types of all registered hooks.

        Returns:
            List of hook classes in registration order.
        """
        with self._lock:
            return [type(h) for h in self._hooks]

    # ------------------------------------------------------------------
    # Lifecycle methods
    # ------------------------------------------------------------------

    def run_pre_build(self, diff_spec, source_map) -> list[str]:
        """Run all pre_build hooks.

        Args:
            diff_spec: DiffSpec for the pending build.
            source_map: Dict mapping file path to source text.

        Returns:
            List of warning messages. The build should be aborted if
            builds_blocked increased.
        """
        warnings: list[str] = []
        with self._lock:
            hooks = list(self._hooks)

        for hook in hooks:
            try:
                result = hook.pre_build(diff_spec, source_map)
                self._stats.total_checks += 1
                if not result:
                    self._stats.builds_blocked += 1
                    msg = f"{type(hook).__name__}.pre_build: blocked"
                    warnings.append(msg)
                    logger.warning("Harness: %s", msg)
            except Exception as exc:
                self._stats.total_checks += 1
                msg = f"{type(hook).__name__}.pre_build: error: {exc}"
                warnings.append(msg)
                logger.error("Harness: %s", msg)

        self._stats.last_run = time.time()
        self._stats.warnings_issued += len(warnings)
        return warnings

    def run_post_build(self, result, diff_spec) -> list[str]:
        """Run all post_build hooks.

        Args:
            result: ContextResult from the build.
            diff_spec: Original DiffSpec.

        Returns:
            List of warning messages. The calling code should decide
            whether to treat warnings as blocking.
        """
        warnings: list[str] = []
        with self._lock:
            hooks = list(self._hooks)

        for hook in hooks:
            try:
                hook_result = hook.post_build(result, diff_spec)
                self._stats.total_checks += 1
                self._stats.builds_validated += 1
                if not hook_result:
                    msg = f"{type(hook).__name__}.post_build: validation failed"
                    warnings.append(msg)
                    logger.warning("Harness: %s", msg)
            except Exception as exc:
                self._stats.total_checks += 1
                msg = f"{type(hook).__name__}.post_build: error: {exc}"
                warnings.append(msg)
                logger.error("Harness: %s", msg)

        # Track tokens saved by enforcement
        if hasattr(result, "total_original_tokens") and hasattr(result, "total_rendered_tokens"):
            self._stats.tokens_saved_by_enforcement += (
                result.total_original_tokens - result.total_rendered_tokens
            )

        self._stats.last_run = time.time()
        self._stats.warnings_issued += len(warnings)

        # Auto-drift check after build
        try:
            if diff_spec and hasattr(self, 'drift_detector'):
                action = AgentAction(
                    action_type="build",
                    target=str(getattr(diff_spec, 'changed_files', [])),
                    timestamp=time.time(),
                )
                alerts = self.drift_detector.record(action)
                if alerts:
                    self._stats.drift_alerts += len(alerts)
                    for alert in alerts:
                        logger.warning("Drift alert [%s]: %s", alert.severity, alert.suggestion)
        except Exception:
            pass

        return warnings

    def record_action(self, action_type: str, target: str, metadata: dict | None = None) -> list:
        """Record an agent action and check for drift. Never raises.

        Args:
            action_type: Type of action (``"build"``, ``"index"``, ``"read"``, etc.).
            target: File path, symbol, or description the action targets.
            metadata: Optional extra data attached to the action.

        Returns:
            List of ``DriftAlert`` instances (empty if no drift detected).
        """
        try:
            if not hasattr(self, 'drift_detector'):
                return []
            action = AgentAction(
                action_type=action_type,
                target=target,
                timestamp=time.time(),
                metadata=metadata or {},
            )
            alerts = self.drift_detector.record(action)
            if alerts:
                self._stats.drift_alerts += len(alerts)
                for alert in alerts:
                    logger.warning("Drift alert [%s]: %s", alert.severity, alert.suggestion)
            return alerts or []
        except Exception:
            return []

    def drift_report(self) -> dict:
        """Return current drift state. Never raises.

        Returns:
            Dict with keys ``alerts``, ``total_actions``, and ``stats``.
        """
        try:
            if not hasattr(self, 'drift_detector'):
                return {"alerts": [], "total_actions": 0, "stats": {"drift_alerts": 0}}
            return {
                "alerts": [str(a) for a in self.drift_detector.alerts],
                "total_actions": len(getattr(self.drift_detector, '_recent_actions', [])),
                "stats": {"drift_alerts": self._stats.drift_alerts},
            }
        except Exception:
            return {"alerts": [], "total_actions": 0, "stats": {"drift_alerts": 0}}

    def run_pre_index(self, root_path: str) -> list[str]:
        """Run all pre_index hooks.

        Args:
            root_path: Repository root path to be indexed.

        Returns:
            List of warning messages.
        """
        warnings: list[str] = []
        with self._lock:
            hooks = list(self._hooks)

        for hook in hooks:
            try:
                result = hook.pre_index(root_path)
                self._stats.total_checks += 1
                if not result:
                    self._stats.indexes_blocked += 1
                    msg = f"{type(hook).__name__}.pre_index: blocked"
                    warnings.append(msg)
                    logger.warning("Harness: %s", msg)
            except Exception as exc:
                self._stats.total_checks += 1
                msg = f"{type(hook).__name__}.pre_index: error: {exc}"
                warnings.append(msg)
                logger.error("Harness: %s", msg)

        self._stats.last_run = time.time()
        self._stats.warnings_issued += len(warnings)
        return warnings

    def run_post_index(self, stats) -> list[str]:
        """Run all post_index hooks.

        Args:
            stats: IndexStats or dict from the indexing run.

        Returns:
            List of warning messages.
        """
        warnings: list[str] = []
        with self._lock:
            hooks = list(self._hooks)

        for hook in hooks:
            try:
                hook_result = hook.post_index(stats)
                self._stats.total_checks += 1
                self._stats.indexes_validated += 1
                if not hook_result:
                    msg = f"{type(hook).__name__}.post_index: validation failed"
                    warnings.append(msg)
                    logger.warning("Harness: %s", msg)
            except Exception as exc:
                self._stats.total_checks += 1
                msg = f"{type(hook).__name__}.post_index: error: {exc}"
                warnings.append(msg)
                logger.error("Harness: %s", msg)

        self._stats.last_run = time.time()
        self._stats.warnings_issued += len(warnings)
        return warnings

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> HarnessStats:
        """Current harness validation statistics.

        Returns:
            HarnessStats dataclass with counters and timestamps.
        """
        with self._lock:
            return HarnessStats(
                builds_validated=self._stats.builds_validated,
                builds_blocked=self._stats.builds_blocked,
                indexes_validated=self._stats.indexes_validated,
                indexes_blocked=self._stats.indexes_blocked,
                warnings_issued=self._stats.warnings_issued,
                drift_alerts=self._stats.drift_alerts,
                tokens_saved_by_enforcement=self._stats.tokens_saved_by_enforcement,
                last_run=self._stats.last_run,
                total_checks=self._stats.total_checks,
            )

    def reset_stats(self) -> None:
        """Reset all validation counters to zero."""
        with self._lock:
            self._stats = HarnessStats()
        logger.debug("Harness: stats reset")


# ===========================================================================
# Drift Detection
# ===========================================================================


@dataclass
class AgentAction:
    """A recorded agent action for drift analysis.

    Attributes:
        action_type: Type of action --- ``"read_file"``, ``"edit_file"``,
            ``"search"``, ``"tool_call"``, etc.
        target: File path, search query, or tool name the action targeted.
        timestamp: Unix timestamp when the action occurred.
        metadata: Free-form key/value pairs for additional context
            (e.g. ``{"lines_changed": 42}``, ``{"search_type": "regex"}``).
    """
    action_type: str
    target: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """A drift alert produced by the DriftDetector.

    Attributes:
        alert_type: Drift signal type --- ``"repeated_reads"``,
            ``"thrashing"``, ``"scope_creep"``, ``"decision_flip"``.
        severity: Severity level --- ``"warning"`` or ``"critical"``.
        message: Human-readable description of the drift signal.
        evidence: List of AgentAction instances that triggered the alert.
        suggestion: Suggested remediation action.
    """
    alert_type: str
    severity: str
    message: str
    evidence: list[AgentAction]
    suggestion: str


class DriftDetector:
    """Detects when the agent's behavior indicates context problems.

    Monitors a sliding window of recent agent actions and checks for
    known drift signals:

    - **repeated_reads**: Same file read 3+ times in the window
      (context may have been lost).
    - **thrashing**: Same file edited 5+ times in the window
      (agent is stuck).
    - **scope_creep**: Targets spreading across too many unique areas
      (agent drifting from original objective).
    - **decision_flip**: Conflicting action types on the same target
      within the window (context collapse).

    Args:
        window_size: Number of recent actions to retain for analysis.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._recent_actions: list[AgentAction] = []
        self._window_size = window_size

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, action: AgentAction) -> None:
        """Record an agent action for drift analysis.

        Maintains a sliding window --- the oldest action is evicted when
        the window exceeds ``window_size``.

        Args:
            action: AgentAction to record.
        """
        self._recent_actions.append(action)
        if len(self._recent_actions) > self._window_size:
            self._recent_actions.pop(0)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def check(self) -> list[DriftAlert]:
        """Check for drift signals in recent agent actions.

        Analyzes the current sliding window and returns all detected
        drift alerts. Multiple alerts of the same type are deduplicated
        (only the most severe instance per type is returned).

        Returns:
            List of DriftAlert instances, one per detected drift signal.
        """
        if len(self._recent_actions) < 2:
            return []

        alerts: list[DriftAlert] = []
        detected_types: set[str] = set()

        # -- repeated_reads: same file read 3+ times in window --
        reads = [a for a in self._recent_actions if a.action_type == "read_file"]
        read_counts: defaultdict[str, list[AgentAction]] = defaultdict(list)
        for r in reads:
            read_counts[r.target].append(r)
        for target, actions in read_counts.items():
            if len(actions) >= 3 and "repeated_reads" not in detected_types:
                detected_types.add("repeated_reads")
                alerts.append(DriftAlert(
                    alert_type="repeated_reads",
                    severity="warning",
                    message=(
                        f"File '{target}' was read {len(actions)} times in "
                        f"the last {self._window_size} actions — context may "
                        f"have been lost or dropped"
                    ),
                    evidence=actions,
                    suggestion="Verify the agent still has the file content "
                               "in context and hasn't exceeded its context window",
                ))

        # -- thrashing: same file edited 5+ times in window --
        edits = [a for a in self._recent_actions if a.action_type == "edit_file"]
        edit_counts: defaultdict[str, list[AgentAction]] = defaultdict(list)
        for e in edits:
            edit_counts[e.target].append(e)
        for target, actions in edit_counts.items():
            if len(actions) >= 5 and "thrashing" not in detected_types:
                detected_types.add("thrashing")
                alerts.append(DriftAlert(
                    alert_type="thrashing",
                    severity="critical",
                    message=(
                        f"File '{target}' was edited {len(actions)} times in "
                        f"the last {self._window_size} actions — agent is stuck "
                        f"or thrashing"
                    ),
                    evidence=actions,
                    suggestion="Review the edit history for the file. The agent "
                               "may be caught in a contradictory loop. Consider "
                               "clearing the conversation or providing a fresh "
                               "starting point",
                ))

        # -- scope_creep: more than half of unique targets are distinct --
        all_targets = [a.target for a in self._recent_actions]
        unique_targets = set(all_targets)
        if len(unique_targets) > max(1, len(all_targets) // 2) and len(all_targets) >= 5:
            if "scope_creep" not in detected_types:
                detected_types.add("scope_creep")
                alerts.append(DriftAlert(
                    alert_type="scope_creep",
                    severity="warning",
                    message=(
                        f"Actions span {len(unique_targets)} unique targets "
                        f"across {len(all_targets)} recent actions — potential "
                        f"scope creep"
                    ),
                    evidence=list(self._recent_actions),
                    suggestion="Consider re-anchoring the agent on the original "
                               "task objective. The agent may have drifted into "
                               "tangential exploration",
                ))

        # -- decision_flip: conflicting action types on same target --
        if len(self._recent_actions) >= 4:
            target_sequences: defaultdict[str, list[AgentAction]] = defaultdict(list)
            for a in self._recent_actions:
                target_sequences[a.target].append(a)

            for target, actions in target_sequences.items():
                if len(actions) >= 3:
                    types = [a.action_type for a in actions]
                    flips = sum(
                        1 for i in range(1, len(types))
                        if types[i] != types[i - 1]
                    )
                    if flips >= 2 and "decision_flip" not in detected_types:
                        detected_types.add("decision_flip")
                        alerts.append(DriftAlert(
                            alert_type="decision_flip",
                            severity="critical",
                            message=(
                                f"Conflicting action pattern detected on "
                                f"'{target}': {flips} type changes in "
                                f"{len(actions)} actions — context collapse "
                                f"may have occurred"
                            ),
                            evidence=actions,
                            suggestion="The agent appears to be changing its "
                                       "mind on the same file. This is a strong "
                                       "signal of context window overflow. "
                                       "Consider summarizing progress and "
                                       "starting a fresh turn",
                        ))

        return alerts

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all recorded actions and reset the detector."""
        self._recent_actions.clear()

    @property
    def recent_actions(self) -> list[AgentAction]:
        """Return a copy of the current action window.

        Returns:
            List of recent AgentAction instances.
        """
        return list(self._recent_actions)

    @property
    def window_size(self) -> int:
        """Current sliding window size."""
        return self._window_size


# ===========================================================================
# HarnessStats
# ===========================================================================


@dataclass
class HarnessStats:
    """Aggregated statistics from the harness lifecycle.

    Attributes:
        builds_validated: Number of builds that passed all hooks.
        builds_blocked: Number of builds blocked by pre/post hooks.
        indexes_validated: Number of indexing runs that passed validation.
        indexes_blocked: Number of indexing runs blocked by hooks.
        warnings_issued: Total warnings emitted across all hook runs.
        drift_alerts: Number of drift alerts raised by the DriftDetector.
        tokens_saved_by_enforcement: Cumulative tokens conserved by
            budget enforcement hooks.
        last_run: Unix timestamp of the most recent hook execution.
        total_checks: Total number of individual hook method calls.
    """
    builds_validated: int = 0
    builds_blocked: int = 0
    indexes_validated: int = 0
    indexes_blocked: int = 0
    warnings_issued: int = 0
    drift_alerts: int = 0
    tokens_saved_by_enforcement: int = 0
    last_run: float = 0.0
    total_checks: int = 0
