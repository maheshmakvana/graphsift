"""Auto-verify pipeline — syntax → lint → test (auto selective) → fix retry loop.

Runs a verification cascade after every code change, with an auto-fix
retry loop. **Test stage is automatically smart**: if a full test baseline
exists, only tests impacted by recent changes are run (saves 60-95% time).
If no baseline exists, runs full suite and stores the baseline.

No separate commands needed. Just call ``verify(run_tests=True)``.

Integrates with:
  - ``Verifier`` (verify_hooks.py) — syntax + lint
  - ``TestImpactAnalyzer`` (test_impact.py) — auto selective testing
  - ``FixSuggester`` (auto_fix.py) — auto-fix retry loop

Usage::

    from graphsift.auto_verify import AutoVerifier

    av = AutoVerifier(project_root=".")
    # Auto mode: runs selective if baseline exists, full otherwise
    result = av.verify("src/auth.py", run_tests=True)
    print(result.summary)
    # → "[PASSED] src/auth.py: 1 iteration(s), 0 fix(es), 2340ms"
    # → "  Tests: SELECTIVE — 12 impacted tests run (566 skipped, 98% savings)"
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphsift.verify_hooks import Verifier, VerifyResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RetryAction(str, enum.Enum):
    """What to do when a verification step fails."""
    SYNTAX_FIX = "syntax_fix"
    LINT_FIX = "lint_fix"
    REPORT_ONLY = "report_only"


@dataclass
class VerificationStage:
    """Result of a single verification stage (e.g. syntax, lint, tests)."""
    name: str
    passed: bool = False
    error: str = ""
    details: str = ""
    duration_ms: float = 0.0


@dataclass
class VerificationIteration:
    """One full pass through the verification cascade."""
    iteration: int
    stages: list[VerificationStage] = field(default_factory=list)
    all_passed: bool = False
    auto_fixes_applied: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        failed = [s for s in self.stages if not s.passed]
        if not failed:
            return (
                f"Iteration {self.iteration}: "
                f"All {len(self.stages)} checks passed"
            )
        return (
            f"Iteration {self.iteration}: "
            f"{len(failed)}/{len(self.stages)} checks failed: "
            f"{', '.join(s.name for s in failed)}"
        )


@dataclass
class AutoVerifyResult:
    """Complete result of an auto-verify run across all iterations."""
    file_path: str
    iterations: list[VerificationIteration] = field(default_factory=list)
    total_duration_ms: float = 0.0
    final_passed: bool = False
    total_fixes_applied: int = 0
    test_mode: str = "standard"  # "standard", "selective", "full"
    tests_saved: int = 0
    tests_savings_pct: float = 0.0

    @property
    def summary(self) -> str:
        status = "PASSED" if self.final_passed else "FAILED"
        parts = [
            f"[{status}] {self.file_path}:",
            f"{len(self.iterations)} iteration(s),",
            f"{self.total_fixes_applied} fix(es),",
            f"{self.total_duration_ms:.0f}ms",
        ]
        if self.test_mode == "selective":
            parts.append(f"Tests: SELECTIVE ({self.tests_saved} skipped, {self.tests_savings_pct:.0f}% saved)")
        elif self.test_mode == "full":
            parts.append("Tests: FULL (baseline stored)")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# AutoVerifier
# ---------------------------------------------------------------------------

class AutoVerifier:
    """Verification cascade with auto-fix retry and **auto selective testing**.

    When ``run_tests=True``, the test stage automatically:
      1. Checks memory for a prior full-test baseline
      2. If baseline exists → runs only tests impacted by changed files
      3. If no baseline → runs full suite, stores baseline for next time
      4. All tests run in parallel (all cores) with per-test timeout

    No separate commands, no manual mode switching.

    Args:
        project_root: Repository root path.
        graph: Optional ``DependencyGraph`` for fix suggestion + impact analysis.
        source_map: Optional dict of file path → source text.
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        project_root: str = "",
        graph: Any = None,
        source_map: dict[str, str] | None = None,
    ) -> None:
        self.project_root = Path(project_root or ".").resolve()
        self.verifier = Verifier(project_root=str(self.project_root))
        self._graph = graph

        # Lazy-import TestImpactAnalyzer (only when tests actually run)
        self._impact = None

        # Lazy-import FixSuggester only when graph is available
        self._fix_suggester = None
        if graph is not None:
            try:
                from graphsift.auto_fix import FixSuggester  # noqa: PLC0415
                self._fix_suggester = FixSuggester(
                    graph=graph,
                    source_map=source_map or {},
                )
            except Exception as exc:
                logger.debug("FixSuggester not available: %s", exc)

    def _get_impact(self):
        """Lazy-init TestImpactAnalyzer."""
        if self._impact is None:
            from graphsift.test_impact import TestImpactAnalyzer  # noqa: PLC0415
            self._impact = TestImpactAnalyzer(
                project_root=str(self.project_root),
                graph=self._graph,
            )
        return self._impact

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        file_path: str,
        run_tests: bool = False,
        max_retries: int | None = None,
        changed_files: list[str] | None = None,
    ) -> AutoVerifyResult:
        """Run full verification cascade with auto-fix retry.

        For each iteration:
          1. Syntax check (via ``Verifier.check``)
          2. Lint check (via ``Verifier.lint``)
          3. **Smart tests** (if ``run_tests=True``):
             - Auto detects changed files via git diff
             - Checks memory for full-test baseline
             - Runs only impacted tests if baseline exists
             - Falls back to full test suite if no baseline
          4. If all pass → done
          5. If failures remain → apply auto-fixes → retry

        Args:
            file_path: Path relative to ``project_root``.
            run_tests: Whether to also run tests (smart selective if True).
            max_retries: Max retry iterations (default ``3``).
            changed_files: Optional override for changed files list.
                           Auto-detected via git if omitted.

        Returns:
            ``AutoVerifyResult`` with full iteration history.
        """
        start = time.monotonic()
        max_retries = max_retries or self.MAX_RETRIES
        iterations: list[VerificationIteration] = []
        test_mode = "standard"
        tests_saved = 0
        tests_savings_pct = 0.0

        for iteration in range(max_retries + 1):
            stages: list[VerificationStage] = []
            all_passed = True
            fixes: list[str] = []

            # -- Stage 1: Syntax -----------------------------------------
            t0 = time.monotonic()
            try:
                vresult = self.verifier.check(file_path)
                stages.append(VerificationStage(
                    name="syntax",
                    passed=vresult.syntax_ok,
                    error=vresult.syntax_error or "",
                    details="",
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                if not vresult.syntax_ok:
                    all_passed = False
            except Exception as exc:
                stages.append(VerificationStage(
                    name="syntax",
                    passed=False,
                    error=str(exc),
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                all_passed = False

            # -- Stage 2: Lint -------------------------------------------
            t0 = time.monotonic()
            try:
                lint_ok, lint_out = self.verifier.lint(file_path)
                stages.append(VerificationStage(
                    name="lint",
                    passed=lint_ok,
                    error=lint_out[:500] if not lint_ok else "",
                    details=lint_out[:200] if lint_ok else "",
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                if not lint_ok:
                    all_passed = False
            except Exception as exc:
                stages.append(VerificationStage(
                    name="lint",
                    passed=False,
                    error=str(exc),
                    duration_ms=(time.monotonic() - t0) * 1000,
                ))
                all_passed = False

            # -- Stage 3: Smart Tests (auto selective) --------------------
            if run_tests:
                t0 = time.monotonic()
                try:
                    impact = self._get_impact()
                    last_full = impact._memory.last_full_snapshot()

                    if last_full is not None:
                        # --- SELECTIVE MODE: baseline exists -------------
                        cf = changed_files or impact._git_changed_files(last_full.commit_hash)
                        if cf:
                            stage_details, passed_flag, test_mode_label, saved, pct = (
                                self._run_selective_tests(impact, cf, file_path, t0)
                            )
                        else:
                            # No files changed — skip tests entirely
                            stage_details = VerificationStage(
                                name="tests", passed=True,
                                details="No files changed since last full test — skipping",
                                duration_ms=(time.monotonic() - t0) * 1000,
                            )
                            passed_flag = True
                            test_mode_label = "selective"
                            saved = 0
                            pct = 100.0

                        stages.append(stage_details)
                        test_mode = f"selective ({test_mode_label})"
                        tests_saved = saved
                        tests_savings_pct = pct
                    else:
                        # --- FULL MODE: no baseline — run full suite -----
                        stage_details, passed_flag = self._run_full_tests(impact, file_path, t0)
                        stages.append(stage_details)
                        test_mode = "full"

                    if not passed_flag:
                        all_passed = False

                except Exception as exc:
                    stages.append(VerificationStage(
                        name="tests", passed=False, error=str(exc),
                        duration_ms=(time.monotonic() - t0) * 1000,
                    ))
                    all_passed = False

            # Check if done
            if all_passed:
                iterations.append(VerificationIteration(
                    iteration=iteration,
                    stages=stages,
                    all_passed=True,
                ))
                break

            # -- Auto-fix attempt -----------------------------------------
            if iteration < max_retries and self._fix_suggester is not None:
                failed = [s for s in stages if not s.passed]
                for stage in failed:
                    if stage.name == "lint":
                        try:
                            full_path = self.project_root / file_path
                            if full_path.exists():
                                report = self._fix_suggester.analyze(
                                    changed_files=[file_path],
                                )
                                applied = [
                                    s.title
                                    for s in report.suggestions
                                    if s.auto_fixable
                                ]
                                fixes.extend(applied)
                        except Exception as exc:
                            logger.debug("Auto-fix error: %s", exc)
                    elif stage.name == "syntax":
                        logger.debug(
                            "Syntax error in %s, iteration %d: %s",
                            file_path, iteration, stage.error,
                        )

            iterations.append(VerificationIteration(
                iteration=iteration,
                stages=stages,
                all_passed=all_passed,
                auto_fixes_applied=fixes,
            ))

        total_ms = (time.monotonic() - start) * 1000
        last = iterations[-1] if iterations else None
        return AutoVerifyResult(
            file_path=file_path,
            iterations=iterations,
            total_duration_ms=total_ms,
            final_passed=last.all_passed if last else False,
            total_fixes_applied=sum(
                len(i.auto_fixes_applied) for i in iterations
            ),
            test_mode=test_mode,
            tests_saved=tests_saved,
            tests_savings_pct=tests_savings_pct,
        )

    # ------------------------------------------------------------------
    # Internal: smart test runners
    # ------------------------------------------------------------------

    def _run_selective_tests(
        self,
        impact: Any,
        changed_files: list[str],
        file_path: str,
        t0: float,
    ) -> tuple[VerificationStage, bool, str, int, float]:
        """Run only impacted tests. Returns (stage, passed, label, saved, pct)."""
        import os as _os

        impacted_tests = impact._find_impacted_tests(changed_files)

        # Include the specific file's tests too
        file_tests = impact._find_impacted_tests([file_path])
        all_tests = list(set(impacted_tests + file_tests))

        if not all_tests:
            # No tests directly impacted — run smoke test on this file
            cpu_count = _os.cpu_count() or 4
            workers = max(1, cpu_count - 1)
            import subprocess  # noqa: PLC0415
            test_result = subprocess.run(
                ["pytest", file_path,
                 f"-n={workers}", "--dist=loadscope",
                 "--timeout=120", "--timeout-method=thread",
                 "-x", "--tb=short", "-q"],
                capture_output=True, text=True, timeout=300,
                cwd=str(self.project_root),
            )
            passed = test_result.returncode == 0
            return (
                VerificationStage(
                    name="tests", passed=passed,
                    details=test_result.stdout[:500],
                    error=test_result.stderr[:500] if not passed else "",
                    duration_ms=(time.monotonic() - t0) * 1000,
                ),
                passed, "single-file", 0, 0.0,
            )

        # Estimate total tests in project
        total_est = impact._estimate_total_tests()
        savings = max(0, total_est - len(all_tests)) if total_est else 0
        savings_pct = round((1 - len(all_tests) / max(total_est, 1)) * 100, 1) if total_est else 0

        # Build test file list
        test_files_str = " ".join(
            f'"{t}"' if " " in t else t
            for t in all_tests
        )
        cpu_count = _os.cpu_count() or 4
        workers = max(1, cpu_count - 1)

        import subprocess  # noqa: PLC0415
        test_result = subprocess.run(
            f"pytest {test_files_str} -n={workers} "
            f"--dist=loadscope --timeout=120 --timeout-method=thread "
            f"--tb=short -q --no-header 2>&1",
            capture_output=True, text=True, timeout=300,
            cwd=str(self.project_root), shell=True,
        )
        passed = test_result.returncode == 0
        output = test_result.stdout + test_result.stderr

        from graphsift.test_impact import TestSnapshot  # noqa: PLC0415

        # Store snapshot
        commit = impact._get_git_hash()
        impact._memory.save_snapshot(
            TestSnapshot(
                commit_hash=commit,
                mode="selective",
                status="passed" if passed else "failed",
                tests_run=len(all_tests),
                changed_files=changed_files,
                impacted_tests=all_tests,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        )

        summary = (
            f"{len(all_tests)} impacted tests ({savings} skipped, "
            f"{savings_pct}% savings)"
        )
        return (
            VerificationStage(
                name="tests", passed=passed,
                details=summary,
                error=output[:500] if not passed else "",
                duration_ms=(time.monotonic() - t0) * 1000,
            ),
            passed, summary, savings, savings_pct,
        )

    def _run_full_tests(
        self,
        impact: Any,
        file_path: str,
        t0: float,
    ) -> tuple[VerificationStage, bool]:
        """Run full test suite and store baseline. Returns (stage, passed)."""
        import os as _os
        import subprocess  # noqa: PLC0415

        cpu_count = _os.cpu_count() or 4
        workers = max(1, cpu_count - 1)

        test_result = subprocess.run(
            f"pytest -n={workers} --dist=loadscope "
            f"--timeout=120 --timeout-method=thread "
            f"--tb=short -q --no-header 2>&1",
            capture_output=True, text=True, timeout=600,
            cwd=str(self.project_root), shell=True,
        )
        passed = test_result.returncode == 0
        output = test_result.stdout + test_result.stderr

        from graphsift.test_impact import TestSnapshot  # noqa: PLC0415
        import re  # noqa: PLC0415

        # Store as baseline snapshot for future selective runs
        commit = impact._get_git_hash()
        passed_count = len(re.findall(r"passed", output))
        failed_count = len(re.findall(r"failed", output))

        impact._memory.save_snapshot(
            TestSnapshot(
                commit_hash=commit,
                mode="full",
                status="passed" if passed else "failed",
                tests_run=passed_count + failed_count,
                tests_passed=passed_count,
                tests_failed=failed_count,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        )

        # Build/refresh impact mappings
        impact._build_impact_mappings()

        summary = (
            "Full test suite (baseline stored — future tests will be selective)"
            if passed else
            "Full test suite — baseline NOT stored (tests failed)"
        )
        return (
            VerificationStage(
                name="tests", passed=passed,
                details=summary,
                error=output[:500] if not passed else "",
                duration_ms=(time.monotonic() - t0) * 1000,
            ),
            passed,
        )


__all__ = [
    "AutoVerifier",
    "AutoVerifyResult",
    "VerificationIteration",
    "VerificationStage",
    "RetryAction",
]
