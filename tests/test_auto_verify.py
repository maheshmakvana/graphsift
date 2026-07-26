"""Tests for the Auto-Verify Pipeline."""

from __future__ import annotations

from graphsift.auto_verify import AutoVerifier, AutoVerifyResult, VerificationStage, VerificationIteration, RetryAction


class TestAutoVerifyModels:
    """Tests for AutoVerify data classes."""

    def test_verification_stage_defaults(self):
        """VerificationStage should have sensible defaults."""
        stage = VerificationStage(name="syntax")
        assert stage.name == "syntax"
        assert not stage.passed
        assert stage.error == ""
        assert stage.duration_ms == 0.0

    def test_verification_stage_passed(self):
        """VerificationStage should reflect passed state."""
        stage = VerificationStage(name="lint", passed=True)
        assert stage.passed
        assert stage.duration_ms == 0.0

    def test_verification_iteration_all_passed(self):
        """VerificationIteration should show correct summary when all pass."""
        stages = [
            VerificationStage(name="syntax", passed=True),
            VerificationStage(name="lint", passed=True),
        ]
        vi = VerificationIteration(iteration=0, stages=stages, all_passed=True)
        summary = vi.summary
        assert "All" in summary
        assert "passed" in summary

    def test_verification_iteration_failures(self):
        """VerificationIteration should list failed stages."""
        stages = [
            VerificationStage(name="syntax", passed=True),
            VerificationStage(name="lint", passed=False, error="Lint error"),
        ]
        vi = VerificationIteration(iteration=1, stages=stages, all_passed=False)
        summary = vi.summary
        assert "failed" in summary
        assert "lint" in summary

    def test_verification_iteration_with_fixes(self):
        """VerificationIteration should track applied fixes."""
        vi = VerificationIteration(iteration=0, stages=[], all_passed=False, auto_fixes_applied=["fix1"])
        assert len(vi.auto_fixes_applied) == 1
        assert vi.auto_fixes_applied[0] == "fix1"

    def test_auto_verify_result_passed(self):
        """AutoVerifyResult should reflect passed state."""
        result = AutoVerifyResult(
            file_path="test.py",
            iterations=[],
            total_duration_ms=100.0,
            final_passed=True,
        )
        summary = result.summary
        assert "PASSED" in summary
        assert "test.py" in summary
        assert result.total_fixes_applied == 0

    def test_auto_verify_result_failed(self):
        """AutoVerifyResult should reflect failed state."""
        result = AutoVerifyResult(
            file_path="bad.py",
            iterations=[VerificationIteration(iteration=0, stages=[])],
            final_passed=False,
        )
        summary = result.summary
        assert "FAILED" in summary

    def test_auto_verify_result_with_fixes(self):
        """AutoVerifyResult should track total fixes."""
        result = AutoVerifyResult(
            file_path="test.py",
            iterations=[
                VerificationIteration(iteration=0, stages=[], auto_fixes_applied=["a", "b"]),
            ],
            total_fixes_applied=2,
        )
        assert result.total_fixes_applied == 2

    def test_retry_action_enum(self):
        """RetryAction enum should have correct values."""
        assert RetryAction.SYNTAX_FIX.value == "syntax_fix"
        assert RetryAction.LINT_FIX.value == "lint_fix"
        assert RetryAction.REPORT_ONLY.value == "report_only"


class TestAutoVerifier:
    """Tests for the AutoVerifier class."""

    def test_create_auto_verifier(self):
        """Creating AutoVerifier without graph should work."""
        av = AutoVerifier(project_root=".")
        assert av is not None
        assert av.MAX_RETRIES == 3

    def test_verify_valid_file(self):
        """Verify should pass on a valid Python file."""
        av = AutoVerifier(project_root=".")
        result = av.verify("graphsift/_version.py")
        assert result.final_passed
        assert len(result.iterations) >= 1
        assert result.iterations[0].all_passed

    def test_verify_invalid_file(self):
        """Verify should fail on a non-existent file."""
        av = AutoVerifier(project_root=".")
        result = av.verify("nonexistent_file_xyz.py")
        assert not result.final_passed

    def test_verify_with_max_retries(self):
        """Verify should respect max_retries parameter."""
        av = AutoVerifier(project_root=".")
        result = av.verify("graphsift/_version.py", max_retries=1)
        assert result.final_passed
        assert len(result.iterations) >= 1

    def test_verify_empty_file_path(self):
        """Verify should handle empty file path (treated as current dir)."""
        av = AutoVerifier(project_root=".")
        result = av.verify("")
        # Empty path resolves to current directory, which may or may not be valid
        assert isinstance(result, AutoVerifyResult)
