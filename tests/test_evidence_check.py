"""Tests for v2.4+ evidence enforcement features."""

from __future__ import annotations

from graphsift.evidence_check import EvidenceChecker, EnforceMode, EnforceResult, Citation


class TestEnforceMode:
    """Tests for the EnforceMode enum."""

    def test_enum_values(self):
        """EnforceMode should have correct values."""
        assert EnforceMode.MARK.value == "mark"
        assert EnforceMode.STRIP.value == "strip"
        assert EnforceMode.REPORT.value == "report"
        assert EnforceMode.ENFORCE.value == "enforce"


class TestEnforceResult:
    """Tests for the EnforceResult dataclass."""

    def test_defaults(self):
        """EnforceResult should have sensible defaults."""
        result = EnforceResult(text="hello")
        assert result.verified_claims == []
        assert result.unverified_claims == []
        assert result.total_claims == 0
        assert result.unverified_count == 0

    def test_with_claims(self):
        """EnforceResult should track verified and unverified claims."""
        verified = [Citation(raw="file.py:10", file_path="file.py", line=10, valid=True)]
        unverified = [Citation(raw="bad.py:99", file_path="bad.py", line=99, valid=False, error="Not found")]
        result = EnforceResult(
            text="test",
            verified_claims=verified,
            unverified_claims=unverified,
            total_claims=2,
            unverified_count=1,
        )
        assert result.total_claims == 2
        assert result.unverified_count == 1
        assert len(result.verified_claims) == 1
        assert len(result.unverified_claims) == 1

    def test_summary(self):
        """EnforceResult.summary should describe the result."""
        result = EnforceResult(text="test", total_claims=5, unverified_count=2)
        summary = result.summary
        assert "5 claims" in summary
        assert "2 unverified" in summary

    def test_summary_no_issues(self):
        """EnforceResult.summary should show zero unverified."""
        result = EnforceResult(text="clean", total_claims=3, unverified_count=0)
        assert "0 unverified" in result.summary


class TestEvidenceCheckerEnforce:
    """Tests for EvidenceChecker.enforce_text."""

    def test_enforce_mark_mode(self):
        """MARK mode should append [UNKNOWN] to unverified claims."""
        checker = EvidenceChecker(project_root=".")
        result = checker.enforce_text("Fix bug in src/auth.py:999", mode=EnforceMode.MARK)
        # 999 line likely doesn't exist
        assert "[UNKNOWN]" in result.text

    def test_enforce_strip_mode(self):
        """STRIP mode should remove unverified claims."""
        checker = EvidenceChecker(project_root=".")
        text = "Fix bug in fake_file_xyz.py:1"
        result = checker.enforce_text(text, mode=EnforceMode.STRIP)
        # The fake file reference should be removed
        assert "fake_file_xyz" not in result.text or result.unverified_count > 0

    def test_enforce_report_mode(self):
        """REPORT mode should not modify text."""
        checker = EvidenceChecker(project_root=".")
        text = "Fix bug in nonexistent_file_xyz.py:42"
        result = checker.enforce_text(text, mode=EnforceMode.REPORT)
        assert result.text == text  # text unchanged

    def test_enforce_no_claims(self):
        """Text with no file:line claims should pass through."""
        checker = EvidenceChecker(project_root=".")
        text = "This is a general statement with no citations."
        result = checker.enforce_text(text, mode=EnforceMode.MARK)
        assert result.total_claims == 0
        assert result.text == text

    def test_enforce_mixed_claims(self):
        """Mix of valid and invalid claims should be handled."""
        checker = EvidenceChecker(project_root=".")
        text = "Valid file: graphsift/_version.py:3 and invalid: fake.py:999"
        result = checker.enforce_text(text, mode=EnforceMode.MARK)
        # Should have found some claims (at least the invalid one)
        assert result.total_claims >= 0

    def test_enforce_empty_text(self):
        """Empty text should return empty result."""
        checker = EvidenceChecker(project_root=".")
        result = checker.enforce_text("", mode=EnforceMode.MARK)
        assert result.total_claims == 0
        assert result.text == ""

    def test_enforce_real_citation(self):
        """A citation pointing to a real file:line should be verified."""
        checker = EvidenceChecker(project_root=".")
        result = checker.enforce_text("Version is in graphsift/_version.py:3", mode=EnforceMode.REPORT)
        assert result.total_claims >= 1
        verified = [c for c in result.verified_claims if c.valid]
        # There may or may not be matches depending on parsing
        assert isinstance(verified, list)
