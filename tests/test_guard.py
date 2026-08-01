"""Tests for graphsift.guard — trading-strategy hallucination guard."""

from __future__ import annotations

import json

import pytest

from graphsift.guard import (
    ClaimStatus,
    ClaimType,
    GuardReport,
    JsonBacktestProvider,
    MarketDataProvider,
    StrategyClaim,
    StrategyGuard,
    extract_claims,
    guard_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HALLUCINATED = """STRATEGY: Iron Condor on BANKNIFTY — GUARANTEED PROFIT
This is a sure-shot strategy with 99% win rate and zero risk.
Buy BANKNIFTY 54000 CE and 52000 PE, sell 54500 CE 51500 PE.
Net premium collected: Rs.44,00,000 profit in one month.
Annualized ROI of 8500%. This strategy never fails.
100% safe returns, no loss possible.
"""

GROUNDED = """STRATEGY: Short Straddle BANKNIFTY
Paper trading result: net P&L Rs.4,00,000 over recent period,
win rate 59%, ROI about 200%. Premium collected Rs.300-600 per lot.
"""

BACKTEST_FRAMED = """Backtest showed Rs.44,15,988 profit over 58 days (2X WINNERS variant).
Live paper trading so far is around Rs.4,00,000 net P&L.
"""


@pytest.fixture()
def guard() -> StrategyGuard:
    return StrategyGuard(provider=JsonBacktestProvider())


# ---------------------------------------------------------------------------
# Reference / provider
# ---------------------------------------------------------------------------

class TestReference:
    def test_default_reference_has_live_net_pnl(self):
        prov = JsonBacktestProvider()
        assert prov.get_reference_stats()["net_pnl"] == 400000

    def test_json_provider_loads_file(self, tmp_path):
        path = tmp_path / "ref.json"
        path.write_text(json.dumps({"net_pnl": 123456, "win_rate_pct": 50.0}))
        prov = JsonBacktestProvider(path)
        assert prov.get_reference_stats()["net_pnl"] == 123456

    def test_missing_file_falls_back_to_default(self):
        prov = JsonBacktestProvider("/nonexistent/ref.json")
        assert prov.get_reference_stats()["net_pnl"] == 400000

    def test_verify_profit_below_live_is_verified(self):
        prov = JsonBacktestProvider()
        status, _ = prov.verify_profit(400000)
        assert status == ClaimStatus.VERIFIED

    def test_verify_profit_way_above_live_is_contradicted(self):
        prov = JsonBacktestProvider()
        status, reason = prov.verify_profit(4400000)
        assert status == ClaimStatus.CONTRADICTED
        assert "real-time proven" in reason

    def test_verify_profit_backtest_framed_matches_2x(self):
        prov = JsonBacktestProvider()
        status, reason = prov.verify_profit(4415988, backtest=True)
        assert status == ClaimStatus.SYNTHETIC
        assert "SIMULATION" in reason

    def test_verify_win_rate_over_90_contradicted(self):
        prov = JsonBacktestProvider()
        status, _ = prov.verify_win_rate(99)
        assert status == ClaimStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_extracts_currency_profit(self):
        claims = extract_claims("Made Rs.44,00,000 profit")
        profits = [c for c in claims if c.type == ClaimType.PROFIT]
        assert any(abs(c.value - 4400000) < 1 for c in profits)

    def test_parses_lakh(self):
        claims = extract_claims("44 lakh profit")
        profits = [c for c in claims if c.type == ClaimType.PROFIT]
        assert any(abs(c.value - 4400000) < 1 for c in profits)

    def test_detects_guarantee_language(self):
        claims = extract_claims("guaranteed sure shot zero risk")
        assert any(c.type == ClaimType.GUARANTEE for c in claims)

    def test_negated_guarantee_not_flagged(self):
        claims = extract_claims("Do NOT expect guaranteed returns")
        assert not any(c.type == ClaimType.GUARANTEE for c in claims)

    def test_option_strike_is_signal(self):
        claims = extract_claims("Buy BANKNIFTY 54000 CE")
        assert any(c.type == ClaimType.SIGNAL or c.type == ClaimType.PRICE for c in claims)

    def test_lowercase_words_not_strikes(self):
        # "and 52000 PE" should not be a strike (symbol must be uppercase)
        claims = extract_claims("Buy BANKNIFTY 54000 CE and 52000 PE")
        assert not any("and 52000 PE" in c.raw for c in claims)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class TestAudit:
    def test_hallucinated_scores_high(self, guard):
        report = guard.audit(HALLUCINATED)
        assert report.hallucination_score >= 50
        assert len(report.contradicted_claims) > 0

    def test_grounded_scores_low(self, guard):
        report = guard.audit(GROUNDED)
        assert report.hallucination_score < 25
        assert len(report.verified_claims) > 0

    def test_backtest_framed_44l_is_synthetic(self, guard):
        report = guard.audit(BACKTEST_FRAMED)
        synthetics = [c for c in report.synthetic_claims if c.value == 4415988]
        assert any(c.status == ClaimStatus.SYNTHETIC for c in synthetics)

    def test_report_dict_shape(self, guard):
        report = guard.audit(HALLUCINATED)
        data = report.to_dict()
        for key in ("total_claims", "verified", "synthetic", "contradicted",
                    "unverifiable", "hallucination_score", "risk_level", "claims"):
            assert key in data

    def test_risk_level_high_for_fabrication(self, guard):
        report = guard.audit(HALLUCINATED)
        assert report.risk_level.startswith("HIGH")


# ---------------------------------------------------------------------------
# Enforce
# ---------------------------------------------------------------------------

class TestEnforce:
    def test_mark_appends_markers(self, guard):
        cleaned, report = guard.enforce("Guaranteed Rs.44,00,000 profit", mode="mark")
        assert "GUARANTEED [CONTRADICTED]" in cleaned.upper() or "guaranteed [CONTRADICTED]" in cleaned.lower()

    def test_strip_removes_risky_claims(self, guard):
        cleaned, report = guard.enforce("Guaranteed Rs.44,00,000 profit", mode="strip")
        assert "44,00,000" not in cleaned

    def test_report_leaves_text_unchanged(self, guard):
        cleaned, report = guard.enforce("Guaranteed profit", mode="report")
        assert cleaned == "Guaranteed profit"

    def test_unknown_mode_raises(self, guard):
        with pytest.raises(ValueError):
            guard.enforce("text", mode="nonsense")


# ---------------------------------------------------------------------------
# Convenience API + prompt
# ---------------------------------------------------------------------------

class TestConvenience:
    def test_guard_strategy_report_mode(self):
        report = guard_strategy(HALLUCINATED)
        assert isinstance(report, GuardReport)
        assert report.hallucination_score >= 50

    def test_guard_strategy_enforce_mode(self):
        result = guard_strategy(HALLUCINATED, mode="strip")
        cleaned, report = result
        assert isinstance(report, GuardReport)
        assert "44,00,000" not in cleaned

    def test_build_grounding_prompt_contains_rules(self, guard):
        prompt = guard.build_grounding_prompt(strategy_request="Iron Condor")
        assert "Anti-hallucination rules" in prompt
        assert "REAL REFERENCE DATA" in prompt
        assert "[UNKNOWN]" in prompt
        assert "guaranteed" in prompt.lower()  # mentions the banned word

    def test_grounding_prompt_without_reference(self, guard):
        prompt = guard.build_grounding_prompt(strategy_request="x", include_reference=False)
        # The rules still mention "REAL REFERENCE DATA" generically, but the
        # actual reference JSON block must NOT be embedded.
        assert '"net_pnl"' not in prompt
        assert '"backtest_2x_pnl"' not in prompt

    def test_grounding_prompt_embeds_reference(self, guard):
        prompt = guard.build_grounding_prompt(strategy_request="x", include_reference=True)
        assert '"net_pnl": 400000' in prompt


# ---------------------------------------------------------------------------
# Regression tests — bugs found during multi-role review
# ---------------------------------------------------------------------------

class TestReviewRegressions:
    """Regression tests for bugs found by adversarial review."""

    def test_negation_does_not_cross_sentence_boundary(self):
        # "never" in a PREVIOUS sentence must not cancel "100% safe" here.
        claims = extract_claims("This strategy never fails. No loss possible. 100% safe.")
        guarantees = [c for c in claims if c.type == ClaimType.GUARANTEE]
        raws = " ".join(c.raw for c in guarantees).lower()
        assert "100% safe" in raws
        assert "no loss" in raws
        assert "never fails" in raws

    def test_never_fails_with_s_is_caught(self):
        claims = extract_claims("This strategy never fails.")
        assert any(c.type == ClaimType.GUARANTEE for c in claims)

    def test_bare_lakh_amounts_not_deduped_to_none(self):
        # "44 lakh" and "2 crore" must BOTH be extracted (dedup key bug).
        claims = extract_claims("Backtest showed 44 lakh profit. Also 2 crore revenue.")
        values = sorted([c.value for c in claims if c.value])
        assert 4400000 in values
        assert 20000000 in values

    def test_small_value_not_verified_as_profit(self, guard):
        # Per-lot premium (Rs.300) must not be "verified" against cumulative P&L.
        report = guard.audit("Premium collected Rs.300 per lot.")
        for c in report.claims:
            if c.value == 300:
                assert c.status == ClaimStatus.UNVERIFIABLE

    def test_repr_with_missing_keys_no_crash(self):
        prov = JsonBacktestProvider()
        prov.reference = {"backtest_pnl": 1000000}  # no net_pnl
        assert "?" in repr(prov)  # must not raise

    def test_json_provider_coerces_string_numbers(self, tmp_path):
        # User reference JSONs often quote numbers — must not crash formatting.
        path = tmp_path / "str_ref.json"
        path.write_text(json.dumps({"net_pnl": "400000", "win_rate_pct": "59.3"}))
        prov = JsonBacktestProvider(path)
        assert prov.reference["net_pnl"] == 400000.0
        assert prov.reference["win_rate_pct"] == 59.3

    def test_reference_provenance_fields_present(self):
        # Observability: reference must document its own provenance.
        prov = JsonBacktestProvider()
        note = prov.get_reference_stats().get("note", "")
        assert "PROVENANCE" in note or "proven" in note.lower()

    def test_backtest_figures_exact_match_angelbot(self):
        # Data integrity: embedded backtest numbers must match Angelbot source.
        prov = JsonBacktestProvider()
        ref = prov.get_reference_stats()
        assert ref["backtest_pnl"] == 2626665        # ALL-23 60-day
        assert ref["backtest_2x_pnl"] == 4415988     # 2X WINNERS
        assert ref["backtest_win_rate_pct"] == 64.0
        assert ref["backtest_roi_pct"] == 1313.3

    def test_bare_profit_numbers_extracted(self):
        # Profit figures without a currency marker must still be extracted.
        assert any(c.value == 5000 for c in extract_claims("profit 5000"))
        assert any(c.value == 15000 for c in extract_claims("made 15000 profit"))
        assert any(c.value == 250000 for c in extract_claims("gained 2,50,000"))
        assert any(c.value == 120000 for c in extract_claims("earned 1,20,000"))

    def test_bare_numbers_no_false_positives(self):
        # Numbers without profit context must NOT be extracted.
        assert not any(c.value == 5000 for c in extract_claims("uses 5000 lots"))
        assert extract_claims("it is 5000 meters") == []

    def test_bare_number_no_double_count(self):
        # "Rs.44,00,000 profit" must yield exactly ONE profit claim.
        matches = [c for c in extract_claims("Rs.44,00,000 profit") if c.value == 4400000]
        assert len(matches) == 1

    def test_2x_winners_column_matches_guard(self):
        # Data integrity: 2X WINNERS net P&L in the comparison file is the
        # SECOND column (OLD=26,26,666 / 2X=44,15,988) — guard uses 44,15,988.
        prov = JsonBacktestProvider()
        assert prov.get_reference_stats()["backtest_2x_pnl"] == 4415988
