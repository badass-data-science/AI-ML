"""Tests for The Risk Desk MCP skill."""
from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from risk_desk_mcp_server import (
    AccountState,
    MarketRegime,
    PairLiquidity,
    TradeProposal,
    _current_session,
    _dominant_vix_set,
    _load_rules,
    _pair_to_oanda,
    evaluate_trade,
    get_market_regime,
)


@pytest.fixture(scope="session", autouse=True)
def load_rules():
    _load_rules()


# ---------------------------------------------------------------------------
# Input factories — sensible defaults that fire no rules on their own
# ---------------------------------------------------------------------------

def _trade(pair="EUR/USD", direction="long", size_pct=1.0):
    return TradeProposal(pair=pair, direction=direction, size_pct=size_pct)

def _regime(vix_level=15.0, regime="normal"):
    return MarketRegime(vix_level=vix_level, regime=regime)

def _account(balance=10_000.0, weekly_drawdown_pct=0.0, open_positions=0):
    return AccountState(balance=balance, weekly_drawdown_pct=weekly_drawdown_pct, open_positions=open_positions)

def _liquidity(pair="EUR/USD", spread_pips=1.0, session="london"):
    return PairLiquidity(pair=pair, spread_pips=spread_pips, session=session)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _fired(result, rule_id):
    return any(v.rule_id == rule_id for v in result.verdicts)


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

class TestPairToOanda:
    def test_slash_replaced(self):
        assert _pair_to_oanda("EUR/USD") == "EUR_USD"

    def test_jpy(self):
        assert _pair_to_oanda("USD/JPY") == "USD_JPY"

    @pytest.mark.parametrize("pair", ["GBP/USD", "AUD/USD", "NZD/USD", "USD/CHF", "USD/CAD"])
    def test_all_majors(self, pair):
        assert "/" not in _pair_to_oanda(pair)


class TestCurrentSession:
    @pytest.mark.parametrize("hour,expected", [
        (0,  "tokyo"),
        (6,  "tokyo"),
        (7,  "london"),
        (11, "london"),
        (12, "overlap"),
        (15, "overlap"),
        (16, "new-york"),
        (20, "new-york"),
        (21, "sydney"),
        (23, "sydney"),
    ])
    def test_session_boundaries(self, hour, expected):
        fake_now = datetime.datetime(2024, 1, 1, hour, 0, 0, tzinfo=datetime.timezone.utc)
        with patch("risk_desk_mcp_server.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fake_now
            mock_dt.timezone = datetime.timezone
            assert _current_session() == expected


class TestDominantVixSet:
    def test_highest_membership_wins(self):
        memberships = {"low": 0.3, "medium low": 0.7, "medium": 0.1}
        assert _dominant_vix_set(memberships) == "medium low"

    def test_tie_breaks_toward_higher_index(self):
        # "medium" (index 3) beats "low" (index 1) on equal degree
        memberships = {"low": 0.5, "medium": 0.5}
        assert _dominant_vix_set(memberships) == "medium"

    def test_single_entry(self):
        assert _dominant_vix_set({"very high": 1.0}) == "very high"


class TestGetMarketRegime:
    @pytest.mark.parametrize("vix,expected_regime", [
        (10.0, "calm"),
        (15.0, "normal"),
        (25.0, "elevated"),
        (80.0, "crisis"),
    ])
    def test_classifies_live_vix_value(self, vix, expected_regime):
        with patch("risk_desk_mcp_server.get_most_recent_vix", return_value=vix):
            result = get_market_regime()
        assert result.vix_level == vix
        assert result.regime == expected_regime


# ---------------------------------------------------------------------------
# CLIPS rule tests via evaluate_trade
# ---------------------------------------------------------------------------

class TestCleanTrade:
    def test_no_rules_fired_is_approved(self):
        r = evaluate_trade(_trade(), _regime(), _account(), _liquidity())
        assert r.overall == "APPROVED"
        assert r.verdicts == []


class TestRegimeRules:
    def test_001a_blocks_risk_on_long_in_crisis(self):
        r = evaluate_trade(_trade("EUR/USD", "long"), _regime(40.0, "crisis"), _account(), _liquidity())
        assert r.overall == "BLOCKED"
        assert _fired(r, "REGIME-001a")

    def test_001a_does_not_fire_for_risk_off_long(self):
        # USD/CAD long is risk-off — REGIME-001a must not fire
        r = evaluate_trade(_trade("USD/CAD", "long"), _regime(40.0, "crisis"), _account(), _liquidity("USD/CAD"))
        assert not _fired(r, "REGIME-001a")

    def test_001b_blocks_risk_on_short_in_crisis(self):
        # USD/CAD short is risk-on
        r = evaluate_trade(_trade("USD/CAD", "short"), _regime(40.0, "crisis"), _account(), _liquidity("USD/CAD"))
        assert r.overall == "BLOCKED"
        assert _fired(r, "REGIME-001b")

    def test_001b_does_not_fire_for_risk_off_short(self):
        # EUR/USD short is risk-off
        r = evaluate_trade(_trade("EUR/USD", "short"), _regime(40.0, "crisis"), _account(), _liquidity())
        assert not _fired(r, "REGIME-001b")

    def test_002_modifies_oversized_trade_in_elevated(self):
        r = evaluate_trade(_trade(size_pct=2.0), _regime(22.0, "elevated"), _account(), _liquidity())
        assert r.overall == "MODIFIED"
        assert _fired(r, "REGIME-002")

    def test_002_does_not_fire_at_the_cap(self):
        # size_pct == 1.0 is not > 1.0 — rule must stay silent
        r = evaluate_trade(_trade(size_pct=1.0), _regime(22.0, "elevated"), _account(), _liquidity())
        assert not _fired(r, "REGIME-002")

    def test_002_does_not_fire_in_normal_regime(self):
        r = evaluate_trade(_trade(size_pct=2.0), _regime(15.0, "normal"), _account(), _liquidity())
        assert not _fired(r, "REGIME-002")


class TestDrawdownRules:
    def test_dd001_blocks_at_threshold(self):
        r = evaluate_trade(_trade(), _regime(), _account(weekly_drawdown_pct=5.0), _liquidity())
        assert r.overall == "BLOCKED"
        assert _fired(r, "DD-001")

    def test_dd001_blocks_above_threshold(self):
        r = evaluate_trade(_trade(), _regime(), _account(weekly_drawdown_pct=7.5), _liquidity())
        assert _fired(r, "DD-001")

    def test_dd002_warns_in_soft_zone(self):
        r = evaluate_trade(_trade(), _regime(), _account(weekly_drawdown_pct=3.0), _liquidity())
        assert _fired(r, "DD-002")
        verdict = next(v for v in r.verdicts if v.rule_id == "DD-002")
        assert verdict.severity == "warning"

    def test_dd002_fires_at_upper_edge_of_soft_zone(self):
        r = evaluate_trade(_trade(), _regime(), _account(weekly_drawdown_pct=4.9), _liquidity())
        assert _fired(r, "DD-002")
        assert not _fired(r, "DD-001")

    def test_no_drawdown_rules_below_soft_zone(self):
        r = evaluate_trade(_trade(), _regime(), _account(weekly_drawdown_pct=2.9), _liquidity())
        assert not _fired(r, "DD-001")
        assert not _fired(r, "DD-002")


class TestLiquidityRules:
    def test_liq001_blocks_excessive_spread(self):
        r = evaluate_trade(_trade(), _regime(), _account(), _liquidity(spread_pips=3.1))
        assert r.overall == "BLOCKED"
        assert _fired(r, "LIQ-001")

    def test_liq001_does_not_fire_at_threshold(self):
        r = evaluate_trade(_trade(), _regime(), _account(), _liquidity(spread_pips=3.0))
        assert not _fired(r, "LIQ-001")

    def test_liq002_warns_eur_usd_in_tokyo(self):
        r = evaluate_trade(_trade("EUR/USD"), _regime(), _account(), _liquidity("EUR/USD", session="tokyo"))
        assert _fired(r, "LIQ-002")
        verdict = next(v for v in r.verdicts if v.rule_id == "LIQ-002")
        assert verdict.severity == "info"

    def test_liq002_warns_gbp_usd_in_tokyo(self):
        r = evaluate_trade(_trade("GBP/USD"), _regime(), _account(), _liquidity("GBP/USD", session="tokyo"))
        assert _fired(r, "LIQ-002")

    def test_liq002_does_not_fire_for_aud_usd_in_tokyo(self):
        r = evaluate_trade(_trade("AUD/USD"), _regime(), _account(), _liquidity("AUD/USD", session="tokyo"))
        assert not _fired(r, "LIQ-002")

    def test_liq002_does_not_fire_outside_tokyo(self):
        r = evaluate_trade(_trade("EUR/USD"), _regime(), _account(), _liquidity("EUR/USD", session="london"))
        assert not _fired(r, "LIQ-002")


class TestConcentrationRules:
    def test_conc001_blocks_at_hard_limit(self):
        r = evaluate_trade(_trade(), _regime(), _account(open_positions=5), _liquidity())
        assert r.overall == "BLOCKED"
        assert _fired(r, "CONC-001")

    def test_conc001_blocks_above_hard_limit(self):
        r = evaluate_trade(_trade(), _regime(), _account(open_positions=7), _liquidity())
        assert _fired(r, "CONC-001")

    def test_conc002_warns_in_soft_zone(self):
        r = evaluate_trade(_trade(), _regime(), _account(open_positions=3), _liquidity())
        assert _fired(r, "CONC-002")
        verdict = next(v for v in r.verdicts if v.rule_id == "CONC-002")
        assert verdict.severity == "warning"

    def test_conc002_fires_at_upper_edge_of_soft_zone(self):
        r = evaluate_trade(_trade(), _regime(), _account(open_positions=4), _liquidity())
        assert _fired(r, "CONC-002")
        assert not _fired(r, "CONC-001")

    def test_no_concentration_rules_below_soft_zone(self):
        r = evaluate_trade(_trade(), _regime(), _account(open_positions=2), _liquidity())
        assert not _fired(r, "CONC-001")
        assert not _fired(r, "CONC-002")


# ---------------------------------------------------------------------------
# Overall verdict precedence and summary
# ---------------------------------------------------------------------------

class TestOverallVerdictPrecedence:
    def test_blocked_takes_precedence_over_modified(self):
        # DD-001 fires BLOCKED; REGIME-002 fires MODIFIED — overall must be BLOCKED
        r = evaluate_trade(
            _trade(size_pct=2.0),
            _regime(22.0, "elevated"),
            _account(weekly_drawdown_pct=5.0),
            _liquidity(),
        )
        assert r.overall == "BLOCKED"
        assert _fired(r, "DD-001")
        assert _fired(r, "REGIME-002")

    def test_summary_singular(self):
        r = evaluate_trade(_trade(), _regime(), _account(weekly_drawdown_pct=5.0), _liquidity())
        assert r.summary.startswith("1 rule fired")

    def test_summary_plural(self):
        r = evaluate_trade(
            _trade(size_pct=2.0),
            _regime(22.0, "elevated"),
            _account(weekly_drawdown_pct=5.0),
            _liquidity(),
        )
        assert "rules fired" in r.summary
        assert "BLOCKED" in r.summary
