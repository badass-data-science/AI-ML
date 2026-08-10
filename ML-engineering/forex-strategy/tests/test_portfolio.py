from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from forex_strategy.portfolio import (
    bucket_trades_by_day,
    combine_portfolio_daily_pnl,
    max_drawdown,
    pairwise_correlation,
    sharpe_ratio,
)

_NY = ZoneInfo("America/New_York")


def _ts(*args) -> float:
    return datetime.datetime(*args, tzinfo=_NY).timestamp()


def test_bucket_trades_by_day_sums_same_day_trades():
    timestamps = np.array([_ts(2024, 1, 1, 10), _ts(2024, 1, 1, 14), _ts(2024, 1, 2, 9)])
    net_pnl_pct = np.array([1.0, 2.0, -0.5])

    result = bucket_trades_by_day(timestamps, net_pnl_pct)

    assert result[datetime.date(2024, 1, 1)] == 3.0
    assert result[datetime.date(2024, 1, 2)] == -0.5


def test_bucket_trades_by_day_empty_input():
    result = bucket_trades_by_day(np.array([]), np.array([]))
    assert len(result) == 0


def test_bucket_trades_by_day_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        bucket_trades_by_day(np.array([1.0, 2.0]), np.array([1.0]))


def test_combine_portfolio_daily_pnl_weights_and_unions_dates():
    pair_a = pd.Series([10.0], index=[datetime.date(2024, 1, 1)])
    pair_b = pd.Series([4.0], index=[datetime.date(2024, 1, 2)])

    combined = combine_portfolio_daily_pnl(
        {"A": pair_a, "B": pair_b}, weights={"A": 0.5, "B": 0.5},
    )

    # Day 1: only A traded (weighted 0.5 * 10 = 5.0); B contributes 0 that day.
    assert combined[datetime.date(2024, 1, 1)] == 5.0
    # Day 2: only B traded (weighted 0.5 * 4 = 2.0).
    assert combined[datetime.date(2024, 1, 2)] == 2.0


def test_combine_portfolio_daily_pnl_rejects_missing_weight():
    pair_a = pd.Series([1.0], index=[datetime.date(2024, 1, 1)])
    with pytest.raises(ValueError):
        combine_portfolio_daily_pnl({"A": pair_a}, weights={})


def test_sharpe_ratio_matches_manual_calculation():
    daily_pnl = pd.Series([1.0, -0.5, 2.0, 0.5, -1.0])
    expected = daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(daily_pnl) == pytest.approx(expected)


def test_sharpe_ratio_nan_for_zero_variance_or_too_few_points():
    assert np.isnan(sharpe_ratio(pd.Series([1.0])))
    assert np.isnan(sharpe_ratio(pd.Series([1.0, 1.0, 1.0])))  # zero variance


def test_max_drawdown_finds_the_worst_peak_to_trough_decline():
    # Cumulative path: 5, 3, 8, 2, 6 -- worst drawdown is peak 8 -> trough 2 = 6.
    daily_pnl = pd.Series([5.0, -2.0, 5.0, -6.0, 4.0])
    assert max_drawdown(daily_pnl) == pytest.approx(6.0)


def test_max_drawdown_zero_for_monotonically_increasing_pnl():
    daily_pnl = pd.Series([1.0, 1.0, 1.0])
    assert max_drawdown(daily_pnl) == pytest.approx(0.0)


def test_max_drawdown_empty_series_is_zero():
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


def test_pairwise_correlation_perfectly_correlated_pairs():
    dates = [datetime.date(2024, 1, i) for i in range(1, 6)]
    pair_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    pair_b = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0], index=dates)  # exactly 2x A

    corr = pairwise_correlation({"A": pair_a, "B": pair_b})

    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_pairwise_correlation_fills_missing_days_with_zero():
    # A trades every day; B only trades on day 1 -- days 2-3 should count as B=0.0
    # for correlation purposes, not be dropped from the comparison.
    dates_a = [datetime.date(2024, 1, i) for i in range(1, 4)]
    pair_a = pd.Series([1.0, -1.0, 1.0], index=dates_a)
    pair_b = pd.Series([5.0], index=[datetime.date(2024, 1, 1)])

    corr = pairwise_correlation({"A": pair_a, "B": pair_b})

    assert corr.shape == (2, 2)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
