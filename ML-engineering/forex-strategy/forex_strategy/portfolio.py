"""Portfolio-level analysis across multiple pairs' independent backtests --
everything in `backtest.py` evaluates one currency pair in isolation; this module
answers the question that raises: what happens when you trade several validated
pairs at once? Do their trades cluster on the same days (shared risk) or spread
out (real diversification), and does the combined equity curve actually look
better than either pair alone?

Each pair's own `simulate_trades` run already produces one net P&L value per
CLOSED trade, realized at that trade's exit -- this module buckets those
per-trade results into a daily P&L series (the standard unit for portfolio
statistics like Sharpe ratio and drawdown), then combines multiple pairs'
daily series under a capital allocation.
"""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_NY = ZoneInfo("America/New_York")


def bucket_trades_by_day(entry_timestamp: np.ndarray, net_pnl_pct: np.ndarray) -> pd.Series:
    """Sums net_pnl_pct into one value per calendar day (America/New_York, matching
    this project's rollover/session convention elsewhere), indexed by date. A day
    with no trades simply has no entry -- callers combining multiple pairs should
    reindex/fill with 0.0 over their union of dates, not assume every pair traded
    every day.
    """
    if len(entry_timestamp) != len(net_pnl_pct):
        raise ValueError("entry_timestamp and net_pnl_pct must be the same length")
    if len(entry_timestamp) == 0:
        return pd.Series(dtype=float)

    dates = [datetime.datetime.fromtimestamp(ts, tz=_NY).date() for ts in entry_timestamp]
    return pd.Series(net_pnl_pct, index=pd.Index(dates)).groupby(level=0).sum().sort_index()


def combine_portfolio_daily_pnl(daily_pnl_by_pair: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Combines each pair's own daily P&L series under a capital allocation
    (`weights`, e.g. {"USD/CHF": 0.5, "USD/JPY": 0.5}) into one portfolio daily
    P&L series over the union of all dates any pair traded. A pair's weighted
    contribution on a day it didn't trade is 0.0, not missing -- the portfolio
    was still fully allocated that day, just not earning/losing on that pair.
    """
    missing = set(daily_pnl_by_pair) - set(weights)
    if missing:
        raise ValueError(f"weights missing entries for: {sorted(missing)}")

    all_dates = sorted(set().union(*(series.index for series in daily_pnl_by_pair.values())))
    combined = pd.Series(0.0, index=pd.Index(all_dates))
    for pair, series in daily_pnl_by_pair.items():
        combined = combined.add(series.reindex(all_dates, fill_value=0.0) * weights[pair], fill_value=0.0)
    return combined.sort_index()


def sharpe_ratio(daily_pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio (mean / std of the daily P&L series, scaled by
    sqrt(periods_per_year)) -- NOT excess-of-risk-free, since every P&L figure in
    this project is already the net-of-cost return on a trade, not compared
    against a risk-free benchmark. Returns NaN if there isn't enough data (fewer
    than 2 days, or a zero-variance series) to compute a ratio at all.
    """
    if len(daily_pnl) < 2 or daily_pnl.std(ddof=1) == 0:
        return float("nan")
    return float(daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(daily_pnl: pd.Series) -> float:
    """Largest peak-to-trough decline in the CUMULATIVE P&L series, as a positive
    number (e.g. 12.5 means a 12.5-percentage-point drawdown at the worst point) --
    computed on the running total daily_pnl builds up to, not on daily_pnl itself.
    """
    if len(daily_pnl) == 0:
        return 0.0
    cumulative = daily_pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = running_max - cumulative
    return float(drawdown.max())


def pairwise_correlation(daily_pnl_by_pair: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation matrix between pairs' daily P&L series, over the union of all
    dates any pair traded (missing days filled with 0.0 -- a day one pair sat out
    is a real, informative "no P&L" data point for the correlation, not a gap to
    drop). Low/negative off-diagonal correlation is the signature of real
    diversification; a correlation near +1 means the two pairs are effectively
    making the same bet.
    """
    all_dates = sorted(set().union(*(series.index for series in daily_pnl_by_pair.values())))
    aligned = pd.DataFrame({
        pair: series.reindex(all_dates, fill_value=0.0) for pair, series in daily_pnl_by_pair.items()
    })
    return aligned.corr()
