"""Backtests the ACTUAL risk-overlay rule sketched from validate_density_risk_signal.py's
finding: does scaling exposure by fx-pcn's mean_abs_partial_corr (NOT density --
see that script, density was rejected as a signal) actually improve risk-adjusted
returns for a naive equal-weighted 7-pair FX basket, versus holding it at
constant exposure?

This is a DIFFERENT question from validate_density_risk_signal.py's correlation
check: that showed mean_abs_partial_corr predicts forward volatility; this tests
whether ACTING on that prediction (by cutting exposure) actually helps, net of
giving up some upside during periods that turn out fine. There is currently no
validated alpha strategy to attach this overlay to (see this session's memory --
both the forex-ML GBT re-validation and the fx-pcn directional lead-lag work were
rejected), so the naive basket itself stands in as "the strategy being sized" --
this tests the OVERLAY MECHANISM's value in isolation, not any real edge.

Overlay rule (illustrative thresholds -- NOT fitted/optimized on this data, to
avoid the obvious circularity of tuning thresholds against the same series used
to validate the underlying correlation):
    trailing walk-forward percentile of mean_abs_partial_corr (252-trading-day
    rolling window, min_periods=60) ->
        < 0.50               -> 1.00x exposure  (Normal)
        0.50 <= p < 0.80      -> 0.60x exposure  (Elevated)
        p >= 0.80             -> 0.25x exposure  (High)

Non-lookahead: exposure for day t is set from day (t-1)'s percentile bucket --
you can't act on today's network fit before today's bars have already happened.
Same shift-by-one-day discipline as validate_density_risk_signal.py's forward-
return construction.

Run from forex-ML/ with forex-strategy's venv:
    uv run --project ../forex-strategy python scripts/backtest_density_risk_overlay.py
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from validate_density_risk_signal import REGIMES, load_daily_returns

ROLLING_WINDOW_DAYS = 252
ROLLING_MIN_PERIODS = 60
RECENT_CUTOFF = datetime.date(2023, 1, 1)
TRADING_DAYS_PER_YEAR = 252


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Walk-forward percentile RANK of each value within its own trailing window
    (inclusive of itself) -- never uses a date's OWN period to judge a value from
    an EARLIER date, and never uses future dates to define what "high" means at
    any given point in history."""
    return series.rolling(window, min_periods=min_periods).apply(lambda x: (x <= x[-1]).mean(), raw=True)


def exposure_from_percentile(pct: pd.Series) -> pd.Series:
    exposure = pd.Series(1.0, index=pct.index)
    exposure[(pct >= 0.50) & (pct < 0.80)] = 0.60
    exposure[pct >= 0.80] = 0.25
    exposure[pct.isna()] = np.nan  # not enough history yet to classify -- excluded, not defaulted to full size
    return exposure


def sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std(ddof=1) == 0 or len(daily_returns) < 2:
        return float("nan")
    return daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(daily_returns: pd.Series) -> float:
    cum = daily_returns.cumsum()
    running_max = cum.cummax()
    return float((running_max - cum).max())


def summarize(label: str, daily_returns: pd.Series) -> dict:
    ann_return = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "label": label,
        "n_days": len(daily_returns),
        "ann_return_pct": ann_return * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe": sharpe(daily_returns),
        "max_drawdown_pct": max_drawdown(daily_returns) * 100,
    }


def print_summary(rows: list[dict]) -> None:
    header = f"{'':22s} {'n_days':>7s} {'ann_ret%':>9s} {'ann_vol%':>9s} {'sharpe':>7s} {'max_dd%':>8s}"
    print(header)
    for row in rows:
        print(f"{row['label']:22s} {row['n_days']:7d} {row['ann_return_pct']:9.2f} "
              f"{row['ann_vol_pct']:9.2f} {row['sharpe']:7.3f} {row['max_drawdown_pct']:8.2f}")


def run_for_regime(regime_label: str, density_path: str, returns: pd.DataFrame, basket_ret: pd.Series) -> None:
    print(f"\n{'#' * 70}\n# REGIME: {regime_label}\n{'#' * 70}")
    density = pd.read_parquet(density_path).set_index("date").sort_index()

    pct = rolling_percentile(density["mean_abs_partial_corr"], ROLLING_WINDOW_DAYS, ROLLING_MIN_PERIODS)
    exposure = exposure_from_percentile(pct)
    # Non-lookahead: TODAY's exposure comes from YESTERDAY's regime classification.
    exposure_lagged = exposure.shift(1)

    combined = pd.DataFrame({"basket_ret": basket_ret, "exposure": exposure_lagged}).dropna()
    if len(combined) < 50:
        print(f"  too few overlapping rows (n={len(combined)}), skipping")
        return

    baseline_ret = combined["basket_ret"]
    overlay_ret = combined["basket_ret"] * combined["exposure"]

    time_in_regime = exposure_lagged.dropna().value_counts(normalize=True).sort_index()
    print("Time spent at each exposure level (fraction of days, full overlap period):")
    for exp_level, frac in time_in_regime.items():
        print(f"  {exp_level:.2f}x: {frac * 100:5.1f}%")

    print(f"\n{'FULL HISTORY':}")
    print_summary([
        summarize("baseline (1.0x always)", baseline_ret),
        summarize("overlay (scaled)", overlay_ret),
    ])

    recent = combined[combined.index >= RECENT_CUTOFF]
    if len(recent) >= 50:
        print(f"\n{'RECENT (2023+)':}")
        print_summary([
            summarize("baseline (1.0x always)", recent["basket_ret"]),
            summarize("overlay (scaled)", recent["basket_ret"] * recent["exposure"]),
        ])
    else:
        print(f"\nRECENT (2023+): too few rows (n={len(recent)}), skipping")


def main() -> None:
    returns = load_daily_returns()
    basket_ret = returns.mean(axis=1)

    for regime_label in ("default (H1)", "intraday (M15)"):
        run_for_regime(regime_label, REGIMES[regime_label]["path"], returns, basket_ret)

    print("\nALL_DONE")


if __name__ == "__main__":
    main()
