"""ACF/PACF diagnostics to sanity-check `n_back`/`lookahead` in params.yaml.

Those two constants were carried over from the original notebooks with no empirical
justification. Autocorrelation (how correlated a series is with itself at increasing
lags) and partial autocorrelation (the same, with the effect of intermediate lags
removed) tell you roughly where LINEAR memory in a series ends — the lag at which the
autocorrelation's confidence interval starts including zero. That's a floor, not a
ceiling: an LSTM can exploit nonlinear, multi-feature structure an ACF can't see. But
if `n_back` is wildly larger than where ACF says memory ends, that's worth checking
rather than assuming.

Also reports effect sizes, not just "statistically distinguishable from zero" — the
same statistical-vs-practical-significance gap as the ADF/KPSS diagnostics
(`stationarity.py`). With the hundreds-to-thousands of bars typical here, the ACF/PACF
confidence interval narrows enough that even a tiny correlation (0.02) can clear
"significant," which would make `suggested_min_lookback` track sample size rather than
real memory. Two things address that:
  - the raw |ACF|/|PACF| magnitude at the suggested lag and the max magnitude across
    all computed lags, so you can see just how small "still significant" actually is.
  - `practical_min_lookback` — the first lag where |ACF| (or |PACF|) drops below a
    fixed threshold (default 0.1) that does NOT shrink as the sample grows, giving a
    second answer anchored to correlation strength rather than significance.
PACF gets its own `suggested_min_lookback_pacf`/`practical_min_lookback_pacf` rather
than reusing ACF's: PACF is the more standard tool for picking an AR order/cutoff
(it tends to drop sharply at the true order, where ACF decays gradually), so the two
can legitimately disagree and both are worth seeing.
"""

from __future__ import annotations

import argparse
from typing import cast

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from statsmodels.tsa.stattools import acf, pacf

from forex_ml.config import load_params
from forex_ml.paths import non_time_series_parquet_path, pair_key


def compute_acf_pacf(series: np.ndarray, nlags: int, alpha: float = 0.05) -> dict:
    """ACF/PACF values and their (1-alpha) confidence bounds for a 1D series."""
    acf_values, acf_confint = acf(series, nlags=nlags, alpha=alpha, fft=True)
    pacf_values, pacf_confint = pacf(series, nlags=nlags, alpha=alpha)
    return {
        "acf": acf_values,
        "acf_confint": acf_confint,
        "pacf": pacf_values,
        "pacf_confint": pacf_confint,
    }


def suggest_min_lookback(values: np.ndarray, confint: np.ndarray) -> int:
    """First lag (>=1) whose confidence interval includes zero — i.e. the first lag no
    longer statistically distinguishable from "no linear (partial) autocorrelation".
    Works on either ACF or PACF output. Returns the last computed lag if it never
    decays within `nlags`."""
    for lag in range(1, len(values)):
        lo, hi = confint[lag]
        if lo <= 0 <= hi:
            return lag
    return len(values) - 1


def _effect_size(values: np.ndarray, suggested_lag: int, practical_threshold: float) -> dict:
    """Magnitude-based effect size for an ACF or PACF array, to distinguish
    "statistically significant" from "practically meaningful" (see module docstring).
    """
    magnitudes = np.abs(values[1:])  # lag 0 is always 1.0 by definition, not informative
    practical_min_lookback = next(
        (lag for lag in range(1, len(values)) if abs(values[lag]) < practical_threshold),
        len(values) - 1,
    )
    return {
        "magnitude_at_suggested_lookback": float(abs(values[suggested_lag])),
        "max_abs_magnitude": float(magnitudes.max()) if magnitudes.size else 0.0,
        "practical_min_lookback": practical_min_lookback,
    }


def diagnose_series(
    series: np.ndarray, nlags: int, alpha: float = 0.05, practical_threshold: float = 0.1
) -> dict:
    series = series[~np.isnan(series)]
    nlags = min(nlags, len(series) // 2 - 1)
    result = compute_acf_pacf(series, nlags=nlags, alpha=alpha)
    result["n_observations"] = len(series)

    result["suggested_min_lookback"] = suggest_min_lookback(result["acf"], result["acf_confint"])
    acf_effect = _effect_size(result["acf"], result["suggested_min_lookback"], practical_threshold)
    result["acf_magnitude_at_suggested_lookback"] = acf_effect["magnitude_at_suggested_lookback"]
    result["acf_max_abs_magnitude"] = acf_effect["max_abs_magnitude"]
    result["practical_min_lookback"] = acf_effect["practical_min_lookback"]

    result["suggested_min_lookback_pacf"] = suggest_min_lookback(result["pacf"], result["pacf_confint"])
    pacf_effect = _effect_size(result["pacf"], result["suggested_min_lookback_pacf"], practical_threshold)
    result["pacf_magnitude_at_suggested_lookback"] = pacf_effect["magnitude_at_suggested_lookback"]
    result["pacf_max_abs_magnitude"] = pacf_effect["max_abs_magnitude"]
    result["practical_min_lookback_pacf"] = pacf_effect["practical_min_lookback"]

    return result


def diagnose_pair(
    spark: SparkSession,
    output_dir: str,
    instrument: str,
    granularity: str,
    n_back: int,
    lookahead: int,
    column: str = "pd_lead",
    nlags: int = 250,
    practical_threshold: float = 0.1,
) -> dict:
    """Run the diagnostic against a pair's real Stage-1 output (the actual target
    column, by default `pd_lead`) rather than synthetic data."""
    key = pair_key(instrument, granularity, n_back, lookahead)
    pdf = cast(
        pd.DataFrame,
        spark.read.parquet(str(non_time_series_parquet_path(output_dir, key)))
        .orderBy("unix_epoch_s")
        .select(column)
        .toPandas(),
    )
    return diagnose_series(pdf[column].to_numpy(), nlags=nlags, practical_threshold=practical_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACF/PACF diagnostic for a pair's target column, to sanity-check n_back/lookahead."
    )
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--column", default="pd_lead", help="Target column to diagnose (default: pd_lead)")
    parser.add_argument("--nlags", type=int, default=250, help="Max lags to check (default: 250)")
    parser.add_argument(
        "--practical-threshold", type=float, default=0.1,
        help="|ACF|/|PACF| cutoff for practical_min_lookback, independent of sample size (default: 0.1)",
    )
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    args = parser.parse_args()

    params = load_params(args.params) if args.params else load_params()
    # See forex_ml.flows.prepare_data_flow's driver-memory note -- same reason, same fix.
    spark = (
        SparkSession.builder.appName("forex-ml-acf-pacf-diagnostic")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )

    result = diagnose_pair(
        spark, params.feature.output_dir, args.instrument, args.granularity,
        params.feature.n_back, params.feature.lookahead, column=args.column, nlags=args.nlags,
        practical_threshold=args.practical_threshold,
    )

    print(f"{args.instrument} {args.granularity} — {args.column}")
    print(f"  observations:            {result['n_observations']}")
    print(f"  configured n_back:       {params.feature.n_back}")
    print(f"  configured lookahead:    {params.feature.lookahead}")
    print(f"  ACF  suggested min lookback:      {result['suggested_min_lookback']} bars "
          f"(first lag no longer statistically significant); "
          f"|ACF| there = {result['acf_magnitude_at_suggested_lookback']:.3f}, "
          f"max |ACF| = {result['acf_max_abs_magnitude']:.3f}")
    print(f"  ACF  practical min lookback:      {result['practical_min_lookback']} bars "
          f"(first lag with |ACF| < {args.practical_threshold})")
    print(f"  PACF suggested min lookback:      {result['suggested_min_lookback_pacf']} bars "
          f"(first lag no longer statistically significant); "
          f"|PACF| there = {result['pacf_magnitude_at_suggested_lookback']:.3f}, "
          f"max |PACF| = {result['pacf_max_abs_magnitude']:.3f}")
    print(f"  PACF practical min lookback:      {result['practical_min_lookback_pacf']} bars "
          f"(first lag with |PACF| < {args.practical_threshold})")

    if result["suggested_min_lookback"] < params.feature.n_back / 4:
        print(
            f"  NOTE: n_back ({params.feature.n_back}) is more than 4x the ACF-suggested "
            f"minimum — worth checking whether this much history is actually used."
        )
    if result["acf_magnitude_at_suggested_lookback"] < args.practical_threshold:
        print(
            f"  NOTE: the ACF suggested_min_lookback is driven by statistical significance, "
            f"not correlation strength — |ACF| there ({result['acf_magnitude_at_suggested_lookback']:.3f}) "
            f"is already below the practical threshold ({args.practical_threshold}). With this many "
            f"observations, statistical and practical significance have diverged; "
            f"practical_min_lookback ({result['practical_min_lookback']}) is the more conservative read."
        )


if __name__ == "__main__":
    main()
