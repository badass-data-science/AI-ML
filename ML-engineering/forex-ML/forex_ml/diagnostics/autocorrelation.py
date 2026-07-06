"""ACF/PACF diagnostics to sanity-check `n_back`/`lookahead` in params.yaml.

Those two constants were carried over from the original notebooks with no empirical
justification. Autocorrelation (how correlated a series is with itself at increasing
lags) and partial autocorrelation (the same, with the effect of intermediate lags
removed) tell you roughly where LINEAR memory in a series ends — the lag at which the
autocorrelation's confidence interval starts including zero. That's a floor, not a
ceiling: an LSTM can exploit nonlinear, multi-feature structure an ACF can't see. But
if `n_back` is wildly larger than where ACF says memory ends, that's worth checking
rather than assuming.
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


def suggest_min_lookback(acf_values: np.ndarray, acf_confint: np.ndarray) -> int:
    """First lag (>=1) whose ACF confidence interval includes zero — i.e. the first
    lag no longer statistically distinguishable from "no linear autocorrelation".
    Returns the last computed lag if autocorrelation never decays within `nlags`."""
    for lag in range(1, len(acf_values)):
        lo, hi = acf_confint[lag]
        if lo <= 0 <= hi:
            return lag
    return len(acf_values) - 1


def diagnose_series(series: np.ndarray, nlags: int, alpha: float = 0.05) -> dict:
    series = series[~np.isnan(series)]
    nlags = min(nlags, len(series) // 2 - 1)
    result = compute_acf_pacf(series, nlags=nlags, alpha=alpha)
    result["n_observations"] = len(series)
    result["suggested_min_lookback"] = suggest_min_lookback(result["acf"], result["acf_confint"])
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
    return diagnose_series(pdf[column].to_numpy(), nlags=nlags)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACF/PACF diagnostic for a pair's target column, to sanity-check n_back/lookahead."
    )
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--column", default="pd_lead", help="Target column to diagnose (default: pd_lead)")
    parser.add_argument("--nlags", type=int, default=250, help="Max lags to check (default: 250)")
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    args = parser.parse_args()

    params = load_params(args.params) if args.params else load_params()
    spark = SparkSession.builder.appName("forex-ml-acf-pacf-diagnostic").getOrCreate()

    result = diagnose_pair(
        spark, params.feature.output_dir, args.instrument, args.granularity,
        params.feature.n_back, params.feature.lookahead, column=args.column, nlags=args.nlags,
    )

    print(f"{args.instrument} {args.granularity} — {args.column}")
    print(f"  observations:            {result['n_observations']}")
    print(f"  configured n_back:       {params.feature.n_back}")
    print(f"  configured lookahead:    {params.feature.lookahead}")
    print(f"  suggested min lookback:  {result['suggested_min_lookback']} bars "
          f"(first lag where ACF is no longer significant)")
    if result["suggested_min_lookback"] < params.feature.n_back / 4:
        print(
            f"  NOTE: n_back ({params.feature.n_back}) is more than 4x the suggested "
            f"minimum — worth checking whether this much history is actually used."
        )


if __name__ == "__main__":
    main()
