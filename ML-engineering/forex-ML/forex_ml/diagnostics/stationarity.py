"""Stationarity diagnostics for engineered features.

Uses ADF and KPSS together — the standard combination, since they test opposite
null hypotheses and each has blind spots the other catches:
  - ADF: null hypothesis is "the series has a unit root" (is non-stationary).
    Rejecting the null is evidence FOR stationarity.
  - KPSS: null hypothesis is "the series IS stationary". Rejecting the null is
    evidence AGAINST stationarity.
Both agreeing is a strong signal either way; disagreeing means the series is in a
genuinely ambiguous zone worth a human look, not something to silently resolve by
picking one test over the other.
"""

from __future__ import annotations

import argparse
import warnings
from typing import cast

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from statsmodels.tsa.stattools import InterpolationWarning, adfuller, kpss

from forex_ml.config import load_params
from forex_ml.paths import non_time_series_parquet_path, pair_key


def check_stationarity(series: np.ndarray, alpha: float = 0.05) -> dict:
    series = series[~np.isnan(series)]

    _, adf_p_value, *_ = adfuller(series, autolag="AIC")
    # KPSS emits an InterpolationWarning whenever the p-value falls outside its
    # lookup table's range (very small or very large) -- expected and harmless here,
    # not a sign of a bad series; the returned p-value is still a valid clamped bound.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        _, kpss_p_value, *_ = kpss(series, regression="c", nlags="auto")

    adf_stationary = bool(adf_p_value < alpha)
    kpss_stationary = bool(kpss_p_value >= alpha)

    if adf_stationary and kpss_stationary:
        verdict = "stationary"
    elif not adf_stationary and not kpss_stationary:
        verdict = "non-stationary"
    else:
        verdict = "inconclusive (ADF and KPSS disagree)"

    return {
        "adf_p_value": float(adf_p_value),
        "adf_stationary": adf_stationary,
        "kpss_p_value": float(kpss_p_value),
        "kpss_stationary": kpss_stationary,
        "verdict": verdict,
    }


def check_pair_stationarity(
    spark: SparkSession,
    output_dir: str,
    instrument: str,
    granularity: str,
    n_back: int,
    lookahead: int,
    columns: list[str],
    alpha: float = 0.05,
) -> dict[str, dict]:
    """Run the stationarity check against a pair's real Stage-1 output for each
    column in `columns` (typically split.columns_x from params.yaml)."""
    key = pair_key(instrument, granularity, n_back, lookahead)
    pdf = cast(
        pd.DataFrame,
        spark.read.parquet(str(non_time_series_parquet_path(output_dir, key)))
        .orderBy("unix_epoch_s")
        .select(*columns)
        .toPandas(),
    )
    return {column: check_stationarity(pdf[column].to_numpy(), alpha=alpha) for column in columns}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADF/KPSS stationarity check for a pair's engineered feature columns."
    )
    parser.add_argument("--instrument", required=True, help="e.g. EUR/USD")
    parser.add_argument("--granularity", required=True, help="e.g. H1")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--params", default=None, help="Path to params.yaml (default: repo root)")
    args = parser.parse_args()

    params = load_params(args.params) if args.params else load_params()
    spark = SparkSession.builder.appName("forex-ml-stationarity-diagnostic").getOrCreate()

    results = check_pair_stationarity(
        spark, params.feature.output_dir, args.instrument, args.granularity,
        params.feature.n_back, params.feature.lookahead, params.split.columns_x, alpha=args.alpha,
    )

    print(f"{args.instrument} {args.granularity} — stationarity of split.columns_x")
    for column, result in results.items():
        print(
            f"  {column:20s} ADF p={result['adf_p_value']:.4f}  "
            f"KPSS p={result['kpss_p_value']:.4f}  -> {result['verdict']}"
        )


if __name__ == "__main__":
    main()
