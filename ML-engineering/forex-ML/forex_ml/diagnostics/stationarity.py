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

Also reports effect sizes, not just the two p-values. With the hundreds-to-thousands
of bars typical here, ADF/KPSS will tend to reject the unit-root null for almost any
realistic series, even a highly persistent one (AR coefficient near 1, behaving
practically like a random walk over the horizons that matter for n_back/lookahead) —
statistical significance isn't the same as practical significance, and large samples
widen that gap. A short half-life and a long half-life can both get called
"stationary"; only the effect size tells them apart. Two effect sizes are reported,
matched to what each test actually estimates:
  - ADF: the AR(1) coefficient (phi_hat), reported as a half-life in bars — an
    interpretable real-world unit, since the ADF regression directly parametrizes
    persistence.
  - KPSS: the raw LM statistic as a multiple of its own 5% critical value
    (kpss_ratio_to_5pct). KPSS doesn't have as clean a real-units effect size as
    ADF's half-life — its statistic is a normalized measure of a random-walk
    variance component, not something that decomposes into bars or any other
    physical unit — but the raw statistic still carries graduated information a
    p-value alone throws away: a ratio of 3.0 is a much stronger non-stationarity
    signal than 1.01, even though both cross the "significant" line the same way.
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


def _ar1_effect_size(resstore) -> tuple[float, float]:
    """AR(1) coefficient (phi_hat) and half-life (in bars) from the ADF regression's
    own fitted OLS model, rather than a second, separately-fit regression.

    resstore.resols.params[0] is ALWAYS the coefficient on the lagged level term
    y_{t-1} in the ADF regression (delta_y_t = gamma*y_{t-1} + ... + epsilon_t),
    regardless of how many lagged-difference terms autolag picked or which
    deterministic terms (const/trend) were included — confirmed directly from
    statsmodels' adfuller source: the lagged-level column is always placed first,
    before any trend/const columns are appended (see H0/HA on resstore: "the
    coefficient on the lagged level equals 1" / "... is less than 1"). H0: gamma=0
    (phi=1, unit root); HA: gamma<0 (phi<1, mean-reverting).
    """
    gamma_hat = float(resstore.resols.params[0])
    phi_hat = 1.0 + gamma_hat
    half_life = float(np.log(0.5) / np.log(abs(phi_hat))) if abs(phi_hat) < 1 else float("inf")
    return phi_hat, half_life


def check_stationarity(series: np.ndarray, alpha: float = 0.05) -> dict:
    series = series[~np.isnan(series)]

    _, adf_p_value, _, resstore = adfuller(series, autolag="AIC", regression="c", regresults=True)
    phi_hat, half_life = _ar1_effect_size(resstore)

    # KPSS emits an InterpolationWarning whenever the p-value falls outside its
    # lookup table's range (very small or very large) -- expected and harmless here,
    # not a sign of a bad series; the returned p-value is still a valid clamped bound.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_stat, kpss_p_value, _, kpss_crit = kpss(series, regression="c", nlags="auto")
    # kpss_stat and kpss_crit were already being computed internally and thrown away
    # -- kpss_stat grows without bound as the random-walk-variance component KPSS
    # tests for grows, so unlike the p-value alone it carries graduated information;
    # normalizing by the 5% critical value gives a scale-free "how far past the
    # threshold" reading (1.0 = exactly at the boundary).
    kpss_ratio_to_5pct = float(kpss_stat) / float(kpss_crit["5%"])

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
        "phi_hat": phi_hat,
        "half_life_bars": half_life,
        "kpss_stat": float(kpss_stat),
        "kpss_ratio_to_5pct": kpss_ratio_to_5pct,
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
        half_life = result["half_life_bars"]
        half_life_str = f"{half_life:.1f} bars" if np.isfinite(half_life) else "inf"
        print(
            f"  {column:20s} ADF p={result['adf_p_value']:.4f}  "
            f"KPSS p={result['kpss_p_value']:.4f}  -> {result['verdict']:30s} "
            f"phi_hat={result['phi_hat']:.3f}  half_life={half_life_str}  "
            f"kpss_ratio_to_5pct={result['kpss_ratio_to_5pct']:.2f}"
        )


if __name__ == "__main__":
    main()
