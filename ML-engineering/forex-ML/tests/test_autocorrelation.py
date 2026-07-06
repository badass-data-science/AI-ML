from __future__ import annotations

import numpy as np

from forex_ml.config import FeatureParams
from forex_ml.diagnostics.autocorrelation import diagnose_pair, diagnose_series
from forex_ml.flows.prepare_data_flow import engineer_and_save_task


def _ar1_series(rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.normal()
    return series


def test_white_noise_has_a_short_suggested_lookback():
    rng = np.random.default_rng(0)
    series = rng.normal(size=2000)
    result = diagnose_series(series, nlags=50)
    assert result["suggested_min_lookback"] <= 5


def test_ar1_process_suggests_a_lookback_within_the_expected_range():
    # rho=0.8 over n=3000: rho**30 ~= 0.001, so real ACF should fall below
    # significance well before lag 60 -- bounded loosely since real samples are noisy
    # and can cross back and forth near zero.
    series = _ar1_series(rho=0.8, n=3000, seed=1)
    result = diagnose_series(series, nlags=100)
    assert 3 <= result["suggested_min_lookback"] <= 60


def test_more_persistent_process_suggests_a_longer_lookback():
    low_persistence = diagnose_series(_ar1_series(rho=0.3, n=3000, seed=10), nlags=100)
    high_persistence = diagnose_series(_ar1_series(rho=0.95, n=3000, seed=11), nlags=100)
    assert high_persistence["suggested_min_lookback"] > low_persistence["suggested_min_lookback"]


def test_diagnose_pair_runs_against_real_stage1_output(spark, synthetic_candles, tmp_path):
    """End-to-end: run real Stage-1 feature engineering, then diagnose the resulting
    pd_lead column -- proves the Spark read + column selection actually works, not
    just the underlying statsmodels call."""
    params = FeatureParams(
        instruments=["EUR/USD"],
        granularities=["H1"],
        n_back=10,
        lookahead=2,
        ma_lookback_list=[3, 5],
        columns_base=["mid_open", "mid_high", "mid_low", "mid_close", "spread_close", "volume"],
        ma_columns_list=["volatility", "return", "diff_spread_close", "diff_volume"],
        training_and_testing=True,
        min_training_timestamp="2020-01-01T00:00:00",
        output_dir=str(tmp_path),
    )
    engineer_and_save_task(spark, synthetic_candles, "EUR/USD", "H1", params)

    result = diagnose_pair(spark, str(tmp_path), "EUR/USD", "H1", n_back=10, lookahead=2, nlags=20)
    assert result["n_observations"] > 0
    assert result["suggested_min_lookback"] >= 1
