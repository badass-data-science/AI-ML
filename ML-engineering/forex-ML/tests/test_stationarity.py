from __future__ import annotations

import numpy as np

from forex_ml.config import FeatureParams
from forex_ml.diagnostics.stationarity import check_pair_stationarity, check_stationarity
from forex_ml.flows.prepare_data_flow import engineer_and_save_task


def test_white_noise_is_classified_stationary():
    rng = np.random.default_rng(0)
    series = rng.normal(size=1000)
    result = check_stationarity(series)
    assert result["verdict"] == "stationary"
    assert result["adf_stationary"] is True
    assert result["kpss_stationary"] is True


def test_random_walk_is_classified_non_stationary():
    """A random walk (cumulative sum of noise) has a unit root by construction --
    the textbook example of a non-stationary series."""
    rng = np.random.default_rng(1)
    series = rng.normal(size=1000).cumsum()
    result = check_stationarity(series)
    assert result["verdict"] == "non-stationary"
    assert result["adf_stationary"] is False
    assert result["kpss_stationary"] is False


def test_check_pair_stationarity_runs_against_real_stage1_output(spark, synthetic_candles, tmp_path):
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

    results = check_pair_stationarity(
        spark, str(tmp_path), "EUR/USD", "H1", n_back=10, lookahead=2,
        columns=["volatility", "return", "diff_spread_close", "diff_volume", "day_sin", "day_cos"],
    )
    assert set(results.keys()) == {"volatility", "return", "diff_spread_close", "diff_volume", "day_sin", "day_cos"}
    for result in results.values():
        assert result["verdict"] in ("stationary", "non-stationary", "inconclusive (ADF and KPSS disagree)")
