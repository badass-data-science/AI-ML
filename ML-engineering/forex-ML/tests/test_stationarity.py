from __future__ import annotations

import numpy as np
import pytest

from forex_ml.config import FeatureParams
from forex_ml.diagnostics.stationarity import check_pair_stationarity, check_stationarity
from forex_ml.flows.prepare_data_flow import engineer_and_save_task


def _ar1_series(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = phi * series[t - 1] + rng.normal()
    return series


def test_white_noise_is_classified_stationary():
    rng = np.random.default_rng(0)
    series = rng.normal(size=1000)
    result = check_stationarity(series)
    assert result["verdict"] == "stationary"
    assert result["adf_stationary"] is True
    assert result["kpss_stationary"] is True
    assert abs(result["phi_hat"]) < 0.15  # white noise: no real persistence
    assert result["half_life_bars"] < 2
    assert result["kpss_ratio_to_5pct"] < 1.0  # comfortably below the boundary


def test_random_walk_is_classified_non_stationary():
    """A random walk (cumulative sum of noise) has a unit root by construction --
    the textbook example of a non-stationary series."""
    rng = np.random.default_rng(1)
    series = rng.normal(size=1000).cumsum()
    result = check_stationarity(series)
    assert result["verdict"] == "non-stationary"
    assert result["adf_stationary"] is False
    assert result["kpss_stationary"] is False
    assert result["phi_hat"] > 0.95  # a random walk's phi is exactly 1 in theory
    assert result["half_life_bars"] > 50
    assert result["kpss_ratio_to_5pct"] > 1.0  # crosses the boundary


def test_kpss_ratio_is_graduated_not_just_binary():
    """The whole point of surfacing kpss_stat instead of only the p-value: a series
    far past the non-stationarity boundary should show a much larger ratio than one
    only just past it, even though a bare pass/fail read would call both "the same"
    (both non-stationary)."""
    rng = np.random.default_rng(4)
    mild_random_walk = rng.normal(size=200).cumsum()  # short series, weaker signal
    rng2 = np.random.default_rng(5)
    strong_random_walk = rng2.normal(size=3000).cumsum()  # long series, strong signal

    mild_result = check_stationarity(mild_random_walk)
    strong_result = check_stationarity(strong_random_walk)

    assert mild_result["kpss_ratio_to_5pct"] > 1.0
    assert strong_result["kpss_ratio_to_5pct"] > mild_result["kpss_ratio_to_5pct"]


def test_ar1_effect_size_recovers_the_true_phi():
    """The effect size should track the actual generating AR(1) coefficient, not
    just move in the right direction -- pins down the estimate quantitatively rather
    than only checking sign/verdict."""
    true_phi = 0.8
    series = _ar1_series(true_phi, n=3000, seed=2)
    result = check_stationarity(series)

    assert result["phi_hat"] == pytest.approx(true_phi, abs=0.05)
    theoretical_half_life = np.log(0.5) / np.log(true_phi)
    assert result["half_life_bars"] == pytest.approx(theoretical_half_life, rel=0.2)


def test_highly_persistent_series_can_be_statistically_stationary_yet_practically_slow():
    """The exact scenario effect size exists to catch: at phi=0.98, ADF and KPSS
    disagree even at n=5000 (verdict lands on "inconclusive") -- a p-value-only
    report would leave you stuck there with no further signal. The half-life cuts
    through the ambiguity regardless: it's long, clearly showing this series is
    slow to mean-revert in practice, whatever the p-value verdict says."""
    series = _ar1_series(phi=0.98, n=5000, seed=3)
    result = check_stationarity(series)

    assert result["verdict"] in ("stationary", "inconclusive (ADF and KPSS disagree)")
    # theoretical half-life at phi=0.98 is ln(0.5)/ln(0.98) ~= 34 bars; bounded
    # loosely since this is a single noisy sample estimate, not the true parameter
    assert result["half_life_bars"] > 15  # slow to mean-revert in practice, either way


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
