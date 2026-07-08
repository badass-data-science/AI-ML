from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from forex_ml.config import FeatureParams
from forex_ml.diagnostics.feature_impact import (
    _parse_cross_pair_candidates,
    analyze_cross_pair_feature_impact,
    analyze_feature_impact,
    cross_correlation_report,
    granger_causality_report,
    lasso_importance_report,
    load_cross_pair_target_and_candidates,
    load_target_and_candidates,
    var_fevd_report,
)
from forex_ml.flows.prepare_data_flow import engineer_and_save_task


def _make_lead_lag_df(n: int = 2000, lead_lag: int = 5, seed: int = 0) -> pd.DataFrame:
    """`driver` genuinely predicts `target` `lead_lag` bars later; `noise` is
    independent white noise carrying no information about `target` at all."""
    rng = np.random.default_rng(seed)
    driver = rng.normal(size=n)
    noise = rng.normal(size=n)
    target = np.zeros(n)
    target[lead_lag:] = 0.8 * driver[:-lead_lag] + rng.normal(scale=0.2, size=n - lead_lag)
    return pd.DataFrame({"target": target, "driver": driver, "noise": noise})


def test_cross_correlation_report_finds_the_true_lag_and_ignores_noise():
    df = _make_lead_lag_df(lead_lag=5)
    report = cross_correlation_report(df, "target", ["driver", "noise"], max_lag=20)

    assert report["driver"]["best_lag"] == 5
    assert abs(report["driver"]["best_corr"]) > 0.5
    assert report["driver"]["practically_significant"] is True

    assert abs(report["noise"]["best_corr"]) < 0.15
    assert report["noise"]["practically_significant"] is False


def test_granger_causality_report_detects_real_relationship_and_bh_corrects():
    df = _make_lead_lag_df(lead_lag=3)
    report = granger_causality_report(df, "target", ["driver", "noise"], lag=3)

    assert report["driver"]["significant_after_correction"] is True
    assert report["noise"]["significant_after_correction"] is False
    assert "candidate_stationarity" in report["driver"]


def test_var_fevd_report_attributes_more_variance_to_the_true_driver():
    df = _make_lead_lag_df(lead_lag=2, n=3000)
    report = var_fevd_report(df, "target", ["driver", "noise"], lag_order=5, horizon=10)

    driver_share = report["fevd_fraction_of_target_variance"]["driver"]
    noise_share = report["fevd_fraction_of_target_variance"]["noise"]
    assert driver_share > noise_share
    assert report["causality"]["driver"]["significant_after_correction"] is True
    assert report["causality"]["noise"]["significant_after_correction"] is False
    assert report["rank_warning"] is None  # independent white-noise candidates: no structural collinearity


def test_var_fevd_report_flags_exact_collinearity_across_lags_of_a_fixed_period_sinusoid():
    """Regression test for a real finding on real data: a fixed-period sin/cos pair
    is EXACTLY linearly dependent across lags (angle-subtraction identity), which
    silently corrupted a block-exogeneity p-value for a genuinely high-FEVD-share
    candidate (day_cos, 47.7% FEVD share, came back "not significant"). Reproduced
    here with a clean synthetic sinusoid rather than trusting the real-data finding
    alone."""
    n = 2000
    t = np.arange(n)
    omega = 2 * np.pi / 24  # a 24-bar period, same idea as day_sin/day_cos
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "target": rng.normal(size=n),
        "cyc_sin": np.sin(omega * t),
        "cyc_cos": np.cos(omega * t),
    })
    report = var_fevd_report(df, "target", ["cyc_sin", "cyc_cos"], lag_order=10, horizon=5)
    assert report["rank_warning"] is not None
    assert "collinearity" in report["rank_warning"]


def test_lasso_importance_report_shrinks_the_irrelevant_candidate_toward_zero():
    """Lasso soft-thresholding exactly zeros a coefficient only when regularization
    strength exceeds that coefficient's correlation with the residual -- not
    guaranteed for a genuinely-irrelevant column at every finite random draw, so the
    robust thing to assert is "shrunk to a small fraction of the real signal's
    magnitude," not "exactly zero" (all_lags_zeroed is still reported and worth
    checking by eye; it just isn't a safe hard assertion here)."""
    df = _make_lead_lag_df(lead_lag=4)
    report = lasso_importance_report(df, "target", ["driver", "noise"], max_lag=10)

    assert report["driver"]["best_lag"] == 4
    assert report["driver"]["all_lags_zeroed"] is False
    assert report["noise"]["max_abs_coefficient"] < 0.05 * report["driver"]["max_abs_coefficient"]


def test_load_target_and_candidates_accepts_arbitrary_columns_not_in_columns_x(
    spark, synthetic_candles, tmp_path,
):
    """The whole point of this being a repeatable tool for evaluating FUTURE column
    additions: it must work for a column that was never in split.columns_x, as long
    as Stage 1 produced it -- not just the currently-configured feature set."""
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

    df = load_target_and_candidates(
        spark, str(tmp_path), "EUR/USD", "H1", n_back=10, lookahead=2,
        target_column="volatility_lead",
        candidate_columns=["is_tokyo_session", "week_sin"],  # neither is in any columns_x used so far
    )
    assert {"is_tokyo_session", "week_sin", "volatility_lead"}.issubset(df.columns)
    assert len(df) > 0


def test_analyze_feature_impact_runs_against_real_stage1_output(spark, synthetic_candles, tmp_path):
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

    result = analyze_feature_impact(
        spark, str(tmp_path), "EUR/USD", "H1", n_back=10, lookahead=2,
        target_column="volatility_lead", candidate_columns=["volatility", "return", "day_sin"],
        ccf_max_lag=10, granger_lag=3, var_lag_order=3, var_horizon=5, lasso_max_lag=3,
    )
    assert result["n_observations"] > 0
    assert set(result["cross_correlation"].keys()) == {"volatility", "return", "day_sin"}
    assert set(result["granger_causality"].keys()) == {"volatility", "return", "day_sin"}
    assert set(result["var_fevd"]["fevd_fraction_of_target_variance"].keys()) == {"volatility", "return", "day_sin"}
    assert set(result["lasso"].keys()) == {"volatility", "return", "day_sin", "_alpha_selected"}


def test_parse_cross_pair_candidates_handles_multiple_instruments_and_columns():
    parsed = _parse_cross_pair_candidates("GBP/USD:return,diff_spread_close;USD/JPY:volatility")
    assert parsed == [
        ("GBP/USD", ["return", "diff_spread_close"]),
        ("USD/JPY", ["volatility"]),
    ]


def test_parse_cross_pair_candidates_handles_a_single_instrument_and_column():
    assert _parse_cross_pair_candidates("GBP/USD:return") == [("GBP/USD", ["return"])]


def _make_candles_for(instrument: str, seed: int) -> pd.DataFrame:
    """Same shape as conftest.synthetic_candles, but for an arbitrary instrument/
    seed, and on the SAME timestamp grid -- needed so two different "pairs"'
    Stage-1 output actually has overlapping unix_epoch_s values to inner-join on."""
    n = 300
    rng = np.random.default_rng(seed)
    start = int(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
    timestamps = start + np.arange(n) * 3600

    base_price = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))
    mid_open = base_price
    mid_close = base_price + rng.normal(0, 0.0002, size=n)
    mid_high = np.maximum(mid_open, mid_close) + np.abs(rng.normal(0, 0.0002, size=n))
    mid_low = np.minimum(mid_open, mid_close) - np.abs(rng.normal(0, 0.0002, size=n))
    spread_close = np.abs(rng.normal(0.0001, 0.00002, size=n))
    volume = rng.integers(100, 1000, size=n).astype(float)

    return pd.DataFrame({
        "instrument": instrument,
        "granularity": "H1",
        "unix_epoch_s": timestamps,
        "mid_open": mid_open,
        "mid_high": mid_high,
        "mid_low": mid_low,
        "mid_close": mid_close,
        "spread_close": spread_close,
        "volume": volume,
    })


def test_load_cross_pair_target_and_candidates_joins_two_real_pairs_and_renames_columns(spark, tmp_path):
    params = FeatureParams(
        instruments=["EUR/USD", "GBP/USD"],
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
    engineer_and_save_task(spark, _make_candles_for("EUR/USD", seed=42), "EUR/USD", "H1", params)
    engineer_and_save_task(spark, _make_candles_for("GBP/USD", seed=7), "GBP/USD", "H1", params)

    df = load_cross_pair_target_and_candidates(
        spark, str(tmp_path), "EUR/USD", "H1", n_back=10, lookahead=2,
        target_column="volatility_lead",
        candidate_pairs=[("GBP/USD", ["return", "diff_spread_close"])],
    )

    assert len(df) > 0
    assert "volatility_lead" in df.columns
    assert "GBP_USD__return" in df.columns
    assert "GBP_USD__diff_spread_close" in df.columns
    assert "return" not in df.columns  # not renamed for the TARGET pair, but it wasn't requested at all


def test_analyze_cross_pair_feature_impact_runs_against_two_real_pairs(spark, tmp_path):
    params = FeatureParams(
        instruments=["EUR/USD", "GBP/USD"],
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
    engineer_and_save_task(spark, _make_candles_for("EUR/USD", seed=42), "EUR/USD", "H1", params)
    engineer_and_save_task(spark, _make_candles_for("GBP/USD", seed=7), "GBP/USD", "H1", params)

    result = analyze_cross_pair_feature_impact(
        spark, str(tmp_path), "EUR/USD", "H1", n_back=10, lookahead=2,
        target_column="volatility_lead",
        candidate_pairs=[("GBP/USD", ["return", "volatility"])],
        ccf_max_lag=10, granger_lag=3, var_lag_order=3, var_horizon=5, lasso_max_lag=3,
    )

    expected_columns = {"GBP_USD__return", "GBP_USD__volatility"}
    assert result["n_observations"] > 0
    assert set(result["cross_correlation"].keys()) == expected_columns
    assert set(result["granger_causality"].keys()) == expected_columns
    assert set(result["var_fevd"]["fevd_fraction_of_target_variance"].keys()) == expected_columns
    assert set(result["lasso"].keys()) == expected_columns | {"_alpha_selected"}
