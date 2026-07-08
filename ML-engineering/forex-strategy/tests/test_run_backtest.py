from __future__ import annotations

import pytest
from forex_ml.training.train import train_and_evaluate

from conftest import _make_splits, _train_params
from forex_strategy.run_backtest import backtest_from_mlflow


def test_backtest_from_mlflow_runs_end_to_end_against_a_real_registered_model(trained_pd_lead_model):
    splits = trained_pd_lead_model["splits"]
    result = backtest_from_mlflow(trained_pd_lead_model["tracking_uri"], "EUR/USD", "H1")

    assert result.n_rows == splits.test["M"].shape[0]
    assert 0 <= result.n_trades <= result.n_rows
    assert isinstance(result.net_pnl_pct, float)


def test_backtest_from_mlflow_rejects_a_non_directional_target(trained_volatility_lead_model):
    # find_model_version filters on tags.column_y="pd_lead" -- when only a
    # volatility_lead version is registered for this pair, no version matches at all.
    with pytest.raises(ValueError, match="pd_lead"):
        backtest_from_mlflow(trained_volatility_lead_model["tracking_uri"], "EUR/USD", "H1")


def test_backtest_from_mlflow_raises_for_an_unknown_pair(trained_pd_lead_model):
    with pytest.raises(ValueError, match="No registered version"):
        backtest_from_mlflow(trained_pd_lead_model["tracking_uri"], "GBP/USD", "H1")


def test_backtest_from_mlflow_finds_the_pd_lead_version_even_when_a_later_volatility_version_exists(tmp_path):
    """Regression test: filtering find_model_version on tags.column_y="pd_lead"
    matters beyond convenience -- without it, "most recently registered version for
    this pair" would pick whichever target was trained LAST, regardless of which one
    the caller actually wants. Registers volatility_lead first (lower version
    number) then pd_lead second (higher version number) -- if the filter weren't
    applied, the higher version number would still happen to be the right one here,
    so instead this registers pd_lead FIRST (lower) and volatility_lead SECOND
    (higher), proving the lookup doesn't just take the latest version regardless of
    target."""
    params = _train_params(tmp_path, "forex-lstm")
    train_and_evaluate(
        _make_splits(seed=0), params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="pd_lead",
    )
    train_and_evaluate(
        _make_splits(seed=1), params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="volatility_lead",
    )

    result = backtest_from_mlflow(params.mlflow_tracking_uri, "EUR/USD", "H1")
    assert isinstance(result.net_pnl_pct, float)  # would have raised the pd_lead-rejection error otherwise


def test_backtest_from_mlflow_supports_swap_cost_and_flatten_before_rollover(trained_pd_lead_model):
    result = backtest_from_mlflow(
        trained_pd_lead_model["tracking_uri"], "EUR/USD", "H1",
        swap_cost_pct_per_night=0.02, flatten_before_rollover=True,
    )
    splits = trained_pd_lead_model["splits"]
    assert result.n_rows == splits.test["M"].shape[0]
    assert result.n_flattened_for_rollover >= 0


def test_backtest_from_mlflow_uses_volatility_gated_position_sizing(trained_pd_lead_and_volatility_models):
    fixture = trained_pd_lead_and_volatility_models
    result = backtest_from_mlflow(fixture["tracking_uri"], "EUR/USD", "H1", use_volatility_sizing=True)
    assert result.n_rows == fixture["pd_lead_splits"].test["M"].shape[0]


def test_backtest_from_mlflow_rejects_misaligned_volatility_model(tmp_path):
    params = _train_params(tmp_path, "forex-lstm")
    pd_lead_splits = _make_splits(seed=0)
    volatility_splits = _make_splits(seed=1)
    volatility_splits.test["timestamp"] = volatility_splits.test["timestamp"] + 999_999  # deliberately misaligned

    train_and_evaluate(
        pd_lead_splits, params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="pd_lead",
    )
    train_and_evaluate(
        volatility_splits, params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="volatility_lead",
    )

    with pytest.raises(ValueError, match="not row-aligned"):
        backtest_from_mlflow(params.mlflow_tracking_uri, "EUR/USD", "H1", use_volatility_sizing=True)
