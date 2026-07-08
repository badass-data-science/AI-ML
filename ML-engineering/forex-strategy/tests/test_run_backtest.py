from __future__ import annotations

import pytest
from forex_ml.training.train import train_and_evaluate

from conftest import _make_splits, _train_params
from forex_strategy.run_backtest import backtest_from_mlflow


def test_backtest_from_mlflow_runs_end_to_end_against_a_real_registered_model(trained_triple_barrier_model):
    splits = trained_triple_barrier_model["splits"]
    result = backtest_from_mlflow(trained_triple_barrier_model["tracking_uri"], "EUR/USD", "H1")

    assert result.n_rows == splits.test["M"].shape[0]
    assert 0 <= result.n_trades <= result.n_rows
    assert isinstance(result.net_pnl_pct, float)


def test_backtest_from_mlflow_rejects_pre_migration_pd_lead_models(tmp_path):
    """Regression test: a model version registered before the triple-barrier
    migration (still tagged column_y="pd_lead") must not be picked up --
    find_model_version filters on tags.column_y="triple_barrier", so when only a
    pre-migration version is registered for this pair, none match at all."""
    params = _train_params(tmp_path, "forex-lstm")
    train_and_evaluate(
        _make_splits(seed=0), params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="pd_lead",
        profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3, swap_cost_pct_per_night=0.0,
    )
    with pytest.raises(ValueError, match="No registered version"):
        backtest_from_mlflow(params.mlflow_tracking_uri, "EUR/USD", "H1")


def test_backtest_from_mlflow_raises_for_an_unknown_pair(trained_triple_barrier_model):
    with pytest.raises(ValueError, match="No registered version"):
        backtest_from_mlflow(trained_triple_barrier_model["tracking_uri"], "GBP/USD", "H1")


def test_backtest_from_mlflow_finds_the_triple_barrier_version_even_when_a_later_pd_lead_version_exists(tmp_path):
    """Regression test: filtering find_model_version on tags.column_y="triple_barrier"
    matters beyond convenience -- without it, "most recently registered version for
    this pair" would pick whichever version was registered LAST, regardless of
    scheme. Registers the triple_barrier version FIRST (lower version number) and a
    pre-migration pd_lead version SECOND (higher version number), proving the lookup
    doesn't just take the latest version regardless of column_y."""
    params = _train_params(tmp_path, "forex-lstm")
    train_and_evaluate(
        _make_splits(seed=0), params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="triple_barrier",
        profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3, swap_cost_pct_per_night=0.0,
    )
    train_and_evaluate(
        _make_splits(seed=1), params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="pd_lead",
        profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3, swap_cost_pct_per_night=0.0,
    )

    result = backtest_from_mlflow(params.mlflow_tracking_uri, "EUR/USD", "H1")
    assert isinstance(result.net_pnl_pct, float)  # would have raised the triple_barrier-rejection error otherwise


def test_backtest_from_mlflow_supports_swap_cost_and_flatten_before_rollover(trained_triple_barrier_model):
    result = backtest_from_mlflow(
        trained_triple_barrier_model["tracking_uri"], "EUR/USD", "H1",
        swap_cost_pct_per_night=0.02, flatten_before_rollover=True,
    )
    splits = trained_triple_barrier_model["splits"]
    assert result.n_rows == splits.test["M"].shape[0]
    assert result.n_flattened_for_rollover >= 0


def test_backtest_from_mlflow_uses_realized_volatility_position_sizing(trained_triple_barrier_model):
    fixture = trained_triple_barrier_model
    result = backtest_from_mlflow(
        fixture["tracking_uri"], "EUR/USD", "H1",
        size_by_realized_volatility=True, target_volatility=0.002, max_position_size=1.0,
    )
    assert result.n_rows == fixture["splits"].test["M"].shape[0]
