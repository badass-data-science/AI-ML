from __future__ import annotations

import pytest

from forex_strategy.run_backtest import backtest_from_mlflow


def test_backtest_from_mlflow_runs_end_to_end_against_a_real_registered_model(trained_pd_lead_model):
    splits = trained_pd_lead_model["splits"]
    result = backtest_from_mlflow(trained_pd_lead_model["tracking_uri"], "EUR/USD", "H1")

    assert result.n_rows == splits.test["M"].shape[0]
    assert 0 <= result.n_trades <= result.n_rows
    assert isinstance(result.net_pnl_pct, float)


def test_backtest_from_mlflow_rejects_a_non_directional_target(trained_volatility_lead_model):
    with pytest.raises(ValueError, match="pd_lead"):
        backtest_from_mlflow(trained_volatility_lead_model["tracking_uri"], "EUR/USD", "H1")


def test_backtest_from_mlflow_raises_for_an_unknown_pair(trained_pd_lead_model):
    with pytest.raises(ValueError, match="No registered version"):
        backtest_from_mlflow(trained_pd_lead_model["tracking_uri"], "GBP/USD", "H1")
