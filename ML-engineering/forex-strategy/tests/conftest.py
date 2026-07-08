from __future__ import annotations

import numpy as np
import pytest
from forex_ml.config import TrainParams
from forex_ml.data.splitting import Splits
from forex_ml.training.train import train_and_evaluate


def _make_splits(n_back: int = 10, n_features: int = 3, n_classes: int = 3, seed: int = 0) -> Splits:
    rng = np.random.default_rng(seed)

    def _one(n: int) -> dict[str, np.ndarray]:
        M = rng.normal(size=(n, n_back, n_features)).astype("float32")
        y_idx = rng.integers(0, n_classes, size=n)
        y = np.eye(n_classes, dtype="float32")[y_idx]
        return {"M": M, "y": y}

    test = _one(20)
    n_test = test["y"].shape[0]
    test["timestamp"] = np.arange(n_test, dtype="float64") * 3600
    test["price"] = rng.normal(loc=1.10, scale=0.01, size=n_test).astype("float64")
    test["spread"] = rng.uniform(0.0001, 0.0003, size=n_test).astype("float64")
    test["y_raw"] = rng.normal(scale=0.2, size=n_test).astype("float64")

    return Splits(train=_one(60), val=_one(20), test=test)


def _train_params(tmp_path, experiment_name: str) -> TrainParams:
    return TrainParams(
        number_of_cells_per_rnn_layer=[4],
        number_of_cells_per_dense_layer=[4],
        lstm_activation_function="relu",
        dense_activation_function="relu",
        final_dense_activation_function="softmax",
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        loss_function="categorical_crossentropy",
        metrics=["accuracy"],
        l1_regularization_constant=0.0001,
        l2_regularization_constant=0.0001,
        batch_normalization_momentum=0.9,
        dense_dropout_rate=0.1,
        rnn_dropout_rate=0.0,
        rnn_recurrent_dropout_rate=0.0,
        reduce_lr_on_plateau_factor=0.9,
        reduce_lr_on_plateau_patience=1,
        early_stopping_patience=1,
        tensorflow_seed=1,
        mlflow_experiment_name=experiment_name,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


@pytest.fixture
def trained_pd_lead_model(tmp_path):
    """Trains and registers a tiny real pd_lead model into a scratch MLflow store --
    the same pattern forex-ML's own test_train_smoke.py uses -- so forex-strategy's
    model-loading/backtest code can be exercised against a real registered model and
    a real predictions.npz artifact, not a hand-mocked stand-in."""
    params = _train_params(tmp_path, "forex-lstm")
    splits = _make_splits(seed=0)
    train_and_evaluate(splits, params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="pd_lead")
    return {"tracking_uri": params.mlflow_tracking_uri, "splits": splits}


@pytest.fixture
def trained_volatility_lead_model(tmp_path):
    """Same as trained_pd_lead_model, but trained on volatility_lead -- used to check
    that the backtest correctly refuses to run against a non-directional target."""
    params = _train_params(tmp_path, "forex-lstm")
    splits = _make_splits(seed=1)
    train_and_evaluate(splits, params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="volatility_lead")
    return {"tracking_uri": params.mlflow_tracking_uri, "splits": splits}


@pytest.fixture
def trained_pd_lead_and_volatility_models(tmp_path):
    """Trains and registers BOTH a pd_lead and a volatility_lead model for the same
    pair into the SAME scratch MLflow store -- for exercising the volatility-gated
    position-sizing path in run_backtest.py, which needs to look up and combine two
    distinct registered versions. _make_splits' test["timestamp"] doesn't depend on
    `seed`, so both models' test sets are already row-aligned by timestamp, exactly
    as the real pipeline guarantees when both are trained with the same
    n_back/lookahead/split configuration."""
    params = _train_params(tmp_path, "forex-lstm")
    pd_lead_splits = _make_splits(seed=0)
    volatility_splits = _make_splits(seed=1)
    train_and_evaluate(
        pd_lead_splits, params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="pd_lead",
    )
    train_and_evaluate(
        volatility_splits, params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="volatility_lead",
    )
    return {"tracking_uri": params.mlflow_tracking_uri, "pd_lead_splits": pd_lead_splits}
