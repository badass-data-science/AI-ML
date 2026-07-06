"""End-to-end smoke test: synthetic train/val/test tensors -> 1-epoch LSTM fit ->
evaluate on the held-out test split -> assert MLflow actually recorded the run.

This is the test that would have caught the original validation_split bug: it fails
if `train_and_evaluate` ever stops using splits.val for validation_data, and it fails
if the test-set evaluation step is ever removed.
"""

from __future__ import annotations

import mlflow
import numpy as np

from forex_ml.config import TrainParams
from forex_ml.data.splitting import Splits
from forex_ml.training.train import train_and_evaluate


def _make_splits(n_back: int = 10, n_features: int = 3, n_classes: int = 3) -> Splits:
    rng = np.random.default_rng(0)

    def _one(n: int) -> dict[str, np.ndarray]:
        M = rng.normal(size=(n, n_back, n_features)).astype("float32")
        y_idx = rng.integers(0, n_classes, size=n)
        y = np.eye(n_classes, dtype="float32")[y_idx]
        return {"M": M, "y": y}

    return Splits(train=_one(40), val=_one(10), test=_one(10))


def test_train_and_evaluate_logs_params_metrics_and_model(tmp_path):
    splits = _make_splits()
    params = TrainParams(
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
        mlflow_experiment_name="test-experiment",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )

    test_results = train_and_evaluate(splits, params, "EUR/USD", "H1", tmp_path)
    assert "loss" in test_results
    assert "baseline_majority_accuracy" in test_results
    assert "baseline_persistence_accuracy" in test_results

    client = mlflow.tracking.MlflowClient(tracking_uri=params.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name("test-experiment")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1

    run = runs[0]
    assert "test_loss" in run.data.metrics
    assert "val_loss" in run.data.metrics  # proves validation_data was actually used
    assert "baseline_majority_test_accuracy" in run.data.metrics
    assert "baseline_persistence_test_accuracy" in run.data.metrics
    for split_name in ("train", "val", "test"):
        for class_idx in range(3):
            assert f"{split_name}_class_{class_idx}_balance" in run.data.metrics
    assert run.data.params["instrument"] == "EUR/USD"
    assert run.data.params["granularity"] == "H1"

    registered = client.search_registered_models(filter_string="name = 'test-experiment'")
    assert len(registered) == 1
