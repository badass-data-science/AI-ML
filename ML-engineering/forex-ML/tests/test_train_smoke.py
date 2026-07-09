"""End-to-end smoke test: synthetic train/val/test tensors -> 1-epoch LSTM fit ->
evaluate on the held-out test split -> assert MLflow actually recorded the run.

This is the test that would have caught the original validation_split bug: it fails
if `train_and_evaluate` ever stops using splits.val for validation_data, and it fails
if the test-set evaluation step is ever removed.
"""

from __future__ import annotations

import mlflow
import numpy as np

import forex_ml.training.train as train_module
from forex_ml.config import TrainParams
from forex_ml.data.splitting import Splits
from forex_ml.training.model import build_lstm_regressor
from forex_ml.training.train import train_and_evaluate


def _make_splits(n_back: int = 10, n_features: int = 3, n_classes: int = 3) -> Splits:
    rng = np.random.default_rng(0)

    def _one(n: int) -> dict[str, np.ndarray]:
        M = rng.normal(size=(n, n_back, n_features)).astype("float32")
        y_idx = rng.integers(0, n_classes, size=n)
        y = np.eye(n_classes, dtype="float32")[y_idx]
        return {"M": M, "y": y}

    test = _one(10)
    n_test = test["y"].shape[0]
    test["timestamp"] = np.arange(n_test, dtype="float64")
    test["price"] = rng.normal(loc=1.1, scale=0.01, size=n_test).astype("float64")
    test["spread"] = rng.uniform(0.0001, 0.0005, size=n_test).astype("float64")
    test["y_raw"] = rng.normal(size=n_test).astype("float64")
    test["exit_bar_offset"] = rng.integers(1, 4, size=n_test)
    test["realized_volatility"] = rng.uniform(0.0005, 0.005, size=n_test)

    return Splits(train=_one(40), val=_one(10), test=test)


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

    test_results = train_and_evaluate(
        splits, params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="triple_barrier",
        profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=0.0,
    )
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
    assert run.data.params["n_back"] == "10"
    assert run.data.params["lookahead"] == "2"
    assert run.data.params["column_y"] == "triple_barrier"
    assert run.data.params["profit_take_pct"] == "0.5"
    assert run.data.params["stop_loss_pct"] == "0.5"
    assert run.data.params["max_holding_bars"] == "3"
    assert run.data.params["long_swap_cost_pct_per_night"] == "0.0"
    assert run.data.params["short_swap_cost_pct_per_night"] == "0.0"
    assert run.data.tags["recovered_from_batch_checkpoint"] == "False"

    registered = client.search_registered_models(filter_string="name = 'test-experiment'")
    assert len(registered) == 1

    versions = client.search_model_versions("name = 'test-experiment'")
    assert len(versions) == 1
    assert versions[0].tags["instrument"] == "EUR/USD"
    assert versions[0].tags["granularity"] == "H1"
    assert versions[0].tags["config_signature"]
    assert versions[0].tags["column_y"] == "triple_barrier"

    artifact_dir = tmp_path / "downloaded_artifacts"
    predictions_path = next(
        p for p in client.list_artifacts(run.info.run_id) if p.path.endswith("_predictions.npz")
    ).path
    local_path = client.download_artifacts(run.info.run_id, predictions_path, str(artifact_dir))
    predictions = np.load(local_path)
    assert predictions["lstm_pred_proba"].shape == (10, 3)
    np.testing.assert_array_equal(predictions["test_timestamp"], splits.test["timestamp"])
    np.testing.assert_array_equal(predictions["test_price"], splits.test["price"])
    np.testing.assert_array_equal(predictions["test_y_raw"], splits.test["y_raw"])
    np.testing.assert_array_equal(predictions["test_spread"], splits.test["spread"])
    np.testing.assert_array_equal(predictions["test_exit_bar_offset"], splits.test["exit_bar_offset"])
    np.testing.assert_array_equal(predictions["test_realized_volatility"], splits.test["realized_volatility"])


def test_recover_from_divergence_if_needed_no_recovery_when_loss_is_finite(tmp_path):
    """Organically forcing this tiny toy architecture to diverge realistically
    (many good batches, then real NaN) turned out to be unreliable -- a
    pathological learning rate either destabilizes it almost immediately (unlike
    the real 5x300-layer production model, which stayed healthy for most of an
    epoch before diverging) or never crosses from "huge but finite" into literal
    NaN/Inf at this tiny scale. Testing _recover_from_divergence_if_needed
    directly, with a fully controlled fake history dict, is both more reliable
    and a better match for what's actually being tested here: the recovery
    LOGIC, not whether a toy model can be coaxed into failing realistically."""
    model = build_lstm_regressor(
        TrainParams(
            number_of_cells_per_rnn_layer=[2], number_of_cells_per_dense_layer=[2],
            lstm_activation_function="relu", dense_activation_function="relu",
            final_dense_activation_function="softmax", epochs=1, batch_size=4, learning_rate=0.001,
            loss_function="categorical_crossentropy", metrics=["accuracy"],
            l1_regularization_constant=0.0001, l2_regularization_constant=0.0001,
            batch_normalization_momentum=0.9, dense_dropout_rate=0.1, rnn_dropout_rate=0.0,
            rnn_recurrent_dropout_rate=0.0, reduce_lr_on_plateau_factor=0.9,
            reduce_lr_on_plateau_patience=1, early_stopping_patience=1, tensorflow_seed=1,
            mlflow_experiment_name="x", mlflow_tracking_uri="sqlite:///:memory:",
        ),
        (5, 2), 3,
    )
    weights_before = [w.copy() for w in model.get_weights()]

    recovered = train_module._recover_from_divergence_if_needed(
        model, {"loss": [0.9, 0.7], "val_loss": [1.0, 0.8]}, tmp_path / "nonexistent.keras",
    )

    assert recovered is False
    for before, after in zip(weights_before, model.get_weights()):
        np.testing.assert_array_equal(before, after)  # untouched


def test_recover_from_divergence_if_needed_detects_inf_not_just_nan(tmp_path):
    """A diverged run can produce loss: inf with every individual weight still
    finite (observed directly against the real architecture) -- this must trigger
    recovery exactly like NaN does, not just NaN specifically."""
    model = build_lstm_regressor(
        TrainParams(
            number_of_cells_per_rnn_layer=[2], number_of_cells_per_dense_layer=[2],
            lstm_activation_function="relu", dense_activation_function="relu",
            final_dense_activation_function="softmax", epochs=1, batch_size=4, learning_rate=0.001,
            loss_function="categorical_crossentropy", metrics=["accuracy"],
            l1_regularization_constant=0.0001, l2_regularization_constant=0.0001,
            batch_normalization_momentum=0.9, dense_dropout_rate=0.1, rnn_dropout_rate=0.0,
            rnn_recurrent_dropout_rate=0.0, reduce_lr_on_plateau_factor=0.9,
            reduce_lr_on_plateau_patience=1, early_stopping_patience=1, tensorflow_seed=1,
            mlflow_experiment_name="x", mlflow_tracking_uri="sqlite:///:memory:",
        ),
        (5, 2), 3,
    )
    checkpoint_path = tmp_path / "batch_checkpoint.keras"
    model.save(str(checkpoint_path))
    clean_weights = [w.copy() for w in model.get_weights()]
    model.set_weights([w * 0 + 999.0 for w in model.get_weights()])  # simulate a corrupted in-memory model

    recovered = train_module._recover_from_divergence_if_needed(
        model, {"loss": [0.9, float("inf")], "val_loss": [1.0, 0.8]}, checkpoint_path,
    )

    assert recovered is True
    for clean, after in zip(clean_weights, model.get_weights()):
        np.testing.assert_array_equal(clean, after)  # restored from the checkpoint, not left corrupted


def test_recover_from_divergence_if_needed_loads_the_pre_divergence_checkpoint(tmp_path):
    model = build_lstm_regressor(
        TrainParams(
            number_of_cells_per_rnn_layer=[2], number_of_cells_per_dense_layer=[2],
            lstm_activation_function="relu", dense_activation_function="relu",
            final_dense_activation_function="softmax", epochs=1, batch_size=4, learning_rate=0.001,
            loss_function="categorical_crossentropy", metrics=["accuracy"],
            l1_regularization_constant=0.0001, l2_regularization_constant=0.0001,
            batch_normalization_momentum=0.9, dense_dropout_rate=0.1, rnn_dropout_rate=0.0,
            rnn_recurrent_dropout_rate=0.0, reduce_lr_on_plateau_factor=0.9,
            reduce_lr_on_plateau_patience=1, early_stopping_patience=1, tensorflow_seed=1,
            mlflow_experiment_name="x", mlflow_tracking_uri="sqlite:///:memory:",
        ),
        (5, 2), 3,
    )
    checkpoint_path = tmp_path / "batch_checkpoint.keras"
    model.save(str(checkpoint_path))
    clean_weights = [w.copy() for w in model.get_weights()]
    model.set_weights([np.full_like(w, np.nan) for w in model.get_weights()])  # simulate a NaN-corrupted model

    recovered = train_module._recover_from_divergence_if_needed(
        model, {"loss": [float("nan")], "val_loss": [float("nan")]}, checkpoint_path,
    )

    assert recovered is True
    assert not any(np.isnan(w).any() for w in model.get_weights())
    for clean, after in zip(clean_weights, model.get_weights()):
        np.testing.assert_array_equal(clean, after)


def test_recover_from_divergence_if_needed_no_checkpoint_to_recover_from(tmp_path):
    """Divergence within the first _BATCH_CHECKPOINT_FREQ batches means no
    sub-epoch checkpoint was ever saved -- there's nothing to recover, but this
    must not raise; it should just report no recovery happened."""
    model = build_lstm_regressor(
        TrainParams(
            number_of_cells_per_rnn_layer=[2], number_of_cells_per_dense_layer=[2],
            lstm_activation_function="relu", dense_activation_function="relu",
            final_dense_activation_function="softmax", epochs=1, batch_size=4, learning_rate=0.001,
            loss_function="categorical_crossentropy", metrics=["accuracy"],
            l1_regularization_constant=0.0001, l2_regularization_constant=0.0001,
            batch_normalization_momentum=0.9, dense_dropout_rate=0.1, rnn_dropout_rate=0.0,
            rnn_recurrent_dropout_rate=0.0, reduce_lr_on_plateau_factor=0.9,
            reduce_lr_on_plateau_patience=1, early_stopping_patience=1, tensorflow_seed=1,
            mlflow_experiment_name="x", mlflow_tracking_uri="sqlite:///:memory:",
        ),
        (5, 2), 3,
    )

    recovered = train_module._recover_from_divergence_if_needed(
        model, {"loss": [float("nan")], "val_loss": [float("nan")]}, tmp_path / "never_saved.keras",
    )

    assert recovered is False
