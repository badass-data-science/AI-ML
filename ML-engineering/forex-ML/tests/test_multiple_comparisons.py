from __future__ import annotations

import time

import mlflow
import numpy as np
import pytest

from forex_ml.config import TrainParams
from forex_ml.data.splitting import Splits
from forex_ml.evaluation.multiple_comparisons import (
    benjamini_hochberg_report,
    mcnemar_p_value,
    report_across_pairs,
)
from forex_ml.training.train import train_and_evaluate


def test_mcnemar_identical_predictions_give_p_value_of_one():
    rng = np.random.default_rng(0)
    correct = rng.random(200) > 0.4  # both models get exactly the same rows right
    assert mcnemar_p_value(correct, correct) == pytest.approx(1.0)


def test_mcnemar_clearly_better_model_gives_small_p_value():
    rng = np.random.default_rng(0)
    n = 200
    # model_a right on 80% of rows, model_b right on 20%, mostly disagreeing
    model_a_correct = rng.random(n) < 0.8
    model_b_correct = rng.random(n) < 0.2
    p = mcnemar_p_value(model_a_correct, model_b_correct)
    assert p < 0.01


def test_mcnemar_requires_aligned_lengths():
    with pytest.raises(ValueError, match="aligned"):
        mcnemar_p_value(np.array([True, False]), np.array([True, False, True]))


def test_benjamini_hochberg_filters_a_borderline_result_that_naive_alpha_would_pass():
    """The whole point of this feature: with 14 pairs tested at raw alpha=0.05, a
    borderline p=0.04 pair would look "significant" under naive per-pair comparison
    even though, corrected for testing 14 pairs at once, it isn't. The two genuinely
    small p-values remain significant either way."""
    p_values = {
        "real_signal_1": 0.0001,
        "real_signal_2": 0.0005,
        "borderline_noise": 0.04,
        **{f"null_{i}": p for i, p in enumerate([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80])},
    }
    assert len(p_values) == 14

    naive_significant = {pair for pair, p in p_values.items() if p < 0.05}
    assert naive_significant == {"real_signal_1", "real_signal_2", "borderline_noise"}

    report = benjamini_hochberg_report(p_values, alpha=0.05)
    corrected_significant = {pair for pair, r in report.items() if r["significant_after_correction"]}
    assert corrected_significant == {"real_signal_1", "real_signal_2"}
    assert report["borderline_noise"]["significant_after_correction"] is False
    assert report["borderline_noise"]["p_adjusted"] > report["borderline_noise"]["p_value"]


def _make_splits(seed: int, n_back: int = 10, n_features: int = 3, n_classes: int = 3) -> Splits:
    rng = np.random.default_rng(seed)

    def _one(n: int) -> dict[str, np.ndarray]:
        M = rng.normal(size=(n, n_back, n_features)).astype("float32")
        y_idx = rng.integers(0, n_classes, size=n)
        y = np.eye(n_classes, dtype="float32")[y_idx]
        return {"M": M, "y": y}

    return Splits(train=_one(40), val=_one(10), test=_one(10))


def test_report_across_pairs_finds_both_pairs_end_to_end(tmp_path):
    """Trains two tiny synthetic "pairs" into the same MLflow store, then proves
    report_across_pairs can pull both runs back, load their predictions.npz
    artifacts, and produce a BH-corrected report — the full real path, not mocked."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
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
        mlflow_experiment_name="cross-pair-test",
        mlflow_tracking_uri=tracking_uri,
    )

    train_and_evaluate(_make_splits(0), params, "EUR/USD", "H1", tmp_path)
    train_and_evaluate(_make_splits(1), params, "AUD/USD", "H1", tmp_path)

    report = report_across_pairs(tracking_uri, "cross-pair-test", baseline="majority")

    assert set(report.keys()) == {"EUR/USD_H1", "AUD/USD_H1"}
    for result in report.values():
        assert 0.0 <= result["p_value"] <= 1.0
        assert 0.0 <= result["p_adjusted"] <= 1.0


def _log_manual_run(
    tracking_uri: str,
    experiment_name: str,
    instrument: str,
    granularity: str,
    lstm_correct: np.ndarray,
    majority_correct: np.ndarray,
    artifact_path,
) -> None:
    """Logs a run with hand-picked predictions.npz content, bypassing real training
    entirely -- for tests that need deterministic control over which run "wins"
    rather than depending on what a tiny 1-epoch model happens to learn."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({"instrument": instrument, "granularity": granularity})
        np.savez_compressed(
            artifact_path,
            lstm_correct=lstm_correct,
            majority_correct=majority_correct,
            persistence_correct=majority_correct[1:],
        )
        mlflow.log_artifact(str(artifact_path))


def test_report_across_pairs_uses_only_the_most_recent_run_for_a_retrained_pair(tmp_path):
    """Simulates retraining the same pair over time (expected when building up data
    incrementally on a single local GPU rather than training all pairs at once): the
    second run must win, not be silently overwritten by iteration order."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    n = 200
    rng = np.random.default_rng(0)
    majority_correct = rng.random(n) > 0.5

    # first (older) attempt: LSTM just agrees with the baseline everywhere -> no
    # discordant pairs -> p-value == 1.0 (not significant)
    _log_manual_run(
        tracking_uri, "retrain-test", "EUR/USD", "H1",
        lstm_correct=majority_correct, majority_correct=majority_correct,
        artifact_path=tmp_path / "old_predictions.npz",
    )
    time.sleep(0.05)  # ensure a distinct, later start_time for the "retrained" run

    # second (newer) attempt: LSTM right almost everywhere the baseline is wrong,
    # and right everywhere the baseline is right too -> heavily one-sided discordant
    # pairs -> very small p-value (clearly significant)
    lstm_correct_improved = majority_correct | (rng.random(n) > 0.05)
    _log_manual_run(
        tracking_uri, "retrain-test", "EUR/USD", "H1",
        lstm_correct=lstm_correct_improved, majority_correct=majority_correct,
        artifact_path=tmp_path / "new_predictions.npz",
    )

    report = report_across_pairs(tracking_uri, "retrain-test", baseline="majority")

    assert set(report.keys()) == {"EUR/USD_H1"}  # one pair, not two runs double-counted
    assert report["EUR/USD_H1"]["p_value"] < 0.01  # reflects the newer, improved run
    assert report["EUR/USD_H1"]["significant_after_correction"] is True
