from __future__ import annotations

import time

import mlflow
import numpy as np
import pytest

from forex_ml.config import TrainParams
from forex_ml.data.splitting import Splits
from forex_ml.evaluation.multiple_comparisons import (
    benjamini_hochberg_report,
    config_signature_from_params,
    mcnemar_p_value,
    report_across_pairs,
)
from forex_ml.training.train import train_and_evaluate


def test_config_signature_ignores_long_and_short_swap_cost_pct_per_night():
    """long/short_swap_cost_pct_per_night are now resolved from a live InfluxDB
    snapshot (see forex_ml.data.swap_rates), not hyperparameters a human chose --
    two runs of the SAME hyperparameters on different days would otherwise get
    different signatures purely because OANDA's rate ticked in between, silently
    fracturing the "same configuration, retrained" grouping this signature exists
    to support."""
    base = {
        "n_back": 200, "lookahead": 4, "learning_rate": 0.0001,
        "long_swap_cost_pct_per_night": 0.00679, "short_swap_cost_pct_per_night": -0.00234,
    }
    drifted = {**base, "long_swap_cost_pct_per_night": -0.01234, "short_swap_cost_pct_per_night": 0.00987}

    assert config_signature_from_params(base) == config_signature_from_params(drifted)


def test_config_signature_ignores_the_old_pre_bidirectional_swap_cost_key_too():
    """Real historical MLflow runs were logged with the old, single
    swap_cost_pct_per_night key before the bidirectional redesign -- it must stay
    excluded permanently so re-running this against old runs doesn't silently
    change their config signatures."""
    base = {"n_back": 200, "lookahead": 4, "learning_rate": 0.0001, "swap_cost_pct_per_night": 0.00679}
    drifted = {**base, "swap_cost_pct_per_night": -0.01234}

    assert config_signature_from_params(base) == config_signature_from_params(drifted)


def test_config_signature_still_distinguishes_real_hyperparameter_changes():
    base = {
        "n_back": 200, "lookahead": 4, "learning_rate": 0.0001,
        "long_swap_cost_pct_per_night": 0.0, "short_swap_cost_pct_per_night": 0.0,
    }
    different_lr = {**base, "learning_rate": 0.001}

    assert config_signature_from_params(base) != config_signature_from_params(different_lr)


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

    test = _one(10)
    n_test = test["y"].shape[0]
    test["timestamp"] = np.arange(n_test, dtype="float64")
    test["price"] = rng.normal(loc=1.1, scale=0.01, size=n_test).astype("float64")
    test["spread"] = rng.uniform(0.0001, 0.0005, size=n_test).astype("float64")
    test["y_raw"] = rng.normal(size=n_test).astype("float64")
    test["exit_bar_offset"] = rng.integers(1, 4, size=n_test)
    test["realized_volatility"] = rng.uniform(0.0005, 0.005, size=n_test)

    return Splits(train=_one(40), val=_one(10), test=test)


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

    train_and_evaluate(
        _make_splits(0), params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="triple_barrier",
        profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=0.0,
    )
    train_and_evaluate(
        _make_splits(1), params, "AUD/USD", "H1", tmp_path, n_back=10, lookahead=2, column_y="triple_barrier",
        profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=0.0,
    )

    report = report_across_pairs(tracking_uri, "cross-pair-test", baseline="majority")

    assert {(r["instrument"], r["granularity"]) for r in report.values()} == {
        ("EUR/USD", "H1"), ("AUD/USD", "H1"),
    }
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
    extra_params: dict | None = None,
    persistence_correct: np.ndarray | None = None,
    persistence_scored: np.ndarray | None = None,
) -> None:
    """Logs a run with hand-picked predictions.npz content, bypassing real training
    entirely -- for tests that need deterministic control over which run "wins"
    rather than depending on what a tiny 1-epoch model happens to learn.

    `extra_params` simulates other TrainParams (architecture, epochs, etc.) being
    logged alongside instrument/granularity -- used to distinguish "same
    configuration, retrained" from "different configuration, same pair" in
    _model_config_signature.

    `persistence_correct`/`persistence_scored` default to full-length arrays
    (matching lstm_correct/majority_correct's length, all rows scored) so tests
    that only exercise baseline="majority" don't need to care about persistence's
    contract at all -- pass them explicitly for tests targeting baseline="persistence".
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    if persistence_correct is None:
        persistence_correct = majority_correct
    if persistence_scored is None:
        persistence_scored = np.ones_like(majority_correct, dtype=bool)
    with mlflow.start_run():
        mlflow.log_params({
            "instrument": instrument, "granularity": granularity, **(extra_params or {}),
        })
        np.savez_compressed(
            artifact_path,
            lstm_correct=lstm_correct,
            majority_correct=majority_correct,
            persistence_correct=persistence_correct,
            persistence_scored=persistence_scored,
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

    assert len(report) == 1  # one (pair, configuration), not two runs double-counted
    result = next(iter(report.values()))
    assert (result["instrument"], result["granularity"]) == ("EUR/USD", "H1")
    assert result["p_value"] < 0.01  # reflects the newer, improved run
    assert result["significant_after_correction"] is True


def test_report_across_pairs_treats_different_configurations_as_separate_hypotheses(tmp_path):
    """Architecture search on the SAME pair (different number_of_cells_per_rnn_layer,
    say) must NOT collapse to "most recent run" the way a same-configuration retrain
    does -- each configuration tried is its own hypothesis and needs its own slot in
    the BH correction, or picking whichever architecture wins would silently escape
    correction entirely."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    n = 200
    rng = np.random.default_rng(0)
    majority_correct = rng.random(n) > 0.5
    lstm_correct = majority_correct | (rng.random(n) > 0.05)  # clearly beats baseline

    _log_manual_run(
        tracking_uri, "arch-search-test", "EUR/USD", "H1",
        lstm_correct=lstm_correct, majority_correct=majority_correct,
        artifact_path=tmp_path / "arch_a_predictions.npz",
        extra_params={"number_of_cells_per_rnn_layer": "[32]"},
    )
    time.sleep(0.05)
    _log_manual_run(
        tracking_uri, "arch-search-test", "EUR/USD", "H1",
        lstm_correct=lstm_correct, majority_correct=majority_correct,
        artifact_path=tmp_path / "arch_b_predictions.npz",
        extra_params={"number_of_cells_per_rnn_layer": "[64, 32]"},
    )

    report = report_across_pairs(tracking_uri, "arch-search-test", baseline="majority")

    assert len(report) == 2  # both configurations kept, not collapsed to the latest
    assert all((r["instrument"], r["granularity"]) == ("EUR/USD", "H1") for r in report.values())


def test_report_across_pairs_treats_different_n_back_as_separate_hypotheses(tmp_path):
    """Regression test: n_back/lookahead are FeatureParams, not TrainParams, so they
    used to never get logged to MLflow at all -- two runs differing ONLY in n_back
    logged an IDENTICAL set of params and collapsed into "the same configuration,
    just retrained," silently discarding one of them. train_and_evaluate now logs
    n_back/lookahead explicitly so this can't happen, even though the TrainParams
    below are byte-for-byte identical between the two calls."""
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
        mlflow_experiment_name="n-back-test",
        mlflow_tracking_uri=tracking_uri,
    )

    train_and_evaluate(
        _make_splits(0, n_back=10), params, "EUR/USD", "H1", tmp_path, n_back=10, lookahead=2,
        column_y="triple_barrier", profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=0.0,
    )
    train_and_evaluate(
        _make_splits(1, n_back=24), params, "EUR/USD", "H1", tmp_path, n_back=24, lookahead=2,
        column_y="triple_barrier", profit_take_pct=0.5, stop_loss_pct=0.5, max_holding_bars=3,
        long_swap_cost_pct_per_night=0.0, short_swap_cost_pct_per_night=0.0,
    )

    report = report_across_pairs(tracking_uri, "n-back-test", baseline="majority")

    assert len(report) == 2  # both n_back values kept, not collapsed to the latest


def test_report_across_pairs_persistence_baseline_excludes_unscored_rows_from_mcnemar(tmp_path):
    """report_across_pairs must mask BOTH lstm_correct and persistence_correct by
    persistence_scored before McNemar's test -- not just slice off a fixed number
    of rows. Constructed so the first 20 (unscored) rows are maximally discordant
    (lstm always right, persistence always wrong) -- if these leaked into the
    test, the p-value would come out very small. The remaining 180 (scored) rows
    have lstm and persistence agreeing on every single row (zero discordant
    pairs), so the correctly-masked result must be p_value == 1.0."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    n = 200
    scored = np.array([False] * 20 + [True] * 180)
    lstm_correct = np.array([True] * 20 + [True] * 90 + [False] * 90)
    persistence_correct = np.array([False] * 20 + [True] * 90 + [False] * 90)
    assert len(lstm_correct) == len(persistence_correct) == len(scored) == n

    _log_manual_run(
        tracking_uri, "persistence-mask-test", "EUR/USD", "H1",
        lstm_correct=lstm_correct, majority_correct=lstm_correct,  # unused by this test
        artifact_path=tmp_path / "run_predictions.npz",
        persistence_correct=persistence_correct, persistence_scored=scored,
    )

    report = report_across_pairs(tracking_uri, "persistence-mask-test", baseline="persistence")

    assert len(report) == 1
    result = next(iter(report.values()))
    assert result["p_value"] == pytest.approx(1.0)


def test_report_across_pairs_skips_runs_missing_persistence_scored(tmp_path):
    """A predictions.npz logged before the persistence_baseline causal-validity fix
    has no persistence_scored key (and an old, one-row-shorter persistence_correct
    convention) -- report_across_pairs must skip that run gracefully (like it
    already does for a run with no predictions artifact at all) rather than raise
    a bare KeyError."""
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    n = 50
    rng = np.random.default_rng(0)
    lstm_correct = rng.random(n) > 0.5

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("pre-fix-artifact-test")
    artifact_path = tmp_path / "old_predictions.npz"
    with mlflow.start_run():
        mlflow.log_params({"instrument": "EUR/USD", "granularity": "H1"})
        np.savez_compressed(
            artifact_path,
            lstm_correct=lstm_correct,
            majority_correct=lstm_correct,
            persistence_correct=lstm_correct[1:],  # old N-1 convention, no scored mask
        )
        mlflow.log_artifact(str(artifact_path))

    report = report_across_pairs(tracking_uri, "pre-fix-artifact-test", baseline="persistence")

    assert report == {}  # the only run present predates the fix -- nothing to report, no crash
