from __future__ import annotations

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from forex_strategy.model_registry import find_model_version, load_keras_model, load_test_predictions


def test_find_model_version_locates_the_tagged_version(trained_triple_barrier_model):
    client = MlflowClient(tracking_uri=trained_triple_barrier_model["tracking_uri"])
    resolved = find_model_version(client, "forex-lstm", "EUR/USD", "H1")
    assert resolved.instrument == "EUR/USD"
    assert resolved.granularity == "H1"
    assert resolved.config_signature


def test_find_model_version_raises_for_an_unknown_pair(trained_triple_barrier_model):
    client = MlflowClient(tracking_uri=trained_triple_barrier_model["tracking_uri"])
    with pytest.raises(ValueError, match="No registered version"):
        find_model_version(client, "forex-lstm", "GBP/USD", "H1")


def test_load_keras_model_returns_a_working_model(trained_triple_barrier_model):
    client = MlflowClient(tracking_uri=trained_triple_barrier_model["tracking_uri"])
    resolved = find_model_version(client, "forex-lstm", "EUR/USD", "H1")
    model = load_keras_model(resolved)

    splits = trained_triple_barrier_model["splits"]
    pred = model.predict(splits.test["M"], verbose=0)
    assert pred.shape == (splits.test["M"].shape[0], 3)


def test_load_test_predictions_returns_the_expected_arrays(trained_triple_barrier_model, tmp_path):
    client = MlflowClient(tracking_uri=trained_triple_barrier_model["tracking_uri"])
    resolved = find_model_version(client, "forex-lstm", "EUR/USD", "H1")
    predictions = load_test_predictions(client, resolved.run_id, str(tmp_path / "downloaded"))

    splits = trained_triple_barrier_model["splits"]
    assert predictions["lstm_pred_proba"].shape == (splits.test["M"].shape[0], 3)
    np.testing.assert_array_equal(predictions["test_y_raw"], splits.test["y_raw"])
    np.testing.assert_array_equal(predictions["test_price"], splits.test["price"])
    np.testing.assert_array_equal(predictions["test_spread"], splits.test["spread"])
    np.testing.assert_array_equal(predictions["test_exit_bar_offset"], splits.test["exit_bar_offset"])
    np.testing.assert_array_equal(predictions["test_realized_volatility"], splits.test["realized_volatility"])
