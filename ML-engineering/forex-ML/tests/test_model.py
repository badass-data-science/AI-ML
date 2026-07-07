from __future__ import annotations

import numpy as np

from forex_ml.config import TrainParams
from forex_ml.training.model import build_lstm_regressor, compile_model, configure_gpu_memory_growth


def _minimal_train_params(**overrides) -> TrainParams:
    base = dict(
        number_of_cells_per_rnn_layer=[8, 8],
        number_of_cells_per_dense_layer=[4],
        lstm_activation_function="relu",
        dense_activation_function="relu",
        final_dense_activation_function="softmax",
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        loss_function="categorical_crossentropy",
        metrics=["accuracy"],
        l1_regularization_constant=0.0001,
        l2_regularization_constant=0.0001,
        batch_normalization_momentum=0.9,
        dense_dropout_rate=0.1,
        rnn_dropout_rate=0.1,
        rnn_recurrent_dropout_rate=0.1,
        reduce_lr_on_plateau_factor=0.9,
        reduce_lr_on_plateau_patience=2,
        early_stopping_patience=2,
        tensorflow_seed=1,
        mlflow_experiment_name="test",
        mlflow_tracking_uri="sqlite:///:memory:",
    )
    base.update(overrides)
    return TrainParams(**base)


def test_model_output_shape_matches_config():
    params = _minimal_train_params()
    model = build_lstm_regressor(params, input_shape=(10, 3), num_outputs=3)
    compile_model(model, params)

    assert model.output_shape == (None, 3)
    # 2 LSTM layers + Flatten + BatchNorm + (Dense + BatchNorm + Dropout) + final Dense
    assert len(model.layers) == 8


def test_model_forward_pass_produces_valid_softmax_output():
    params = _minimal_train_params()
    model = build_lstm_regressor(params, input_shape=(10, 3), num_outputs=3)
    compile_model(model, params)

    X = np.random.default_rng(0).normal(size=(2, 10, 3)).astype("float32")
    preds = model.predict(X, verbose=0)

    assert preds.shape == (2, 3)
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-4)


def test_single_rnn_layer_config_also_builds():
    params = _minimal_train_params(number_of_cells_per_rnn_layer=[4])
    model = build_lstm_regressor(params, input_shape=(5, 2), num_outputs=3)
    compile_model(model, params)
    assert model.output_shape == (None, 3)


def test_configure_gpu_memory_growth_is_safe_to_call_repeatedly():
    """Regression test: on a GPU-equipped machine, memory growth can only be set
    before the GPU context initializes, so every call after the first one in a
    process (once per train_and_evaluate() call -- once per rolling_cv fold, once
    per test in a shared pytest process) used to raise
    RuntimeError("Physical devices cannot be modified after being initialized").
    A no-op on CPU-only machines either way, since the device loop never executes."""
    configure_gpu_memory_growth()
    configure_gpu_memory_growth()
