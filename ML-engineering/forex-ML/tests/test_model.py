from __future__ import annotations

import numpy as np

from forex_ml.config import TrainParams
from forex_ml.training.model import ClipInputs, build_lstm_regressor, compile_model, configure_gpu_memory_growth


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
    # ClipInputs + 2 LSTM layers + Flatten + BatchNorm + (Dense + BatchNorm + Dropout) + final Dense
    assert len(model.layers) == 9


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


def test_clip_inputs_clamps_extreme_values_and_passes_normal_ones_through():
    """Regression test for the real bug this guards against: a >10-sigma input value
    (real and common in this data -- see the fat-tailed return/volatility columns)
    compounding through many unrolled LSTM timesteps overflowed to Inf/NaN in the
    forward pass, which gradient clipping (a backward-pass-only fix) couldn't
    prevent. clip_value=5 here for a clean, easy-to-check boundary."""
    layer = ClipInputs(clip_value=5.0)
    x = np.array([[-19.7, -5.0, -1.0, 0.0, 1.0, 5.0, 19.7]], dtype="float32")
    out = layer(x).numpy()
    np.testing.assert_allclose(out, [[-5.0, -5.0, -1.0, 0.0, 1.0, 5.0, 5.0]])


def test_clip_inputs_survives_save_and_load_round_trip(tmp_path):
    """The whole point of a real Layer subclass instead of a bare Lambda: it must
    serialize/deserialize correctly through keras.models.load_model(), since
    ModelCheckpoint saves and later reloads this exact architecture in
    train_and_evaluate()."""
    import keras

    params = _minimal_train_params(input_clip_value=3.0)
    model = build_lstm_regressor(params, input_shape=(5, 2), num_outputs=3)
    compile_model(model, params)

    path = tmp_path / "model.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)

    clip_layer = reloaded.layers[0]
    assert isinstance(clip_layer, ClipInputs)
    assert clip_layer.clip_value == 3.0


def test_configure_gpu_memory_growth_is_safe_to_call_repeatedly():
    """Regression test: on a GPU-equipped machine, memory growth can only be set
    before the GPU context initializes, so every call after the first one in a
    process (once per train_and_evaluate() call -- once per rolling_cv fold, once
    per test in a shared pytest process) used to raise
    RuntimeError("Physical devices cannot be modified after being initialized").
    A no-op on CPU-only machines either way, since the device loop never executes."""
    configure_gpu_memory_growth()
    configure_gpu_memory_growth()
