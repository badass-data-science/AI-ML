"""LSTM model architecture and compilation.

Ported from lstm.py's build_generic_LSTM_regressor / compile_generic_regressor,
parameterized by forex_ml.config.TrainParams instead of a raw **config dict.
"""

from __future__ import annotations

import tensorflow as tf
from keras import Sequential, layers, regularizers
from keras.optimizers import Adam
from keras.saving import register_keras_serializable

from forex_ml.config import TrainParams


@register_keras_serializable(package="forex_ml")
class ClipInputs(layers.Layer):
    """Clips inputs to [-clip_value, clip_value] before the first LSTM layer.

    Gradient clipping (see compile_model's clipnorm) only bounds the BACKWARD pass --
    it can't help if the FORWARD pass itself produces Inf/NaN, which is exactly what
    happens when a single extreme input value compounds through many unrolled LSTM
    timesteps across several stacked layers. This isn't hypothetical here: fat-tailed
    features like `return`/`volatility` have thousands of >10-sigma values in the
    real training data, and a real n_back=200 training run went to NaN mid-epoch
    (once at epoch 1, again at epoch 13 even with clipnorm already in place) before
    this was added. Clipping the input directly removes the mechanism that causes
    that, independent of and complementary to gradient clipping -- not a replacement
    for it, since clipnorm still bounds how aggressively the model can react to a
    clipped-but-still-large input.

    A proper Layer subclass (not a bare Lambda) so it serializes/deserializes
    correctly through keras.models.load_model() -- ModelCheckpoint saves and later
    reloads this exact architecture.
    """

    def __init__(self, clip_value: float, **kwargs):
        super().__init__(**kwargs)
        self.clip_value = clip_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, -self.clip_value, self.clip_value)

    def get_config(self):
        config = super().get_config()
        config["clip_value"] = self.clip_value
        return config


def configure_gpu_memory_growth() -> None:
    """Replaces the original `from numba import cuda; device.reset()` hack with the
    standard TensorFlow idiom for not pre-allocating all GPU memory. A no-op on
    CPU-only machines instead of raising, unlike the numba.cuda call it replaces.

    Memory growth can only be set before the GPU context initializes -- this function
    is called at the top of every train_and_evaluate() call (once per fold in
    rolling_cv, once per test in a shared pytest process, etc.), so every call after
    the first one in a process hits an already-initialized context and TF raises
    RuntimeError. That's expected, not a failure: the first call already took effect,
    there's nothing left to configure, so it's caught and ignored here.
    """
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def build_lstm_regressor(params: TrainParams, input_shape: tuple[int, int], num_outputs: int) -> Sequential:
    model = Sequential()
    model.add(ClipInputs(params.input_clip_value, input_shape=input_shape))

    for n_units in params.number_of_cells_per_rnn_layer:
        model.add(
            layers.LSTM(
                n_units,
                activation=params.lstm_activation_function,
                return_sequences=True,
                dropout=params.rnn_dropout_rate,
                recurrent_dropout=params.rnn_recurrent_dropout_rate,
            )
        )

    model.add(layers.Flatten())
    model.add(layers.BatchNormalization(momentum=params.batch_normalization_momentum))

    for n_units in params.number_of_cells_per_dense_layer:
        model.add(
            layers.Dense(
                n_units,
                activation=params.dense_activation_function,
                kernel_regularizer=regularizers.L1L2(l1=params.l1_regularization_constant, l2=params.l2_regularization_constant),
                bias_regularizer=regularizers.L2(params.l2_regularization_constant),
                activity_regularizer=regularizers.L2(params.l1_regularization_constant),
            )
        )
        model.add(layers.BatchNormalization(momentum=params.batch_normalization_momentum))
        model.add(layers.Dropout(rate=params.dense_dropout_rate))

    model.add(
        layers.Dense(
            num_outputs,
            activation=params.final_dense_activation_function,
            kernel_regularizer=regularizers.L1L2(l1=params.l1_regularization_constant, l2=params.l2_regularization_constant),
            bias_regularizer=regularizers.L2(params.l2_regularization_constant),
            activity_regularizer=regularizers.L2(params.l1_regularization_constant),
        )
    )
    return model


def compile_model(model: Sequential, params: TrainParams) -> None:
    # clipnorm bounds the gradient norm per step -- without it, deep backprop-through-
    # time (5 stacked LSTM layers x a 200-bar n_back is a lot of unrolled depth) can
    # explode the loss to NaN partway through the very first epoch. This was always
    # missing (confirmed absent in the original lstm.py too, via
    # compile_generic_regressor), it just never got exercised until training was
    # actually run at n_back=200 against real full-history data -- shorter windows in
    # tests/earlier real runs weren't deep enough to trigger it.
    model.compile(
        optimizer=Adam(learning_rate=params.learning_rate, clipnorm=params.gradient_clip_norm),
        loss=params.loss_function,
        metrics=list(params.metrics),
    )
