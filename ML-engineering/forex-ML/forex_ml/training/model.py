"""LSTM model architecture and compilation.

Ported from lstm.py's build_generic_LSTM_regressor / compile_generic_regressor,
parameterized by forex_ml.config.TrainParams instead of a raw **config dict.
"""

from __future__ import annotations

import tensorflow as tf
from keras import Sequential, layers, regularizers
from keras.optimizers import Adam

from forex_ml.config import TrainParams


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

    for i, n_units in enumerate(params.number_of_cells_per_rnn_layer):
        if i == 0:
            model.add(
                layers.LSTM(
                    n_units,
                    activation=params.lstm_activation_function,
                    return_sequences=True,
                    input_shape=input_shape,
                    dropout=params.rnn_dropout_rate,
                    recurrent_dropout=params.rnn_recurrent_dropout_rate,
                )
            )
        else:
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
