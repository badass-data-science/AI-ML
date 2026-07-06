
#
# Load useful libraries
#
import os
import pickle
import uuid
import json

import tensorflow
from tensorflow.random import set_seed

from keras import layers
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from numba import cuda 

#
# get a unique ID
#
uid = str(uuid.uuid4())

print()
print(uid)
print()

#
# user settings
#
config = {

    'filename_data_pickled' : '/home/emily/Desktop/projects/cc_test/AI-ML/ML-engineering/forex-ML/output/data.pickled',

    'use_variable_learning_rate' : True,
    
    'use_batch_normalization_layers' : True,
    'use_dropout_layers' : True,
    
    'tensorflow_seed' : 54,

    'number_of_cells_per_RNN_layer_list' : [300, 300, 300, 300, 300],
    'number_of_cells_per_dense_layer_list' : [7],

    'lstm_activation_function' : 'relu',
    'dense_activation_function' : 'relu',
    'final_dense_activation_function' : 'softmax',
    
    'epochs' : 2,
    'batch_size' : 16,

    'learning_rate' : 0.0001,

    'loss_function' : 'categorical_crossentropy',
    'metrics_to_store' : ['accuracy'],

    'model_checkpoint_monitor' : 'val_loss',
    'model_checkpoint_save_best_only' : True,

    'validation_split' : 0.2,
    
    'l1_regularization_constant' : 0.0001,
    'l2_regularization_constant' : 0.0001,

    'batch_normalization_momentum' : 0.9,
    
    'dense_dropout_rate' : 0.4,
    'rnn_dropout_rate' : 0.4,
    'rnn_recurrent_dropout_rate' : 0.4,

    
    'callbacks_dict' : {
        'ReduceLROnPlateau' : {
            'monitor' : 'val_loss',
            'factor' : 0.9,
            'patience' : 3,
        }
    },

    'json_config_output_path' : 'output/models/' + uid + '_regressor_config.json',
    'checkpoint_file_path' : 'output/models/' + uid + '_regressor_model_checkpoints.keras',
    'model_json_path' : 'output/models/' + uid + '_model_regressor.json',
    'model_final_weights_path' : 'output/models/' + uid + '_final_weights_regressor.weights.h5',
    'model_final_history_path' : 'output/models/' + uid + '_final_history_regressor.pickled',
}

#
# Reset device
#
device = cuda.get_current_device()
device.reset()

#
# set seeds
#
set_seed(config['tensorflow_seed'])

#
# Load data
#
with open(config['filename_data_pickled'], 'rb') as f:
    train_val_test_dict = pickle.load(f)

    #QA
    print(train_val_test_dict['train']['M'].shape)
    print(train_val_test_dict['train']['y'].shape)

M = train_val_test_dict['train']['M']
y = train_val_test_dict['train']['y']

#
# calculate input and output matrix/array shapes
#
config['calculated_input_shape'] = (M.shape[1], M.shape[2])
config['calculated_number_of_outputs'] = y.shape[1]

#
# save configuration
#
with open(config['json_config_output_path'], 'w') as f:
    json.dump(config, f, indent = 2)

#
# build a generic Keras LSTM regressor
#
def build_generic_LSTM_regressor(**config):
    model = Sequential()

    #
    # build RNN layers (this will always produce at least one, optionally more)
    #
    for i, n_units_in_layer in enumerate(config['number_of_cells_per_RNN_layer_list']):

        if i == 0:

            #
            # define input layer
            #
            model.add(
                layers.LSTM(
                    n_units_in_layer,
                    activation = config['lstm_activation_function'],
                    return_sequences = True,
                    input_shape = config['calculated_input_shape'],
                    dropout = config['rnn_dropout_rate'],
                    recurrent_dropout = config['rnn_recurrent_dropout_rate'],
                )
            )

        else:
            model.add(
                layers.LSTM(
                    n_units_in_layer,
                    activation = config['lstm_activation_function'],
                    return_sequences = True,
                    dropout = config['rnn_dropout_rate'],
                    recurrent_dropout = config['rnn_recurrent_dropout_rate'],
                )
            )

    #
    # flatten
    #
    model.add(layers.Flatten())

    #
    # first batch normalization
    #
    model.add(layers.BatchNormalization(momentum = config['batch_normalization_momentum']))
    
    #
    # Build dense layers
    #
    for n_units_in_layer in config['number_of_cells_per_dense_layer_list']:

        model.add(
            layers.Dense(
                n_units_in_layer,
                activation = config['dense_activation_function'],

                # https://keras.io/api/layers/regularizers/
                kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.L2(1e-4),
                activity_regularizer=regularizers.L2(1e-5),
            )
        )

        model.add(layers.BatchNormalization(momentum = config['batch_normalization_momentum']))

        model.add(
            layers.Dropout(
                rate = config['dense_dropout_rate'],
            )
        )

    #
    # define output layer
    #
    model.add(
        layers.Dense(
            config['calculated_number_of_outputs'],
            activation = config['final_dense_activation_function'],

            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
            bias_regularizer=regularizers.L2(1e-4),
            activity_regularizer=regularizers.L2(1e-5),
        )
    )

    return model

#
# compile the generic Keras LSTM regressor given above
#
def compile_generic_regressor(model, **config): #loss = 'mse', metrics = ['mse']):
    model.compile(
        optimizer = Adam(learning_rate = config['learning_rate']),
        loss = config['loss_function'],
        metrics = config['metrics_to_store'],
        )

#
# fit the generic Keras LSTM classifer given above
#
def fit_generic_regressor(model, train_X, train_y, **config):

    #
    # set callbacks list
    #
    callbacks_list = [
        ReduceLROnPlateau(
            monitor = config['callbacks_dict']['ReduceLROnPlateau']['monitor'],
            factor = config['callbacks_dict']['ReduceLROnPlateau']['factor'],
            patience = config['callbacks_dict']['ReduceLROnPlateau']['patience'],
        ),
        ModelCheckpoint(
            filepath = config['checkpoint_file_path'],
            monitor = config['model_checkpoint_monitor'],
            save_best_only = config['model_checkpoint_save_best_only'],
        ),
    ]

    print()
    print(train_X.shape)
    print(train_y.shape)
    print()

    if config['use_variable_learning_rate']:
        history = model.fit(
            train_X,
            train_y,

            #validation_data = (val_X, val_y),
            validation_split = config['validation_split'],

            epochs = config['epochs'],
            batch_size = config['batch_size'],
            callbacks = callbacks_list,
        )
    else:
        history = model.fit(
            train_X,
            train_y,

            #validation_data = (val_X, val_y),
            validation_split = config['validation_split'],

            epochs = config['epochs'],
            batch_size = config['batch_size'],
        )
        

    return history

#
# build model
#
model = build_generic_LSTM_regressor(**config)

#
# save model to JSON
#
model_json = model.to_json()
f = open(config['model_json_path'], 'w')
f.write(model_json)
f.close()

#
# compile model
#
compile_generic_regressor(
    model,
    **config,
)

#
# fit model
#
history = fit_generic_regressor(
    model,
    M,
    y,
    **config,
)

#
# save final weights
#
model.save_weights(config['model_final_weights_path'])

#
# save history
#
with open(config['model_final_history_path'], 'wb') as f:
    pickle.dump(history.history, f)


    
#
# display unique ID
#
print()
print(uid)
print()
