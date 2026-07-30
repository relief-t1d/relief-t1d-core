"""
Neural network architectures for continuous glucose prediction.
"""

from tensorflow import keras
from keras.layers import Dense, LSTM, Input, Conv1D, MaxPooling1D, Flatten
from keras.models import Model


# ==========================================
# 1. ARCHITECTURE BUILDERS
# ==========================================

def build_linear_model(history_length: int, nb_output_units: int = 1) -> Model:
    """
    Builds a simple Linear (Dense) regression model.
    Effectively acts as a flattened perceptron mapping history to horizon.
    """
    i = Input(shape=(history_length,))
    x = Flatten()(i)
    x = Dense(nb_output_units, activation=None)(x)

    return Model(inputs=[i], outputs=[x], name="Linear_Model")


def build_lstm_model(history_length: int,
                     nb_output_units: int = 1,
                     nb_hidden_units: int = 128,
                     dropout_rate: float = 0.1,
                     recurrent_dropout_rate: float = 0.0) -> Model:
    """
    Builds a Long Short-Term Memory (LSTM) recurrent neural network.
    Designed to capture temporal dependencies in sequential glucose data.
    """
    # Note: LSTMs require 3D input shape (batch_size, time_steps, features)
    i = Input(shape=(history_length, 1))
    x = LSTM(nb_hidden_units,
             dropout=dropout_rate,
             recurrent_dropout=recurrent_dropout_rate)(i)
    x = Dense(nb_output_units, activation=None)(x)

    return Model(inputs=[i], outputs=[x], name=f"LSTM_{nb_hidden_units}_Units")


def build_cnn_model(history_length: int, nb_output_units: int = 1) -> Model:
    """
    Builds a 1D Convolutional Neural Network (1D-CNN).
    Useful for extracting local shape features (trends, peaks) from the glucose curve.
    """
    i = Input(shape=(history_length, 1))

    x = Conv1D(filters=16, kernel_size=2, activation='relu', padding='same')(i)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Conv1D(filters=32, kernel_size=2, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(units=50, activation='relu')(x)
    # Using 'linear' activation for the final regression output
    x = Dense(units=nb_output_units, activation='linear')(x)

    return Model(inputs=[i], outputs=[x], name="CNN_1D_Model")


# ==========================================
# 2. COMPILATION & DISPATCHER
# ==========================================

def get_compiled_model(algo_name: str,
                       history_length: int,
                       loss_function,
                       metrics_list: list,
                       learning_rate: float = 0.001) -> Model:
    """
    Dispatcher function to fetch, build and compile a model by its string name.

    Args:
        algo_name: Name of the algorithm ('Linear', 'LSTM', 'CNN').
        history_length: Number of past time steps used as input.
        loss_function: The custom or standard Keras loss function.
        metrics_list: List of Keras metrics to track during training.
        learning_rate: Optimizer learning rate.

    Returns:
        A compiled tf.keras.Model ready for training.
    """
    # 1. Build the requested architecture
    if algo_name == 'Linear':
        model = build_linear_model(history_length)
    elif algo_name == 'LSTM':
        model = build_lstm_model(history_length)
    elif algo_name == 'CNN':
        model = build_cnn_model(history_length)
    else:
        raise ValueError(f"Unknown algorithm name: '{algo_name}'. Supported: 'Linear', 'LSTM', 'CNN'.")

    # 2. Compile the model with the requested loss and metrics
    model.compile(
        loss=loss_function,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=metrics_list
    )

    return model