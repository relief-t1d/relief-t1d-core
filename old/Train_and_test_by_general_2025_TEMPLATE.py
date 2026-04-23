# ----------------------- Imports -----------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from keras.layers import Dense, LSTM, GRU, Lambda, dot, concatenate, Activation, Input
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow import keras
import numpy as np
import tensorflow as tf
import random
import os
import sys
import pandas as pd
from pandas import DataFrame
import time
import matplotlib.pyplot as plt
import datetime
import platform
import socket
import psutil
import subprocess
# ----------------------- End imports -----------------------

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

###################### CONFIG EXPERIMENT #############################
batch_size = 4096
patience = 10
max_epoch = 2
dropout = 0.0  # Dropout for LSTM layers
recurrent_dropout = 0.0  # Dropout rate for LSTM layers

# Path to the windows data (must include final slash)
windows_path = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/folds/folds_deduplicated_intra_patient_balanced_splits_DiaTrend/'  # e.g., 'C:/users/user1/windows/'

history_length = 8  # 2h

do_cross_validation = True  # If True, it will use the cross-validation folds in the dataset

horizons = [2]
algorithms = ['Linear', 'LSTM', 'CNN']

# Sensor reading limits
maximum_sensor_reading = 500
minimum_sensor_reading = 40

###################### END CONFIG EXPERIMENT #############################

# List of folder names
folders = ["hist", "models", "plots", "test_results"]

# Create each folder if it does not exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            if f not in (sys.stdout, sys.stderr):
                f.close()


# Open a log file to write the results
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = open(f"{time_str}_train_log.out", "a", encoding="utf-8")

# Redirect stdout and stderr
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# ----------------------- Print the system information -----------------------

def log_experiment_info():
    print("📅 Date & Time")
    print("-----------------------------------")
    print(f"Experiment Start : {datetime.datetime.now()}")

    print("/n🖥️  System & Hardware")
    print("-----------------------------------")
    print(f"Hostname         : {socket.gethostname()}")
    print(f"OS               : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture     : {platform.machine()}")
    print(f"CPU              : {platform.processor()}")
    print(f"CPU Cores        : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical")

    ram = psutil.virtual_memory()
    print(f"RAM Total        : {round(ram.total / (1024**3), 2)} GB")
    print(f"RAM Available    : {round(ram.available / (1024**3), 2)} GB")
    print(f"RAM Used         : {round(ram.used / (1024**3), 2)} GB")
    print(f"RAM Usage (%)    : {ram.percent}%")

    print("/n🐍 Python & Libraries")
    print("-----------------------------------")
    print(f"Python Version   : {platform.python_version()}")
    print(f"Executable Path  : {sys.executable}")
    print(f"TensorFlow       : {tf.__version__}")

    print("/n🧠 GPU (via TensorFlow)")
    print("-----------------------------------")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}           : {gpu.name}")
    else:
        print("No GPUs detected by TensorFlow.")

    print("/n🎮 GPU Details (via nvidia-smi)")
    print("-----------------------------------")
    try:
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader"
        ])
        for i, line in enumerate(result.decode().strip().split('/n')):
            name, mem_total, mem_used, usage, driver = map(str.strip, line.split(','))
            print(f"GPU {i}           : {name}")
            print(f"  Total Memory   : {mem_total}")
            print(f"  Used Memory    : {mem_used}")
            print(f"  Utilization    : {usage}")
            print(f"  Driver Version : {driver}")
    except FileNotFoundError:
        print("⚠️  'nvidia-smi' not found or NVIDIA driver not installed.")
    except Exception as e:
        print(f"⚠️  Error running nvidia-smi: {e}")


log_experiment_info()

print("-----------------------------------")
print()

# ----------------------- End print the system information -----------------------

try:
    ## Algorithms architectures ################################################
    
    # LSTM
    class LSTMModel:
        def __init__(self,
                     input_shape,
                     nb_output_units,
                     nb_hidden_units=128,
                     dropout_rate=dropout,  # <<< NEW parameter
                     recurrent_dropout_rate=recurrent_dropout):  # <<< NEW parameter
            self.input_shape = input_shape
            self.nb_output_units = nb_output_units
            self.nb_hidden_units = nb_hidden_units
            self.dropout = dropout_rate  # <<< SAVE
            self.recurrent_dropout = recurrent_dropout_rate  # <<< SAVE

        def __repr__(self):
            #  Simplificamos para reflejar los nuevos campos
            return (f"LSTM_{self.nb_hidden_units}_h"
                    f"_drop{self.dropout}_recDrop{self.recurrent_dropout}")  # <<< MOD

        def build(self):
            i = Input(shape=self.input_shape)

            # LSTM con dropout en entradas y (opcional) en recurrencia
            x = LSTM(
                self.nb_hidden_units,
                dropout=self.dropout,  # <<< ADD
                recurrent_dropout=self.recurrent_dropout  # <<< ADD
            )(i)

            x = Dense(self.nb_output_units, activation=None)(x)
            return Model(inputs=[i], outputs=[x])

    # Linear
    class LinearModel:
        def __init__(self, input_shape, nb_output_units):
            self.input_shape = input_shape
            self.nb_output_units = nb_output_units

        def __repr__(self):
            return 'Linear'

        def build(self):
            i = Input(shape=self.input_shape)
            x = Dense(self.nb_output_units, activation=None)(i)

            return Model(inputs=[i], outputs=[x])

    # CNN
    class CNNModel:
        def __init__(self, input_shape, nb_output_units):
            self.input_shape = input_shape
            self.nb_output_units = nb_output_units

        def __repr__(self):
            return 'CNN_{0}_units'.format(self.nb_hidden_units)

        def build(self):
            i = Input(shape=self.input_shape)
            x = Conv1D(filters=16, kernel_size=2, activation='relu')(i)
            x = MaxPooling1D(pool_size=2)(x)
            x = Conv1D(filters=32, kernel_size=2, activation='relu')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)
            x = Dense(units=50, activation='relu')(x)
            x = Dense(units=self.nb_output_units, activation='linear')(x)

            return Model(inputs=[i], outputs=[x])

    ## Model Functions ################################################
    def RMSE(output, target):
        output = tf.cast(output, 'float32')
        target = tf.cast(target, 'float32')

        return tf.sqrt(tf.reduce_mean((output - target) ** 2))

    def build_model(model,
                    weights=''):
        # build & compile model
        m = model.build()

        m.compile(loss=RMSE,
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanSquaredError(), RMSE])  # Use RMSE as a metric
        if weights:
            print(f"Weights: {weights}")
            m.load_weights(weights)
        return m


    def callbacks(filepath,
                  early_stopping_patience):
        callbacks = []
        callbacks.append(ModelCheckpoint(filepath=filepath + ".weights.h5",
                                         monitor='val_loss',
                                         mode='min',
                                         save_best_only=True,
                                         save_weights_only=True,
                                         verbose=1))
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       patience=early_stopping_patience,
                                       restore_best_weights=True,  # Restore the best weights after early stopping
                                       verbose=1))
        return callbacks

    def prepare_model_LSTM(history_length,
                           nb_hidden_units=128,
                           weights=''):
        model = LSTMModel(input_shape=(history_length, 1), nb_output_units=1, nb_hidden_units=nb_hidden_units)
        return build_model(model, weights)


    def prepare_model_linear(history_length,
                             weights=''):
        model = LinearModel(input_shape=(history_length,), nb_output_units=1)
        return build_model(model, weights)


    def prepare_model_CNN(history_length, weights=''):
        model = CNNModel(input_shape=(history_length, 1), nb_output_units=1)
        return build_model(model, weights)

    ############## TRAIN FUNCTION #####################################
    def train(x_train,
              y_train,
              x_val,
              y_val,
              model,
              history_length,
              save_filepath="",
              early_stopping_patience=patience):
        x_train = np.reshape(x_train, (x_train.shape[0], history_length, 1))

        x_val = np.reshape(x_val, (x_val.shape[0], history_length, 1))
        validation_data = (x_val, y_val)

        hist = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         validation_data=validation_data,
                         epochs=max_epoch,
                         # shuffle=True,
                         callbacks=callbacks(save_filepath,
                                             early_stopping_patience)
                         )

        return hist, model

    def get_train_plots_loss(hist: keras.callbacks, name: str):
        fig, ax = plt.subplots()

        # data
        x_epoch = hist.epoch
        y_val_loss = hist.history['val_loss']
        y_train_loss = hist.history['loss']

        # Create a line plots
        ax.plot(x_epoch, y_val_loss, label='Validation loss', color='orange', linestyle='-')
        ax.plot(x_epoch, y_train_loss, label='Train loss', color='blue', linestyle='-')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and validation loss')

        ax.legend()

        fig.savefig(name + '_Loss.pdf', dpi=350, bbox_inches='tight')

    def get_train_plots_RMSE(hist: keras.callbacks, name: str):
        fig, ax = plt.subplots()

        # Sample data
        x_epoch = hist.epoch
        y_val_rmse = hist.history['val_root_mean_squared_error']
        y_train_rmse = hist.history['root_mean_squared_error']

        # Create a line plot
        ax.plot(x_epoch, y_val_rmse, label='Validation RMSE', color='orange', linestyle='-')
        ax.plot(x_epoch, y_train_rmse, label='Train RMSE', color='blue', linestyle='-')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Training and validation RMSE')

        # Add a legend
        ax.legend()

        # Display the plot
        fig.savefig(name + '_RMSE.pdf', dpi=350, bbox_inches='tight')

    def get_train_plots_MSE(hist: keras.callbacks, name: str):
        fig, ax = plt.subplots()

        # Sample data
        x_epoch = hist.epoch
        y_val_mse = hist.history['val_mean_squared_error']
        y_train_mse = hist.history['mean_squared_error']

        # Create a line plot
        ax.plot(x_epoch, y_val_mse, label='Validation MSE', color='orange', linestyle='-')
        ax.plot(x_epoch, y_train_mse, label='Train MSE', color='blue', linestyle='-')

        # Add labels and title
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Training and validation MSE')

        # Add a legend
        ax.legend()

        # Display the plot
        fig.savefig(name + '_MSE.pdf', dpi=350, bbox_inches='tight')

    # Results    ################################################

    ## Functions ################################################
    def clarke_error_grid(ref_values, pred_values, title_string, show_plot=False):
        """
          This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
         of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
        """
        # Checking to see if the lengths of the reference and prediction arrays are the same
        assert (len(ref_values) == len(
            pred_values)), "Unequal number of values (reference : {0}) (prediction : {1}).".format(len(ref_values),
                                                                                                   len(pred_values))

        # Checks to see if the values are within the normal sensor measurement range, otherwise it gives a warning
        # Reference values
        if min(ref_values) < minimum_sensor_reading:
            print(f'Input Warning: the minimum reference value ({min(ref_values):.2f}) is below the limit of sensor design (Inferior limit: {minimum_sensor_reading} mg/dL).')
            number_ref_values_low_limit = sum(value < minimum_sensor_reading for value in ref_values)
            print(f"Number of reference values below the limit: {number_ref_values_low_limit}")
            print()

        if max(ref_values) > maximum_sensor_reading:
            print(f'Input Warning: the maximum reference value ({max(ref_values):.2f}) is above the limit of sensor design (Superior limit: {maximum_sensor_reading} mg/dL).')
            number_ref_values_up_limit = sum(value > maximum_sensor_reading for value in ref_values)
            print(f"Number of reference values above the limit: {number_ref_values_up_limit}")
            print()

        # Predicted values
        min_ref_value = min(pred_values)
        min_ref_value = round(min_ref_value, 2)
        number_pred_values_low_limit = 0
        if min_ref_value < minimum_sensor_reading:
            print(f'Input Warning: the minimum predicted value ({min_ref_value:.2f}) is below the minimum input value (Inferior limit: {minimum_sensor_reading} mg/dL).')
            number_pred_values_low_limit = sum(value < minimum_sensor_reading for value in pred_values)
            # list_of_values_low_limit = [value for value in pred_values if value < minimum_sensor_reading]
            print(f"Number of predicted values below the limit: {number_pred_values_low_limit}")
            print()

        max_ref_value = max(pred_values)
        max_ref_value = round(max_ref_value, 2)
        number_pred_values_up_limit = 0
        if max_ref_value > maximum_sensor_reading:
            print(f'Input Warning: the maximum predicted value ({max_ref_value:.2f}) is above the maximum input value (Superior limit: {maximum_sensor_reading} mg/dL).')
            number_pred_values_up_limit = sum(value > maximum_sensor_reading for value in pred_values)
            # list_of_values_up_limit = [value for value in pred_values if value > maximum_sensor_reading]
            print(f"Number of predicted values above the limit: {number_pred_values_up_limit}")
            print()

        # list of out of range values
        out_range_values = [min_ref_value, number_pred_values_low_limit, max_ref_value, number_pred_values_up_limit]

        if show_plot:
            # #Clear plot
            plt.clf()

            # Set up plot
            plt.scatter(ref_values, pred_values, marker='o', color='darkgray', s=8)
            # plt.title(title_string + " Clarke Error Grid")
            plt.xlabel("Reference Concentration (mg/dl)")
            plt.ylabel("model_prediction_scripts Concentration (mg/dl)")
            plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
            plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
            plt.gca().set_facecolor('white')

            # Set axes lengths
            plt.gca().set_xlim([0, 500])
            plt.gca().set_ylim([0, 500])
            plt.gca().set_aspect((500) / (500))

            # Plot zone lines
            plt.plot([0, 500], [0, 500], ':', c='black', linewidth=1)  # Theoretical 45 regression line
            plt.plot([0, 175 / 3], [70, 70], '-', c='black', linewidth=1)
            # plt.plot([175/3, 320], [70, 500], '-', c='black', linewidth=1)
            plt.plot([175 / 3, 500 / 1.2], [70, 500], '-',
                     c='black',
                     linewidth=1)  # Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
            plt.plot([70, 70], [84, 500], '-', c='black', linewidth=1)
            plt.plot([0, 70], [180, 180], '-', c='black', linewidth=1)
            plt.plot([70, 290], [180, 500], '-', c='black', linewidth=1)
            # plt.plot([70, 70], [0, 175/3], '-', c='black', linewidth=1)
            plt.plot([70, 70], [0, 56], '-', c='black',
                     linewidth=1)  # Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
            # plt.plot([70, 500],[175/3, 320],'-', c='black', linewidth=1)

            plt.plot([70, 500], [56, 400.79], '-', c='black', linewidth=1)  ##########################

            plt.plot([180, 180], [0, 70], '-', c='black', linewidth=1)
            plt.plot([180, 500], [70, 70], '-', c='black', linewidth=1)
            plt.plot([240, 240], [70, 180], '-', c='black', linewidth=1)
            plt.plot([240, 500], [180, 180], '-', c='black', linewidth=1)
            plt.plot([130, 180], [0, 70], '-', c='black', linewidth=1)

            # Add zone titles
            plt.text(30, 15, "A", fontsize=15, fontweight='bold')
            plt.text(400, 250, "B", fontsize=15, fontweight='bold')
            plt.text(270, 400, "B", fontsize=15, fontweight='bold')
            plt.text(150, 400, "C", fontsize=15, fontweight='bold')
            plt.text(155, 15, "C", fontsize=15, fontweight='bold')
            plt.text(30, 120, "D", fontsize=15, fontweight='bold')
            plt.text(370, 120, "D", fontsize=15, fontweight='bold')
            plt.text(30, 370, "E", fontsize=15, fontweight='bold')
            plt.text(370, 15, "E", fontsize=15, fontweight='bold')

            plt.savefig(f'plots/Clarke_Error_Grid{title_string}.png', dpi=350, bbox_inches='tight')

        # Statistics from the data
        zone = [0] * 5
        for i in range(len(ref_values)):
            if (ref_values[i] <= 70 and pred_values[i] <= 70) or (
                    pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[i]):
                zone[0] += 1  # Zone A

            elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
                zone[4] += 1  # Zone E

            elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or (
                    (ref_values[i] >= 130 and ref_values[i] <= 180) and (
                    pred_values[i] <= (7 / 5) * ref_values[i] - 182)):
                zone[2] += 1  # Zone C
            elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (
                    ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[i] >= 70) or (
                    (ref_values[i] >= 175 / 3 and ref_values[i] <= 70) and pred_values[i] >= (6 / 5) * ref_values[i]):
                zone[3] += 1  # Zone D
            else:
                zone[1] += 1  # Zone B

        return zone, out_range_values

    def get_values_per_zone(values):
        keys = ['A', 'B', 'C', 'D', 'E']
        return {key: value for key, value in zip(keys, values)}

    def test_by_range(df_result_vector: pd.DataFrame, grid_name: str) -> tuple[DataFrame, DataFrame, DataFrame]:
        df_result_vector_TBR_2 = df_result_vector[df_result_vector['y_test'] < 54]

        df_result_vector_TBR_1 = df_result_vector[
            (df_result_vector['y_test'] >= 54) & (df_result_vector['y_test'] < 70)]

        df_result_vector_TIR = df_result_vector[(df_result_vector['y_test'] >= 70) & (df_result_vector['y_test'] < 181)]

        df_result_vector_TAR_1 = df_result_vector[
            (df_result_vector['y_test'] >= 181) & (df_result_vector['y_test'] < 251)]

        df_result_vector_TAR_2 = df_result_vector[(df_result_vector['y_test'] >= 251)]

        list_dataframe_by_range = [df_result_vector,
                                   df_result_vector_TBR_2, df_result_vector_TBR_1,
                                   df_result_vector_TIR,
                                   df_result_vector_TAR_1, df_result_vector_TAR_2]
        list_dataframe_by_range_name = ['ENTIRE', 'TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']

        df_metrics_summary = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E', 'A + B', 'RMSE', 'MSE', 'MAE', 'MAPE'])
        df_zones_values = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E'])
        df_limits = pd.DataFrame(columns=['Range', 'Minimum value', 'Number values below limit', 'Maximum value', 'Number values above limit'])

        for i in range(6):
            reference_values = list_dataframe_by_range[i]['y_test']
            pred_values = list_dataframe_by_range[i]['y_predict']
            print(f'------- Zone: {list_dataframe_by_range_name[i]} -------')
            if list_dataframe_by_range_name[i] == 'ENTIRE':  # Plot the graph for the ENTIRE zone
                zone, out_of_range_values = clarke_error_grid(ref_values=reference_values.values,
                                                              pred_values=pred_values.values,
                                                              title_string=grid_name,
                                                              show_plot=True)
            else:  # Do not plot the graph for the other zones
                zone, out_of_range_values = clarke_error_grid(ref_values=reference_values.values,
                                                              pred_values=pred_values.values,
                                                              title_string=grid_name + list_dataframe_by_range_name[i],
                                                              show_plot=False)

            # Clinical metrics
            print(f'Zones: [A, B, C, D, E]')
            print(f'Number of points by zone: {zone}')
            zone_percentages = round(pd.Series(zone) / reference_values.shape[0] * 100, 2)

            new_row_zone_values = {'Range': list_dataframe_by_range_name[i],
                                   'A': zone[0],
                                   'B': zone[1],
                                   'C': zone[2],
                                   'D': zone[3],
                                   'E': zone[4]}
            new_index_zone_values = len(df_zones_values)
            df_zones_values.loc[new_index_zone_values] = new_row_zone_values

            print(f'Percentage of point by zone: {zone_percentages.tolist()}')
            print()

            # Non-clinical metrics
            mse = mean_squared_error(reference_values, pred_values)
            rmse = np.sqrt(mean_squared_error(reference_values, pred_values))
            mae = mean_absolute_error(reference_values, pred_values)
            mape = mean_absolute_percentage_error(reference_values, pred_values)

            print(f'Mean Squared Error (MSE): {mse:.2f}')
            print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
            print(f'Mean absolute Error (MAE): {mae:.2f}')
            print(f'Mean absolute percentage Error (MAPE): {mape:.2f}')

            new_row_metrics = {'Range': list_dataframe_by_range_name[i],
                               'A': zone_percentages[0],
                               'B': zone_percentages[1],
                               'C': zone_percentages[2],
                               'D': zone_percentages[3],
                               'E': zone_percentages[4],
                               'A + B': zone_percentages[0] + zone_percentages[1],
                               'RMSE': rmse,
                               'MSE': mse,
                               'MAE': mae,
                               'MAPE': mape}
            new_index_metrics = len(df_metrics_summary)
            df_metrics_summary.loc[new_index_metrics] = new_row_metrics

            # Save the limits of the values
            new_row_limits = {'Range': list_dataframe_by_range_name[i],
                              'Minimum value': out_of_range_values[0],
                              'Number values below limit': out_of_range_values[1],
                              'Maximum value': out_of_range_values[2],
                              'Number values above limit': out_of_range_values[3]}
            new_index_limits = len(df_limits)
            df_limits.loc[new_index_limits] = new_row_limits

            print()
            print('-' * 100)
            print()

        return df_metrics_summary, df_zones_values, df_limits
        
    #################################################### TRAIN AND TEST ##########################################################

    for current_horizon in horizons:  # For all horizons -------------------------------------
        
        used_columns = [f"x{i}" for i in range(history_length)] + ["y"]  # Columns to be used for training, validation, and testing

        if do_cross_validation:
            k_folds = 5
            print(f"Cross-validation enabled with {k_folds} folds.")

        else:
            k_folds = 1
            print("Cross-validation is disabled. Using the first fold for training, validation, and testing.")

        results_vector_list = []

        for current_algorithm in algorithms:  # For all models ---------------------------------------

            if current_algorithm == 'Linear':  # prepare linear model
                model = prepare_model_linear(history_length)

            elif current_algorithm == 'LSTM':  # prepare LSTM model
                model = prepare_model_LSTM(history_length)

            else:  # prepare CNN model
                model = prepare_model_CNN(history_length)

            for current_fold in range(0, k_folds):  # For all folds -------------------------------------
                current_name = f"{current_algorithm}_H{current_horizon}_Fold{current_fold + 1}"

                df = pd.read_parquet(f"{windows_path}windows_with_folds_horizon_{current_horizon}.parquet")

                # Filter the DataFrame for training for the current fold
                df_train = df[df[f'fold_{current_fold}'] == 'train']
                patient_number_in_train = df_train['patient_id'].nunique()
                df_train.reset_index(drop=True, inplace=True)
                df_train = df_train[used_columns]  # Select only the used columns

                # Filter the DataFrame for validation for the current fold
                df_val = df[df[f'fold_{current_fold}'] == 'val']
                patient_number_in_val = df_val['patient_id'].nunique()
                df_val.reset_index(drop=True, inplace=True)
                df_val = df_val[used_columns]  # Select only the used columns

                # Filter the DataFrame for testing for the current fold
                df_test = df[df[f'fold_{current_fold}'] == 'test']
                patient_number_in_test = df_test['patient_id'].nunique()
                df_test.reset_index(drop=True, inplace=True)
                df_test = df_test[used_columns]  # Select only the used columns

                num_patients = patient_number_in_train + patient_number_in_val + patient_number_in_test

                print(f"\n---- Fold {current_fold + 1}: -----")
                print("-- Totals --")
                print(f"  Train: {patient_number_in_train} patients. Total windows: {len(df_train)}")
                print(f"  Val:   {patient_number_in_val} patients. Total windows: {len(df_val)}")
                print(f"  Test:  {patient_number_in_test} patients. Total windows: {len(df_test)}")

                print("\n -- Percentages --")
                print(f"Train: {(patient_number_in_train / num_patients * 100):.2f} Percentage of patients. Percentage of windows: {(len(df_train) / len(df) * 100):.2f}")
                print(f"Val:   {(patient_number_in_val / num_patients * 100):.2f} Percentage of patients. Percentage of windows: {(len(df_val) / len(df) * 100):.2f}")
                print(f"Test:  {(patient_number_in_test / num_patients * 100):.2f} Percentage of patients. Percentage of windows: {(len(df_test) / len(df) * 100):.2f}")

                x_train = df_train.iloc[:, :-1].to_numpy()  # Get all columns but the last two
                y_train = df_train.iloc[:, -1:].to_numpy()  # Get just the one before the last column

                x_val = df_val.iloc[:, :-1].to_numpy()  # Get all columns but the last one
                y_val = df_val.iloc[:, -1:].to_numpy()  # Get just the last column

                x_test = df_test.iloc[:, :-1].to_numpy()  # Get all columns but the last two
                y_test = df_test.iloc[:, -1:].to_numpy()  # Get just the one before the last column

                del df, df_train, df_val, df_test  # Free memory

                print()
                print("######################################################################## - ")
                print(f'START training {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1}')
                print()

                # Record the TRAINING start time
                start_time_training = time.time()

                hist, model_trained = train(x_train=x_train,
                                            y_train=y_train,
                                            x_val=x_val,
                                            y_val=y_val,
                                            model=model,
                                            history_length=history_length,
                                            save_filepath=f"models/{current_name}")

                # Record the TRAINING end time
                end_time_training = time.time()
                # Calculate the elapsed time of TRAINING
                training_time = end_time_training - start_time_training                
                elapsed_for_training = datetime.timedelta(seconds=training_time)
                
                print()
                print(f'END training {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1}')
                print("########################################################################")
                print()

                print(f"Training time for {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1}: {elapsed_for_training}")
                print()

                print("######################################################################## - ")
                print(f'START saving logs for {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1}')
                print()

                get_train_plots_loss(hist, f"plots/{current_name}")
                get_train_plots_MSE(hist, f"plots/{current_name}")

                df_hist = pd.DataFrame(data=hist.history)
                df_hist.to_csv(f'hist/df_hist_{current_name}.csv')

                hist_best = df_hist.iloc[df_hist['loss'].idxmin()]
                hist_best['epoch'] = df_hist.shape[0]
                hist_best.to_csv(f'hist/hist_best_{current_name}.csv')
                
                print(f'END saving logs for {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1}')
                print("########################################################################")
                print()

                # ## TESTING ################################

                print("######################################################################## - ")
                print(f'START testing for {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1}')
                print()

                # Record the TEST start time
                start_time_testing = time.time()

                if current_algorithm == 'Linear':  # load linear model
                    model_load = prepare_model_linear(history_length,
                                                      weights=f"models/{current_name}.weights.h5")
                                                      
                elif current_algorithm == 'LSTM':  # load LSTM model
                    model_load = prepare_model_LSTM(history_length,
                                                    weights=f"models/{current_name}.weights.h5")
                                                    
                else:  # load CNN model
                    model_load = prepare_model_CNN(history_length,
                                                   weights=f"models/{current_name}.weights.h5")

                bg_predict = model_load.predict(x_test)

                # Record the TEST end time
                end_time_testing = time.time()
                
                # Calculate the elapsed time of TEST
                testing_time = end_time_testing - start_time_testing
                
                # Convertir a timedelta
                elapsed_testing = datetime.timedelta(seconds=testing_time)

                print()
                print(f"Testing time for {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1}: {elapsed_testing}")

                print()
                print(f'Test results for {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - Fold={current_fold + 1} - DONE')
                print()

                # Save predictions and references
                df_test_results = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': bg_predict.ravel()})
                df_test_results.to_parquet(f'test_results/df_test_results_vectors_{current_name}.parquet')

                results_vector_list.append(df_test_results)
                

            # Concatenate all results for the current algorithm
            df_test_results = pd.concat(results_vector_list, ignore_index=True)

            # Save the concatenated results
            df_test_results.to_parquet(f'test_results/df_test_results_vectors_{current_algorithm}_H{current_horizon}.parquet')

            print(f'Metrics for {current_algorithm} - model_prediction_scripts Horizon={current_horizon} - All folds aggregated')

            print(f"Total test results for {current_algorithm} - model_prediction_scripts Horizon={current_horizon}: {len(df_test_results)} rows")

            df_metrics_by_range, df_zone_values, df_limits = test_by_range(df_test_results,
                                                                       f'_{current_algorithm}_H{current_horizon}')
            df_metrics_by_range = df_metrics_by_range.round(2)
            df_metrics_by_range.to_csv(f'test_results/metrics_performance_by_range_{current_algorithm}_H{current_horizon}.csv',
                                   index=False)

            df_zone_values.to_csv(f'test_results/number_of points_by_zones_{current_algorithm}_H{current_horizon}.csv',
                              index=False)

            df_limits.to_csv(f'test_results/predictions_out_of_limits_{current_algorithm}_H{current_horizon}.csv',
                         index=False)

            print(f'Test results of {current_algorithm} - model_prediction_scripts Horizon={current_horizon}  ALL FOLDS - DONE')
            print('____________________________________________________________________________')
            print()
finally:
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()