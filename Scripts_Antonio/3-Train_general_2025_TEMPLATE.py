# ----------------------- Imports -----------------------
from Test_predictions import *
from custom_loss_functions import *

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
import time
import datetime
import platform
import socket
import psutil
import subprocess

# ----------------------- End imports -----------------------

# ----------------------- Custom loss function -----------------------
def c_sig_p01(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.1)

def c_sig_p06(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.6)

def c_sig_MSE_p01(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.1) * MSE(y_true, y_pred)

def c_sig_MSE_p06(y_true, y_pred):
    return c_sig(y_true, y_pred, p=0.6) * MSE(y_true, y_pred)

loss_functions_dic = {"EloyPenaltyp01": c_sig_p01,  # Former: EloyPenaltyp01.  # c_sig (documento: Clarke Sigmoid-shaped Cost Function (C-Sig)
                      "EloyPenaltyp06": c_sig_p06,  # Former: EloyPenaltyp06,
                      "RMSE": RMSE,  # rmse
                      "gMSEitaliana": gMSE_italiana,  # gMSE
                      "gMADitaliana": gMAD_italiana,  # gMAD
                      "EloyPenaltyMSEp01": c_sig_MSE_p01, #  Former: EloyPenaltyMSEp01,
                      "EloyPenaltyMSEp06": c_sig_MSE_p06, #  Former: EloyPenaltyMSEp06,
                      }

loss_function_name_list = ['RMSE']

# ----------------------- ENd Custom loss function -----------------------

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

###################### CONFIG EXPERIMENT #############################

batch_size = 4096
patience = 10
max_epoch = 500
dropout = 0.1  # Dropout for LSTM layers
recurrent_dropout = 0.0  # Dropout rate for LSTM layers

# Path to the windows data (must include final slash)
windows_path = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/folds/folds_deduplicated_intra_patient_balanced_splits_DiaTrend_v3/'

history_length = 8  # 2h

do_cross_validation = True  # If True, it will use the cross-validation folds in the dataset

horizons = [4]
algorithms = ['LSTM']

# Sensor reading limits
minimum_sensor_reading = 38.96
maximum_sensor_reading = 401

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
    print(f"RAM Total        : {round(ram.total / (1024 ** 3), 2)} GB")
    print(f"RAM Available    : {round(ram.available / (1024 ** 3), 2)} GB")
    print(f"RAM Used         : {round(ram.used / (1024 ** 3), 2)} GB")
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
    def build_model(model, weights='', loss_function=loss_functions_dic['RMSE']):
        # build & compile model
        m = model.build()

        m.compile(loss=loss_function,
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanSquaredError(),
                           keras.metrics.RootMeanSquaredError(),
                           ClinicalPenaltyLoss()])  # DUDA: que es la ClinicalPenaltyLoss()
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
        # callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=early_stopping_patience,
        #                                                   min_lr=1e-6))
        callbacks.append(keras.callbacks.TensorBoard(log_dir=filepath + "_logs",
                                                     histogram_freq=0,
                                                     write_graph=True,
                                                     write_images=True,
                                                     update_freq='epoch',
                                                     profile_batch=0,
                                                     embeddings_freq=0,
                                                     embeddings_metadata=None))
        callbacks.append(keras.callbacks.CSVLogger(filepath + "_history.csv", separator=',',
                                                   append=False))  # DUDA: pq hay tanto cambio en la definición de la callback?
        return callbacks

    def prepare_model_LSTM(history_length, nb_hidden_units=128, weights='', loss_function=loss_functions_dic['RMSE']):
        model = LSTMModel(input_shape=(history_length, 1), nb_output_units=1, nb_hidden_units=nb_hidden_units)
        return build_model(model, weights, loss_function)


    def prepare_model_linear(history_length, weights='', loss_function=loss_functions_dic['RMSE']):
        model = LinearModel(input_shape=(history_length,), nb_output_units=1)
        return build_model(model, weights, loss_function)


    def prepare_model_CNN(history_length, weights='', loss_function=loss_functions_dic['RMSE']):
        model = CNNModel(input_shape=(history_length, 1), nb_output_units=1)
        return build_model(model, weights, loss_function)

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

    #################################################### TRAIN AND TEST ##########################################################

    for loss_function_name in loss_function_name_list:  # For all loss functions -------------------------------------
        loss_function = loss_functions_dic[loss_function_name]
        print(f" ----------- Loss function name: {loss_function_name} ----------- \n")

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
                    model = prepare_model_linear(history_length=history_length,
                                                 loss_function=loss_function)

                elif current_algorithm == 'LSTM':  # prepare LSTM model
                    model = prepare_model_LSTM(history_length=history_length,
                                               loss_function=loss_function)

                else:  # prepare CNN model
                    model = prepare_model_CNN(history_length=history_length,
                                              loss_function=loss_function)

                for current_fold in range(0, k_folds):  # For all folds -------------------------------------
                    current_name = f"{loss_function_name}_{current_algorithm}_H{current_horizon}_Fold{current_fold + 1}"

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
                    print(f'START training {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1}')
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
                    print(f'END training {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1}')
                    print("########################################################################")
                    print()

                    print(f"Training time for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1}: {elapsed_for_training}")
                    print()

                    print("######################################################################## - ")
                    print(f'START saving logs for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1}')
                    print()

                    get_train_plots_loss(hist, f"plots/{current_name}")
                    get_train_plots_MSE(hist, f"plots/{current_name}")

                    df_hist = pd.DataFrame(data=hist.history)
                    df_hist.to_csv(f'hist/df_hist_{current_name}.csv')

                    hist_best = df_hist.iloc[df_hist['loss'].idxmin()]
                    hist_best['epoch'] = df_hist.shape[0]
                    hist_best.to_csv(f'hist/hist_best_{current_name}.csv')

                    print(f'END saving logs for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1}')
                    print("########################################################################")
                    print()

                    # ## TESTING ################################

                    print("######################################################################## - ")
                    print(f'START testing for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1}')
                    print()

                    # Record the TEST start time
                    start_time_testing = time.time()

                    if current_algorithm == 'Linear':  # load linear model
                        model_load = prepare_model_linear(history_length,
                                                          loss_function=loss_function,
                                                          weights=f"models/{current_name}.weights.h5")

                    elif current_algorithm == 'LSTM':  # load LSTM model
                        model_load = prepare_model_LSTM(history_length,
                                                        loss_function=loss_function,
                                                         weights=f"models/{current_name}.weights.h5")

                    else:  # load CNN model
                        model_load = prepare_model_CNN(history_length,
                                                       loss_function=loss_function,
                                                       weights=f"models/{current_name}.weights.h5")

                    bg_predict = model_load.predict(x_test)

                    # Record the TEST end time
                    end_time_testing = time.time()

                    # Calculate the elapsed time of TEST
                    testing_time = end_time_testing - start_time_testing

                    # Convertir a timedelta
                    elapsed_testing = datetime.timedelta(seconds=testing_time)

                    print()
                    print(f"Testing time for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1}: {elapsed_testing}")

                    print()
                    print(f'Test results for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - Fold={current_fold + 1} - DONE')
                    print()

                    # Save predictions and references
                    df_test_results = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': bg_predict.ravel()})
                    df_test_results.to_parquet(f'test_results/df_test_results_vectors_{current_name}.parquet')

                    results_vector_list.append(df_test_results)


                # Concatenate all results for the current algorithm
                df_test_results = pd.concat(results_vector_list, ignore_index=True)

                # Save the concatenated results
                df_test_results.to_parquet(f'test_results/df_test_results_vectors_{loss_function_name}_{current_algorithm}_H{current_horizon}.parquet')

                print(f'Metrics for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon} - All folds aggregated')

                print(f"Total test results for {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon}: {len(df_test_results)} rows")

                df_metrics_by_range, df_zone_values, df_limits = test_by_range(df_test_results,
                                                                               f'_{loss_function_name}_{current_algorithm}_H{current_horizon}',
                                                                               minimum_sensor_reading,
                                                                               maximum_sensor_reading)
                df_metrics_by_range = df_metrics_by_range.round(2)
                df_metrics_by_range.to_csv(f'test_results/metrics_performance_by_range_{loss_function_name}_{current_algorithm}_H{current_horizon}.csv',
                                       index=False)

                df_zone_values.to_csv(f'test_results/number_of points_by_zones_{loss_function_name}_{current_algorithm}_H{current_horizon}.csv',
                                  index=False)

                df_limits.to_csv(f'test_results/predictions_out_of_limits_{loss_function_name}_{current_algorithm}_H{current_horizon}.csv',
                             index=False)

                print(f'Test results of {current_algorithm} - Loss function {loss_function_name} - Prediction Horizon={current_horizon}  ALL FOLDS - DONE')
                print('____________________________________________________________________________')
                print()
finally:
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()