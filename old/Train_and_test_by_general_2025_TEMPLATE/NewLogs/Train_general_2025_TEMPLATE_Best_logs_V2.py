# -*- coding: utf-8 -*-
"""
RELIEF-T1D — Training & model_prediction_scripts Script (English)

Directory layout per (loss, algo, PH):
models/
  <prefix>.weights.h5
  <prefix>_history.csv
  <prefix>.best.json
  <prefix>_logs/  # TensorBoard (tfevents)
plots/
  training/
    <prefix>_Loss.pdf
results/
  predictions/
    df_test_results_vectors_<prefix>.parquet
    df_test_results_vectors_<loss>_<algo>_H<PH>_ALL.parquet
  evaluation/
    metrics/
      metrics_performance_by_range_<loss>_<algo>_H<PH>_ALL.csv
      number_of_points_by_zones_<loss>_<algo>_H<PH>_ALL.csv
      predictions_out_of_limits_<loss>_<algo>_H<PH>_ALL.csv
    figures/
      Clarke_Error_Grid_<...>_ALL.png  # created by test_by_range(plot_dir=figures_dir)

Global stdout/stderr is duplicated to a timestamped .out file at the repo root.
"""

# ----------------------- Imports -----------------------
from Test_predictions import test_by_range, get_train_plots_loss  # test_by_range ONLY after ALL folds
from custom_loss_functions import *  # includes ClinicalPenaltyLoss and custom losses

from keras.layers import Dense, LSTM, Input, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
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
import json
import re
# ----------------------- End imports -----------------------

# ----------------------- Custom losses mapping -----------------------
loss_functions_dic = {
    "cSigP01": c_sig_p01,
    "cSigP06": c_sig_p06,
    "RMSE": RMSE,
    "gMSE": gMSE,
    "gMAD": gMAD,
    "cSigMSEP01": c_sig_MSE_p01,
    "cSigMSEP06": c_sig_MSE_p06,
}
loss_function_name_list = ['RMSE', 'cSigP01', 'cSigP06', 'gMSE', 'gMAD', 'cSigMSEP01', 'cSigMSEP06'] # all
#loss_function_name_list = ['cSigMSEP06']
# ----------------------- End custom losses mapping -----------------------

# ----------------------- Reproducibility -----------------------
np.random.seed(50)
tf.random.set_seed(50)
random.seed(50)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
# ----------------------- End reproducibility -----------------------

# ----------------------- Config -----------------------
batch_size = 4096
patience = 10
max_epoch = 500
dropout = 0.1
recurrent_dropout = 0.0

# Cross-platform base path to parquet windows (must end with os.sep)
windows_path = os.path.join(os.sep, 'opt', 'datasets', 'folds', 'T1DiabetesGranada') + os.sep

history_length = 8
do_cross_validation = True
horizons = [2, 4]  # PH values in (samples)
algorithms = ['Linear', 'LSTM', 'CNN']

# Sensor reading limits
maximum_sensor_reading = 500
minimum_sensor_reading = 40

# Context label for naming experiment directories
experiment_context = "CustomFunction"
# ----------------------- End Config -----------------------

# ----------------------- Console tee to file (global) -----------------------
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def close(self):
        for f in self.files:
            if f not in (sys.stdout, sys.stderr):
                f.close()

RUN_TS = datetime.datetime.now()
GLOBAL_RUN_ID = RUN_TS.strftime("%Y%m%d_%H%M%S")
global_log_file = open(f"{GLOBAL_RUN_ID}_train_log.out", "a", encoding="utf-8")
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(sys.stdout, global_log_file)
sys.stderr = Tee(sys.stderr, global_log_file)
# ----------------------- End console tee -----------------------

# ----------------------- System & Git helpers -----------------------
def print_system_info():
    print("📅 Date & Time")
    print("-----------------------------------")
    print(f"Experiment Start : {datetime.datetime.now()}")
    print("\n🖥️  System & Hardware")
    print("-----------------------------------")
    print(f"Hostname         : {socket.gethostname()}")
    print(f"OS               : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture     : {platform.machine()}")
    print(f"CPU              : {platform.processor()}")
    try:
        phys = psutil.cpu_count(logical=False)
    except Exception:
        phys = None
    print(f"CPU Cores        : {phys} physical / {psutil.cpu_count()} logical")
    ram = psutil.virtual_memory()
    print(f"RAM Total        : {round(ram.total / (1024 ** 3), 2)} GB")
    print(f"RAM Available    : {round(ram.available / (1024 ** 3), 2)} GB")
    print(f"RAM Used         : {round(ram.used / (1024 ** 3), 2)} GB")
    print(f"RAM Usage (%)    : {ram.percent}%")
    print("\n🐍 Python & Libraries")
    print("-----------------------------------")
    print(f"Python Version   : {platform.python_version()}")
    print(f"Executable Path  : {sys.executable}")
    print(f"TensorFlow       : {tf.__version__}")
    print("\n🧠 GPU (via TensorFlow)")
    print("-----------------------------------")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}           : {getattr(gpu, 'name', str(gpu))}")
        else:
            print("No GPUs detected by TensorFlow.")
    except Exception as e:
        print(f"GPU detection error: {e}")
    print("\n🎮 GPU Details (via nvidia-smi)")
    print("-----------------------------------")
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT
        )
        for i, line in enumerate(out.decode(errors="ignore").strip().split("\n")):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                name, mem_total, mem_used, util, driver = parts[:5]
                print(f"GPU {i}           : {name}")
                print(f"  Total Memory   : {mem_total}")
                print(f"  Used Memory    : {mem_used}")
                print(f"  Utilization    : {util}")
                print(f"  Driver Version : {driver}")
    except Exception as e:
        print(f"nvidia-smi not available: {e}")

def get_git_info():
    info = {"commit": None, "branch": None, "remote": None}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        try:
            info["remote"] = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"]).decode().strip()
        except subprocess.CalledProcessError:
            info["remote"] = None
    except Exception:
        pass
    return info
# ----------------------- End system & git helpers -----------------------

# ----------------------- Experiment directory helpers -----------------------
def safe_token(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))

def safe_relpath(p, start=None):
    try:
        return os.path.relpath(p, start=start) if start else os.path.relpath(p)
    except Exception:
        return p

def next_versioned_dir(base_name: str) -> str:
    rx = re.compile(r"^(.*?)(?:-v(\d+))?$")
    m = rx.match(base_name)
    stem, v = m.group(1), m.group(2)
    vnum = int(v) if v else 1
    while True:
        candidate = f"{stem}-v{vnum}"
        if not os.path.exists(candidate):
            return candidate
        vnum += 1

def build_experiment_dir(algo: str, loss_name: str, ph_value, history_len: int, context: str) -> str:
    """
    EXP-<YYYYMMDD>-prediction-<ALGO>-<LOSS>-H<history>-PH<ph>-<Context>-vN
    """
    date_str = RUN_TS.strftime("%Y%m%d")
    algo_t = safe_token(algo)
    loss_t = safe_token(loss_name)
    ph_t = safe_token(ph_value)
    ctx_t = safe_token(context)
    base = f"EXP-{date_str}-prediction-{algo_t}-{loss_t}-H{history_len}-PH{ph_t}-{ctx_t}"
    exp_dir = next_versioned_dir(base)
    for sub in ["models", "plots", "results", "summaries", "readmes"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    # nested result dirs
    os.makedirs(os.path.join(exp_dir, "plots", "training"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results", "predictions"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results", "evaluation", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "results", "evaluation", "figures"), exist_ok=True)
    return exp_dir
# ----------------------- End experiment directory helpers -----------------------

# ----------------------- Models -----------------------
class LSTMModel:
    def __init__(self, input_shape, nb_output_units, nb_hidden_units=128,
                 dropout_rate=dropout, recurrent_dropout_rate=recurrent_dropout):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
        self.nb_hidden_units = nb_hidden_units
        self.dropout = dropout_rate
        self.recurrent_dropout = recurrent_dropout_rate
    def __repr__(self):
        return (f"LSTM_{self.nb_hidden_units}_h"
                f"_drop{self.dropout}_recDrop{self.recurrent_dropout}")
    def build(self):
        i = Input(shape=self.input_shape)
        x = LSTM(self.nb_hidden_units,
                 dropout=self.dropout,
                 recurrent_dropout=self.recurrent_dropout)(i)
        x = Dense(self.nb_output_units, activation=None)(x)
        return Model(inputs=[i], outputs=[x])

class LinearModel:
    def __init__(self, input_shape, nb_output_units):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
    def __repr__(self):
        return 'Linear'
    def build(self):
            i = Input(shape=self.input_shape)           # (history_length, 1)
            x = Flatten()(i)
            x = Dense(self.nb_output_units, activation=None)(x)
            
            return Model(inputs=[i], outputs=[x])

class CNNModel:
    def __init__(self, input_shape, nb_output_units):
        self.input_shape = input_shape
        self.nb_output_units = nb_output_units
    def __repr__(self):
        return 'CNN'
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

def build_model(model, weights='', loss_function=loss_functions_dic['RMSE']):
    m = model.build()
    m.compile(
        loss=loss_function,
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.RootMeanSquaredError(), ClinicalPenaltyLoss()]
    )
    if weights:
        print(f"Weights: {weights}")
        m.load_weights(weights)
    return m

def prepare_model_LSTM(history_len, nb_hidden_units=128, weights='', loss_function=loss_functions_dic['RMSE']):
    model = LSTMModel(input_shape=(history_len, 1), nb_output_units=1, nb_hidden_units=nb_hidden_units)
    return build_model(model, weights, loss_function)

def prepare_model_linear(history_len, weights='', loss_function=loss_functions_dic['RMSE']):
    model = LinearModel(input_shape=(history_len,), nb_output_units=1)
    return build_model(model, weights, loss_function)

def prepare_model_CNN(history_len, weights='', loss_function=loss_functions_dic['RMSE']):
    model = CNNModel(input_shape=(history_len, 1), nb_output_units=1)
    return build_model(model, weights, loss_function)
# ----------------------- End models -----------------------

# ----------------------- Keras callbacks -----------------------
def make_callbacks(filepath_prefix: str, early_stopping_patience: int):
    """
    filepath_prefix: prefix for artifacts. ModelCheckpoint writes prefix + '.weights.h5'
    CSVLogger writes prefix + '_history.csv'
    TensorBoard writes logs into prefix + '_logs/' as TF event files.
    """
    cbs = []
    cbs.append(ModelCheckpoint(filepath=filepath_prefix + ".weights.h5",
                               monitor='val_loss',
                               mode='min',
                               save_best_only=True,
                               save_weights_only=True,
                               verbose=1))
    cbs.append(EarlyStopping(monitor='val_loss',
                             mode='min',
                             patience=early_stopping_patience,
                             restore_best_weights=True,
                             verbose=1))
    cbs.append(keras.callbacks.TensorBoard(
        log_dir=filepath_prefix + "_logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,     # leaner TB logs; faster I/O
        update_freq='epoch',
        profile_batch=0
    ))
    cbs.append(keras.callbacks.CSVLogger(filepath_prefix + "_history.csv",
                                         separator=',', append=False))
    return cbs
# ----------------------- End callbacks -----------------------

# ----------------------- Train function -----------------------
def train(x_train, 
          y_train, 
          x_val, 
          y_val, 
          model, 
          history_len, 
          save_prefix="", 
          early_stopping_patience=patience):
    need_3d = (len(model.input_shape) == 3)
    if need_3d:
        x_train = x_train.reshape(x_train.shape[0], history_length, 1)
        x_val   = x_val.reshape(x_val.shape[0], history_length, 1)
        
    validation_data = (x_val, y_val)
    
    hist = model.fit(x_train, 
                     y_train,
                     batch_size=batch_size,
                     validation_data=validation_data,
                    epochs=max_epoch,                                     
                    callbacks=make_callbacks(save_prefix, 
                                             early_stopping_patience)
    )
    return hist, model
# ----------------------- End train function -----------------------

# ----------------------- Per-fold JSON/MD writers -----------------------
def write_fold_json_and_md(
    exp_dir, loss_name, algo, ph, fold_idx,
    fold_started_at,
    train_sec, train_hms, pred_sec, pred_hms,
    weights_path, hist_csv_path, best_json_path, preds_parquet_path, training_plot_prefix,
    history_len, do_cv, min_sensor, max_sensor,
    data_file_path,
    data_counts
):
    # System info
    sys_info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "arch": platform.machine(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "tensorflow": getattr(tf, "__version__", None),
    }
    try:
        gpus = tf.config.list_physical_devices('GPU')
        sys_info["gpus_tf"] = [getattr(g, "name", "GPU") for g in gpus] if gpus else []
    except Exception:
        sys_info["gpus_tf"] = []

    nvsmi = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT
        )
        for line in out.decode(errors="ignore").strip().split("\n"):
            if not line.strip():
                continue
            name, mem_total, mem_used, util, driver = [p.strip() for p in line.split(",")]
            nvsmi.append({
                "name": name, "memory_total": mem_total, "memory_used": mem_used,
                "utilization": util, "driver_version": driver
            })
    except Exception as e:
        nvsmi = [{"error": str(e)}]
    sys_info["nvidia_smi"] = nvsmi

    # JSON payload (fold summary)
    payload = {
        "experiment": {
            "loss": loss_name,
            "algorithm": algo,
            "prediction_horizon": ph,
            "fold": fold_idx,
            "history_length": history_len,
            "cross_validation": do_cv,
            "sensor_limits": {"min": min_sensor, "max": max_sensor},
            "started_at": fold_started_at.isoformat(),
            "run_id": GLOBAL_RUN_ID
        },
        "data": {
            "file": os.path.abspath(data_file_path),
            "counts": data_counts
        },
        "timing": {
            "train_seconds": round(train_sec, 3),
            "train_elapsed": str(train_hms),
            "predict_seconds": round(pred_sec, 3),
            "predict_elapsed": str(pred_hms)
        },
        "artifacts": {
            "weights": safe_relpath(weights_path, start=exp_dir),
            "history_csv": safe_relpath(hist_csv_path, start=exp_dir),
            "best_json": safe_relpath(best_json_path, start=exp_dir),
            "preds_parquet": safe_relpath(preds_parquet_path, start=exp_dir),
            "training_plot_prefix": safe_relpath(training_plot_prefix, start=exp_dir)
        },
        "git": get_git_info(),
        "system": sys_info
    }
    json_name = os.path.join(exp_dir, "summaries", f"summary_{loss_name}_{algo}_H{ph}_Fold{fold_idx}.json")
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Markdown (human-readable)
    lines = []
    lines.append(f"Experiment Start : {fold_started_at}")
    lines.append("")
    lines.append("🖥️  System & Hardware")
    lines.append("-----------------------------------")
    lines.append(f"Hostname         : {sys_info['hostname']}")
    lines.append(f"OS               : {sys_info['os']}")
    lines.append(f"Architecture     : {sys_info['arch']}")
    lines.append(f"CPU              : {sys_info['cpu']}")
    try:
        phys = psutil.cpu_count(logical=False)
    except Exception:
        phys = None
    lines.append(f"CPU Cores        : {phys} physical / {psutil.cpu_count()} logical")
    ram = psutil.virtual_memory()
    lines.append(f"RAM Total        : {round(ram.total / (1024 ** 3), 2)} GB")
    lines.append(f"RAM Available    : {round(ram.available / (1024 ** 3), 2)} GB")
    lines.append(f"RAM Used         : {round(ram.used / (1024 ** 3), 2)} GB")
    lines.append(f"RAM Usage (%)    : {ram.percent}%")
    lines.append("")
    lines.append("🐍 Python & Libraries")
    lines.append("-----------------------------------")
    lines.append(f"Python Version   : {sys_info['python']}")
    lines.append(f"Executable Path  : {sys.executable}")
    lines.append(f"TensorFlow       : {sys_info['tensorflow']}")
    lines.append("")
    lines.append("🧠 GPU (via TensorFlow)")
    lines.append("-----------------------------------")
    if sys_info["gpus_tf"]:
        for i, gname in enumerate(sys_info["gpus_tf"]):
            lines.append(f"GPU {i}           : {gname}")
    else:
        lines.append("No GPUs detected by TensorFlow.")
    lines.append("")
    lines.append("🎮 GPU Details (via nvidia-smi)")
    lines.append("-----------------------------------")
    if nvsmi and "error" not in nvsmi[0]:
        for i, g in enumerate(nvsmi):
            lines.append(f"GPU {i}           : {g['name']}")
            lines.append(f"  Total Memory   : {g['memory_total']}")
            lines.append(f"  Used Memory    : {g['memory_used']}")
            lines.append(f"  Utilization    : {g['utilization']}")
            lines.append(f"  Driver Version : {g['driver_version']}")
    else:
        lines.append(f"nvidia-smi not available: {nvsmi[0].get('error', 'unknown')}")
    lines.append("-----------------------------------")
    lines.append("")
    lines.append(f" ----------- Loss function name: {loss_name} ----------- ")
    lines.append("")
    if do_cv:
        lines.append("Cross-validation enabled with 5 folds.")
    else:
        lines.append("Cross-validation is disabled. Using the first fold for training, validation, and prediction.")
    lines.append("")
    lines.append(f"---- Fold {fold_idx}: -----")
    lines.append("-- Totals --")
    lines.append(f"  Train: {data_counts['patients']['train']} patients. Total windows: {data_counts['windows']['train']}")
    lines.append(f"  Val:   {data_counts['patients']['val']} patients. Total windows: {data_counts['windows']['val']}")
    lines.append(f"  Test:  {data_counts['patients']['test']} patients. Total windows: {data_counts['windows']['test']}")
    lines.append("")
    lines.append(" -- Percentages --")
    lines.append(f"Train: {data_counts['percent']['patients']['train']:.2f} Percentage of patients. "
                 f"Percentage of windows: {data_counts['percent']['windows']['train']:.2f}")
    lines.append(f"Val:   {data_counts['percent']['patients']['val']:.2f} Percentage of patients. "
                 f"Percentage of windows: {data_counts['percent']['windows']['val']:.2f}")
    lines.append(f"Test:  {data_counts['percent']['patients']['test']:.2f} Percentage of patients. "
                 f"Percentage of windows: {data_counts['percent']['windows']['test']:.2f}")
    lines.append("")
    lines.append(f"Data file: {os.path.abspath(data_file_path)}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- Weights: `{safe_relpath(weights_path, start=exp_dir)}`")
    lines.append(f"- Training history (CSV): `{safe_relpath(hist_csv_path, start=exp_dir)}`")
    lines.append(f"- Best summary (JSON): `{safe_relpath(best_json_path, start=exp_dir)}`")
    lines.append(f"- Test predictions (Parquet): `{safe_relpath(preds_parquet_path, start=exp_dir)}`")
    lines.append(f"- Training curve (PDF prefix): `{safe_relpath(training_plot_prefix, start=exp_dir)}`")
    lines.append("")
    lines.append("## Repro")
    lines.append(f"- Loss: `{loss_name}`  •  Algo: `{algo}`  •  Horizon: `{ph}`  •  Fold: `{fold_idx}`")
    lines.append("")
    lines.append("*(File generated automatically by the training script.)*")
    md_name = os.path.join(exp_dir, "readmes", f"README_{loss_name}_{algo}_H{ph}_Fold{fold_idx}.md")
    with open(md_name, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
# ----------------------- End per-fold JSON/MD writers -----------------------

# ----------------------- Main Loop: train + predict per fold; metrics only after ALL folds -----------------------
print_system_info()
print("-----------------------------------\n")

try:
    for loss_function_name in loss_function_name_list: # For all loss functions -------------------------------------
        loss_function = loss_functions_dic[loss_function_name]
        print(f" ----------- Loss function name: {loss_function_name} ----------- \n")

        for current_horizon in horizons: # For all prediction horizons -------------------------------------
            tf.keras.backend.clear_session()
            np.random.seed(50)
            tf.random.set_seed(50)
            random.seed(50)                                    
            used_columns = [f"x{i}" for i in range(history_length)] + ["y"] # Columns to be used for training, validation, and testing

            if do_cross_validation:
                k_folds = 5
                print(f"Cross-validation enabled with {k_folds} folds.")
            else:
                k_folds = 1
                print("Cross-validation is disabled. Using the first fold for training, validation, and prediction.")

            for current_algorithm in algorithms: # For all algorithms ---------------------------------------               
                # Per-(loss, algo, PH) experiment directory (includes LOSS)
                EXP_DIR = build_experiment_dir(current_algorithm, loss_function_name, current_horizon, history_length, experiment_context)
                MODELS_DIR = os.path.join(EXP_DIR, "models")
                PLOTS_TRAIN_DIR = os.path.join(EXP_DIR, "plots", "training")
                PRED_DIR = os.path.join(EXP_DIR, "results", "predictions")
                METRICS_DIR = os.path.join(EXP_DIR, "results", "evaluation", "metrics")
                FIGURES_DIR = os.path.join(EXP_DIR, "results", "evaluation", "figures")

                aggregated_results = []  # collect per-fold predictions for this (loss, algo, PH)

                # Prepare model instance template (new one per fold anyway)
                for current_fold in range(0, k_folds):
                    current_name = f"{loss_function_name}_{current_algorithm}_H{current_horizon}_Fold{current_fold + 1}"
                    data_file_path = f"{windows_path}windows_with_folds_horizon_{current_horizon}.parquet"

                    df = pd.read_parquet(data_file_path)

                    # Split by fold
                    # Training split
                    df_train = df[df[f'fold_{current_fold}'] == 'train']
                    patients_train_number = df_train['patient_id'].nunique()
                    df_train = df_train.reset_index(drop=True)[used_columns]

                    # Validation split
                    df_val = df[df[f'fold_{current_fold}'] == 'val']
                    patients_val_number = df_val['patient_id'].nunique()
                    df_val = df_val.reset_index(drop=True)[used_columns]

                    # Testing split
                    df_test = df[df[f'fold_{current_fold}'] == 'test']
                    patients_test_number = df_test['patient_id'].nunique()
                    df_test = df_test.reset_index(drop=True)[used_columns]

                    total_patients = patients_train_number + patients_val_number + patients_test_number

                    print(f"\n---- Fold {current_fold + 1}: -----")
                    print("-- Totals --")
                    print(f"  Train: {patients_train_number} patients. Total windows: {len(df_train)}")
                    print(f"  Val:   {patients_val_number} patients. Total windows: {len(df_val)}")
                    print(f"  Test:  {patients_test_number} patients. Total windows: {len(df_test)}")

                    print("\n -- Percentages --")
                    print(f"Train: {(patients_train_number / total_patients * 100):.2f} Percentage of patients. "
                          f"Percentage of windows: {(len(df_train) / len(df) * 100):.2f}")
                    print(f"Val:   {(patients_val_number / total_patients * 100):.2f} Percentage of patients. "
                          f"Percentage of windows: {(len(df_val) / len(df) * 100):.2f}")
                    print(f"Test:  {(patients_test_number / total_patients * 100):.2f} Percentage of patients. "
                          f"Percentage of windows: {(len(df_test) / len(df) * 100):.2f}")

                    # Numpy arrays
                    x_train = df_train.iloc[:, :-1].to_numpy() # Get all columns but the last one
                    y_train = df_train.iloc[:, -1:].to_numpy() # Get just the last column
                    
                    x_val = df_val.iloc[:, :-1].to_numpy() # Get all columns but the last one
                    y_val = df_val.iloc[:, -1:].to_numpy() # Get just the last column
                    
                    x_test = df_test.iloc[:, :-1].to_numpy() # Get all columns but the last one
                    y_test = df_test.iloc[:, -1:].to_numpy() # Get just the last column
                    
                    del df, df_train, df_val, df_test # Free memory

                    # Data counts payload for per-fold logs
                    total_windows = len(x_train) + len(x_val) + len(x_test)
                    data_counts = {
                        "patients": {
                            "train": int(patients_train_number),
                            "val": int(patients_val_number),
                            "test": int(patients_test_number),
                            "total": int(total_patients)
                        },
                        "windows": {
                            "train": int(len(x_train)),
                            "val": int(len(x_val)),
                            "test": int(len(x_test)),
                            "total": int(total_windows)
                        },
                        "percent": {
                            "patients": {
                                "train": (patients_train_number / total_patients * 100.0) if total_patients else 0.0,
                                "val":   (patients_val_number / total_patients * 100.0) if total_patients else 0.0,
                                "test":  (patients_test_number / total_patients * 100.0) if total_patients else 0.0
                            },
                            "windows": {
                                "train": (len(x_train) / total_windows * 100.0) if total_windows else 0.0,
                                "val":   (len(x_val)   / total_windows * 100.0) if total_windows else 0.0,
                                "test":  (len(x_test)  / total_windows * 100.0) if total_windows else 0.0
                            }
                        }
                    }

                    print()
                    print("######################################################################## - ")
                    print(f'START training {current_algorithm} - Loss {loss_function_name} - PH={current_horizon} - Fold={current_fold + 1}')
                    print()

                    # Prepare a fresh model for this fold
                    if current_algorithm == 'Linear': # prepare linear model
                        model = prepare_model_linear(history_length, loss_function=loss_function)
                        
                    elif current_algorithm == 'LSTM': # prepare LSTM model
                        model = prepare_model_LSTM(history_length, loss_function=loss_function)
                        
                    else: # prepare CNN model
                        model = prepare_model_CNN(history_length, loss_function=loss_function)

                    # Timings
                    fold_started_at = datetime.datetime.now()
                    start_time_training = time.time()

                    save_prefix = os.path.join(MODELS_DIR, current_name)
                    hist, model_trained = train(
                        x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val,
                        model=model,
                        history_len=history_length,
                        save_prefix=save_prefix
                    )

                    end_time_training = time.time()
                    training_time = end_time_training - start_time_training
                    elapsed_for_training = datetime.timedelta(seconds=training_time)

                    print()
                    print(f'END training {current_algorithm} - Loss {loss_function_name} - PH={current_horizon} - Fold={current_fold + 1}')
                    print("########################################################################\n")
                    print(f"Training time: {elapsed_for_training}\n")

                    print("######################################################################## - ")
                    print(f'START saving logs {current_algorithm} - Loss {loss_function_name} - PH={current_horizon} - Fold={current_fold + 1}\n')

                    # Training curve PDF into plots/training/
                    training_curve_prefix = os.path.join(PLOTS_TRAIN_DIR, current_name)
                    get_train_plots_loss(hist, training_curve_prefix)

                    # Path to canonical CSV (callback output)
                    hist_csv_path = save_prefix + "_history.csv"

                    # ---- Compute BEST from the CSV and write <prefix>.best.json ----
                    try:
                        df_hist = pd.read_csv(hist_csv_path)
                        if 'val_loss' in df_hist.columns:
                            best_idx = int(df_hist['val_loss'].idxmin())
                        else:
                            best_idx = int(df_hist['loss'].idxmin())

                        row = df_hist.iloc[best_idx].to_dict()
                        best_payload = {
                            "monitor": "val_loss" if 'val_loss' in df_hist.columns else "loss",
                            "epoch_idx": best_idx,                 # 0-based
                            "epoch_count": int(df_hist.shape[0]),  # total epochs executed
                            "metrics": row,                        # all available scalar columns for that epoch
                            "artifacts": {
                                "weights_path": os.path.abspath(save_prefix + ".weights.h5"),
                                "history_csv_path": os.path.abspath(hist_csv_path)
                            },
                            "context": {
                                "loss": loss_function_name,
                                "algorithm": current_algorithm,
                                "prediction_horizon": current_horizon,
                                "history_length": history_length,
                                "fold": current_fold + 1,
                                "run_id": GLOBAL_RUN_ID
                            },
                            "created_at": datetime.datetime.now().isoformat()
                        }
                        best_json_path = save_prefix + ".best.json"
                        with open(best_json_path, "w", encoding="utf-8") as f:
                            json.dump(best_payload, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"WARNING: could not write best JSON for {current_name}: {e}")
                        best_json_path = save_prefix + ".best.json"  # still pass the path onward

                    print(f'END saving logs {current_algorithm} - Loss {loss_function_name} - PH={current_horizon} - Fold={current_fold + 1}')
                    print("########################################################################\n")

                    # ===================== PREDICTION (within fold) — NO METRICS HERE =====================
                    print("######################################################################## - ")
                    print(f'START prediction {current_algorithm} - Loss {loss_function_name} - PH={current_horizon} - Fold={current_fold + 1}\n')

                    start_time_pred = time.time()

                    # Reload model with best weights
                    if current_algorithm == 'Linear':
                        model_load = prepare_model_linear(history_length,
                                                          loss_function=loss_function,
                                                          weights=save_prefix + ".weights.h5")
                    elif current_algorithm == 'LSTM':
                        model_load = prepare_model_LSTM(history_length,
                                                        loss_function=loss_function,
                                                        weights=save_prefix + ".weights.h5")
                    else:
                        model_load = prepare_model_CNN(history_length,
                                                       loss_function=loss_function,
                                                       weights=save_prefix + ".weights.h5")

                    need_3d_test = (len(model_load.input_shape) == 3)   # (None, history_length, 1)
                    if need_3d_test:
                        x_test = x_test.reshape(x_test.shape[0], history_length, 1)
                        
                    y_pred = model_load.predict(x_test)

                    end_time_pred = time.time()
                    pred_time = end_time_pred - start_time_pred
                    elapsed_pred = datetime.timedelta(seconds=pred_time)

                    print(f"\nmodel_prediction_scripts time: {elapsed_pred}\n")
                    print(f'model_prediction_scripts finished {current_algorithm} - Loss {loss_function_name} - PH={current_horizon} - Fold={current_fold + 1}\n')

                    # Save predictions vs references to results/predictions/
                    df_test_results = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': y_pred.ravel()})
                    test_vectors_path = os.path.join(PRED_DIR, f'df_test_results_vectors_{current_name}.parquet')
                    df_test_results.to_parquet(test_vectors_path)

                    aggregated_results.append(df_test_results)

                    # Per-fold JSON & README (timings + artifacts, NO metrics here)
                    weights_path = save_prefix + ".weights.h5"
                    write_fold_json_and_md(
                        EXP_DIR, loss_function_name, current_algorithm, current_horizon, current_fold + 1,
                        fold_started_at,
                        training_time, elapsed_for_training, pred_time, elapsed_pred,
                        weights_path, hist_csv_path, best_json_path, test_vectors_path, training_curve_prefix,
                        history_length, do_cross_validation, minimum_sensor_reading, maximum_sensor_reading,
                        data_file_path,
                        data_counts
                    )
                    # ===================== END PREDICTION =====================

                # --------------- AFTER ALL FOLDS (for this loss, algo, PH) ---------------
                # Aggregate predictions across folds and compute metrics/CEG ONCE
                df_all_folds = pd.concat(aggregated_results, ignore_index=True)
                agg_vectors_path = os.path.join(
                    PRED_DIR, f'df_test_results_vectors_{loss_function_name}_{current_algorithm}_H{current_horizon}.parquet'
                )
                df_all_folds.to_parquet(agg_vectors_path)

                print(f'Aggregated ALL folds — {current_algorithm} | Loss {loss_function_name} | PH={current_horizon}')
                print(f"Total aggregated rows: {len(df_all_folds)}\n")

                # Metrics
                df_metrics_by_range_all, df_zone_values_all, df_limits_all = test_by_range(
                    df_all_folds,
                    f'_{loss_function_name}_{current_algorithm}_H{current_horizon}_ALL',
                    plot_dir=FIGURES_DIR,  # CEG figures go to results/evaluation/figures/
                    minimum_sensor_reading=minimum_sensor_reading,
                    maximum_sensor_reading=maximum_sensor_reading
                )
                df_metrics_by_range_all = df_metrics_by_range_all.round(2)

                metrics_by_range_csv_all = os.path.join(
                    METRICS_DIR, f'metrics_performance_by_range_{loss_function_name}_{current_algorithm}_H{current_horizon}_ALL.csv'
                )
                zone_values_csv_all = os.path.join(
                    METRICS_DIR, f'number_of_points_by_zones_{loss_function_name}_{current_algorithm}_H{current_horizon}_ALL.csv'
                )
                limits_csv_all = os.path.join(
                    METRICS_DIR, f'predictions_out_of_limits_{loss_function_name}_{current_algorithm}_H{current_horizon}_ALL.csv'
                )
                df_metrics_by_range_all.to_csv(metrics_by_range_csv_all, index=False)
                df_zone_values_all.to_csv(zone_values_csv_all, index=False)
                df_limits_all.to_csv(limits_csv_all, index=False)

                print(f'ALL FOLDS metrics written for {current_algorithm} - Loss {loss_function_name} - PH={current_horizon}')
                print('____________________________________________________________________________\n')

finally:
    try:
        pass
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        global_log_file.close()
