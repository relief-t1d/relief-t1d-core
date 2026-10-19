# -*- coding: utf-8 -*-
"""
RELIEF-T1D: Model Training and Evaluation Pipeline

This script orchestrates the end-to-end training process for blood glucose
prediction models. It handles the loading of preprocessed temporal windows,
model instantiation (e.g., LSTMs, CNNs), k-fold cross-validation, and
comprehensive evaluation using both statistical and clinical metrics.

All experiment hyperparameters, sensor limits, and paths are governed by
an external YAML configuration file for full reproducibility.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
import random
import numpy as np

# Load environment variables from a local .env file into os.environ if present
load_dotenv()

# ---------------------------------------------------------
# 1. SYSTEM PATH AND CONFIGURATION LOADING
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train_prediction_models.yaml"

def parse_arguments():
    """Parse CLI arguments for configuration injection."""
    parser = argparse.ArgumentParser(description="Training script using YAML config.")
    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML configuration file."
    )
    return parser.parse_args()

def get_experiment_pairs(hyperparams_config):
    """
    Extract (history_length, horizon) pairs. Supports both proportional history
    multipliers (e.g., [0.25, 0.5, 1.0]) and explicit history length lists.
    """
    pairs = []

    if 'history_multipliers' in hyperparams_config:
        multipliers = hyperparams_config['history_multipliers']
        for ph in hyperparams_config['horizons']:
            calculated_h = sorted(list(set(
                max(1, int(round(ph * m))) for m in multipliers
            )))
            for h in calculated_h:
                pairs.append((h, ph))
    elif 'history_lengths' in hyperparams_config:
        for h in hyperparams_config['history_lengths']:
            for ph in hyperparams_config['horizons']:
                pairs.append((h, ph))
    else:
        raise KeyError("Experiment section must contain either 'history_multipliers' or 'history_lengths'.")

    return pairs

# Load the configuration file and dynamically expand environment variables (e.g., ${PATH_TO_DATA})
cli_args = parse_arguments()
with open(cli_args.config, 'r', encoding='utf-8') as f:
    raw_yaml_content = f.read()
    expanded_yaml_content = os.path.expandvars(raw_yaml_content)
    config = yaml.safe_load(expanded_yaml_content)

# ---------------------------------------------------------
# 2. GLOBAL REPRODUCIBILITY AND DETERMINISM (Before TensorFlow)
# ---------------------------------------------------------

# Extract seed from config
global_seed = config['experiment']['random_seed']

# Force GPU and environment to be deterministic BEFORE loading TensorFlow
os.environ['PYTHONHASHSEED'] = str(global_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set core Python and NumPy seeds
random.seed(global_seed)
np.random.seed(global_seed)

# Now it is safe to import TensorFlow and set its global seeds
import tensorflow as tf
tf.random.set_seed(global_seed)
tf.keras.utils.set_random_seed(global_seed)
from tensorflow import keras

# ---------------------------------------------------------
# 3. STANDARD AND THIRD-PARTY LIBRARIES
# ---------------------------------------------------------
import time
import datetime
import platform
import socket
import json
import re
import psutil
import subprocess
import pandas as pd
import shutil

# ---------------------------------------------------------
# 4. IMPORTS FROM INTERNAL MODULAR LIBRARY (src/)
# ---------------------------------------------------------
from src.models.losses import *
from src.models.architectures import get_compiled_model
from src.evaluation.metrics import test_by_range, get_train_plots_loss, fold_results_aggregation

# ---------------------------------------------------------
# 5. CONFIGURATION VARIABLES INJECTION
# ---------------------------------------------------------
# Mapping dictionary linking string keys in YAML to concrete loss functions in Python
loss_functions_dic = {
    "cSigP01": c_sig_p01,
    "cSigP06": c_sig_p06,
    "RMSE": RMSE,
    "gMSE": gMSE,
    "gMAD": gMAD,
    "cSigMSEP01": c_sig_p01_MSE,
    "cSigMSEP06": c_sig_p06_MSE,
}

# Extracting experiment configuration
loss_function_name_list = config['experiment']['loss_functions']
algorithms = config['experiment']['algorithms']
do_cross_validation = config['experiment']['do_cross_validation']
k_folds = config['experiment']['k_folds']

# Dynamically derive (history_length, horizon) experiment pairs
experiment_pairs = get_experiment_pairs(config['experiment'])

# Extracting training hyperparameters
batch_size = config['training_hyperparameters']['batch_size']
patience = config['training_hyperparameters']['patience']
max_epoch = config['training_hyperparameters']['max_epoch']
dropout = config['training_hyperparameters']['dropout']
recurrent_dropout = config['training_hyperparameters']['recurrent_dropout']

# Extracting sensor limits
minimum_sensor_reading = config['sensor_limits']['minimum_reading']
maximum_sensor_reading = config['sensor_limits']['maximum_reading']

# Extracting paths
windows_path = config['paths']['windows_path']
experiment_base_path = config['paths']['experiment_base_path']
central_models_path = config['paths']['central_models_path']

# ---------------------------------------------------------
# 6. METADATA AND LOGGING VARIABLES (README)
# ---------------------------------------------------------
dataset_name = config['metadata']['dataset_name']
dataset_link = config['metadata']['link']

training_data = config['metadata']['training_data']
validation_data = config['metadata']['validation_data']
testing_data = config['metadata']['testing_data']
wise_level = config['metadata']['wise_level']

# Dynamic variables based on experiment setup
cross_validation = 'k-fold' if do_cross_validation else 'No'
Folds = str(k_folds) if do_cross_validation else '1'

# ----------------------- Console Tee Logic -----------------------
class Tee:
    def __init__(self, *files):
        self.files = [f for f in files if f]

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                try:
                    f.write(obj)
                    f.flush()
                except (IOError, ValueError):
                    pass

    def flush(self):
        for f in self.files:
            if not f.closed:
                try:
                    f.flush()
                except (IOError, ValueError):
                    pass

# RUN_TS = datetime.datetime.now()
# GLOBAL_RUN_ID = RUN_TS.strftime("%Y%m%d_%H%M%S")
original_stdout = sys.stdout
original_stderr = sys.stderr

# ----------------------- System & Git helpers -----------------------
def print_system_info(experiment_start_time):
    print("📅 Date & Time")
    print("-----------------------------------")
    print(f"Experiment Start Time: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # print(f"Current Time: {datetime.datetime.now()}")
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
    info = {"commit": "N/A", "branch": "N/A", "remote": "N/A"}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        info["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        info["remote"] = subprocess.check_output(["git", "config", "--get", "remote.origin.url"]).decode().strip()
    except Exception:
        pass
    return info


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


def build_experiment_dir(base_path: str, dataset_name: str, algorithm_name: str, loss_name: str) -> str:
    date_str = RUN_TS.strftime("%Y%m%d%H%M")
    exp_folder_name = f"{date_str}-pred-{safe_token(dataset_name)}-{safe_token(algorithm_name)}-{safe_token(loss_name)}"

    # Combine the base path with the experiment folder name
    base = os.path.join(base_path, exp_folder_name)

    # exp_dir = next_versioned_dir(base)
    exp_dir = base  # Use the base name directly without versioning
    for sub in ["models", "plots/training", "results/predictions", "results/evaluation/metrics",
                "results/evaluation/figures", "summaries", "readmes"]:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)
    return exp_dir


# ----------------------- Keras callbacks -----------------------
def make_callbacks(filepath_prefix: str, early_stopping_patience: int):
    return [
        keras.callbacks.ModelCheckpoint(filepath=f"{filepath_prefix}.weights.h5", monitor='val_loss', mode='min',
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=f"{filepath_prefix}_logs", histogram_freq=0, write_graph=True,
                                    write_images=False, update_freq='epoch', profile_batch=0),
        keras.callbacks.CSVLogger(filename=f"{filepath_prefix}_history.csv", separator=',', append=False)
    ]


# ----------------------- Train function -----------------------
def train(x_train, y_train, x_val, y_val, model, history_len, save_prefix="", early_stopping_patience=patience):
    if len(model.input_shape) == 3:
        x_train = x_train.reshape(x_train.shape[0], history_len, 1)
        x_val = x_val.reshape(x_val.shape[0], history_len, 1)
    hist = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=max_epoch,
                     callbacks=make_callbacks(save_prefix, early_stopping_patience))
    return hist, model


# ----------------------- Per-fold JSON/MD writers -----------------------
def write_fold_json_and_md(exp_dir, base_name, loss_name, algo, ph, fold_idx, fold_started_at, train_sec, train_hms, pred_sec,
                           pred_hms, weights_path, hist_csv_path, best_json_path, preds_parquet_path,
                           training_plot_prefix, history_len, do_cv, min_sensor, max_sensor, data_file_path,
                           data_counts):
    payload = {
        "experiment": {"loss": loss_name, "algorithm": algo, "prediction_horizon": ph, "fold": fold_idx,
                       "history_length": history_len, "cross_validation": do_cv,
                       "sensor_limits": {"min": min_sensor, "max": max_sensor},
                       "started_at": fold_started_at.isoformat(), "run_id": GLOBAL_RUN_ID},
        "data": {"file": os.path.abspath(data_file_path), "counts": data_counts},
        "timing": {"train_seconds": round(train_sec, 3), "train_elapsed": str(train_hms),
                   "predict_seconds": round(pred_sec, 3), "predict_elapsed": str(pred_hms)},
        "artifacts": {
            "weights_path": weights_path, "history_csv_path": hist_csv_path, "best_json_path": best_json_path,
            "predictions_parquet_path": preds_parquet_path, "training_plot_prefix": training_plot_prefix
        },
        "git": get_git_info(),
        "system": {"hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
                   "python": platform.python_version(), "tensorflow": tf.__version__}
    }

    for key, val in payload["artifacts"].items():
        payload["artifacts"][key] = safe_relpath(val, start=exp_dir)

    json_name = os.path.join(exp_dir, "summaries", f"summary_{base_name}_Fold{fold_idx}.json")
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_name = os.path.join(exp_dir, "readmes", f"README_{base_name}_Fold{fold_idx}.md")
    with open(md_name, "w", encoding="utf-8") as f:
        f.write(f"# Fold {fold_idx} Summary for {loss_name}/{algo}/PH={ph}\n\n")
        f.write(f"**Data File**: `{os.path.abspath(data_file_path)}`\n\n")
        f.write("| Split | Patient Count | Window Count | % Patients | % Windows |\n")
        f.write("|-------|---------------|--------------|------------|-----------|\n")
        for split in ["train", "val", "test"]:
            f.write(
                f"| {split.capitalize()} | {data_counts['patients'][split]} | {data_counts['windows'][split]} | {data_counts['percent']['patients'][split]:.2f}% | {data_counts['percent']['windows'][split]:.2f}% |\n")
        f.write("\n## Timings\n")
        f.write(f"- **Training**: {train_hms}\n- **Prediction**: {pred_hms}\n\n")
        f.write("## Artifacts\n")
        for key, val in payload["artifacts"].items():
            f.write(f"- **{key.replace('_path', '').replace('_', ' ').capitalize()}**: `{val}`\n")


# ----------------------- Main README writer -----------------------
def write_experiment_readme(exp_dir, base_name, loss_name, algo, ph, history_len, k_folds):
    exp_name, git_info = os.path.basename(exp_dir), get_git_info()
    prod_prefix = f"{base_name}_PRODUCTION"
    
    lines = [f"# Experiment Summary: `{exp_name}`\n",
             f"- **Run ID**: `{GLOBAL_RUN_ID}`", f"- **Date**: `{RUN_TS.strftime('%Y-%m-%d %H:%M:%S')}`\n",
             "## 🧪 Experiment Configuration\n",
             f"- **Algorithm**: `{algo}`", f"- **Loss Function**: `{loss_name}`",
             f"- **Prediction Horizon (PH)**: `{ph}`", f"- **History Length (H)**: `{history_len}`",
             f"- **batch_size**: `{batch_size}`", f"- **patience**: `{patience}`", f"- **max_epoch**: `{max_epoch}`\n",
             "## 📊 Quick Links\n",
             "- [TensorBoard Logs](./models/)",
             f"- [Aggregated Predictions](./results/predictions/df_test_results_vectors_{base_name}_ALL.parquet)",
             "- [Aggregated Metrics](./results/evaluation/metrics/)",
             "- [Clarke Error Grid (CEG) Plots](./results/evaluation/figures/)",
             "- [Detailed Log File](./experiment_log.out)\n",
             "---", 
             "\n## 🚀 Production Model (Deployment-Ready)\n",
             "This model is trained on a dedicated dataset split guaranteeing zero data leakage.\n",
             f"- **Weights (.h5)**: `models/{prod_prefix}.weights.h5`\n",
             f"- **Best Epoch Metadata**: `models/{prod_prefix}.best.json`\n",
             f"- **Training History**: `models/{prod_prefix}_history.csv`\n",
             "---\n", f"\n## 📂 Fold Details ({k_folds} folds)\n"]
             
    for i in range(1, k_folds + 1):
        p = f"{base_name}_Fold{i}"
        lines.extend(
            [f"### Fold {i}\n", f"- **Weights**: `models/{p}.weights.h5`", f"- **History**: `models/{p}_history.csv`",
             f"- **Best Epoch**: `models/{p}.best.json`",
             f"- **Predictions**: `results/predictions/test_results_vectors_{p}.parquet`",
             f"- **Detailed Summary**: `readmes/README_{p}.md`\n"])
    with open(os.path.join(exp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------------- Main Loop -----------------------
try:
    sys.stdout = Tee(original_stdout)
    print("Starting script execution...")

    # Recorrido unificado de las parejas (historia, horizonte)
    for current_historic, current_PH in experiment_pairs:
        used_columns = [f"x{i}" for i in range(current_historic)] + ["y"]
        info_columns = ['patient_id', f"x{current_historic - 1}_time", f"x{current_historic - 1}_date"]

        for loss_function_name in loss_function_name_list:
            loss_function = loss_functions_dic[loss_function_name]
            tf.keras.backend.clear_session()
            np.random.seed(global_seed)
            tf.random.set_seed(global_seed)
            random.seed(global_seed)
            k_folds = 5 if do_cross_validation else 1

            for current_algorithm in algorithms:
                global RUN_TS, GLOBAL_RUN_ID
                RUN_TS = datetime.datetime.now()
                GLOBAL_RUN_ID = RUN_TS.strftime("%Y%m%d_%H%M%S")

                dataset_file_name = f"{dataset_name}_H{current_historic}_PH{current_PH}_kfold{k_folds}"
                dataset_file_name_nofold = f"{dataset_name}_H{current_historic}_PH{current_PH}"
                # SINGLE SOURCE OF TRUTH NAMES
                base_name = f"{dataset_file_name_nofold}-{current_algorithm}-{loss_function_name}"
                
                EXP_DIR = build_experiment_dir(base_path=experiment_base_path,
                                               dataset_name=dataset_file_name_nofold,
                                               algorithm_name=current_algorithm, loss_name=loss_function_name)
                exp_log_file = None
                try:
                    exp_log_file_path = os.path.join(EXP_DIR, "experiment_log.out")
                    exp_log_file = open(exp_log_file_path, "a", encoding="utf-8")
                    sys.stdout = sys.stderr = Tee(original_stdout, exp_log_file)

                    print(f"================\nStarting Experiment: {os.path.basename(EXP_DIR)}\n================")
                    print_system_info(RUN_TS)
                    print("-----------------------------------\n")

                    MODELS_DIR = os.path.join(EXP_DIR, "models")
                    PRED_DIR = os.path.join(EXP_DIR, "results", "predictions")
                    PLOTS_TRAIN_DIR = os.path.join(EXP_DIR, "plots", "training")
                    METRICS_DIR = os.path.join(EXP_DIR, "results", "evaluation", "metrics")
                    FIGURES_DIR = os.path.join(EXP_DIR, "results", "evaluation", "figures")

                    CENTRAL_EXP_MODELS_DIR = os.path.join(central_models_path, os.path.basename(EXP_DIR))
                    os.makedirs(CENTRAL_EXP_MODELS_DIR, exist_ok=True)

                    aggregated_results = []
                    aggregated_results_performance = []
                    aggregated_results_out_of_range = []
                    aggregated_results_number_of_points = []

                    for i in range(k_folds):
                        current_fold = i + 1
                        current_name = f"{base_name}_Fold{current_fold}"
                        data_file_path = f"{windows_path}{dataset_file_name}.parquet"

                        print(f"\n---- Loading data for Fold {current_fold}/{k_folds} from: {data_file_path} ----")
                        df = pd.read_parquet(data_file_path)

                        df_train = df[df[f'fold{i}'] == 'train']
                        patients_train_number = df_train['patient_id'].nunique()
                        df_train = df_train.reset_index(drop=True)[used_columns]

                        df_val = df[df[f'fold{i}'] == 'val']
                        patients_val_number = df_val['patient_id'].nunique()
                        df_val = df_val.reset_index(drop=True)[used_columns]

                        df_test = df[df[f'fold{i}'] == 'test']
                        patients_test_number = df_test['patient_id'].nunique()
                        df_test_info = df_test.reset_index(drop=True)[info_columns]
                        df_test = df_test.reset_index(drop=True)[used_columns]

                        total_patients = patients_train_number + patients_val_number + patients_test_number

                        data_counts = {
                            "patients": {"train": int(patients_train_number), "val": int(patients_val_number),
                                         "test": int(patients_test_number)},
                            "windows": {"train": len(df_train), "val": len(df_val), "test": len(df_test)},
                            "percent": {
                                "patients": {"train": (patients_train_number / total_patients * 100),
                                             "val": (patients_val_number / total_patients * 100),
                                             "test": (patients_test_number / total_patients * 100)},
                                "windows": {"train": (len(df_train) / len(df) * 100),
                                            "val": (len(df_val) / len(df) * 100),
                                            "test": (len(df_test) / len(df) * 100)}
                            }
                        }

                        print("\n-- Data Split --")
                        print(f"  Train: {patients_train_number} patients. Total windows: {len(df_train)}")
                        print(f"  Val:   {patients_val_number} patients. Total windows: {len(df_val)}")
                        print(f"  Test:  {patients_test_number} patients. Total windows: {len(df_test)}")

                        x_train, y_train = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1:].to_numpy()
                        x_val, y_val = df_val.iloc[:, :-1].to_numpy(), df_val.iloc[:, -1:].to_numpy()
                        x_test, y_test = df_test.iloc[:, :-1].to_numpy(), df_test.iloc[:, -1:].to_numpy()
                        del df, df_train, df_val, df_test  # Free memory

                        print(f'\nSTART training {current_algorithm} - H={current_historic} | PH={current_PH} - Fold={current_fold}')

                        # 1. Clean instantiation from architectures.py using the exact default metrics
                        model = get_compiled_model(
                            algo_name=current_algorithm,
                            history_length=current_historic,
                            loss_function=loss_function,
                            metrics_list=[keras.metrics.RootMeanSquaredError(), ClinicalPenaltyLoss()]
                        )

                        fold_started_at = datetime.datetime.now()
                        start_time = time.time()
                        save_prefix = os.path.join(MODELS_DIR, current_name)
                        hist, model_trained = train(x_train, y_train, x_val, y_val, model, current_historic, save_prefix)
                        train_time = time.time() - start_time
                        print(
                            f'END training {current_algorithm} - Fold={current_fold} (Time: {datetime.timedelta(seconds=train_time)})\n')

                        print('START prediction & saving artifacts...')
                        weights_file = f"{save_prefix}.weights.h5"

                        # 2. Instantiating model for prediction and loading the saved weights manually
                        model_load = get_compiled_model(
                            algo_name=current_algorithm,
                            history_length=current_historic,
                            loss_function=loss_function,
                            metrics_list=[keras.metrics.RootMeanSquaredError(), ClinicalPenaltyLoss()]
                        )
                        model_load.load_weights(weights_file)

                        start_pred_time = time.time()
                        if len(model_load.input_shape) == 3:
                            x_test = x_test.reshape(x_test.shape[0], current_historic, 1)
                        y_pred = model_load.predict(x_test)
                        pred_time = time.time() - start_pred_time

                        # Concat arrays and tracking indices to generate test vectors
                        df_res = pd.DataFrame({'y_test': y_test.ravel(), 'y_predict': y_pred.ravel()})
                        df_res = pd.concat([df_res, df_test_info], axis=1)
                        preds_path = os.path.join(PRED_DIR, f'test_results_vectors_{current_name}.parquet')
                        df_res.to_parquet(preds_path)
                        aggregated_results.append(df_res)

                        training_plot_prefix = os.path.join(PLOTS_TRAIN_DIR, current_name)
                        get_train_plots_loss(hist, training_plot_prefix)

                        hist_csv_path = f"{save_prefix}_history.csv"
                        best_json_path = f"{save_prefix}.best.json"

                        try:
                            df_hist = pd.read_csv(hist_csv_path)
                            best_idx = int(df_hist['val_loss'].idxmin() if 'val_loss' in df_hist.columns else df_hist[
                                'loss'].idxmin())
                            row = df_hist.iloc[best_idx].to_dict()
                            best_payload = {
                                "monitor": "val_loss" if 'val_loss' in df_hist.columns else "loss",
                                "epoch_idx": best_idx, "epoch_count": int(df_hist.shape[0]), "metrics": row,
                                "artifacts": {"weights_path": os.path.abspath(weights_file),
                                              "history_csv_path": os.path.abspath(hist_csv_path)},
                                "context": {"loss": loss_function_name, "algorithm": current_algorithm,
                                            "prediction_horizon": current_PH, "history_length": current_historic,
                                            "fold": current_fold, "run_id": GLOBAL_RUN_ID},
                                "created_at": datetime.datetime.now().isoformat()
                            }
                            with open(best_json_path, "w", encoding="utf-8") as f:
                                json.dump(best_payload, f, indent=2)
                        except Exception as e:
                            print(f"WARNING: could not write best.json for {current_name}: {e}")

                        write_fold_json_and_md(
                            EXP_DIR, base_name, loss_function_name, current_algorithm, current_PH, current_fold,
                            fold_started_at, train_time, datetime.timedelta(seconds=train_time),
                            pred_time, datetime.timedelta(seconds=pred_time),
                            weights_path=weights_file, hist_csv_path=hist_csv_path, best_json_path=best_json_path,
                            preds_parquet_path=preds_path, training_plot_prefix=training_plot_prefix,
                            history_len=current_historic, do_cv=do_cross_validation,
                            min_sensor=minimum_sensor_reading, max_sensor=maximum_sensor_reading,
                            data_file_path=data_file_path, data_counts=data_counts
                        )

                        # Duplicate weights and best.json metadata to the models repository
                        try:
                            if os.path.exists(weights_file):
                                shutil.copy(weights_file, CENTRAL_EXP_MODELS_DIR)
                            if os.path.exists(best_json_path):
                                shutil.copy(best_json_path, CENTRAL_EXP_MODELS_DIR)
                            print(
                                f"Successfully copied fold {current_fold} model artifacts to central hub: {CENTRAL_EXP_MODELS_DIR}")
                        except Exception as e:
                            print(f"WARNING: Could not copy model artifacts to central hub: {e}")

                        print(
                            f'END prediction & saving artifacts for Fold={current_fold} (Time: {datetime.timedelta(seconds=pred_time)})\n')

                    # ====================================================================
                    # PRODUCTION MODEL TRAINING (DEPLOYMENT MODEL)
                    # ====================================================================
                    print(f"\n====================================================================")
                    print(
                        f"🚀 STARTING PRODUCTION MODEL TRAINING: {current_algorithm} - H={current_historic} | PH={current_PH}")
                    print(f"====================================================================")

                    # 1. Load data specifically for the production model
                    print(f"Loading data for Production Model from: {data_file_path}")
                    df_prod = pd.read_parquet(data_file_path)

                    df_prod_train = df_prod[df_prod['production_split'] == 'train'].reset_index(drop=True)[
                        used_columns]
                    df_prod_val = df_prod[df_prod['production_split'] == 'val'].reset_index(drop=True)[used_columns]

                    x_prod_train, y_prod_train = df_prod_train.iloc[:, :-1].to_numpy(), df_prod_train.iloc[:,
                                                                                        -1:].to_numpy()
                    x_prod_val, y_prod_val = df_prod_val.iloc[:, :-1].to_numpy(), df_prod_val.iloc[:,
                                                                                  -1:].to_numpy()

                    # Free memory
                    del df_prod, df_prod_train, df_prod_val

                    # 2. Instantiate a clean model for production
                    prod_model = get_compiled_model(
                        algo_name=current_algorithm,
                        history_length=current_historic,
                        loss_function=loss_function,
                        metrics_list=[keras.metrics.RootMeanSquaredError(), ClinicalPenaltyLoss()]
                    )

                    # 3. Define paths and train
                    prod_name = f"{base_name}_PRODUCTION"
                    prod_save_prefix = os.path.join(MODELS_DIR, prod_name)

                    prod_start_time = time.time()
                    prod_hist, prod_model_trained = train(
                        x_prod_train, y_prod_train,
                        x_prod_val, y_prod_val,
                        prod_model, current_historic, prod_save_prefix
                    )
                    prod_train_time = time.time() - prod_start_time

                    print(f'END training PRODUCTION MODEL (Time: {datetime.timedelta(seconds=prod_train_time)})\n')

                    # 4. Save training plots
                    prod_training_plot_prefix = os.path.join(PLOTS_TRAIN_DIR, prod_name)
                    get_train_plots_loss(prod_hist, prod_training_plot_prefix)

                    # 5. Extract best epoch metrics and save metadata (best.json)
                    prod_weights_file = f"{prod_save_prefix}.weights.h5"
                    prod_hist_csv_path = f"{prod_save_prefix}_history.csv"
                    prod_best_json_path = f"{prod_save_prefix}.best.json"

                    try:
                        df_hist_prod = pd.read_csv(prod_hist_csv_path)
                        best_idx_prod = int(
                            df_hist_prod['val_loss'].idxmin() if 'val_loss' in df_hist_prod.columns else
                            df_hist_prod['loss'].idxmin())
                        row_prod = df_hist_prod.iloc[best_idx_prod].to_dict()

                        prod_best_payload = {
                            "monitor": "val_loss" if 'val_loss' in df_hist_prod.columns else "loss",
                            "epoch_idx": best_idx_prod,
                            "epoch_count": int(df_hist_prod.shape[0]),
                            "metrics": row_prod,
                            "artifacts": {
                                "weights_path": os.path.abspath(prod_weights_file),
                                "history_csv_path": os.path.abspath(prod_hist_csv_path)
                            },
                            "context": {
                                "loss": loss_function_name,
                                "algorithm": current_algorithm,
                                "prediction_horizon": current_PH,
                                "history_length": current_historic,
                                "fold": "PRODUCTION",
                                "run_id": GLOBAL_RUN_ID
                            },
                            "created_at": datetime.datetime.now().isoformat()
                        }
                        with open(prod_best_json_path, "w", encoding="utf-8") as f:
                            json.dump(prod_best_payload, f, indent=2)
                    except Exception as e:
                        print(f"WARNING: could not write best.json for {prod_name}: {e}")

                    # 6. Copy production artifacts to the central model hub
                    try:
                        if os.path.exists(prod_weights_file):
                            shutil.copy(prod_weights_file, CENTRAL_EXP_MODELS_DIR)
                        if os.path.exists(prod_best_json_path):
                            shutil.copy(prod_best_json_path, CENTRAL_EXP_MODELS_DIR)
                        print(
                            f"Successfully copied PRODUCTION model artifacts to central hub: {CENTRAL_EXP_MODELS_DIR}")
                    except Exception as e:
                        print(f"WARNING: Could not copy PRODUCTION model artifacts to central hub: {e}")

                    print("Production Model training and artifact saving completed successfully.\n")

                    # ------------------ Metric-Then-Aggregate approach ------------------
                    fold_number = 1
                    for df_current_result in aggregated_results:
                        print()
                        print('Agregating metrics for fold ', fold_number)

                        df_metrics_performance, df_number_of_points, df_out_of_limits = test_by_range(
                            df_result_vector=df_current_result,
                            grid_name=f'_{base_name}_Fold{fold_number}',
                            show_plot=False,
                            plot_dir=FIGURES_DIR,
                            minimum_sensor_reading=minimum_sensor_reading,
                            maximum_sensor_reading=maximum_sensor_reading
                        )
                        df_metrics_performance.round(2).to_csv(os.path.join(METRICS_DIR,
                                                                            f'metrics_performance_{base_name}_Fold{fold_number}.csv'),
                                                               index=False)
                        df_metrics_performance['Fold'] = fold_number

                        df_number_of_points.to_csv(os.path.join(METRICS_DIR,
                                                                f'number_of_points_by_zones_{base_name}_Fold{fold_number}.csv'),
                                                   index=False)
                        df_number_of_points['Fold'] = fold_number

                        df_out_of_limits.to_csv(os.path.join(METRICS_DIR,
                                                             f'predictions_out_of_limits_{base_name}_Fold{fold_number}.csv'),
                                                index=False)
                        df_out_of_limits['Fold'] = fold_number

                        aggregated_results_performance.append(df_metrics_performance)
                        aggregated_results_number_of_points.append(df_number_of_points)
                        aggregated_results_out_of_range.append(df_out_of_limits)

                        print()
                        print('Aggregated metrics for fold ', fold_number, ' done.')
                        fold_number += 1

                    print()
                    print('Calculating aggregated metrics across folds (Metric-Then-Aggregate)...')

                    df_metrics_performance_agg = pd.concat(aggregated_results_performance, ignore_index=True)
                    df_metrics_performance_agg.round(2).to_csv(os.path.join(METRICS_DIR,
                                                                            f'metrics_performance_{base_name}_ALL_metric_then_aggregate.csv'),
                                                               index=False)
                    df_metrics_performance_agg = df_metrics_performance_agg.drop(columns=['Fold'])
                    df_metrics_performance_agg_flat, df_metrics_performance_agg_report = fold_results_aggregation(
                        df_metrics_performance_agg)
                    df_metrics_performance_agg_flat.round(2).to_csv(os.path.join(METRICS_DIR,
                                                                                 f'metrics_performance_{base_name}_metric_then_aggregate_flat.csv'),
                                                                    index=False)
                    df_metrics_performance_agg_report.round(2).to_csv(os.path.join(METRICS_DIR,
                                                                                   f'metrics_performance_{base_name}_metric_then_aggregate_report.csv'),
                                                                      index=False)

                    df_number_of_points_agg = pd.concat(aggregated_results_number_of_points, ignore_index=True)
                    df_number_of_points_agg.to_csv(os.path.join(METRICS_DIR,
                                                                f'number_of_points_by_zones_{base_name}_ALL_metric_then_aggregate.csv'),
                                                   index=False)
                    df_number_of_points_agg = df_number_of_points_agg.drop(columns=['Fold'])
                    df_number_of_points_agg_flat, df_number_of_points_report = fold_results_aggregation(
                        df_number_of_points_agg)
                    df_number_of_points_agg_flat.to_csv(os.path.join(METRICS_DIR,
                                                                     f'number_of_points_by_zones_{base_name}_metric_then_aggregate_flat.csv'),
                                                        index=False)
                    df_number_of_points_report.to_csv(os.path.join(METRICS_DIR,
                                                                   f'number_of_points_by_zones_{base_name}_metric_then_aggregate_report.csv'),
                                                      index=False)

                    df_out_of_range_agg = pd.concat(aggregated_results_out_of_range, ignore_index=True)
                    df_out_of_range_agg.to_csv(os.path.join(METRICS_DIR,
                                                            f'predictions_out_of_limits_{base_name}_ALL_metric_then_aggregate.csv'),
                                               index=False)
                    df_out_of_range_agg = df_out_of_range_agg.drop(columns=['Fold'])
                    df_out_of_range_agg_flat, df_out_of_range_report = fold_results_aggregation(df_out_of_range_agg)
                    df_out_of_range_agg_flat.to_csv(os.path.join(METRICS_DIR,
                                                                 f'predictions_out_of_limits_{base_name}_metric_then_aggregate_flat.csv'),
                                                    index=False)
                    df_out_of_range_report.to_csv(os.path.join(METRICS_DIR,
                                                               f'predictions_out_of_limits_{base_name}_metric_then_aggregate_report.csv'),
                                                  index=False)

                    print('Aggregated metrics across folds (Metric-Then-Aggregate) calculated and saved.')
                    print()

                    # ------------------ Aggregate-Then-Metric approach ------------------
                    print()
                    print('Calculating aggregated metrics across folds (Aggregate-Then-Metric)...')

                    df_all_folds = pd.concat(aggregated_results, ignore_index=True)
                    df_all_folds.to_parquet(os.path.join(PRED_DIR,
                                                         f'df_test_results_vectors_{base_name}_ALL.parquet'))
                    print(f'\nAggregated all {k_folds} folds. Total rows: {len(df_all_folds)}')

                    df_metrics_performance, df_number_of_points, df_out_of_limits = test_by_range(df_all_folds,
                                                                                                  f'_{base_name}_ALL',
                                                                                                  show_plot=True,
                                                                                                  plot_dir=FIGURES_DIR,
                                                                                                  minimum_sensor_reading=minimum_sensor_reading,
                                                                                                  maximum_sensor_reading=maximum_sensor_reading)
                    df_metrics_performance.round(2).to_csv(os.path.join(METRICS_DIR,
                                                                        f'metrics_performance_{base_name}_ALL_agregate_then_metric.csv'),
                                                           index=False)
                    df_number_of_points.to_csv(os.path.join(METRICS_DIR,
                                                            f'number_of_points_by_zones_{base_name}_ALL_agregate_then_metric.csv'),
                                               index=False)
                    df_out_of_limits.to_csv(os.path.join(METRICS_DIR,
                                                         f'predictions_out_of_limits_{base_name}_ALL_agregate_then_metric.csv'),
                                            index=False)
                    print('Metrics for all folds calculated and saved.')

                    write_experiment_readme(EXP_DIR, base_name, loss_function_name, current_algorithm, current_PH,
                                            current_historic, k_folds)
                    print(f"Main README.md created at: {os.path.join(EXP_DIR, 'README.md')}")

                    experiment_end_time = datetime.datetime.now()
                    duration = experiment_end_time - RUN_TS

                    timing_payload = {
                        "run_id": GLOBAL_RUN_ID,
                        "algorithm": current_algorithm,
                        "loss_function": loss_function_name,
                        "history_length": current_historic,
                        "prediction_horizon": current_PH,
                        "start_time": RUN_TS.isoformat(),
                        "end_time": experiment_end_time.isoformat(),
                        "total_duration_str": str(duration),
                        "total_duration_seconds": round(duration.total_seconds(), 3)
                    }

                    timing_json_path = os.path.join(EXP_DIR, "experiment_timing.json")
                    with open(timing_json_path, "w", encoding="utf-8") as f:
                        json.dump(timing_payload, f, indent=4)

                    print(f"Timing JSON saved at: {timing_json_path}")

                    print(f"================\nFinished Experiment: {os.path.basename(EXP_DIR)}")
                    print(f"Experiment End Time: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Total Duration: {duration}\n================\n\n")

                finally:
                    if exp_log_file:
                        exp_log_file.close()
                    sys.stdout = sys.stderr = Tee(original_stdout)

finally:
    print("All experiments finished.")
    sys.stdout = original_stdout
    sys.stderr = original_stderr