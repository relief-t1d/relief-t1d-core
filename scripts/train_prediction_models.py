# -*- coding: utf-8 -*-
"""
RELIEF-T1D: Model Training and Evaluation Pipeline

This script orchestrates the end-to-end training process for blood glucose
prediction models. It handles the loading of preprocessed temporal windows,
model instantiation (e.g., LSTMs, CNNs), k-fold cross-validation, and
comprehensive evaluation using both statistical and clinical metrics.

If the windows/folds file for a given (H, PH) pair does not exist, it is
generated on-the-fly through the dataset_builder API using the 'dataset',
'folds' and 'paths' sections of the configuration file.

All experiment hyperparameters, sensor limits, and paths are governed by
an external YAML configuration file for full reproducibility.
"""

import os
import re
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

# Identifiers taken from the configuration become fields of the experiment directory name, where '_'
# separates fields. Inside an identifier only letters, digits, '.' and '-' are allowed (e.g. 'REPLACE-BG-v1.0').
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9.-]+")

def validate_identifiers(names, config_key):
    """
    Fails fast if any configured identifier cannot be used as a field of the experiment directory name.

    Args:
        names (list): Identifiers read from the configuration.
        config_key (str): YAML key the identifiers come from, used in the error message.

    Raises:
        ValueError: If any identifier is not a string matching IDENTIFIER_PATTERN.
    """
    invalid = [name for name in names if not isinstance(name, str) or not IDENTIFIER_PATTERN.fullmatch(name)]
    if invalid:
        raise ValueError(
            f"Invalid value(s) in '{config_key}': {invalid}. Only letters, digits, '.' and '-' are allowed "
            "('_' is reserved as the field separator of the experiment directory name), e.g. 'REPLACE-BG-v1.0'."
        )

# Load the configuration file and dynamically expand environment variables (e.g., ${PATH_TO_DATA})
cli_args = parse_arguments()
with open(cli_args.config, 'r', encoding='utf-8') as f:
    raw_yaml_content = f.read()
    expanded_yaml_content = os.path.expandvars(raw_yaml_content)
    config = yaml.safe_load(expanded_yaml_content)

# Fail fast if any environment variable could not be resolved (os.path.expandvars leaves them untouched)
unresolved_env_vars = sorted(set(re.findall(r"\$\{[^}]+\}", expanded_yaml_content)))
if unresolved_env_vars:
    raise EnvironmentError(
        f"Unresolved environment variables in '{cli_args.config}': {unresolved_env_vars}. "
        "Check that they are defined in the .env file or in the system environment."
    )

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
import psutil
import subprocess
import pandas as pd
import shutil
from typing import Optional

# ---------------------------------------------------------
# 4. IMPORTS FROM INTERNAL MODULAR LIBRARY (src/)
# ---------------------------------------------------------
from src.models.losses import *
from src.models.architectures import get_compiled_model
from src.evaluation.metrics import test_by_range, get_train_plots_loss, fold_results_aggregation
from src.preprocessing.dataset_builder import build_single_dataset

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

# Extracting dataset configuration
dataset_name = config['dataset']['name']
glucose_column_name = config['dataset']['glucose_column']
id_column_name = config['dataset']['id_column']

# Extracting fold generation configuration
k_folds = config['folds']['k_folds']
folds_random_seed = config['folds']['random_seed']
deduplicate_intra_patient = not config['folds']['disable_deduplication']

if k_folds < 2:
    raise ValueError(f"folds.k_folds must be >= 2 (got {k_folds}). "
                     "To run without cross-validation set experiment.do_cross_validation to false.")

# Extracting experiment configuration
loss_function_name_list = config['experiment']['loss_functions']
algorithms = config['experiment']['algorithms']
do_cross_validation = config['experiment']['do_cross_validation']

# Fail fast, before any dataset generation or training, on identifiers that would break the directory name
validate_identifiers([dataset_name], "dataset.name")
validate_identifiers(algorithms, "experiment.algorithms")
validate_identifiers(loss_function_name_list, "experiment.loss_functions")

# Number of folds actually trained and evaluated:
#   - Cross-validation: every fold of the k-fold dataset.
#   - No cross-validation (hold-out): only fold0 of the same k-fold dataset.
n_folds_to_train = k_folds if do_cross_validation else 1

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
source_data_path = config['paths']['path_to_load']
windows_path = config['paths']['windows_path']
experiment_base_path = config['paths']['experiment_base_path']
central_models_path = config['paths']['central_models_path']

# ---------------------------------------------------------
# 6. METADATA AND LOGGING VARIABLES (README)
# ---------------------------------------------------------
dataset_link = config['metadata']['link']

training_data = config['metadata']['training_data']
validation_data = config['metadata']['validation_data']
testing_data = config['metadata']['testing_data']
wise_level = config['metadata']['wise_level']

# Dynamic variables based on experiment setup
cross_validation = 'k-fold' if do_cross_validation else 'No (hold-out on fold0)'
Folds = str(n_folds_to_train)

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

original_stdout = sys.stdout
original_stderr = sys.stderr

# ----------------------- System & Git helpers -----------------------
def print_system_info(experiment_start_time):
    print("📅 Date & Time")
    print("-----------------------------------")
    print(f"Experiment Start Time: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
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
# ISO 8601 basic format with second resolution (e.g. 20261028T080012). Used as experiment directory prefix
# and as run ID.
RUN_TS_FORMAT = "%Y%m%dT%H%M%S"

EXPERIMENT_SUBDIRS = ["models", "plots/training", "results/predictions", "results/evaluation/metrics",
                      "results/evaluation/figures", "summaries", "readmes"]


def safe_relpath(p, start=None):
    try:
        return os.path.relpath(p, start=start) if start else os.path.relpath(p)
    except Exception:
        return p


# ----------------------- Artifact naming -----------------------
# The experiment directory identifies the run (dataset, H, PH, algorithm, loss), so artifact names inside it
# only describe what each file is. Naming rule (lowercase snake_case):
#   - Per-fold artifacts: {type}_fold_{k}.{ext}, where k is 0-based and matches the 'fold{k}' dataset column.
#   - Global artifacts:   {type}.{ext}, i.e. the production model in models/ and plots/training/, and the
#                         aggregation over all evaluated folds in results/.
def artifact_suffix(fold_idx: Optional[int] = None) -> str:
    """
    Returns the artifact name suffix for a fold, or an empty suffix for a global artifact.

    Args:
        fold_idx (Optional[int]): 0-based fold index, or None for a global artifact.

    Returns:
        str: '_fold_{fold_idx}' for a fold, '' otherwise.
    """
    return "" if fold_idx is None else f"_fold_{fold_idx}"


def model_artifact_paths(models_dir: str, plots_dir: str, fold_idx: Optional[int] = None) -> dict:
    """
    Builds the paths of the training artifacts of one model (fold model or production model).

    Args:
        models_dir (str): Directory of the model artifacts (weights, best epoch, history, TensorBoard logs).
        plots_dir (str): Directory of the training plots.
        fold_idx (Optional[int]): 0-based fold index, or None for the production model.

    Returns:
        dict: Paths keyed by 'weights', 'best_epoch', 'history', 'tensorboard' and 'loss_plot'.
    """
    suffix = artifact_suffix(fold_idx)
    return {
        # Keras 3 requires weights-only files to end with '.weights.h5'
        "weights": os.path.join(models_dir, f"model{suffix}.weights.h5"),
        "best_epoch": os.path.join(models_dir, f"best_epoch{suffix}.json"),
        "history": os.path.join(models_dir, f"history{suffix}.csv"),
        "tensorboard": os.path.join(models_dir, f"tensorboard{suffix}"),
        "loss_plot": os.path.join(plots_dir, f"loss{suffix}.pdf"),
    }


def evaluation_table_path(metrics_dir: str, table_name: str, suffix: str = "") -> str:
    """
    Builds the path of an evaluation table.

    Args:
        metrics_dir (str): Directory of the evaluation tables.
        table_name (str): Table type: 'performance_by_range', 'ceg_zone_counts' or 'prediction_limits'.
        suffix (str): '_fold_{k}' for one fold, or the aggregation scope: '_per_fold' (fold tables stacked),
            '_metric_then_aggregate', '_metric_then_aggregate_report' or '_aggregate_then_metric'.

    Returns:
        str: Path of the CSV file.
    """
    return os.path.join(metrics_dir, f"{table_name}{suffix}.csv")


def build_experiment_dir_name(run_ts: datetime.datetime, dataset: str, history_len: int, ph: int,
                              algorithm_name: str, loss_name: str) -> str:
    """
    Builds the experiment directory name: '{timestamp}_pred_{dataset}_H{h}_PH{ph}_{algorithm}_{loss}'.

    Fields are separated by '_'; identifiers only contain letters, digits, '.' and '-' (see
    validate_identifiers), so the name can be split on '_' unambiguously.

    Args:
        run_ts (datetime.datetime): Start time of the experiment.
        dataset (str): Dataset name (e.g. 'REPLACE-BG-v1.0').
        history_len (int): History length (H).
        ph (int): Prediction horizon (PH).
        algorithm_name (str): Algorithm name (e.g. 'Linear').
        loss_name (str): Loss function name (e.g. 'RMSE').

    Returns:
        str: Experiment directory name (e.g. '20261028T080012_pred_REPLACE-BG-v1.0_H8_PH4_Linear_RMSE').
    """
    return (f"{run_ts.strftime(RUN_TS_FORMAT)}_pred_{dataset}_H{history_len}_PH{ph}"
            f"_{algorithm_name}_{loss_name}")


def create_experiment_dirs(base_path: str, central_path: str, exp_folder_name: str) -> tuple[str, str]:
    """
    Creates the experiment directory (with its subdirectories) and its folder in the central models hub.

    Both must be new: artifact names inside them are generic, so reusing an existing directory would
    silently overwrite the artifacts of another run.

    Args:
        base_path (str): Base directory of the experiments.
        central_path (str): Base directory of the central models hub.
        exp_folder_name (str): Experiment directory name (see build_experiment_dir_name).

    Returns:
        tuple[str, str]: Experiment directory and its central models hub directory.

    Raises:
        FileExistsError: If either directory already exists. Nothing is left behind in that case.
    """
    exp_dir = os.path.join(base_path, exp_folder_name)
    hub_dir = os.path.join(central_path, exp_folder_name)

    # os.makedirs with exist_ok=False is atomic for the last component, so concurrent runs cannot share a folder
    try:
        os.makedirs(exp_dir, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Experiment directory already exists: {exp_dir}. "
                              "Another run with the same configuration started in the same second.") from None
    try:
        os.makedirs(hub_dir, exist_ok=False)
    except FileExistsError:
        os.rmdir(exp_dir)  # Still empty: remove it so that the failed run leaves no trace
        raise FileExistsError(f"Central models hub directory already exists: {hub_dir}. "
                              "Remove it or check which run produced it.") from None

    for sub in EXPERIMENT_SUBDIRS:
        os.makedirs(os.path.join(exp_dir, sub), exist_ok=False)
    return exp_dir, hub_dir


# ----------------------- Keras callbacks -----------------------
def make_callbacks(artifact_paths: dict, early_stopping_patience: int):
    return [
        keras.callbacks.ModelCheckpoint(filepath=artifact_paths["weights"], monitor='val_loss', mode='min',
                                        save_best_only=True, save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=artifact_paths["tensorboard"], histogram_freq=0, write_graph=True,
                                    write_images=False, update_freq='epoch', profile_batch=0),
        keras.callbacks.CSVLogger(filename=artifact_paths["history"], separator=',', append=False)
    ]


# ----------------------- Train function -----------------------
def train(x_train, y_train, x_val, y_val, model, history_len, artifact_paths, early_stopping_patience=patience):
    if len(model.input_shape) == 3:
        x_train = x_train.reshape(x_train.shape[0], history_len, 1)
        x_val = x_val.reshape(x_val.shape[0], history_len, 1)
    hist = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_val, y_val), epochs=max_epoch,
                     callbacks=make_callbacks(artifact_paths, early_stopping_patience))
    return hist, model


# ----------------------- Best epoch metadata -----------------------
def write_best_epoch_json(artifact_paths: dict, context: dict) -> None:
    """
    Writes the best-epoch metadata of a trained model, read from its CSV training history.

    Args:
        artifact_paths (dict): Model artifact paths returned by `model_artifact_paths`.
        context (dict): Identification of the model (model type, fold, loss, algorithm, H, PH, run ID).
            It is stored in the file because the generic file name does not identify the model.
    """
    try:
        df_hist = pd.read_csv(artifact_paths["history"])
        monitor = "val_loss" if 'val_loss' in df_hist.columns else "loss"
        best_idx = int(df_hist[monitor].idxmin())
        best_payload = {
            "monitor": monitor,
            "epoch_idx": best_idx,
            "epoch_count": int(df_hist.shape[0]),
            "metrics": df_hist.iloc[best_idx].to_dict(),
            "artifacts": {
                "weights_path": os.path.abspath(artifact_paths["weights"]),
                "history_csv_path": os.path.abspath(artifact_paths["history"])
            },
            "context": context,
            "created_at": datetime.datetime.now().isoformat()
        }
        with open(artifact_paths["best_epoch"], "w", encoding="utf-8") as f:
            json.dump(best_payload, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not write {artifact_paths['best_epoch']}: {e}")


# ----------------------- Per-fold JSON/MD writers -----------------------
def write_fold_json_and_md(exp_dir, loss_name, algo, ph, fold_idx, fold_started_at, train_sec, train_hms, pred_sec,
                           pred_hms, artifact_paths, preds_parquet_path, history_len, do_cv, min_sensor, max_sensor,
                           data_file_path, data_counts):
    payload = {
        "experiment": {"loss": loss_name, "algorithm": algo, "prediction_horizon": ph, "fold": fold_idx,
                       "history_length": history_len, "cross_validation": do_cv,
                       "sensor_limits": {"min": min_sensor, "max": max_sensor},
                       "started_at": fold_started_at.isoformat(), "run_id": GLOBAL_RUN_ID},
        "data": {"file": os.path.abspath(data_file_path), "counts": data_counts},
        "timing": {"train_seconds": round(train_sec, 3), "train_elapsed": str(train_hms),
                   "predict_seconds": round(pred_sec, 3), "predict_elapsed": str(pred_hms)},
        "artifacts": {
            "weights_path": artifact_paths["weights"], "history_csv_path": artifact_paths["history"],
            "best_epoch_json_path": artifact_paths["best_epoch"], "tensorboard_dir": artifact_paths["tensorboard"],
            "predictions_parquet_path": preds_parquet_path, "training_plot_path": artifact_paths["loss_plot"]
        },
        "git": get_git_info(),
        "system": {"hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
                   "python": platform.python_version(), "tensorflow": tf.__version__}
    }

    for key, val in payload["artifacts"].items():
        payload["artifacts"][key] = safe_relpath(val, start=exp_dir)

    json_name = os.path.join(exp_dir, "summaries", f"summary{artifact_suffix(fold_idx)}.json")
    with open(json_name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_name = os.path.join(exp_dir, "readmes", f"readme{artifact_suffix(fold_idx)}.md")
    with open(md_name, "w", encoding="utf-8") as f:
        f.write(f"# Fold {fold_idx} Summary for {loss_name}/{algo}/H={history_len}/PH={ph}\n\n")
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
def write_experiment_readme(exp_dir, loss_name, algo, ph, history_len, n_folds):
    exp_name, git_info = os.path.basename(exp_dir), get_git_info()

    lines = [f"# Experiment Summary: `{exp_name}`\n",
             f"- **Run ID**: `{GLOBAL_RUN_ID}`", f"- **Date**: `{RUN_TS.strftime('%Y-%m-%d %H:%M:%S')}`\n",
             "## 🧪 Experiment Configuration\n",
             f"- **Algorithm**: `{algo}`", f"- **Loss Function**: `{loss_name}`",
             f"- **Prediction Horizon (PH)**: `{ph}`", f"- **History Length (H)**: `{history_len}`",
             f"- **Cross-validation**: `{cross_validation}`",
             f"- **Folds trained**: `{n_folds}` of `{k_folds}`",
             f"- **batch_size**: `{batch_size}`", f"- **patience**: `{patience}`", f"- **max_epoch**: `{max_epoch}`\n",
             "## 📊 Quick Links\n",
             "- [TensorBoard Logs](./models/)",
             "- [Aggregated Predictions](./results/predictions/predictions.parquet)",
             "- [Aggregated Metrics](./results/evaluation/metrics/)",
             "- [Clarke Error Grid (CEG) Plots](./results/evaluation/figures/)",
             "- [Detailed Log File](./experiment.log)\n",
             "## 🗂️ Artifact Naming\n",
             "- `{type}_fold_{k}.{ext}`: artifact of fold `k` (0-based; dataset column `fold{k}`).",
             "- `{type}.{ext}` (no suffix): production model in `models/` and `plots/training/`; "
             "aggregation over all evaluated folds in `results/`.\n",
             "---",
             "\n## 🚀 Production Model (Deployment-Ready)\n",
             "This model is trained on a dedicated dataset split guaranteeing zero data leakage.\n",
             "- **Weights (.h5)**: `models/model.weights.h5`\n",
             "- **Best Epoch Metadata**: `models/best_epoch.json`\n",
             "- **Training History**: `models/history.csv`\n",
             "- **Training Curve**: `plots/training/loss.pdf`\n",
             "---\n", f"\n## 📂 Fold Details ({n_folds} folds)\n"]

    for fold_idx in range(n_folds):
        s = artifact_suffix(fold_idx)
        lines.extend(
            [f"### Fold {fold_idx}\n", f"- **Weights**: `models/model{s}.weights.h5`",
             f"- **History**: `models/history{s}.csv`",
             f"- **Best Epoch**: `models/best_epoch{s}.json`",
             f"- **Predictions**: `results/predictions/predictions{s}.parquet`",
             f"- **Detailed Summary**: `readmes/readme{s}.md`\n"])
    with open(os.path.join(exp_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ----------------------- Main Loop -----------------------
try:
    sys.stdout = Tee(original_stdout)
    print("Starting script execution...")
    print(f"Evaluation mode: {cross_validation} | Folds to train: {n_folds_to_train} of {k_folds}")

    # Unified iteration over (history, horizon) pairs
    for current_historic, current_PH in experiment_pairs:

        # SINGLE SOURCE OF TRUTH for dataset file names and paths
        dataset_file_name = f"{dataset_name}_H{current_historic}_PH{current_PH}_kfold{k_folds}"
        data_file_path = os.path.join(windows_path, f"{dataset_file_name}.parquet")

        # Check if the dataset file already exists; generate on-the-fly via dataset_builder if missing
        if not os.path.exists(data_file_path):
            print(f"\n[INFO] Dataset artifact not found: {data_file_path}")
            print(f"[INFO] Invoking dataset_builder API to generate H={current_historic}, PH={current_PH}...")

            build_single_dataset(
                source_data_path=source_data_path,
                dataset_name=dataset_name,
                glucose_column_name=glucose_column_name,
                id_column_name=id_column_name,
                historic=current_historic,
                ph=current_PH,
                deduplicate_intra_patient=deduplicate_intra_patient,
                save_pure_windows=False,
                output_pure_windows_dir=None,
                generate_folds=True,
                output_folds_dir=windows_path,
                k_folds=k_folds,
                random_seed=folds_random_seed
            )

            if not os.path.exists(data_file_path):
                raise FileNotFoundError(
                    f"Dataset generation finished but the expected file was not found: {data_file_path}. "
                    "Check that 'paths.windows_path' ends with a path separator."
                )
            print(f"[INFO] Dataset generation completed successfully.\n")

        used_columns = [f"x{i}" for i in range(current_historic)] + ["y"]
        info_columns = ['patient_id', f"x{current_historic - 1}_time", f"x{current_historic - 1}_date"]

        for loss_function_name in loss_function_name_list:
            loss_function = loss_functions_dic[loss_function_name]
            tf.keras.backend.clear_session()
            np.random.seed(global_seed)
            tf.random.set_seed(global_seed)
            random.seed(global_seed)

            for current_algorithm in algorithms:
                global RUN_TS, GLOBAL_RUN_ID
                RUN_TS = datetime.datetime.now()
                GLOBAL_RUN_ID = RUN_TS.strftime(RUN_TS_FORMAT)

                exp_folder_name = build_experiment_dir_name(RUN_TS, dataset_name, current_historic, current_PH,
                                                            current_algorithm, loss_function_name)
                EXP_DIR, CENTRAL_EXP_MODELS_DIR = create_experiment_dirs(experiment_base_path, central_models_path,
                                                                         exp_folder_name)
                exp_log_file = None
                try:
                    exp_log_file_path = os.path.join(EXP_DIR, "experiment.log")
                    exp_log_file = open(exp_log_file_path, "a", encoding="utf-8")
                    sys.stdout = sys.stderr = Tee(original_stdout, exp_log_file)

                    print(f"================\nStarting Experiment: {os.path.basename(EXP_DIR)}\n================")
                    print(f"Evaluation mode: {cross_validation} | Folds to train: {n_folds_to_train} of {k_folds}")
                    print_system_info(RUN_TS)
                    print("-----------------------------------\n")

                    MODELS_DIR = os.path.join(EXP_DIR, "models")
                    PRED_DIR = os.path.join(EXP_DIR, "results", "predictions")
                    PLOTS_TRAIN_DIR = os.path.join(EXP_DIR, "plots", "training")
                    METRICS_DIR = os.path.join(EXP_DIR, "results", "evaluation", "metrics")
                    FIGURES_DIR = os.path.join(EXP_DIR, "results", "evaluation", "figures")

                    aggregated_results = []
                    aggregated_results_performance = []
                    aggregated_results_out_of_range = []
                    aggregated_results_number_of_points = []

                    # Folds are 0-based everywhere: fold_idx k is the dataset column 'fold{k}' and the
                    # '_fold_{k}' suffix of every artifact of that fold.
                    for fold_idx in range(n_folds_to_train):
                        fold_paths = model_artifact_paths(MODELS_DIR, PLOTS_TRAIN_DIR, fold_idx)

                        print(f"\n---- Loading data for Fold {fold_idx} [{fold_idx + 1}/{n_folds_to_train}] "
                              f"from: {data_file_path} ----")
                        df = pd.read_parquet(data_file_path)

                        df_train = df[df[f'fold{fold_idx}'] == 'train']
                        patients_train_number = df_train['patient_id'].nunique()
                        df_train = df_train.reset_index(drop=True)[used_columns]

                        df_val = df[df[f'fold{fold_idx}'] == 'val']
                        patients_val_number = df_val['patient_id'].nunique()
                        df_val = df_val.reset_index(drop=True)[used_columns]

                        df_test = df[df[f'fold{fold_idx}'] == 'test']
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

                        print(f'\nSTART training {current_algorithm} - H={current_historic} | PH={current_PH} - Fold={fold_idx}')

                        # 1. Clean instantiation from architectures.py using the exact default metrics
                        model = get_compiled_model(
                            algo_name=current_algorithm,
                            history_length=current_historic,
                            loss_function=loss_function,
                            metrics_list=[keras.metrics.RootMeanSquaredError(), ClinicalPenaltyLoss()]
                        )

                        fold_started_at = datetime.datetime.now()
                        start_time = time.time()
                        hist, model_trained = train(x_train, y_train, x_val, y_val, model, current_historic, fold_paths)
                        train_time = time.time() - start_time
                        print(
                            f'END training {current_algorithm} - Fold={fold_idx} (Time: {datetime.timedelta(seconds=train_time)})\n')

                        print('START prediction & saving artifacts...')
                        weights_file = fold_paths["weights"]

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
                        preds_path = os.path.join(PRED_DIR, f'predictions{artifact_suffix(fold_idx)}.parquet')
                        df_res.to_parquet(preds_path)
                        aggregated_results.append(df_res)

                        get_train_plots_loss(hist, fold_paths["loss_plot"])

                        write_best_epoch_json(fold_paths, context={
                            "model_type": "fold", "fold": fold_idx, "loss": loss_function_name,
                            "algorithm": current_algorithm, "prediction_horizon": current_PH,
                            "history_length": current_historic, "run_id": GLOBAL_RUN_ID
                        })

                        write_fold_json_and_md(
                            EXP_DIR, loss_function_name, current_algorithm, current_PH, fold_idx,
                            fold_started_at, train_time, datetime.timedelta(seconds=train_time),
                            pred_time, datetime.timedelta(seconds=pred_time),
                            artifact_paths=fold_paths, preds_parquet_path=preds_path,
                            history_len=current_historic, do_cv=do_cross_validation,
                            min_sensor=minimum_sensor_reading, max_sensor=maximum_sensor_reading,
                            data_file_path=data_file_path, data_counts=data_counts
                        )

                        # Duplicate weights and best epoch metadata to the models repository
                        try:
                            if os.path.exists(fold_paths["weights"]):
                                shutil.copy(fold_paths["weights"], CENTRAL_EXP_MODELS_DIR)
                            if os.path.exists(fold_paths["best_epoch"]):
                                shutil.copy(fold_paths["best_epoch"], CENTRAL_EXP_MODELS_DIR)
                            print(
                                f"Successfully copied fold {fold_idx} model artifacts to central hub: {CENTRAL_EXP_MODELS_DIR}")
                        except Exception as e:
                            print(f"WARNING: Could not copy model artifacts to central hub: {e}")

                        print(
                            f'END prediction & saving artifacts for Fold={fold_idx} (Time: {datetime.timedelta(seconds=pred_time)})\n')

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

                    # 3. Define paths and train (production artifacts have no suffix)
                    prod_paths = model_artifact_paths(MODELS_DIR, PLOTS_TRAIN_DIR)

                    prod_start_time = time.time()
                    prod_hist, prod_model_trained = train(
                        x_prod_train, y_prod_train,
                        x_prod_val, y_prod_val,
                        prod_model, current_historic, prod_paths
                    )
                    prod_train_time = time.time() - prod_start_time

                    print(f'END training PRODUCTION MODEL (Time: {datetime.timedelta(seconds=prod_train_time)})\n')

                    # 4. Save training plots
                    get_train_plots_loss(prod_hist, prod_paths["loss_plot"])

                    # 5. Extract best epoch metrics and save metadata
                    write_best_epoch_json(prod_paths, context={
                        "model_type": "production", "fold": None, "loss": loss_function_name,
                        "algorithm": current_algorithm, "prediction_horizon": current_PH,
                        "history_length": current_historic, "run_id": GLOBAL_RUN_ID
                    })

                    # 6. Copy production artifacts to the central model hub
                    try:
                        if os.path.exists(prod_paths["weights"]):
                            shutil.copy(prod_paths["weights"], CENTRAL_EXP_MODELS_DIR)
                        if os.path.exists(prod_paths["best_epoch"]):
                            shutil.copy(prod_paths["best_epoch"], CENTRAL_EXP_MODELS_DIR)
                        print(
                            f"Successfully copied PRODUCTION model artifacts to central hub: {CENTRAL_EXP_MODELS_DIR}")
                    except Exception as e:
                        print(f"WARNING: Could not copy PRODUCTION model artifacts to central hub: {e}")

                    print("Production Model training and artifact saving completed successfully.\n")

                    # ------------------ Metric-Then-Aggregate approach ------------------
                    # aggregated_results is filled in fold order, so its position is the 0-based fold index
                    for fold_idx, df_current_result in enumerate(aggregated_results):
                        print()
                        print(f'Aggregating metrics for fold {fold_idx}')
                        fold_suffix = artifact_suffix(fold_idx)

                        df_metrics_performance, df_number_of_points, df_out_of_limits = test_by_range(
                            df_result_vector=df_current_result,
                            figure_name=f'ceg{fold_suffix}',
                            show_plot=False,
                            plot_dir=FIGURES_DIR,
                            minimum_sensor_reading=minimum_sensor_reading,
                            maximum_sensor_reading=maximum_sensor_reading
                        )
                        df_metrics_performance.round(2).to_csv(
                            evaluation_table_path(METRICS_DIR, 'performance_by_range', fold_suffix), index=False)
                        df_metrics_performance['Fold'] = fold_idx

                        df_number_of_points.to_csv(
                            evaluation_table_path(METRICS_DIR, 'ceg_zone_counts', fold_suffix), index=False)
                        df_number_of_points['Fold'] = fold_idx

                        df_out_of_limits.to_csv(
                            evaluation_table_path(METRICS_DIR, 'prediction_limits', fold_suffix), index=False)
                        df_out_of_limits['Fold'] = fold_idx

                        aggregated_results_performance.append(df_metrics_performance)
                        aggregated_results_number_of_points.append(df_number_of_points)
                        aggregated_results_out_of_range.append(df_out_of_limits)

                        print()
                        print(f'Aggregated metrics for fold {fold_idx} done.')

                    print()
                    print('Calculating aggregated metrics across folds (Metric-Then-Aggregate)...')
                    if n_folds_to_train == 1:
                        print('NOTE: only one fold was trained (hold-out mode); standard deviations will be NaN.')

                    # '_per_fold': fold tables stacked (not aggregated); '_metric_then_aggregate': mean and std
                    # across folds; '_metric_then_aggregate_report': the same as "mean ± std" text.
                    df_metrics_performance_agg = pd.concat(aggregated_results_performance, ignore_index=True)
                    df_metrics_performance_agg.round(2).to_csv(
                        evaluation_table_path(METRICS_DIR, 'performance_by_range', '_per_fold'), index=False)
                    df_metrics_performance_agg = df_metrics_performance_agg.drop(columns=['Fold'])
                    df_metrics_performance_agg_flat, df_metrics_performance_agg_report = fold_results_aggregation(
                        df_metrics_performance_agg)
                    df_metrics_performance_agg_flat.round(2).to_csv(
                        evaluation_table_path(METRICS_DIR, 'performance_by_range', '_metric_then_aggregate'),
                        index=False)
                    df_metrics_performance_agg_report.round(2).to_csv(
                        evaluation_table_path(METRICS_DIR, 'performance_by_range', '_metric_then_aggregate_report'),
                        index=False)

                    df_number_of_points_agg = pd.concat(aggregated_results_number_of_points, ignore_index=True)
                    df_number_of_points_agg.to_csv(
                        evaluation_table_path(METRICS_DIR, 'ceg_zone_counts', '_per_fold'), index=False)
                    df_number_of_points_agg = df_number_of_points_agg.drop(columns=['Fold'])
                    df_number_of_points_agg_flat, df_number_of_points_report = fold_results_aggregation(
                        df_number_of_points_agg)
                    df_number_of_points_agg_flat.to_csv(
                        evaluation_table_path(METRICS_DIR, 'ceg_zone_counts', '_metric_then_aggregate'), index=False)
                    df_number_of_points_report.to_csv(
                        evaluation_table_path(METRICS_DIR, 'ceg_zone_counts', '_metric_then_aggregate_report'),
                        index=False)

                    df_out_of_range_agg = pd.concat(aggregated_results_out_of_range, ignore_index=True)
                    df_out_of_range_agg.to_csv(
                        evaluation_table_path(METRICS_DIR, 'prediction_limits', '_per_fold'), index=False)
                    df_out_of_range_agg = df_out_of_range_agg.drop(columns=['Fold'])
                    df_out_of_range_agg_flat, df_out_of_range_report = fold_results_aggregation(df_out_of_range_agg)
                    df_out_of_range_agg_flat.to_csv(
                        evaluation_table_path(METRICS_DIR, 'prediction_limits', '_metric_then_aggregate'), index=False)
                    df_out_of_range_report.to_csv(
                        evaluation_table_path(METRICS_DIR, 'prediction_limits', '_metric_then_aggregate_report'),
                        index=False)

                    print('Aggregated metrics across folds (Metric-Then-Aggregate) calculated and saved.')
                    print()

                    # ------------------ Aggregate-Then-Metric approach ------------------
                    print()
                    print('Calculating aggregated metrics across folds (Aggregate-Then-Metric)...')

                    df_all_folds = pd.concat(aggregated_results, ignore_index=True)
                    df_all_folds.to_parquet(os.path.join(PRED_DIR, 'predictions.parquet'))
                    print(f'\nAggregated all {n_folds_to_train} folds. Total rows: {len(df_all_folds)}')

                    df_metrics_performance, df_number_of_points, df_out_of_limits = test_by_range(df_all_folds,
                                                                                                  figure_name='ceg',
                                                                                                  show_plot=True,
                                                                                                  plot_dir=FIGURES_DIR,
                                                                                                  minimum_sensor_reading=minimum_sensor_reading,
                                                                                                  maximum_sensor_reading=maximum_sensor_reading)
                    df_metrics_performance.round(2).to_csv(
                        evaluation_table_path(METRICS_DIR, 'performance_by_range', '_aggregate_then_metric'),
                        index=False)
                    df_number_of_points.to_csv(
                        evaluation_table_path(METRICS_DIR, 'ceg_zone_counts', '_aggregate_then_metric'), index=False)
                    df_out_of_limits.to_csv(
                        evaluation_table_path(METRICS_DIR, 'prediction_limits', '_aggregate_then_metric'), index=False)
                    print('Metrics for all folds calculated and saved.')

                    write_experiment_readme(EXP_DIR, loss_function_name, current_algorithm, current_PH,
                                            current_historic, n_folds_to_train)
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

                    timing_json_path = os.path.join(EXP_DIR, "timing.json")
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
