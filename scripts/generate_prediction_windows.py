# -*- coding: utf-8 -*-
"""
RELIEF-T1D: Temporal Windows and Folds Generation Pipeline

This script reads the source blood glucose dataset (already filtered upstream), generates sliding temporal windows
based on specified history multipliers or history lengths and prediction horizons,
and splits patients into balanced k-fold cross-validation sets.

All configuration parameters and input/output paths are governed by an external
YAML configuration file supporting environment variable expansion (e.g., ${PATH_TO_DATA}).
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from a local .env file into os.environ if present
load_dotenv()

# ---------------------------------------------------------
# 1. SYSTEM PATH CONFIGURATION
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "generate_prediction_windows.yaml"

# ---------------------------------------------------------
# 2. IMPORTS FROM INTERNAL MODULAR LIBRARY (src/)
# ---------------------------------------------------------
from src.preprocessing.dataset_builder import build_single_dataset


class Tee(object):
    """Duplicates stdout/stderr outputs to both the console and a log file."""
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def parse_arguments():
    """Configure and parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Pipeline to generate temporal windows using a YAML config.")
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

    # Proportional mode: dynamic multipliers relative to prediction horizon
    if 'history_multipliers' in hyperparams_config:
        multipliers = hyperparams_config['history_multipliers']
        for ph in hyperparams_config['horizons']:
            # Calculate H, enforce minimum of 1, round to integer, and deduplicate
            calculated_h = sorted(list(set(
                max(1, int(round(ph * m))) for m in multipliers
            )))
            for h in calculated_h:
                pairs.append((h, ph))

    # Traditional mode: static history length list
    elif 'history_lengths' in hyperparams_config:
        for h in hyperparams_config['history_lengths']:
            for ph in hyperparams_config['horizons']:
                pairs.append((h, ph))
    else:
        raise KeyError("Hyperparameters section must contain either 'history_multipliers' or 'history_lengths'.")

    return pairs


# ==========================================
# MAIN PIPELINE
# ==========================================
def main(config_path):
    # --- Load YAML configuration and expand environment variables ---
    with open(config_path, 'r', encoding='utf-8') as file:
        raw_yaml_content = file.read()
        expanded_yaml_content = os.path.expandvars(raw_yaml_content)
        config = yaml.safe_load(expanded_yaml_content)

    # --- Extract variables ---
    dataset_name = config['dataset']['name']
    glucose_column_name = config['dataset']['glucose_column']
    id_column_name = config['dataset']['id_column']

    # Dynamically derive (history_length, horizon) experiment pairs
    experiment_pairs = get_experiment_pairs(config['hyperparameters'])

    k_folds = config['folds']['k_folds']
    random_seed = config['folds']['random_seed']
    deduplicate_intra_patient = not config['folds']['disable_deduplication']

    path_to_load_data = config['paths']['path_to_load']
    base_output_dir = config['paths']['base_output_dir']

    # Extract optional generation flags if present in config, defaulting to True for standard pipeline
    save_pure_windows = config.get('options', {}).get('save_pure_windows', True)
    generate_folds = config.get('options', {}).get('generate_folds', True)

    # --- Setup Global Log Redirection ---
    os.makedirs(base_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(base_output_dir, f"execution_log_{dataset_name}_{timestamp}.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_filename, "a", encoding="utf-8")

    # Intercept all prints and errors from ALL imported modules
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        print(f"Reading configuration from: {config_path}")
        print(f"Execution output will be logged to: {log_filename}")
        print(f"--- Starting data generation pipeline for {dataset_name} ---")

        # --- Automatically create subdirectories ---
        path_pure_windows = os.path.join(base_output_dir, 'windows/')
        path_folds = os.path.join(base_output_dir, 'folds/')

        os.makedirs(path_pure_windows, exist_ok=True)
        os.makedirs(path_folds, exist_ok=True)

        # 2.1 Iterate over dynamic experiment pairs (H, PH)
        for current_historic, current_horizon in experiment_pairs:
            print("\n====================================================================")
            print(f" 🕒 RUNNING COMBINATION: H={current_historic} steps ({current_historic * 15} mins) | PH={current_horizon}")
            print("====================================================================")

            # Delegate end-to-end processing to the Dataset Builder
            build_single_dataset(
                source_data_path=path_to_load_data,
                dataset_name=dataset_name,
                glucose_column_name=glucose_column_name,
                id_column_name=id_column_name,
                historic=current_historic,
                ph=current_horizon,
                deduplicate_intra_patient=deduplicate_intra_patient,
                save_pure_windows=save_pure_windows,
                output_pure_windows_dir=path_pure_windows,
                generate_folds=generate_folds,
                output_folds_dir=path_folds,
                k_folds=k_folds,
                random_seed=random_seed
            )

        print("\n--- All combinations generated and pipeline finished successfully! ---")

    finally:
        # Restore standard output streams and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    # Parse CLI arguments to get the config file path
    cli_args = parse_arguments()

    # Run the main pipeline passing the config path
    main(cli_args.config)
