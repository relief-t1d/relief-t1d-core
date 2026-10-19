# -*- coding: utf-8 -*-
"""
RELIEF-T1D: Temporal Windows and Folds Generation Pipeline

This script reads raw blood glucose datasets, generates sliding temporal windows
based on specified history multipliers or history lengths and prediction horizons,
and splits patients into balanced k-fold cross-validation sets.

All configuration parameters and input/output paths are governed by an external
YAML configuration file supporting environment variable expansion (e.g., ${PATH_TO_LOAD}).
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd
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
from src.preprocessing.make_windows import get_windows_one_step_walk_forward
from src.preprocessing.make_folds import (
    make_folds_greedy_splits_valanced_priority,
    validate_folds,
    tag_folds_and_save
)


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

        # 2.1 Load and prepare raw data
        print(f"Loading raw data from: {path_to_load_data}")
        df_data = pd.read_parquet(path_to_load_data)

        # Filter valid rows
        df_valid_rows = df_data[df_data[glucose_column_name].notna()]

        # Sort columns
        df_valid_rows = df_valid_rows.sort_values(by=[id_column_name, glucose_column_name])
        df_valid_rows.reset_index(drop=True, inplace=True)

        # Group by ID
        patient_dict = {
            pid: group['measurement'].to_numpy()
            for pid, group in df_valid_rows.groupby(id_column_name)
        }

        date_dict = {
            pid: group[glucose_column_name].dt.date.to_numpy()
            for pid, group in df_valid_rows.groupby(id_column_name)
        }

        time_dict = {
            pid: group[glucose_column_name].dt.time.to_numpy()
            for pid, group in df_valid_rows.groupby(id_column_name)
        }

        print(f"Number of patients with valid data: {len(patient_dict)}")

        # 2.2 Iterate over dynamic experiment pairs (H, PH)
        for current_historic, current_horizon in experiment_pairs:
            print("\n====================================================================")
            print(f" 🕒 RUNNING COMBINATION: H={current_historic} steps ({current_historic * 15} mins) | PH={current_horizon}")
            print("====================================================================")
            print(f"  -> Processing Combination: H={current_historic} | PH={current_horizon}")
            print("  --------------------------------------------------")

            # --- Step A: Generate Pure Windows ---
            print("  Generating sliding windows...")
            df_pure_windows = get_windows_one_step_walk_forward(
                bgl_measurement_dict=patient_dict,
                history_length=current_historic,
                horizon=current_horizon,
                deduplicate_intra_patient=deduplicate_intra_patient,
                date_dict=date_dict,
                time_dict=time_dict
            )
            print(f"  Total windows generated: {len(df_pure_windows)}")

            # --- Step B: Save Pure Windows ---
            pure_filename = f'{dataset_name}_H{current_historic}_PH{current_horizon}_windows.parquet'
            pure_filepath = os.path.join(path_pure_windows, pure_filename)
            df_pure_windows.to_parquet(pure_filepath, index=False)
            print(f"  ✓ Pure windows saved to: {pure_filename}")

            # --- Step C: Calculate patient loads ---
            patient_window_counts = df_pure_windows[id_column_name].value_counts().to_dict()

            # --- Step D: Create Folds ---
            print(f"  Creating {k_folds}-fold patient-wise splits and Production Split...")
            folds_split, production_split = make_folds_greedy_splits_valanced_priority(
                                            patient_windows=patient_window_counts,
                                            k=k_folds,
                                            seed=random_seed
                                            )

            # --- Step D.1: Print Fold Statistics ---
            print(f"\n  ------- Statistics of folds created for H={current_historic} | PH={current_horizon} -------")
            num_patients = len(patient_window_counts)
            total_windows_all = sum(patient_window_counts.values())

            for i, fold in enumerate(folds_split):
                train_wins = sum(patient_window_counts[pid] for pid in fold['train'])
                val_wins = sum(patient_window_counts[pid] for pid in fold['val'])
                test_wins = sum(patient_window_counts[pid] for pid in fold['test'])

                print(f"\n  ---- Fold {i + 1}: -----")
                print("  -- Totals --")
                print(f"    Train: {len(fold['train'])} patients. Total windows: {train_wins}")
                print(f"    Val:   {len(fold['val'])} patients. Total windows: {val_wins}")
                print(f"    Test:  {len(fold['test'])} patients. Total windows: {test_wins}")

                print("\n  -- Percentages --")
                print(f"    Train: {(len(fold['train']) / num_patients * 100):.2f}% patients. Windows: {(train_wins / total_windows_all * 100):.2f}%")
                print(f"    Val:   {(len(fold['val']) / num_patients * 100):.2f}% patients. Windows: {(val_wins / total_windows_all * 100):.2f}%")
                print(f"    Test:  {(len(fold['test']) / num_patients * 100):.2f}% patients. Windows: {(test_wins / total_windows_all * 100):.2f}%")

                # --- Statistics for Production Split ---
                prod_train_wins = sum(patient_window_counts[pid] for pid in production_split['train'])
                prod_val_wins = sum(patient_window_counts[pid] for pid in production_split['val'])

                print(f"\n  ---- Production Split (Deployment Model): -----")
                print("  -- Totals --")
                print(f"    Train: {len(production_split['train'])} patients. Total windows: {prod_train_wins}")
                print(f"    Val:   {len(production_split['val'])} patients. Total windows: {prod_val_wins}")

                print("\n  -- Percentages --")
                print(f"    Train: {(len(production_split['train']) / num_patients * 100):.2f}% patients. Windows: {(prod_train_wins / total_windows_all * 100):.2f}%")
                print(f"    Val:   {(len(production_split['val']) / num_patients * 100):.2f}% patients. Windows: {(prod_val_wins / total_windows_all * 100):.2f}%")

            print(f"\n  End of statistics for folds created for H={current_historic} | PH={current_horizon}")

            # --- Step E: Validate Folds ---
            validate_folds(
                folds=folds_split,
                patient_measurements=patient_window_counts,
                horizon=current_horizon
            )

            # --- Step F: Tag and Save ---
            tag_folds_and_save(
                df_windows=df_pure_windows,
                folds_splits=folds_split,
                production_fold=production_split,
                output_path=path_folds,
                dataset_name=dataset_name,
                historic=current_historic,
                horizon=current_horizon,
                k_folds=k_folds
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