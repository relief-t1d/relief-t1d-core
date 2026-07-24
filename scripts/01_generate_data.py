# -*- coding: utf-8 -*-
"""
Execution script to generate temporal windows and cross-validation folds.
This script strictly follows the data preparation steps from generate_windows.py,
using an external YAML configuration file for reproducibility.
"""

# ---------------------------------------------------------
# SYSTEM PATH CONFIGURATION (Essential for Remote Server)
# ---------------------------------------------------------
import os
import sys

# Get the absolute path of the directory containing this script (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory and append it to sys.path
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------
# IMPORTS STANDARD, THIRD-PARTY, AND  INTERNAL MODULAR LIBRARY
# ---------------------------------------------------------
import argparse
from pathlib import Path
import yaml
import pandas as pd

# Import our custom modules from the 'src' directory
from src.data.make_windows import get_windows_one_step_walk_forward
from src.data.make_folds import (
    make_folds_greedy_splits_valanced_priority,
    validate_folds,
    tag_folds_and_save
)

# Get the absolute path to the folder where this script is located ('scripts')
SCRIPT_DIR = Path(__file__).resolve().parent
# Get the absolute path to the project root (one level up from 'scripts')
PROJECT_ROOT = SCRIPT_DIR.parent
# Build the path to 'configs/preprocessing_prediction_data.yaml' at the root
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "preprocessing_prediction_data.yaml"


def parse_arguments():
    """Configure and read the CLI arguments."""
    parser = argparse.ArgumentParser(description="Pipeline to generate temporal windows using a YAML config.")
    parser.add_argument(
        '--config',
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML configuration file."
    )
    return parser.parse_args()


# ==========================================
# MAIN PIPELINE
# ==========================================
def main(config_path):
    # --- Read configuration from YAML file ---
    print(f"Reading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    # --- Extract variables ---
    # Bloque: dataset
    dataset_name = config['dataset']['name']
    glucose_column_name = config['dataset']['glucose_column']
    id_column_name = config['dataset']['id_column']

    # Bloque: hyperparameters
    history_lengths = config['hyperparameters']['history_lengths']
    horizons = config['hyperparameters']['horizons']

    # Bloque: folds
    k_folds = config['folds']['k_folds']
    random_seed = config['folds']['random_seed']
    # Como en el YAML se llama 'disable_deduplication' (true/false),
    # invertimos el booleano si tu variable indica si SE deduplica o no:
    deduplicate_intra_patient = not config['folds']['disable_deduplication']

    # Bloque: paths
    path_to_load_data = config['paths']['path_to_load']
    base_output_dir = config['paths']['base_output_dir']

    print(f"--- Starting data generation pipeline for {dataset_name} ---")

    # --- Automatically create the two subdirectories within the base path ---
    path_pure_windows = os.path.join(base_output_dir, 'windows/')
    path_folds = os.path.join(base_output_dir, 'folds/')

    os.makedirs(path_pure_windows, exist_ok=True)
    os.makedirs(path_folds, exist_ok=True)

    # 2.1 Load and prepare the raw data
    print(f"Loading raw data from: {path_to_load_data}")
    df_data = pd.read_parquet(path_to_load_data)

    # Filter valid rows (15min is not NaT)
    df_valid_rows = df_data[df_data[glucose_column_name].notna()]

    # Sort columns by ID and Glucose column
    df_valid_rows = df_valid_rows.sort_values(by=[id_column_name, glucose_column_name])
    df_valid_rows.reset_index(drop=True, inplace=True)  # Reset index after filtering

    # Group by ID and convert 'measurement' to numpy array
    patient_dict = {
        pid: group['measurement'].to_numpy()
        for pid, group in df_valid_rows.groupby(id_column_name)
    }

    # Group by ID and convert glucose column to numpy array of dates
    date_dict = {
        pid: group[glucose_column_name].dt.date.to_numpy()
        for pid, group in df_valid_rows.groupby(id_column_name)
    }

    # Group by ID and convert glucose column to numpy array of times
    time_dict = {
        pid: group[glucose_column_name].dt.time.to_numpy()
        for pid, group in df_valid_rows.groupby(id_column_name)
    }
    # =========================================================================

    print(f"Number of patients with valid data: {len(patient_dict)}")

    # 2.2 Iterate over each history lengths
    for current_history in history_lengths:
        print(f"\n====================================================================")
        print(f" 🕒 RUNNING HISTORY LENGTH (H): {current_history} steps ({current_history * 15} mins)")
        print(f"====================================================================")

        for current_horizon in horizons:
            print(f"\n  -> Processing Combination: H={current_history} | PH={current_horizon}")
            print(f"  --------------------------------------------------")

            # --- Step A: Generate Pure Windows ---
            print("  Generating sliding windows...")
            df_pure_windows = get_windows_one_step_walk_forward(
                bgl_measurement_dict=patient_dict,
                history_length=current_history,
                horizon=current_horizon,
                deduplicate_intra_patient=deduplicate_intra_patient,
                date_dict=date_dict,
                time_dict=time_dict
            )
            print(f"  Total windows generated: {len(df_pure_windows)}")

            # --- Step B: Save Pure Windows ---
            pure_filename = f"windows_H{current_history}_PH{current_horizon}_{dataset_name}.parquet"
            pure_filepath = os.path.join(path_pure_windows, pure_filename)
            df_pure_windows.to_parquet(pure_filepath, index=False)
            print(f"  ✓ Pure windows saved to: {pure_filename}")

            # --- Step C: Calculate patient loads (window counts) ---
            patient_window_counts = df_pure_windows[id_column_name].value_counts().to_dict()

            # --- Step D: Create Folds ---
            print(f"  Creating {k_folds}-fold patient-wise splits...")
            folds_split = make_folds_greedy_splits_valanced_priority(
                patient_windows=patient_window_counts,
                k=k_folds,
                seed=random_seed
            )

            # --- Step E: Validate Folds ---
            validate_folds(
                folds=folds_split,
                patient_measurements=patient_window_counts,
                horizon=current_horizon
            )

            # --- Step F: Tag and Save to Parquet (Windows + K-Fold columns added) ---
            tag_folds_and_save(
                df_windows=df_pure_windows,
                folds=folds_split,
                output_path=path_folds,
                filename=f"windows_with_{k_folds}folds_{dataset_name}_H{current_history}",
                horizon=current_horizon
            )

    print("\n--- All combinations generated and pipeline finished successfully! ---")


if __name__ == "__main__":
    # Parse CLI arguments to get the config file path
    cli_args = parse_arguments()

    # Run the main pipeline passing the config path
    main(cli_args.config)
