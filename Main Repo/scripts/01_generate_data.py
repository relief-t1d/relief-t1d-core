# -*- coding: utf-8 -*-
"""
Execution script to generate temporal windows and cross-validation folds.
This script strictly follows the data preparation steps from generate_windows.py
"""

import os
import pandas as pd

# Import our custom modules from the 'src' directory
from src.data.make_windows import get_windows_one_step_walk_forward
from src.data.make_folds import ( 
    make_folds_greedy_splits_valanced_priority,
    validate_folds,
    tag_folds_and_save
)

# ==========================================
# 1. CONFIGURATION AND HYPERPARAMETERS
# ==========================================
DATASET_NAME = "DiaTrend"

# List of historical data to test (in 15-minute increments)
# 4 = 1 hour, 6 = 1.5 hours, 8 = 2 hours, 12 = 3 hours
HISTORY_LENGTHS = [8]
HORIZONS = [2, 4]  # 2 for 30 min, 4 for 60 min, etc.

DEDUPLICATE_INTRA_PATIENT = True
K_FOLDS = 5
RANDOM_SEED = 42

GLUCOSE_COLUMN_NAME = '15min'
ID_COLUMN_NAME = 'patient_id'

PATH_TO_LOAD_DATA = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/Datasets/2025-12-05/Preprocessed/diatrend/Final_versions_V4/Filtered_for_prediction/Glucose_measurements_DiaTrend_FILTERED_2026-03-27.parquet'
BASE_OUTPUT_DIR = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/Glucose_prediction_2026/Main_repo/Data_test/different_history_horizon_combinations/'


# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def main():
    print(f"--- Starting data generation pipeline for {DATASET_NAME} ---")

    # --- Automatically create the two subdirectories within the base path ---
    path_pure_windows = os.path.join(BASE_OUTPUT_DIR, 'windows/')
    path_folds = os.path.join(BASE_OUTPUT_DIR, 'folds/')

    os.makedirs(path_pure_windows, exist_ok=True)
    os.makedirs(path_folds, exist_ok=True)

    # 2.1 Load and prepare the raw data
    print(f"Loading raw data from: {PATH_TO_LOAD_DATA}")
    df_data = pd.read_parquet(PATH_TO_LOAD_DATA)

    # Filter valid rows (15min is not NaT)
    df_valid_rows = df_data[df_data[GLUCOSE_COLUMN_NAME].notna()]

    # Sort columns by ID_COLUMN_NAME and GLUCOSE_COLUMN_NAME
    df_valid_rows = df_valid_rows.sort_values(by=[ID_COLUMN_NAME, GLUCOSE_COLUMN_NAME])
    df_valid_rows.reset_index(drop=True, inplace=True)  # Reset index after filtering

    # Group by ID_COLUMN_NAME and convert 'Measurement' to numpy array
    patient_dict = {
        pid: group['measurement'].to_numpy()
        for pid, group in df_valid_rows.groupby(ID_COLUMN_NAME)
    }

    # Group by ID_COLUMN_NAME and convert GLUCOSE_COLUMN_NAME to numpy array of dates
    date_dict = {
        pid: group[GLUCOSE_COLUMN_NAME].dt.date.to_numpy()
        for pid, group in df_valid_rows.groupby(ID_COLUMN_NAME)
    }

    # Group by ID_COLUMN_NAME and convert GLUCOSE_COLUMN_NAME to numpy array of times
    time_dict = {
        pid: group[GLUCOSE_COLUMN_NAME].dt.time.to_numpy()
        for pid, group in df_valid_rows.groupby(ID_COLUMN_NAME)
    }
    # =========================================================================

    print(f"Number of patients with valid data: {len(patient_dict)}")

    # 2.2 Iterate over each history lengths 
    for current_history in HISTORY_LENGTHS:
        print(f"\n====================================================================")
        print(f" 🕒 RUNNING HISTORY LENGTH (H): {current_history} steps ({current_history * 15} mins)")
        print(f"====================================================================")

        for current_horizon in HORIZONS:
            print(f"\n  -> Processing Combination: H={current_history} | PH={current_horizon}")
            print(f"  --------------------------------------------------")

            # --- Step A: Generate Pure Windows ---
            print("  Generating sliding windows...")
            df_pure_windows = get_windows_one_step_walk_forward(
                bgl_measurement_dict=patient_dict,
                history_length=current_history,
                horizon=current_horizon,
                deduplicate_intra_patient=DEDUPLICATE_INTRA_PATIENT,
                date_dict=date_dict,
                time_dict=time_dict
            )
            print(f"  Total windows generated: {len(df_pure_windows)}")

            # --- Step B: Save Pure Windows (ONLY windows, dates, times and patient_id) ---
            pure_filename = f"windows_H{current_history}_PH{current_horizon}_{DATASET_NAME}.parquet"
            pure_filepath = os.path.join(path_pure_windows, pure_filename)
            df_pure_windows.to_parquet(pure_filepath, index=False)
            print(f"  ✓ Pure windows saved to: {pure_filename}")

            # --- Step C: Calculate patient loads (window counts) ---
            patient_window_counts = df_pure_windows[ID_COLUMN_NAME].value_counts().to_dict()

            # --- Step D: Create Folds ---
            print(f"  Creating {K_FOLDS}-fold patient-wise splits...")
            folds_split = make_folds_greedy_splits_valanced_priority(
                patient_windows=patient_window_counts,
                k=K_FOLDS,
                seed=RANDOM_SEED
            )

            # --- Step E: Validate Folds ---
            validate_folds(
                folds=folds_split,
                patient_measurements=patient_window_counts,
                horizon=current_horizon
            )

            # --- Step F: Tag and Save to Parquet (Windows + 5-Fold columns added) ---
            tag_folds_and_save(
                df_windows=df_pure_windows,
                folds=folds_split,
                output_path=path_folds,
                filename=f"windows_with_5folds_{DATASET_NAME}_H{current_history}",
                horizon=current_horizon
            )

    print("\n--- All combinations generated and pipeline finished successfully! ---")


if __name__ == "__main__":
    main()