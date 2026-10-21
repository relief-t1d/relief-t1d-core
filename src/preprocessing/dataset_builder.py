# -*- coding: utf-8 -*-
"""
Dataset Builder Module
Encapsulates the end-to-end logic to generate a single dataset (pure windows + folds + production split)
for a specific combination of Historic (H) and Prediction Horizon (PH).
"""

import os
import pandas as pd

# Import internal preprocessing functions
from src.preprocessing.make_windows import get_windows_one_step_walk_forward
from src.preprocessing.make_folds import (
    make_folds_greedy_splits_valanced_priority,
    tag_folds_and_save
)


def build_single_dataset(
        raw_data_path: str,
        dataset_name: str,
        glucose_column_name: str,
        id_column_name: str,
        historic: int,
        ph: int,
        deduplicate_intra_patient: bool = True,
        # --- Flags and paths for output generation ---
        save_pure_windows: bool = False,
        output_pure_windows_dir: str = None,
        generate_folds: bool = True,
        output_folds_dir: str = None,
        k_folds: int = 5,
        random_seed: int = 42
) -> bool:
    """
    Executes the data generation pipeline for a single (H, PH) combination.
    Loads raw data, generates temporal sliding windows, and optionally saves pure windows
    and/or generates and saves k-folds with a production split.

    Args:
        raw_data_path (str): Path to the raw Parquet data file.
        dataset_name (str): Base name of the dataset for the output file.
        glucose_column_name (str): Name of the column containing the timestamps/dates.
        id_column_name (str): Name of the column containing patient IDs.
        historic (int): Number of historical steps used to predict (H).
        ph (int): Prediction horizon steps (PH).
        deduplicate_intra_patient (bool): Whether to remove intra-patient duplicates.
        save_pure_windows (bool): Flag to save the generated pure windows.
        output_pure_windows_dir (str): Directory to save the pure windows dataset.
        generate_folds (bool): Flag to calculate, tag, and save the cross-validation folds.
        output_folds_dir (str): Directory where the final folds dataset should be saved.
        k_folds (int): Number of folds for cross-validation.
        random_seed (int): Seed for reproducibility in fold creation.

    Returns:
        bool: True if the generation process completed successfully.
    """
    print(f"\n[Dataset Builder] Starting generation for H={historic}, PH={ph}...")

    # ---------------------------------------------------------
    # 1. Load and prepare raw data
    # ---------------------------------------------------------
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_data_path}")

    print(f"  -> Loading raw data from: {raw_data_path}")
    df_data = pd.read_parquet(raw_data_path)

    # Filter valid rows (ensure the datetime/glucose column is not null)
    df_valid_rows = df_data[df_data[glucose_column_name].notna()]

    # Sort columns by patient ID and time
    df_valid_rows = df_valid_rows.sort_values(by=[id_column_name, glucose_column_name])
    df_valid_rows.reset_index(drop=True, inplace=True)

    # Group measurements, dates, and times by patient ID
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

    # ---------------------------------------------------------
    # 2. Generate Temporal Sliding Windows
    # ---------------------------------------------------------
    print("  -> Generating pure sliding windows...")
    df_pure_windows = get_windows_one_step_walk_forward(
        bgl_measurement_dict=patient_dict,
        history_length=historic,
        horizon=ph,
        deduplicate_intra_patient=deduplicate_intra_patient,
        date_dict=date_dict,
        time_dict=time_dict
    )

    if df_pure_windows.empty:
        raise ValueError(f"No valid windows could be generated for H={historic}, PH={ph}.")

    # ---------------------------------------------------------
    # 3. Save Pure Windows (Optional)
    # ---------------------------------------------------------
    if save_pure_windows:
        if not output_pure_windows_dir:
            raise ValueError("output_pure_windows_dir must be provided if save_pure_windows is True.")

        os.makedirs(output_pure_windows_dir, exist_ok=True)
        pure_filename = f'{dataset_name}_H{historic}_PH{ph}_pure_windows.parquet'
        pure_filepath = os.path.join(output_pure_windows_dir, pure_filename)
        df_pure_windows.to_parquet(pure_filepath, index=False)
        print(f"  -> Pure windows saved to: {pure_filepath}")

    # ---------------------------------------------------------
    # 4. Generate and Save Folds (Optional)
    # ---------------------------------------------------------
    if generate_folds:
        if not output_folds_dir:
            raise ValueError("output_folds_dir must be provided if generate_folds is True.")

        os.makedirs(output_folds_dir, exist_ok=True)

        # Calculate Patient Loads
        patient_window_counts = df_pure_windows[id_column_name].value_counts().to_dict()

        # Create Folds and Production Split
        print(f"  -> Creating {k_folds}-fold splits and production split...")
        folds_split, production_split = make_folds_greedy_splits_valanced_priority(
            patient_windows=patient_window_counts,
            k=k_folds,
            seed=random_seed
        )

        # Tag Folds and Save Final Dataset
        print(f"  -> Tagging datasets and saving to Parquet...")
        tag_folds_and_save(
            df_windows=df_pure_windows,
            folds_splits=folds_split,
            production_fold=production_split,
            output_path=output_folds_dir,
            dataset_name=dataset_name,
            historic=historic,
            horizon=ph,
            k_folds=k_folds
        )
        print(f"  -> Folds dataset saved in: {output_folds_dir}")

    print(f"[Dataset Builder] Successfully finished process for H={historic}, PH={ph}.")
    return True