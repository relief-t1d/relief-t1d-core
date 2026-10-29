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
    validate_folds,
    tag_folds_and_save
)


def build_single_dataset(
        source_data_path: str,
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
    Executes the end-to-end data generation pipeline for a single Historic (H) and Prediction Horizon (PH) combination.

    This function loads the source Parquet data (CGM series already filtered upstream), creates temporal sliding
    windows, and optionally saves the pure windows. It also computes patient statistics, validates the splits, and
    generates k-folds along with a standalone production split for final model deployment.

    Args:
        source_data_path (str): Absolute or relative path to the source Parquet file containing the measurements.
            This file is the input of the pipeline and has already been filtered upstream (it is not raw data).
        dataset_name (str): Base name of the dataset used as a prefix for the output files (e.g., 'ohiot1dm').
        glucose_column_name (str): Name of the DataFrame column containing the timestamps/dates of the measurements.
        id_column_name (str): Name of the DataFrame column containing the unique patient identifiers.
        historic (int): Number of historical steps used as input to predict the future (H).
        ph (int): Prediction horizon in number of future steps to predict (PH).
        deduplicate_intra_patient (bool, optional): If True, removes windows with identical input/output values
            within the same patient. Defaults to True.
        save_pure_windows (bool, optional): Flag to enable saving the generated pure windows (without fold tags)
            to disk. Defaults to False.
        output_pure_windows_dir (str, optional): Directory path to save the pure windows dataset.
            Strictly required if `save_pure_windows` is True. Defaults to None.
        generate_folds (bool, optional): Flag to calculate, tag, validate, and save the cross-validation folds
            and the independent production split. Defaults to True.
        output_folds_dir (str, optional): Directory path where the final tagged dataset should be saved.
            Strictly required if `generate_folds` is True. Defaults to None.
        k_folds (int, optional): Number of folds to create for the cross-validation splits. Defaults to 5.
        random_seed (int, optional): Seed used for reproducibility during the greedy patient splitting process. Defaults to 42.

    Returns:
        bool: True if the generation, validation, and saving processes completed successfully.

    Raises:
        FileNotFoundError: If the specified `source_data_path` does not exist.
        ValueError: If no valid windows could be generated, or if output directories are missing when their respective flags are True.
    """
    print(f"\n[Dataset Builder] Starting generation for H={historic}, PH={ph}...")

    # ---------------------------------------------------------
    # 1. Load and prepare source data
    # ---------------------------------------------------------
    if not os.path.exists(source_data_path):
        raise FileNotFoundError(f"Source data file not found at: {source_data_path}")

    print(f"  -> Loading source data from: {source_data_path}")
    df_data = pd.read_parquet(source_data_path)

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

    print(f"  -> Number of patients with valid data: {len(patient_dict)}")

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

    print(f"  -> Total windows generated: {len(df_pure_windows)}")

    # ---------------------------------------------------------
    # 3. Save Pure Windows (Optional)
    # ---------------------------------------------------------
    if save_pure_windows:
        if not output_pure_windows_dir:
            raise ValueError("output_pure_windows_dir must be provided if save_pure_windows is True.")

        os.makedirs(output_pure_windows_dir, exist_ok=True)
        pure_filename = f'{dataset_name}_H{historic}_PH{ph}_windows.parquet'
        pure_filepath = os.path.join(output_pure_windows_dir, pure_filename)
        df_pure_windows.to_parquet(pure_filepath, index=False)
        print(f"  ✓ Pure windows saved to: {pure_filename}")

    # ---------------------------------------------------------
    # 4. Generate, Validate and Save Folds (Optional)
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

        # --- Statistics Logging ---
        print(f"\n  ------- Statistics of folds created for H={historic} | PH={ph} -------")
        num_patients = len(patient_window_counts)
        total_windows_all = sum(patient_window_counts.values())

        for i, fold in enumerate(folds_split):
            train_wins = sum(patient_window_counts[pid] for pid in fold['train'])
            val_wins = sum(patient_window_counts[pid] for pid in fold['val'])
            test_wins = sum(patient_window_counts[pid] for pid in fold['test'])

            print(f"\n  ---- Fold {i} (column 'fold{i}'): -----")
            print("  -- Totals --")
            print(f"    Train: {len(fold['train'])} patients. Total windows: {train_wins}")
            print(f"    Val:   {len(fold['val'])} patients. Total windows: {val_wins}")
            print(f"    Test:  {len(fold['test'])} patients. Total windows: {test_wins}")

            print("\n  -- Percentages --")
            print(
                f"    Train: {(len(fold['train']) / num_patients * 100):.2f}% patients. Windows: {(train_wins / total_windows_all * 100):.2f}%")
            print(
                f"    Val:   {(len(fold['val']) / num_patients * 100):.2f}% patients. Windows: {(val_wins / total_windows_all * 100):.2f}%")
            print(
                f"    Test:  {(len(fold['test']) / num_patients * 100):.2f}% patients. Windows: {(test_wins / total_windows_all * 100):.2f}%")

        # --- Statistics for Production Split (printed once, outside the fold loop) ---
        prod_train_wins = sum(patient_window_counts[pid] for pid in production_split['train'])
        prod_val_wins = sum(patient_window_counts[pid] for pid in production_split['val'])

        print(f"\n  ---- Production Split (Deployment Model): -----")
        print("  -- Totals --")
        print(f"    Train: {len(production_split['train'])} patients. Total windows: {prod_train_wins}")
        print(f"    Val:   {len(production_split['val'])} patients. Total windows: {prod_val_wins}")

        print("\n  -- Percentages --")
        print(
            f"    Train: {(len(production_split['train']) / num_patients * 100):.2f}% patients. Windows: {(prod_train_wins / total_windows_all * 100):.2f}%")
        print(
            f"    Val:   {(len(production_split['val']) / num_patients * 100):.2f}% patients. Windows: {(prod_val_wins / total_windows_all * 100):.2f}%")

        print(f"\n  End of statistics for folds created for H={historic} | PH={ph}")

        # --- Validate Folds ---
        validate_folds(
            folds=folds_split,
            patient_measurements=patient_window_counts,
            horizon=ph
        )

        # --- Tag Folds and Save Final Dataset ---
        print(f"  -> Tagging datasets and saving to Parquet...")
        tag_folds_and_save(
            df_windows=df_pure_windows,
            folds_splits=folds_split,
            production_fold=production_split,
            output_path=output_folds_dir,
            dataset_name=dataset_name,
            historic=str(historic),
            horizon=ph,
            k_folds=k_folds
        )
        print(f"  ✓ Folds dataset saved in: {output_folds_dir}")

    print(f"[Dataset Builder] Successfully finished process for H={historic}, PH={ph}.")
    return True
