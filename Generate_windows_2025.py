import pickle
import numpy as np
import pandas as pd
import os

# Define the file path (Replace with your filepath)
file_path_train = "C:/Users/Ciro/Mi Unidad (PDI)/_UGR_PDI_2024-05-30/RELIEF-T1D/Python_projects/HeterogeneityInfluence/Dataset/Preprocessed data/measurements_by_patient_train.pkl"
file_path_test = "C:/Users/Ciro/Mi Unidad (PDI)/_UGR_PDI_2024-05-30/RELIEF-T1D/Python_projects/HeterogeneityInfluence/Dataset/Preprocessed data/measurements_by_patient_test.pkl"
file_path_val = "C:/Users/Ciro/Mi Unidad (PDI)/_UGR_PDI_2024-05-30/RELIEF-T1D/Python_projects/HeterogeneityInfluence/Dataset/Preprocessed data/measurements_by_patient_val.pkl"

# Open files
with open(file_path_train, 'rb') as file:
    train_dict = pickle.load(file)

with open(file_path_test, 'rb') as file:
    test_dict = pickle.load(file)

with open(file_path_val, 'rb') as file:
    val_dict = pickle.load(file)

# Windows creation parameters
history_length = 8  # 120 min
horizons = [2, 4, 6, 8, 12, 16]  # 2 for 30 min, 4 for 60 min, etc.

path = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/windows/new_2025/'
if not os.path.exists(path):
    # Si no existe, crearlo
    os.makedirs(path)


# Auxiliary function
def get_windows_one_step_walk_forward(bgl_measurement_dict, history_length, ph) -> pd.DataFrame:
    """
      :param bgl_measurement_dict: Dictionary with patient_id as key and numpy array of temporal series of BGL measurements
      :param history_length: Number of samples used to predict (8 for 120 minutes)
      :param ph: Prediction horizon in number of windows (2 for 30 minutes, 4 for 60 minutes, and so on)
      :return: Get windows without missing values from one step sliding windows
    """
    df_all_patient_set = pd.DataFrame()

    for patient_id, patient_series in bgl_measurement_dict.items():
        # print(f"Clave: {patient_id} — Valor: {patient_series}")

        # Creating the one step sliding window
        x = np.lib.stride_tricks.sliding_window_view(patient_series[:-ph], history_length)
        y = np.lib.stride_tricks.sliding_window_view(patient_series[history_length:], ph)

        # Removing rows with missing values (NaN)
        nan_rows_x = np.isnan(x).any(axis=1)  # In input values
        nan_rows_y = np.isnan(y).any(axis=1)  # In output values
        x = x[~(nan_rows_x | nan_rows_y)]  # Remove rows with NaN in either x or y from x
        y = y[~(nan_rows_x | nan_rows_y)]  # Remove rows with NaN in either x or y from y

        # Create DataFrame for the current patient
        df_x = pd.DataFrame(x)  # Input windows
        df_y = pd.DataFrame(y[:, -1])  # Output windows (last column of y)
        df_set = pd.concat([df_x, df_y], axis=1)  # Combine inputs and outputs
        df_set = df_set.astype(int)  # Convert to integer type

        # Renaming columns and adding patient_id.
        # Input: x0 a x7, Output: y
        new_cols = [f"x{i}" for i in range(8)] + ["y"]
        df_set.columns = new_cols
        df_set['patient_id'] = patient_id

        # Concat the current patient's DataFrame to the main DataFrame
        df_all_patient_set = pd.concat([df_all_patient_set, df_set], ignore_index=True)

    return df_all_patient_set


for current_horizon in horizons:
    print(f"Processing horizon: {current_horizon}")
    df_train_set = get_windows_one_step_walk_forward(train_dict, history_length, current_horizon)
    df_test_set = get_windows_one_step_walk_forward(test_dict, history_length, current_horizon)
    df_val_set = get_windows_one_step_walk_forward(val_dict, history_length, current_horizon)

    # NOTE: Replace with your filepath
    df_train_set.to_parquet(f'{path}/train_windows_horizon_{current_horizon}.parquet')
    df_test_set.to_parquet(f'{path}/test_windows_horizon_{current_horizon}.parquet')
    df_val_set.to_parquet(f'{path}/val_windows_horizon_{current_horizon}.parquet')
    print(f"Saved windows for horizon {current_horizon} to parquet files.")
    print("--------------------------------------------------")
    print()
