# Loading data
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

data_dict = {}
data_dict.update(train_dict)
data_dict.update(val_dict)
data_dict.update(test_dict)
print(f'{len(data_dict)} patients loaded')

# Windows creation parameters
history_length = 8  # 120 min
horizons = [2, 4, 6, 8, 12, 16]  # 2 for 30 min, 4 for 60 min, etc.
remove_duplicates_intra_patient = True  # Set to True if you want to remove duplicates within each patient

path_to_save = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/windows/no_split_2025_deduplicated_intra_patient/'

if not os.path.exists(path_to_save):
    # Si no existe, crearlo
    os.makedirs(path_to_save)

# Auxiliary function
def get_windows_one_step_walk_forward(bgl_measurement_dict, history_length, horizon,
                                      deduplicate_intra_patient) -> pd.DataFrame:
    """
      :param bgl_measurement_dict: Dictionary with patient_id as key and numpy array of temporal series of BGL measurements
      :param history_length: Number of samples used to predict (8 for 120 minutes)
      :param horizon: Prediction horizon in number of windows (2 for 30 minutes, 4 for 60 minutes, and so on)
      :return: Get windows without missing values from one step sliding windows
    """
    df_all_patient_set = pd.DataFrame()

    for patient_id, patient_series in bgl_measurement_dict.items():
        # print(f"Clave: {patient_id} — Valor: {patient_series}")

        # Creating the one step sliding window
        x = np.lib.stride_tricks.sliding_window_view(patient_series[:-horizon], history_length)
        y = np.lib.stride_tricks.sliding_window_view(patient_series[history_length:], horizon)

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

    if deduplicate_intra_patient:
        # Drop duplicated instances in a patient. All columns are considered for duplication, therefore, if a patient has the same input and output values, it will be considered a duplicate, but if there are two equal instances but from different patients, they will not be considered duplicates.
        print(f"Removing duplicates within patients ...")
        len_before_deduplication = len(df_all_patient_set)

        df_all_patient_set = df_all_patient_set.drop_duplicates()
        df_all_patient_set.reset_index(drop=True, inplace=True)

        len_after_deduplication = len(df_all_patient_set)
        if len_after_deduplication == 0:
            raise ValueError("All windows were removed during deduplication, check your data.")
        elif len_after_deduplication < len_before_deduplication:
            print(f"Removed {len_before_deduplication - len_after_deduplication} duplicated windows.")
        else:
            print(" ---------- No duplicates found ---------- ")

    return df_all_patient_set


for current_horizon in horizons:
    print(f"Processing horizon: {current_horizon}")
    df_windows = get_windows_one_step_walk_forward(bgl_measurement_dict=data_dict, history_length=history_length,
                                                   horizon=current_horizon, deduplicate_intra_patient=remove_duplicates_intra_patient)

    # NOTE: Replace with your filepath
    df_windows.to_parquet(f'{path_to_save}/windows_horizon_{current_horizon}.parquet')
    print(f"Saved windows for horizon {current_horizon} to parquet files.")
    print("--------------------------------------------------")
    print()
