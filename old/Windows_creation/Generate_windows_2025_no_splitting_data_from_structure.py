import numpy as np
import pandas as pd
import os

# Load the dataset
path_to_load_data = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/Datasets/DiaTrend/Glucose_measurements_v3_DiaTrend.parquet'
df_data = pd.read_parquet(path_to_load_data)

# Filter valid rows (15min is not NaT)
df_valid_rows = df_data[df_data['15min'].notna()]

# Sort columns by 'Patient_ID' and '15min'
df_valid_rows.loc[:, 'Patient_ID'] = df_valid_rows['Patient_ID'].astype(int)  # Ensure Patient_ID is int type
df_valid_rows = df_valid_rows.sort_values(by=['Patient_ID', '15min'])
df_valid_rows.reset_index(drop=True, inplace=True)  # Reset index after filtering

# Group by 'Patient_ID' and convert 'Measurement' to numpy array
patient_dict = {
    pid: group['Measurement'].to_numpy()  # Convert to int32
    for pid, group in df_valid_rows.groupby('Patient_ID')
}
print(f"Number of patients with valid data: {len(patient_dict)}")

# Windows creation parameters
history_length = 8  # 120 min
horizons = [2, 4]  # 2 for 30 min, 4 for 60 min
deduplicate_intra_patient = True  # Remove duplicated instances in a patient

path_to_save_windows = 'c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/windows/no_split_2025_DiaTrend_V3/'
if not os.path.exists(path_to_save_windows):
    # If it does not exist, create it
    os.makedirs(path_to_save_windows)


# Auxiliary function
def get_windows_one_step_walk_forward(bgl_measurement_dict,
                                      history_length,
                                      horizon,
                                      deduplicate_intra_patient) -> pd.DataFrame:
    """
      :param bgl_measurement_dict: Dictionary with patient_id as key and numpy array of temporal series of BGL measurements
      :param history_length: Number of samples used to predict (8 for 120 minutes)
      :param horizon: model_prediction_scripts horizon in number of windows (2 for 30 minutes, 4 for 60 minutes, and so on)
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
        print("Deduplicating intra-patient instances...")
        print(f"Initial DataFrame shape: {df_all_patient_set.shape}")
        df_all_patient_set = df_all_patient_set.drop_duplicates()
        df_all_patient_set.reset_index(drop=True, inplace=True)
        print(f"Deduplicated DataFrame shape: {df_all_patient_set.shape}")

    return df_all_patient_set


for current_horizon in horizons:
    print(f"Processing horizon: {current_horizon}")
    df_windows = get_windows_one_step_walk_forward(bgl_measurement_dict=patient_dict,
                                                   history_length=history_length,
                                                   horizon=current_horizon,
                                                   deduplicate_intra_patient=deduplicate_intra_patient)

    # NOTE: Replace with your filepath
    df_windows.to_parquet(f'{path_to_save_windows}/windows_horizon_{current_horizon}.parquet')
    print(f"Saved windows for horizon {current_horizon} to parquet files.")
    print("--------------------------------------------------")
    print()
