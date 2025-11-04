# -*- coding: utf-8 -*-
"""
Recalculate Metrics with Clipped Predictions Script (Hardcoded Path Version)

This script takes a prediction parquet file, clips the predicted values
to a clinically relevant range, and then recalculates the performance metrics
using the new, bounded predictions.

Instructions:
1.  Modify the INPUT_PARQUET_PATH variable below to point to your file.
2.  Run the script: python your_script_name.py
"""

import pandas as pd
import os
import sys

# We assume that the 'Test_predictions.py' file is in the same directory
# or accessible in the Python path.
try:
    from Test_predictions import test_by_range
except ImportError:
    print("ERROR: No se pudo importar 'test_by_range' desde 'Test_predictions.py'.")
    print("Por favor, asegúrate de que 'Test_predictions.py' esté en la misma carpeta que este script.")
    sys.exit(1)

# --- Configuration ---
# MODIFICA ESTA LÍNEA CON LA RUTA A TU ARCHIVO PARQUET
INPUT_PARQUET_PATH = "c:/Users/Ciro/Mi Unidad (PDI)/_UGR_PDI_2024-05-30/RELIEF-T1D/Python_projects/Glucose_prediction_2025/Test_scripts/Limit_predictions/df_test_results_vectors_RMSE_LSTM_H4.parquet"

# Clinical limits for glucose sensor readings.
MINIMUM_SENSOR_READING = 40
MAXIMUM_SENSOR_READING = 500


# --- End Configuration ---

def clip_and_evaluate(input_parquet_path: str):
    """
    Loads a parquet file, clips predictions, and recalculates metrics.

    Args:
        input_parquet_path (str): The full path to the input parquet file.
    """
    if not os.path.exists(input_parquet_path):
        print(f"Error: Archivo de entrada no encontrado en '{input_parquet_path}'")
        print("Por favor, actualiza la variable INPUT_PARQUET_PATH en el script.")
        return

    print(f"🚀 Procesando archivo: {os.path.basename(input_parquet_path)}")
    df = pd.read_parquet(input_parquet_path)

    if 'y_test' not in df.columns or 'y_predict' not in df.columns:
        print("Error: El archivo parquet debe contener las columnas 'y_test' y 'y_predict'.")
        return

    print(f"\n📊 Datos antes del recorte:")
    print(f"Total de predicciones: {len(df)}")

    predictions_above = df[df['y_predict'] > MAXIMUM_SENSOR_READING]
    predictions_below = df[df['y_predict'] < MINIMUM_SENSOR_READING]

    print(
        f"Predicciones por encima de {MAXIMUM_SENSOR_READING}: {len(predictions_above)} ({(len(predictions_above) / len(df) * 100):.2f}%)")
    if not predictions_above.empty:
        print(f"  - Valor máximo predicho: {predictions_above['y_predict'].max():.2f}")

    print(
        f"Predicciones por debajo de {MINIMUM_SENSOR_READING}: {len(predictions_below)} ({(len(predictions_below) / len(df) * 100):.2f}%)")
    if not predictions_below.empty:
        print(f"  - Valor mínimo predicho: {predictions_below['y_predict'].min():.2f}")

    df_clipped = df.copy()
    df_clipped['y_predict'] = df_clipped['y_predict'].clip(
        lower=MINIMUM_SENSOR_READING,
        upper=MAXIMUM_SENSOR_READING
    )

    print("\n✂️  Las predicciones han sido recortadas.")

    base_dir = os.path.dirname(input_parquet_path)
    file_name = os.path.splitext(os.path.basename(input_parquet_path))[0]
    output_dir = os.path.join(base_dir, f"{file_name}_clipped_results")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Los resultados se guardarán en: {output_dir}")

    clipped_parquet_path = os.path.join(output_dir, f"{file_name}_clipped.parquet")
    df_clipped.to_parquet(clipped_parquet_path)
    print(f"✅ Predicciones recortadas guardadas en: {os.path.basename(clipped_parquet_path)}")

    print("\n📈 Recalculando métricas con los datos recortados...")

    # --- CORRECCIÓN ---
    # Guardamos el directorio actual y nos movemos al de salida
    # para que test_by_range guarde los gráficos en el lugar correcto.
    original_cwd = os.getcwd()
    try:
        os.chdir(output_dir)

        metrics_suffix = "_clipped_ALL"

        df_metrics, df_zones, df_limits = test_by_range(
            df_clipped,
            metrics_suffix,
            # Se ha eliminado el argumento 'plot_dir'
            minimum_sensor_reading=MINIMUM_SENSOR_READING,
            maximum_sensor_reading=MAXIMUM_SENSOR_READING
        )

        df_metrics = df_metrics.round(2)

        # Los archivos se guardarán en el directorio actual (que ahora es output_dir)
        metrics_csv_path = f'metrics_performance_by_range{metrics_suffix}.csv'
        zones_csv_path = f'number_of_points_by_zones{metrics_suffix}.csv'
        limits_csv_path = f'predictions_out_of_limits{metrics_suffix}.csv'

        df_metrics.to_csv(metrics_csv_path, index=False)
        df_zones.to_csv(zones_csv_path, index=False)
        df_limits.to_csv(limits_csv_path, index=False)

        print("✅ Las nuevas métricas y el gráfico de Clarke han sido generados correctamente.")

    finally:
        # Volvemos al directorio original para no afectar otras operaciones
        os.chdir(original_cwd)
        print("\n🎉 Proceso completado.")


if __name__ == "__main__":
    clip_and_evaluate(INPUT_PARQUET_PATH)