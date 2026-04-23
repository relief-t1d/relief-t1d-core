import os
import glob
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

# ==============================================================================
# FUNCIONES DE EVALUACIÓN (CON LA CORRECCIÓN)
# ==============================================================================

def clarke_error_grid(ref_values,
                      pred_values,
                      title_string,
                      plot_dir='plots',
                      show_plot=False,
                      minimum_sensor_reading=40,
                      maximum_sensor_reading=500):
    """
    Esta función toma los valores de referencia y predicción y devuelve
    un recuento de puntos por zona (A, B, C, D, E) y genera el gráfico.
    """
    # Arrays compactos para bajar RAM
    ref_values = np.asarray(ref_values, dtype=np.float32)
    pred_values = np.asarray(pred_values, dtype=np.float32)

    assert (len(ref_values) == len(
        pred_values)), "Unequal number of values (reference : {0}) (prediction : {1}).".format(len(ref_values),
                                                                                               len(pred_values))

    # --- Comprobaciones de límites de sensor ---
    # (Se mantiene tu lógica original de advertencias)
    number_ref_values_low_limit = 0
    if len(ref_values) > 0 and min(ref_values) < minimum_sensor_reading:
        print(
            f'  Input Warning: minimum reference value ({min(ref_values):.2f}) is below sensor limit ({minimum_sensor_reading} mg/dL).')
        number_ref_values_low_limit = sum(value < minimum_sensor_reading for value in ref_values)
        print(f"  Number of reference values below limit: {number_ref_values_low_limit}")

    number_ref_values_up_limit = 0
    if len(ref_values) > 0 and max(ref_values) > maximum_sensor_reading:
        print(
            f'  Input Warning: maximum reference value ({max(ref_values):.2f}) is above sensor limit ({maximum_sensor_reading} mg/dL).')
        number_ref_values_up_limit = sum(value > maximum_sensor_reading for value in ref_values)
        print(f"  Number of reference values above limit: {number_ref_values_up_limit}")

    min_pred_value = np.nan
    number_pred_values_low_limit = 0
    if len(pred_values) > 0:
        min_pred_value = min(pred_values)
        min_pred_value = round(min_pred_value, 2)
        if min_pred_value < minimum_sensor_reading:
            print(
                f'  Input Warning: minimum predicted value ({min_pred_value:.2f}) is below sensor limit ({minimum_sensor_reading} mg/dL).')
            number_pred_values_low_limit = sum(value < minimum_sensor_reading for value in pred_values)
            print(f"  Number of predicted values below limit: {number_pred_values_low_limit}")

    max_pred_value = np.nan
    number_pred_values_up_limit = 0
    if len(pred_values) > 0:
        max_pred_value = max(pred_values)
        max_pred_value = round(max_pred_value, 2)
        if max_pred_value > maximum_sensor_reading:
            print(
                f'  Input Warning: maximum predicted value ({max_pred_value:.2f}) is above sensor limit ({maximum_sensor_reading} mg/dL).')
            number_pred_values_up_limit = sum(value > maximum_sensor_reading for value in pred_values)
            print(f"  Number of predicted values above limit: {number_pred_values_up_limit}")
    
    out_range_values = [min_pred_value, number_pred_values_low_limit, max_pred_value, number_pred_values_up_limit]
    # --- Fin de comprobaciones ---


    if show_plot:
        plt.clf()
        ax = plt.gca()
        colors = ['indianred', 'red', 'limegreen', 'yellow', 'orange']
        masks = (ref_values < 54,
                (ref_values >= 54) & (ref_values < 70),
                (ref_values >= 70) & (ref_values <= 180),
                (ref_values > 180) & (ref_values <= 250),
                (ref_values > 250),
                )

        for m, c in zip(masks, colors):
            if np.any(m):
                ax.scatter(ref_values[m], pred_values[m], marker='o', s=1, c=c, rasterized=True)
        
        plt.xlabel("Reference Concentration (mg/dl)")
        plt.ylabel("model_prediction_scripts Concentration (mg/dl)")
        plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        plt.gca().set_facecolor('white')
        plt.gca().set_xlim([0, 500])
        plt.gca().set_ylim([0, 500])
        plt.gca().set_aspect((500) / (500))

        # Plot zone lines (tu código original)
        plt.plot([0, 500], [0, 500], ':', c='black', linewidth=1)
        plt.plot([0, 175 / 3], [70, 70], '-', c='black', linewidth=1)
        plt.plot([175 / 3, 500 / 1.2], [70, 500], '-', c='black', linewidth=1)
        plt.plot([70, 70], [84, 500], '-', c='black', linewidth=1)
        plt.plot([0, 70], [180, 180], '-', c='black', linewidth=1)
        plt.plot([70, 290], [180, 500], '-', c='black', linewidth=1)
        plt.plot([70, 70], [0, 56], '-', c='black', linewidth=1)
        plt.plot([70, 500], [56, 400.79], '-', c='black', linewidth=1)
        plt.plot([180, 180], [0, 70], '-', c='black', linewidth=1)
        plt.plot([180, 500], [70, 70], '-', c='black', linewidth=1)
        plt.plot([240, 240], [70, 180], '-', c='black', linewidth=1)
        plt.plot([240, 500], [180, 180], '-', c='black', linewidth=1)
        plt.plot([130, 180], [0, 70], '-', c='black', linewidth=1)

        # Add zone titles
        plt.text(30, 15, "A", fontsize=15, fontweight='bold')
        plt.text(400, 250, "B", fontsize=15, fontweight='bold')
        plt.text(270, 400, "B", fontsize=15, fontweight='bold')
        plt.text(150, 400, "C", fontsize=15, fontweight='bold')
        plt.text(155, 15, "C", fontsize=15, fontweight='bold')
        plt.text(30, 120, "D", fontsize=15, fontweight='bold')
        plt.text(370, 120, "D", fontsize=15, fontweight='bold')
        plt.text(30, 340, "E", fontsize=15, fontweight='bold')
        plt.text(370, 15, "E", fontsize=15, fontweight='bold')

        os.makedirs(plot_dir, exist_ok=True)
        base = f'{plot_dir}/Clarke_Error_Grid{title_string}'
        plt.savefig(base + '.png', dpi=350, bbox_inches='tight')
        plt.savefig(base + '.pdf', bbox_inches='tight')
        
        import matplotlib as mpl; mpl.rcParams['svg.fonttype'] = 'none'
        plt.savefig(base + '.svg', bbox_inches='tight')
        plt.close()

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        # ---------------------------------------------------------------------------------
        # ZONE A (Clinical Accurate)
        # ---------------------------------------------------------------------------------
        if ((ref_values[i] < 70 and pred_values[i] < 70)  # Condition 1 for Zone A (LEFT BOTTOM)
                or
            (pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[i])):  # Condition 2 for Zone A (RIGHT DIAGONAL)
            zone[0] += 1

        # EXPLANATION:
        # (ref & pred < 70) - EXCLUSIVE: True to the clinical definition in original Clarke paper text.
        # (<= 1.2*ref and >= 0.8*ref) - INCLUSIVE: Captures the ±20% range in original Clarke paper text.


        # End of Zone A conditions --------------------

        # ---------------------------------------------------------------------------------
        # ZONE E (Erroneous Treatment)
        # ---------------------------------------------------------------------------------
        elif ((ref_values[i] > 180 and pred_values[i] < 70)  # Condition 1 for Zone E (BOTTOM RIGHT)
              or
              (ref_values[i] < 70 and pred_values[i] > 180)):  # Condition 2 for Zone E (TOP LEFT)

            zone[4] += 1  # Zone E

        # EXPLANATION:
        # (< 70 and > 180) - EXCLUSIVE: True to the original Clarke paper text.
        # A real value of 70 or 180 is considered "in range," not "hypo" or "hyper."
        # Therefore, (180, 69) or (70, 181) are not Zone E.

        # End of Zone E conditions --------------------

        # ---------------------------------------------------------------------------------
        # ZONE C (Overcorrection)
        # ---------------------------------------------------------------------------------
        elif (((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110)  # Condition 1 for Zone C (TOP)
              or
              ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7 / 5) * ref_values[i] - 182))):  # Condition 2 for Zone C (BOTTOM)

            zone[2] += 1  # Zone C

        # EXPLANATION:
        # Community consensus to use inclusive limits for Zone C conditions

        # End of Zone C conditions --------------------

        # ---------------------------------------------------------------------------------
        # ZONE D (Failure to Detect)
        # #---------------------------------------------------------------------------------
        elif ((ref_values[i] > 240 and (
                pred_values[i] >= 70 and pred_values[i] <= 180))  # Condition 1 for Zone D (RIGHT) # DUDA
              or
              (ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[i] >= 70)  # Condition 2 for Zone D (LEFT)
              or
              ((ref_values[i] >= 175 / 3 and ref_values[i] < 70) and pred_values[i] >= (6 / 5) * ref_values[i])):  # Condition 3 for Zone D (LEFT)

            zone[3] += 1

            # EXPLANATION OF THE CHANGES:
            # The original code used (ref <= 70) in condition D3.
            # This caused a point like (70, 90) to be classified as Zone D.
            # But (70, 90) is clinically a Zone B error,
            # since 70 is the edge of the target range, not hypoglycemia.
            # By changing it to (ref < 70), (70, 90) fails this condition and
            # correctly falls through to the 'else' (Zone B).
            #
            # Likewise, condition D1 was changed from (ref >= 240) to (ref > 240).
            # This change maintains consistency with the original Clarke paper text,
            # which textually defines this specific clinical failure as (ref > 240).
            #
            # The other prediction value limits (>= 70, <= 180) in D1 and D2 are correct because
            # they define the "target range" that the meter erroneously predicts.

            # End of Zone D conditions --------------------

        else:
            zone[1] += 1  # Zone B

    return zone, out_range_values


def test_by_range(df_result_vector: pd.DataFrame,
                  grid_name: str,
                  plot_dir: str = 'plots',
                  minimum_sensor_reading=40,
                  maximum_sensor_reading=500) -> tuple[DataFrame, DataFrame, DataFrame]:
    
    # --- Definición de rangos (tu código original) ---
    df_result_vector_TBR_2 = df_result_vector[df_result_vector['y_test'] < 54]

    df_result_vector_TBR_1 = df_result_vector[
        (df_result_vector['y_test'] >= 54) & (df_result_vector['y_test'] < 70)]

    df_result_vector_TIR = df_result_vector[(df_result_vector['y_test'] >= 70) & (df_result_vector['y_test'] < 181)]

    df_result_vector_TAR_1 = df_result_vector[
        (df_result_vector['y_test'] >= 181) & (df_result_vector['y_test'] < 251)]

    df_result_vector_TAR_2 = df_result_vector[(df_result_vector['y_test'] >= 251)]

    list_dataframe_by_range = [df_result_vector,
                               df_result_vector_TBR_2, df_result_vector_TBR_1,
                               df_result_vector_TIR,
                               df_result_vector_TAR_1, df_result_vector_TAR_2]
    list_dataframe_by_range_name = ['ENTIRE', 'TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']

    df_metrics_summary = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E', 'A + B', 'RMSE', 'MSE', 'MAE', 'MAPE'])
    df_zones_values = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E'])
    df_limits = pd.DataFrame(
        columns=['Range', 'Minimum value', 'Number values below limit', 'Maximum value', 'Number values above limit'])
    
    # --- Loop de evaluación por rangos ---
    for i in range(6):
        reference_values = list_dataframe_by_range[i]['y_test']
        pred_values = list_dataframe_by_range[i]['y_predict']
        print(f'  ------- Zone: {list_dataframe_by_range_name[i]} -------')
        
        # Evitar errores si un rango no tiene puntos
        if reference_values.shape[0] == 0:
            print("  No data points in this range. Skipping.")
            zone = [0] * 5
            zone_percentages = pd.Series([np.nan] * 5, index=['A', 'B', 'C', 'D', 'E'])
            mse, rmse, mae, mape = [np.nan] * 4
            out_of_range_values = [np.nan] * 4
        else:
            if list_dataframe_by_range_name[i] == 'ENTIRE':  # Plot the graph for the ENTIRE zone
                zone, out_of_range_values = clarke_error_grid(ref_values=reference_values.values,
                                                              pred_values=pred_values.values,
                                                              title_string=grid_name,
                                                              plot_dir=plot_dir,
                                                              show_plot=True,
                                                              minimum_sensor_reading=minimum_sensor_reading,
                                                              maximum_sensor_reading=maximum_sensor_reading)
            else:  # Do not plot the graph for the other zones
                zone, out_of_range_values = clarke_error_grid(ref_values=reference_values.values,
                                                              pred_values=pred_values.values,
                                                              title_string=grid_name + list_dataframe_by_range_name[i],
                                                              plot_dir=plot_dir,
                                                              show_plot=False,
                                                              minimum_sensor_reading=minimum_sensor_reading,
                                                              maximum_sensor_reading=maximum_sensor_reading)

            # Clinical metrics
            print(f'  Zones: [A, B, C, D, E]')
            print(f'  Number of points by zone: {zone}')
            zone_percentages = round(pd.Series(zone, index=['A', 'B', 'C', 'D', 'E']) / reference_values.shape[0] * 100, 2)
            print(f'  Percentage of point by zone: {zone_percentages.tolist()}')
            print()

            # Non-clinical metrics
            mse = mean_squared_error(reference_values, pred_values)
            rmse = np.sqrt(mean_squared_error(reference_values, pred_values))
            mae = mean_absolute_error(reference_values, pred_values)
            mape = mean_absolute_percentage_error(reference_values, pred_values)

            print(f'  Mean Squared Error (MSE): {mse:.2f}')
            print(f'  Root Mean Squared Error (RMSE): {rmse:.2f}')
            print(f'  Mean absolute Error (MAE): {mae:.2f}')
            print(f'  Mean absolute percentage Error (MAPE): {mape:.2f}')

        # Guardar resultados
        new_row_zone_values = {'Range': list_dataframe_by_range_name[i],
                               'A': zone[0], 'B': zone[1], 'C': zone[2], 'D': zone[3], 'E': zone[4]}
        df_zones_values.loc[len(df_zones_values)] = new_row_zone_values

        new_row_metrics = {'Range': list_dataframe_by_range_name[i],
                           'A': zone_percentages['A'], 'B': zone_percentages['B'], 'C': zone_percentages['C'], 'D': zone_percentages['D'], 'E': zone_percentages['E'],
                           'A + B': zone_percentages['A'] + zone_percentages['B'],
                           'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE': mape}
        df_metrics_summary.loc[len(df_metrics_summary)] = new_row_metrics

        new_row_limits = {'Range': list_dataframe_by_range_name[i],
                          'Minimum value': out_of_range_values[0], 'Number values below limit': out_of_range_values[1],
                          'Maximum value': out_of_range_values[2], 'Number values above limit': out_of_range_values[3]}
        df_limits.loc[len(df_limits)] = new_row_limits

        print()
        print('  ' + '-' * 98)
        print()

    return df_metrics_summary, df_zones_values, df_limits

# ==============================================================================
# SCRIPT PRINCIPAL DE AUTOMATIZACIÓN
# ==============================================================================

if __name__ == "__main__":

    # Directorio donde se guardarán todas las nuevas evaluaciones
    NEW_OUTPUT_DIR = "EVALUACION_CORREGIDA_2025-11-11"
    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)
    
    # Directorio para los nuevos plots (se creará dentro de NEW_OUTPUT_DIR)
    PLOT_OUTPUT_DIR = os.path.join(NEW_OUTPUT_DIR, "plots_corregidos")
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

    print(f"Iniciando re-evaluación con lógica de Zona D corregida.")
    print(f"Los resultados se guardarán en: {NEW_OUTPUT_DIR}")
    print(f"Los gráficos se guardarán en: {PLOT_OUTPUT_DIR}")

    # 1. Encontrar todos los archivos parquet de predicciones
    #    Buscamos recursivamente dentro de todas las carpetas EXP-*
    search_pattern = "EXP-*/results/predictions/df_test_results_vectors_*.parquet"
    all_parquet_files = glob.glob(search_pattern, recursive=True)

    if not all_parquet_files:
        print(f"Error: No se encontraron archivos parquet con el patrón: {search_pattern}")
        print("Asegúrate de ejecutar este script desde el directorio raíz que contiene las carpetas EXP-")
        exit()

    # 2. Filtrar los archivos: solo procesar los que NO contienen "_Fold"
    filtered_parquet_files = [f for f in all_parquet_files if "_Fold" not in f]
    
    print(f"\nSe encontraron {len(all_parquet_files)} archivos parquet en total.")
    print(f"Se procesarán {len(filtered_parquet_files)} archivos (ignorando los que contienen '_Fold').\n")

    # 3. Compilar regex para extraer el nombre del identificador
    #    Ej: de "df_test_results_vectors_LSTM_H2.parquet" -> extrae "LSTM_H2"
    parquet_name_pattern = re.compile(r"df_test_results_vectors_(.*)\.parquet")

    # 4. Iterar sobre cada archivo filtrado, procesarlo y guardar los resultados
    total_files_to_process = len(filtered_parquet_files)
    for i, path_predictions in enumerate(filtered_parquet_files):
        print("\n" + "="*100)
        print(f"Procesando archivo {i+1}/{total_files_to_process}: {path_predictions}")

        file_name = os.path.basename(path_predictions)
        match = parquet_name_pattern.search(file_name)

        if not match:
            print(f"  -> ADVERTENCIA: No se pudo extraer el identificador del archivo: {file_name}. Saltando.")
            continue

        # Este es el identificador único, ej: "LSTM_H2"
        file_identifier = match.group(1)
        print(f"  -> Identificador del experimento: {file_identifier}")

        try:
            # Cargar los datos de predicciones
            df_test_results = pd.read_parquet(path_predictions)

            # Definir los parámetros para la evaluación (fijos según tu ejemplo)
            minimum_sensor_reading = 40
            maximum_sensor_reading = 500
            
            # El grid_name se usa para el título del plot
            grid_name = f"_{file_identifier}"

            # Ejecutar la evaluación por rangos con el código CORREGIDO
            df_metrics_by_range, df_zone_values, df_limits = test_by_range(
                df_result_vector=df_test_results,
                grid_name=grid_name,
                plot_dir=PLOT_OUTPUT_DIR,
                minimum_sensor_reading=minimum_sensor_reading,
                maximum_sensor_reading=maximum_sensor_reading
            )
            
            df_metrics_by_range = df_metrics_by_range.round(2)

            # 5. Guardar los 3 CSV en el nuevo directorio
            
            # Construir los nombres de archivo de salida
            metrics_file = f"metrics_performance_by_range_{file_identifier}.csv"
            zones_file = f"number_of points_by_zones_{file_identifier}.csv"
            limits_file = f"predictions_out_of_limits_{file_identifier}.csv"

            # Guardar los archivos CSV
            df_metrics_by_range.to_csv(os.path.join(NEW_OUTPUT_DIR, metrics_file), index=False)
            df_zone_values.to_csv(os.path.join(NEW_OUTPUT_DIR, zones_file), index=False)
            df_limits.to_csv(os.path.join(NEW_OUTPUT_DIR, limits_file), index=False)
            
            print(f"  -> ¡Éxito! Se guardaron 3 CSVs corregidos en {NEW_OUTPUT_DIR}")

        except Exception as e:
            print(f"  -> ERROR procesando el archivo {file_name}: {e}")
            print("     Saltando al siguiente archivo.")
            continue
    
    print("\n" + "="*100)
    print("Re-evaluación completada.")