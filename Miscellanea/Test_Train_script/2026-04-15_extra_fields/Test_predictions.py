from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras
import pandas as pd
from pandas import DataFrame

import os
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'

 # Just paint the grid
def clarke_error_grid_graphic(ref_values,
                              pred_values,
                              title_string,
                              plot_dir='plots',
                             ):


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
    plt.xticks(np.arange(0, 551, 50))
    plt.yticks(np.arange(0, 551, 50))

    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    plt.gca().set_facecolor('white')
    plt.gca().set_xlim([0, 500])
    plt.gca().set_ylim([0, 500])
    plt.gca().set_aspect((500) / (500))

    # Plot zone lines
    plt.plot([0, 500], [0, 500], ':', c='black', linewidth=1)  # Línea 45° (y = x)

    # --- LÍMITES DE ZONA D/E INFERIOR ---
    plt.plot([0, 175 / 3], [70, 70], '-', c='black', linewidth=1)  # Línea horizontal inferior izquierda (E)
    # plt.plot([175/3, 320], [70, 500], '-', c='black', linewidth=1)
    plt.plot([175 / 3, 500 / 1.2], [70, 500], '-', c='black', linewidth=1)  # Límite superior de zona A (y = 1.2x)

    # --- LÍMITES VERTICALES IZQUIERDA ---
    plt.plot([70, 70], [84, 500], '-', c='black', linewidth=1)  # Límite vertical izquierdo (zona B/C/E)
    plt.plot([0, 70], [180, 180], '-', c='black', linewidth=1)  # Línea horizontal izquierda (C/E)

    # 🔧 CORRECCIÓN #1 — línea de zona C superior
    plt.plot([70, 390], [180, 500], '-', c='black', linewidth=1)  # ← antes [70, 290], ajustada a 500x500

    # plt.plot([70, 70], [0, 175/3], '-', c='black', linewidth=1)
    plt.plot([70, 70], [0, 56], '-', c='black', linewidth=1)  # Límite A/C inferior (y=0.8x en x=70)

    # 🔧 CORRECCIÓN #2 — línea de zona A inferior
    plt.plot([70, 500], [56, 400], '-', c='black', linewidth=1)  # ← antes 400.79, corregido exacto a y=0.8x

    # --- RESTO DE LÍNEAS (CORRECTAS) ---
    plt.plot([180, 180], [0, 70], '-', c='black', linewidth=1)
    plt.plot([180, 500], [70, 70], '-', c='black', linewidth=1)
    plt.plot([240, 240], [70, 180], '-', c='black', linewidth=1)
    plt.plot([240, 500], [180, 180], '-', c='black', linewidth=1)
    plt.plot([130, 180], [0, 70], '-', c='black', linewidth=1)

    # Add zone titles
    plt.text(380, 420, "A", fontsize=12)
    plt.text(420, 380, "A", fontsize=12)
    plt.text(170, 100, "B", fontsize=12)
    plt.text(100, 170, "B", fontsize=12)
    plt.text(150, 400, "C", fontsize=12)
    plt.text(155, 15, "C", fontsize=12)
    plt.text(30, 120, "D", fontsize=12)
    plt.text(370, 120, "D", fontsize=12)
    plt.text(30, 340, "E", fontsize=12)
    plt.text(370, 15, "E", fontsize=12)

    base = f'{plot_dir}/Clarke_Error_Grid{title_string}'
    plt.savefig(base + '.png', dpi=350, bbox_inches='tight')
    plt.savefig(base + '.pdf', bbox_inches='tight')

    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.savefig(base + '.svg', bbox_inches='tight')

    plt.close()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def clarke_error_grid_hexbin(ref_values,
                             pred_values,
                             title_string,
                             plot_dir='plots'):
    plt.clf()
    ax = plt.gca()

    # Configuración inicial del gráfico
    plt.gca().set_facecolor('white')
    plt.gca().set_xlim([0, 500])
    plt.gca().set_ylim([0, 500])
    plt.gca().set_aspect('equal')  # Importante para que los hexágonos sean regulares

    # -----------------------------------------------------------
    # 1. IMPLEMENTACIÓN HEXBIN (Reemplaza al scatter loop)
    # -----------------------------------------------------------

    # gridsize: ajusta este número. 50 suele ir bien (divide el eje en 50 hexágonos).
    # cmap: 'inferno', 'viridis', 'cividis' funcionan bien. 'inferno' resalta mucho los picos.
    # mincnt=1: hace que las zonas sin puntos sean transparentes (blancas).
    # bins='log': CRÍTICO. Usa escala logarítmica para el color. Si no lo usas,
    #             el pico de densidad ocultará los puntos raros (zonas D/E).

    hb = plt.hexbin(ref_values, pred_values,
                    gridsize=50,
                    cmap='inferno',
                    mincnt=1,
                    bins='log',
                    edgecolors='none')  # 'none' quita bordes al hexágono para suavizar

    # Añadimos la barra de color para entender la densidad
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Densidad de puntos (log10)')

    # --- SECCIÓN ELIMINADA (Scatter manual por rangos) ---
    # colors = ['indianred', 'red', 'limegreen', 'yellow', 'orange']
    # masks = (...)
    # for m, c in zip(masks, colors):
    #     if np.any(m):
    #         ax.scatter(...)
    # -----------------------------------------------------

    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("model_prediction_scripts Concentration (mg/dl)")
    plt.xticks(np.arange(0, 551, 50))
    plt.yticks(np.arange(0, 551, 50))

    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    # -----------------------------------------------------------
    # 2. DIBUJO DE LÍNEAS (Tu código original intacto)
    # -----------------------------------------------------------
    # Plot zone lines
    plt.plot([0, 500], [0, 500], ':', c='black', linewidth=1)  # Línea 45°

    # LÍMITES DE ZONA D/E INFERIOR
    plt.plot([0, 175 / 3], [70, 70], '-', c='black', linewidth=1)
    plt.plot([175 / 3, 500 / 1.2], [70, 500], '-', c='black', linewidth=1)

    # LÍMITES VERTICALES IZQUIERDA
    plt.plot([70, 70], [84, 500], '-', c='black', linewidth=1)
    plt.plot([0, 70], [180, 180], '-', c='black', linewidth=1)

    # CORRECCIÓN #1
    plt.plot([70, 390], [180, 500], '-', c='black', linewidth=1)

    plt.plot([70, 70], [0, 56], '-', c='black', linewidth=1)

    # CORRECCIÓN #2
    plt.plot([70, 500], [56, 400], '-', c='black', linewidth=1)

    # RESTO DE LÍNEAS
    plt.plot([180, 180], [0, 70], '-', c='black', linewidth=1)
    plt.plot([180, 500], [70, 70], '-', c='black', linewidth=1)
    plt.plot([240, 240], [70, 180], '-', c='black', linewidth=1)
    plt.plot([240, 500], [180, 180], '-', c='black', linewidth=1)
    plt.plot([130, 180], [0, 70], '-', c='black', linewidth=1)

    # Add zone titles (Te sugiero ponerles un fondo blanco o un borde para que se lean sobre el color)
    zones = [
        (380, 420, "A"), (420, 380, "A"), (170, 100, "B"), (100, 170, "B"),
        (150, 400, "C"), (155, 15, "C"), (30, 120, "D"), (370, 120, "D"),
        (30, 340, "E"), (370, 15, "E")
    ]
    for x, y, txt in zones:
        plt.text(x, y, txt, fontsize=12, color='black', weight='bold')

    base = f'{plot_dir}/Clarke_Error_Grid_Hexbin{title_string}'
    plt.savefig(base + '.png', dpi=350, bbox_inches='tight')
    plt.savefig(base + '.pdf', bbox_inches='tight')

    import matplotlib as mpl
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.savefig(base + '.svg', bbox_inches='tight')

    plt.close()

def clarke_error_grid(ref_values,
                      pred_values,
                      title_string,
                      plot_dir='plots',
                      show_plot=False,
                      minimum_sensor_reading=40,
                      maximum_sensor_reading=500):
    """
      This function takes in the reference values and the prediction values as lists and returns a list with each index corresponding to the total number
     of points within that zone (0=A, 1=B, 2=C, 3=D, 4=E) and the plot
    """
    # Checking to see if the lengths of the reference and prediction arrays are the same
    # Arrays compactos para bajar RAM
    ref_values = np.asarray(ref_values, dtype=np.float32)
    pred_values = np.asarray(pred_values, dtype=np.float32)

    assert (len(ref_values) == len(
        pred_values)), "Unequal number of values (reference : {0}) (prediction : {1}).".format(len(ref_values),
                                                                                               len(pred_values))

    # Checks to see if the values are within the normal sensor measurement range, otherwise it gives a warning
    # Reference values
    if min(ref_values) < minimum_sensor_reading:
        print(
            f'Input Warning: the minimum reference value ({min(ref_values):.2f}) is below the limit of sensor design (Inferior limit: {minimum_sensor_reading} mg/dL).')
        number_ref_values_low_limit = sum(value < minimum_sensor_reading for value in ref_values)
        print(f"Number of reference values below the limit: {number_ref_values_low_limit}")
        print()

    if max(ref_values) > maximum_sensor_reading:
        print(
            f'Input Warning: the maximum reference value ({max(ref_values):.2f}) is above the limit of sensor design (Superior limit: {maximum_sensor_reading} mg/dL).')
        number_ref_values_up_limit = sum(value > maximum_sensor_reading for value in ref_values)
        print(f"Number of reference values above the limit: {number_ref_values_up_limit}")
        print()

    # Predicted values
    min_ref_value = min(pred_values)
    min_ref_value = round(min_ref_value, 2)
    number_pred_values_low_limit = 0
    if min_ref_value < minimum_sensor_reading:
        print(
            f'Input Warning: the minimum predicted value ({min_ref_value:.2f}) is below the minimum input value (Inferior limit: {minimum_sensor_reading} mg/dL).')
        number_pred_values_low_limit = sum(value < minimum_sensor_reading for value in pred_values)
        # list_of_values_low_limit = [value for value in pred_values if value < minimum_sensor_reading]
        print(f"Number of predicted values below the limit: {number_pred_values_low_limit}")
        print()

    max_ref_value = max(pred_values)
    max_ref_value = round(max_ref_value, 2)
    number_pred_values_up_limit = 0
    if max_ref_value > maximum_sensor_reading:
        print(
            f'Input Warning: the maximum predicted value ({max_ref_value:.2f}) is above the maximum input value (Superior limit: {maximum_sensor_reading} mg/dL).')
        number_pred_values_up_limit = sum(value > maximum_sensor_reading for value in pred_values)
        # list_of_values_up_limit = [value for value in pred_values if value > maximum_sensor_reading]
        print(f"Number of predicted values above the limit: {number_pred_values_up_limit}")
        print()

    # list of out of range values
    out_range_values = [min_ref_value, number_pred_values_low_limit, max_ref_value, number_pred_values_up_limit]

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
        # plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        # plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        plt.xticks(np.arange(0, 551, 50))
        plt.yticks(np.arange(0, 551, 50))

        # plt.xticks([0, 20, 60, 100, 140, 180, 220, 260, 300, 400])
        # plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])

        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

        plt.gca().set_facecolor('white')
        plt.gca().set_xlim([0, 500])
        plt.gca().set_ylim([0, 500])
        plt.gca().set_aspect((500) / (500))

        # Plot zone lines
        plt.plot([0, 500], [0, 500], ':', c='black', linewidth=1)  # Línea 45° (y = x)

        # --- LÍMITES DE ZONA D/E INFERIOR ---
        plt.plot([0, 175 / 3], [70, 70], '-', c='black', linewidth=1)  # Línea horizontal inferior izquierda (E)
        # plt.plot([175/3, 320], [70, 500], '-', c='black', linewidth=1)
        plt.plot([175 / 3, 500 / 1.2], [70, 500], '-', c='black', linewidth=1)  # Límite superior de zona A (y = 1.2x)

        # --- LÍMITES VERTICALES IZQUIERDA ---
        plt.plot([70, 70], [84, 500], '-', c='black', linewidth=1)  # Límite vertical izquierdo (zona B/C/E)
        plt.plot([0, 70], [180, 180], '-', c='black', linewidth=1)  # Línea horizontal izquierda (C/E)

        # 🔧 CORRECCIÓN #1 — línea de zona C superior
        plt.plot([70, 390], [180, 500], '-', c='black', linewidth=1)  # ← antes [70, 290], ajustada a 500x500

        # plt.plot([70, 70], [0, 175/3], '-', c='black', linewidth=1)
        plt.plot([70, 70], [0, 56], '-', c='black', linewidth=1)  # Límite A/C inferior (y=0.8x en x=70)

        # 🔧 CORRECCIÓN #2 — línea de zona A inferior
        plt.plot([70, 500], [56, 400], '-', c='black', linewidth=1)  # ← antes 400.79, corregido exacto a y=0.8x

        # --- RESTO DE LÍNEAS (CORRECTAS) ---
        plt.plot([180, 180], [0, 70], '-', c='black', linewidth=1)
        plt.plot([180, 500], [70, 70], '-', c='black', linewidth=1)
        plt.plot([240, 240], [70, 180], '-', c='black', linewidth=1)
        plt.plot([240, 500], [180, 180], '-', c='black', linewidth=1)
        plt.plot([130, 180], [0, 70], '-', c='black', linewidth=1)

        # plt.plot([450, 450], [0, 500], '-', c='red', linewidth=1)
        # plt.plot([0, 500], [400, 400], '-', c='red', linewidth=1)

        # Add zone titles
        plt.text(380, 420, "A", fontsize=12)
        plt.text(420, 380, "A", fontsize=12)
        plt.text(170, 100, "B", fontsize=12)
        plt.text(100, 170, "B", fontsize=12)
        plt.text(150, 400, "C", fontsize=12)
        plt.text(155, 15, "C", fontsize=12)
        plt.text(30, 120, "D", fontsize=12)
        plt.text(370, 120, "D", fontsize=12)
        plt.text(30, 340, "E", fontsize=12)
        plt.text(370, 15, "E", fontsize=12)

        base = f'{plot_dir}/Clarke_Error_Grid{title_string}'
        plt.savefig(base + '.png', dpi=350, bbox_inches='tight')
        plt.savefig(base + '.pdf', bbox_inches='tight')

        import matplotlib as mpl
        mpl.rcParams['svg.fonttype'] = 'none'
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
                (pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[
                    i])):  # Condition 2 for Zone A (RIGHT DIAGONAL)
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
        elif (((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[
            i] + 110)  # Condition 1 for Zone C (TOP)
              or
              ((ref_values[i] >= 130 and ref_values[i] <= 180) and (
                      pred_values[i] <= (7 / 5) * ref_values[i] - 182))):  # Condition 2 for Zone C (BOTTOM)

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
              (ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[
                  i] >= 70)  # Condition 2 for Zone D (LEFT)
              or
              ((ref_values[i] >= 175 / 3 and ref_values[i] < 70) and pred_values[i] >= (6 / 5) * ref_values[
                  i])):  # Condition 3 for Zone D (LEFT)

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


def clarke_error_grid_normalized_by_dataset(ref_values,
                                            pred_values,
                                            title_string,
                                            plot_dir='Final_test',
                                            vmin_manual=None,
                                            vmax_manual=None,
                                            show_plot=False,
                                            minimum_sensor_reading=40,
                                            maximum_sensor_reading=500):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Saturate values
    ref_values = ref_values.clip(0, 500)
    pred_values = pred_values.clip(0, 500)

    # Checking to see if the lengths of the reference and prediction arrays are the same
    # Arrays compactos para bajar RAM
    ref_values = np.asarray(ref_values, dtype=np.float32)
    pred_values = np.asarray(pred_values, dtype=np.float32)

    assert (len(ref_values) == len(
        pred_values)), "Unequal number of values (reference : {0}) (prediction : {1}).".format(len(ref_values),
                                                                                               len(pred_values))

    # Checks to see if the values are within the normal sensor measurement range, otherwise it gives a warning
    # Reference values
    if min(ref_values) < minimum_sensor_reading:
        print(
            f'Input Warning: the minimum reference value ({min(ref_values):.2f}) is below the limit of sensor design (Inferior limit: {minimum_sensor_reading} mg/dL).')
        number_ref_values_low_limit = sum(value < minimum_sensor_reading for value in ref_values)
        print(f"Number of reference values below the limit: {number_ref_values_low_limit}")
        print()

    if max(ref_values) > maximum_sensor_reading:
        print(
            f'Input Warning: the maximum reference value ({max(ref_values):.2f}) is above the limit of sensor design (Superior limit: {maximum_sensor_reading} mg/dL).')
        number_ref_values_up_limit = sum(value > maximum_sensor_reading for value in ref_values)
        print(f"Number of reference values above the limit: {number_ref_values_up_limit}")
        print()

    # Predicted values
    min_ref_value = min(pred_values)
    min_ref_value = round(min_ref_value, 2)
    number_pred_values_low_limit = 0
    if min_ref_value < minimum_sensor_reading:
        print(
            f'Input Warning: the minimum predicted value ({min_ref_value:.2f}) is below the minimum input value (Inferior limit: {minimum_sensor_reading} mg/dL).')
        number_pred_values_low_limit = sum(value < minimum_sensor_reading for value in pred_values)
        # list_of_values_low_limit = [value for value in pred_values if value < minimum_sensor_reading]
        print(f"Number of predicted values below the limit: {number_pred_values_low_limit}")
        print()

    max_ref_value = max(pred_values)
    max_ref_value = round(max_ref_value, 2)
    number_pred_values_up_limit = 0
    if max_ref_value > maximum_sensor_reading:
        print(
            f'Input Warning: the maximum predicted value ({max_ref_value:.2f}) is above the maximum input value (Superior limit: {maximum_sensor_reading} mg/dL).')
        number_pred_values_up_limit = sum(value > maximum_sensor_reading for value in pred_values)
        # list_of_values_up_limit = [value for value in pred_values if value > maximum_sensor_reading]
        print(f"Number of predicted values above the limit: {number_pred_values_up_limit}")
        print()

    # list of out of range values
    out_range_values = [min_ref_value, number_pred_values_low_limit, max_ref_value, number_pred_values_up_limit]

    if show_plot:
        # Plot start
        # plt.clf()
        fig, ax = plt.subplots()

        # --- CONFIGURACIÓN DE BINS ---
        n_bins = 500
        my_range = [[0, 500], [0, 500]]
        my_bins = [n_bins, n_bins]

        # --- CÁLCULO DE LÍMITES (AUTOMÁTICO O MANUAL) ---
        # Si nos pasan los límites (vmin_manual), los usamos. Si no, calculamos.
        if vmin_manual is not None and vmax_manual is not None:
            vmin_global = vmin_manual
            vmax_global = vmax_manual
        else:
            # Cálculo local estándar si no se comparte escala
            h_temp = np.histogram2d(ref_values, pred_values, bins=my_bins, range=my_range)
            # El count mínimo siempre es 1 (para log), el máximo es el max del histograma
            vmin_global = 1
            vmax_global = np.max(h_temp[0]) if np.max(h_temp[0]) > 0 else 10

        # Normalización compartida
        global_norm = LogNorm(vmin=vmin_global, vmax=vmax_global)

        # --- DEFINICIÓN DE COLORES ---
        cmap_tbr2 = LinearSegmentedColormap.from_list("custom_indian", ["#ffffff", "indianred", "#8B3A3A"])
        cmap_tbr1 = LinearSegmentedColormap.from_list("custom_red", ["#ffffff", "red", "#8B0000"])
        cmap_tir = LinearSegmentedColormap.from_list("custom_lime", ["#ffffff", "limegreen", "#006400"])
        cmap_tar1 = LinearSegmentedColormap.from_list("custom_yellow", ["#ffffff", "yellow", "#808000"])
        cmap_tar2 = LinearSegmentedColormap.from_list("custom_orange", ["#ffffff", "orange", "#8B4513"])
        cmaps_ordered = [cmap_tbr2, cmap_tbr1, cmap_tir, cmap_tar1, cmap_tar2]

        # --- DIBUJO ---
        ranges = [
            (ref_values < 54, cmap_tbr2),
            ((ref_values >= 54) & (ref_values < 70), cmap_tbr1),
            ((ref_values >= 70) & (ref_values <= 180), cmap_tir),
            ((ref_values > 180) & (ref_values <= 250), cmap_tar1),
            (ref_values > 250, cmap_tar2)
        ]

        for mask, cmap in ranges:
            if np.any(mask):
                # Importante: pasamos norm=global_norm para respetar la escala compartida
                ax.hist2d(ref_values[mask], pred_values[mask],
                          bins=my_bins, range=my_range, cmap=cmap, cmin=1, norm=global_norm, zorder=2)

        # --- BARRA DE LEYENDA (Striped) ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)

        n_steps = 200
        Y_bounds = np.logspace(np.log10(vmin_global), np.log10(vmax_global), n_steps + 1)
        Z_values = np.logspace(np.log10(vmin_global), np.log10(vmax_global), n_steps)
        Z_slice = Z_values.reshape(-1, 1)

        for i in range(5):
            X_bounds = np.array([i, i + 1])
            X_mesh, Y_mesh = np.meshgrid(X_bounds, Y_bounds)
            cax.pcolormesh(X_mesh, Y_mesh, Z_slice, cmap=cmaps_ordered[i], norm=global_norm, shading='flat')

        cax.set_yscale('log')
        cax.set_xlim(0, 5)
        cax.yaxis.tick_right()
        # cax.yaxis.set_label_position("right")
        # cax.set_ylabel('Point Density (Log Scale)', rotation=90)
        cax.set_xticks([])  # Sin etiquetas abajo

        # --- ESTÉTICA FINAL (Inglés) ---
        lp = {'c': 'black', 'linewidth': 0.8, 'zorder': 10, 'alpha': 0.5}
        # lp = {'c': 'black', 'linewidth': 1}
        ax.plot([0, 500], [0, 500], ':', **lp)

        ax.plot([0, 175 / 3], [70, 70], '-', **lp)
        ax.plot([175 / 3, 500 / 1.2], [70, 500], '-', **lp)

        ax.plot([70, 70], [84, 500], '-', **lp)

        ax.plot([0, 70], [180, 180], '-', **lp)
        ax.plot([70, 390], [180, 500], '-', **lp)
        ax.plot([70, 70], [0, 56], '-', **lp)

        ax.plot([70, 500], [56, 400], '-', **lp)

        ax.plot([180, 180], [0, 70], '-', **lp)
        ax.plot([180, 500], [70, 70], '-', **lp)
        ax.plot([240, 240], [70, 180], '-', **lp)
        ax.plot([240, 500], [180, 180], '-', **lp)
        ax.plot([130, 180], [0, 70], '-', **lp)

        zones_ceg = [(380, 440, "A"), (440, 380, "A"), (170, 100, "B"), (100, 170, "B"),
                     (150, 400, "C"), (155, 15, "C"), (30, 120, "D"), (370, 120, "D"), (30, 340, "E"), (370, 15, "E")]
        for x, y, txt in zones_ceg:
            ax.text(x, y, txt, fontsize=12, fontweight='bold', color='black', alpha=0.6, zorder=11)
            # ax.text(x, y, txt, fontsize=12) #, fontweight='bold', color='black', alpha=0.6, zorder=11)

        ax.set_xlabel("Reference Concentration (mg/dl)")
        ax.set_ylabel("model_prediction_scripts Concentration (mg/dl)")
        ax.set_xticks(np.arange(0, 551, 50));
        ax.set_yticks(np.arange(0, 551, 50))
        ax.xaxis.set_minor_locator(MultipleLocator(10));
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlim([0, 501]);
        ax.set_ylim([-1, 500])
        ax.set_aspect('equal')

        base = f'{plot_dir}/CEG_{title_string}'
        plt.savefig(base + '.png', dpi=900, bbox_inches='tight')
        plt.savefig(base + '.pdf', bbox_inches='tight')
        plt.savefig(base + '.svg', bbox_inches='tight')
        
        plt.close('all')
        print(f"Generado: {base}")
        # Plot End

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        # ---------------------------------------------------------------------------------
        # ZONE A (Clinical Accurate)
        # ---------------------------------------------------------------------------------
        if ((ref_values[i] < 70 and pred_values[i] < 70)  # Condition 1 for Zone A (LEFT BOTTOM)
                or
                (pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[
                    i])):  # Condition 2 for Zone A (RIGHT DIAGONAL)
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
        elif (((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[
            i] + 110)  # Condition 1 for Zone C (TOP)
              or
              ((ref_values[i] >= 130 and ref_values[i] <= 180) and (
                      pred_values[i] <= (7 / 5) * ref_values[i] - 182))):  # Condition 2 for Zone C (BOTTOM)

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
              (ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[
                  i] >= 70)  # Condition 2 for Zone D (LEFT)
              or
              ((ref_values[i] >= 175 / 3 and ref_values[i] < 70) and pred_values[i] >= (6 / 5) * ref_values[
                  i])):  # Condition 3 for Zone D (LEFT)

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
                  show_plot: bool = False,
                  plot_dir: str = 'plots',
                  minimum_sensor_reading=40,
                  maximum_sensor_reading=500) -> tuple[DataFrame, DataFrame, DataFrame]:

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

    for i in range(6):
        reference_values = list_dataframe_by_range[i]['y_test']
        pred_values = list_dataframe_by_range[i]['y_predict']
        print(f'------- Zone: {list_dataframe_by_range_name[i]} -------')
        if list_dataframe_by_range_name[i] == 'ENTIRE':  # Plot the graph for the ENTIRE zone
            zone, out_of_range_values = clarke_error_grid_normalized_by_dataset(ref_values=reference_values.values,
                                                          pred_values=pred_values.values,
                                                          title_string=grid_name,
                                                          plot_dir=plot_dir,
                                                          show_plot=show_plot,
                                                          minimum_sensor_reading=minimum_sensor_reading,
                                                          maximum_sensor_reading=maximum_sensor_reading)
        else:  # Do not plot the graph for the other zones
            zone, out_of_range_values = clarke_error_grid_normalized_by_dataset(ref_values=reference_values.values,
                                                          pred_values=pred_values.values,
                                                          title_string=grid_name + list_dataframe_by_range_name[i],
                                                          plot_dir=plot_dir,
                                                          show_plot=False,
                                                          minimum_sensor_reading=minimum_sensor_reading,
                                                          maximum_sensor_reading=maximum_sensor_reading)

        # Clinical metrics
        print(f'Zones: [A, B, C, D, E]')
        print(f'Number of points by zone: {zone}')
        zone_percentages = round(pd.Series(zone) / reference_values.shape[0] * 100, 2)

        new_row_zone_values = {'Range': list_dataframe_by_range_name[i],
                               'A': zone[0],
                               'B': zone[1],
                               'C': zone[2],
                               'D': zone[3],
                               'E': zone[4]}
        new_index_zone_values = len(df_zones_values)
        df_zones_values.loc[new_index_zone_values] = new_row_zone_values

        print(f'Percentage of point by zone: {zone_percentages.tolist()}')
        print()

        # Non-clinical metrics
        mse = mean_squared_error(reference_values, pred_values)
        rmse = np.sqrt(mean_squared_error(reference_values, pred_values))
        mae = mean_absolute_error(reference_values, pred_values)
        mape = mean_absolute_percentage_error(reference_values, pred_values)

        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
        print(f'Mean absolute Error (MAE): {mae:.2f}')
        print(f'Mean absolute percentage Error (MAPE): {mape:.2f}')

        new_row_metrics = {'Range': list_dataframe_by_range_name[i],
                           'A': zone_percentages[0],
                           'B': zone_percentages[1],
                           'C': zone_percentages[2],
                           'D': zone_percentages[3],
                           'E': zone_percentages[4],
                           'A + B': zone_percentages[0] + zone_percentages[1],
                           'RMSE': rmse,
                           'MSE': mse,
                           'MAE': mae,
                           'MAPE': mape}
        new_index_metrics = len(df_metrics_summary)
        df_metrics_summary.loc[new_index_metrics] = new_row_metrics

        # Save the limits of the values
        new_row_limits = {'Range': list_dataframe_by_range_name[i],
                          'Minimum value': out_of_range_values[0],
                          'Number values below limit': out_of_range_values[1],
                          'Maximum value': out_of_range_values[2],
                          'Number values above limit': out_of_range_values[3]}
        new_index_limits = len(df_limits)
        df_limits.loc[new_index_limits] = new_row_limits

        print()
        print('-' * 100)
        print()

    return df_metrics_summary, df_zones_values, df_limits

def get_train_plots_loss(hist: keras.callbacks, name: str):
    fig, ax = plt.subplots()

    # data
    x_epoch = hist.epoch
    y_val_loss = hist.history['val_loss']
    y_train_loss = hist.history['loss']

    # Create a line plots
    ax.plot(x_epoch, y_val_loss, label='Validation loss', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_loss, label='Train loss', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and validation loss')

    ax.legend()

    fig.savefig(name + '_Loss.pdf', dpi=350, bbox_inches='tight')

def get_train_plots_RMSE(hist: keras.callbacks, name: str):
    fig, ax = plt.subplots()

    # Sample data
    x_epoch = hist.epoch
    y_val_rmse = hist.history['val_root_mean_squared_error']
    y_train_rmse = hist.history['root_mean_squared_error']

    # Create a line plot
    ax.plot(x_epoch, y_val_rmse, label='Validation RMSE', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_rmse, label='Train RMSE', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Training and validation RMSE')

    # Add a legend
    ax.legend()

    # Display the plot
    fig.savefig(name + '_RMSE.pdf', dpi=350, bbox_inches='tight')

def get_train_plots_MSE(hist: keras.callbacks, name: str):
    fig, ax = plt.subplots()

    # Sample data
    x_epoch = hist.epoch
    y_val_mse = hist.history['val_mean_squared_error']
    y_train_mse = hist.history['mean_squared_error']

    # Create a line plot
    ax.plot(x_epoch, y_val_mse, label='Validation MSE', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_mse, label='Train MSE', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Training and validation MSE')

    # Add a legend
    ax.legend()

    # Display the plot
    fig.savefig(name + '_MSE.pdf', dpi=350, bbox_inches='tight')

def fold_results_aggregation(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a DataFrame with results from multiple folds and aggregates the results by calculating mean and
    standard deviation by 'Range'.
    :param df: dataframe with results from multiple folds
    :return: two DataFrames: one with flat structure (suitable for CSV) and another formatted for reports
    """
    logical_order = ['ENTIRE', 'TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']

    # Select only numeric columns for calculation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ---------------------------------------------------------
    # APPROACH A: Flat CSV Structure (Numeric)
    # Best for saving to CSV and further analysis
    # ---------------------------------------------------------

    # 1. Group and calculate Mean and Std
    df_grouped = df.groupby('Range')[numeric_cols].agg(['mean', 'std'])

    # 2. FLATTEN THE COLUMNS (The key step)
    # This changes MultiIndex ('A', 'mean') to single index 'A_Mean'
    df_grouped.columns = [f"{col}_{stat.capitalize()}" for col, stat in df_grouped.columns]

    # 3. Reorder rows logically and reset index to make 'Range' a normal column
    df_flat = df_grouped.reindex(logical_order).reset_index()

    # print("\n=== OPTION A: Flat Structure (Ready for CSV) ===")

    # To save:
    # df_flat.to_csv('statistical_analysis.csv', index=False)

    # ---------------------------------------------------------
    # APPROACH B: Report Structure (Text format "Mean ± Std")
    # Best for presentations/publications
    # ---------------------------------------------------------

    df_report = pd.DataFrame(index=logical_order)
    df_report.index.name = 'Range'

    # Calculate base stats again (or reuse from above)
    grouped = df.groupby('Range')[numeric_cols]
    means = grouped.mean()
    stds = grouped.std()

    # Format columns
    for col in numeric_cols:
        df_report[col] = (
                means[col].map('{:.2f}'.format) +
                " ± " +
                stds[col].map('{:.2f}'.format)
        )

    df_report = df_report.reset_index()

    # print("\n=== OPTION B: Report Structure (Formatted) ===")

    # To save:
    # df_report.to_csv('final_report.csv', index=False)

    return df_flat, df_report


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator


def clarke_error_grid_density_colored(ref_values, pred_values, title_string, plot_dir='plots'):
    plt.clf()
    ax = plt.gca()

    # 1. Definimos los colores base que ya usabas
    base_colors = ['indianred', 'red', 'limegreen', 'gold', 'darkorange']

    # Creamos colormaps que van desde un tono muy claro (casi blanco) hasta el color saturado
    # 'white' -> color_base -> negro (para zonas de máxima densidad)
    cmaps = [
        LinearSegmentedColormap.from_list("c1", ["#fff5f5", "indianred", "#5c1a1a"]),
        LinearSegmentedColormap.from_list("c2", ["#fff0f0", "red", "#660000"]),
        LinearSegmentedColormap.from_list("c3", ["#f0fff0", "limegreen", "#004d00"]),
        LinearSegmentedColormap.from_list("c4", ["#fffff0", "yellow", "#666600"]),
        LinearSegmentedColormap.from_list("c5", ["#fff7e6", "orange", "#663300"]),
    ]

    masks = (
        ref_values < 54,
        (ref_values >= 54) & (ref_values < 70),
        (ref_values >= 70) & (ref_values <= 180),
        (ref_values > 180) & (ref_values <= 250),
        (ref_values > 250),
    )

    # 2. Dibujamos un Hexbin por cada rango de glucosa
    for m, cmap in zip(masks, cmaps):
        if np.any(m):
            # Usamos hexbin solo para los datos de esa máscara
            ax.hexbin(ref_values[m], pred_values[m],
                      gridsize=30,
                      cmap=cmap,
                      mincnt=1,
                      bins='log',  # Escala logarítmica para ver bien los extremos
                      edgecolors='none')

    # --- Configuración estética (Igual que tu código) ---
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("model_prediction_scripts Concentration (mg/dl)")
    plt.xticks(np.arange(0, 551, 50))
    plt.yticks(np.arange(0, 551, 50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().set_facecolor('white')
    plt.gca().set_xlim([0, 500])
    plt.gca().set_ylim([0, 500])
    plt.gca().set_aspect('equal')

    # 3. Dibujo de líneas del Clarke Error Grid (Tus líneas originales)
    line_params = {'c': 'black', 'linewidth': 1}
    plt.plot([0, 500], [0, 500], ':', **line_params)
    plt.plot([0, 175 / 3], [70, 70], '-', **line_params)
    plt.plot([175 / 3, 500 / 1.2], [70, 500], '-', **line_params)
    plt.plot([70, 70], [84, 500], '-', **line_params)
    plt.plot([0, 70], [180, 180], '-', **line_params)
    plt.plot([70, 390], [180, 500], '-', **line_params)
    plt.plot([70, 70], [0, 56], '-', **line_params)
    plt.plot([70, 500], [56, 400], '-', **line_params)
    plt.plot([180, 180], [0, 70], '-', **line_params)
    plt.plot([180, 500], [70, 70], '-', **line_params)
    plt.plot([240, 240], [70, 180], '-', **line_params)
    plt.plot([240, 500], [180, 180], '-', **line_params)
    plt.plot([130, 180], [0, 70], '-', **line_params)

    # Títulos de zonas
    zones = [(380, 420, "A"), (420, 380, "A"), (170, 100, "B"), (100, 170, "B"),
             (150, 400, "C"), (155, 15, "C"), (30, 120, "D"), (370, 120, "D"),
             (30, 340, "E"), (370, 15, "E")]
    for x, y, txt in zones:
        plt.text(x, y, txt, fontsize=12, fontweight='bold', alpha=0.7)

    # Guardar
    plt.savefig(f'{plot_dir}/CEG_Density_Colored_{title_string}.png', dpi=300, bbox_inches='tight')
    plt.close()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

def clarke_error_grid_uniform_hex(ref_values, pred_values, title_string, plot_dir='plots'):
    plt.clf()
    ax = plt.gca()

    # 1. Definimos los colormaps (de claro a oscuro) para cada rango
    cmaps = [
        LinearSegmentedColormap.from_list("c1", ["#fff5f5", "indianred", "#5c1a1a"]),
        LinearSegmentedColormap.from_list("c2", ["#fff0f0", "red", "#660000"]),
        LinearSegmentedColormap.from_list("c3", ["#f0fff0", "limegreen", "#004d00"]),
        LinearSegmentedColormap.from_list("c4", ["#fffff0", "yellow", "#666600"]),
        LinearSegmentedColormap.from_list("c5", ["#fff7e6", "orange", "#663300"]),
    ]

    masks = (
        ref_values < 54,
        (ref_values >= 54) & (ref_values < 70),
        (ref_values >= 70) & (ref_values <= 180),
        (ref_values > 180) & (ref_values <= 250),
        (ref_values > 250),
    )

    # --- PARÁMETROS DE UNIFORMIDAD ---
    # gridsize: Cuántos hexágonos caben en el eje X.
    # extent: [xmin, xmax, ymin, ymax]. Esto es lo que iguala el tamaño en todo el rango.
    g_size = 100
    g_extent = [0, 500, 0, 500]

    # 2. Dibujamos cada zona
    for m, cmap in zip(masks, cmaps):
        if np.any(m):
            ax.hexbin(ref_values[m], pred_values[m],
                      gridsize=g_size,
                      extent=g_extent, # <--- CRÍTICO: misma base para todos
                      cmap=cmap,
                      mincnt=1,
                      bins='log',
                      # bins=30,
                      edgecolors='none')

    # 3. Estética y límites (Tu configuración original)
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("model_prediction_scripts Concentration (mg/dl)")
    plt.xticks(np.arange(0, 551, 50))
    plt.yticks(np.arange(0, 551, 50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 500])
    ax.set_aspect('equal')

    # 4. Líneas del Clarke Error Grid
    lp = {'c': 'black', 'linewidth': 1, 'zorder': 5} # zorder alto para que las líneas queden sobre los hexágonos
    plt.plot([0, 500], [0, 500], ':', **lp)
    plt.plot([0, 175/3], [70, 70], '-', **lp)
    plt.plot([175/3, 500/1.2], [70, 500], '-', **lp)
    plt.plot([70, 70], [84, 500], '-', **lp)
    plt.plot([0, 70], [180, 180], '-', **lp)
    plt.plot([70, 390], [180, 500], '-', **lp)
    plt.plot([70, 70], [0, 56], '-', **lp)
    plt.plot([70, 500], [56, 400], '-', **lp)
    plt.plot([180, 180], [0, 70], '-', **lp)
    plt.plot([180, 500], [70, 70], '-', **lp)
    plt.plot([240, 240], [70, 180], '-', **lp)
    plt.plot([240, 500], [180, 180], '-', **lp)
    plt.plot([130, 180], [0, 70], '-', **lp)

    # Títulos de zonas
    zones = [(380,420,"A"),(420,380,"A"),(170,100,"B"),(100,170,"B"),
             (150,400,"C"),(155,15,"C"),(30,120,"D"),(370,120,"D"),
             (30,340,"E"),(370,15,"E")]
    for x, y, txt in zones:
        plt.text(x, y, txt, fontsize=12, fontweight='bold', alpha=0.6, zorder=6)

    plt.savefig(f'{plot_dir}/CEG_Perfect_Grid_{title_string}.png', dpi=300, bbox_inches='tight')
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator


def clarke_error_grid_heatmap(ref_values, pred_values, title_string, plot_dir='plots'):
    plt.clf()
    ax = plt.gca()

    # 1. Colormaps personalizados (de claro a muy oscuro)
    cmaps = [
        LinearSegmentedColormap.from_list("c1", ["#ffffff", "indianred", "#5c1a1a"]),
        LinearSegmentedColormap.from_list("c2", ["#ffffff", "red", "#660000"]),
        LinearSegmentedColormap.from_list("c3", ["#ffffff", "limegreen", "#004d00"]),
        LinearSegmentedColormap.from_list("c4", ["#ffffff", "yellow", "#666600"]),
        LinearSegmentedColormap.from_list("c5", ["#ffffff", "orange", "#663300"]),
    ]

    masks = (
        ref_values < 54,
        (ref_values >= 54) & (ref_values < 70),
        (ref_values >= 70) & (ref_values <= 180),
        (ref_values > 180) & (ref_values <= 250),
        (ref_values > 250),
    )

    # --- PARÁMETROS DEL HEATMAP ---
    # bins: define cuántos "píxeles" habrá. 100 hace que cada cuadrado sea de 5x5 mg/dl.
    # range: fuerza a que la rejilla se calcule de 0 a 500 para todas las capas.
    n_bins = 500
    h_range = [[0, 500], [0, 500]]

    # 2. Dibujamos un histograma 2D por cada zona
    for m, cmap in zip(masks, cmaps):
        if np.any(m):
            # cmin=1: hace que los cuadros con cero puntos sean blancos/transparentes
            ax.hist2d(ref_values[m], pred_values[m],
                      bins=[n_bins, n_bins],
                      range=h_range,
                      cmap=cmap,
                      cmin=1,
                      norm='log')  # Usamos escala logarítmica para la densidad

    # 3. Configuración de ejes y aspecto (Idéntico a tu original)
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("model_prediction_scripts Concentration (mg/dl)")
    plt.xticks(np.arange(0, 551, 50))
    plt.yticks(np.arange(0, 551, 50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 500])
    ax.set_aspect('equal')

    # 4. Líneas del Clarke Error Grid (Capa superior)
    lp = {'c': 'black', 'linewidth': 1, 'zorder': 10}
    plt.plot([0, 500], [0, 500], ':', **lp)
    plt.plot([0, 175 / 3], [70, 70], '-', **lp)
    plt.plot([175 / 3, 500 / 1.2], [70, 500], '-', **lp)
    plt.plot([70, 70], [84, 500], '-', **lp)
    plt.plot([0, 70], [180, 180], '-', **lp)
    plt.plot([70, 390], [180, 500], '-', **lp)
    plt.plot([70, 70], [0, 56], '-', **lp)
    plt.plot([70, 500], [56, 400], '-', **lp)
    plt.plot([180, 180], [0, 70], '-', **lp)
    plt.plot([180, 500], [70, 70], '-', **lp)
    plt.plot([240, 240], [70, 180], '-', **lp)
    plt.plot([240, 500], [180, 180], '-', **lp)
    plt.plot([130, 180], [0, 70], '-', **lp)

    # Títulos de zonas
    zones = [(380, 420, "A"), (420, 380, "A"), (170, 100, "B"), (100, 170, "B"),
             (150, 400, "C"), (155, 15, "C"), (30, 120, "D"), (370, 120, "D"),
             (30, 340, "E"), (370, 15, "E")]
    for x, y, txt in zones:
        plt.text(x, y, txt, fontsize=12, fontweight='bold', zorder=11)

    plt.savefig(f'{plot_dir}/CEG_Heatmap_{title_string}.png', dpi=300, bbox_inches='tight')
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import PowerNorm

def clarke_error_grid_heatmap_single(ref_values, pred_values, title_string, plot_dir='plots'):
    plt.clf()
    ax = plt.gca()

    # --- CONFIGURACIÓN DEL HEATMAP ---
    # Usamos un solo colormap. 'viridis', 'magma' o 'inferno' son excelentes para densidad.
    # n_bins=100 significa que cada cuadro es de 5x5 mg/dl (500/100).
    n_bins = 500
    h_range = [[0, 500], [0, 500]]

    # 1. Dibujamos el histograma 2D global
    # norm='log' es fundamental para ver errores raros y densidad alta a la vez.
    h = ax.hist2d(ref_values, pred_values,
                  bins=[n_bins, n_bins],
                  range=h_range,
                  cmap='YlOrRd',
                  cmin=1,
                  norm='log',
                  # alpha=0.8
                  )  # alpha para que las líneas del grid se vean mejor

    # Añadimos una barra de color para que se entienda la escala de densidad
    cb = plt.colorbar(h[3], ax=ax, pad=0.02)
    cb.set_label('Número de puntos (escala log)')

    # 2. Configuración de ejes (Tu estándar)
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("model_prediction_scripts Concentration (mg/dl)")
    plt.xticks(np.arange(0, 551, 50))
    plt.yticks(np.arange(0, 551, 50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_xlim([0, 500])
    ax.set_ylim([0, 500])
    ax.set_aspect('equal')

    # 3. Líneas del Clarke Error Grid (Z-order alto para que brillen sobre el mapa)
    lp = {'c': 'cyan', 'linewidth': 1, 'zorder': 10, 'alpha': 0.8}  # Cyan destaca bien sobre fondo oscuro

    plt.plot([0, 500], [0, 500], ':', **lp)
    plt.plot([0, 175 / 3], [70, 70], '-', **lp)
    plt.plot([175 / 3, 500 / 1.2], [70, 500], '-', **lp)
    plt.plot([70, 70], [84, 500], '-', **lp)
    plt.plot([0, 70], [180, 180], '-', **lp)
    plt.plot([70, 390], [180, 500], '-', **lp)
    plt.plot([70, 70], [0, 56], '-', **lp)
    plt.plot([70, 500], [56, 400], '-', **lp)
    plt.plot([180, 180], [0, 70], '-', **lp)
    plt.plot([180, 500], [70, 70], '-', **lp)
    plt.plot([240, 240], [70, 180], '-', **lp)
    plt.plot([240, 500], [180, 180], '-', **lp)
    plt.plot([130, 180], [0, 70], '-', **lp)

    # Títulos de zonas
    zones = [(380, 420, "A"), (420, 380, "A"), (170, 100, "B"), (100, 170, "B"),
             (150, 400, "C"), (155, 15, "C"), (30, 120, "D"), (370, 120, "D"),
             (30, 340, "E"), (370, 15, "E")]
    for x, y, txt in zones:
        plt.text(x, y, txt, fontsize=12, fontweight='bold', color='Black', zorder=11)

    plt.savefig(f'{plot_dir}/CEG_Heatmap_Global_{title_string}.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = pd.read_parquet('c:/Users/Ciro/C/_UGR_PDI/RELIEF-T1D/Glucose_prediction_2026/df_test_results_vectors_RMSE_LSTM_H4.parquet')
    clarke_error_grid_heatmap_single(ref_values=df['y_test'],
                                     pred_values=df['y_predict'],
                                     title_string='Test_heatmap_single_log_YlOrRd_bins500',)
    # path_predictions = 'c:/Users/Ciro/Mi Unidad (PDI)/_UGR_PDI_2024-05-30/RELIEF-T1D/Python_projects/Glucose_prediction_2025/Custom_loss_functions/Exp1/test_results/df_test_results_vectors_CNN_H2_EloyPenaltyp01_Fold1.parquet'
    # path_predictions = 'test_results/df_test_results_vectors_LSTM_H2.parquet'

    # df_test_results = pd.DataFrame({'y_test': [0], 'y_predict': [0]})
    #
    # # df_test_results = pd.read_parquet(path_predictions)
    #
    # current_algorithm = 'CNN_500'  # Change this to the algorithm you are using
    # current_horizon = 2  # Change this to the horizon you are using
    # minimum_sensor_reading = 38.96  # Minimum sensor reading limit
    # maximum_sensor_reading = 401  # Maximum sensor reading limit
    #
    # df_metrics_by_range, df_zone_values, df_limits = test_by_range(df_result_vector=df_test_results,
    #                                                                   grid_name='',
    #                                                                   show_plot=True,
    #                                                                   minimum_sensor_reading=40,
    #                                                                   maximum_sensor_reading=500)
    # df_metrics_by_range = df_metrics_by_range.round(2)
    #
    # df_metrics_by_range.to_csv(f'test_results/metrics_performance_by_range_{current_algorithm}_H{current_horizon}.csv',
    #                        index=False)
    #
    # df_zone_values.to_csv(f'test_results/number_of points_by_zones_{current_algorithm}_H{current_horizon}.csv',
    #                   index=False)
    #
    # df_limits.to_csv(f'test_results/predictions_out_of_limits_{current_algorithm}_H{current_horizon}.csv',
    #              index=False)

    # df = pd.read_csv('/home/cirorodriguezleon/RELIEF-T1D/python_projects/Glucose_prediction_2026/Test_scripts/2026-01-14 - Both approach cross val/EXP-20260114-prediction-RMSE-Linear-H8-PH2-both_cross_val-v8/results/evaluation/metrics/metrics_performance_by_range_RMSE_Linear_H2_ALL_metric_then_aggregate.csv')
    # df_flat, df_report = fold_results_aggregation(df.drop(columns=['Fold']))
    # print(df_flat)
    # print(df_report)

    # Get values from parquet



    # clarke_error_grid(np.array([-1, -1]),
    #                   np.array([-1, -1]),
    #                   'Empty',
    #                   show_plot=True,
    #                   minimum_sensor_reading=40,
    #                   maximum_sensor_reading=500)