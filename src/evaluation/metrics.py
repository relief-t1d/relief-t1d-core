from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras
import pandas as pd
from pandas import DataFrame
import os
import matplotlib as mpl
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

mpl.rcParams['svg.fonttype'] = 'none'


def clarke_error_grid_as_heatmap(ref_values,
                                 pred_values,
                                 title_string,
                                 plot_dir='ceg_as_heatmap',
                                 vmin_manual=None,
                                 vmax_manual=None,
                                 show_plot=False,
                                 minimum_sensor_reading=40,
                                 maximum_sensor_reading=500,
                                 verbose=False):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Saturate and convert values to numpy arrays to reduce memory and ensure speed
    ref_values = np.clip(np.asarray(ref_values, dtype=np.float32), 0, 500)
    pred_values = np.clip(np.asarray(pred_values, dtype=np.float32), 0, 500)

    assert len(ref_values) == len(pred_values), f"Unequal number of values ({len(ref_values)} vs {len(pred_values)})."

    # ---------------------------------------------------------
    # BOUNDARY CHECKS AND WARNINGS (HIGH-SPEED NUMPY)
    # ---------------------------------------------------------

    # 1. Prediction calculations (always needed for the final CSV)
    min_pred_value = round(float(np.min(pred_values)), 2)
    max_pred_value = round(float(np.max(pred_values)), 2)
    number_pred_values_low_limit = int(np.sum(pred_values < minimum_sensor_reading))
    number_pred_values_up_limit = int(np.sum(pred_values > maximum_sensor_reading))

    if verbose:
        # --- Warnings for Reference values (y_true) ---
        if np.any(ref_values < minimum_sensor_reading):
            min_ref = float(np.min(ref_values))
            num_ref_low = int(np.sum(ref_values < minimum_sensor_reading))
            print(
                f'Input Warning: the minimum reference value ({min_ref:.2f}) is below the limit of sensor design (Inferior limit: {minimum_sensor_reading} mg/dL).')
            print(f"Number of reference values below the limit: {num_ref_low}\n")

        if np.any(ref_values > maximum_sensor_reading):
            max_ref = float(np.max(ref_values))
            num_ref_up = int(np.sum(ref_values > maximum_sensor_reading))
            print(
                f'Input Warning: the maximum reference value ({max_ref:.2f}) is above the limit of sensor design (Superior limit: {maximum_sensor_reading} mg/dL).')
            print(f"Number of reference values above the limit: {num_ref_up}\n")

        # --- Warnings for Predicted values (y_pred) ---
        if min_pred_value < minimum_sensor_reading:
            print(
                f'Input Warning: the minimum predicted value ({min_pred_value:.2f}) is below the minimum input value (Inferior limit: {minimum_sensor_reading} mg/dL).')
            print(f"Number of predicted values below the limit: {number_pred_values_low_limit}\n")

        if max_pred_value > maximum_sensor_reading:
            print(
                f'Input Warning: the maximum predicted value ({max_pred_value:.2f}) is above the maximum input value (Superior limit: {maximum_sensor_reading} mg/dL).')
            print(f"Number of predicted values above the limit: {number_pred_values_up_limit}\n")

    # 2. Final list of out-of-range values
    out_range_values = [
        min_pred_value,
        number_pred_values_low_limit,
        max_pred_value,
        number_pred_values_up_limit
    ]

    # ---- PLOT CONFIGURATION START ----
    if show_plot:
        fig, ax = plt.subplots()
        n_bins, my_range = 500, [[0, 500], [0, 500]]
        my_bins = [n_bins, n_bins]

        # --- LIMITS CALCULATION (AUTOMATIC OR MANUAL) ---
        # If manual limits (vmin_manual) are provided, we use them. Otherwise, we calculate them.
        if vmin_manual is not None and vmax_manual is not None:
            vmin_global, vmax_global = vmin_manual, vmax_manual
        else:
            # Standard local calculation if not sharing scale
            h_temp = np.histogram2d(ref_values, pred_values, bins=my_bins, range=my_range)
            # The minimum count is always 1 (for log), the maximum is the max of the histogram
            vmin_global, vmax_global = 1, max(float(np.max(h_temp[0])), 10.0)
        
        # Shared normalization across all histograms ensures that the color intensity is comparable between them.
        global_norm = LogNorm(vmin=vmin_global, vmax=vmax_global)

        # Colors & Ranges definition
        ranges = [
            (ref_values < 54, LinearSegmentedColormap.from_list("c_tbr2", ["#ffffff", "indianred", "#8B3A3A"])),
            ((ref_values >= 54) & (ref_values < 70),
             LinearSegmentedColormap.from_list("c_tbr1", ["#ffffff", "red", "#8B0000"])),
            ((ref_values >= 70) & (ref_values <= 180),
             LinearSegmentedColormap.from_list("c_tir", ["#ffffff", "limegreen", "#006400"])),
            ((ref_values > 180) & (ref_values <= 250),
             LinearSegmentedColormap.from_list("c_tar1", ["#ffffff", "yellow", "#808000"])),
            (ref_values > 250, LinearSegmentedColormap.from_list("c_tar2", ["#ffffff", "orange", "#8B4513"]))
        ]

        for mask, cmap in ranges:
            if np.any(mask):
                ax.hist2d(ref_values[mask], pred_values[mask], bins=my_bins, range=my_range, cmap=cmap, cmin=1,
                          norm=global_norm, zorder=2)

        # Legend Color Bar Setup
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        n_steps = 200
        Y_bounds = np.logspace(np.log10(vmin_global), np.log10(vmax_global), n_steps + 1)
        Z_slice = np.logspace(np.log10(vmin_global), np.log10(vmax_global), n_steps).reshape(-1, 1)
        cmaps_ordered = [r[1] for r in ranges]

        for i in range(5):
            X_mesh, Y_mesh = np.meshgrid(np.array([i, i + 1]), Y_bounds)
            cax.pcolormesh(X_mesh, Y_mesh, Z_slice, cmap=cmaps_ordered[i], norm=global_norm, shading='flat')

        cax.set_yscale('log')
        cax.set_xlim(0, 5)
        cax.yaxis.tick_right()
        cax.set_xticks([]) # No labels on the x-axis of the legend

        # Grid lines (A, B, C, D, E)
        lp = {'c': 'black', 'linewidth': 0.8, 'zorder': 10, 'alpha': 0.5}
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

        ax.set_xlabel("Reference Concentration (mg/dl)")
        ax.set_ylabel("Prediction Concentration (mg/dl)")
        ax.set_xticks(np.arange(0, 551, 50))
        ax.set_yticks(np.arange(0, 551, 50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlim([-1, 501]);
        ax.set_ylim([-1, 500]);
        ax.set_aspect('equal')

        base = f'{plot_dir}/CEG_{title_string}'
        plt.savefig(base + '.png', dpi=900, bbox_inches='tight')
        plt.savefig(base + '.pdf', dpi=900, bbox_inches='tight')
        plt.savefig(base + '.svgz')  # This will save the SVG in a compressed format, which is more efficient for large files.

        plt.close('all')
        if verbose:
            print(f"Generado: {base}")

    # ---------------------------------------------------------------------------------
    # VECTORIZED ENVIRONMENT (Replaces the slow FOR loop)
    # ---------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------
    # ZONE A (Clinical Accurate)
    # ---------------------------------------------------------------------------------
    # EXPLANATION:
    # (ref & pred < 70) - EXCLUSIVE: True to the clinical definition in original Clarke paper text.
    # (<= 1.2*ref and >= 0.8*ref) - INCLUSIVE: Captures the ±20% range in original Clarke paper text.
    cond_A = (
                ((ref_values < 70) & (pred_values < 70)) # Condition 1 for Zone A (LEFT BOTTOM)
              |
                ((pred_values <= 1.2 * ref_values) & (pred_values >= 0.8 * ref_values)) # Condition 2 for Zone A (RIGHT DIAGONAL)
    )
    # ---------------------------------------------------------------------------------
    # ZONE E (Erroneous Treatment)
    # ---------------------------------------------------------------------------------
    # EXPLANATION:
    # (< 70 and > 180) - EXCLUSIVE: True to the original Clarke paper text.
    # A real value of 70 or 180 is considered "in range," not "hypo" or "hyper."
    # Therefore, (180, 69) or (70, 181) are not Zone E.
    cond_E = (
                ((ref_values > 180) & (pred_values < 70))  # Condition 1 for Zone E (BOTTOM RIGHT)
              |
                ((ref_values < 70) & (pred_values > 180))  # Condition 2 for Zone E (TOP LEFT)
    )
    
    # ---------------------------------------------------------------------------------
    # ZONE C (Overcorrection)
    # ---------------------------------------------------------------------------------
    # EXPLANATION:
    # Community consensus to use inclusive limits for Zone C conditions
    cond_C = ((
                    ((ref_values >= 70) & (ref_values <= 290)) & (pred_values >= ref_values + 110))
              |
                    (((ref_values >= 130) & (ref_values <= 180)) & (pred_values <= 1.4 * ref_values - 182))
    )
    
    # ---------------------------------------------------------------------------------
    # ZONE D (Failure to Detect)
    # ---------------------------------------------------------------------------------
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
    cond_D = (((ref_values > 240) & (pred_values >= 70) & (pred_values <= 180))  # Condition 1 for Zone D (RIGHT)
              |
              ((ref_values <= 175.0 / 3.0) & (pred_values <= 180) & (pred_values >= 70)) # Condition 2 for Zone D (LEFT 1: before line that split top zone B and A)
              |
              (((ref_values >= 175.0 / 3.0) & (ref_values < 70)) & (pred_values >= 1.2 * ref_values))  # Condition 3 for Zone D (LEFT 2: after line that split top zone B and A)
    )

    # ---------------------------------------------------------------------------------
    # EXCLUSIVE RESOLUTION (Mimics the original if-elif-else hierarchy)
    # ---------------------------------------------------------------------------------
    is_A = cond_A
    is_E = ~is_A & cond_E
    is_C = ~is_A & ~is_E & cond_C
    is_D = ~is_A & ~is_E & ~is_C & cond_D
    
    # Zone B catches everything else not classified above
    is_B = ~is_A & ~is_E & ~is_C & ~is_D

    zone = [int(np.sum(is_A)), int(np.sum(is_B)), int(np.sum(is_C)), int(np.sum(is_D)), int(np.sum(is_E))]
    return zone, out_range_values


def test_by_range(df_result_vector: pd.DataFrame,
                  grid_name: str,
                  show_plot: bool = False,
                  plot_dir: str = 'plots',
                  minimum_sensor_reading=40,
                  maximum_sensor_reading=500,
                  verbose=False) -> tuple[DataFrame, DataFrame, DataFrame]:
                      
    # Blood glucose levels ranges segmentation
    ranges_dict = {
        'ENTIRE': df_result_vector,
        'TBR_2': df_result_vector[df_result_vector['y_test'] < 54],
        'TBR_1': df_result_vector[(df_result_vector['y_test'] >= 54) & (df_result_vector['y_test'] < 70)],
        'TIR': df_result_vector[(df_result_vector['y_test'] >= 70) & (df_result_vector['y_test'] <= 180)],
        'TAR_1': df_result_vector[(df_result_vector['y_test'] > 180) & (df_result_vector['y_test'] <= 250)],
        'TAR_2': df_result_vector[(df_result_vector['y_test'] > 250)]
    }

    df_metrics_summary = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E', 'A + B', 'RMSE', 'MSE', 'MAE', 'MARD'])
    df_zones_values = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E'])
    df_limits = pd.DataFrame(columns=['Range', 'Minimum value', 'Number values below limit', 'Maximum value', 'Number values above limit'])

    for name, df_sub in ranges_dict.items():
        if verbose: print(f'------- Zone: {name} -------')

        # SAFEGUARD: Prevent a division-by-zero error if the range is empty
        if df_sub.empty:
            if verbose: print(f"Warning: No hay datos disponibles para el rango {name}. Saltando...")
            # Rellenar con ceros o NaNs para mantener consistencia estructural en los CSVs
            df_zones_values.loc[len(df_zones_values)] = {'Range': name, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
            df_metrics_summary.loc[len(df_metrics_summary)] = {'Range': name, 'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0,
                                                               'A + B': 0, 'RMSE': np.nan, 'MSE': np.nan, 'MAE': np.nan,
                                                               'MARD': np.nan}
            df_limits.loc[len(df_limits)] = {'Range': name, 'Minimum value': np.nan, 'Number values below limit': 0,
                                             'Maximum value': np.nan, 'Number values above limit': 0}
            continue

        reference_values = df_sub['y_test'].values
        pred_values = df_sub['y_predict'].values

        plot_this = show_plot if name == 'ENTIRE' else False
        title_suffix = grid_name if name == 'ENTIRE' else grid_name + name

        zone, out_of_range_values = clarke_error_grid_as_heatmap(
            ref_values=reference_values, pred_values=pred_values,
            title_string=title_suffix, plot_dir=plot_dir, show_plot=plot_this,
            minimum_sensor_reading=minimum_sensor_reading, maximum_sensor_reading=maximum_sensor_reading,
            verbose=verbose
        )

        total_points = reference_values.shape[0]
        zone_percentages = round(pd.Series(zone) / total_points * 100, 2)

        df_zones_values.loc[len(df_zones_values)] = {'Range': name, 'A': zone[0], 'B': zone[1], 'C': zone[2],
                                                     'D': zone[3], 'E': zone[4]}
        
        print(f'Percentage of point by zone in {name} range: {zone_percentages.tolist()}')
        print()
        
        # Non-clinical metrics
        mse = mean_squared_error(reference_values, pred_values)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(reference_values, pred_values)
        mard = mean_absolute_percentage_error(reference_values, pred_values)

        df_metrics_summary.loc[len(df_metrics_summary)] = {
            'Range': name, 'A': zone_percentages[0], 'B': zone_percentages[1], 'C': zone_percentages[2],
            'D': zone_percentages[3], 'E': zone_percentages[4],
            'A + B': zone_percentages[0] + zone_percentages[1], 'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MARD': mard
        }

        df_limits.loc[len(df_limits)] = {
            'Range': name, 'Minimum value': out_of_range_values[0], 'Number values below limit': out_of_range_values[1],
            'Maximum value': out_of_range_values[2], 'Number values above limit': out_of_range_values[3]
        }
        
        print()
        print('-' * 100)
        print()

    return df_metrics_summary, df_zones_values, df_limits


# DRY MERGER: A single generic function for plotting any training metric
def plot_training_metric(hist: keras.callbacks.History, name: str, metric_key: str, y_label: str, title: str):
    
    fig, ax = plt.subplots()
    x_epoch = hist.epoch
    y_val = hist.history[f'val_{metric_key}']
    y_train = hist.history[metric_key]

    ax.plot(x_epoch, y_val, label=f'Validation {y_label}', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train, label=f'Train {y_label}', color='blue', linestyle='-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    fig.savefig(f'{name}_{y_label}.pdf', dpi=350, bbox_inches='tight')
    plt.close(fig)


# Supported wrapper methods for calls to external scripts
def get_train_plots_loss(hist, name):
    plot_training_metric(hist, name, 'loss', 'Loss', 'Training and validation loss')


def get_train_plots_RMSE(hist, name):
    plot_training_metric(hist, name, 'root_mean_squared_error', 'RMSE', 'Training and validation RMSE')


def get_train_plots_MSE(hist, name):
    plot_training_metric(hist, name, 'mean_squared_error', 'MSE', 'Training and validation MSE')


def fold_results_aggregation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    
    # ---------------------------------------------------------
    # APPROACH B: Report Structure (Text format "Mean ± Std")
    # Best for presentations/publications
    # ---------------------------------------------------------
    df_report = pd.DataFrame(index=logical_order)
    df_report.index.name = 'Range'
    
    # Calculate base stats again (or reuse from above)
    grouped = df.groupby('Range')[numeric_cols]
    means, stds = grouped.mean(), grouped.std()
    
    # Format columns
    for col in numeric_cols:
        df_report[col] = means[col].map('{:.2f}'.format) + " ± " + stds[col].map('{:.2f}'.format)

    return df_flat, df_report.reset_index()


if __name__ == "__main__":
    # # ----- Entry point for testing the functions V1. ------
    # # In case you want to test with your own data and get only the plot of the CLarke Error Grid, just change the path to the parquet file and make sure it has the correct structure (columns 'y_test' and 'y_predict').
    # parquet_path = ''
    # df = pd.read_parquet(parquet_path)
    # clarke_error_grid_as_heatmap(ref_values=df['y_test'],
    #                              pred_values=df['y_predict'],
    #                              title_string='cSigP01_Linear_H4',
    #                              plot_dir='test_results',
    #                              show_plot=True)
    #
    # # ----- Entry point for testing the functions V2. ------
    # # In case you want to test with your own data and want all metrics by range, just change the path to the parquet file and make sure it has the correct structure (columns 'y_test' and 'y_predict').
    # parquet_path = ''
    # df = pd.read_parquet(parquet_path)
    #
    # # Variable names
    # algorithm_name = 'cSigP01_Linear'
    # grid_name = f'{algorithm_name}_H4'
    # show_plot = False
    # plot_dir = 'test_results'
    # minimum_sensor_reading = 40
    # maximum_sensor_reading = 500
    #
    # df_metrics_by_range, df_zone_values, df_limits = test_by_range( df_result_vector=df,
    #                                                                 grid_name=grid_name,
    #                                                                 show_plot=show_plot,
    #                                                                 plot_dir=plot_dir,
    #                                                                 minimum_sensor_reading=minimum_sensor_reading,
    #                                                                 maximum_sensor_reading=maximum_sensor_reading)
    #
    # # Save results to CSV
    # df_metrics_by_range.to_csv(f'{plot_dir}/metrics_performance_by_range_{algorithm_name}_H4.csv', index=False)
    # df_zone_values.to_csv(f'{plot_dir}/number_of_points_by_zones_{algorithm_name}_H4.csv', index=False)
    # df_limits.to_csv(f'{plot_dir}/predictions_out_of_limits_{algorithm_name}_H4.csv', index=False)

    # ----- Entry point to create an empty clarke error grid ------
    # Create empty dataframe with the correct structure to generate an empty Clarke Error Grid.
    df_empty = pd.DataFrame({'y_test': [0-1], 'y_predict': [-1]})
    clarke_error_grid_as_heatmap(ref_values=df_empty['y_test'],
                                    pred_values=df_empty['y_predict'],
                                    title_string='Empty',
                                    plot_dir='test_results',
                                    show_plot=True,
                                    minimum_sensor_reading=40,
                                    maximum_sensor_reading=500)
    print("Empty Clarke Error Grid generated successfully.")