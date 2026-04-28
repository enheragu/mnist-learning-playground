#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools
import copy

from collections import OrderedDict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, shapiro, kurtosis
from utils.log_utils import log, logTable, bcolors, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list, c_alpha_red, c_alpha_darkgrey, c_alpha_grey, c_alpha_blue


# Whether to plot or just store images in disk
only_store = True
plt.rcParams.update({'font.size': 24,
                     'legend.fontsize': 21})

def plot_metric_normaldistribution(data_y, ax, color, mean=None, std=None):
    # Ajuste de la distribución normal para el modelo
    mean = np.mean(data_y) if mean is None else mean
    std = np.std(data_y) if std is None else std
    if std > 0.000001:
        x_right = max(mean + std * 4, max(data_y))
        x_left = min(mean - std * 4, min(data_y))
        x = np.linspace(x_left, x_right, 100)
        y = norm.pdf(x, mean, std)
        sns.lineplot(x=x, y=y, linestyle='--', linewidth=2, color=color, ax=ax)
        for pos in np.arange(mean - std * 3, mean + std * 3, std):
            linewidth = 1 if abs(pos-mean) >= 0.01 else 2
            linestyle = ':' if abs(pos-mean) >= 0.01 else 'solid'
            ax.vlines(x=pos, ymin=0, ymax=norm.pdf(pos, mean, std), colors=color, linewidth=linewidth, linestyles=linestyle)

def plot_metric_gammadistribution(data_y, ax, color, mean=None, std=None):

    if np.any(np.array(data_y) <= 0):
        log(f"⚠️  Invalid data for Gamma distribution: {np.sum(np.array(data_y) <= 0)} values <= 0", color=bcolors.WARNING)
        return
    if np.std(np.array(data_y)) == 0:
        log("⚠️  Zero variance data for Gamma distribution. Cannot compute.", color=bcolors.WARNING)
        return
    
    # Ajuste de la distribución gamma para el modelo
    data_y = np.array(data_y)
    data_corrected = copy.deepcopy(data_y)
    data_corrected = data_corrected.astype(float)
    data_corrected[data_corrected == 0] = 1e-8 # Data needs to be at lesat floc value
    # print(f"{data_corrected.min() = }; {data_corrected.max() = }")
    shape, loc, scale = gamma.fit(data_corrected, floc=0)
    x_right = max(shape * scale * 3, max(data_y))
    x = np.linspace(0, x_right, 100)
    y = gamma.pdf(x, shape, loc, scale)

    sns.lineplot(x=x, y=y, linestyle='--', linewidth=2, color=color, ax=ax)

    for pos in [0.05,0.25,0.5,0.750,0.95]:
        pos_percentile = gamma.ppf(pos, shape, loc, scale)
        linewidth = 1 if pos != 0.5 else 2
        linestyle = ':' if pos != 0.5 else 'solid'
        ax.vlines(x=pos_percentile, ymin=0, ymax=gamma.pdf(pos_percentile, shape, loc, scale), colors=color, linewidth=linewidth, linestyles=linestyle)


"""
    Plots theoretical normal distribution based on given data, and plots binned distribution or real data
"""
def plot_metric_distribution(metrics_data, train_duration_data = None, metric_label = 'accuracy (%)', plot_func = plot_metric_normaldistribution, 
                             color_palette = color_palette_list, vertical_lines_acc = None, analysis_path = None,
                             plot_filename = None, plot_mean = None, plot_std = None,
                             bin_size = 20,
                             title = None,
                             x_label = None,
                             y_label = None,
                             hist_stat = "density",
                             x_max_quantile = None,
                             hist_kwargs = None):
    
    if analysis_path is None:
        log(f"[Error] [plot_metric_distribution] no analysis_path was provided for {metric_label}")
        return

    if vertical_lines_acc is None:
        vertical_lines_acc = []
    if hist_kwargs is None:
        hist_kwargs = {}
    
    _, ax = plt.subplots(figsize=(14, 11))
    max_density = 0
    x_range = [float('inf'), float('-inf')]
    all_x_values = []

    for index, (model_name, data_y) in enumerate(metrics_data.items()):
        # train_duration = np.mean(train_duration_data[model_name])
        # train_duration_str = f"{int(train_duration // 60)}min {train_duration % 60:.2f}s"
        color = color_palette[index]

        # Specific case for constant plot 
        if np.ptp(data_y) == 0:
            x0 = data_y[0]
            ax.axvline(x0, color=color, linewidth=4, alpha=0.7, 
                    label=f"{model_name} (n={len(data_y)}), X = {x0:.4f}") #; mean={np.mean(data_y):.2f}; std={np.std(data_y):.2f})") #   (mean train: {train_duration_str})")
            max_density = max(max_density, 100)
        else:
            hist = sns.histplot(data_y, bins=bin_size, stat=hist_stat, alpha=0.4, label=f"{model_name}   (n={len(data_y)})", #; mean={np.mean(data_y):.2f}; std={np.std(data_y):.2f})", #   (mean train: {train_duration_str})",
                        color=color, edgecolor='none', ax=ax, **hist_kwargs)
        
            patch_heights = [patch.get_height() for patch in hist.patches]
            max_density = max(max_density, max(patch_heights))
        
        x_range[0] = min(x_range[0], min(data_y))
        x_range[1] = max(x_range[1], max(data_y))
        all_x_values.extend(np.asarray(data_y, dtype=float).tolist())
        if plot_func is not None:
            plot_func(data_y, ax, color, plot_mean, plot_std)

    if metric_label == "Accuracy (%)":
        if vertical_lines_acc:
            lines_in_range = [line for line in vertical_lines_acc if x_range[0] <= line <= x_range[1]]
            lines_out_range = [x for x in vertical_lines_acc if not (x_range[0] <= x <= x_range[1])]
            count_in_range = len(lines_in_range)
            print(f"[{metric_label}] Models {'_'.join(list(metrics_data.keys()))} with range {x_range} include {count_in_range} models from review.")

            # Líneas fuera del rango (gris, punteadas, sin label)
            if lines_out_range:
                ax.vlines(x=lines_out_range, ymin=ax.get_ylim()[0], ymax=max_density,
                        colors=c_alpha_grey, linewidth=1, linestyles=':', label=None, zorder=1)

            # Líneas dentro del rango (rojas, sólidas, solo la primera con label)
            if lines_in_range:
                ax.vlines(x=lines_in_range, ymin=ax.get_ylim()[0], ymax=max_density,
                        colors=c_alpha_red, linewidth=2, linestyles='solid',
                        label=f'{count_in_range} models in range', zorder=1)
    
    if x_max_quantile is not None and len(all_x_values) > 0:
        x_left = min(all_x_values)
        x_right = np.percentile(np.asarray(all_x_values), x_max_quantile)
        if x_right > x_left:
            ax.set_xlim(x_left, x_right)

    ax.set_title(title if title is not None else f"{metric_label} distribution")
    ax.set_xlabel(x_label if x_label is not None else metric_label)
    ax.set_ylabel(y_label if y_label is not None else ("Density" if hist_stat == "density" else "Count"))
    ax.legend()

    plt.tight_layout()

    if plot_filename is not None:
        path_name = os.path.join(analysis_path, f"{plot_filename}.pdf")
    else:
        path_name = os.path.join(analysis_path,f"plot_{metric_label.replace(' (%)','').replace(' (s)','').replace(' ','_').lower()}_{'_'.join(list(metrics_data.keys()))}.pdf")
    plt.savefig(path_name, format="pdf")
    # print(f"\t· Stored file in {path_name}")

    if only_store:
        plt.close()

"""
    Intermediate function to handle each distribution plot wanted
"""
def plotDataDistribution(metrics_data, models_plot_list = [[]], color_list = color_palette_list, 
                         vertical_lines_acc = [], analysis_path=None, single_plots = True):
    
    if analysis_path is None:
        log(f"[Error] [plotDataDistribution] no analysis_path was provided for plots {models_plot_list = }")
        return
    
    color_iterator = itertools.cycle([color_palette_list])

    train_duration_data = None # default
    plot_train_duration = metrics_data is not None and 'train_duration' in list(list(metrics_data.values())[0].values())[0].keys()
    plot_best_epoch = metrics_data is not None and 'best_epoch' in list(list(metrics_data.values())[0].values())[0].keys()
    
    # if plot_train_duration:
    #     print(f"Plot train duration will be printed for {models_plot_list}")
    # else:
    #     print(f"Plot train duration will NOT be printed for {models_plot_list}")
    # if plot_best_epoch:
    #     print(f"Plot best epoch will be printed for {models_plot_list}")
    # else:
    #     print(f"Plot best epoch will NOT be printed for {models_plot_list}")

    # Plot all models in single plot
    if single_plots:
        for model in metrics_data.keys():
            color = next(color_iterator)
            accuracy_data = {model: [entry['accuracy']*100 for entry in metrics_data[model].values()]}
            
            if plot_best_epoch:
                best_epoch_data = {model: [entry['best_epoch'] for entry in metrics_data[model].values()]}
                plot_metric_distribution(best_epoch_data, train_duration_data, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=os.path.join(analysis_path,'single_model'))
            
            if plot_train_duration:
                train_duration_data = {model: [entry['train_duration'] for entry in metrics_data[model].values()]}
                plot_metric_distribution(train_duration_data, train_duration_data, metric_label = 'Train Duration (s)', plot_func=plot_metric_gammadistribution, color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=os.path.join(analysis_path,'single_model'))        
        
        plot_metric_distribution(accuracy_data, train_duration_data, metric_label = 'Accuracy (%)', color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=os.path.join(analysis_path,'single_model'))

    for plot, color_scheme in zip(models_plot_list, color_list):
        print(f"Generating plot for {plot} model{'s' if len(plot)>1 else ''}")
        accuracy_data = OrderedDict()
        if plot_train_duration: 
            train_duration_data = OrderedDict()
        if plot_best_epoch: 
            best_epoch_data = OrderedDict()

        for model in plot:
            if model in metrics_data.keys():
                accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
                if plot_train_duration: 
                    train_duration_data[model] = [entry['train_duration'] for entry in metrics_data[model].values()]
                if plot_best_epoch: 
                    best_epoch_data[model] = [entry['best_epoch'] for entry in metrics_data[model].values()]
            else:
                print(f"\t· [WARNING] {model} not in metrics available: {metrics_data.keys()}")

        if plot_train_duration: 
            plot_metric_distribution(train_duration_data, train_duration_data, metric_label = 'Train Duration (s)', plot_func=plot_metric_gammadistribution, color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        if plot_best_epoch: 
            plot_metric_distribution(best_epoch_data, train_duration_data, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        
        plot_metric_distribution(accuracy_data, train_duration_data, metric_label = 'Accuracy (%)', color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        
    # plt.show()


def plot_survival_function(metrics_data,
                           analysis_path,
                           metric_label='Accuracy (%)',
                           percentile_start=89,
                           percentile_step=0.1,
                           table_filename=None,
                           plot_prefix='survival'):
    """Empirical percentile exceedance calibration — one line per model.

    metric-agnostic: pass any pre-extracted list of floats per model, e.g.
        {'CNN_14L': [99.1, 99.4, ...]}           # accuracy
        {'yolo_task': [0.72, 0.81, ...]}         # mAP50
        {'yolo_task': [0.55, 0.61, ...]}         # mAP50-95

    X axis: percentile P.
    Y axis: empirical P(X >= x_p) [%].
    Dashed line: theoretical 100-P % (continuous distribution reference).
    """
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)

    percentiles = np.arange(percentile_start, 100 + percentile_step, percentile_step, dtype=float)
    percentiles = percentiles[percentiles <= 100]

    key_ps = [50, 75, 90, 95, 97.5, 99]
    table_data = [['Model', 'N'] + [f'p{p:g}: empirical % > threshold (theoretical {100-p:g}%)' for p in key_ps]]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Theoretical reference: P(X > x_p) = (100-P)% by definition of normal percentiles
    ax.plot(percentiles, 100 - percentiles, color='black', linestyle='--',
            linewidth=1.5, label='Theoretical normal: 100 − P %', zorder=5)

    for idx, (model_name, data_values) in enumerate(metrics_data.items()):
        data = np.asarray(data_values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        n = len(data)

        # Thresholds from fitted normal — NOT from the data itself.
        # This makes the comparison meaningful: empirical tail vs. theoretical tail.
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-9:
            continue
        theoretical_thresholds = norm.ppf(percentiles / 100.0, loc=mean, scale=std)

        empirical_rates = [np.mean(data > t) * 100.0 for t in theoretical_thresholds]

        color = color_palette_list[idx % len(color_palette_list)]
        ax.plot(percentiles, empirical_rates,
                color=color, linewidth=1.6, label=f"{model_name} (N={n})", alpha=0.85)

        key_vals = [f"{np.mean(data > norm.ppf(p/100.0, loc=mean, scale=std)) * 100:.1f}%" for p in key_ps]
        table_data.append([model_name, n] + key_vals)

    ax.set_xlabel('Percentile P (of fitted normal)')
    ax.set_ylabel('Empirical P(X > x_p)  [%]')
    ax.set_xlim(percentile_start, 100)
    ax.set_xticks(np.arange(int(np.ceil(percentile_start)), 101, 1))
    ax.autoscale(axis='y')
    ax.set_ylim(bottom=0)
    ax.set_title(f"{metric_label}: P(X ≥ x_p)")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_plot = os.path.join(analysis_path, f"{plot_prefix}.pdf")
    plt.savefig(output_plot, format='pdf')
    if only_store:
        plt.close()

    if len(table_data) > 1:
        logTable(
            row_data=table_data,
            output_path=f"{analysis_path}/tables",
            filename=table_filename or f"Survival Function Calibration {metric_label}",
            colalign=['left', 'right'] + ['right'] * len(key_ps)
        )

