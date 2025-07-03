
#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, shapiro, kurtosis
from utils.log_utils import log, logTable, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list

bin_size = 20

# Whether to plot or just store images in disk
only_store = True


def plot_metric_normaldistribution(data_y, ax, color, mean=None, std=None):
    # Ajuste de la distribución normal para el modelo
    mean = np.mean(data_y) if mean is None else mean
    std = np.std(data_y) if std is None else std
    if std > 0.000001:
        x = np.linspace(mean - std * 4, mean + std * 4, 100)
        y = norm.pdf(x, mean, std)
        sns.lineplot(x=x, y=y, linestyle='--', linewidth=2, color=color, ax=ax)
        for pos in np.arange(mean - std * 3, mean + std * 3, std):
            linewidth = 1 if abs(pos-mean) >= 0.01 else 2
            linestyle = ':' if abs(pos-mean) >= 0.01 else 'solid'
            ax.vlines(x=pos, ymin=0, ymax=norm.pdf(pos, mean, std), colors=color, linewidth=linewidth, linestyles=linestyle)

def plot_metric_gammadistribution(data_y, ax, color, mean=None, std=None):
    # Ajuste de la distribución gamma para el modelo
    shape, loc, scale = gamma.fit(data_y, floc=0)
    x = np.linspace(0, shape * scale * 3, 100)
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
                             color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = None,
                             plot_filename = None, plot_mean = None, plot_std = None):
    
    if analysis_path is None:
        log(f"[Error] [plot_metric_distribution] no analysis_path was provided for {metric_label}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 9))
    color_iterator = itertools.cycle(color_palette)

    max_density = 0
    x_range = [float('inf'), float('-inf')]

    for model_name, data_y in metrics_data.items():
        # train_duration = np.mean(train_duration_data[model_name])
        # train_duration_str = f"{int(train_duration // 60)}min {train_duration % 60:.2f}s"
        color = next(color_iterator)

        hist = sns.histplot(data_y, bins=bin_size, stat="density", alpha=0.4, label=f"{model_name}   (n={len(data_y)})", #; mean={np.mean(data_y):.2f}; std={np.std(data_y):.2f})", #   (mean train: {train_duration_str})",
                     color=color, edgecolor='none', ax=ax)
        
        patch_heights = [patch.get_height() for patch in hist.patches]
        max_density = max(max_density, max(patch_heights))
        patch_heights_x = [patch.get_height() for patch in hist.patches]
        
        x_range[0] = min(x_range[0], min(data_y))
        x_range[1] = max(x_range[1], max(data_y))
        plot_func(data_y, ax, color, plot_mean, plot_std)

    if metric_label == "Accuracy (%)":
        for vertical_coord in vertical_lines_acc:
            ymin, ymax = ax.get_ylim()
            ax.vlines(x=vertical_coord, ymin=ymin, ymax=max_density, colors=c_grey, linewidth=1, linestyles=':')

        if vertical_lines_acc:
            lines_in_range = [line for line in vertical_lines_acc if x_range[0] <= line <= x_range[1]]
            count_in_range = len(lines_in_range)
            print(f"[{metric_label}] Models {'_'.join(list(metrics_data.keys()))} with range {x_range} include {count_in_range} models from review.")


    ax.set_title(f"{metric_label} distribution")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()

    if plot_filename is not None:
        path_name = os.path.join(analysis_path, f"{plot_filename}.png")
    else:
        path_name = os.path.join(analysis_path,f"plot_{metric_label.replace(' (%)','').replace(' (s)','').replace(' ','_').lower()}_{'_'.join(list(metrics_data.keys()))}.png")
    plt.savefig(path_name)
    # print(f"\t· Stored file in {path_name}")

    if only_store:
        plt.close()

"""
    Intermediate function to handle each distribution plot wanted
"""
def plotDataDistribution(metrics_data, models_plot_list = [['all']], color_list = color_palette_list, 
                         vertical_lines_acc = [], analysis_path=None):
    
    if analysis_path is None:
        log(f"[Error] [plotDataDistribution] no analysis_path was provided for plots {models_plot_list = }")
        return
    
    color_iterator = itertools.cycle(color_list)

    # Plot all models in single plot
    for index, (model, data) in enumerate(metrics_data.items()):
        color = next(color_iterator)
        accuracy_data = {model: [entry['accuracy']*100 for entry in metrics_data[model].values()]}
        train_duration_data = {model: [entry['train_duration'] for entry in metrics_data[model].values()]}
        best_epoch_data = {model: [entry['best_epoch'] for entry in metrics_data[model].values()]}

        plot_metric_distribution(best_epoch_data, train_duration_data, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        plot_metric_distribution(train_duration_data, train_duration_data, metric_label = 'Train Duration (s)', plot_func=plot_metric_gammadistribution, color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        plot_metric_distribution(accuracy_data, train_duration_data, metric_label = 'Accuracy (%)', color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)

    for plot, color_scheme in zip(models_plot_list, color_list):
        print(f"Generating plot for {plot} model{'s' if len(plot)>1 else ''}")
        train_duration_data = {}
        best_epoch_data = {}
        accuracy_data = {}
        for model, data in metrics_data.items():
            if plot == 'all' or model in plot:
                accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
                train_duration_data[model] = [entry['train_duration'] for entry in metrics_data[model].values()]
                best_epoch_data[model] = [entry['best_epoch'] for entry in metrics_data[model].values()]
        
        for model_plot in plot:
            if model_plot not in metrics_data.keys():
                print(f"\t· [WARNING] {model_plot} not in metrics available: {metrics_data.keys()}")
        plot_metric_distribution(best_epoch_data, train_duration_data, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        plot_metric_distribution(train_duration_data, train_duration_data, metric_label = 'Train Duration (s)', plot_func=plot_metric_gammadistribution, color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        plot_metric_distribution(accuracy_data, train_duration_data, metric_label = 'Accuracy (%)', color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path)
        
    # plt.show()