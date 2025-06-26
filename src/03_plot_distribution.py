#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, shapiro, kurtosis

from utils.yaml_utils import getMetricsLogFile 
from utils.log_utils import log, logTable, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list
from train_models import output_path

# Whether to plot or just store images in disk
only_store = True

analysis_path = './analysis_results/distributions'
bin_size = 20


def getAllModelData():
    metrics_file_name = "randomseed_training_metrics.yaml"
    metrics_data = {}

    # List each folder (assumed to be a model) in output_path
    for model_name in os.listdir(output_path):
        model_path = os.path.join(output_path, model_name)
        
        # Check if it's a directory and contains the metrics file
        if os.path.isdir(model_path):
            metrics_file = os.path.join(model_path, metrics_file_name)
            if os.path.exists(metrics_file):
                # Load metrics data for the model
                metrics_data[model_name] = getMetricsLogFile(metrics_file)
            else:
                log(f"No metrics file found for model: {model_name}")
        else:
            log(f"{model_name} is not a directory, skipping...")
        
    return metrics_data

def plot_metric_normaldistribution(data_y, ax, color):
    # Ajuste de la distribución normal para el modelo
    mean = np.mean(data_y)
    std = np.std(data_y)
    if std > 0.000001:
        x = np.linspace(mean - std * 4, mean + std * 4, 100)
        y = norm.pdf(x, mean, std)
        sns.lineplot(x=x, y=y, linestyle='--', linewidth=2, color=color, ax=ax)
        for pos in np.arange(mean - std * 3, mean + std * 3, std):
            linewidth = 1 if abs(pos-mean) >= 0.01 else 2
            linestyle = ':' if abs(pos-mean) >= 0.01 else 'solid'
            ax.vlines(x=pos, ymin=0, ymax=norm.pdf(pos, mean, std), colors=color, linewidth=linewidth, linestyles=linestyle)

def plot_metric_gammadistribution(data_y, ax, color):
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
def plot_metric_distribution(metrics_data, train_duration_data, metric_label = 'accuracy (%)', plot_func = plot_metric_normaldistribution, 
                             color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = analysis_path):
    
    fig, ax = plt.subplots(figsize=(12, 9))
    color_iterator = itertools.cycle(color_palette)

    max_density = 0
    x_range = [float('inf'), float('-inf')]

    for model_name, data_y in metrics_data.items():
        train_duration = np.mean(train_duration_data[model_name])
        train_duration_str = f"{int(train_duration // 60)}min {train_duration % 60:.2f}s"
        color = next(color_iterator)

        hist = sns.histplot(data_y, bins=bin_size, stat="density", alpha=0.4, label=f"{model_name}   (n={len(data_y)})", #   (mean train: {train_duration_str})",
                     color=color, edgecolor='none', ax=ax)
        
        patch_heights = [patch.get_height() for patch in hist.patches]
        max_density = max(max_density, max(patch_heights))
        patch_heights_x = [patch.get_height() for patch in hist.patches]
        
        x_range[0] = min(x_range[0], min(data_y))
        x_range[1] = max(x_range[1], max(data_y))
        plot_func(data_y, ax, color)

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

    path_name = os.path.join(analysis_path,f"plot_{metric_label.replace(' (%)','').replace(' (s)','').replace(' ','_').lower()}_{'_'.join(list(metrics_data.keys()))}.png")
    plt.savefig(path_name)
    # print(f"\t· Stored file in {path_name}")

    if only_store:
        plt.close()

"""
    Intermediate function to handle each distribution plot wanted
"""
def plotDataDistribution(metrics_data, models_plot_list = [['all']], color_list = color_palette_list, 
                         vertical_lines_acc = [], analysis_path=analysis_path):
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

"""
    Just plots the amplitude of the distribution against the number of params
    to see if they somehow relate
"""
def plotParamAmplitudeRelation(metrics_data):
    import models
    from train_models import input_size, num_classes, learning_rate, patience

    y = []
    x = []
    labels = []
    for model_name, data in metrics_data.items():

        object_class = getattr(models, model_name)
        model = object_class(input_size=input_size, num_classes=num_classes, learning_rate=learning_rate, patience=patience, output_path=None)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accuracy_data = [entry['accuracy']*100 for entry in data.values()]
        amplitude = np.max(accuracy_data) - np.min(accuracy_data)
        x.append(trainable_params)
        y.append(amplitude)
        labels.append(model_name)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)

    for i, label in enumerate(labels):
        plt.text(x[i], y[i], label, fontsize=9, ha='right')

    plt.title('Accuracy amplitude vs Trainable Params')
    plt.ylabel('Accuracy amplitude (%)')
    plt.xlabel('Trainable params (N)')

    # Mostrar la gráfica
    plt.grid(True)
    plt.savefig(os.path.join(analysis_path,f"amplitude_relation.png"))

    if only_store:
        plt.close()

"""
    Just plots the sampling error for each model against sample size
"""
def plotSamplingError(metrics_data, metric = 'accuracy'):
    sample_sizes = np.arange(1, 101)  # Tamaño de muestra de 1 a 100

    metric_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        metric_data[model] = [entry[metric]*100 for entry in metrics_data[model].values()]
        

    fig, ax = plt.subplots(figsize=(12, 9))
    color_iterator = itertools.cycle(color_palette_list)
    eq = r'Sampling Error = $\sigma/\sqrt{n}$'

    row_data = [['Model', '1 Sample (%)', '5 Samples (%)', '10 Samples (%)', '25 Samples (%)', '50 Samples (%)', '100 Samples (%)']]
    for model_name, data in metric_data.items():        
        g_std = np.std(data)
        errors = g_std / np.sqrt(sample_sizes)
        plt.plot(sample_sizes, errors, label=f'{model_name}', color=next(color_iterator), linewidth=2)
        
        row_data.append([
            model_name, 
            f"{errors[0]:.3f}",   # Error for 1 observation
            f"{errors[4]:.3f}",   # Error for 5 observations
            f"{errors[9]:.3f}",   # Error for 10 observations
            f"{errors[24]:.3f}",   # Error for 25 observations
            f"{errors[49]:.3f}",   # Error for 50 observations
            f"{errors[99]:.3f}"   # Error for 100 observations
        ])

    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) * 0.5
    y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) * 0.5
    ax.text(x_center, y_center, eq, color="Black", alpha=0.5, fontsize=17, ha="center", va="center")
        
    plt.title('Sampling error')
    plt.ylabel('SamplingError (%)')
    plt.xlabel('Sample Size (N)')

    # plt.xscale('log')
    plt.grid(visible=True, color=c_grey, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_path,f"sampling_error.png"))

    log(f"\nSummary Table of {metric.title()} Sampling Errors (%):")
    logTable(row_data, analysis_path, f"{metric.title()} Sampling Errors", colalign=['left', 'right', 'right', 'right', 'right', 'right', 'right'])

    if only_store:
        plt.close()

"""
    Checks the normality of each distribution with:
    - Kurtosis with Fisher definition (close to 0 is normal)
    - Skewness (median-mean correspondance)
    - The Shapiro-Wilk test: https://es.wikipedia.org/wiki/Prueba_de_Shapiro-Wilk
"""
def normalityTest(metrics_data, metric = 'accuracy'):
    metric_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        metric_data[model] = [entry[metric]*100 for entry in metrics_data[model].values()]
        
    row_data = [['Model', 'Median (%)', 'Mean (%)', 'Kurtosis (Fisher)', 'Shapiro-Wilk: W', 'Shapiro-Wilk: p-value (%)']]
    for model_name, data in metric_data.items():  
        estadistico, p_valor = shapiro(data) 
        kurt = kurtosis(data, fisher=True) 
        row_data.append([f"{model_name} (n={len(data)})", 
                         f"{np.median(data):.3f}", 
                         f"{np.mean(data):.3f}", 
                         f"{kurt:.4f}", 
                         f"{estadistico:.4f}", 
                         f"{p_valor:.4f}"])

    log(f"\nSummary Table of {metric.title()} Shapiro-Wilk normality test:")
    logTable(row_data, analysis_path, f"{metric.title()} Normality test", colalign=['left', 'right', 'right', 'right', 'right', 'right'])


def normal_amplitude(data):
    mean = np.mean(data)
    std = np.std(data)
    
    percentile_0_5 = norm.ppf(0.001, loc=mean, scale=std)
    percentile_99_5 = norm.ppf(0.999, loc=mean, scale=std)
    amplitude_99 = percentile_99_5 - percentile_0_5
    return amplitude_99


def gamma_amplitude(data):
    k, loc, scale = gamma.fit(data, floc=0)
    
    percentile_0_5 = gamma.ppf(0.001, k, loc, scale)
    percentile_99_5 = gamma.ppf(0.999, k, loc, scale)
    amplitude_99 = percentile_99_5 - percentile_0_5
    return amplitude_99

"""
    Computes max amplitude
"""
def maxAmplitude(metrics_data, metric='accuracy', unit=" (%)", format='.3f', amplitude_function = normal_amplitude):
    metric_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        metric_data[model] = [entry[metric]*100 for entry in metrics_data[model].values()]
        
    row_data = [['Model', f'min{unit}', f'max{unit}', f'Data Amplitude{unit}', f'Amplitude 99.9% Interval Distribution{unit}']]
    for model_name, data in metric_data.items():
        
        amplitude_distribution = normal_amplitude(data)
        row_data.append([f"{model_name} (n={len(data)})", 
                         f"{np.min(data):{format}}", 
                         f"{np.max(data):{format}}", 
                         f"{np.max(data)-np.min(data):{format}}", 
                         f"{amplitude_distribution:{format}}"])
    
    row_data_sorted = sorted(row_data[1:], key=lambda x: float(x[4].replace(f"{unit}", "")), reverse=True)
    row_data_sorted.insert(0, row_data[0])  # Agregar la fila de encabezado al inicio

    log(f"\nSummary {metric.title()} max amplitude:")
    logTable(row_data_sorted, analysis_path, f"{metric.title()} Max amplitude", colalign=['left', 'right', 'right', 'right'])


def count_trials(metrics_data):

    row_data = [['Model''N']]
    metric_data = {}
    for model, data in metrics_data.items():
        metric_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
        
    for model_name, data in metric_data.items():  
        row_data.append([model_name,len(data)])        

    log(f"\nTrials on each model:")
    logTable(row_data, analysis_path, f"N trials on each model")



if __name__ == "__main__":
    os.makedirs(analysis_path, exist_ok=True)
    plt.rcParams.update({'font.size': 18})
    metrics_data = getAllModelData()

    all_models = metrics_data.keys()
    log(f"Model availability: {all_models}")
    # log(f"{metrics_data = }")
    
    # Once all models' metrics have been gathered, plot the distributions
    if metrics_data:
        plotDataDistribution(metrics_data,
                              [['SimplePerceptron'],
                               ['DNN_6L', 'HiddenLayerPerceptron'],
                               ['CNN_5L', 'CNN_3L', 'CNN_14L', 'CNN_4L'],
                               ['CNN_14L_B10', 'CNN_14L', 'CNN_14L_B25', 'CNN_14L_B50'],
                               all_models],
                              [[c_green],
                               [c_blue,c_darkgrey],
                               [c_yellow, c_red, c_purple, c_grey],
                               [c_yellow, c_red, c_purple, c_grey],
                               color_palette_list])
        # plotParamAmplitudeRelation(metrics_data)
        plotSamplingError(metrics_data)
        normalityTest(metrics_data)
        maxAmplitude(metrics_data)
        
        maxAmplitude(metrics_data, 'train_duration', unit=' (s)', format='.1f', amplitude_function=gamma_amplitude)
        maxAmplitude(metrics_data, 'best_epoch', unit='', format='.0f', amplitude_function=gamma_amplitude)

        count_trials(metrics_data)
        print(f"Search Juyang (John) Weng for info about initialization bias and similar stuff https://www.google.com/search?client=ubuntu&channel=fs&q=Juyang+%28John%29+Weng")
    else:
        log("No models found or no metrics to plot.")
    
    if not only_store:
        plt.show()