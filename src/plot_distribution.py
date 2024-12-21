#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools
import tabulate

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, kurtosis

from utils.yaml_utils import getMetricsLogFile 
from main import output_path

bin_size = 20

## Custom color definitions
c_blue = "#0171ba"
c_green = "#78b01c"
c_yellow = "#f6ae2d"
c_red = "#f23535" 
c_purple = "#a66497"
c_grey = "#769393"
c_darkgrey = "#2a2b2e"

color_palette_list = [c_blue,c_green,c_yellow,c_red,c_purple,c_grey,c_darkgrey]


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
                print(f"No metrics file found for model: {model_name}")
        else:
            print(f"{model_name} is not a directory, skipping...")
        
    return metrics_data

"""
    Plots theoretical normal distribution based on given data, and plots binned distribution or real data
"""
def plot_metric_distribution(metrics_data, train_duration_data, metric_label = 'accuracy (%)'):
    
    fig, ax = plt.subplots(figsize=(12, 9))
    color_iterator = itertools.cycle(color_palette_list)

    for model_name, data_y in metrics_data.items():
        train_duration = np.mean(train_duration_data[model_name])
        train_duration_str = f"{int(train_duration // 60)}min {train_duration % 60:.2f}s"
        color = next(color_iterator)

        sns.histplot(data_y, bins=bin_size, stat="density", alpha=0.5, label=f"{model_name}   (n = {len(data_y)})   (mean train: {train_duration_str})",
                     color=color, edgecolor='none', ax=ax)

        # Ajuste de la distribución normal para el modelo
        mean = np.mean(data_y)
        std = np.std(data_y)
        if std > 0.000001:
            x = np.linspace(mean - std * 4, mean + std * 4, 100)
            y = norm.pdf(x, mean, std)
            sns.lineplot(x=x, y=y, linestyle='--', linewidth=2, color=color, ax=ax)
            for pos in np.arange(mean - std * 3, mean + std * 3, std):
                ax.vlines(x=pos, ymin=0, ymax=norm.pdf(pos, mean, std), colors=color, linewidth=1, linestyles='solid')

    ax.set_title(f"{metric_label} distribution")
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()

"""
    Intermediate function to handle each distribution plot wanted
"""
def plotDataDistribution(metrics_data, models_plot_list = [['all']], color_list = color_palette_list):
    global color_palette_list
    
    for plot, color_scheme in zip(models_plot_list, color_list):
        color_palette_list = color_scheme

        train_duration_data = {}
        accuracy_data = {}
        for model, data in metrics_data.items():
            if plot == 'all' or model in plot:
                accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
                train_duration_data[model] = [entry['train_duration'] for entry in metrics_data[model].values()]

        # plot_metric_distribution(train_duration_data, train_duration_data, metric_label = 'train_duration (s)')
        plot_metric_distribution(accuracy_data, train_duration_data, metric_label = 'accuracy (%)')
    # plt.show()

"""
    Just plots the amplitude of the distribution against the number of params
    to see if they somehow relate
"""
def plotParamAmplitudeRelation(metrics_data):
    import models
    from main import input_size, num_classes, learning_rate, patience

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


"""
    Just plots the sampling error for each model against sample size
"""
def plotSamplingError(metrics_data):
    sample_sizes = np.arange(1, 101)  # Tamaño de muestra de 1 a 100

    accuracy_data = {}
    print("Data available is:")
    for model, data in metrics_data.items():
        accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
        

    fig, ax = plt.subplots(figsize=(12, 9))
    color_iterator = itertools.cycle(color_palette_list)
    eq = r'Sampling Error = $\sigma/\sqrt{n}$'

    row_data = [['Model', '1 Sample', '5 Samples', '10 Samples', '25 Samples', '50 Samples', '100 Samples']]
    for model_name, data in accuracy_data.items():        
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

    print("\nSummary Table of Sampling Errors (%):")
    print(tabulate.tabulate(row_data, headers="firstrow", tablefmt="fancy_grid"))

"""
    Checks the normality of each distribution with:
    - Kurtosis with Fisher definition (close to 0 is normal)
    - Skewness (median-mean correspondance)
    - The Shapiro-Wilk test: https://es.wikipedia.org/wiki/Prueba_de_Shapiro-Wilk
"""
def normalityTest(metrics_data):
    accuracy_data = {}
    print("Data available is:")
    for model, data in metrics_data.items():
        accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
        
    row_data = [['Model', 'Median', 'Mean', 'Kurtosis (Fisher)', 'Shapiro-Wilk: W', 'Shapiro-Wilk: p-value']]
    for model_name, data in accuracy_data.items():  
        estadistico, p_valor = shapiro(data) 
        kurt = kurtosis(data, fisher=True) 
        row_data.append([model_name, np.median(data), np.mean(data), kurt, estadistico, p_valor])

    print("\nSummary Table of Shapiro-Wilk normality test:")
    print(tabulate.tabulate(row_data, headers="firstrow", tablefmt="fancy_grid"))

if __name__ == "__main__":
    metrics_data = getAllModelData()

    all_models = metrics_data.keys()
    print(f"Model availability: {all_models}")
    
    # Once all models' metrics have been gathered, plot the distributions
    if metrics_data:
        plotDataDistribution(metrics_data,[['SimplePerceptron'],
                               ['DNN_6L', 'HiddenLayerPerceptron'],
                               ['CNN_5L', 'CNN_2L', 'CNN_13L', 'CNN_4L'],
                               all_models],
                              [[c_green],
                               [c_blue,c_darkgrey],
                               [c_yellow, c_red, c_purple, c_grey],
                               color_palette_list])
        # plotParamAmplitudeRelation(metrics_data)
        plotSamplingError(metrics_data)
        normalityTest(metrics_data)
    else:
        print("No models found or no metrics to plot.")
    plt.show()