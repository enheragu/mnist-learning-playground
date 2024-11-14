#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils.yaml_utils import getMetricsLogFile 
from main import output_path

bin_size = 9

## Custom color definitions
c_blue = "#0171ba"
c_green = "#78b01c"
c_yellow = "#f6ae2d"
c_red = "#f23535" 
c_purple = "#a66497"
c_grey = "#769393"
c_darkgrey = "#2a2b2e"

color_palette_list = [c_blue,c_green,c_yellow,c_red,c_purple,c_grey,c_darkgrey]

"""
    Plots theoretical normal distribution based on given data, and plots binned distribution or real data
"""
def plot_metric_distribution(metrics_data, metric_name = 'accuracy'):
    
    fig, ax = plt.subplots(figsize=(12, 9))
    color_iterator = itertools.cycle(color_palette_list)

    for model_name in metrics_data.keys():
        data_y = [entry[metric_name] for entry in metrics_data[model_name].values()]
        color = next(color_iterator)

        sns.histplot(data_y, bins=bin_size, stat="density", alpha=0.4, label=f"{model_name} (n = {len(data_y)})",
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

    ax.set_title(f"{metric_name} distribution across models")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
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
    
    # Once all models' metrics have been gathered, plot the distributions
    if metrics_data:
        plot_metric_distribution(metrics_data)
    else:
        print("No models found or no metrics to plot.")