#!/usr/bin/env python3
# encoding: utf-8

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from utils.yaml_utils import getMetricsLogFile 

bin_size = 9

"""
    Plots theoretical normal distribution based on given data, and plots binned distribution or real data
"""
def plot_metric_distribution(metrics, metric_name, ax):
    
    data_y = [entry[metric_name] for entry in metrics.values()]
    sns.histplot(data_y, bins=bin_size, stat="density", alpha=0.4, label=f"{metric_name} (n = {len(data_y)})",
                 color="blue", edgecolor='none', ax=ax)

    mean = np.mean(data_y)
    std = np.std(data_y)
    if std > 0.000001:
        x = np.linspace(mean - std * 4, mean + std * 4, 100)
        y = norm.pdf(x, mean, std)
        sns.lineplot(x=x, y=y, linestyle='--', linewidth=2, color="red", ax=ax)
        for pos in np.arange(mean - std * 3, mean + std * 3, std):
            ax.vlines(x=pos, ymin=0, ymax=norm.pdf(pos, mean, std), colors="red", linewidth=1, linestyles='solid')

    ax.set_title(f"{metric_name} distribution")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ax.legend()


"""
    Creates subplots for each metric to plot
"""
def plot_all_metrics_distributions(metrics_data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    metrics = ['accuracy', 'f1_score', 'precision', 'recall']
    
    for i, metric in enumerate(metrics):
        ax = axs[i//2, i%2]
        plot_metric_distribution(metrics_data, metric, ax)
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    output_path = "./output_data/SimplePerceptron"
    metrics_data = getMetricsLogFile(os.path.join(output_path,"randomseed_training_metrics.yaml"))
    plot_all_metrics_distributions(metrics_data)
