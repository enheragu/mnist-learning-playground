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
from utils import getAllModelData
from utils.plot_distribution import plotDataDistribution, only_store, plot_metric_distribution, bin_size
from utils import output_path

analysis_path = './analysis_results/distributions'


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
        
    row_data = [['Model', f'mean{unit}', f'min{unit}', f'max{unit}', f'Data Amplitude{unit}', f'Amplitude 99.9% Interval Distribution{unit}']]
    for model_name, data in metric_data.items():
        
        amplitude_distribution = normal_amplitude(data)
        row_data.append([f"{model_name} (n={len(data)})", 
                         f"{np.mean(data):{format}}", 
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

"""
    Penalizes the proximity of elements in a subset to get a distribution with more spread data
"""
def proximity_penalty(subset, min_dist):
    subset = np.sort(subset)
    penalty = 0
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
            if abs(subset[j] - subset[i]) < min_dist:
                penalty += 1
    return penalty

def biased_resample(data, target_mean, target_std, subset_size, max_iter=30000, 
                    alpha_std = 3.0, alpha_proximity=4.0, min_dist_factor=20):
    data = np.array(data)
    n = len(data)
    distances = np.diff(np.sort(data))
    original_min_distance = np.min(distances[distances > 0])
    min_dist = original_min_distance * min_dist_factor
    # log(f"Original min distance: {original_min_distance:.4f}, using min_dist={min_dist:.4f} and alpha={alpha} for proximity penalty")

    # Random subset at init
    index = np.random.choice(n, subset_size, replace=False)
    subset = data[index]

    best_score = (
        abs(np.mean(subset) - target_mean) +
        alpha_std * abs(np.std(subset, ddof=1) - target_std) +
        alpha_proximity * proximity_penalty(subset, min_dist)
    )

    for _ in range(max_iter):
        current_mean = np.mean(subset)
        current_std = np.std(subset, ddof=1)
        out_candidates = np.setdiff1d(np.arange(n), index)
        if len(out_candidates) == 0:
            break

        # Choose based on the current mean and std
        if abs(current_mean - target_mean) > abs(current_std - target_std):
            # Changes min or max value depending on the average
            if current_mean < target_mean:
                in_pos = np.argmin(subset)
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmax(data[out_candidates])]
            else:
                in_pos = np.argmax(subset)
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmin(data[out_candidates])]
        else:
            # Changes closest or farthest value depending on the std
            if current_std < target_std:
                mean_val = np.mean(subset)
                in_pos = np.argmin(np.abs(subset - mean_val))
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmax(np.abs(data[out_candidates] - target_mean))]
            else:
                mean_val = np.mean(subset)
                in_pos = np.argmax(np.abs(subset - mean_val))
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmin(np.abs(data[out_candidates] - target_mean))]

        # Recompute score
        new_index = index.copy()
        new_index[np.where(index == in_idx)[0][0]] = out_idx
        new_subset = data[new_index]
        new_score = (
            abs(np.mean(new_subset) - target_mean) +
            alpha_std * abs(np.std(new_subset, ddof=1) - target_std) +
            alpha_proximity * proximity_penalty(new_subset, min_dist)
        )

        # If its better keep it :)
        if new_score < best_score:
            index = new_index
            subset = new_subset
            best_score = new_score

    return data[index]


"""
    Plots biaserd distributions (same normal with mean and std) with all data centered,
    or no data at right side or left side
"""
def plot_example_distributions(metrics_data, new_sample_size=50, analysis_path=analysis_path):
    
    data = np.array([entry['accuracy']*100 for entry in metrics_data['CNN_14L'].values()])
    mean = np.mean(data)
    std = np.std(data)
    log(f"[plot_example_distributions] Original data mean: {mean:.2f}, std: {std:.2f}. Max: {np.max(data):.2f}, Min: {np.min(data):.2f}, N: {len(data)}")

    # Resamples giving different probability to be chose in resampling based on value position
    data_right_bias = data[data < (mean + std*0.5)]
    resampled_right_bias = biased_resample(data_right_bias, target_mean=mean, target_std=std, subset_size=new_sample_size)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_right_bias):.2f}, std: {np.std(resampled_right_bias):.2f}. Max: {np.max(data_right_bias):.2f}, Min: {np.min(data_right_bias):.2f}, N: {len(data_right_bias)}")
    plot_metric_distribution({'CNN_14L': resampled_right_bias}, metric_label='Accuracy (%)', color_palette=[c_red], analysis_path=analysis_path, plot_filename="example_CNN_14L_right_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with right bias")

    data_centered_bias = data[(data < (mean + std*1.4)) & (data > (mean - std*1.4))]
    resampled_centered_bias = biased_resample(data_centered_bias, target_mean=mean, target_std=std, subset_size=new_sample_size, alpha_std=1.0)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_centered_bias):.2f}, std: {np.std(resampled_centered_bias):.2f}. Max: {np.max(data_centered_bias):.2f}, Min: {np.min(data_centered_bias):.2f}, N: {len(data_centered_bias)}")
    plot_metric_distribution({'CNN_14L': resampled_centered_bias}, metric_label='Accuracy (%)', color_palette=[c_yellow], analysis_path=analysis_path, plot_filename="example_CNN_14L_center_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with centered bias")
    
    data_left_bias = data[data > (mean - std*0.5)]
    resampled_left_bias = biased_resample(data_left_bias, target_mean=mean, target_std=std, subset_size=new_sample_size)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_left_bias):.2f}, std: {np.std(resampled_left_bias):.2f}. Max: {np.max(data_left_bias):.2f}, Min: {np.min(data_left_bias):.2f}, N: {len(data_left_bias)}")
    plot_metric_distribution({'CNN_14L': resampled_left_bias}, metric_label='Accuracy (%)', color_palette=[c_green], analysis_path=analysis_path, plot_filename="ezample_CNN_14L_left_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with left bias")
    

if __name__ == "__main__":
    os.makedirs(analysis_path, exist_ok=True)
    plt.rcParams.update({'font.size': 18})
    metrics_data = getAllModelData(output_path)

    all_models = metrics_data.keys()
    log(f"Model availability: {all_models}")
    # log(f"{metrics_data = }")
    
    plot_example_distributions(metrics_data, new_sample_size=60, analysis_path=analysis_path)
    exit()

    # Once all models' metrics have been gathered, plot the distributions
    if metrics_data:
        plotDataDistribution(metrics_data=metrics_data,
                             models_plot_list=[['SimplePerceptron'],
                               ['DNN_6L', 'HiddenLayerPerceptron'],
                               ['CNN_5L', 'CNN_3L', 'CNN_14L', 'CNN_4L'],
                               ['CNN_14L_B10', 'CNN_14L', 'CNN_14L_B25', 'CNN_14L_B50'],
                               all_models],
                             color_list=[[c_green],
                               [c_blue,c_darkgrey],
                               [c_yellow, c_red, c_purple, c_grey],
                               [c_yellow, c_red, c_purple, c_grey],
                               color_palette_list],
                             analysis_path = analysis_path)
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