#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import numpy as np
import scipy.stats as stats

from utils import output_path
from utils import getAllModelData
from utils.log_utils import log, logTable, bcolors, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list
from utils.plot_distribution import plot_metric_distribution
from utils.compute_switched_probability import computeSwtichedProbability, montecarlo_samples, bootstrap_samples

analysis_path = './analysis_results/analysis'



"""
    Given a set of data, computes the amount of repetitions a model should be trained to get
    a better result than the one a 5% top (of real data). Based on the approximated normal
    distribution to the given data
"""
def computeMonteCarloBetterResultSampleSize(data, percentile = 96, percentile_value = None, n_samples = montecarlo_samples):
    data = np.array(data)
    if percentile_value is not None:
        percentile = percentile = np.sum(data < percentile_value) / (np.sum(data < percentile_value) + np.sum(data > percentile_value)) * 100
        value = percentile_value
    else:
        value = np.percentile(data, percentile)

    g_mean = np.mean(data)
    g_std = np.std(data)

    n_iterations_monte_carlo = []
    for _ in range(n_samples):
        n_iterations_monte_carlo.append(0)
        while True:
            observation = np.random.normal(loc=g_mean, scale=g_std, size=1)
            if observation > value:
                break

            n_iterations_monte_carlo[-1]+=1
    
    train_iterations = np.mean(n_iterations_monte_carlo)
    train_iterations_std = np.std(n_iterations_monte_carlo) 

    plot_metric_distribution({'': n_iterations_monte_carlo}, train_duration_data = None, metric_label = f'number of samples to reach {percentile} percentile based on a Monte Carlo simulation', plot_func = None, 
                             color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = analysis_path,
                             plot_filename = f'monte_carlo_sample_simulation_{value:.2f}',
                             bin_size=40)
    
    log(f"\t[MonteCarlo] Samples computed: {train_iterations:.4f} (std {train_iterations_std:.4f})")



"""
    Given a set of data, computes the amount of repetitions a model should be trained to get
    a better result than the one a 5% top (of real data). Based on the approximated normal
    distribution to the given data
"""
def computeBootstrapBetterResultSampleSize(data, percentile = 96, percentile_value = None, n_samples = bootstrap_samples):
    data = np.array(data)
    if percentile_value is not None:
        percentile = percentile = np.sum(data < percentile_value) / (np.sum(data < percentile_value) + np.sum(data > percentile_value)) * 100
        value = percentile_value
    else:
        value = np.percentile(data, percentile)

    n_iterations_boosttrap = []
    for _ in range(n_samples):
        n_iterations_boosttrap.append(0)
        while True:
            observation = np.random.choice(data, size=1, replace=True)
            if observation > value:
                break

            n_iterations_boosttrap[-1]+=1
                
    train_iterations = np.mean(n_iterations_boosttrap)
    train_iterations_std = np.std(n_iterations_boosttrap)

    plot_metric_distribution({'': n_iterations_boosttrap}, train_duration_data = None, metric_label = f'number of samples to reach {percentile} percentile based on a Bootstrap simulation', plot_func = None, 
                             color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = analysis_path,
                             plot_filename = f'bootstrap_sample_simulation_{value:.2f}',
                             bin_size=40)

    log(f"\t[Bootstrap] Samples computed: {train_iterations:.4f} (std {train_iterations_std:.4f})")


"""
    Wrap function to compute train iterations to get a result better than the percentile of provided data
    both using approximated normal distribution and bootstrap approach
"""
def computeBetterResultSampleSize(dict_data, g_names, percentile = 96, accuracy_value = None):
    for name in g_names:
        if accuracy_value is not None:
            percentile_value = accuracy_value
            data = np.array(dict_data[name])
            percentile = np.sum(data < percentile_value) / (np.sum(data < percentile_value) + np.sum(data > percentile_value)) * 100
        else:
            percentile_value = np.percentile(dict_data[name], percentile)
        log(f"Analysis of train size for {name}, to get value > {percentile_value:.4f} (percentile {percentile} of data), you would need:")

        computeMonteCarloBetterResultSampleSize(dict_data[name], percentile=percentile, percentile_value=accuracy_value)
        computeBootstrapBetterResultSampleSize(dict_data[name], percentile=percentile, percentile_value=accuracy_value)



"""
    Templated version (for both bootstrap and MonteCarlo) to compute the estimation error of each
    method when estimating basic statistics (mean and std) of the provided data
"""
def computeEstimationErrorT(data, samples_list, sampling_method, iterations=1):
    g_mean = np.mean(data)
    g_std = np.std(data)

    table_data = [['Samples', 'Estimate Mean', 'Error Mean  (%)', 'Estimate Std', 'Error Std (%)']]
    for n_samples in samples_list:
        error_mean_list = []
        error_std_list = []
        for iter in range(iterations):
            estimates = sampling_method(n_samples=n_samples)
            estimate_mean = np.mean(estimates)
            estimate_std = np.std(estimates)

            error_mean_list.append((abs(estimate_mean - g_mean)/abs(g_mean))*100)
            error_std_list.append((abs(estimate_std - g_std)/abs(g_std))*100)
        
        error_mean = np.mean(error_mean_list)
        error_std = np.mean(error_std_list)
        table_data.append([n_samples, f"{estimate_mean:.8f}", f"{error_mean:.8f}", f"{estimate_std:.8f}", f"{error_std:.8f}"])
    
    logTable(table_data)



"""
    Wrap function to compute the estimation error between the diferent methods
"""
def computeEstimationError(dict_data, g_names):
    error_estimation_iterations = 400
    log(f"Analysis of estimation error between methods:")
    for name in g_names:
        data = dict_data[name]
        g_mean = np.mean(data)
        g_std = np.std(data)
        # montecarlo_samples_list = np.linspace(40*400, montecarlo_samples, 5).astype(int)
        bootstrap_samples_list = np.linspace(40*400, bootstrap_samples, 5).astype(int)

        log(f"MonteCarlo Estimation Error for {name} ({error_estimation_iterations} iterations):", color=bcolors.OKCYAN)
        montecarlo_sampling = lambda n_samples: np.random.normal(loc=g_mean, scale=g_std, size=n_samples)
        computeEstimationErrorT(data=data, samples_list=bootstrap_samples_list, sampling_method=montecarlo_sampling, iterations=error_estimation_iterations)

        log(f"Bootstrap Estimation Error for {name} ({error_estimation_iterations} iterations):", color=bcolors.OKCYAN)
        bootstrap_sampling = lambda n_samples: np.random.choice(data, size=n_samples, replace=True)
        computeEstimationErrorT(data=data, samples_list=bootstrap_samples_list, sampling_method=bootstrap_sampling, iterations=error_estimation_iterations)



if __name__ == "__main__":
    os.makedirs(f"{analysis_path}", exist_ok=True)
    metrics_data = getAllModelData(output_path)

    accuracy_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
        log(f"\t· [{model}] samples: {len(accuracy_data[model])}")
    # log(f"Accuracy data filtered: {accuracy_data}")

    computeEstimationError(accuracy_data, ['CNN_3L', 'CNN_4L', 'CNN_5L','CNN_14L'])

    computeBetterResultSampleSize(accuracy_data, ['SimplePerceptron'])
    computeBetterResultSampleSize(accuracy_data, ['CNN_14L'], percentile=92)  

    computeSwtichedProbability(accuracy_data, ['HiddenLayerPerceptron','DNN_6L'])
    computeSwtichedProbability(accuracy_data, ['CNN_3L', 'CNN_4L', 'CNN_5L','CNN_14L'])
