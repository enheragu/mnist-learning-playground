#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import numpy as np
import scipy.stats as stats

from utils import output_path
from utils import getAllModelData
from utils.log_utils import log, logTable, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list
from utils.plot_distribution import plot_metric_distribution


analysis_path = './analysis_results/analysis'

montecarlo_samples = 500000 # Slow version :) -> 1000000
bootstrap_samples = 100000 # Slow version :) -> 100000

"""
    Given 2 sets of data it computes the probability of, when getting one random sample
    from each of them, the order is switched:
        So having data from g1 and g2 and mean(g1) > mean(g2); 
        computes the p(sample2) > p(sample1)
    The probability is based on a MonteCarlo simulation (with n_samples) based on an 
    approximation based on a Normal distribution of the provided data.
"""
def computeMonteCarloSwitchedProbability(dict_data = {'group1': [1,3], 'group2': [0,4]}, g_names = ["group1", "group2"], n_samples = montecarlo_samples):
    
    max_list = [np.max(dict_data[name]) for name in g_names]
    original_order = np.argsort(max_list) # Get index that sort the array min to max
    
    means_list = [np.mean(dict_data[name]) for name in g_names]
    std_list = [np.std(dict_data[name]) for name in g_names]

    samples_list = [np.random.normal(means_list[index], std_list[index], n_samples) for index in range(len(means_list))]    

    switched_count = 0
    for i in range(n_samples):
        observations_list = [samples_list[j][i] for j in range(len(g_names))]
        observations_order = np.argsort(observations_list)

        if not np.array_equal(observations_order, original_order):
            switched_count += 1
    
    switched_probability = switched_count / n_samples
    
    log(f"\t[MonteCarlo] {n_samples = }")
    log(f"\t[MonteCarlo] p(!original-order) = {switched_probability:.4f}")
    log(f"\t[MonteCarlo] p(original-order) = {1-switched_probability:.4f}")
    return switched_probability


"""
    Does the same as computeMonteCarloSwitchedProbability function, but in this case
    instead of MonteCarlo, the data is resampled and then values are extracted,
    as in a Bootstrap approach, based on the original data instead of the approximated 
    Normal distribution.
"""
def computeBootstrapSwitchedProbability(dict_data = {'group1': [1,3], 'group2': [0,4]}, g_names = ["group1", "group2"], n_samples = bootstrap_samples):
    
    max_list = [np.max(dict_data[name]) for name in g_names]
    original_order = np.argsort(max_list) # Get index that sort the array min to max

    samples_list = [np.random.choice(dict_data[name], size=n_samples, replace=True) for name in g_names]
    switched_count = 0
    for i in range(n_samples):
        observations_list = [samples_list[j][i] for j in range(len(g_names))]
        observations_order = np.argsort(observations_list)

        if not np.array_equal(observations_order, original_order):
            switched_count += 1

    switched_probability = switched_count / n_samples

    log(f"\t[Bootstrap] {n_samples = }")
    log(f"\t[Bootstrap] p(!original-order) = {switched_probability:.4f}")
    log(f"\t[Bootstrap] p(original-order) = {1-switched_probability:.4f}")
    return switched_probability

"""
    Wrap function to compute switched probability for a given data dict and tags with both MonteCarlo and
    Bootstrap approach
"""
def computeSwtichedProbability(dict_data, g_names):
    max_list = [np.max(dict_data[name]) for name in g_names]
    original_order = np.argsort(max_list) # Get index that sort the array min to max
    
    log(f"Analysis of switched probability for {g_names} models:")
    log(f"Original order: {' < '.join([g_names[index] for index in original_order])}")
    computeMonteCarloSwitchedProbability(dict_data, g_names)
    computeBootstrapSwitchedProbability(dict_data, g_names)



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

if __name__ == "__main__":
    os.makedirs(f"{analysis_path}", exist_ok=True)
    metrics_data = getAllModelData(output_path)

    accuracy_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
        log(f"\t· [{model}] samples: {len(accuracy_data[model])}")
    # log(f"Accuracy data filtered: {accuracy_data}")

    computeBetterResultSampleSize(accuracy_data, ['SimplePerceptron'])
    computeBetterResultSampleSize(accuracy_data, ['CNN_14L'], percentile=92)  

    computeSwtichedProbability(accuracy_data, ['HiddenLayerPerceptron','DNN_6L'])
    computeSwtichedProbability(accuracy_data, ['CNN_3L', 'CNN_4L', 'CNN_5L','CNN_14L'])
