#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from utils.log_utils import log

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
