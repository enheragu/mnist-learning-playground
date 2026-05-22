#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from utils.log_utils import log, logTable

montecarlo_samples = 40000 # Slow version :) -> 1000000
bootstrap_samples = 40000 # Slow version :) -> 100000

"""
    Given 2 sets of data it computes the probability of, when getting one random sample
    from each of them, the order is switched:
        So having data from g1 and g2 and mean(g1) > mean(g2);
        computes p(sample2) > p(sample1)
    The probability is based on a MonteCarlo simulation (with n_simulations) based on an
    approximation based on a Normal distribution of the provided data.
"""
def computeSwitchedProbabilityT(dict_data=None, g_names=None, n_simulations = montecarlo_samples, resampling_method = "MonteCarlo", analysis_path = "results/analysis"):
    
    max_list = [np.max(dict_data[name]) for name in g_names]
    original_order = np.argsort(max_list) # Get index that sort the array min to max
    
    row_list = [["N Samples", "P(Switched order)", "P(Original order)"]]
    for n_samples in range(1,6):
        if resampling_method == "MonteCarlo":
            means_list = [np.mean(dict_data[name]) for name in g_names]
            std_list = [np.std(dict_data[name]) for name in g_names]

            # Draw `n_samples` observations per simulation for each group
            samples_list = [np.random.normal(means_list[index], std_list[index], size=(n_simulations, n_samples)) for index in range(len(means_list))]

        elif resampling_method == "Bootstrap":
            # Draw `n_samples` with replacement per simulation for each group
            samples_list = [np.random.choice(dict_data[name], size=(n_simulations, n_samples), replace=True) for name in g_names]

        else:
            raise ValueError(f"Unknown resampling_method='{resampling_method}'. Use 'MonteCarlo' or 'Bootstrap'.")

        # Per simulation, keep the best value from the n_samples drawn for each group.
        # Then compare ordering against the original ordering.
        best_per_group = np.column_stack([np.max(group_samples, axis=1) for group_samples in samples_list])
        simulated_orders = np.argsort(best_per_group, axis=1)
        switched_count = np.count_nonzero(np.any(simulated_orders != original_order, axis=1))
        
        switched_probability = switched_count / n_simulations
        row_list.append([n_samples, switched_probability, 1-switched_probability])
    
    # log(f"\t[{resampling_method}] n_simulations = {n_simulations}")
    # log(f"\t[{resampling_method}] p(!original-order) = {switched_probability:.4f}")
    # log(f"\t[{resampling_method}] p(original-order) = {1-switched_probability:.4f}")
    logTable(row_list, f"{analysis_path}/tables", f"Switched Probability - {resampling_method} - {' vs '.join(g_names)}")
    return row_list




"""
    Wrap function to compute switched probability for a given data dict and tags with both MonteCarlo and
    Bootstrap approach
"""
def computeSwitchedProbability(dict_data, g_names, analysis_path):
    max_list = [np.max(dict_data[name]) for name in g_names]
    original_order = np.argsort(max_list) # Get index that sort the array min to max
    
    log(f"Analysis of switched probability for {g_names} models:")
    log(f"Original order: {' < '.join([g_names[index] for index in original_order])}")
    table_mc = computeSwitchedProbabilityT(dict_data, g_names, resampling_method="MonteCarlo", n_simulations=montecarlo_samples,  analysis_path=analysis_path)
    table_boot = computeSwitchedProbabilityT(dict_data, g_names, resampling_method="Bootstrap", n_simulations=bootstrap_samples,  analysis_path=analysis_path)

    table_combined = [["N Samples", "P(Switched order) - MonteCarlo", "P(Switched order) - Bootstrap"]]
    for i in range(1, len(table_mc)):
        table_combined.append([table_mc[i][0], table_mc[i][1], table_boot[i][1]])
    logTable(table_combined, f"{analysis_path}/tables", f"Switched Probability - Combined - {' vs '.join(g_names)}")