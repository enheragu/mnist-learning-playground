#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from utils.log_utils import log


def compute_sampling_error_curve(data, sample_sizes, method='analytical', n_iterations=3000, parameter = 'mean'):
    data = np.asarray(data, dtype=float)
    sample_sizes = np.asarray(sample_sizes, dtype=int)

    if np.any(sample_sizes <= 0):
        raise ValueError("sample_sizes must contain only positive integers")
    if data.size == 0:
        raise ValueError("data must contain at least one sample")

    method = method.lower()
    if method not in ['analytical', 'bootstrap', 'montecarlo']:
        raise ValueError(f"Unknown method '{method}'. Expected one of: analytical, bootstrap, montecarlo")
    if parameter not in ['mean', 'std']:
        raise ValueError(f"Unknown parameter '{parameter}'. Expected one of: mean, std")


    if method == 'analytical':
        sigma = np.std(data, ddof=1)
        std_errors = []
        for n_samples in sample_sizes:
            if parameter == 'mean':
                std_error = sigma / np.sqrt(n_samples)
            else:
                if n_samples < 2:
                    std_error = np.nan
                else:
                    std_error = sigma / np.sqrt(2 * (n_samples - 1))
            std_errors.append(std_error)
        return np.array(std_errors)

    estimates = []
    for n_samples in sample_sizes:
        if n_samples < 2 and parameter == 'std':
            estimates.append(np.nan)
            continue
        if method == 'bootstrap':
            sampled_values = np.random.choice(data, size=(n_iterations, n_samples), replace=True)
        elif method == 'montecarlo':
            g_mean = np.mean(data)
            g_std = np.std(data, ddof=1)
            sampled_values = np.random.normal(loc=g_mean, scale=g_std, size=(n_iterations, n_samples))
        
        if parameter == 'mean':
            sample_estimates = np.mean(sampled_values, axis=1)
        else: # parameter == 'std'
            sample_estimates = np.std(sampled_values, axis=1, ddof=1)
        
        estimates.append(np.std(sample_estimates, ddof=1))
    
    return np.array(estimates)  # Already has std of the estimates, which is already the sampling error


def compute_at_least_one_exceedance_probability_curve(data, sample_sizes, percentile=95, method='analytical', n_iterations=4000):
    data = np.asarray(data, dtype=float)
    sample_sizes = np.asarray(sample_sizes, dtype=int)

    if np.any(sample_sizes <= 0):
        raise ValueError("sample_sizes must contain only positive integers")
    if data.size == 0:
        raise ValueError("data must contain at least one sample")
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be in [0, 100]")

    method = method.lower()
    if method not in ['analytical', 'bootstrap', 'montecarlo']:
        raise ValueError(f"Unknown method '{method}'. Expected one of: analytical, bootstrap, montecarlo")

    if method == "analytical":
        p_single_success = 1 - (percentile / 100)
        probability = 1 - (1 - p_single_success) ** sample_sizes
        return probability

    threshold = np.percentile(data, percentile)

    probabilities = []

    if method == 'bootstrap':
        for n_samples in sample_sizes:
            sampled_values = np.random.choice(data, size=(n_iterations, n_samples), replace=True)
            trial_successes = np.any(sampled_values > threshold, axis=1)
            probabilities.append(np.mean(trial_successes))
        return np.asarray(probabilities)

    g_mean = np.mean(data)
    g_std = np.std(data)
    for n_samples in sample_sizes:
        sampled_values = np.random.normal(loc=g_mean, scale=g_std, size=(n_iterations, n_samples))
        trial_successes = np.any(sampled_values > threshold, axis=1)
        probabilities.append(np.mean(trial_successes))
    return np.asarray(probabilities)


def get_sampling_method_label(method):
    labels = {
        'analytical': 'Analytical Formula',
        'bootstrap': 'Bootstrap',
        'montecarlo': 'MonteCarlo',
    }
    return labels.get(method.lower(), method)
