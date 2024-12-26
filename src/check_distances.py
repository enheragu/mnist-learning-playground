#!/usr/bin/env python3
# encoding: utf-8

import os 
import numpy as np
from itertools import combinations

from utils.log_utils import log, logTable

analysis_path = './analysis_results'

# Error rates from Table 2 -> https://www.mdpi.com/2076-3417/9/15/3169
error_rates = [
    0.40, 0.83, 0.82, 0.84, 0.88, 0.94, 0.95, 0.24, 0.24, 0.25, 
    0.28, 0.29, 0.29, 0.31, 0.31, 0.32, 0.32, 0.32, 0.33, 0.37,
    0.37, 0.38, 0.39, 0.39, 0.42, 0.44, 0.45, 0.45, 0.46, 0.47,
    0.49, 0.52, 0.53, 0.57, 0.59, 0.60, 0.60, 0.62, 0.68, 0.69, 
    0.71, 0.83, 0.87
]




if __name__ == "__main__":

    os.makedirs(analysis_path, exist_ok=True)
    log_data = [['Parameter', 'Value']]
    log_data.append([f"Items", f"{len(error_rates)}"])
    log_data.append([f"Minr error rate", f"{np.min(error_rates)} %"])
    log_data.append([f"Max error rate", f"{np.max(error_rates)} %"])
    log_data.append([f"Error amplitude", f"{np.max(error_rates)-np.min(error_rates)} %"])


    mean_error = np.mean(error_rates)
    distances_to_mean = [abs(rate - mean_error) for rate in error_rates]
    average_distance_to_mean = np.mean(distances_to_mean)
    log_data.append([f"Mean error rate", f"{mean_error:.3f} %"])
    log_data.append([f"Average distance from mean", f"{average_distance_to_mean:.3f} %"])


    # Absolute distances between data pairs
    pairwise_distances = np.array([
        abs(rate1 - rate2) for rate1, rate2 in combinations(error_rates, 2)
    ])

    average_pairwise_distance = np.mean(pairwise_distances)
    log_data.append([f"Mean distance between pairs", f"{average_pairwise_distance:.3f} %"])
    log_data.append([f"Min distance between pairs", f"{np.min(pairwise_distances):.3f} %"])
    log_data.append([f"Min distance between pairs (excluding zeros)", f"{np.min(pairwise_distances[pairwise_distances>0]):.3f} %"])
    log_data.append([f"Max distance between pairs", f"{np.max(pairwise_distances):.3f} %"])

    logTable(log_data, analysis_path, 'Distances Analysis')