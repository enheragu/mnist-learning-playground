#!/usr/bin/env python3
# encoding: utf-8

import os 
import numpy as np
from itertools import combinations

from utils.log_utils import log, logTable, color_palette_list
from plot_distribution import plotDataDistribution
from utils import output_path
from utils import getAllModelData

analysis_path = './analysis_results/distances'

# Error rates from Table 2 -> https://www.mdpi.com/2076-3417/9/15/3169
error_rates = {
'no_data_augmentation': {
    "HOPE+DNN with unsupervised learning features [47]": 0.40,
    "Deep convex net [10]": 0.83,
    "CDBN [54]": 0.82,
    "S-SC + linear SVM [56]": 0.84,
    "2-layer MP-DBM [58]": 0.88,
    "DNet-kNN [55]": 0.94,
    "2-layer Boltzmann machine [57]": 0.95,
    "Batch-normalized maxout network-in-network [29]": 0.24,
    "Committees of evolved CNNs (CEA-CNN) [65]": 0.24,
    "Genetically evolved committee of CNNs [66]": 0.25,
    "Committees of 7 neuroevolved CNNs [64]": 0.28,
    "CNN with gated pooling function [30]": 0.29,
    "Inception-Recurrent CNN + LSUV + EVE [60]": 0.29,
    "Recurrent CNN [31]": 0.31,
    "CNN with norm. layers and piecewise linear activation units [32]": 0.31,
    "CNN (5 conv, 3 dense) with full training [45]": 0.32,
    "MetaQNN (ensemble) [61]": 0.32,
    "Fractional max-pooling CNN with random overlapping [34] ": 0.32,
    "CNN with competitive multi-scale conv. filters [33]": 0.33,
    "CNN neuroevolved with GE [63]": 0.37,
    "Fast-learning shallow CNN [35]": 0.37,
    "CNN FitNet with LSUV initialization and SVM [59]": 0.38,
    "Deeply supervised CNN [27]": 0.39,
    "Convolutional kernel networks [36]": 0.39,
    "CNN with Multi-loss regularization [37]": 0.42,
    "MetaQNN [61]": 0.44,
    "CNN (3 conv maxout, 1 dense) with dropout [17]": 0.45,
    "Convolutional highway networks [38]": 0.45,
    "CNN (5 conv, 3 dense) with retraining [45]": 0.46,
    "Network-in-network [39]": 0.47,
    "CNN (3 conv, 1 dense), stochastic pooling [25]": 0.49,
    "CNN (2 conv, 1 dense, relu) with dropout [24]": 0.52,
    "CNN, unsup pretraining [17]": 0.53,
    "CNN (2 conv, 1 dense, relu) with DropConnect [24]": 0.57,
    "SparseNet + SVM [15]": 0.59,
    "CNN (2 conv, 1 dense), unsup pretraining [16]": 0.60,
    "DEvol [62]": 0.60,
    "CNN (2 conv, 2 dense) [40]": 0.62,
    "Boosted Gabor CNN [42]": 0.68,
    "CNN (2 conv, 1 dense) with L-BFGS [43]": 0.69,
    "Fastfood 1024/2048 CNN [44]": 0.71,
    "Feature Extractor + SVM [14]": 0.83,
    "Dual-hidden Layer Feedforward Network [21]": 0.87,
    "CNN LeNet-5 [4]": 0.95},
'data_augmentated': {
    "NN 6-layer 5,700 hidden units [12]": 0.35,
    "MSRV C-SVDDNet [46]": 0.35,
    "Committee of 25 NN 2-layer 800 hidden units [11]": 0.39,
    "RNN [48]": 0.45,
    "K-NN (P2DHMDM) [6]": 0.52,
    "COSFIRE [49]": 0.52,
    "K-NN (IDM) [6]": 0.54,
    "Task-driven dictionary learning [51]": 0.54,
    "Virtual SVM, deg-9 poly, 2-pixel jit [8]": 0.56,
    "RF-C-ELM, 15,000 hidden units [21]": 0.57,
    "PCANet (LDANet-2) [50]": 0.62,
    "K-NN (shape context) [5]": 0.63,
    "Pooling + SVM [52]": 0.64,
    "Virtual SVM, deg-9 poly, 1-pixel jit [8]": 0.68,
    "NN 2-layer 800 hidden units, XE loss [9]": 0.70,
    "SOAE-with sparse connectivity and activity [53]": 0.75,
    "SVM, deg-9 poly [4]": 0.80,
    "Product of stumps on Haar f. [7]": 0.87,
    "NN 2-layer 800 hidden units, MSE loss [9]": 0.90,
    "CNN (2 conv, 1 dense, relu) with DropConnect [24]": 0.21,
    "Committee of 25 CNNs [20]": 0.23,
    "CNN with APAC [28]": 0.23,
    "CNN (2 conv, 1 relu, relu) with dropout [24]": 0.27,
    "Committee of 7 CNNs [19]": 0.27,
    "Deep CNN [18]": 0.35,
    "CNN (2 conv, 1 dense), unsup pretraining [16]": 0.39,
    "CNN, XE loss [9]": 0.40,
    "Scattering convolution networks + SVM [41]": 0.43,
    "Feature Extractor + SVM [14]": 0.54,
    "CNN Boosted LeNet-4 [4]": 0.70,
    "CNN LeNet-5 [4]": 0.80
}}

def compute_metrics(error_rates):
    metrics = {}
    metrics["Items"] = len(error_rates)
    metrics["Min error rate"] = np.min(error_rates)
    metrics["Max error rate"] = np.max(error_rates)
    metrics["Error amplitude"] = np.max(error_rates) - np.min(error_rates)

    mean_error = np.mean(error_rates)
    distances_to_mean = [abs(rate - mean_error) for rate in error_rates]
    metrics["Mean error rate"] = mean_error
    metrics["Average distance from mean"] = np.mean(distances_to_mean)

    pairwise_distances = np.array([
        abs(rate1 - rate2) for rate1, rate2 in combinations(error_rates, 2)
    ])
    metrics["Mean distance between pairs"] = np.mean(pairwise_distances)
    metrics["Min distance between pairs"] = np.min(pairwise_distances)
    metrics["Min distance between pairs (excluding zeros)"] = (
        np.min(pairwise_distances[pairwise_distances > 0]) if len(pairwise_distances[pairwise_distances > 0]) > 0 else 0
    )
    metrics["Max distance between pairs"] = np.max(pairwise_distances)


    unique_elements = set(error_rates)
    repeated_elements = [x for x in unique_elements if error_rates.count(x) > 1]
    total_repeated = sum(error_rates.count(x) - 1 for x in repeated_elements)
    metrics["Items with error repeated"] = total_repeated

    return metrics


if __name__ == "__main__":

    os.makedirs(analysis_path, exist_ok=True)
    
    error_rates_da = list(error_rates['data_augmentated'].values())
    error_rates_nda = list(error_rates['no_data_augmentation'].values())
    metrics_da = compute_metrics(error_rates_da)
    metrics_nda = compute_metrics(error_rates_nda)
    log_data = [['Parameter', 'With Data Augmentation', 'Without DA']]
    
    for key in metrics_da.keys():
        log_data.append([
            key,
            f"{metrics_da[key]:.3f}" if isinstance(metrics_da[key], float) else f"{metrics_da[key]}",
            f"{metrics_nda[key]:.3f}" if isinstance(metrics_nda[key], float) else f"{metrics_nda[key]}"
        ])
    
    logTable(log_data, analysis_path, f'Distances Analysis')


    metrics_data = getAllModelData(output_path)
    all_models = metrics_data.keys()

    accuracy_agugmentation = [100-item for item in error_rates["data_augmentated"].values()]
    accuracy_no_augmentation = [100-item for item in error_rates["no_data_augmentation"].values()]
    vertical_lines_acc=accuracy_agugmentation+accuracy_no_augmentation
    # print(f"Vertical lines at coord: {vertical_lines_acc}")
    plotDataDistribution(metrics_data,
                        [all_models],
                        [color_palette_list],
                        vertical_lines_acc=vertical_lines_acc,
                        analysis_path=analysis_path)