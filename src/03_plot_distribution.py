#!/usr/bin/env python3
# encoding: utf-8

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.log_utils import log, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list
from utils import getAllModelData, getAblationModelData, getAllAndAblationModelData
from utils.plot_distribution import plotDataDistribution, only_store, plot_metric_distribution, plot_survival_function
from utils import output_path, ablation_data_file
from utils.distribution_analysis import normalityTest, maxAmplitude, count_trials, gamma_amplitude

analysis_path = './analysis_results/distributions'

compute_analysis_metrics = False
plot_distributions = False
plot_example_distributions = False
compute_survival_function = True


"""
    Just plots the amplitude of the distribution against the number of params
    to see if they somehow relate
"""
def plotParamAmplitudeRelation(metrics_data, plot_models = 'all', title_tag=''):
    import models
    # from train_models import input_size, num_classes, learning_rate, patience
    # General configuration taken from 00_train_models.py that cannot be imported as such
    input_size = 28 * 28  # Size of each image flattened
    num_classes = 10  # Numbers from 0 to 9
    learning_rate = 0.001
    patience = 10

    y = []
    x = []
    labels = []
    for model_name, data in metrics_data.items():
        if plot_models != 'all' and model_name not in plot_models:
            continue

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
    plt.ylabel('Accuracy amplitude')
    plt.xlabel('Trainable params (N)')

    # Mostrar la gráfica
    plt.grid(True)
    plt.savefig(os.path.join(analysis_path,f"amplitude_relation_{title_tag}.pdf"), format="pdf")

    if only_store:
        plt.close()


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
def plot_example_distributions(metrics_data, new_sample_size=50, analysis_path=analysis_path, vertical_lines_acc=[99.57],model='CNN_14L'):
    
    data = np.array([entry['accuracy']*100 for entry in metrics_data[model].values()])
    mean = np.mean(data)
    std = np.std(data)
    log(f"[plot_example_distributions] Original data mean: {mean:.2f}, std: {std:.2f}. Max: {np.max(data):.2f}, Min: {np.min(data):.2f}, N: {len(data)}")

    # Resamples giving different probability to be chose in resampling based on value position
    data_right_bias = data[data < (mean + std*0.5)]
    resampled_right_bias = biased_resample(data_right_bias, target_mean=mean, target_std=std, subset_size=new_sample_size)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_right_bias):.2f}, std: {np.std(resampled_right_bias):.2f}. Max: {np.max(data_right_bias):.2f}, Min: {np.min(data_right_bias):.2f}, N: {len(data_right_bias)}")
    plot_metric_distribution({model: resampled_right_bias}, metric_label='Accuracy (%)', color_palette=[c_red], vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path, plot_filename="example_CNN_14L_right_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with right bias")

    data_centered_bias = data[(data < (mean + std*1.4)) & (data > (mean - std*1.4))]
    resampled_centered_bias = biased_resample(data_centered_bias, target_mean=mean, target_std=std, subset_size=new_sample_size, alpha_std=1.0)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_centered_bias):.2f}, std: {np.std(resampled_centered_bias):.2f}. Max: {np.max(data_centered_bias):.2f}, Min: {np.min(data_centered_bias):.2f}, N: {len(data_centered_bias)}")
    plot_metric_distribution({model: resampled_centered_bias}, metric_label='Accuracy (%)', color_palette=[c_yellow], analysis_path=analysis_path, plot_filename="example_CNN_14L_center_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with centered bias")
    
    data_left_bias = data[data > (mean - std*0.5)]
    resampled_left_bias = biased_resample(data_left_bias, target_mean=mean, target_std=std, subset_size=new_sample_size)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_left_bias):.2f}, std: {np.std(resampled_left_bias):.2f}. Max: {np.max(data_left_bias):.2f}, Min: {np.min(data_left_bias):.2f}, N: {len(data_left_bias)}")
    plot_metric_distribution({model: resampled_left_bias}, metric_label='Accuracy (%)', color_palette=[c_green], analysis_path=analysis_path, plot_filename="example_CNN_14L_left_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with left bias")
    

if __name__ == "__main__":
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)
    os.makedirs(os.path.join(analysis_path, 'single_model'), exist_ok=True)
    plt.rcParams.update({'font.size': 18})
    metrics_data, ablation_metrics = getAllAndAblationModelData(output_path, ablation_data_file)

    all_models = metrics_data.keys()
    ablatipon_models = ablation_metrics.keys()
    log(f"Model availability: {all_models}")
    log(f"Model availability ablation tests: {ablatipon_models}")
    # log(f"{metrics_data = }")
    
    if ablation_metrics:
        if plot_distributions:
            plotDataDistribution(metrics_data=ablation_metrics,
                                models_plot_list=[ablatipon_models],
                                color_list=[color_palette_list],
                                analysis_path=analysis_path)
        if compute_analysis_metrics:
            normalityTest(metrics_data=ablation_metrics, title_tag='ablation', analysis_path=analysis_path)
            maxAmplitude(metrics_data=ablation_metrics, title_tag='ablation', analysis_path=analysis_path)
            maxAmplitude(metrics_data=ablation_metrics, metric='train_duration', unit=' (s)', unit_multiplier=1, format='.1f', amplitude_function=gamma_amplitude, title_tag='ablation', analysis_path=analysis_path)
            maxAmplitude(metrics_data=ablation_metrics, metric='best_epoch', unit='', unit_multiplier=1, format='.0f', amplitude_function=gamma_amplitude, title_tag='ablation', analysis_path=analysis_path)

            count_trials(metrics_data=ablation_metrics, title_tag='ablation', analysis_path=analysis_path)

    # Once all models' metrics have been gathered, plot the distributions
    if metrics_data:

        all_models_no_overfit = [ # ignore overfited models and repeated models for this analysis
            'BatchNormMaxoutNetInNet',
            'CNN_14L',
            # 'CNN_14L_B10',
            # 'CNN_14L_B25',
            # 'CNN_14L_B50',
            # 'CNN_14L_B80',
            # 'CNN_14L_overfit_0.02',
            'CNN_3L',
            'CNN_4L',
            'CNN_5L',
            'DNN_6L',
            # 'DNN_6L_overfit_0.02',
            'HiddenLayerPerceptron',
            'SimplePerceptron']
        if plot_distributions:
            plotParamAmplitudeRelation(metrics_data, plot_models=all_models_no_overfit)
        
            plotDataDistribution(metrics_data=metrics_data,
                                models_plot_list=[['SimplePerceptron'],
                                ['CNN_14L'],
                                ['DNN_6L', 'HiddenLayerPerceptron'],
                                ['CNN_3L', 'CNN_4L', 'CNN_5L', 'CNN_14L'],
                                ['CNN_14L', 'CNN_14L_B10', 'CNN_14L_B25', 'CNN_14L_B50'],
                                ['CNN_14L_overfit_0.02'], 
                                ['DNN_6L'], 
                                ['DNN_6L_overfit_0.02'],
                                #    all_models],
                                ],
                                color_list=[[c_green],
                                [c_purple],
                                [c_blue,c_darkgrey],
                                [c_yellow, c_grey, c_red, c_purple], 
                                [c_purple, c_yellow, c_red, c_grey],
                                [c_blue],
                                [c_grey],
                                [c_darkgrey],
                                #    color_palette_list],
                                ],
                                analysis_path = analysis_path)
        
        if compute_analysis_metrics:
            normalityTest(metrics_data=metrics_data, analysis_path=analysis_path)
            maxAmplitude(metrics_data=metrics_data, analysis_path=analysis_path)
            
            maxAmplitude(metrics_data=metrics_data, metric='train_duration', unit=' (s)', unit_multiplier=1, format='.1f', amplitude_function=gamma_amplitude, analysis_path=analysis_path)
            maxAmplitude(metrics_data=metrics_data, metric='best_epoch', unit='', unit_multiplier=1, format='.0f', amplitude_function=gamma_amplitude, analysis_path=analysis_path)

            count_trials(metrics_data=metrics_data, analysis_path=analysis_path)


        if plot_example_distributions:
            plot_example_distributions(metrics_data=metrics_data, new_sample_size=60, analysis_path=analysis_path)

        print(f"Search Juyang (John) Weng for info about initialization bias and similar stuff https://www.google.com/search?client=ubuntu&channel=fs&q=Juyang+%28John%29+Weng")

    

    combined_models = ablation_metrics.copy()
    combined_models.update(metrics_data)

    if compute_survival_function:
        exceedance_metric_data = {
            model_name: [entry['accuracy'] * 100 for entry in model_data.values()]
            for model_name, model_data in combined_models.items()
            if isinstance(model_data, dict) and len(model_data) > 0
        }
        if exceedance_metric_data:
            plot_survival_function(
                metrics_data=exceedance_metric_data,
                analysis_path=analysis_path,
                metric_label='Accuracy (%)',
                table_filename='Survival Function Key Percentiles Combined Models',
                plot_prefix='survival_combined',
            )
            log("Survival function plot generated", c_green)
        else:
            log("Skipping survival function: no valid data", c_yellow)
    else:
        log("Skipping survival function computation...", c_yellow)
    
    if not only_store:
        plt.show()