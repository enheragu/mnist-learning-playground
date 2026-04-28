#!/usr/bin/env python3
# encoding: utf-8

import os
import math
import numpy as np
import scipy.stats as stats

from utils import output_path, ablation_data_file
from utils import getAllModelData, getAblationModelData, getAllAndAblationModelData
from utils.log_utils import log, logTable, bcolors, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list
from utils.plot_distribution import plot_metric_distribution
from utils.yaml_utils import dumpYaml
from utils.compute_switched_probability import (
    computeSwitchedProbability,
    montecarlo_samples as default_montecarlo_simulations,
    bootstrap_samples as default_bootstrap_simulations,
)
from utils.plot_sampling_graph import plot_all_sampling_errors, plot_all_percentile_probabilities, plot_all_percentile_probabilities_average

analysis_path = './analysis_results/analysis'

store_accuracy_standalonde_data = False
compute_estimation_error = False
compute_better_result_sample_size = False
compute_switched_probability = False
compute_sampling_error_graphs = True
compute_percentile_probability_graphs = True
simulation_x_max_quantile = 99.5

estimation_error_repetitions = 100


"""
    Given a set of data, computes the amount of repetitions a model should be trained to get
    a better result than the one a 5% top (of real data). Based on the approximated normal
    distribution to the given data
"""
def computeMonteCarloBetterResultSampleSize(data, percentile = 96, percentile_value = None, n_simulations = default_montecarlo_simulations, x_max_quantile=None):
    data = np.array(data)
    if percentile_value is not None:
        percentile = percentile = np.sum(data < percentile_value) / (np.sum(data < percentile_value) + np.sum(data > percentile_value)) * 100
        value = percentile_value
    else:
        value = np.percentile(data, percentile)

    g_mean = np.mean(data)
    g_std = np.std(data)

    n_iterations_monte_carlo = []
    for _ in range(n_simulations):
        n_iterations_monte_carlo.append(0)
        while True:
            observation = np.random.normal(loc=g_mean, scale=g_std, size=1)
            if observation > value:
                break

            n_iterations_monte_carlo[-1]+=1
    
    train_iterations = np.mean(n_iterations_monte_carlo)
    train_iterations_std = np.std(n_iterations_monte_carlo) 

    plot_metric_distribution({'': n_iterations_monte_carlo}, train_duration_data = None, metric_label = f'N to hit p{percentile:.0f} (Monte Carlo)', plot_func = None, 
                             color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = analysis_path,
                             plot_filename = f'monte_carlo_sample_simulation_{value:.2f}',
                             bin_size=40,
                             title=f'Iterations to exceed p{percentile:.0f} threshold (Monte Carlo)',
                             x_label='N Samples',
                             y_label='Percentage (%)',
                             hist_stat='percent',
                             hist_kwargs={'discrete': True},
                             x_max_quantile=x_max_quantile)
    
    log(f"\t[MonteCarlo] Samples computed: {train_iterations:.4f} (std {train_iterations_std:.4f})")



"""
    Given a set of data, computes the amount of repetitions a model should be trained to get
    a better result than the one a 5% top (of real data). Based on the approximated normal
    distribution to the given data
"""
def computeBootstrapBetterResultSampleSize(data, percentile = 96, percentile_value = None, n_simulations = default_bootstrap_simulations, x_max_quantile=None):
    data = np.array(data)
    if percentile_value is not None:
        percentile = percentile = np.sum(data < percentile_value) / (np.sum(data < percentile_value) + np.sum(data > percentile_value)) * 100
        value = percentile_value
    else:
        value = np.percentile(data, percentile)

    n_iterations_boosttrap = []
    for _ in range(n_simulations):
        n_iterations_boosttrap.append(0)
        while True:
            observation = np.random.choice(data, size=1, replace=True)
            if observation > value:
                break

            n_iterations_boosttrap[-1]+=1
                
    train_iterations = np.mean(n_iterations_boosttrap)
    train_iterations_std = np.std(n_iterations_boosttrap)

    plot_metric_distribution({'': n_iterations_boosttrap}, train_duration_data = None, metric_label = f'N to hit p{percentile:.0f} (Bootstrap)', plot_func = None, 
                             color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = analysis_path,
                             plot_filename = f'bootstrap_sample_simulation_{value:.2f}',
                             bin_size=40,
                             title=f'Iterations to exceed p{percentile:.0f} threshold (Bootstrap)',
                             x_label='N Samples',
                             y_label='Percentage (%)',
                             hist_stat='percent',
                             hist_kwargs={'discrete': True},
                             x_max_quantile=x_max_quantile)

    log(f"\t[Bootstrap] Samples computed: {train_iterations:.4f} (std {train_iterations_std:.4f})")


"""
    Wrap function to compute train iterations to get a result better than the percentile of provided data
    both using approximated normal distribution and bootstrap approach
"""
def computeBetterResultSampleSize(dict_data, g_names, percentile = 95, accuracy_value = None, x_max_quantile=None):
    for name in g_names:
        if accuracy_value is not None:
            percentile_value = accuracy_value
            data = np.array(dict_data[name])
            percentile = np.sum(data < percentile_value) / (np.sum(data < percentile_value) + np.sum(data > percentile_value)) * 100
        else:
            percentile_value = np.percentile(dict_data[name], percentile)
        log(f"Analysis of train size for {name}, to get value > {percentile_value:.4f} (percentile {percentile} of data), you would need:")

        computeMonteCarloBetterResultSampleSize(dict_data[name], percentile=percentile, percentile_value=accuracy_value, x_max_quantile=x_max_quantile)
        computeBootstrapBetterResultSampleSize(dict_data[name], percentile=percentile, percentile_value=accuracy_value, x_max_quantile=x_max_quantile)

def computeBetterResultSampleSizeMultiPercentiles(dict_data, g_names, percentiles=(90, 92, 95), n_simulations=50000, x_max_quantile=None):
    for name in g_names:
        data = np.array(dict_data[name])
        mc_metric_data = {}
        boot_metric_data = {}

        for percentile in percentiles:
            threshold = np.percentile(data, percentile)

            g_mean = np.mean(data)
            g_std = np.std(data)

            n_iterations_monte_carlo = []
            for _ in range(n_simulations):
                n_iterations_monte_carlo.append(0)
                while True:
                    observation = np.random.normal(loc=g_mean, scale=g_std, size=1)
                    if observation > threshold:
                        break
                    n_iterations_monte_carlo[-1] += 1

            n_iterations_bootstrap = []
            for _ in range(n_simulations):
                n_iterations_bootstrap.append(0)
                while True:
                    observation = np.random.choice(data, size=1, replace=True)
                    if observation > threshold:
                        break
                    n_iterations_bootstrap[-1] += 1

            mc_metric_data[f"p{percentile:.0f}"] = n_iterations_monte_carlo
            boot_metric_data[f"p{percentile:.0f}"] = n_iterations_bootstrap

        plot_metric_distribution(mc_metric_data,
                                 train_duration_data=None,
                                 metric_label='N Samples',
                                 plot_func=None,
                                 color_palette=color_palette_list,
                                 vertical_lines_acc=[],
                                 analysis_path=analysis_path,
                                 plot_filename=f"monte_carlo_sample_simulation_multi_p_{name}",
                                 bin_size=40,
                                 title=f"Iterations to exceed threshold (MC) - {name}",
                                 x_label='N Samples',
                                 y_label='Percentage (%)',
                                 hist_stat='percent',
                                 hist_kwargs={'discrete': True},
                                 x_max_quantile=x_max_quantile)

        plot_metric_distribution(boot_metric_data,
                                 train_duration_data=None,
                                 metric_label='N Samples',
                                 plot_func=None,
                                 color_palette=color_palette_list,
                                 vertical_lines_acc=[],
                                 analysis_path=analysis_path,
                                 plot_filename=f"bootstrap_sample_simulation_multi_p_{name}",
                                 bin_size=40,
                                 title=f"Iterations to exceed threshold (Boot.) - {name}",
                                 x_label='N Samples',
                                 y_label='Percentage (%)',
                                 hist_stat='percent',
                                 hist_kwargs={'discrete': True},
                                 x_max_quantile=x_max_quantile)



"""
    Templated version (for both bootstrap and MonteCarlo) to compute the estimation error of each
    method when estimating basic statistics (mean and std) of the provided data
"""
def computeEstimationErrorT(data, estimation_sample_sizes, sampling_fn, n_repetitions=200, table_title=''):
    g_mean = np.mean(data)
    g_std = np.std(data)
    
    table_data = [['Sample Size', 'Repetitions', 'Estimate Mean', 'Error Mean', 'Estimate Std', 'Error Std']]
    for sample_size in estimation_sample_sizes:
        mean_estimates = []
        std_estimates = []

        for _ in range(n_repetitions):
            estimates = sampling_fn(sample_size=sample_size)
            mean_estimates.append(np.mean(estimates))
            std_estimates.append(np.std(estimates))

        mean_estimates = np.asarray(mean_estimates)
        std_estimates = np.asarray(std_estimates)

        estimate_mean = np.mean(mean_estimates)
        estimate_std = np.mean(std_estimates)

        error_mean = (np.mean(np.abs(mean_estimates - g_mean)) if g_mean != 0 else 0)
        error_std = (np.mean(np.abs(std_estimates - g_std)) if g_std != 0 else 0)
        table_data.append([sample_size, n_repetitions, 
                           f"{estimate_mean:.8f}", f"{error_mean:.8f}", 
                           f"{estimate_std:.8f}", f"{error_std:.8f}"])

    if table_title:
        logTable(table_data, output_path=f"{analysis_path}/tables", filename=table_title)
    else:
        logTable(table_data)

    return table_data


"""
    Wrap function to compute the estimation error between the diferent methods
"""
def computeEstimationError(dict_data, g_names, bootstrap_simulations=default_bootstrap_simulations, montecarlo_simulations=default_montecarlo_simulations, n_repetitions=200):
    log(f"Analysis of estimation error between methods (repetitions per sample size: {n_repetitions}):")
    min_estimation_samples = 40*400
    tables_monte_carlo = []
    tables_bootstrap = []
    for name in g_names:
        data = dict_data[name]
        g_mean = np.mean(data)
        g_std = np.std(data)

        bootstrap_sample_sizes = np.array([1000, 4000, 8000, 16000, 25000, 50000, 75000, 100000, 150000]).astype(int)
        monte_carlo_sample_sizes = np.array([1000, 4000, 8000, 16000, 50000, 100000, 200000, 400000, 600000]).astype(int)

        log(f"MonteCarlo Estimation Error for {name} (mean: {g_mean:.4f}, std: {g_std:.4f}):", color=bcolors.OKCYAN)
        montecarlo_sampling = lambda sample_size: np.random.normal(loc=g_mean, scale=g_std, size=sample_size)
        tables_monte_carlo.append(computeEstimationErrorT(data=data,
                       estimation_sample_sizes=monte_carlo_sample_sizes,
                       sampling_fn=montecarlo_sampling,
                       n_repetitions=n_repetitions,
                       table_title=f"{name} MonteCarlo Estimation Error"))

        log(f"Bootstrap Estimation Error for {name} (mean: {g_mean:.4f}, std: {g_std:.4f}):", color=bcolors.OKCYAN)
        bootstrap_sampling = lambda sample_size: np.random.choice(data, size=sample_size, replace=True)
        tables_bootstrap.append(computeEstimationErrorT(data=data,
                       estimation_sample_sizes=bootstrap_sample_sizes,
                       sampling_fn=bootstrap_sampling,
                       n_repetitions=n_repetitions,
                       table_title=f"{name} Bootstrap Estimation Error"))

    # Table with [min-max] on each cell
    combined_table_mc = [['Sample Size', 'MonteCarlo Mean Error (min-max)', 'MonteCarlo Std Error (min-max)']]
    combined_table_boot = [['Sample Size', 'Bootstrap Mean Error (min-max)', 'Bootstrap Std Error (min-max)']]
    
    for table_in, table_out in [(tables_monte_carlo,combined_table_mc), 
                                (tables_bootstrap,combined_table_boot)]:
        for i in range(1, len(table_in[0])):
            sample_size = table_in[0][i][0]
            mean_errors = [float(t[i][3]) for t in table_in]
            std_errors = [float(t[i][5]) for t in table_in]
            
            table_out.append([
                sample_size,
                f"{min(mean_errors):.8f} - {max(mean_errors):.8f}",
                f"{min(std_errors):.8f} - {max(std_errors):.8f}"
            ])

    logTable(combined_table_mc, output_path=f"{analysis_path}/tables", filename=f"MonteCarlo Estimation Error (min-max) Accumulated")
    logTable(combined_table_boot, output_path=f"{analysis_path}/tables", filename=f"Bootstrap Estimation Error (min-max) Accumulated")

if __name__ == "__main__":
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)
    metrics_data, ablation_metrics = getAllAndAblationModelData(output_path, ablation_data_file)

    all_models = sorted(metrics_data.keys())
    
    accuracy_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
        log(f"\t· [{model}] samples: {len(accuracy_data[model])}")
    # log(f"Accuracy data filtered: {accuracy_data}")

    if store_accuracy_standalonde_data:
        ablation_accuracy_data = {}
        if ablation_metrics is not None:
            log("Ablation data available is:")
            for model, data in ablation_metrics.items():
                ablation_accuracy_data[model] = [entry['accuracy']*100 for entry in ablation_metrics[model].values()]
                log(f"\t· [{model}] samples: {len(ablation_accuracy_data[model])}")
        dumpYaml(accuracy_data, f"{analysis_path}/accuracy_data_raw.yaml")
        dumpYaml(ablation_accuracy_data, f"{analysis_path}/ablation_accuracy_data_raw.yaml")


    if not compute_estimation_error:
        log("Skipping estimation error computation...", bcolors.WARNING)
    if not compute_better_result_sample_size:
        log("Skipping better result sample size computation...", bcolors.WARNING)
    if not compute_switched_probability:
        log("Skipping switched probability computation...", bcolors.WARNING) 
    if not compute_sampling_error_graphs:
        log("Skipping sampling error graphs computation...", bcolors.WARNING)
    if not compute_percentile_probability_graphs:
        log("Skipping percentile probability graphs computation...", bcolors.WARNING)

    if compute_estimation_error:
        computeEstimationError(dict_data=accuracy_data, 
                               g_names=['CNN_3L', 'CNN_4L', 'CNN_5L','CNN_14L'], 
                               bootstrap_simulations=default_bootstrap_simulations*2, 
                               montecarlo_simulations=default_montecarlo_simulations*2,
                               n_repetitions=estimation_error_repetitions)

    if compute_better_result_sample_size:
        # computeBetterResultSampleSize(accuracy_data, ['SimplePerceptron'])
        # computeBetterResultSampleSize(accuracy_data, ['CNN_14L'], percentile=92)
        # computeBetterResultSampleSize(accuracy_data, ['CNN_14L'], percentile=90)
        computeBetterResultSampleSizeMultiPercentiles(accuracy_data, ['CNN_14L'], percentiles=(90, 95), x_max_quantile=simulation_x_max_quantile)

    if compute_switched_probability:
        computeSwitchedProbability(accuracy_data, ['HiddenLayerPerceptron','DNN_6L'], analysis_path=analysis_path)
        computeSwitchedProbability(accuracy_data, ['CNN_3L', 'CNN_4L', 'CNN_5L','CNN_14L'], analysis_path=analysis_path)


    combined_models = ablation_metrics.copy()
    combined_models.update(metrics_data)
    combined_models_list = all_models + list(ablation_metrics.keys())
    if compute_sampling_error_graphs:
        if ablation_metrics is not None:
            # plot_all_sampling_errors(metrics_data=ablation_metrics, analysis_path=analysis_path, title_tag='ablation')
            plot_all_sampling_errors(metrics_data=combined_models, analysis_path=analysis_path, plot_models=combined_models_list)
        # plot_all_sampling_errors(metrics_data=metrics_data, analysis_path=analysis_path, plot_models=all_models)
        # plot_all_sampling_errors(metrics_data=metrics_data, analysis_path=analysis_path, title_tag='informed_training', plot_models=['SimplePerceptron','CNN_3L', 'CNN_4L', 'CNN_5L', 'CNN_14L'], color_list=[c_green, c_yellow, c_grey, c_red, c_purple])
        # plot_all_sampling_errors(metrics_data=metrics_data, analysis_path=analysis_path, title_tag='CNN_14L_variations', plot_models=['CNN_14L_B10', 'CNN_14L', 'CNN_14L_B25', 'CNN_14L_B50'])

    if compute_percentile_probability_graphs:
        probability_percentile = 90
        probability_percentile_range = [80,100]
        if ablation_metrics is not None:
            # plot_all_percentile_probabilities(metrics_data=ablation_metrics, analysis_path=analysis_path, percentile=probability_percentile, title_tag='ablation')
            plot_all_percentile_probabilities(metrics_data=combined_models, analysis_path=analysis_path, percentile=probability_percentile, plot_models=combined_models_list)
            plot_all_percentile_probabilities_average(metrics_data=combined_models, analysis_path=analysis_path, percentile_range=probability_percentile_range, plot_models=combined_models_list)
        # plot_all_percentile_probabilities(metrics_data=metrics_data, analysis_path=analysis_path, percentile=probability_percentile, plot_models=all_models)
        # plot_all_percentile_probabilities(metrics_data=metrics_data, analysis_path=analysis_path, percentile=probability_percentile, title_tag='informed_training', plot_models=['SimplePerceptron','CNN_3L', 'CNN_4L', 'CNN_5L', 'CNN_14L'], color_list=[c_green, c_yellow, c_grey, c_red, c_purple])
        # plot_all_percentile_probabilities(metrics_data=metrics_data, analysis_path=analysis_path, percentile=probability_percentile, title_tag='CNN_14L_variations', plot_models=['CNN_14L_B10', 'CNN_14L', 'CNN_14L_B25', 'CNN_14L_B50'])