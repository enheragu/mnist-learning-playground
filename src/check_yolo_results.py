#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools
import csv
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import seaborn as sns
from scipy.stats import norm, gamma, shapiro, kurtosis
from torch import seed

from utils.plot_distribution import plot_metric_distribution, plot_metric_gammadistribution, plot_metric_normaldistribution, plot_survival_function
from utils.yaml_utils import getMetricsLogFile, updateMetricsLogFile, print_yaml_structure, print_dict_keys, dumpYaml
from utils.log_utils import log, logTable, bcolors, color_palette_list, print_dict, print_dict_keys


from utils.indexer import Indexer
from utils import output_path, deposit_path, read_deposit, analysis_seed
from utils.deposit_data import getDepositYoloResults, getDepositYoloFinalMetrics
from utils.plot_sampling_graph import plot_all_sampling_errors, plot_all_percentile_probabilities, plot_all_percentile_probabilities_average
from utils.yolo_overfit_utils import (
    runYoloOverfitAnalysis,
    collectYoloFinalMetrics,
)
from utils.compute_switched_probability import (
    computeSwitchedProbability,
    montecarlo_samples as default_montecarlo_simulations,
    bootstrap_samples as default_bootstrap_simulations,
)
from utils.yolo_run_validity import updateYoloRunsInventory, getValidYoloRuns

analysis_path = './analysis_results/yolo_analysis'
output_path = os.path.join(output_path, "YOLO")
parse_mode = "incremental"

# Only valid runs (see utils/yolo_run_validity.py); the inventory with the validity of every run is stored with the data
only_valid_runs = True
yolo_runs_inventory_file = os.path.join(output_path, 'yolo_runs_inventory.csv')

# Metric names as written in the article, for titles and axis labels (file names keep the plain names)
yolo_metric_labels = {'mAP50': r'mAP$_{50}$', 'mAP50-95': r'mAP$_{50\text{-}95}$'}

store_metric_standalone_data = False
enable_yolos_swithced_probability = False
enable_yolo_overfit_analysis = False
enable_yolo_survival_function = True
enable_yolo_plot_distributions = False
enable_yolo_sampling_plots = True
enable_yolo_ablation_tests = False

# overfit-style analysis from YOLO results.csv files.
yolo_overfit_analysis_path = os.path.join(analysis_path, 'overfit')
yolo_overfit_plot_all_runs = False
yolo_overfit_alpha = 0.12
yolo_overfit_loss_log_scale = 'auto'  # auto | True | False
yolo_overfit_loss_component = 'all'  # total | box | cls | dfl | all
yolo_overfit_quality_metric_key = 'mAP50'  # mAP50 | mAP50-95 | precision | recall


# Configurable parse modes:
# - "cached": Only use existing cached/summary data, do not parse raw data. If cache/summary is missing, it will be skipped.
# - "incremental": Parse raw data only if it's newer than existing cache/summary, otherwise reuse cache/summary. If cache/summary is missing, parse raw data.
# - "scratch": Always parse raw data and regenerate cache/summary, ignoring existing cache/summary timestamps.  
VALID_PARSE_MODES = {"cached", "incremental", "scratch"}


def sanitize_parse_mode(parse_mode):
    if parse_mode not in VALID_PARSE_MODES:
        log(f"[config] Unknown parse_mode='{parse_mode}', fallback to 'incremental'", bcolors.WARNING)
        return "incremental"
    return parse_mode


parse_mode = sanitize_parse_mode(parse_mode)

"""
    Recursively filters a dict and returns only branches that contain provided keys
"""
def filter_dict_by_keys(data_dict, target_keys):
    if isinstance(data_dict, dict):
        direct_matches = {key: value for key, value in data_dict.items() if key in target_keys}
        
        nested_matches = {key: filter_dict_by_keys(value, target_keys) 
                          for key, value in data_dict.items() if key not in direct_matches}
        
        combined = {**direct_matches, **{key: value for key, value in nested_matches.items() if value}}
        return combined if combined else None
    
    elif isinstance(data_dict, list):
        filtered_list = [filter_dict_by_keys(item, target_keys) for item in data_dict]
        return [item for item in filtered_list if item]
    
    return None

"""
    Combines data from differents dict into a list (if keys match)
"""
def combine_dicts(dicts):
    combined = defaultdict(list)

    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict):
                sub_dicts = [c.get(key, {}) for c in dicts if key in c]
                combined[key] = combine_dicts(sub_dicts)
            else:
                if isinstance(value, list):
                    combined[key].extend(value)
                else:
                    combined[key].append(value)

    return dict(combined)

def getClassId(names_dict, class_name):
    for key, values in names_dict.items():
        if class_name in values:
            return key

    return None

def process_dir(args):
    dirpath, filenames, parse_mode = args
    results = []
    try:
        has_cached = "results.yaml.cached" in filenames
        has_raw = "results.yaml" in filenames and 'predictions.json' in filenames

        raw_path = os.path.join(dirpath, "results.yaml")
        cached_path = os.path.join(dirpath, "results.yaml.cached")

        use_cached = False
        parse_raw = False

        if parse_mode == "scratch":
            parse_raw = has_raw
        elif parse_mode == "cached":
            use_cached = has_cached
            parse_raw = (not has_cached) and has_raw
        elif parse_mode == "incremental":
            if has_raw:
                if not has_cached:
                    parse_raw = True
                else:
                    raw_mtime = os.path.getmtime(raw_path)
                    cached_mtime = os.path.getmtime(cached_path)
                    parse_raw = raw_mtime > cached_mtime
                    use_cached = not parse_raw
            elif has_cached:
                use_cached = True

        if use_cached:
            # log(f"[process_dir] Parse cached data from {cached_path}")
            data = getMetricsLogFile(cached_path)
            results.append(data)
        elif parse_raw:
            args_file = os.path.join(dirpath, "args.yaml")
            # log(f"[process_dir] Parsing data from {raw_path}")
            data = getMetricsLogFile(raw_path)
            # log(f"[process_dir] Parsed results.yaml file.")
            data_args = getMetricsLogFile(args_file)
            # log(f"[process_dir] Parsed args.yaml file.")
            data = filter_dict_by_keys(data, ['P', 'R', 'mAP50', 'mAP50-95', 'names', 'train_data', 'rgb_equalization', 'seed', 'thermal_equalization', 'train_duration_h', 'epoch_best_fit_index'])
            data['batch'] = data_args.get('batch', None)
            updateMetricsLogFile(data, cached_path)
            results.append(data)
    except Exception as e:
        log(f"[process_dir] Exception catched: {e}", bcolors.ERROR)
        raise
    return results

def getYOLOVarianceData(analysis_path, output_path, test_tag="", valid_runs=None):
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)

    summary_file = os.path.join(output_path, 'summary_variance_data.yaml' if valid_runs is None else 'summary_variance_data_valid.yaml')
    if parse_mode == "cached" and os.path.exists(summary_file):
        data_combined = getMetricsLogFile(summary_file)
    else:
        data_combined = []
        tasks = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            # In sorted dirpath order, so that the order of the runs (and the seeded resampling that depends on it) does
            # not change with the file system or with which thread finishes first
            for dirpath, dirnames, filenames in sorted(os.walk(output_path), key=lambda entry: entry[0]):
                if valid_runs is not None and 'args.yaml' in filenames and os.path.normpath(dirpath) not in valid_runs:
                    continue
                tasks.append(executor.submit(process_dir, (dirpath, filenames, parse_mode)))
            for future in tasks:
                try:
                    result = future.result()
                    if result:
                        data_combined.extend(result)
                except Exception as e:
                    log(f"[getYOLOVarianceData] Exception catched: {e}", bcolors.ERROR)

        if not data_combined:
            log(f"[getYOLOVarianceData] No runs to analyze for {test_tag} in {output_path}", bcolors.WARNING)
            return {}

        data_combined = combine_dicts(data_combined)
        # print(f"After combination:\n{data_combined}")
        updateMetricsLogFile(data_combined, summary_file)

    n_data = 0
    headers = ['Id', 'Class', 'mAP50 (mean)', 'mAP50 (std)', 'mAP50-95 (mean)', 'mAP50-95 (std)']
    row_data = []
    all_row = []
    for index, (class_key, data) in enumerate(data_combined['validation_0']['data'].items()):
        class_id = getClassId(data_combined['dataset_info']['names'],class_key)
        n_data = len(data['mAP50'])
        mean1 = np.mean(data['mAP50'])
        std1 = np.std(data['mAP50'])
        mean2 = np.mean(data['mAP50-95'])
        std2 = np.std(data['mAP50-95'])
        row = [class_id, class_key,mean1,std1,mean2,std2]
        if class_key == "all":
            all_row = [0] + row[1:] # Just to order it at the end
            continue
        row_data.append(row)
    

    data_array = np.array(row_data)
    numerical_data = data_array[:, 2:].astype(float)
    column_means = np.mean(numerical_data, axis=0)
    column_stds = np.std(numerical_data, axis=0)
    summary_row_mean = ['1','Summary (Mean each col)'] + column_means.tolist()
    summary_row_std = ['2','Summary (Std each col)'] + column_stds.tolist()

    
    sorted_rows = sorted(row_data, key=lambda x: x[0])
    row_data = [headers] + sorted_rows
    logTable(row_data, f"{analysis_path}/tables", f"Variance of YOLO with {test_tag} with {n_data} trials.", 
             colalign=['left', 'right', 'right', 'right', 'right'])

    summary_data = [headers]
    summary_data.extend([summary_row_mean,summary_row_std,all_row])
    logTable(summary_data, f"{analysis_path}/tables", f"Summary variance of YOLO with {test_tag} with {n_data} trials.", 
             colalign=['left', 'right', 'right', 'right', 'right'])
    
    log(f"[getYOLOVarianceData] Finished processing data from {output_path}")
    return {test_tag: data_combined}

def plotYOLODistribution(data_plot, analysis_path, class_tag = 'all', metric_name = 'mAP50', tag_name = ''):
        data = {}
        for key, data_items in data_plot.items():
            data[key] = data_items['validation_0']['data'][class_tag][metric_name]

        plot_metric_distribution(data, train_duration_data = None, metric_label = yolo_metric_labels.get(metric_name, metric_name), plot_func=plot_metric_normaldistribution,
                                color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = analysis_path,
                                plot_filename = f'{tag_name}_{metric_name}', paper_size=True)
        
def plotYOLOTrainDurationDistribution(data_plot, analysis_path, tag_name = ''):
    data_epoch = {}
    data_time = {}
    for key, data_items in data_plot.items():
        data_epoch[key] = [item for item in data_items['train_data']['epoch_best_fit_index']] # To seconds
        data_time[key] = [item*60 for item in data_items['train_data']['train_duration_h']] # To seconds
        
        # data_epoch[key].remove(max(data_epoch[key])) # Keep only 40 samples..
        # data_time[key].remove(max(data_time[key]))

    plot_metric_distribution(data_epoch, train_duration_data = None, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, 
                             color_palette=color_palette_list, vertical_lines_acc=[], analysis_path=analysis_path,
                             plot_filename = f'plot_best_epoch_{tag_name}',
                             bin_size=30, paper_size=True)
    plot_metric_distribution(data_time, train_duration_data = None, metric_label = 'Train duration (min)', plot_func=plot_metric_gammadistribution, 
                             color_palette=color_palette_list, vertical_lines_acc=[], analysis_path=analysis_path,
                             plot_filename = f'plot_time_duration_{tag_name}',
                             bin_size=30, paper_size=True)


def yolo_to_metrics_data(yolo_results, metric='mAP50', class_tag='all'):
    """Convert YOLO results structure to metrics_data expected by plotting utilities.

    Returns dict: {model_name: {run_id: {metric: value}, ...}, ...}
    """
    metrics = {}
    for model_name, items in yolo_results.items():
        try:
            vals = items['validation_0']['data'][class_tag][metric]
        except Exception:
            # fallback: try other possible locations or skip
            continue
        # Ensure iterable
        vals = list(vals)
        metrics[model_name] = {str(i): {metric: float(v)} for i, v in enumerate(vals)}
    return metrics


def plot_sampling_from_yolo(yolo_results, analysis_path, metric='mAP50', class_tag='all', n_iterations=None, percentile=95, percentile_range=[80,100]):
    """Convenience wrapper: convert YOLO results and call sampling/percentile plotting.

    - Calls `plot_all_sampling_errors` and `plot_all_percentile_probabilities` with converted data.
    """
    metrics_data = yolo_to_metrics_data(yolo_results, metric=metric, class_tag=class_tag)
    if not metrics_data:
        log('[plot_sampling_from_yolo] No metrics found for provided data', bcolors.WARNING)
        return

    # Sampling errors (bootstrap/montecarlo/analytical are selected inside plot functions)
    plot_all_sampling_errors(metrics_data=metrics_data, title_tag=f"yolo_{metric}", analysis_path=analysis_path, metric=metric, n_iterations=n_iterations)

    # Percentile probabilities
    plot_all_percentile_probabilities(metrics_data=metrics_data, title_tag=f"yolo_{metric}", analysis_path=analysis_path, metric=metric, percentile=percentile, n_iterations=n_iterations)
    plot_all_percentile_probabilities_average(metrics_data=metrics_data, title_tag=f"yolo_{metric}", analysis_path=analysis_path, metric=metric, percentile_range=percentile_range, n_iterations=n_iterations)

def computeSwitchedProbabilityYolo(yolo_results, model_keys, metric_name='mAP50', analysis_path=None):
    metrics_data = yolo_to_metrics_data(yolo_results, metric=metric_name, class_tag='all')
    switched_data = {}

    for model_name in model_keys:
        model_metrics = metrics_data.get(model_name)
        if not model_metrics:
            log(f"[computeSwitchedProbabilityYolo] Skipping '{model_name}': no data available", bcolors.WARNING)
            continue

        metric_values = []
        for sample_data in model_metrics.values():
            if metric_name in sample_data:
                metric_values.append(float(sample_data[metric_name]))

        if not metric_values:
            log(f"[computeSwitchedProbabilityYolo] Skipping '{model_name}': metric '{metric_name}' not found", bcolors.WARNING)
            continue

        switched_data[model_name] = metric_values

    if len(switched_data) < 2:
        log(f"[computeSwitchedProbabilityYolo] Not enough models with valid data for switched probability: {model_keys}", bcolors.WARNING)
        return

    computeSwitchedProbability(dict_data=switched_data, g_names=list(switched_data.keys()), analysis_path=analysis_path)


if __name__ == "__main__":
    distributions_path = f"{analysis_path}/distributions"
    sampling_path = f"{analysis_path}/sampling"
    ablation_path = f"{analysis_path}/ablation"
    os.makedirs(distributions_path, exist_ok=True)
    os.makedirs(sampling_path, exist_ok=True)
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)
    os.makedirs(f"{sampling_path}/tables", exist_ok=True)
    os.makedirs(f"{ablation_path}/tables", exist_ok=True)

    if parse_mode == "scratch":
        log(f"[main] parse_mode='scratch': parse raw data and regenerate cache/summary", bcolors.OKGREEN)
    elif parse_mode == "incremental":
        log(f"[main] parse_mode='incremental': detect new/updated raw data and reuse valid cache", bcolors.OKGREEN)
    else:
        log(f"[main] parse_mode='cached': use summary/cache without refreshing", bcolors.WARNING)
    
    tasks = [
        # Output Path, Data Path, Tag name :)
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'coco'), 'COCO'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_day_rgbt'), 'kaist_day_rgbt'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_day_vths_v2'), 'kaist_day_vths_v2'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_day_hsvt'), 'kaist_day_hsvt'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths_llvip/variance_vths_llvip_no_equalization'), 'LLVIP_VTHS_no_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths_llvip/variance_vths_llvip_rgb_th_equalization'), 'LLVIP_VTHS_rgb_th_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths_llvip/variance_vths_llvip_th_equalization'), 'LLVIP_VTHS_th_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths_llvip/variance_vths_llvip_rgb_equalization'), 'LLVIP_VTHS_rgb_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vt_llvip/variance_vt_llvip_no_equalization'), 'LLVIP_VT_no_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vt_llvip/variance_vt_llvip_rgb_th_equalization'), 'LLVIP_VT_rgb_th_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vt_llvip/variance_vt_llvip_th_equalization'), 'LLVIP_VT_th_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vt_llvip/variance_vt_llvip_rgb_equalization'), 'LLVIP_VT_rgb_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_llvip_night_lwir'), 'LLVIP_lwir'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_llvip_night_yoloCh4v3'), 'LLVIP_Ch4v3')
    ]

    valid_runs = None
    if read_deposit:
        # The deposit holds the valid runs only
        log(f"[main] Reading the YOLOv8 runs from the data deposit in {deposit_path}", bcolors.OKGREEN)
        results = getDepositYoloResults(deposit_path)
    else:
        if only_valid_runs:
            yolo_runs_inventory = updateYoloRunsInventory(tasks, output_path, yolo_runs_inventory_file, rescan=(parse_mode == "scratch"))
            valid_runs = getValidYoloRuns(yolo_runs_inventory, output_path)

        results = {}
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(getYOLOVarianceData, *task, valid_runs) for task in tasks]
            # In the order of tasks, so that the figures do not depend on which group finishes first
            for future in futures:
                try:
                    results.update(future.result())
                except Exception as e:
                    log(f"[main] Exception in future: {e}", bcolors.ERROR)
    
    if store_metric_standalone_data:        
        for metric, class_tag in zip(['P', 'R', 'mAP50', 'mAP50-95'], ['person', 'person', 'all', 'all']):
            data_raw = yolo_to_metrics_data(results, metric=metric, class_tag=class_tag)
            metric_data = {}
            for model, data in data_raw.items():
                if model not in metric_data:
                    metric_data[model] = []
                for entry in data.values():
                    metrid_item = entry.get(metric)
                    metric_data[model].append(metrid_item)
            dumpYaml(metric_data, f"{analysis_path}/{metric}_data_raw.yaml")


    if enable_yolos_swithced_probability:
        np.random.seed(analysis_seed)
        computeSwitchedProbabilityYolo(results, ['kaist_day_hsvt','kaist_day_vths_v2'], metric_name='mAP50', analysis_path=analysis_path)
        computeSwitchedProbabilityYolo(results, ['LLVIP_VTHS_no_equalization','LLVIP_VTHS_rgb_th_equalization','LLVIP_VTHS_th_equalization','LLVIP_VTHS_rgb_equalization'], metric_name='mAP50', analysis_path=analysis_path)
        computeSwitchedProbabilityYolo(results, ['LLVIP_VT_no_equalization','LLVIP_VT_rgb_th_equalization','LLVIP_VT_th_equalization','LLVIP_VT_rgb_equalization'], metric_name="mAP50", analysis_path=analysis_path)

    if enable_yolo_overfit_analysis and read_deposit:
        log("[main] The overfit analysis reads the results.csv files of the runs; skipped with the data deposit", bcolors.WARNING)
    elif enable_yolo_overfit_analysis:
        # Empty list means: include all tags present in `tasks`.
        yolo_overfit_include_tags = ['kaist_day_rgbt', 'kaist_day_vths_v2']
        runYoloOverfitAnalysis(
            variance_tasks=tasks,
            analysis_output_path=yolo_overfit_analysis_path,
            include_tags=yolo_overfit_include_tags,
            plot_all_runs=yolo_overfit_plot_all_runs,
            all_runs_alpha=yolo_overfit_alpha,
            loss_log_scale=yolo_overfit_loss_log_scale,
            loss_component=yolo_overfit_loss_component,
            quality_metric_key=yolo_overfit_quality_metric_key,
            valid_runs=valid_runs,
        )

    if enable_yolo_survival_function:
        # Final evaluation of each run, the same values as the tables (never the maximum over the epochs)
        if read_deposit:
            survival_map50, survival_map5095 = getDepositYoloFinalMetrics(deposit_path)
        else:
            survival_map50, survival_map5095 = collectYoloFinalMetrics(results)
        if survival_map50:
            plot_survival_function(
                metrics_data=survival_map50,
                analysis_path=distributions_path,
                metric_label=yolo_metric_labels['mAP50'],
                table_filename='Survival Function mAP50',
                plot_prefix='survival_mAP50',
                paper_size=True,
            )
            log("Survival function plot saved for mAP50", bcolors.OKGREEN)
        if survival_map5095:
            plot_survival_function(
                metrics_data=survival_map5095,
                analysis_path=distributions_path,
                metric_label=yolo_metric_labels['mAP50-95'],
                table_filename='Survival Function mAP50-95',
                plot_prefix='survival_mAP50-95',
                paper_size=True,
            )
            log("Survival function plot saved for mAP50-95", bcolors.OKGREEN)

    if enable_yolo_plot_distributions:
        if 'COCO' in results:
            plotYOLODistribution({'COCO': results['COCO']}, distributions_path, metric_name = 'mAP50', tag_name = 'COCO')
            plotYOLODistribution({'COCO': results['COCO']}, distributions_path, metric_name = 'mAP50-95', tag_name = 'COCO')

        data_plot_vths = {}
        for key in ['LLVIP_VTHS_no_equalization',
                    'LLVIP_VTHS_rgb_th_equalization',
                    'LLVIP_VTHS_rgb_equalization',
                    'LLVIP_VTHS_th_equalization'
                    ]:
            data_plot_vths[key] = results[key]
        plotYOLODistribution(data_plot_vths, distributions_path, metric_name = 'mAP50', tag_name = 'LLVIP_VTHS_equalization')
        plotYOLODistribution(data_plot_vths, distributions_path, metric_name = 'mAP50-95', tag_name = 'LLVIP_VTHS_equalization')
        
        data_plot_vt = {}
        for key in ['LLVIP_VT_no_equalization',
                    'LLVIP_VT_rgb_th_equalization',
                    'LLVIP_VT_rgb_equalization',
                    'LLVIP_VT_th_equalization'
                    ]:
            data_plot_vt[key] = results[key]
        plotYOLODistribution(data_plot_vt, distributions_path, metric_name = 'mAP50', tag_name = 'LLVIP_VT_equalization')
        plotYOLODistribution(data_plot_vt, distributions_path, metric_name = 'mAP50-95', tag_name = 'LLVIP_VT_equalization')

        # Fusion methods compared with the same equalization (overlapping distributions)
        data_plot_fusion = {key: results[key] for key in ['LLVIP_VT_no_equalization', 'LLVIP_VTHS_no_equalization']}
        plotYOLODistribution(data_plot_fusion, distributions_path, metric_name = 'mAP50', tag_name = 'LLVIP_fusion_no_equalization')
        plotYOLODistribution(data_plot_fusion, distributions_path, metric_name = 'mAP50-95', tag_name = 'LLVIP_fusion_no_equalization')

        ablation_equalization = data_plot_vths.copy()
        ablation_equalization.update(data_plot_vt)
        plotYOLODistribution(ablation_equalization, distributions_path, metric_name = 'mAP50', tag_name = 'LLVIP_equalization_ablation')
        plotYOLODistribution(ablation_equalization, distributions_path, metric_name = 'mAP50-95', tag_name = 'LLVIP_equalization_ablation')

        data_plot_day = {}
        for key in ['kaist_day_rgbt', 'kaist_day_vths_v2', 'kaist_day_hsvt']:
            if key in results:
                data_plot_day[key] = results[key]

        plotYOLODistribution(data_plot_day, distributions_path, metric_name = 'mAP50', tag_name = 'KAIST_day')
        plotYOLODistribution(data_plot_day, distributions_path, metric_name = 'mAP50-95', tag_name = 'KAIST_day')

        # Single-group distributions and best epoch (introduction figure)
        for key in ['kaist_day_rgbt', 'kaist_day_vths_v2', 'kaist_day_hsvt', 'LLVIP_lwir']:
            if key not in results:
                continue
            plotYOLODistribution({key: results[key]}, distributions_path, metric_name = 'mAP50', tag_name = key)
            plotYOLODistribution({key: results[key]}, distributions_path, metric_name = 'mAP50-95', tag_name = key)
            plotYOLOTrainDurationDistribution({key: results[key]}, distributions_path, tag_name = key)

        if 'LLVIP_Ch4v3' in results:
            plotYOLODistribution({'LLVIP_Ch4v3': results['LLVIP_Ch4v3']}, distributions_path, metric_name = 'mAP50', tag_name = 'LLVIP_Ch4v3')
            plotYOLODistribution({'LLVIP_Ch4v3': results['LLVIP_Ch4v3']}, distributions_path, metric_name = 'mAP50-95', tag_name = 'LLVIP_Ch4v3')
            plotYOLOTrainDurationDistribution({'LLVIP_Ch4v3': results['LLVIP_Ch4v3']}, distributions_path, tag_name = 'LLVIP_Ch4v3')

    if enable_yolo_sampling_plots:
        np.random.seed(analysis_seed)
        plot_sampling_from_yolo(results, analysis_path=sampling_path, metric='mAP50', percentile=90)
        np.random.seed(analysis_seed)
        plot_sampling_from_yolo(results, analysis_path=sampling_path, metric='mAP50-95', percentile=90)


    if enable_yolo_ablation_tests:
        indexer = Indexer()
        table_data = [["index", "fusion", "rgb_eq", "th_eq", "R", "P", "mAP50", "mAP50-95"]]

        ablation_equalization = {}
        for key in ['LLVIP_VT_no_equalization',
                    'LLVIP_VT_rgb_th_equalization',
                    'LLVIP_VT_rgb_equalization',
                    'LLVIP_VT_th_equalization',
                    'LLVIP_VTHS_no_equalization',
                    'LLVIP_VTHS_rgb_th_equalization',
                    'LLVIP_VTHS_rgb_equalization',
                    'LLVIP_VTHS_th_equalization'
                    ]:
            for iter_n, mAP50 in enumerate(results[key]['validation_0']['data']['person']['mAP50']):
                ablation_equalization[key] = results[key]
                index = indexer.get_index()
                fusion = "VTHS" if "VTHS" in key else "VT"
                # print_dict(results[key])
                # print_dict_keys(results[key])
                rgb_eq = results[key]['rgb_equalization'][iter_n]
                th_eq = results[key]['thermal_equalization'][iter_n]
                p = results[key]['validation_0']['data']['person']['P'][iter_n]
                r = results[key]['validation_0']['data']['person']['R'][iter_n]
                mAP50_95 = results[key]['validation_0']['data']['person']['mAP50-95'][iter_n]
                table_data.append([index, fusion, rgb_eq, th_eq, r, p, mAP50, mAP50_95])

        with open(f"{ablation_path}/tables/ablation_table_summary.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(table_data)
