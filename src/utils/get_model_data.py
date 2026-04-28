#!/usr/bin/env python3
# encoding: utf-8

import os
from concurrent.futures import ThreadPoolExecutor
from utils.log_utils import log
from utils.yaml_utils import getMetricsLogFile


def _load_single_model_metrics(model_name, model_path, metrics_file_name):
    if not os.path.isdir(model_path):
        return 'not_directory', model_name, None

    metrics_file = os.path.join(model_path, metrics_file_name)
    if not os.path.exists(metrics_file):
        return 'missing_file', model_name, None

    return 'ok', model_name, getMetricsLogFile(metrics_file)


def getAllModelData(output_path):
    metrics_file_name = "randomseed_training_metrics.yaml"
    metrics_data = {}

    model_names = sorted(os.listdir(output_path))
    if len(model_names) == 0:
        return metrics_data

    max_workers = min(len(model_names), (os.cpu_count() or 4) * 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        model_args = []
        for model_name in model_names:
            model_path = os.path.join(output_path, model_name)

            if os.path.exists(os.path.join(model_path, '.exclude_from_analysis')):
                log(f"Skipping model '{model_name}' due to .exclude_from_analysis marker")
                continue

            model_args.append((model_name, model_path, metrics_file_name))

        for status, model_name, model_metrics in executor.map(lambda args: _load_single_model_metrics(*args), model_args):
            if status == 'ok':
                metrics_data[model_name] = model_metrics
            elif status == 'missing_file':
                log(f"No metrics file found for model: {model_name}")
            else:
                log(f"{model_name} is not a directory, skipping...")
        
    return metrics_data


"""
    Extracts ablation model data from a YAML file in a compatible format 
    with previous functions and single execution YAML format.
"""
def getAblationModelData(ablation_data_file):
    ablation_metrics = getMetricsLogFile(ablation_data_file)

    metrics_data = {}
    for key, iteration in ablation_metrics.items():
        for trial_idx, (condition, trial) in enumerate(iteration.items()):
            batchs = trial['batch_size']
            learningr = trial['learning_rate']

            if not f'CNN_14L_B{batchs}_L{learningr}' in metrics_data:
                metrics_data[f'CNN_14L_B{batchs}_L{learningr}'] = {}

            metrics_data[f'CNN_14L_B{batchs}_L{learningr}'][trial['finish_timetag']] = \
                {'accuracy': trial['accuracy'],
                 'train_duration': trial['train_duration'],
                 'best_epoch': trial['best_epoch']
                }
    
    return metrics_data


def getAllAndAblationModelData(output_path, ablation_data_file):
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_metrics = executor.submit(getAllModelData, output_path)
        future_ablation = executor.submit(getAblationModelData, ablation_data_file)

        metrics_data = future_metrics.result()
        ablation_metrics = future_ablation.result()

    return metrics_data, ablation_metrics
