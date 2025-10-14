#!/usr/bin/env python3
# encoding: utf-8

import os
from utils.log_utils import log, logTable
from utils.yaml_utils import getMetricsLogFile


def getAllModelData(output_path):
    metrics_file_name = "randomseed_training_metrics.yaml"
    metrics_data = {}

    # List each folder (assumed to be a model) in output_path
    for model_name in os.listdir(output_path):
        model_path = os.path.join(output_path, model_name)
        
        # Check if it's a directory and contains the metrics file
        if os.path.isdir(model_path):
            metrics_file = os.path.join(model_path, metrics_file_name)
            if os.path.exists(metrics_file):
                # Load metrics data for the model
                metrics_data[model_name] = getMetricsLogFile(metrics_file)
            else:
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
