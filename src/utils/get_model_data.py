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