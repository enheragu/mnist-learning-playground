#!/usr/bin/env python3
# encoding: utf-8

import os

from utils import output_path
from utils.yaml_utils import getMetricsLogFile
from utils.log_utils import log, bcolors
from utils.overfit_analysis import analyzeAndPlotOverfitCurves


analysis_path = './analysis_results/overfit_analysis'
metrics_file_name = 'randomseed_training_metrics.yaml'

# If empty, all folders that match *_overfit_* are analyzed.
requested_models = []

# Plot configuration
plot_all_runs = False
all_runs_alpha = 0.12
loss_log_scale = True


def _discover_overfit_models(base_output_path, metrics_file_name):
    model_names = []
    for entry in sorted(os.listdir(base_output_path)):
        model_dir = os.path.join(base_output_path, entry)
        if not os.path.isdir(model_dir):
            continue
        if '_overfit_' not in entry:
            continue
        metrics_file = os.path.join(model_dir, metrics_file_name)
        if os.path.exists(metrics_file):
            model_names.append(entry)
    return model_names


def _resolve_models(base_output_path, metrics_file_name, requested_models):
    if requested_models:
        return requested_models
    return _discover_overfit_models(base_output_path, metrics_file_name)


def analyze_model(model_name,
                  analysis_path,
                  metrics_file_name,
                  plot_all_runs,
                  all_runs_alpha,
                  loss_log_scale):
    metrics_path = os.path.join(output_path, model_name, metrics_file_name)
    if not os.path.exists(metrics_path):
        log(f"[07_check_overfit_analysis] Missing metrics file for '{model_name}': {metrics_path}", bcolors.WARNING)
        return

    metrics_data = getMetricsLogFile(metrics_path)
    if not metrics_data:
        log(f"[07_check_overfit_analysis] Empty metrics data for '{model_name}'", bcolors.WARNING)
        return

    model_analysis_path = os.path.join(analysis_path)
    result = analyzeAndPlotOverfitCurves(
        metrics_data=metrics_data,
        analysis_path=model_analysis_path,
        model_name=model_name,
        plot_all_runs=plot_all_runs,
        all_runs_alpha=all_runs_alpha,
        loss_log_scale=loss_log_scale,
    )

    log(f"[07_check_overfit_analysis] Stored overfit plot for '{model_name}' at {result['plot_path']}", bcolors.OKGREEN)


if __name__ == '__main__':
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)

    models_to_analyze = _resolve_models(output_path, metrics_file_name, requested_models)
    if len(models_to_analyze) == 0:
        log('[07_check_overfit_analysis] No overfit models found to analyze.', bcolors.WARNING)
        raise SystemExit(0)

    log(f"[07_check_overfit_analysis] Models to analyze: {models_to_analyze}", bcolors.OKCYAN)

    for model_name in models_to_analyze:
        analyze_model(
            model_name=model_name,
            analysis_path=analysis_path,
            metrics_file_name=metrics_file_name,
            plot_all_runs=plot_all_runs,
            all_runs_alpha=all_runs_alpha,
            loss_log_scale=loss_log_scale,
        )
