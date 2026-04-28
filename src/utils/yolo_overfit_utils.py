#!/usr/bin/env python3
# encoding: utf-8

import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from utils.log_utils import log, logTable, bcolors, color_palette_list
from utils.overfit_analysis import analyzeAndPlotOverfitCurves


def _to_float_or_nan(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def yolo_results_csv_to_metrics_run(csv_path):
    if not os.path.exists(csv_path):
        log(f"[yolo_results_csv_to_metrics_run] CSV not found: {csv_path}", bcolors.WARNING)
        return None

    train_loss_curve = []
    test_loss_curve = []
    train_box_loss_curve = []
    train_cls_loss_curve = []
    train_dfl_loss_curve = []
    test_box_loss_curve = []
    test_cls_loss_curve = []
    test_dfl_loss_curve = []
    test_map50_curve = []
    test_map50_95_curve = []
    test_precision_curve = []
    test_recall_curve = []
    lr_mean_curve = []

    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file, skipinitialspace=True)
        if reader.fieldnames is None:
            log(f"[yolo_results_csv_to_metrics_run] Empty CSV or missing header: {csv_path}", bcolors.WARNING)
            return None

        normalized_fieldnames = [name.strip() for name in reader.fieldnames]

        for raw_row in reader:
            row = {key.strip(): value for key, value in raw_row.items() if key is not None}

            train_box = _to_float_or_nan(row.get('train/box_loss', np.nan))
            train_cls = _to_float_or_nan(row.get('train/cls_loss', np.nan))
            train_dfl = _to_float_or_nan(row.get('train/dfl_loss', np.nan))
            val_box = _to_float_or_nan(row.get('val/box_loss', np.nan))
            val_cls = _to_float_or_nan(row.get('val/cls_loss', np.nan))
            val_dfl = _to_float_or_nan(row.get('val/dfl_loss', np.nan))

            train_total = np.nansum([train_box, train_cls, train_dfl])
            val_total = np.nansum([val_box, val_cls, val_dfl])

            map50_key = 'metrics/mAP50(B)' if 'metrics/mAP50(B)' in normalized_fieldnames else 'metrics/mAP50'
            map50_95_key = 'metrics/mAP50-95(B)' if 'metrics/mAP50-95(B)' in normalized_fieldnames else 'metrics/mAP50-95'
            precision_key = 'metrics/precision(B)' if 'metrics/precision(B)' in normalized_fieldnames else 'metrics/precision'
            recall_key = 'metrics/recall(B)' if 'metrics/recall(B)' in normalized_fieldnames else 'metrics/recall'

            map50_value = _to_float_or_nan(row.get(map50_key, np.nan))
            map50_95_value = _to_float_or_nan(row.get(map50_95_key, np.nan))
            precision_value = _to_float_or_nan(row.get(precision_key, np.nan))
            recall_value = _to_float_or_nan(row.get(recall_key, np.nan))

            lr_pg0 = _to_float_or_nan(row.get('lr/pg0', np.nan))
            lr_pg1 = _to_float_or_nan(row.get('lr/pg1', np.nan))
            lr_pg2 = _to_float_or_nan(row.get('lr/pg2', np.nan))
            lr_candidates = [lr_pg0, lr_pg1, lr_pg2]
            lr_mean = np.nanmean(lr_candidates) if np.any(np.isfinite(lr_candidates)) else np.nan

            train_loss_curve.append(train_total)
            test_loss_curve.append(val_total)
            train_box_loss_curve.append(train_box)
            train_cls_loss_curve.append(train_cls)
            train_dfl_loss_curve.append(train_dfl)
            test_box_loss_curve.append(val_box)
            test_cls_loss_curve.append(val_cls)
            test_dfl_loss_curve.append(val_dfl)
            test_map50_curve.append(map50_value)
            test_map50_95_curve.append(map50_95_value)
            test_precision_curve.append(precision_value)
            test_recall_curve.append(recall_value)
            lr_mean_curve.append(lr_mean)

    if len(train_loss_curve) == 0:
        log(f"[yolo_results_csv_to_metrics_run] No epochs parsed from: {csv_path}", bcolors.WARNING)
        return None

    return {
        'train_loss_plot': train_loss_curve,
        'test_loss_plot': test_loss_curve,
        'train_box_loss_plot': train_box_loss_curve,
        'train_cls_loss_plot': train_cls_loss_curve,
        'train_dfl_loss_plot': train_dfl_loss_curve,
        'test_box_loss_plot': test_box_loss_curve,
        'test_cls_loss_plot': test_cls_loss_curve,
        'test_dfl_loss_plot': test_dfl_loss_curve,
        'mAP50': test_map50_curve,
        'mAP50-95': test_map50_95_curve,
        'precision': test_precision_curve,
        'recall': test_recall_curve,
        'lr_mean_plot': lr_mean_curve,
    }


def discoverYoloResultsCSV(experiment_output_path):
    csv_paths = []
    missing_csv_dirs = []

    for dirpath, dirnames, filenames in os.walk(experiment_output_path):
        if '.exclude_from_analysis' in filenames:
            dirnames.clear()  # prune subtree
            continue

        has_args = 'args.yaml' in filenames
        has_results_csv = 'results.csv' in filenames

        if has_results_csv:
            csv_paths.append(os.path.join(dirpath, 'results.csv'))
        elif has_args:
            missing_csv_dirs.append(dirpath)

    return sorted(csv_paths), sorted(missing_csv_dirs)


def _curves_to_matrix(curves):
    valid_curves = [np.asarray(curve, dtype=float) for curve in curves if curve is not None and len(curve) > 0]
    if len(valid_curves) == 0:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(curve) for curve in valid_curves)
    matrix = np.full((len(valid_curves), max_len), np.nan, dtype=float)
    for idx, curve in enumerate(valid_curves):
        matrix[idx, :len(curve)] = curve
    return matrix


def _extract_metric_curves(metrics_data, metric_key):
    curves = []
    for _, run_data in sorted(metrics_data.items(), key=lambda item: item[0]):
        if not isinstance(run_data, dict):
            continue
        curve = run_data.get(metric_key)
        if curve is None:
            continue
        curves.append(curve)
    return curves


def resolve_loss_log_scale(log_scale_setting, metric_curves):
    if isinstance(log_scale_setting, bool):
        return log_scale_setting

    setting = str(log_scale_setting).strip().lower()
    if setting in {'true', '1', 'yes'}:
        return True
    if setting in {'false', '0', 'no'}:
        return False

    finite_values = []
    for curve in metric_curves:
        if curve is None:
            continue
        arr = np.asarray(curve, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = arr[arr > 0]
        if arr.size:
            finite_values.append(arr)

    if not finite_values:
        return False

    values = np.concatenate(finite_values)
    if values.size < 10:
        return False

    low = np.nanpercentile(values, 5)
    high = np.nanpercentile(values, 95)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0:
        return False

    dynamic_ratio = high / low
    return dynamic_ratio >= 20.0


def plotYoloLossComponentsOverview(metrics_data,
                                   analysis_output_path,
                                   model_name,
                                   loss_log_scale=True,
                                   plot_all_runs=False,
                                   all_runs_alpha=0.08):
    component_specs = [
        ('box', 'train_box_loss_plot', 'test_box_loss_plot', 'Box loss'),
        ('cls', 'train_cls_loss_plot', 'test_cls_loss_plot', 'Cls loss'),
        ('dfl', 'train_dfl_loss_plot', 'test_dfl_loss_plot', 'Dfl loss'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    plotted_any = False

    for axis, (_, train_key, test_key, label) in zip(axes, component_specs):
        train_matrix = _curves_to_matrix(_extract_metric_curves(metrics_data, train_key))
        test_matrix = _curves_to_matrix(_extract_metric_curves(metrics_data, test_key))

        if train_matrix.size == 0 and test_matrix.size == 0:
            axis.text(0.5, 0.5, f'{label}: no data', ha='center', va='center', transform=axis.transAxes)
            axis.grid(True, alpha=0.3)
            continue

        plotted_any = True
        max_len = max(train_matrix.shape[1] if train_matrix.size else 0,
                      test_matrix.shape[1] if test_matrix.size else 0)
        epochs = np.arange(1, max_len + 1)

        train_mean = np.full(max_len, np.nan, dtype=float)
        train_min = np.full(max_len, np.nan, dtype=float)
        train_max = np.full(max_len, np.nan, dtype=float)
        test_mean = np.full(max_len, np.nan, dtype=float)
        test_min = np.full(max_len, np.nan, dtype=float)
        test_max = np.full(max_len, np.nan, dtype=float)

        if train_matrix.size:
            train_mean[:train_matrix.shape[1]] = np.nanmean(train_matrix, axis=0)
            train_min[:train_matrix.shape[1]] = np.nanmin(train_matrix, axis=0)
            train_max[:train_matrix.shape[1]] = np.nanmax(train_matrix, axis=0)
            if plot_all_runs:
                for curve in train_matrix:
                    curve_plot = np.where(curve > 0, curve, np.nan) if loss_log_scale else curve
                    axis.plot(epochs[:train_matrix.shape[1]], curve_plot, color=color_palette_list[0], alpha=all_runs_alpha, linewidth=0.8)

        if test_matrix.size:
            test_mean[:test_matrix.shape[1]] = np.nanmean(test_matrix, axis=0)
            test_min[:test_matrix.shape[1]] = np.nanmin(test_matrix, axis=0)
            test_max[:test_matrix.shape[1]] = np.nanmax(test_matrix, axis=0)
            if plot_all_runs:
                for curve in test_matrix:
                    curve_plot = np.where(curve > 0, curve, np.nan) if loss_log_scale else curve
                    axis.plot(epochs[:test_matrix.shape[1]], curve_plot, color=color_palette_list[3], alpha=all_runs_alpha, linewidth=0.8)

        if loss_log_scale:
            train_mean = np.where(train_mean > 0, train_mean, np.nan)
            train_min = np.where(train_min > 0, train_min, np.nan)
            train_max = np.where(train_max > 0, train_max, np.nan)
            test_mean = np.where(test_mean > 0, test_mean, np.nan)
            test_min = np.where(test_min > 0, test_min, np.nan)
            test_max = np.where(test_max > 0, test_max, np.nan)

        valid_train = np.isfinite(train_mean)
        valid_test = np.isfinite(test_mean)

        if np.any(valid_train):
            axis.fill_between(epochs[valid_train], train_min[valid_train], train_max[valid_train], color=color_palette_list[0], alpha=0.15)
            axis.plot(epochs[valid_train], train_mean[valid_train], color=color_palette_list[0], linewidth=2.0, label='Train (min-mean-max)')
        if np.any(valid_test):
            axis.fill_between(epochs[valid_test], test_min[valid_test], test_max[valid_test], color=color_palette_list[3], alpha=0.15)
            axis.plot(epochs[valid_test], test_mean[valid_test], color=color_palette_list[3], linewidth=2.0, label='Validation (min-mean-max)')

        ylabel = f'{label} (log scale)' if loss_log_scale else label
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(loc='best', fontsize='small')
        if loss_log_scale:
            axis.set_yscale('log')

    axes[-1].set_xlabel('Epoch')

    if not plotted_any:
        plt.close(fig)
        return None

    n_samples = len(metrics_data)
    axes[0].set_title(f'Training Process - {model_name} - {n_samples} Samples - Box/Cls/Dfl')
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_file = os.path.join(analysis_output_path, f'overfit_components_{model_name}.pdf')
    fig.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return output_file


def runYoloOverfitAnalysis(variance_tasks,
                           analysis_output_path,
                           include_tags=None,
                           plot_all_runs=False,
                           all_runs_alpha=0.12,
                           loss_log_scale=True,
                           loss_component='total',
                           quality_metric_key='mAP50',
                           quality_label=None):
    os.makedirs(analysis_output_path, exist_ok=True)
    os.makedirs(f"{analysis_output_path}/tables", exist_ok=True)

    include_tags = include_tags or []
    quality_label = quality_label or quality_metric_key
    valid_loss_components = {'total', 'box', 'cls', 'dfl', 'all'}
    if loss_component not in valid_loss_components:
        log(f"[runYoloOverfitAnalysis] Unknown loss_component='{loss_component}', fallback to 'total'", bcolors.WARNING)
        loss_component = 'total'

    component_specs = {
        'total': ('train_loss_plot', 'test_loss_plot', 'Loss'),
        'box': ('train_box_loss_plot', 'test_box_loss_plot', 'Box loss'),
        'cls': ('train_cls_loss_plot', 'test_cls_loss_plot', 'Cls loss'),
        'dfl': ('train_dfl_loss_plot', 'test_dfl_loss_plot', 'Dfl loss'),
    }
    selected_components = ['total', 'box', 'cls', 'dfl'] if loss_component == 'all' else [loss_component]

    for _, experiment_output_path, tag in variance_tasks:
        if include_tags and tag not in include_tags:
            continue
        if os.path.exists(os.path.join(experiment_output_path, '.exclude_from_analysis')):
            log(f"[runYoloOverfitAnalysis] Skipping '{tag}' due to .exclude_from_analysis marker", bcolors.WARNING)
            continue

        csv_paths, missing_csv_dirs = discoverYoloResultsCSV(experiment_output_path)
        log(f"[runYoloOverfitAnalysis] Discovered {len(csv_paths)} results.csv files for '{tag}'", bcolors.OKBLUE)
        if missing_csv_dirs:
            log(f"[runYoloOverfitAnalysis] Found {len(missing_csv_dirs)} runs with args.yaml but without results.csv for '{tag}'", bcolors.WARNING)

        metrics_data = {}
        for index, csv_path in enumerate(csv_paths):
            run_data = yolo_results_csv_to_metrics_run(csv_path)
            if run_data is None:
                continue
            metrics_data[f'run_{index}'] = run_data

        if not metrics_data:
            log(f"[runYoloOverfitAnalysis] No valid CSV runs for '{tag}' in {experiment_output_path}", bcolors.WARNING)
            continue

        component_interpretations = []
        for selected_component in selected_components:
            train_loss_metric_key, test_loss_metric_key, loss_label = component_specs[selected_component]
            model_name = tag if selected_component == 'total' else f"{tag}_{selected_component}"

            train_curves = _extract_metric_curves(metrics_data, train_loss_metric_key)
            test_curves = _extract_metric_curves(metrics_data, test_loss_metric_key)
            component_loss_log_scale = resolve_loss_log_scale(loss_log_scale, train_curves + test_curves)

            analysis_result = analyzeAndPlotOverfitCurves(
                metrics_data=metrics_data,
                analysis_path=analysis_output_path,
                model_name=model_name,
                plot_all_runs=plot_all_runs,
                all_runs_alpha=all_runs_alpha,
                loss_log_scale=component_loss_log_scale,
                train_loss_metric_key=train_loss_metric_key,
                test_loss_metric_key=test_loss_metric_key,
                loss_label=loss_label,
                quality_metric_key=quality_metric_key,
                train_quality_metric_key=None,
                quality_label=quality_label,
                lr_metric_key='lr_mean_plot',
                train_component_metric_keys=['train_box_loss_plot', 'train_cls_loss_plot', 'train_dfl_loss_plot'],
                test_component_metric_keys=['test_box_loss_plot', 'test_cls_loss_plot', 'test_dfl_loss_plot'],
            )
            component_interpretations.append((selected_component, analysis_result.get('interpretation', '-')))
            scale_tag = 'log' if component_loss_log_scale else 'linear'
            log(f"[runYoloOverfitAnalysis] Scale for '{model_name}': {scale_tag}", bcolors.OKBLUE)
            log(f"[runYoloOverfitAnalysis] Stored overfit analysis for '{model_name}' from {len(metrics_data)} CSV runs", bcolors.OKGREEN)

        if loss_component == 'all' and component_interpretations:
            overview_curves = []
            for train_key in ['train_box_loss_plot', 'train_cls_loss_plot', 'train_dfl_loss_plot']:
                overview_curves.extend(_extract_metric_curves(metrics_data, train_key))
            for test_key in ['test_box_loss_plot', 'test_cls_loss_plot', 'test_dfl_loss_plot']:
                overview_curves.extend(_extract_metric_curves(metrics_data, test_key))
            overview_loss_log_scale = resolve_loss_log_scale(loss_log_scale, overview_curves)

            overview_plot = plotYoloLossComponentsOverview(
                metrics_data=metrics_data,
                analysis_output_path=analysis_output_path,
                model_name=tag,
                loss_log_scale=overview_loss_log_scale,
                plot_all_runs=plot_all_runs,
                all_runs_alpha=all_runs_alpha,
            )
            if overview_plot is not None:
                scale_tag = 'log' if overview_loss_log_scale else 'linear'
                log(f"[runYoloOverfitAnalysis] Scale for component overview '{tag}': {scale_tag}", bcolors.OKBLUE)
                log(f"[runYoloOverfitAnalysis] Stored component overview plot for '{tag}' at {overview_plot}", bcolors.OKGREEN)

            interpretation_counts = {'overfit': 0, 'underfit': 0, 'mixed/balanced': 0}
            for _, label in component_interpretations:
                if label in interpretation_counts:
                    interpretation_counts[label] += 1

            if interpretation_counts['overfit'] >= 2:
                consensus_label = 'overfit'
            elif interpretation_counts['underfit'] >= 2:
                consensus_label = 'underfit'
            else:
                consensus_label = 'mixed/balanced'

            consensus_table = [[
                'Component',
                'Interpretation'
            ]]
            for component_name, label in component_interpretations:
                consensus_table.append([component_name, label])
            consensus_table.append(['consensus_label', consensus_label])

            logTable(
                row_data=consensus_table,
                output_path=f"{analysis_output_path}/tables",
                filename=f"Overfit consensus - {tag}",
                colalign=['left', 'left']
            )
            log(f"[runYoloOverfitAnalysis] Consensus for '{tag}': {consensus_label} ({interpretation_counts})", bcolors.OKBLUE)


def collectYoloBestMetrics(variance_tasks, include_tags=None):
    """For each task collect the best (max over epochs) mAP50 and mAP50-95 per run.

    Returns:
        survival_map50_data:   {tag: [best_mAP50_per_run]}
        survival_map5095_data: {tag: [best_mAP5095_per_run]}
    """
    include_tags = include_tags or []
    survival_map50_data = {}
    survival_map5095_data = {}

    for _, experiment_output_path, tag in variance_tasks:
        if include_tags and tag not in include_tags:
            continue
        if os.path.exists(os.path.join(experiment_output_path, '.exclude_from_analysis')):
            log(f"[collectYoloBestMetrics] Skipping '{tag}' due to .exclude_from_analysis marker", bcolors.WARNING)
            continue
        csv_paths, _ = discoverYoloResultsCSV(experiment_output_path)
        best_map50 = []
        best_map5095 = []
        for csv_path in csv_paths:
            run_data = yolo_results_csv_to_metrics_run(csv_path)
            if not run_data:
                continue
            curve50 = run_data.get('mAP50', [])
            curve5095 = run_data.get('mAP50-95', [])
            if curve50:
                v = np.nanmax(np.asarray(curve50, dtype=float))
                if np.isfinite(v):
                    best_map50.append(v)
            if curve5095:
                v = np.nanmax(np.asarray(curve5095, dtype=float))
                if np.isfinite(v):
                    best_map5095.append(v)
        if best_map50:
            survival_map50_data[tag] = best_map50
        if best_map5095:
            survival_map5095_data[tag] = best_map5095

    return survival_map50_data, survival_map5095_data
