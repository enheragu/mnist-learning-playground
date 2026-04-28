#!/usr/bin/env python3
# encoding: utf-8

import os

import numpy as np
import matplotlib.pyplot as plt

from utils.log_utils import logTable, color_palette_list


def _to_float_curve(values):
    if values is None:
        return np.array([], dtype=float)
    return np.asarray(values, dtype=float)


def _extract_curves(metrics_data, metric_key):
    if metric_key is None:
        return []
    curves = []
    for _, run_data in sorted(metrics_data.items(), key=lambda item: item[0]):
        if not isinstance(run_data, dict):
            continue
        curve = _to_float_curve(run_data.get(metric_key))
        if curve.size > 0:
            curves.append(curve)
    return curves


def _curves_to_matrix(curves):
    if len(curves) == 0:
        return np.empty((0, 0), dtype=float)

    max_len = max(len(curve) for curve in curves)
    matrix = np.full((len(curves), max_len), np.nan, dtype=float)

    for idx, curve in enumerate(curves):
        matrix[idx, :len(curve)] = curve

    return matrix


def _safe_value(arr, index):
    if arr.size == 0 or index is None or index < 0 or index >= arr.shape[0]:
        return np.nan
    return arr[index]


def _plot_metric_evolution(ax,
                           train_matrix,
                           test_matrix,
                           metric_label,
                           train_label,
                           test_label,
                           higher_is_better,
                           plot_all_runs,
                           all_runs_alpha,
                           show_xlabel,
                           y_scale='linear'):
    if train_matrix.size == 0 and test_matrix.size == 0:
        y_axis_label = f"{metric_label} (log scale)" if y_scale == 'log' else metric_label
        if show_xlabel:
            ax.set_xlabel("Epoch")
        ax.set_ylabel(y_axis_label)
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        return None

    max_len = max(train_matrix.shape[1] if train_matrix.size else 0,
                  test_matrix.shape[1] if test_matrix.size else 0)

    epochs = np.arange(1, max_len + 1)
    train_color = color_palette_list[0]
    test_color = color_palette_list[3]

    train_mean = np.full(max_len, np.nan, dtype=float)
    train_min = np.full(max_len, np.nan, dtype=float)
    train_max = np.full(max_len, np.nan, dtype=float)
    test_mean = np.full(max_len, np.nan, dtype=float)
    test_min = np.full(max_len, np.nan, dtype=float)
    test_max = np.full(max_len, np.nan, dtype=float)

    def _sanitize_for_log(values):
        return np.where(values > 0, values, np.nan)

    if train_matrix.size:
        train_mean[:train_matrix.shape[1]] = np.nanmean(train_matrix, axis=0)
        train_min[:train_matrix.shape[1]] = np.nanmin(train_matrix, axis=0)
        train_max[:train_matrix.shape[1]] = np.nanmax(train_matrix, axis=0)
        if plot_all_runs:
            for curve in train_matrix:
                display_curve = _sanitize_for_log(curve) if y_scale == 'log' else curve
                ax.plot(epochs[:train_matrix.shape[1]], display_curve, color=train_color, alpha=all_runs_alpha, linewidth=0.8)

    if test_matrix.size:
        test_mean[:test_matrix.shape[1]] = np.nanmean(test_matrix, axis=0)
        test_min[:test_matrix.shape[1]] = np.nanmin(test_matrix, axis=0)
        test_max[:test_matrix.shape[1]] = np.nanmax(test_matrix, axis=0)
        if plot_all_runs:
            for curve in test_matrix:
                display_curve = _sanitize_for_log(curve) if y_scale == 'log' else curve
                ax.plot(epochs[:test_matrix.shape[1]], display_curve, color=test_color, alpha=all_runs_alpha, linewidth=0.8)

    display_train_mean = _sanitize_for_log(train_mean) if y_scale == 'log' else train_mean
    display_train_min = _sanitize_for_log(train_min) if y_scale == 'log' else train_min
    display_train_max = _sanitize_for_log(train_max) if y_scale == 'log' else train_max
    display_test_mean = _sanitize_for_log(test_mean) if y_scale == 'log' else test_mean
    display_test_min = _sanitize_for_log(test_min) if y_scale == 'log' else test_min
    display_test_max = _sanitize_for_log(test_max) if y_scale == 'log' else test_max

    valid_train = np.isfinite(display_train_mean)
    valid_test = np.isfinite(display_test_mean)

    if np.any(valid_train):
        ax.fill_between(epochs[valid_train], display_train_min[valid_train], display_train_max[valid_train], color=train_color, alpha=0.15)
        ax.plot(epochs[valid_train], display_train_mean[valid_train], color=train_color, linewidth=2.2, label=f"{train_label} (min-mean-max)")

    if np.any(valid_test):
        ax.fill_between(epochs[valid_test], display_test_min[valid_test], display_test_max[valid_test], color=test_color, alpha=0.15)
        ax.plot(epochs[valid_test], display_test_mean[valid_test], color=test_color, linewidth=2.2, label=f"{test_label} (min-mean-max)")

    best_epoch = None
    best_idx = None
    per_run_best_epochs = []
    best_test_value = np.nan
    best_train_value = np.nan
    if np.any(valid_test):
        if higher_is_better:
            best_idx = int(np.nanargmax(test_mean))
        else:
            best_idx = int(np.nanargmin(test_mean))
        best_epoch = best_idx + 1
        best_test_value = _safe_value(test_mean, best_idx)
        best_train_value = _safe_value(train_mean, best_idx)
        ax.axvline(
            best_epoch,
            color='black',
            linestyle='--',
            linewidth=1.1,
            alpha=0.7,
            label=f"Best test epoch: {best_epoch}"
        )

        for run_curve in test_matrix:
            finite_mask = np.isfinite(run_curve)
            if not np.any(finite_mask):
                continue
            finite_indices = np.where(finite_mask)[0]
            finite_values = run_curve[finite_mask]
            if higher_is_better:
                local_best_pos = int(np.nanargmax(finite_values))
            else:
                local_best_pos = int(np.nanargmin(finite_values))
            per_run_best_epochs.append(int(finite_indices[local_best_pos]) + 1)

    if per_run_best_epochs:
        per_run_best_epochs_arr = np.asarray(per_run_best_epochs, dtype=float)
        best_epoch_run_median = float(np.nanmedian(per_run_best_epochs_arr))
        best_epoch_run_min = float(np.nanmin(per_run_best_epochs_arr))
        best_epoch_run_max = float(np.nanmax(per_run_best_epochs_arr))
        best_epoch_run_std = float(np.nanstd(per_run_best_epochs_arr))
        best_epoch_run_count = float(per_run_best_epochs_arr.size)
    else:
        best_epoch_run_median = np.nan
        best_epoch_run_min = np.nan
        best_epoch_run_max = np.nan
        best_epoch_run_std = np.nan
        best_epoch_run_count = 0.0

    last_train_idx = (
        int(np.where(np.isfinite(train_mean))[0][-1])
        if np.any(np.isfinite(train_mean)) else None
    )
    last_test_idx = (
        int(np.where(np.isfinite(test_mean))[0][-1])
        if np.any(np.isfinite(test_mean)) else None
    )

    train_final = _safe_value(train_mean, last_train_idx)
    test_final = _safe_value(test_mean, last_test_idx)

    if np.isfinite(train_final) and np.isfinite(test_final):
        if higher_is_better:
            final_gap = train_final - test_final
        else:
            final_gap = test_final - train_final
    else:
        final_gap = np.nan

    def _post_best_divergence(train_values, test_values, best_index):
        if best_index is None:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if best_index + 1 >= len(test_values):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        train_tail = train_values[best_index + 1:]
        test_tail = test_values[best_index + 1:]
        valid_tail = np.isfinite(test_tail)
        if np.sum(valid_tail) < 5:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        test_tail_valid = test_tail[valid_tail]
        train_tail_valid = train_tail[valid_tail] if train_tail.shape[0] == test_tail.shape[0] else train_tail[:test_tail.shape[0]][valid_tail]

        test_ref = _safe_value(test_values, best_index)
        train_ref = _safe_value(train_values, best_index)
        if not np.isfinite(test_ref):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if higher_is_better:
            test_shift = test_ref - float(np.nanmedian(test_tail_valid))
            test_degrade_fraction = float(np.mean(test_tail_valid <= (test_ref - 0.005)))
            test_sustained = test_shift >= 0.01 and test_degrade_fraction >= 0.6

            if np.isfinite(train_ref) and np.any(np.isfinite(train_tail_valid)):
                train_shift = float(np.nanmedian(train_tail_valid)) - train_ref
                train_improve_fraction = float(np.mean(train_tail_valid >= (train_ref + 0.002)))
                train_sustained = train_shift >= 0.005 and train_improve_fraction >= 0.6
            else:
                train_shift = np.nan
                train_improve_fraction = np.nan
                train_sustained = False
        else:
            test_shift = float(np.nanmedian(test_tail_valid)) - test_ref
            test_degrade_fraction = float(np.mean(test_tail_valid >= (test_ref + 0.02)))
            test_sustained = test_shift >= 0.03 and test_degrade_fraction >= 0.6

            if np.isfinite(train_ref) and np.any(np.isfinite(train_tail_valid)):
                train_shift = train_ref - float(np.nanmedian(train_tail_valid))
                train_improve_fraction = float(np.mean(train_tail_valid <= (train_ref - 0.005)))
                train_sustained = train_shift >= 0.01 and train_improve_fraction >= 0.6
            else:
                train_shift = np.nan
                train_improve_fraction = np.nan
                train_sustained = False

        sustained_divergence = float(test_sustained and train_sustained)
        n_tail_points = float(np.sum(valid_tail))
        return test_shift, train_shift, test_degrade_fraction, train_improve_fraction, sustained_divergence, n_tail_points

    (
        post_best_test_shift,
        post_best_train_shift,
        post_best_test_degrade_fraction,
        post_best_train_improve_fraction,
        sustained_divergence,
        post_best_points,
    ) = _post_best_divergence(train_mean, test_mean, best_idx)

    y_axis_label = f"{metric_label} (log scale)" if y_scale == 'log' else metric_label
    if show_xlabel:
        ax.set_xlabel("Epoch")
    ax.set_ylabel(y_axis_label)
    if y_scale == 'log':
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    return {
        'best_epoch': best_epoch,
        'best_epoch_run_median': best_epoch_run_median,
        'best_epoch_run_min': best_epoch_run_min,
        'best_epoch_run_max': best_epoch_run_max,
        'best_epoch_run_std': best_epoch_run_std,
        'best_epoch_run_count': best_epoch_run_count,
        'best_train': best_train_value,
        'best_test': best_test_value,
        'final_train': train_final,
        'final_test': test_final,
        'final_gap': final_gap,
        'post_best_test_shift': post_best_test_shift,
        'post_best_train_shift': post_best_train_shift,
        'post_best_test_degrade_fraction': post_best_test_degrade_fraction,
        'post_best_train_improve_fraction': post_best_train_improve_fraction,
        'sustained_divergence': sustained_divergence,
        'post_best_points': post_best_points,
    }


def analyzeAndPlotOverfitCurves(metrics_data,
                                analysis_path,
                                model_name,
                                plot_all_runs=True,
                                all_runs_alpha=0.12,
                                loss_log_scale=True,
                                train_loss_metric_key='train_loss_plot',
                                test_loss_metric_key='test_loss_plot',
                                loss_label='Loss',
                                quality_metric_key='accuracy_plot',
                                train_quality_metric_key='train_accuracy_plot',
                                quality_label='Accuracy',
                                lr_metric_key=None,
                                train_component_metric_keys=None,
                                test_component_metric_keys=None):
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)

    train_loss_curves = _extract_curves(metrics_data, train_loss_metric_key)
    test_loss_curves = _extract_curves(metrics_data, test_loss_metric_key)
    train_quality_curves = _extract_curves(metrics_data, train_quality_metric_key)
    test_quality_curves = _extract_curves(metrics_data, quality_metric_key)

    train_loss_matrix = _curves_to_matrix(train_loss_curves)
    test_loss_matrix = _curves_to_matrix(test_loss_curves)
    train_quality_matrix = _curves_to_matrix(train_quality_curves)
    test_quality_matrix = _curves_to_matrix(test_quality_curves)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)

    loss_summary = _plot_metric_evolution(
        ax=axes[0],
        train_matrix=train_loss_matrix,
        test_matrix=test_loss_matrix,
        metric_label=loss_label,
        train_label=f'Train {loss_label.lower()}',
        test_label=f'Validation {loss_label.lower()}',
        higher_is_better=False,
        plot_all_runs=plot_all_runs,
        all_runs_alpha=all_runs_alpha,
        show_xlabel=False,
        y_scale='log' if loss_log_scale else 'linear',
    )

    quality_summary = _plot_metric_evolution(
        ax=axes[1],
        train_matrix=train_quality_matrix,
        test_matrix=test_quality_matrix,
        metric_label=quality_label,
        train_label=f'Train {quality_label.lower()}',
        test_label=f'Validation {quality_label.lower()}',
        higher_is_better=True,
        plot_all_runs=plot_all_runs,
        all_runs_alpha=all_runs_alpha,
        show_xlabel=True,
    )

    n_samples = len(metrics_data)
    axes[0].set_title(f"Training Process - {model_name} - {n_samples} Samples")
    # fig.subplots_adjust(hspace=0.08)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    plot_filename = os.path.join(analysis_path, f"overfit_evolution_{model_name}.pdf")
    fig.savefig(plot_filename, format='pdf', bbox_inches='tight')
    plt.close(fig)

    summary_table = [[
        'Metric',
        'Best epoch (mean test)',
        'Train@best',
        'Test@best',
        'Train@final',
        'Test@final',
        'Gap@final',
        'Interpretation'
    ]]

    def _fmt(value, digits=4):
        if value is None or not np.isfinite(value):
            return '-'
        return f"{value:.{digits}f}"

    def _curve_best_final_delta(train_key, test_key, best_epoch):
        if best_epoch is None:
            return np.nan, np.nan, False

        train_curves = _extract_curves(metrics_data, train_key)
        test_curves = _extract_curves(metrics_data, test_key)
        train_matrix = _curves_to_matrix(train_curves)
        test_matrix = _curves_to_matrix(test_curves)
        if train_matrix.size == 0 or test_matrix.size == 0:
            return np.nan, np.nan, False

        train_mean = np.nanmean(train_matrix, axis=0)
        test_mean = np.nanmean(test_matrix, axis=0)
        best_idx = min(max(0, best_epoch - 1), train_mean.shape[0] - 1, test_mean.shape[0] - 1)

        train_best = _safe_value(train_mean, best_idx)
        test_best = _safe_value(test_mean, best_idx)

        train_last_idx = int(np.where(np.isfinite(train_mean))[0][-1]) if np.any(np.isfinite(train_mean)) else None
        test_last_idx = int(np.where(np.isfinite(test_mean))[0][-1]) if np.any(np.isfinite(test_mean)) else None
        train_final = _safe_value(train_mean, train_last_idx)
        test_final = _safe_value(test_mean, test_last_idx)

        if not np.isfinite(train_best) or not np.isfinite(test_best) or not np.isfinite(train_final) or not np.isfinite(test_final):
            return np.nan, np.nan, False

        train_drop = max(0.0, train_best - train_final)
        test_rise = max(0.0, test_final - test_best)
        return train_drop, test_rise, True

    def _lr_signals(best_epoch):
        if lr_metric_key is None or best_epoch is None:
            return np.nan, np.nan, np.nan

        lr_curves = _extract_curves(metrics_data, lr_metric_key)
        lr_matrix = _curves_to_matrix(lr_curves)
        if lr_matrix.size == 0:
            return np.nan, np.nan, np.nan

        lr_mean = np.nanmean(lr_matrix, axis=0)
        best_idx = min(max(0, best_epoch - 1), lr_mean.shape[0] - 1)
        lr_best = _safe_value(lr_mean, best_idx)
        lr_last_idx = int(np.where(np.isfinite(lr_mean))[0][-1]) if np.any(np.isfinite(lr_mean)) else None
        lr_final = _safe_value(lr_mean, lr_last_idx)

        if np.isfinite(lr_best) and np.isfinite(lr_final) and lr_best > 0:
            return lr_best, lr_final, lr_final / lr_best
        return lr_best, lr_final, np.nan

    def _build_interpretation(quality_summary, loss_summary):
        if quality_summary is None or loss_summary is None:
            return '-', np.nan, np.nan, {}

        quality_gap = quality_summary['final_gap']  # train - validation
        loss_gap = loss_summary['final_gap']  # test - train
        quality_sustained_divergence = quality_summary.get('sustained_divergence', np.nan)
        loss_sustained_divergence = loss_summary.get('sustained_divergence', np.nan)
        test_quality_drop = max(0.0, quality_summary['best_test'] - quality_summary['final_test'])
        test_loss_rise = max(0.0, loss_summary['final_test'] - loss_summary['best_test'])
        train_quality_gain_post_best = max(0.0, quality_summary['final_train'] - quality_summary['best_train'])
        train_loss_drop_post_best = max(0.0, loss_summary['best_train'] - loss_summary['final_train'])

        def _isfinite(value):
            return np.isfinite(value)

        has_quality_gap = _isfinite(quality_gap)
        best_epoch_ref = quality_summary['best_epoch'] if quality_summary['best_epoch'] is not None else loss_summary['best_epoch']

        component_pairs = list(zip(train_component_metric_keys or [], test_component_metric_keys or []))
        component_available = 0
        component_divergence_count = 0
        component_train_drop_mean = []
        component_test_rise_mean = []
        for train_key, test_key in component_pairs:
            train_drop, test_rise, is_available = _curve_best_final_delta(train_key, test_key, best_epoch_ref)
            if not is_available:
                continue
            component_available += 1
            component_train_drop_mean.append(train_drop)
            component_test_rise_mean.append(test_rise)
            if train_drop >= 0.01 and test_rise >= 0.02:
                component_divergence_count += 1

        mean_component_train_drop = np.nanmean(component_train_drop_mean) if component_train_drop_mean else np.nan
        mean_component_test_rise = np.nanmean(component_test_rise_mean) if component_test_rise_mean else np.nan

        lr_best, lr_final, lr_final_to_best_ratio = _lr_signals(best_epoch_ref)

        overfit_score = 0.0
        underfit_score = 0.0

        # Overfit evidence: train keeps improving while test degrades + widening gap.
        if has_quality_gap and quality_gap >= 0.03:
            overfit_score += 2.0
        # Absolute train-vs-validation loss gap is less reliable when train quality is unavailable
        # (common in YOLO results.csv), so only use it when quality gap is also available.
        if has_quality_gap and _isfinite(loss_gap) and loss_gap >= 0.05:
            overfit_score += 2.0
        if _isfinite(test_quality_drop) and test_quality_drop >= 0.01:
            overfit_score += 1.5
        if _isfinite(test_loss_rise) and test_loss_rise >= 0.03:
            overfit_score += 1.5
        if _isfinite(train_quality_gain_post_best) and _isfinite(test_quality_drop) and train_quality_gain_post_best >= 0.005 and test_quality_drop >= 0.005:
            overfit_score += 1.0
        if _isfinite(train_loss_drop_post_best) and _isfinite(test_loss_rise) and train_loss_drop_post_best >= 0.01 and test_loss_rise >= 0.02:
            overfit_score += 1.0
        if _isfinite(quality_sustained_divergence) and quality_sustained_divergence >= 1.0:
            overfit_score += 2.0
        if _isfinite(loss_sustained_divergence) and loss_sustained_divergence >= 1.0 and (has_quality_gap or component_available >= 2):
            overfit_score += 1.5
        if component_available >= 2 and component_divergence_count >= 2:
            overfit_score += 1.5
        if _isfinite(lr_final_to_best_ratio) and lr_final_to_best_ratio <= 0.2 and _isfinite(test_quality_drop) and test_quality_drop >= 0.01:
            overfit_score += 0.5

        # Underfit evidence: both train and test stay weak, with little separation.
        best_train_quality = quality_summary['best_train']
        best_test_quality = quality_summary['best_test']
        best_train_loss = loss_summary['best_train']
        best_test_loss = loss_summary['best_test']

        low_train_and_test_quality = _isfinite(best_train_quality) and _isfinite(best_test_quality) and (best_train_quality < 0.85 and best_test_quality < 0.85)
        low_test_quality_only = _isfinite(best_test_quality) and (best_test_quality < 0.6)
        small_final_gap = _isfinite(quality_gap) and _isfinite(loss_gap) and (quality_gap < 0.02 and loss_gap < 0.05)
        little_curve_divergence = _isfinite(test_quality_drop) and _isfinite(test_loss_rise) and (test_quality_drop < 0.005 and test_loss_rise < 0.02)
        persistently_high_loss = _isfinite(best_train_loss) and _isfinite(best_test_loss) and (best_train_loss > 0.4 and best_test_loss > 0.4)

        if low_train_and_test_quality:
            underfit_score += 2.0
        if low_test_quality_only:
            underfit_score += 1.5
        if small_final_gap:
            underfit_score += 1.5
        if little_curve_divergence:
            underfit_score += 1.0
        if persistently_high_loss:
            underfit_score += 1.5
        if component_available >= 2 and component_divergence_count == 0:
            underfit_score += 0.5
        if _isfinite(lr_final_to_best_ratio) and lr_final_to_best_ratio >= 0.4 and _isfinite(test_quality_drop) and test_quality_drop < 0.01:
            underfit_score += 0.5
        if _isfinite(quality_sustained_divergence) and quality_sustained_divergence == 0:
            underfit_score += 0.5
        if _isfinite(loss_sustained_divergence) and loss_sustained_divergence == 0:
            underfit_score += 0.5

        strong_overfit_pattern = (
            (_isfinite(quality_sustained_divergence) and quality_sustained_divergence >= 1.0) or
            (_isfinite(loss_sustained_divergence) and loss_sustained_divergence >= 1.0 and component_divergence_count >= 2) or
            (_isfinite(test_quality_drop) and test_quality_drop >= 0.02 and _isfinite(test_loss_rise) and test_loss_rise >= 0.04)
        )

        signals = {
            'quality_gap': quality_gap,
            'quality_gap_available': 1.0 if has_quality_gap else 0.0,
            'loss_gap': loss_gap,
            'quality_sustained_divergence': quality_sustained_divergence,
            'loss_sustained_divergence': loss_sustained_divergence,
            'quality_post_best_test_shift': quality_summary.get('post_best_test_shift', np.nan),
            'quality_post_best_train_shift': quality_summary.get('post_best_train_shift', np.nan),
            'quality_post_best_test_degrade_fraction': quality_summary.get('post_best_test_degrade_fraction', np.nan),
            'quality_post_best_train_improve_fraction': quality_summary.get('post_best_train_improve_fraction', np.nan),
            'quality_post_best_points': quality_summary.get('post_best_points', np.nan),
            'loss_post_best_test_shift': loss_summary.get('post_best_test_shift', np.nan),
            'loss_post_best_train_shift': loss_summary.get('post_best_train_shift', np.nan),
            'loss_post_best_test_degrade_fraction': loss_summary.get('post_best_test_degrade_fraction', np.nan),
            'loss_post_best_train_improve_fraction': loss_summary.get('post_best_train_improve_fraction', np.nan),
            'loss_post_best_points': loss_summary.get('post_best_points', np.nan),
            'quality_best_epoch_run_median': quality_summary.get('best_epoch_run_median', np.nan),
            'quality_best_epoch_run_min': quality_summary.get('best_epoch_run_min', np.nan),
            'quality_best_epoch_run_max': quality_summary.get('best_epoch_run_max', np.nan),
            'quality_best_epoch_run_std': quality_summary.get('best_epoch_run_std', np.nan),
            'quality_best_epoch_run_count': quality_summary.get('best_epoch_run_count', np.nan),
            'loss_best_epoch_run_median': loss_summary.get('best_epoch_run_median', np.nan),
            'loss_best_epoch_run_min': loss_summary.get('best_epoch_run_min', np.nan),
            'loss_best_epoch_run_max': loss_summary.get('best_epoch_run_max', np.nan),
            'loss_best_epoch_run_std': loss_summary.get('best_epoch_run_std', np.nan),
            'loss_best_epoch_run_count': loss_summary.get('best_epoch_run_count', np.nan),
            'test_quality_drop': test_quality_drop,
            'test_loss_rise': test_loss_rise,
            'train_quality_gain_post_best': train_quality_gain_post_best,
            'train_loss_drop_post_best': train_loss_drop_post_best,
            'component_available': float(component_available),
            'component_divergence_count': float(component_divergence_count),
            'component_train_drop_mean': mean_component_train_drop,
            'component_test_rise_mean': mean_component_test_rise,
            'lr_best': lr_best,
            'lr_final': lr_final,
            'lr_final_to_best_ratio': lr_final_to_best_ratio,
            'best_train_quality': best_train_quality,
            'best_test_quality': best_test_quality,
            'best_train_loss': best_train_loss,
            'best_test_loss': best_test_loss,
        }

        if overfit_score >= 3.0 and overfit_score - underfit_score >= 1.0 and strong_overfit_pattern:
            return 'overfit', overfit_score, underfit_score, signals
        if underfit_score >= 3.0 and underfit_score - overfit_score >= 1.0:
            return 'underfit', overfit_score, underfit_score, signals
        return 'mixed/balanced', overfit_score, underfit_score, signals

    interpretation, overfit_score, underfit_score, signals = _build_interpretation(quality_summary, loss_summary)
    interpretation_tag = interpretation if not np.isfinite(overfit_score) else f"{interpretation} (O={overfit_score:.1f}, U={underfit_score:.1f})"

    if loss_summary is not None:
        summary_table.append([
            loss_label,
            loss_summary['best_epoch'] if loss_summary['best_epoch'] is not None else '-',
            _fmt(loss_summary['best_train']),
            _fmt(loss_summary['best_test']),
            _fmt(loss_summary['final_train']),
            _fmt(loss_summary['final_test']),
            _fmt(loss_summary['final_gap']),
            interpretation_tag,
        ])

    if quality_summary is not None:
        summary_table.append([
            quality_label,
            quality_summary['best_epoch'] if quality_summary['best_epoch'] is not None else '-',
            _fmt(quality_summary['best_train']),
            _fmt(quality_summary['best_test']),
            _fmt(quality_summary['final_train']),
            _fmt(quality_summary['final_test']),
            _fmt(quality_summary['final_gap']),
            interpretation_tag,
        ])

    logTable(
        row_data=summary_table,
        output_path=f"{analysis_path}/tables",
        filename=f"Overfit summary - {model_name}",
        colalign=['left', 'right', 'right', 'right', 'right', 'right', 'right', 'left']
    )

    diagnostics_table = [[
        'Signal',
        'Value'
    ]]
    for signal_name in [
        'quality_gap',
        'quality_gap_available',
        'loss_gap',
        'quality_sustained_divergence',
        'loss_sustained_divergence',
        'quality_post_best_test_shift',
        'quality_post_best_train_shift',
        'quality_post_best_test_degrade_fraction',
        'quality_post_best_train_improve_fraction',
        'quality_post_best_points',
        'quality_best_epoch_run_median',
        'quality_best_epoch_run_min',
        'quality_best_epoch_run_max',
        'quality_best_epoch_run_std',
        'quality_best_epoch_run_count',
        'loss_post_best_test_shift',
        'loss_post_best_train_shift',
        'loss_post_best_test_degrade_fraction',
        'loss_post_best_train_improve_fraction',
        'loss_post_best_points',
        'loss_best_epoch_run_median',
        'loss_best_epoch_run_min',
        'loss_best_epoch_run_max',
        'loss_best_epoch_run_std',
        'loss_best_epoch_run_count',
        'test_quality_drop',
        'test_loss_rise',
        'train_quality_gain_post_best',
        'train_loss_drop_post_best',
        'component_available',
        'component_divergence_count',
        'component_train_drop_mean',
        'component_test_rise_mean',
        'lr_best',
        'lr_final',
        'lr_final_to_best_ratio',
        'best_train_quality',
        'best_test_quality',
        'best_train_loss',
        'best_test_loss',
    ]:
        diagnostics_table.append([signal_name, _fmt(signals.get(signal_name, np.nan), digits=6)])
    diagnostics_table.append(['overfit_score', _fmt(overfit_score, digits=3)])
    diagnostics_table.append(['underfit_score', _fmt(underfit_score, digits=3)])
    diagnostics_table.append(['final_label', interpretation])

    logTable(
        row_data=diagnostics_table,
        output_path=f"{analysis_path}/tables",
        filename=f"Overfit diagnostics - {model_name}",
        colalign=['left', 'right']
    )

    return {
        'plot_path': plot_filename,
        'loss_summary': loss_summary,
        'quality_summary': quality_summary,
        'accuracy_summary': quality_summary,
        'interpretation': interpretation,
        'overfit_score': float(overfit_score) if np.isfinite(overfit_score) else np.nan,
        'underfit_score': float(underfit_score) if np.isfinite(underfit_score) else np.nan,
        'n_runs_loss': int(train_loss_matrix.shape[0] if train_loss_matrix.size else 0),
        'n_runs_accuracy': int(train_quality_matrix.shape[0] if train_quality_matrix.size else 0),
    }
