#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

from utils.log_utils import log, logTable, c_grey, color_palette_list
from utils.compute_switched_probability import bootstrap_samples, montecarlo_samples
from utils.sampling_param_compute import (
    compute_sampling_error_curve,
    compute_at_least_one_exceedance_probability_curve,
    get_sampling_method_label,
)


def _resolve_simulation_iterations(sampling_method, n_iterations):
    if n_iterations is not None:
        return n_iterations
    if sampling_method == 'bootstrap':
        return bootstrap_samples
    if sampling_method == 'montecarlo':
        return montecarlo_samples
    return 0


def plot_sampling_graph(metrics_data,
                        analysis_path,
                        metric='accuracy',
                        title_tag='',
                        plot_models=None,
                        color_list=color_palette_list,
                        sampling_method='analytical',
                        n_iterations=None,
                        curve_method=compute_sampling_error_curve,
                        curve_kwargs=None,
                        title_template='Sampling curve ({method_label})',
                        y_label='Value',
                        filename_prefix='sampling_curve',
                        table_title_prefix='Sampling Curve',
                        close_plot=True,
                        plot_derivative=False):
    sample_sizes = np.arange(1, 16)
    plot_models = plot_models or []
    curve_kwargs = curve_kwargs or {}

    metric_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        metric_data[model] = [entry[metric] * 100 for entry in metrics_data[model].values()]

    if plot_derivative:
        # increase height so stacked subplots remain readable
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 16), gridspec_kw={'height_ratios': [2.8, 1.2]})
        ax2.set_ylabel('Derivative')
    else:
        fig, ax = plt.subplots(figsize=(14, 13))
        ax2 = None

    color_iterator = itertools.cycle(color_list)
    method_label = get_sampling_method_label(sampling_method)
    simulation_iterations = _resolve_simulation_iterations(sampling_method, n_iterations)
    display_method_label = method_label if sampling_method == 'analytical' else f"{method_label} - {simulation_iterations} samples"
    
    idx_n1 = 0
    idx_n5 = 4
    idx_n10 = 9
    idx_q25 = int(len(sample_sizes) * 0.25)
    idx_q50 = int(len(sample_sizes) * 0.5)
    idx_nlast = -1

    table_checkpoints = [
        (f'N={sample_sizes[idx_n1]}', idx_n1, sample_sizes[idx_n1]),
        (f'N={sample_sizes[idx_n5]}', idx_n5, sample_sizes[idx_n5]),
        (f'N={sample_sizes[idx_n10]}', idx_n10, sample_sizes[idx_n10]),
        (f'N≈25% (N={sample_sizes[idx_q25]})', idx_q25, sample_sizes[idx_q25]),
        (f'N≈50% (N={sample_sizes[idx_q50]})', idx_q50, sample_sizes[idx_q50]),
        (f'N={sample_sizes[idx_nlast]}', idx_nlast, sample_sizes[idx_nlast]),
    ]
    table_checkpoints = sorted(table_checkpoints, key=lambda item: item[2])

    row_data = [['Model'] + [label for label, _, _ in table_checkpoints]]
    curve_values_all = []
    for model_name, data in metric_data.items():
        if plot_models and model_name not in plot_models:
            # log(f"[plot_sampling_graph] Skipping model {model_name} as it is not in plot_models {plot_models}.")
            continue

        curve_values = curve_method(
            data=data,
            sample_sizes=sample_sizes,
            method=sampling_method,
            n_iterations=simulation_iterations,
            **curve_kwargs,
        )
        curve_values_all.append(curve_values)
        curve_values = np.asarray(curve_values, dtype=float)
        valid_points = np.isfinite(curve_values)
        primary_color = next(color_iterator)
        ax.plot(sample_sizes[valid_points], curve_values[valid_points], linestyle='--', alpha=0.5, label=f'{model_name}', color=primary_color, linewidth=2)

        if plot_derivative and np.sum(valid_points) >= 2 and ax2 is not None:
            deriv_vals = np.gradient(curve_values[valid_points], sample_sizes[valid_points])
            # plot derivative without legend label so the global legend shows only primary curves
            ax2.plot(sample_sizes[valid_points], deriv_vals, linestyle='--', color=primary_color, linewidth=1.5, alpha=0.9)

        def _fmt_value(value):
            return '-' if not np.isfinite(value) else f"{value:.3f}"

        row_data.append([model_name] + [_fmt_value(curve_values[idx]) for _, idx, _ in table_checkpoints])

    avg_curve = np.mean(curve_values_all, axis=0)
    valid_points = np.isfinite(avg_curve)
    ax.plot(sample_sizes[valid_points], avg_curve[valid_points], label=f'Average', color='black', linewidth=3)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Combine legends from both axes when derivative subplot exists and place a single legend below the figure
    handles, labels = ax.get_legend_handles_labels()

    # Just orders so that long ones appear in the same column, if not both columns might be long and cover the figure
    if labels:
        max_short_len = 20
        paired = list(zip(handles, labels))
        short_pairs = [p for p in paired if len(p[1]) <= max_short_len]
        long_pairs = [p for p in paired if len(p[1]) > max_short_len]

        ordered = short_pairs + long_pairs

        if ordered:
            h_all, l_all = zip(*ordered)
            # ax.legend(list(h_all), list(l_all), loc='best', ncol=2, fontsize='small', frameon=True, framealpha=0.9, edgecolor='0.2')
            if plot_derivative:
                box_anchor = (0.5, -0.23)
            else:
                box_anchor = (0.5, -0.28)
            fig.legend(list(h_all), list(l_all), loc='lower center', ncol=3, fontsize='small', frameon=True, framealpha=0.9, edgecolor='0.2',
                      bbox_to_anchor=box_anchor)

    ax.set_title(title_template.format(method_label=display_method_label, **curve_kwargs))
    ax.set_ylabel(y_label)
    # Put X label on the bottom subplot when derivative is plotted so it appears under both plots
    if plot_derivative and ax2 is not None:
        ax2.set_xlabel('Sample Size (N)')
    else:
        ax.set_xlabel('Sample Size (N)')

    # Set X ticks to include first and last and a few intermediate ticks
    try:
        max_ticks = 8
        n = len(sample_sizes)
        tick_count = min(max_ticks, n)
        indices = np.unique(np.round(np.linspace(0, n - 1, tick_count)).astype(int))
        xticks = sample_sizes[indices]
        ax.set_xticks(xticks)
        if plot_derivative and ax2 is not None:
            ax2.set_xticks(xticks)
    except Exception:
        pass

    # Align Y labels of stacked subplots for visual consistency
    try:
        fig.align_ylabels([ax, ax2] if ax2 is not None else [ax])
    except Exception:
        # fallback: set same label coordinates
        if ax2 is not None:
            ax.yaxis.set_label_coords(-0.05, 0.5)
            ax2.yaxis.set_label_coords(-0.05, 0.5)

    # Apply grid to both axes (main and derivative) when present
    ax.grid(visible=True, color=c_grey, linestyle='--', linewidth=0.5, alpha=0.7)
    if plot_derivative and ax2 is not None:
        ax2.grid(visible=True, color=c_grey, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    extra_title = f"_{title_tag}" if title_tag else ""
    method_suffix = sampling_method.lower()
    filename = f"{filename_prefix}{extra_title}.pdf" if method_suffix == 'analytical' else f"{filename_prefix}_{method_suffix}{extra_title}.pdf"
    plt.savefig(os.path.join(analysis_path, filename), format="pdf", bbox_inches='tight')

    table_title = f"{metric.title()} {table_title_prefix} ({display_method_label}) {title_tag.replace('_', ' ').title()}"
    log(f"\nSummary Table of {table_title}:")
    colalign = ['left'] + ['right'] * (len(row_data[0]) - 1)
    logTable(row_data=row_data, output_path=f"{analysis_path}/tables", filename=table_title, colalign=colalign)

    if close_plot:
        plt.close()


def plot_sampling_graph_average(metrics_data,
                        analysis_path,
                        metric='accuracy',
                        title_tag='',
                        plot_models=None,
                        color_list=color_palette_list,
                        sampling_method='analytical',
                        n_iterations=None,
                        curve_method=compute_sampling_error_curve,
                        percentile_range=None,
                        title_template='Sampling curve ({method_label})',
                        y_label='Value',
                        filename_prefix='sampling_curve',
                        table_title_prefix='Sampling Curve',
                        close_plot=True,
                        plot_derivative=False):
    sample_sizes = np.arange(1, 16)
    plot_models = plot_models or []
    percentile_range = percentile_range or [90, 100]

    metric_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        metric_data[model] = [entry[metric] * 100 for entry in metrics_data[model].values()]

    if plot_derivative:
        # increase height so stacked subplots remain readable
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 16), gridspec_kw={'height_ratios': [2.8, 1.2]})
        ax2.set_ylabel('Derivative')
    else:
        fig, ax = plt.subplots(figsize=(14, 11))
        ax2 = None

    color_iterator = itertools.cycle(color_list)
    method_label = get_sampling_method_label(sampling_method)
    simulation_iterations = _resolve_simulation_iterations(sampling_method, n_iterations)
    display_method_label = method_label if sampling_method == 'analytical' else f"{method_label} - {simulation_iterations} samples"
    
    idx_n1 = 0
    idx_n5 = 4
    idx_n10 = 9
    idx_q25 = int(len(sample_sizes) * 0.25)
    idx_q50 = int(len(sample_sizes) * 0.5)
    idx_nlast = -1

    table_checkpoints = [
        (f'N={sample_sizes[idx_n1]}', idx_n1, sample_sizes[idx_n1]),
        (f'N={sample_sizes[idx_n5]}', idx_n5, sample_sizes[idx_n5]),
        (f'N={sample_sizes[idx_n10]}', idx_n10, sample_sizes[idx_n10]),
        (f'N≈25% (N={sample_sizes[idx_q25]})', idx_q25, sample_sizes[idx_q25]),
        (f'N≈50% (N={sample_sizes[idx_q50]})', idx_q50, sample_sizes[idx_q50]),
        (f'N={sample_sizes[idx_nlast]}', idx_nlast, sample_sizes[idx_nlast]),
    ]
    table_checkpoints = sorted(table_checkpoints, key=lambda item: item[2])

    row_data = [['Model'] + [label for label, _, _ in table_checkpoints]]
    for percentile in range(percentile_range[0], percentile_range[1] + 1):
        curve_values = []
        for model_name, data in metric_data.items():
            if plot_models and model_name not in plot_models:
                # log(f"[plot_sampling_graph] Skipping model {model_name} as it is not in plot_models {plot_models}.")
                continue

            curve_values_i = curve_method(
                data=data,
                sample_sizes=sample_sizes,
                method=sampling_method,
                n_iterations=simulation_iterations,
                **{'percentile': percentile}
            )
            curve_values.append(curve_values_i)
        
        curve_values = np.array(curve_values)
        curve_values = np.mean(curve_values, axis=0)
        valid_points = np.isfinite(curve_values)

        primary_color = next(color_iterator)
        ax.plot(sample_sizes[valid_points], curve_values[valid_points], label=f'Avg p{percentile}', color=primary_color, linewidth=2)

        if plot_derivative and np.sum(valid_points) >= 2 and ax2 is not None:
            deriv_vals = np.gradient(curve_values[valid_points], sample_sizes[valid_points])
            # plot derivative without legend label so the global legend shows only primary curves
            ax2.plot(sample_sizes[valid_points], deriv_vals, linestyle='--', color=primary_color, linewidth=1.5, alpha=0.9)

        def _fmt_value(value):
            return '-' if not np.isfinite(value) else f"{value:.3f}"

        row_data.append([model_name] + [_fmt_value(curve_values[idx]) for _, idx, _ in table_checkpoints])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Combine legends from both axes when derivative subplot exists and place a single legend below the figure
    handles, labels = ax.get_legend_handles_labels()

    # Just orders so that long ones appear in the same column, if not both columns might be long and cover the figure
    if labels:
        max_short_len = 20
        paired = list(zip(handles, labels))
        short_pairs = [p for p in paired if len(p[1]) <= max_short_len]
        long_pairs = [p for p in paired if len(p[1]) > max_short_len]

        ordered = short_pairs + long_pairs

        if ordered:
            h_all, l_all = zip(*ordered)
            # ax.legend(list(h_all), list(l_all), loc='best', ncol=2, fontsize='small', frameon=True, framealpha=0.9, edgecolor='0.2')
            fig.legend(list(h_all), list(l_all), loc='center left', ncol=1, fontsize='small', frameon=True, framealpha=0.9, edgecolor='0.2',
                      bbox_to_anchor=(1.0, 0.5))

    ax.set_title(title_template.format(method_label=display_method_label, percentile=f'p∈[percentile_range]'))
    ax.set_ylabel(y_label)
    # Put X label on the bottom subplot when derivative is plotted so it appears under both plots
    if plot_derivative and ax2 is not None:
        ax2.set_xlabel('Sample Size (N)')
    else:
        ax.set_xlabel('Sample Size (N)')

    # Set X ticks to include first and last and a few intermediate ticks
    try:
        max_ticks = 8
        n = len(sample_sizes)
        tick_count = min(max_ticks, n)
        indices = np.unique(np.round(np.linspace(0, n - 1, tick_count)).astype(int))
        xticks = sample_sizes[indices]
        ax.set_xticks(xticks)
        if plot_derivative and ax2 is not None:
            ax2.set_xticks(xticks)
    except Exception:
        pass

    # Align Y labels of stacked subplots for visual consistency
    try:
        fig.align_ylabels([ax, ax2] if ax2 is not None else [ax])
    except Exception:
        # fallback: set same label coordinates
        if ax2 is not None:
            ax.yaxis.set_label_coords(-0.05, 0.5)
            ax2.yaxis.set_label_coords(-0.05, 0.5)

    # Apply grid to both axes (main and derivative) when present
    ax.grid(visible=True, color=c_grey, linestyle='--', linewidth=0.5, alpha=0.7)
    if plot_derivative and ax2 is not None:
        ax2.grid(visible=True, color=c_grey, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    extra_title = f"_{title_tag}" if title_tag else ""
    method_suffix = sampling_method.lower()
    filename = f"{filename_prefix}{extra_title}.pdf" if method_suffix == 'analytical' else f"{filename_prefix}_{method_suffix}{extra_title}.pdf"
    plt.savefig(os.path.join(analysis_path, filename), format="pdf", bbox_inches='tight')

    table_title = f"{metric.title()} {table_title_prefix} ({display_method_label}) {title_tag.replace('_', ' ').title()}"
    log(f"\nSummary Table of {table_title}:")
    colalign = ['left'] + ['right'] * (len(row_data[0]) - 1)
    logTable(row_data=row_data, output_path=f"{analysis_path}/tables", filename=table_title, colalign=colalign)

    if close_plot:
        plt.close()



def plot_all_sampling_errors(metrics_data,
                             analysis_path,
                             metric='accuracy',
                             title_tag='',
                             plot_models=None,
                             color_list=color_palette_list,
                             n_iterations=None,
                             close_plot=True):
    for sampling_method in ['analytical', 'bootstrap', 'montecarlo']:
        for parameter in ['mean', 'std']:
            plot_sampling_graph(
                metrics_data=metrics_data,
                analysis_path=analysis_path,
                metric=metric,
                title_tag=title_tag,
                plot_models=plot_models,
                sampling_method=sampling_method,
                color_list=color_list,
                n_iterations=n_iterations,
                curve_method=compute_sampling_error_curve,
                curve_kwargs={'parameter': parameter},
                title_template='Sampling error ({method_label})',
                y_label=f'SamplingError ({parameter} {metric})',
                filename_prefix=f'sampling_error_{parameter}',
                table_title_prefix=f'Sampling Errors ({parameter})',
                close_plot=close_plot,
                plot_derivative=True
            )


def plot_all_percentile_probabilities(metrics_data,
                                      analysis_path,
                                      percentile=95,
                                      metric='accuracy',
                                      title_tag='',
                                      plot_models=None,
                                      color_list=color_palette_list,
                                      n_iterations=None,
                                      close_plot=True):
    extra_title = f"_{title_tag}" if title_tag else ""
    for sampling_method in ['analytical', 'bootstrap', 'montecarlo']:
        plot_sampling_graph(
            metrics_data=metrics_data,
            analysis_path=analysis_path,
            metric=metric,
            title_tag='',
            plot_models=plot_models,
            sampling_method=sampling_method,
            color_list=color_list,
            n_iterations=n_iterations,
            curve_method=compute_at_least_one_exceedance_probability_curve,
            curve_kwargs={'percentile': percentile},
            title_template='P(at least 1 > p{percentile}) ({method_label})',
            y_label='Probability',
            filename_prefix=f'sampling_probability{extra_title}_p{int(percentile)}',
            table_title_prefix=f'Sampling Probability p{percentile}',
            close_plot=close_plot,
        )


def plot_all_percentile_probabilities_average(metrics_data,
                                              analysis_path,
                                              percentile_range=[90, 100],
                                              metric='accuracy',
                                              title_tag='',
                                              plot_models=None,
                                              color_list=color_palette_list,
                                              n_iterations=None,
                                              close_plot=True):
    # Custom color degradation based on the number of percentiles in the range, 
    # using the base colors (first 5 colors from color_palette_list)
    num_colors = percentile_range[1] - percentile_range[0] + 1
    base_colors = color_palette_list[0:7]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", base_colors)
    gradient_colors = [cmap(i/(num_colors-1)) for i in range(num_colors)]

    extra_title = f"_{title_tag}" if title_tag else ""
    for sampling_method in ['analytical', 'bootstrap', 'montecarlo']:
        plot_sampling_graph_average(
            metrics_data=metrics_data,
            analysis_path=analysis_path,
            metric=metric,
            title_tag='',
            plot_models=plot_models,
            sampling_method=sampling_method,
            color_list=gradient_colors,
            n_iterations=n_iterations,
            curve_method=compute_at_least_one_exceedance_probability_curve,
            percentile_range=percentile_range,
            title_template='P(at least 1 > percentile) ({method_label})',
            y_label='Probability',
            filename_prefix=f'sampling_probability{extra_title}_p{str(percentile_range)}',
            table_title_prefix=f'Sampling Probability p{str(percentile_range)}',
            close_plot=close_plot,
        )