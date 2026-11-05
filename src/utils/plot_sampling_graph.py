#!/usr/bin/env python3
# encoding: utf-8

import os
import re
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

from utils.log_utils import log, logTable, c_grey, c_darkgrey, color_palette_list
from utils.plot_distribution import display_name
from utils.compute_switched_probability import bootstrap_samples, montecarlo_samples
from utils.sampling_param_compute import (
    compute_sampling_error_curve,
    compute_at_least_one_exceedance_probability_curve,
    get_sampling_method_label,
)

# Figures are drawn at their printed size (column width of the paper, cas-dc: 3.3 in) so the typeset text is not
# scaled; sizes as in STYLE_GUIDE 12.2 of the paper series.
FIG_WIDTH_IN = 3.3
PLOT_HEIGHT_IN = 2.7
PLOT_DERIVATIVE_HEIGHT_IN = 3.6
LEGEND_FONTSIZE = 6.5
PAPER_RC = {'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
            'ytick.labelsize': 7.5, 'legend.fontsize': LEGEND_FONTSIZE, 'axes.linewidth': 0.6}

_PLOT_METHOD_LABEL = {'analytical': 'Analytical formula', 'bootstrap': 'Bootstrap', 'montecarlo': 'Monte Carlo'}
_METRIC_LABEL = {'accuracy': 'accuracy', 'mAP50': 'mAP$_{50}$', 'mAP50-95': 'mAP$_{50-95}$', 'P': 'precision',
                 'R': 'recall'}
_MNIST_LABEL = {'SimplePerceptron': 'Simple Perceptron', 'HiddenLayerPerceptron': 'Hidden Layer Perceptron'}


def _resolve_simulation_iterations(sampling_method, n_iterations):
    if n_iterations is not None:
        return n_iterations
    if sampling_method == 'bootstrap':
        return bootstrap_samples
    if sampling_method == 'montecarlo':
        return montecarlo_samples
    return 0


def _curve_label(name):
    """Legend label in the vocabulary of the paper: YOLOv8 groups through display_name(); the CNN_14L batch size
    variants as 'CNN_14L, BS b' and the ablation conditions as 'BS b, LR l' (batch size, learning rate)."""
    match = re.fullmatch(r'CNN_14L_B(\d+)_L([\d.]+)', name)
    if match:
        return f'BS {match.group(1)}, LR {match.group(2)}'
    match = re.fullmatch(r'CNN_14L_B(\d+)', name)
    if match:
        return f'CNN_14L, BS {match.group(1)}'
    return _MNIST_LABEL.get(name, display_name(name))


def _plot_method_label(sampling_method, simulation_iterations):
    label = _PLOT_METHOD_LABEL.get(sampling_method, sampling_method)
    return label if sampling_method == 'analytical' else f"{label}, {simulation_iterations:,} samples"


def _new_figure(plot_derivative):
    if plot_derivative:
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(FIG_WIDTH_IN, PLOT_DERIVATIVE_HEIGHT_IN),
                                      gridspec_kw={'height_ratios': [2.6, 1.0]})
        ax2.set_ylabel('Derivative')
    else:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, PLOT_HEIGHT_IN))
        ax2 = None
    return fig, ax, ax2


def _legend_at_axes_width(fig, ax, handles, labels, ncol, y_anchor):
    """Legend below the axes with its frame exactly as wide as the axes frame (STYLE_GUIDE 12.2): the spacing
    between columns absorbs the free width and the frame keeps its inner margins."""
    fig.canvas.draw()
    box = ax.get_position()
    unit = LEGEND_FONTSIZE * fig.dpi / 72
    spacing = 1.0
    if ncol == 1:
        leg = fig.legend(handles, labels, loc='lower left', ncol=1, fontsize=LEGEND_FONTSIZE, frameon=True,
                         framealpha=0.9, edgecolor='0.2', borderpad=0.9, handlelength=1.4, handletextpad=0.5, labelspacing=0.3,
                         mode='expand', borderaxespad=0, bbox_to_anchor=(box.x0, y_anchor, box.width, 0))
        fig.canvas.draw()
        return leg
    for _ in range(8):
        leg = fig.legend(handles, labels, loc='lower center', ncol=ncol, fontsize=LEGEND_FONTSIZE, frameon=True,
                         framealpha=0.9, edgecolor='0.2', borderpad=0.9, handlelength=1.4, handletextpad=0.5, labelspacing=0.3,
                         bbox_to_anchor=(box.x0 + box.width / 2, y_anchor), columnspacing=spacing)
        fig.canvas.draw()
        gap = ax.get_window_extent().width - leg.get_window_extent().width
        if abs(gap) < 0.5:
            break
        leg.remove()
        spacing += gap / unit / max(ncol - 1, 1)
    else:
        raise AssertionError('legend width does not converge to the axes width')
    if spacing < 0.5:
        leg.remove()
        raise AssertionError('legend entries do not fit in the axes width')
    renderer = fig.canvas.get_renderer()
    frame = leg.get_frame().get_window_extent(renderer)
    handles_drawn = getattr(leg, 'legend_handles', None) or leg.legendHandles
    left = (min(h.get_window_extent(renderer).x0 for h in handles_drawn) - frame.x0) * 72 / fig.dpi
    right = (frame.x1 - max(t.get_window_extent(renderer).x1 for t in leg.get_texts())) * 72 / fig.dpi
    if min(left, right) < 4:
        leg.remove()
        raise AssertionError(f'legend entries too close to the frame ({left:.1f} / {right:.1f} pt)')
    return leg


def _finish_figure(fig, ax, ax2, handles, labels, title, y_label, sample_sizes, max_ncol):
    """Titles, axes, grid and a legend below the plot, as wide as the axes; the figure grows to hold the legend."""
    ax.set_title(title)
    ax.set_ylabel(y_label)
    (ax2 if ax2 is not None else ax).set_xlabel('Number of runs (N)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # X ticks include the first and last sample size and a few intermediate ones
    n = len(sample_sizes)
    indices = np.unique(np.round(np.linspace(0, n - 1, min(8, n))).astype(int))
    for axis in [a for a in (ax, ax2) if a is not None]:
        axis.set_xticks(sample_sizes[indices])
        axis.grid(visible=True, color=c_grey, linestyle='--', linewidth=0.5, alpha=0.7)
        axis.tick_params(length=2.5, pad=2)
    if ax2 is not None:
        fig.align_ylabels([ax, ax2])

    fig.tight_layout(pad=0.3)
    if not labels:
        return

    # First pass: the most columns that fit in the axes width, and the height the legend needs
    for ncol in range(min(max_ncol, len(labels)), 0, -1):
        try:
            leg = _legend_at_axes_width(fig, ax, handles, labels, ncol, 0.0)
            break
        except AssertionError:
            continue
    else:
        raise RuntimeError('legend does not fit in the axes width')
    legend_height_in = leg.get_window_extent().height / fig.dpi
    leg.remove()

    # Second pass: grow the figure below the plot and place the legend there
    gap_in = 0.45 if ax2 is None else 0.4
    plot_height_in = fig.get_figheight()
    total_height_in = plot_height_in + legend_height_in + gap_in
    fig.set_size_inches(FIG_WIDTH_IN, total_height_in)
    fig.tight_layout(pad=0.3, rect=(0, (legend_height_in + gap_in) / total_height_in, 1, 1))
    leg = _legend_at_axes_width(fig, ax, handles, labels, ncol, 0.02 / total_height_in)
    fig.canvas.draw()
    log(f"  legend: {ncol} columns, {leg.get_window_extent().width / fig.dpi:.3f} in wide, "
        f"axes {ax.get_window_extent().width / fig.dpi:.3f} in")


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
                        title_template='Sampling curve\n{method_label}',
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

    with plt.rc_context(PAPER_RC):
        fig, ax, ax2 = _new_figure(plot_derivative)

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
            ax.plot(sample_sizes[valid_points], curve_values[valid_points], linestyle='--', alpha=0.6,
                    label=_curve_label(model_name), color=primary_color, linewidth=1.0)

            if plot_derivative and np.sum(valid_points) >= 2 and ax2 is not None:
                deriv_vals = np.gradient(curve_values[valid_points], sample_sizes[valid_points])
                # plot derivative without legend label so the global legend shows only primary curves
                ax2.plot(sample_sizes[valid_points], deriv_vals, linestyle='--', color=primary_color, linewidth=0.8, alpha=0.9)

            def _fmt_value(value):
                return '-' if not np.isfinite(value) else f"{value:.3f}"

            row_data.append([model_name] + [_fmt_value(curve_values[idx]) for _, idx, _ in table_checkpoints])

        avg_curve = np.mean(curve_values_all, axis=0)
        valid_points = np.isfinite(avg_curve)
        ax.plot(sample_sizes[valid_points], avg_curve[valid_points], label='Average', color=c_darkgrey, linewidth=1.6)

        # Short labels first so that the long ones share columns
        handles, labels = ax.get_legend_handles_labels()
        ordered = sorted(zip(handles, labels), key=lambda pair: len(pair[1]) > 20)
        handles, labels = [list(x) for x in zip(*ordered)] if ordered else ([], [])

        title = title_template.format(method_label=_plot_method_label(sampling_method, simulation_iterations),
                                      **curve_kwargs)
        _finish_figure(fig, ax, ax2, handles, labels, title, y_label, sample_sizes, max_ncol=3)

        extra_title = f"_{title_tag}" if title_tag else ""
        method_suffix = sampling_method.lower()
        filename = f"{filename_prefix}{extra_title}.pdf" if method_suffix == 'analytical' else f"{filename_prefix}_{method_suffix}{extra_title}.pdf"
        fig.savefig(os.path.join(analysis_path, filename), format="pdf", bbox_inches='tight', pad_inches=0.02)

    table_title = f"{metric.title()} {table_title_prefix} ({display_method_label}) {title_tag.replace('_', ' ').title()}"
    log(f"\nSummary Table of {table_title}:")
    colalign = ['left'] + ['right'] * (len(row_data[0]) - 1)
    logTable(row_data=row_data, output_path=f"{analysis_path}/tables", filename=table_title, colalign=colalign)

    if close_plot:
        plt.close(fig)


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
                        title_template='Sampling curve\n{method_label}',
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

    with plt.rc_context(PAPER_RC):
        fig, ax, ax2 = _new_figure(plot_derivative)

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
            ax.plot(sample_sizes[valid_points], curve_values[valid_points], label=f'$P_{{{percentile}}}$',
                    color=primary_color, linewidth=1.2)

            if plot_derivative and np.sum(valid_points) >= 2 and ax2 is not None:
                deriv_vals = np.gradient(curve_values[valid_points], sample_sizes[valid_points])
                # plot derivative without legend label so the global legend shows only primary curves
                ax2.plot(sample_sizes[valid_points], deriv_vals, linestyle='--', color=primary_color, linewidth=0.8, alpha=0.9)

            def _fmt_value(value):
                return '-' if not np.isfinite(value) else f"{value:.3f}"

            row_data.append([model_name] + [_fmt_value(curve_values[idx]) for _, idx, _ in table_checkpoints])

        handles, labels = ax.get_legend_handles_labels()
        title = title_template.format(method_label=_plot_method_label(sampling_method, simulation_iterations),
                                      p_min=percentile_range[0], p_max=percentile_range[1])
        _finish_figure(fig, ax, ax2, handles, labels, title, y_label, sample_sizes, max_ncol=7)

        extra_title = f"_{title_tag}" if title_tag else ""
        method_suffix = sampling_method.lower()
        filename = f"{filename_prefix}{extra_title}.pdf" if method_suffix == 'analytical' else f"{filename_prefix}_{method_suffix}{extra_title}.pdf"
        fig.savefig(os.path.join(analysis_path, filename), format="pdf", bbox_inches='tight', pad_inches=0.02)

    table_title = f"{metric.title()} {table_title_prefix} ({display_method_label}) {title_tag.replace('_', ' ').title()}"
    log(f"\nSummary Table of {table_title}:")
    colalign = ['left'] + ['right'] * (len(row_data[0]) - 1)
    logTable(row_data=row_data, output_path=f"{analysis_path}/tables", filename=table_title, colalign=colalign)

    if close_plot:
        plt.close(fig)


def plot_all_sampling_errors(metrics_data,
                             analysis_path,
                             metric='accuracy',
                             title_tag='',
                             plot_models=None,
                             color_list=color_palette_list,
                             n_iterations=None,
                             close_plot=True):
    metric_label = _METRIC_LABEL.get(metric, metric)
    for sampling_method in ['analytical', 'bootstrap', 'montecarlo']:
        for parameter in ['mean', 'std']:
            parameter_label = 'mean' if parameter == 'mean' else 'standard deviation'
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
                title_template=f'Sampling error of the {parameter_label}\n{{method_label}}',
                y_label=f'Sampling error ({metric_label}, points)',
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
            title_template='P(at least one run > $P_{{{percentile}}}$)\n{method_label}',
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
            title_template='Mean P(at least one run > $P_X$), X in [{p_min}, {p_max}]\n{method_label}',
            y_label='Probability',
            filename_prefix=f'sampling_probability{extra_title}_p{str(percentile_range)}',
            table_title_prefix=f'Sampling Probability p{str(percentile_range)}',
            close_plot=close_plot,
        )
