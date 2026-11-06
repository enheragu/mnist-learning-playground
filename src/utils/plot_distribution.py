#!/usr/bin/env python3
# encoding: utf-8

import os
import re
import itertools
import copy

from collections import OrderedDict
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, shapiro, kurtosis
from utils.log_utils import log, logTable, bcolors, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list, c_alpha_red, c_alpha_darkgrey, c_alpha_grey, c_alpha_blue


# Whether to plot or just store images in disk
only_store = True
plt.rcParams.update({'font.size': 24,
                     'legend.fontsize': 15})

# Legend vocabulary of the articles: fusion names (RGBT, VTHS, HSVT, VT, LWIR) and equalization conditions
# (no / visible / combined / thermal equalization); the CNN_14L batch size variants as 'CNN_14L, BS b' and the ablation
# conditions as 'BS b, LR l' (batch size, learning rate), as in plot_sampling_graph.py. Other model names pass through.
_DISPLAY_EXACT = {'LLVIP_lwir': 'LLVIP LWIR', 'LLVIP_Ch4v3': 'LLVIP Ch4v3', 'kaist_day_rgbt': 'KAIST day RGBT',
                  'kaist_day_vths_v2': 'KAIST day VTHS', 'kaist_day_hsvt': 'KAIST day HSVT'}
_DISPLAY_EQ = [('_rgb_th_equalization', ', combined equalization'), ('_rgb_equalization', ', visible equalization'),
               ('_th_equalization', ', thermal equalization'), ('_no_equalization', ', no equalization')]


# Printed size of the figures in the articles: half the text width of the two-column template (cas-dc, 3.3 in), with
# the font sizes of the series at that width (STYLE_GUIDE 12.2), as in plot_sampling_graph.py. paper_size=True draws
# at this size; without it the plots keep the large canvas that other callers scale with their own rc settings.
PAPER_FIG_SIZE_IN = (3.3, 2.6)
PAPER_RC = {'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
            'ytick.labelsize': 7.5, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6}
PAPER_LINE_SCALE = 0.5  # line widths of the large canvas, scaled to the printed size
_WRAP_LINESTYLES = ['-', '--', ':', '-.']  # one per pass over the palette in the survival plot


def display_name(name):
    """Legend label for a results group: paper vocabulary for the YOLO groups, unchanged for everything else."""
    if name in _DISPLAY_EXACT:
        return _DISPLAY_EXACT[name]
    match = re.fullmatch(r'CNN_14L_B(\d+)_L([\d.]+)', name)
    if match:
        return f'BS {match.group(1)}, LR {match.group(2)}'
    match = re.fullmatch(r'CNN_14L_B(\d+)', name)
    if match:
        return f'CNN_14L, BS {match.group(1)}'
    for suffix, text in _DISPLAY_EQ:
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            if base.startswith('LLVIP_'):
                base = 'LLVIP ' + base[6:]
            return base + text
    return name


def plot_metric_normaldistribution(data_y, ax, color, mean=None, std=None, linestyle='--', line_scale=1.0):
    # Normal distribution fit for the model
    mean = np.mean(data_y) if mean is None else mean
    std = np.std(data_y) if std is None else std
    if std > 0.000001:
        x_right = max(mean + std * 4, max(data_y))
        x_left = min(mean - std * 4, min(data_y))
        x = np.linspace(x_left, x_right, 100)
        y = norm.pdf(x, mean, std)
        sns.lineplot(x=x, y=y, linestyle=linestyle, linewidth=2 * line_scale, color=color, ax=ax)
        # Mean (k = 0) and -3 to +3 standard deviations; an integer k does not depend on the scale of the data
        for k in range(-3, 4):
            pos = mean + k * std
            linewidth = (2 if k == 0 else 1) * line_scale
            linestyle = 'solid' if k == 0 else ':'
            ax.vlines(x=pos, ymin=0, ymax=norm.pdf(pos, mean, std), colors=color, linewidth=linewidth, linestyles=linestyle)

def plot_metric_gammadistribution(data_y, ax, color, mean=None, std=None, linestyle='--', line_scale=1.0):

    if np.any(np.array(data_y) <= 0):
        log(f"⚠️  Invalid data for Gamma distribution: {np.sum(np.array(data_y) <= 0)} values <= 0", color=bcolors.WARNING)
        return
    if np.std(np.array(data_y)) == 0:
        log("⚠️  Zero variance data for Gamma distribution. Cannot compute.", color=bcolors.WARNING)
        return

    # Gamma distribution fit for the model
    data_y = np.array(data_y)
    data_corrected = copy.deepcopy(data_y)
    data_corrected = data_corrected.astype(float)
    data_corrected[data_corrected == 0] = 1e-8 # Data needs to be at lesat floc value
    # print(f"{data_corrected.min() = }; {data_corrected.max() = }")
    shape, loc, scale = gamma.fit(data_corrected, floc=0)
    x_right = max(shape * scale * 3, max(data_y))
    x = np.linspace(0, x_right, 100)
    y = gamma.pdf(x, shape, loc, scale)

    sns.lineplot(x=x, y=y, linestyle=linestyle, linewidth=2 * line_scale, color=color, ax=ax)

    for pos in [0.05,0.25,0.5,0.750,0.95]:
        pos_percentile = gamma.ppf(pos, shape, loc, scale)
        linewidth = (1 if pos != 0.5 else 2) * line_scale
        linestyle = ':' if pos != 0.5 else 'solid'
        ax.vlines(x=pos_percentile, ymin=0, ymax=gamma.pdf(pos_percentile, shape, loc, scale), colors=color, linewidth=linewidth, linestyles=linestyle)


"""
    Plots theoretical normal distribution based on given data, and plots binned distribution or real data.
    paper_size draws the figure at its printed size in the articles (PAPER_FIG_SIZE_IN, PAPER_RC).
"""
def plot_metric_distribution(*args, paper_size=False, **kwargs):
    if not paper_size:
        return _plotMetricDistribution(*args, **kwargs)
    with plt.rc_context(PAPER_RC):
        return _plotMetricDistribution(*args, figsize=PAPER_FIG_SIZE_IN, line_scale=PAPER_LINE_SCALE, **kwargs)


def _plotMetricDistribution(metrics_data, train_duration_data = None, metric_label = 'accuracy (%)', plot_func = plot_metric_normaldistribution,
                             color_palette = color_palette_list, vertical_lines_acc = None, analysis_path = None,
                             plot_filename = None, plot_mean = None, plot_std = None,
                             bin_size = 20,
                             title = None,
                             x_label = None,
                             y_label = None,
                             hist_stat = "density",
                             x_max_quantile = None,
                             hist_kwargs = None,
                             show_histogram = True,
                             show_rug = False,
                             force_legend_loc = None,
                             figsize = (14, 11),
                             line_scale = 1.0):
    
    if analysis_path is None:
        log(f"[Error] [plot_metric_distribution] no analysis_path was provided for {metric_label}")
        return

    if vertical_lines_acc is None:
        vertical_lines_acc = []
    if hist_kwargs is None:
        hist_kwargs = {}
    
    _, ax = plt.subplots(figsize=figsize)
    max_density = 0
    x_range = [float('inf'), float('-inf')]
    all_x_values = []

    for index, (model_name, data_y) in enumerate(metrics_data.items()):
        # train_duration = np.mean(train_duration_data[model_name])
        # train_duration_str = f"{int(train_duration // 60)}min {train_duration % 60:.2f}s"
        color = color_palette[index]

        # Specific case for constant plot
        if np.ptp(data_y) == 0:
            x0 = data_y[0]
            ax.axvline(x0, color=color, linewidth=4 * line_scale, alpha=0.7,
                    label=f"{display_name(model_name)} (n={len(data_y)}), X = {x0:.4f}")
            max_density = max(max_density, 100)
        elif show_histogram:
            hist = sns.histplot(data_y, bins=bin_size, stat=hist_stat, alpha=0.4, label=f"{display_name(model_name)}   (n={len(data_y)})",
                        color=color, edgecolor='none', ax=ax, **hist_kwargs)
            patch_heights = [patch.get_height() for patch in hist.patches]
            max_density = max(max_density, max(patch_heights))
        else:
            ax.plot([], [], color=color, label=f"{display_name(model_name)}   (n={len(data_y)})")
            mean_v = np.mean(data_y) if plot_mean is None else plot_mean
            std_v = np.std(data_y) if plot_std is None else plot_std
            if std_v > 1e-6:
                from scipy.stats import norm as _norm
                max_density = max(max_density, _norm.pdf(mean_v, mean_v, std_v))
        
        x_range[0] = min(x_range[0], min(data_y))
        x_range[1] = max(x_range[1], max(data_y))
        all_x_values.extend(np.asarray(data_y, dtype=float).tolist())
        if plot_func is not None:
            curve_linestyle = '--' if show_histogram else '-'
            plot_func(data_y, ax, color, plot_mean, plot_std, linestyle=curve_linestyle,
                      **({'line_scale': line_scale} if line_scale != 1.0 else {}))

    if metric_label == "Accuracy (%)":
        if vertical_lines_acc:
            # At two decimals, the resolution of the values (so that float noise does not move a line across a bound),
            # and strict: a value equal to a bound of the range is not inside it
            range_low, range_high = round(x_range[0], 2), round(x_range[1], 2)
            lines_in_range = [line for line in vertical_lines_acc if range_low < round(line, 2) < range_high]
            lines_out_range = [line for line in vertical_lines_acc if not range_low < round(line, 2) < range_high]
            count_in_range = len(lines_in_range)
            print(f"[{metric_label}] Models {'_'.join(list(metrics_data.keys()))} with range {[range_low, range_high]} include {count_in_range} models from review.")

            # Lines outside the range (grey, dotted, no label)
            if lines_out_range:
                ax.vlines(x=lines_out_range, ymin=ax.get_ylim()[0], ymax=max_density,
                        colors=c_alpha_grey, linewidth=1 * line_scale, linestyles=':', label=None, zorder=1)

            # Lines inside the range (red, solid, only the first one labelled)
            if lines_in_range:
                ax.vlines(x=lines_in_range, ymin=ax.get_ylim()[0], ymax=max_density,
                        colors=c_alpha_red, linewidth=2 * line_scale, linestyles='solid',
                        label=f'{count_in_range} models in range', zorder=1)
    
    if x_max_quantile is not None and len(all_x_values) > 0:
        x_left = min(all_x_values)
        x_right = np.percentile(np.asarray(all_x_values), x_max_quantile)
        if x_right > x_left:
            ax.set_xlim(x_left, x_right)

    if show_rug:
        # Short vertical marks for the observed values (optional, off by default).
        # Height relative to the tallest fitted curve/bar.
        rug_h = 0.06 * max_density if max_density > 0 else 1.0
        for index, (model_name, data_y) in enumerate(metrics_data.items()):
            ax.vlines(np.asarray(data_y, dtype=float), 0, rug_h, colors=color_palette[index],
                      linewidth=2.5 * line_scale, zorder=5)
    ax.set_title(title if title is not None else f"{metric_label} distribution")
    ax.set_xlabel(x_label if x_label is not None else metric_label)
    ax.set_ylabel(y_label if y_label is not None else ("Density" if hist_stat == "density" else "Count"))

    # Try legend placements: corners first, then below
    legend = ax.legend(loc=force_legend_loc if force_legend_loc is not None else 'upper left')

    # The collision check below uses each element's axis-aligned bounding
    # box, so a fit curve spanning most of the x-range can appear to
    # "cover" every corner even where it is visually clear (its bbox runs
    # from the baseline up to its peak across nearly the full width).
    # force_legend_loc skips that heuristic for callers who already know
    # which corner is actually free.
    if legend is not None and force_legend_loc is None:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        
        # Get legend bounds
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend_labels = [label for label in ax.get_legend_handles_labels()[1] if label and not label.startswith('_')]
        
        # Check collision with actual plot elements (patches and lines)
        def has_collision_with_elements():
            # At printed size the legend is relatively larger: only the bars and the fitted curves count, and only
            # an overlap of more than 15% of the legend (the reference lines drawn behind the data are ignored)
            plot_elements = list(ax.patches) + list(ax.lines) + (list(ax.collections) if line_scale == 1.0 else [])
            
            for element in plot_elements:
                if not element.get_visible():
                    continue
                try:
                    elem_bbox = element.get_window_extent(renderer=renderer)
                    if elem_bbox and legend_bbox.overlaps(elem_bbox):
                        # Calculate overlap area
                        lx0, ly0, lx1, ly1 = legend_bbox.extents
                        ex0, ey0, ex1, ey1 = elem_bbox.extents
                        overlap_w = max(0, min(lx1, ex1) - max(lx0, ex0))
                        overlap_h = max(0, min(ly1, ey1) - max(ly0, ey0))
                        overlap_area = overlap_w * overlap_h
                        
                        # Only consider it a collision if >1000 px² (histograms/thick lines are visible)
                        min_overlap = 1000 if line_scale == 1.0 else 0.15 * legend_bbox.width * legend_bbox.height
                        if overlap_area > min_overlap:
                            return True
                except Exception:
                    continue
            
            return False
        
        # At printed size a legend of more than four entries would cover the data: it goes below
        long_legend = line_scale != 1.0 and len(legend_labels) > 4
        has_collision = long_legend or has_collision_with_elements()
        
        # If collision, try other corners
        if has_collision and not long_legend:
            orig_handles, orig_labels = ax.get_legend_handles_labels()
            for loc in ['upper right', 'lower left', 'lower right']:
                legend.remove()
                legend = ax.legend(handles=orig_handles, labels=orig_labels, loc=loc)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                legend_bbox = legend.get_window_extent(renderer=renderer)
                
                if not has_collision_with_elements():
                    # Found a corner that fits!
                    has_collision = False
                    break
        
        # At printed size a short legend stays inside, where it covers the least data (matplotlib 'best'), so that the
        # figure keeps the size of the others; a long one goes below
        if has_collision and line_scale != 1.0 and len(legend_labels) <= 4:
            orig_handles, orig_labels = ax.get_legend_handles_labels()
            legend.remove()
            legend = ax.legend(handles=orig_handles, labels=orig_labels, loc='best')
            has_collision = False

        # If still colliding after trying all corners, move below
        if has_collision:
            legend.remove()
            legend_labels = [label for label in ax.get_legend_handles_labels()[1] if label and not label.startswith('_')]
            n_labels = len(legend_labels)
            ncol = 1 if n_labels <= 1 else (2 if n_labels <= 8 else 3)
            n_rows = (n_labels + ncol - 1) // ncol
            bottom_margin_frac = 0.08 + 0.035 * n_rows
            bottom_margin_frac = min(bottom_margin_frac, 0.30)
            
            if line_scale == 1.0:
                plt.tight_layout(rect=[0, bottom_margin_frac, 1, 1])
            
            legend = fig.legend(
                handles=[h for h in ax.get_legend_handles_labels()[0] if h is not None],
                labels=legend_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, 0.02),
                ncol=ncol,
                # fontsize='small',
                frameon=True,
                fancybox=True,
                facecolor='white',
                framealpha=0.95,
                edgecolor='0.3',
            )
            if line_scale != 1.0:
                # At printed size the figure grows by the height of the legend instead of shrinking the plot
                _fitLegendBelow(fig, legend, ncol)
        else:
            plt.tight_layout()
    else:
        plt.tight_layout()

    nohistogram_suffix = '' if show_histogram else '_nohistogram'
    if plot_filename is not None:
        path_name = os.path.join(analysis_path, f"{plot_filename}{nohistogram_suffix}.pdf")
    else:
        path_name = os.path.join(analysis_path,f"plot_{metric_label.replace(' (%)','').replace(' (s)','').replace(' ','_').lower()}_{'_'.join(list(metrics_data.keys()))}{nohistogram_suffix}.pdf")
    plt.savefig(path_name, format="pdf")
    # print(f"\t· Stored file in {path_name}")

    if only_store:
        plt.close()

"""
    Legend placed below the axes at printed size (paper_size): fewer columns while it is wider than the figure (a legend
    of more than ten entries in a smaller, compact layout), and the figure grows by its height so that the plot area
    keeps its size.
"""
def _fitLegendBelow(fig, legend, ncol, pad_in=0.05):
    handles, labels = legend.legend_handles, [text.get_text() for text in legend.get_texts()]
    compact = dict(fontsize=6, handlelength=1.0, columnspacing=0.6, handletextpad=0.3) if len(labels) > 10 else {}
    renderer = fig.canvas.get_renderer()
    while True:
        legend.remove()
        legend = fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=ncol,
                            frameon=True, fancybox=True, facecolor='white', framealpha=0.95, edgecolor='0.3', **compact)
        fig.canvas.draw()
        if ncol == 1 or legend.get_window_extent(renderer).width <= 0.98 * fig.bbox.width:
            break
        ncol -= 1
    legend_h_in = legend.get_window_extent(renderer).height / fig.dpi
    width, height = fig.get_size_inches()
    total = height + legend_h_in + pad_in
    fig.set_size_inches(width, total)
    legend.set_bbox_to_anchor((0.5, pad_in / 2 / total), transform=fig.transFigure)
    fig.tight_layout(rect=[0, (legend_h_in + pad_in) / total, 1, 1])
    return legend


"""
    Intermediate function to handle each distribution plot wanted
"""
def plotDataDistribution(metrics_data, models_plot_list = [[]], color_list = color_palette_list,
                         vertical_lines_acc = [], analysis_path=None, single_plots = True,
                         show_histogram = True, plot_filename = None, paper_size = False):
    
    if analysis_path is None:
        log(f"[Error] [plotDataDistribution] no analysis_path was provided for plots {models_plot_list = }")
        return
    
    color_iterator = itertools.cycle([color_palette_list])

    train_duration_data = None # default
    plot_train_duration = metrics_data is not None and 'train_duration' in list(list(metrics_data.values())[0].values())[0].keys()
    plot_best_epoch = metrics_data is not None and 'best_epoch' in list(list(metrics_data.values())[0].values())[0].keys()
    
    # if plot_train_duration:
    #     print(f"Plot train duration will be printed for {models_plot_list}")
    # else:
    #     print(f"Plot train duration will NOT be printed for {models_plot_list}")
    # if plot_best_epoch:
    #     print(f"Plot best epoch will be printed for {models_plot_list}")
    # else:
    #     print(f"Plot best epoch will NOT be printed for {models_plot_list}")

    # Plot all models in single plot
    if single_plots:
        for model in metrics_data.keys():
            color = next(color_iterator)
            accuracy_data = {model: [entry['accuracy']*100 for entry in metrics_data[model].values()]}
            
            if plot_best_epoch:
                best_epoch_data = {model: [entry['best_epoch'] for entry in metrics_data[model].values()]}
                plot_metric_distribution(best_epoch_data, train_duration_data, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=os.path.join(analysis_path,'single_model'), show_histogram=show_histogram, paper_size=paper_size)

            if plot_train_duration:
                train_duration_data = {model: [entry['train_duration'] for entry in metrics_data[model].values()]}
                plot_metric_distribution(train_duration_data, train_duration_data, metric_label = 'Train Duration (s)', plot_func=plot_metric_gammadistribution, color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=os.path.join(analysis_path,'single_model'), show_histogram=show_histogram, paper_size=paper_size)

            plot_metric_distribution(accuracy_data, train_duration_data, metric_label = 'Accuracy (%)', color_palette=color, vertical_lines_acc=vertical_lines_acc, analysis_path=os.path.join(analysis_path,'single_model'), show_histogram=show_histogram, paper_size=paper_size)

    for plot, color_scheme in zip(models_plot_list, color_list):
        print(f"Generating plot for {plot} model{'s' if len(plot)>1 else ''}")
        accuracy_data = OrderedDict()
        if plot_train_duration: 
            train_duration_data = OrderedDict()
        if plot_best_epoch: 
            best_epoch_data = OrderedDict()

        for model in plot:
            if model in metrics_data.keys():
                accuracy_data[model] = [entry['accuracy']*100 for entry in metrics_data[model].values()]
                if plot_train_duration: 
                    train_duration_data[model] = [entry['train_duration'] for entry in metrics_data[model].values()]
                if plot_best_epoch: 
                    best_epoch_data[model] = [entry['best_epoch'] for entry in metrics_data[model].values()]
            else:
                print(f"\t· [WARNING] {model} not in metrics available: {metrics_data.keys()}")

        # None of the models of this entry is available: there is nothing to plot
        if not accuracy_data:
            continue

        if plot_train_duration:
            plot_metric_distribution(train_duration_data, train_duration_data, metric_label = 'Train Duration (s)', plot_func=plot_metric_gammadistribution, color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path, show_histogram=show_histogram, paper_size=paper_size)
        if plot_best_epoch:
            plot_metric_distribution(best_epoch_data, train_duration_data, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path, show_histogram=show_histogram, paper_size=paper_size)

        plot_metric_distribution(accuracy_data, train_duration_data, metric_label = 'Accuracy (%)', color_palette=color_scheme, vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path, show_histogram=show_histogram, plot_filename=plot_filename, paper_size=paper_size)
        
    # plt.show()


def plot_survival_function(*args, paper_size=False, **kwargs):
    """Survival plot (see _plotSurvivalFunction); paper_size draws it at its printed size in the articles, with the
    legend below adding to the height of the figure."""
    if not paper_size:
        return _plotSurvivalFunction(*args, **kwargs)
    with plt.rc_context(PAPER_RC):
        return _plotSurvivalFunction(*args, figsize=PAPER_FIG_SIZE_IN, line_scale=PAPER_LINE_SCALE, **kwargs)


def _plotSurvivalFunction(metrics_data,
                          analysis_path,
                          metric_label='Accuracy (%)',
                          percentile_start=89,
                          percentile_step=0.1,
                          table_filename=None,
                          plot_prefix='survival',
                          legend_ncol=2,
                          figsize=(12, 9),
                          line_scale=1.0):
    """Empirical percentile exceedance calibration: one line per model.

    metric-agnostic: pass any pre-extracted list of floats per model, e.g.
        {'CNN_14L': [99.1, 99.4, ...]}           # accuracy
        {'yolo_task': [0.72, 0.81, ...]}         # mAP50
        {'yolo_task': [0.55, 0.61, ...]}         # mAP50-95

    X axis: percentile P.
    Y axis: empirical P(X > x_p) [%].
    Dashed line: theoretical 100-P % (continuous distribution reference).
    """
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)

    percentiles = np.arange(percentile_start, 100 + percentile_step, percentile_step, dtype=float)
    percentiles = percentiles[percentiles <= 100]

    key_ps = [50, 75, 90, 95, 97.5, 99]
    table_data = [['Model', 'N'] + [f'p{p:g}: empirical % > threshold (theoretical {100-p:g}%)' for p in key_ps]]

    fig, ax = plt.subplots(figsize=figsize)

    # Theoretical reference: P(X > x_p) = (100-P)% by definition of normal percentiles
    ax.plot(percentiles, 100 - percentiles, color='black', linestyle='--',
            linewidth=1.5 * line_scale, label='Theoretical normal: 100 − P %', zorder=5)

    for idx, (model_name, data_values) in enumerate(metrics_data.items()):
        data = np.asarray(data_values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            continue
        n = len(data)

        # Thresholds from fitted normal, NOT from the data itself.
        # This makes the comparison meaningful: empirical tail vs. theoretical tail.
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-9:
            continue
        theoretical_thresholds = norm.ppf(percentiles / 100.0, loc=mean, scale=std)

        empirical_rates = [np.mean(data > t) * 100.0 for t in theoretical_thresholds]

        color = color_palette_list[idx % len(color_palette_list)]
        # Once the palette wraps, the line style tells apart the curves that share a color
        linestyle = _WRAP_LINESTYLES[(idx // len(color_palette_list)) % len(_WRAP_LINESTYLES)]
        ax.plot(percentiles, empirical_rates, linestyle=linestyle,
                color=color, linewidth=1.6 * line_scale, label=f"{display_name(model_name)} (n={n})", alpha=0.85)

        key_vals = [f"{np.mean(data > norm.ppf(p/100.0, loc=mean, scale=std)) * 100:.1f}%" for p in key_ps]
        table_data.append([model_name, n] + key_vals)

    ax.set_xlabel('Percentile P (of fitted normal)')
    ax.set_ylabel('Empirical P(X > x_p)  [%]')
    ax.set_xlim(percentile_start, 100)
    ax.set_xticks(np.arange(int(np.ceil(percentile_start)), 101, 1))
    ax.autoscale(axis='y')
    ax.set_ylim(bottom=0)
    ax.set_title(f"{metric_label}: P(X > x_p)")
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.grid(True, alpha=0.3)
    if legend_labels and line_scale != 1.0:
        # At printed size the legend takes the columns that fit in the width of the figure, and the figure grows by
        # its height instead of shrinking the plot
        ncol = max(1, int(legend_ncol))
        legend = fig.legend(handles=legend_handles, labels=legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
                            ncol=ncol, frameon=True, fancybox=True, facecolor='white', framealpha=0.95, edgecolor='0.3')
        _fitLegendBelow(fig, legend, ncol)
    else:
        bottom_margin_frac = 0.07
        if legend_labels:
            ncol = max(1, int(legend_ncol))
            legend = fig.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc='lower left',
                bbox_to_anchor=(0.03, 0.015, 0.94, 0.0),
                mode='expand',
                ncol=ncol,
                # fontsize=12,
                frameon=True,
                fancybox=True,
                facecolor='white',
                framealpha=0.95,
                edgecolor='0.3',
            )

            # Reserve exactly the space the rendered legend occupies (+small padding).
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_bbox_fig = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
            required_bottom_margin = max(legend_bbox_fig.y1 + 0.006, 0.06)
            bottom_margin_frac = min(required_bottom_margin, 0.80)
        plt.tight_layout(rect=[0, bottom_margin_frac, 1, 1])

    output_plot = os.path.join(analysis_path, f"{plot_prefix}.pdf")
    plt.savefig(output_plot, format='pdf')
    if only_store:
        plt.close()

    if len(table_data) > 1:
        logTable(
            row_data=table_data,
            output_path=f"{analysis_path}/tables",
            filename=table_filename or f"Survival Function Calibration {metric_label}",
            colalign=['left', 'right'] + ['right'] * len(key_ps)
        )

