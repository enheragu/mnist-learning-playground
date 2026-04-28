#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from scipy.stats import norm, gamma, shapiro, kurtosis

from utils.log_utils import log, logTable


def normal_amplitude(data):
    mean = np.mean(data)
    std = np.std(data)

    percentile_0_5 = norm.ppf(0.001, loc=mean, scale=std)
    percentile_99_5 = norm.ppf(0.999, loc=mean, scale=std)
    amplitude_99 = percentile_99_5 - percentile_0_5
    return amplitude_99


def gamma_amplitude(data):
    data_array = np.asarray(data, dtype=float)
    positive_data = data_array[np.isfinite(data_array) & (data_array > 0)]

    if positive_data.size < 2:
        return 0.0

    k, loc, scale = gamma.fit(positive_data, floc=0)

    percentile_0_5 = gamma.ppf(0.001, k, loc, scale)
    percentile_99_5 = gamma.ppf(0.999, k, loc, scale)
    amplitude_99 = percentile_99_5 - percentile_0_5
    return amplitude_99


def normalityTest(metrics_data,
                  metric='accuracy',
                  title_tag='',
                  analysis_path='./analysis_results/distributions',
                  unit_multiplier=100):
    metric_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        metric_data[model] = [entry[metric] * unit_multiplier for entry in metrics_data[model].values()]

    row_data = [['Model', 'Median (%)', 'Mean (%)', 'Kurtosis (Fisher)', 'Shapiro-Wilk: W', 'Shapiro-Wilk: p-value (%)']]
    for model_name, data in metric_data.items():
        estadistico, p_valor = shapiro(data)
        kurt = kurtosis(data, fisher=True)
        row_data.append([
            f"{model_name} (n={len(data)})",
            f"{np.median(data):.3f}",
            f"{np.mean(data):.3f}",
            f"{kurt:.4f}",
            f"{estadistico:.4f}",
            f"{p_valor:.4f}",
        ])

    log(f"\nSummary Table of {metric.title()} Shapiro-Wilk normality test:")
    title_suffix = "" if title_tag == '' else f" {title_tag.replace('_', ' ').title()}"
    logTable(
        row_data,
        f"{analysis_path}/tables",
        f"{metric.title()} Normality test{title_suffix}",
        colalign=['left', 'right', 'right', 'right', 'right', 'right']
    )


def maxAmplitude(metrics_data,
                 metric='accuracy',
                 unit=' (%)',
                 unit_multiplier=100,
                 format='.3f',
                 amplitude_function=normal_amplitude,
                 title_tag='',
                 analysis_path='./analysis_results/distributions'):
    metric_data = {}
    log("Data available is:")
    for model, data in metrics_data.items():
        metric_data[model] = [entry[metric] * unit_multiplier for entry in metrics_data[model].values()]

    row_data = [['Model', f'mean{unit}', f'min{unit}', f'max{unit}', f'Data Amplitude{unit}', f'Amplitude 99.9% Interval Distribution{unit}']]
    sortable_rows = []
    for model_name, data in metric_data.items():
        data_array = np.asarray(data, dtype=float)
        finite_data = data_array[np.isfinite(data_array)]

        if finite_data.size == 0:
            amplitude_distribution = 0.0
            data_amplitude = 0.0
            mean_value = 0.0
            min_value = 0.0
            max_value = 0.0
        else:
            data_amplitude = float(np.max(finite_data) - np.min(finite_data))
            mean_value = float(np.mean(finite_data))
            min_value = float(np.min(finite_data))
            max_value = float(np.max(finite_data))
            try:
                amplitude_distribution = float(amplitude_function(finite_data))
                if not np.isfinite(amplitude_distribution):
                    amplitude_distribution = data_amplitude
            except Exception:
                amplitude_distribution = data_amplitude

        row = [
            f"{model_name} (n={len(data)})",
            f"{mean_value:{format}}",
            f"{min_value:{format}}",
            f"{max_value:{format}}",
            f"{data_amplitude:{format}}",
            f"{amplitude_distribution:{format}}",
        ]
        sortable_rows.append((data_amplitude, row))

    for _, row in sorted(sortable_rows, key=lambda item: item[0], reverse=True):
        row_data.append(row)

    log(f"\nSummary {metric.title()} max amplitude:")
    logTable(
        row_data,
        f"{analysis_path}/tables",
        f"{metric.title()} Max amplitude {title_tag.replace('_', ' ').title()}",
        colalign=['left', 'right', 'right', 'right', 'right', 'right']
    )


def count_trials(metrics_data,
                 title_tag='',
                 analysis_path='./analysis_results/distributions'):
    row_data = [['Model', 'N']]
    for model_name, data in metrics_data.items():
        row_data.append([model_name, len(data)])

    log("\nTrials on each model:")
    logTable(
        row_data,
        f"{analysis_path}/tables",
        f"N trials on each model {title_tag.replace('_', ' ').title()}"
    )
