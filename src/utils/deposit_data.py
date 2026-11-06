#!/usr/bin/env python3
# encoding: utf-8

"""
    Reads the public data deposit of the article (the export of 08_export_zenodo.py, the files archived in Zenodo)
    into the same structures that the analysis scripts build from the training logs, so that the figures and tables
    can be regenerated without the training outputs. Only the fields that the analysis uses are filled.
"""

import os
import csv


def _readRows(deposit_path, filename):
    with open(os.path.join(deposit_path, filename), newline="") as f:
        return list(csv.DictReader(f))


def _mnistTrial(row):
    trial = {'accuracy': float(row['accuracy']),
             'best_epoch': int(row['best_epoch']),
             'total_epochs': int(row['total_epochs']),
             'train_duration': float(row['train_duration_s'])}
    if row['seed']:
        trial['seed'] = int(row['seed'])
    return trial


"""
    Trials of the main and batch size experiments per model, as getAllModelData() returns them from
    randomseed_training_metrics.yaml: {model: {trial_id: trial}}, models in alphabetical order and trials in the order
    of the training log.
"""
def getDepositModelData(deposit_path):
    metrics_data = {}
    for row in _readRows(deposit_path, "mnist_trials.csv"):
        if row['experiment'] in ('main', 'batch_size'):
            metrics_data.setdefault(row['condition'], {})[row['trial_id']] = _mnistTrial(row)
    return dict(sorted(metrics_data.items()))


"""
    Trials of the ablation as in its training log (CNN_14L_Ablation/training_metrics.yaml): {iteration: {condition:
    trial}}, iterations in training order and conditions by batch size and learning rate, as they are in the log.
"""
def getDepositAblationLog(deposit_path):
    rows = [row for row in _readRows(deposit_path, "mnist_trials.csv") if row['experiment'] == 'ablation']
    rows.sort(key=lambda row: (int(row['iteration']), int(row['batch_size']), float(row['learning_rate'])))
    ablation_log = {}
    for row in rows:
        trial = _mnistTrial(row)
        trial['batch_size'] = int(row['batch_size'])
        trial['learning_rate'] = float(row['learning_rate'])
        # Key of the trial in getAblationModelData(); the deposit has no timestamps
        trial['finish_timetag'] = row['trial_id']
        iteration = ablation_log.setdefault(f"{row['iteration']}_iteration_{row['seed']}", {})
        iteration[f"{trial['batch_size']}-{trial['learning_rate']}"] = trial
    return ablation_log


"""
    Valid YOLOv8 runs per group with the fields of the run summaries (results.yaml.cached) that the analysis uses, as
    getYOLOVarianceData() combines them: {group: {'validation_0': {'data': {class: {metric: [one value per run]}}},
    'train_data': {...}, ...}}. Groups in the order of yolo_groups.csv, runs in the order of yolo_runs.csv.
"""
def getDepositYoloResults(deposit_path):
    groups = _readRows(deposit_path, "yolo_groups.csv")
    runs = _readRows(deposit_path, "yolo_runs.csv")
    results = {}
    for group in groups:
        group_runs = [run for run in runs if run['group'] == group['group']]
        results[group['group']] = {
            'validation_0': {'data': {
                'all': {'mAP50': [float(run['mAP50']) for run in group_runs],
                        'mAP50-95': [float(run['mAP50_95']) for run in group_runs]},
                'person': {'P': [float(run['P_person']) for run in group_runs],
                           'R': [float(run['R_person']) for run in group_runs],
                           'mAP50': [float(run['mAP50_person']) for run in group_runs],
                           'mAP50-95': [float(run['mAP50_95_person']) for run in group_runs]}}},
            'train_data': {'epoch_best_fit_index': [int(run['best_epoch']) for run in group_runs],
                           'train_duration_h': [float(run['train_duration_h']) for run in group_runs]},
            'seed': [int(run['seed']) for run in group_runs],
            'batch': [int(group['batch_size'])] * len(group_runs),
            'rgb_equalization': [group['rgb_equalization']] * len(group_runs),
            'thermal_equalization': [group['thermal_equalization']] * len(group_runs),
        }
    return results


"""
    mAP50 and mAP50-95 of the final evaluation of each valid YOLOv8 run, the values of the tables (see
    getDepositYoloResults() and collectYoloFinalMetrics()): two dicts {group: [one value per run]}.
"""
def getDepositYoloFinalMetrics(deposit_path):
    final_map50, final_map5095 = {}, {}
    for run in _readRows(deposit_path, "yolo_runs.csv"):
        final_map50.setdefault(run['group'], []).append(float(run['mAP50']))
        final_map5095.setdefault(run['group'], []).append(float(run['mAP50_95']))
    return final_map50, final_map5095
