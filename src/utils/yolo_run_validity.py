#!/usr/bin/env python3
# encoding: utf-8

"""
    Inventory of the YOLO variance runs and the validity criterion used in the analysis.

    A run is valid when:
      - it finished training and has a final evaluation (train_data and validation_0 in results.yaml; interrupted runs
        only log the per-epoch validation),
      - its real seed (args.yaml) is not 1: the training launcher forced seed=1 in args.yaml from 2024-06-19 to
        2025-10-08, and the seed logged in results.yaml was not the one used for training,
      - it is the first finished run with that seed in its group (seeds come from randint(0, 300), repeated seeds share
        the initialization).
"""

import os
import csv

import yaml

from utils.log_utils import log, bcolors

yaml_loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
inventory_fields = ['tag', 'run', 'start', 'args_seed', 'cache', 'final_eval', 'seed1', 'dup_seed', 'valid']


def loadYaml(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=yaml_loader) or {}


def scanYoloRun(run_path, tag, yolo_root):
    args = loadYaml(os.path.join(run_path, 'args.yaml'))
    results = loadYaml(os.path.join(run_path, 'results.yaml'))

    train_data = results.get('train_data') or {}
    all_data = ((results.get('validation_0') or {}).get('data') or {}).get('all') or {}
    return {'tag': tag, 'run': os.path.relpath(run_path, yolo_root),
            'start': str(train_data.get('train_start_time', '') or '')[:19],
            'args_seed': str(args.get('seed', '')), 'cache': str(args.get('cache', '')),
            'final_eval': bool(train_data) and all_data.get('mAP50') is not None}


def updateYoloRunsInventory(variance_tasks, yolo_root, inventory_file, rescan=False):
    """Scan the run folders of each task (args.yaml), reusing the rows already in the inventory unless rescan is set,
       and recompute the validity of every run. Returns the list of rows."""
    known = {}
    if os.path.exists(inventory_file) and not rescan:
        with open(inventory_file, newline='', encoding='utf-8') as file:
            for row in csv.DictReader(file):
                row['final_eval'] = row['final_eval'] == 'True'
                known[row['run']] = row

    rows = []
    for _, experiment_output_path, tag in variance_tasks:
        for dirpath, dirnames, filenames in os.walk(experiment_output_path):
            if 'args.yaml' not in filenames:
                continue
            run = os.path.relpath(dirpath, yolo_root)
            rows.append(known[run] if run in known else scanYoloRun(dirpath, tag, yolo_root))

    seen_seeds = set()
    for row in sorted(rows, key=lambda row: (row['tag'], row['start'], row['run'])):
        row['seed1'] = row['args_seed'] == '1'
        row['dup_seed'] = row['final_eval'] and not row['seed1'] and (row['tag'], row['args_seed']) in seen_seeds
        row['valid'] = row['final_eval'] and not row['seed1'] and not row['dup_seed']
        if row['final_eval'] and not row['seed1']:
            seen_seeds.add((row['tag'], row['args_seed']))

    rows = sorted(rows, key=lambda row: (row['tag'], row['start'], row['run']))
    with open(inventory_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=inventory_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in inventory_fields} for row in rows])

    for tag in dict.fromkeys(row['tag'] for row in rows):
        tag_rows = [row for row in rows if row['tag'] == tag]
        n_valid = sum(row['valid'] for row in tag_rows)
        log(f"[updateYoloRunsInventory] {tag}: {n_valid} valid of {len(tag_rows)} runs "
            f"(seed 1: {sum(row['seed1'] for row in tag_rows)}, repeated seed: {sum(row['dup_seed'] for row in tag_rows)}, "
            f"no final evaluation: {sum(not row['final_eval'] for row in tag_rows)})",
            bcolors.OKBLUE if n_valid else bcolors.WARNING)
    return rows


def getValidYoloRuns(inventory_rows, yolo_root):
    """Set of normalized absolute paths of the valid runs."""
    return {os.path.normpath(os.path.join(yolo_root, row['run'])) for row in inventory_rows if row['valid']}
