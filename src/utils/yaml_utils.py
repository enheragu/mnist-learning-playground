#!/usr/bin/env python3
# encoding: utf-8

import os
import yaml

from utils.file_lock import FileLock

# Check if all values in the dictionary/list are of basic types
def is_basic_types(values):
    basic_types = (int, str, bool, float)
    return all(isinstance(value, basic_types) for value in values)

# Custom representation function for dictionaries
def represent_dict(dumper, data):
    if is_basic_types(data.values()):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data)
    else:
        return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)

# Custom representation function for lists
def represent_list(dumper, data):
    if is_basic_types(data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    else:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)




def updateMetricsLogFile(metrics, file_path="training_metrics.yaml"):
    lock_file = f"{file_path}.lock"

    with FileLock(lock_file) as lock1:
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                all_metrics = yaml.safe_load(file) or {}
        else:
            all_metrics = {}

        all_metrics.update(metrics)

    with FileLock(lock_file) as lock1:
        with open(file_path, "w") as file:
            # Add custom representation functions to the YAML dumper
            yaml.add_representer(list, represent_list)
            yaml.add_representer(dict, represent_dict)
            yaml.dump(all_metrics, file, default_flow_style=False, allow_unicode=True, indent=4, width=5000)


        print(f"Updated metrics file in: {file_path}")


def getMetricsLogFile(file_path="training_metrics.yaml"):
    lock_file = f"{file_path}.lock"

    if os.path.exists(file_path):
        with FileLock(lock_file) as lock1:
            with open(file_path, "r") as file:
                all_metrics = yaml.safe_load(file) or {}
                return all_metrics
    print(f"[Error] File not found: {file_path}")
    return {}