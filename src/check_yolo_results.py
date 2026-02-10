#!/usr/bin/env python3
# encoding: utf-8

import os
import itertools
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, shapiro, kurtosis

from utils.plot_distribution import plot_metric_distribution, plot_metric_gammadistribution
from utils.yaml_utils import getMetricsLogFile, updateMetricsLogFile, print_yaml_structure, print_dict_keys
from utils.log_utils import log, logTable, bcolors, color_palette_list

from utils import output_path

analysis_path = './analysis_results/yolo_analysis'
output_path = os.path.join(output_path, "YOLO")
parse_from_scratch = False

"""
    Recursively filters a dict and returns only branches that contain provided keys
"""
def filter_dict_by_keys(data_dict, target_keys):
    if isinstance(data_dict, dict):
        direct_matches = {key: value for key, value in data_dict.items() if key in target_keys}
        
        nested_matches = {key: filter_dict_by_keys(value, target_keys) 
                          for key, value in data_dict.items() if key not in direct_matches}
        
        combined = {**direct_matches, **{key: value for key, value in nested_matches.items() if value}}
        return combined if combined else None
    
    elif isinstance(data_dict, list):
        filtered_list = [filter_dict_by_keys(item, target_keys) for item in data_dict]
        return [item for item in filtered_list if item]
    
    return None

"""
    Combines data from differents dict into a list (if keys match)
"""
def combine_dicts(dicts):
    combined = defaultdict(list)

    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict):
                sub_dicts = [c.get(key, {}) for c in dicts if key in c]
                combined[key] = combine_dicts(sub_dicts)
            else:
                if isinstance(value, list):
                    combined[key].extend(value)
                else:
                    combined[key].append(value)

    return dict(combined)

def getClassId(names_dict, class_name):
    for key, values in names_dict.items():
        if class_name in values:
            return key

    return None

def process_dir(args):
    dirpath, filenames, parse_from_scratch = args
    results = []
    try:
        if "results.yaml.cached" in filenames and not parse_from_scratch:
            file_path = os.path.join(dirpath, "results.yaml.cached")
            log(f"[process_dir] Parse cached data from {file_path}")
            data = getMetricsLogFile(file_path)
            results.append(data)
        elif "results.yaml" in filenames and 'predictions.json' in filenames:
            file_path = os.path.join(dirpath, "results.yaml")
            args_file = os.path.join(dirpath, "args.yaml")
            log(f"[process_dir] Parsing data from {file_path}")
            data = getMetricsLogFile(file_path)
            log(f"[process_dir] Parsed results.yaml file.")
            data_args = getMetricsLogFile(args_file)
            log(f"[process_dir] Parsed args.yaml file.")
            data = filter_dict_by_keys(data, ['P', 'R', 'mAP50', 'mAP50-95', 'names', 'train_data', 'rgb_equalization', 'seed', 'thermal_equalization', 'train_duration_h', 'epoch_best_fit_index'])
            data['batch'] = data_args.get('batch', None)
            updateMetricsLogFile(data, f"{file_path}.cached")
            results.append(data)
    except Exception as e:
        log(f"[process_dir] Exception catched: {e}", bcolors.ERROR)
        raise
    return results

def getYOLOVarianceData(analysis_path, output_path, test_tag=""):
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)
    
    if 'summary_variance_data.yaml' in os.listdir(output_path) and not parse_from_scratch:
        data_combined = getMetricsLogFile(os.path.join(output_path, 'summary_variance_data.yaml'))
    else:
        data_combined = []
        tasks = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            for dirpath, dirnames, filenames in os.walk(output_path):
                tasks.append(executor.submit(process_dir, (dirpath, filenames, parse_from_scratch)))
            for future in as_completed(tasks):
                try:
                    result = future.result()
                    if result:
                        data_combined.extend(result)
                except Exception as e:
                    log(f"[getYOLOVarianceData] Exception catched: {e}", bcolors.ERROR)

        data_combined = combine_dicts(data_combined)
        # print(f"After combination:\n{data_combined}")
        updateMetricsLogFile(data_combined, os.path.join(output_path, 'summary_variance_data.yaml'))
    
    n_data = 0
    headers = ['Id', 'Class', 'mAP50 (mean)', 'mAP50 (std)', 'mAP50-95 (mean)', 'mAP50-95 (std)']
    row_data = []
    all_row = []
    for index, (class_key, data) in enumerate(data_combined['validation_0']['data'].items()):
        class_id = getClassId(data_combined['dataset_info']['names'],class_key)
        n_data = len(data['mAP50'])
        mean1 = np.mean(data['mAP50'])
        std1 = np.std(data['mAP50'])
        mean2 = np.mean(data['mAP50-95'])
        std2 = np.std(data['mAP50-95'])
        row = [class_id, class_key,mean1,std1,mean2,std2]
        if class_key == "all":
            all_row = [0] + row[1:] # Just to order it at the end
            continue
        row_data.append(row)
    

    data_array = np.array(row_data)
    numerical_data = data_array[:, 2:].astype(float)
    column_means = np.mean(numerical_data, axis=0)
    column_stds = np.std(numerical_data, axis=0)
    summary_row_mean = ['1','Summary (Mean each col)'] + column_means.tolist()
    summary_row_std = ['2','Summary (Std each col)'] + column_stds.tolist()

    
    sorted_rows = sorted(row_data, key=lambda x: x[0])
    row_data = [headers] + sorted_rows
    logTable(row_data, f"{analysis_path}/tables", f"Variance of YOLO with {test_tag} with {n_data} trials.", 
             colalign=['left', 'right', 'right', 'right', 'right'])

    summary_data = [headers]
    summary_data.extend([summary_row_mean,summary_row_std,all_row])
    logTable(summary_data, f"{analysis_path}/tables", f"Summary variance of YOLO with {test_tag} with {n_data} trials.", 
             colalign=['left', 'right', 'right', 'right', 'right'])
    
    log(f"[getYOLOVarianceData] Finished processing data from {output_path}")
    return {test_tag: data_combined}

def plotYOLODistribution(data_plot, analysis_path, class_tag = 'all', metric_name = 'mAP50', tag_name = ''):
        data = {}
        for key, data_items in data_plot.items():
            data[key] = data_items['validation_0']['data'][class_tag][metric_name]

        plot_metric_distribution(data, train_duration_data = None, metric_label = f'{metric_name} {tag_name}',
                                color_palette = color_palette_list, vertical_lines_acc = [], analysis_path = analysis_path,
                                plot_filename = f'{tag_name}_{metric_name}')
        
def plotYOLOTrainDurationDistribution(data_plot, analysis_path, tag_name = ''):
    data_epoch = {}
    data_time = {}
    for key, data_items in data_plot.items():
        data_epoch[key] = [item for item in data_items['train_data']['epoch_best_fit_index']] # To seconds
        data_time[key] = [item*60 for item in data_items['train_data']['train_duration_h']] # To seconds
        
        # data_epoch[key].remove(max(data_epoch[key])) # Keep only 40 samples..
        # data_time[key].remove(max(data_time[key]))

    plot_metric_distribution(data_epoch, train_duration_data = None, metric_label = 'Best Epoch', plot_func=plot_metric_gammadistribution, 
                             color_palette=color_palette_list, vertical_lines_acc=[], analysis_path=analysis_path,
                             plot_filename = f'plot_best_epoch_{tag_name}',
                             bin_size=30)
    plot_metric_distribution(data_time, train_duration_data = None, metric_label = 'Train duration (min)', plot_func=plot_metric_gammadistribution, 
                             color_palette=color_palette_list, vertical_lines_acc=[], analysis_path=analysis_path,
                             plot_filename = f'plot_time_duration_{tag_name}',
                             bin_size=30)
if __name__ == "__main__":
    if parse_from_scratch:
        log(f"[main] Parsing all data from scratch and regenerate cached files", bcolors.OKGREEN)
    else:
        log(f"[main] Data is parsed from cached files without updating it", bcolors.WARNING)
    
    tasks = [
        # Output Path, Data Path, Tag name :)
        # (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'coco'), 'COCO')#,
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_day_rgbt'), 'kaist_day_rgbt'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths/variance_vths_llvip_no_equalization'), 'LLVIP_no_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths/variance_vths_llvip_rgb_th_equalization'), 'LLVIP_rgb_th_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths/variance_vths_llvip_th_equalization'), 'LLVIP_th_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_vths/variance_vths_llvip_rgb_equalization'), 'LLVIP_rgb_equalization'),
        (os.path.join(analysis_path,'tables'), os.path.join(output_path, 'variance_llvip_night_lwir'), 'LLVIP_lwir')
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(getYOLOVarianceData, *task) for task in tasks]
        for future in as_completed(futures):
            try:
                results.update(future.result())
            except Exception as e:
                log(f"[main] Exception in future: {e}", bcolors.ERROR)
    
    data_plot = {}
    for key in ['LLVIP_no_equalization',
                'LLVIP_rgb_th_equalization',
                'LLVIP_rgb_equalization',
                'LLVIP_th_equalization'
                ]:
        data_plot[key] = results[key]
    plotYOLODistribution(data_plot, analysis_path, metric_name = 'mAP50', tag_name = 'LLVIP_vths_equalization')
    plotYOLODistribution(data_plot, analysis_path, metric_name = 'mAP50-95', tag_name = 'LLVIP_vths_equalization')
    plotYOLODistribution({'kaist_day_rgbt': results['kaist_day_rgbt']}, analysis_path, metric_name = 'mAP50', tag_name = 'kaist_day_rgbt')
    plotYOLOTrainDurationDistribution({'kaist_day_rgbt': results['kaist_day_rgbt']}, analysis_path, tag_name = 'kaist_day_rgbt')

    plotYOLODistribution({'LLVIP_lwir': results['LLVIP_lwir']}, analysis_path, metric_name = 'mAP50', tag_name = 'LLVIP_lwir')
    plotYOLOTrainDurationDistribution({'LLVIP_lwir': results['LLVIP_lwir']}, analysis_path, tag_name = 'LLVIP_lwir')
