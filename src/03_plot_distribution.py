#!/usr/bin/env python3
# encoding: utf-8

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from utils.log_utils import log, c_blue, c_green, c_yellow, c_red, c_purple, c_grey, c_darkgrey, color_palette_list
from utils import getAllModelData, getAblationModelData, getAllAndAblationModelData
from utils.plot_distribution import plotDataDistribution, only_store, plot_metric_distribution, plot_survival_function
from utils import output_path, ablation_data_file
from utils.distribution_analysis import normalityTest, maxAmplitude, count_trials, gamma_amplitude

analysis_path = './analysis_results/distributions'

PLOT_AMPLITUDE_RELATION = False
COMPUTE_ANALYSIS_METRICS = False
PLOT_DISTRIBUTIONS = True
PLOT_EXAMPLE_DISTRIBUTIONS = False
COMPUTE_SURVIVAL_FUNCTION = False


"""
    Just plots the amplitude of the distribution against the number of params
    to see if they somehow relate
"""
def plotParamAmplitudeRelation(metrics_data, plot_models = 'all', title_tag=''):
    import models
    # from train_models import input_size, num_classes, learning_rate, patience
    # General configuration taken from 00_train_models.py that cannot be imported as such
    input_size = 28 * 28  # Size of each image flattened
    num_classes = 10  # Numbers from 0 to 9
    learning_rate = 0.001
    patience = 10

    y = []
    x = []
    labels = []
    for model_name, data in metrics_data.items():
        if plot_models != 'all' and model_name not in plot_models:
            continue

        # Try to get the model class
        try:
            object_class = getattr(models, model_name)
            model = object_class(input_size=input_size, num_classes=num_classes, learning_rate=learning_rate, patience=patience, output_path=None)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except AttributeError:
            # Handle ablation model names like 'CNN_14L_B10_L0.001'
            # Extract base model name and parameters
            if 'CNN_14L' in model_name and '_B' in model_name and '_L' in model_name:
                try:
                    # Parse format: CNN_14L_B{batch_size}_L{learning_rate}
                    parts = model_name.split('_')
                    base_model_name = 'CNN_14L'
                    # Find B and L values
                    batch_size_str = [p for p in parts if p.startswith('B')][0][1:]  # Remove 'B' prefix
                    learning_rate_val = float([p for p in parts if p.startswith('L')][0][1:])  # Remove 'L' prefix
                    
                    object_class = getattr(models, base_model_name)
                    model = object_class(input_size=input_size, num_classes=num_classes, learning_rate=learning_rate_val, patience=patience, output_path=None)
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                except Exception as e:
                    log(f"Could not parse ablation model {model_name}: {e}", c_red)
                    continue
            else:
                log(f"Could not find model class for {model_name}", c_red)
                continue
        
        accuracy_data = [entry['accuracy']*100 for entry in data.values()]
        amplitude = np.max(accuracy_data) - np.min(accuracy_data)
        x.append(trainable_params)
        y.append(amplitude)
        labels.append(model_name)

    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, zorder=3)

    # Posicionar etiquetas dinámicamente para evitar solapamientos en X e Y
    x_range = np.max(x) - np.min(x)
    x_middle = x_range / 2 + np.min(x)
    y_range = np.max(y) - np.min(y)
    proximity_threshold = y_range * 0.02  # Umbral: 2% del rango Y
    
    # Crear índices ordenados por Y para detectar puntos cercanos
    y_indices = np.argsort(y)
    
    # Determinar lado para cada punto considerando proximidad a otros
    sides = {}  # Mapeo de índice a 'left' o 'right'
    for i in range(len(labels)):
        idx = y_indices[i]
        
        # Determinar lado inicial basado en X
        if x[idx] < x_middle:
            initial_side = 'left'
        else:
            initial_side = 'right'
        
        # Verificar si hay puntos muy cercanos en Y
        nearby_indices = [j for j in range(len(labels)) if abs(y[j] - y[idx]) < proximity_threshold and j != idx]
        
        if nearby_indices:
            # Si hay puntos cercanos, alternar lado para uno de ellos
            nearby_sides = [sides.get(j, 'unknown') for j in nearby_indices if j in sides]
            if 'left' in nearby_sides:
                # Si alguno va a izquierda, este va a derecha
                sides[idx] = 'right'
            elif 'right' in nearby_sides:
                # Si alguno va a derecha, este va a izquierda
                sides[idx] = 'left'
            else:
                # Ambos nuevos: asignar al lado opuesto del primero
                sides[idx] = 'right' if initial_side == 'left' else 'left'
        else:
            sides[idx] = initial_side
    
    # Dibujar etiquetas con los lados determinados
    # Usar offsets en puntos (independientes de la escala de datos) con annotate
    ax = plt.gca()
    for i, label in enumerate(labels):
        ha = sides[i]
        # Detectar si es nombre de ablation (tiene _B y _L)
        is_ablation = ('_B' in label and '_L' in label)

        # Offset en puntos (px). Cortos cerca, ablation más lejos.
        base_dx = 10 if is_ablation else 7
        dx = base_dx if ha == 'left' else -base_dx

        # Si hay puntos muy cercanos en Y, desplazar también en Y en puntos para evitar solapamiento
        # Contar cuántos puntos están cerca en Y por encima del umbral
        nearby_count = sum(1 for j in range(len(labels)) if j != i and abs(y[j] - y[i]) < proximity_threshold)
        # Alternar desplazamiento vertical según índice para separar visualmente
        if nearby_count > 0:
            # small vertical offset per nearby point
            dy = (i % 2) * 6  # 0 or 6 points
        else:
            dy = 0

        ax.annotate(
            label,
            xy=(x[i], y[i]),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=9,
            ha=ha,
            va='center',
            zorder=4,
        )

    plt.title(r'Accuracy amplitude $\mathit{vs}.$ Trainable Params')
    plt.ylabel('Accuracy amplitude')
    plt.xlabel('Trainable params (N)')

    # Mostrar la gráfica con grid atrás
    plt.grid(True, zorder=0)
    plt.savefig(os.path.join(analysis_path,f"amplitude_relation_{title_tag}.pdf"), format="pdf")

    if only_store:
        plt.close()


"""
    Penalizes the proximity of elements in a subset to get a distribution with more spread data
"""
def proximity_penalty(subset, min_dist):
    subset = np.sort(subset)
    penalty = 0
    for i in range(len(subset)):
        for j in range(i+1, len(subset)):
            if abs(subset[j] - subset[i]) < min_dist:
                penalty += 1
    return penalty

def biased_resample(data, target_mean, target_std, subset_size, max_iter=30000, 
                    alpha_std = 3.0, alpha_proximity=4.0, min_dist_factor=20):
    data = np.array(data)
    n = len(data)
    distances = np.diff(np.sort(data))
    original_min_distance = np.min(distances[distances > 0])
    min_dist = original_min_distance * min_dist_factor
    # log(f"Original min distance: {original_min_distance:.4f}, using min_dist={min_dist:.4f} and alpha={alpha} for proximity penalty")

    # Random subset at init
    index = np.random.choice(n, subset_size, replace=False)
    subset = data[index]

    best_score = (
        abs(np.mean(subset) - target_mean) +
        alpha_std * abs(np.std(subset, ddof=1) - target_std) +
        alpha_proximity * proximity_penalty(subset, min_dist)
    )

    for _ in range(max_iter):
        current_mean = np.mean(subset)
        current_std = np.std(subset, ddof=1)
        out_candidates = np.setdiff1d(np.arange(n), index)
        if len(out_candidates) == 0:
            break

        # Choose based on the current mean and std
        if abs(current_mean - target_mean) > abs(current_std - target_std):
            # Changes min or max value depending on the average
            if current_mean < target_mean:
                in_pos = np.argmin(subset)
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmax(data[out_candidates])]
            else:
                in_pos = np.argmax(subset)
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmin(data[out_candidates])]
        else:
            # Changes closest or farthest value depending on the std
            if current_std < target_std:
                mean_val = np.mean(subset)
                in_pos = np.argmin(np.abs(subset - mean_val))
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmax(np.abs(data[out_candidates] - target_mean))]
            else:
                mean_val = np.mean(subset)
                in_pos = np.argmax(np.abs(subset - mean_val))
                in_idx = index[in_pos]
                out_idx = out_candidates[np.argmin(np.abs(data[out_candidates] - target_mean))]

        # Recompute score
        new_index = index.copy()
        new_index[np.where(index == in_idx)[0][0]] = out_idx
        new_subset = data[new_index]
        new_score = (
            abs(np.mean(new_subset) - target_mean) +
            alpha_std * abs(np.std(new_subset, ddof=1) - target_std) +
            alpha_proximity * proximity_penalty(new_subset, min_dist)
        )

        # If its better keep it :)
        if new_score < best_score:
            index = new_index
            subset = new_subset
            best_score = new_score

    return data[index]


"""
    Plots biaserd distributions (same normal with mean and std) with all data centered,
    or no data at right side or left side
"""
def plot_example_distributions(metrics_data, new_sample_size=50, analysis_path=analysis_path, vertical_lines_acc=[99.57],model='CNN_14L'):
    
    data = np.array([entry['accuracy']*100 for entry in metrics_data[model].values()])
    mean = np.mean(data)
    std = np.std(data)
    log(f"[plot_example_distributions] Original data mean: {mean:.2f}, std: {std:.2f}. Max: {np.max(data):.2f}, Min: {np.min(data):.2f}, N: {len(data)}")

    # Resamples giving different probability to be chose in resampling based on value position
    data_right_bias = data[data < (mean + std*0.5)]
    resampled_right_bias = biased_resample(data_right_bias, target_mean=mean, target_std=std, subset_size=new_sample_size)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_right_bias):.2f}, std: {np.std(resampled_right_bias):.2f}. Max: {np.max(data_right_bias):.2f}, Min: {np.min(data_right_bias):.2f}, N: {len(data_right_bias)}")
    plot_metric_distribution({model: resampled_right_bias}, metric_label='Accuracy (%)', color_palette=[c_red], vertical_lines_acc=vertical_lines_acc, analysis_path=analysis_path, plot_filename="example_CNN_14L_right_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with right bias")

    data_centered_bias = data[(data < (mean + std*1.4)) & (data > (mean - std*1.4))]
    resampled_centered_bias = biased_resample(data_centered_bias, target_mean=mean, target_std=std, subset_size=new_sample_size, alpha_std=1.0)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_centered_bias):.2f}, std: {np.std(resampled_centered_bias):.2f}. Max: {np.max(data_centered_bias):.2f}, Min: {np.min(data_centered_bias):.2f}, N: {len(data_centered_bias)}")
    plot_metric_distribution({model: resampled_centered_bias}, metric_label='Accuracy (%)', color_palette=[c_yellow], analysis_path=analysis_path, plot_filename="example_CNN_14L_center_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with centered bias")
    
    data_left_bias = data[data > (mean - std*0.5)]
    resampled_left_bias = biased_resample(data_left_bias, target_mean=mean, target_std=std, subset_size=new_sample_size)
    log(f"[plot_example_distributions] Resampled data mean: {np.mean(resampled_left_bias):.2f}, std: {np.std(resampled_left_bias):.2f}. Max: {np.max(data_left_bias):.2f}, Min: {np.min(data_left_bias):.2f}, N: {len(data_left_bias)}")
    plot_metric_distribution({model: resampled_left_bias}, metric_label='Accuracy (%)', color_palette=[c_green], analysis_path=analysis_path, plot_filename="example_CNN_14L_left_biased", plot_mean=mean, plot_std=std)
    log(f"[plot_example_distributions] Plot resampled distribution with left bias")
    

if __name__ == "__main__":
    os.makedirs(f"{analysis_path}/tables", exist_ok=True)
    os.makedirs(os.path.join(analysis_path, 'single_model'), exist_ok=True)
    plt.rcParams.update({'font.size': 18})
    metrics_data, ablation_metrics = getAllAndAblationModelData(output_path, ablation_data_file)

    all_models = metrics_data.keys()
    ablatipon_models = ablation_metrics.keys()
    log(f"Model availability: {all_models}")
    log(f"Model availability ablation tests: {ablatipon_models}")
    # log(f"{metrics_data = }")
    
    if ablation_metrics:
        if PLOT_AMPLITUDE_RELATION:
            plotParamAmplitudeRelation(ablation_metrics, plot_models=ablatipon_models, title_tag='ablation')
       
        if PLOT_DISTRIBUTIONS:
            plotDataDistribution(metrics_data=ablation_metrics,
                                models_plot_list=[ablatipon_models],
                                color_list=[color_palette_list],
                                analysis_path=analysis_path)
            plotDataDistribution(metrics_data=ablation_metrics,
                                models_plot_list=[ablatipon_models],
                                color_list=[color_palette_list],
                                analysis_path=analysis_path,
                                show_histogram=False,
                                plot_filename='plot_accuracy_ablation')
        if COMPUTE_ANALYSIS_METRICS:
            normalityTest(metrics_data=ablation_metrics, title_tag='ablation', analysis_path=analysis_path)
            maxAmplitude(metrics_data=ablation_metrics, title_tag='ablation', analysis_path=analysis_path)
            maxAmplitude(metrics_data=ablation_metrics, metric='train_duration', unit=' (s)', unit_multiplier=1, format='.1f', amplitude_function=gamma_amplitude, title_tag='ablation', analysis_path=analysis_path)
            maxAmplitude(metrics_data=ablation_metrics, metric='best_epoch', unit='', unit_multiplier=1, format='.0f', amplitude_function=gamma_amplitude, title_tag='ablation', analysis_path=analysis_path)

            count_trials(metrics_data=ablation_metrics, title_tag='ablation', analysis_path=analysis_path)

    # Once all models' metrics have been gathered, plot the distributions
    if metrics_data:

        all_models_no_overfit = [ # ignore overfited models and repeated models for this analysis
            'BatchNormMaxoutNetInNet',
            'CNN_14L',
            # 'CNN_14L_B10',
            # 'CNN_14L_B25',
            # 'CNN_14L_B50',
            # 'CNN_14L_B80',
            # 'CNN_14L_overfit_0.02',
            'CNN_3L',
            'CNN_4L',
            'CNN_5L',
            'DNN_6L',
            # 'DNN_6L_overfit_0.02',
            'HiddenLayerPerceptron',
            'SimplePerceptron']
        
        if PLOT_AMPLITUDE_RELATION:
            plotParamAmplitudeRelation(metrics_data, plot_models=all_models_no_overfit)
        
        if PLOT_DISTRIBUTIONS:
            plotDataDistribution(metrics_data=metrics_data,
                                models_plot_list=[['SimplePerceptron'],
                                ['CNN_14L'],
                                ['DNN_6L', 'HiddenLayerPerceptron'],
                                ['CNN_3L', 'CNN_4L', 'CNN_5L', 'CNN_14L'],
                                ['CNN_14L', 'CNN_14L_B10', 'CNN_14L_B25', 'CNN_14L_B50'],
                                ['CNN_14L_overfit_0.02'], 
                                ['DNN_6L'], 
                                ['DNN_6L_overfit_0.02'],
                                #    all_models],
                                ],
                                color_list=[[c_green],
                                [c_purple],
                                [c_blue,c_darkgrey],
                                [c_yellow, c_grey, c_red, c_purple], 
                                [c_purple, c_yellow, c_red, c_grey],
                                [c_blue],
                                [c_grey],
                                [c_darkgrey],
                                #    color_palette_list],
                                ],
                                analysis_path = analysis_path)
        
        if COMPUTE_ANALYSIS_METRICS:
            normalityTest(metrics_data=metrics_data, analysis_path=analysis_path)
            maxAmplitude(metrics_data=metrics_data, analysis_path=analysis_path)
            
            maxAmplitude(metrics_data=metrics_data, metric='train_duration', unit=' (s)', unit_multiplier=1, format='.1f', amplitude_function=gamma_amplitude, analysis_path=analysis_path)
            maxAmplitude(metrics_data=metrics_data, metric='best_epoch', unit='', unit_multiplier=1, format='.0f', amplitude_function=gamma_amplitude, analysis_path=analysis_path)

            count_trials(metrics_data=metrics_data, analysis_path=analysis_path)


        if PLOT_EXAMPLE_DISTRIBUTIONS:
            plot_example_distributions(metrics_data=metrics_data, new_sample_size=60, analysis_path=analysis_path)

        print(f"Search Juyang (John) Weng for info about initialization bias and similar stuff https://www.google.com/search?client=ubuntu&channel=fs&q=Juyang+%28John%29+Weng")

    

    combined_models = ablation_metrics.copy()
    combined_models.update(metrics_data)

    if COMPUTE_SURVIVAL_FUNCTION:
        exceedance_metric_data = {
            model_name: [entry['accuracy'] * 100 for entry in model_data.values()]
            for model_name, model_data in combined_models.items()
            if isinstance(model_data, dict) and len(model_data) > 0
        }
        if exceedance_metric_data:
            plot_survival_function(
                metrics_data=exceedance_metric_data,
                analysis_path=analysis_path,
                metric_label='Accuracy (%)',
                table_filename='Survival Function Key Percentiles Combined Models',
                plot_prefix='survival_combined',
                legend_ncol=2,
            )
            log("Survival function plot generated", c_green)
        else:
            log("Skipping survival function: no valid data", c_yellow)
    else:
        log("Skipping survival function computation...", c_yellow)
    
    if not only_store:
        plt.show()