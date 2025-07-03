#!/usr/bin/env python3
# encoding: utf-8

"""
    Evaluates a model under different initialization conditions. For each condition a hiper-parameter is changed, so that
    under the same condition one parameter changed to check if distances are invariant between different initializations.
    CNN_14L model is used and batch size and learning rate is modified from the default version.
"""

## WHEN CODE IS CHANGED PLEASE CONFIRM REPETIBILITY WR TO PREVOUS TESTS IF THEY NEED TO BE COMPARED

import os
import sys 
import re
import random
import traceback 
import copy
import itertools

import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.log_utils import log, logTable, bcolors
from utils.yaml_utils import updateMetricsLogFile, getMetricsLogFile
from utils.set_seed import set_seed
from utils import output_path
from models import CNN_14L

# Dummy class for better logging
class CNN_14L_Ablation(CNN_14L):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_14L_Ablation, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)

metrics_log_file = os.path.join(output_path, 'CNN_14L_Ablation', f"training_metrics.yaml")

# How many train loops are executed to study its variance
total_iterations = 298

# General configuration
input_size = 28 * 28  # Size of each image flattened
num_classes = 10  # Numbers from 0 to 9
num_epochs = 100
patience=10

# Default has batch_size = 64
batch_conditions = [10,40,70]
learning_rates = [0.01,0.001,0.005]

def trainModel(model, batch_size, learning_rate, train_dataset, test_dataset, seed):
    set_seed(seed)
    train_model = copy.deepcopy(model)
    train_model.batch_size = batch_size  
    train_model.learning_rate = learning_rate  
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) 

    metrics = train_model.spinTrainEval(train_loader, test_loader, num_epochs = num_epochs)
    
    date_tag = next(iter(metrics))
    data = metrics[date_tag]
    data.update({'batch_size': batch_size, 'learning_rate': learning_rate, 'finish_timetag': date_tag})
    key = f"{batch_size}-{learning_rate}"
    return {key: data}

def runAblationTests(batch_conditions, learning_rates, metrics_log_file, train_dataset, test_dataset):
    seed = 0
    while True:
        metrics = getMetricsLogFile(metrics_log_file)
        iteration = sum(1 for key in metrics.keys() if 'iteration_' in key)
        if not iteration < total_iterations:
            break

        seed = int(random.randint(0, 2**32 - 1))  # Random 32 bits number
        set_seed(seed)

        # Init model, loss and optimizer
        model = CNN_14L_Ablation(input_size=input_size, num_classes=num_classes, learning_rate=learning_rates[0], patience=patience, seed=seed, output_path=output_path)
        model.save_architecture()

        for batch_size, learning_rate in itertools.product(batch_conditions, learning_rates):
            data = trainModel(model, batch_size, learning_rate, train_dataset, test_dataset, seed)
            log(f"[CNN_14L_Ablation] Iteration: {iteration}/{total_iterations} ({batch_size = }, {learning_rate = })")
            new_metrics = {f"{os.getpid()}_iteration_{seed}": data}
            updateMetricsLogFile(new_metrics, metrics_log_file)

    log(f"Finished all iterations configured for all models", color=bcolors.OKGREEN)

def completeAblationTests(batch_conditions, learning_rates, metrics_log_file, train_dataset, test_dataset):
    metrics = getMetricsLogFile(metrics_log_file)

    all_conditions = [f"{batch}-{lr}" for batch, lr in itertools.product(batch_conditions, learning_rates)]
    for key, iteration in metrics.items():
        missing_conditions = copy.deepcopy(all_conditions)
        for trial_idx, (condition, trial) in enumerate(iteration.items()):
            if condition in missing_conditions:
                missing_conditions.remove(condition)

        log(f"For {key} test there are {missing_conditions} tests mising")
        for condition in missing_conditions:
            log(f"Complete {key} -> {condition}")
            batch_size = int(condition.split('-')[0])
            learning_rate = float(condition.split('-')[1])
            seed = int(key.split('_')[2])
            set_seed(seed)

            model = CNN_14L_Ablation(input_size=input_size, num_classes=num_classes, learning_rate=learning_rates[0], patience=patience, seed=seed, output_path=output_path)
            model.save_architecture()
            
            data = trainModel(model, batch_size, learning_rate, train_dataset, test_dataset, seed)
            new_metrics = {key: data}
            log(f"Completed {key} -> {condition}")
            updateMetricsLogFile(new_metrics, metrics_log_file)

    log(f"Completed all missing tests", color=bcolors.OKGREEN)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Choose which test functions to run.")
    parser.add_argument('-r', '--run', action='store_true', help="Run the main tests")
    parser.add_argument('-c', '--complete', action='store_true', help="Run the complete tests")
    args = parser.parse_args()


    # Load MNIST dataset just once
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    exception_messages = ""
    try: 
        if (not args.run and not args.complete) or args.run:
            runAblationTests(batch_conditions=batch_conditions, 
                            learning_rates=learning_rates,
                            metrics_log_file=metrics_log_file,
                            train_dataset =train_dataset,
                            test_dataset =test_dataset)
            
        if args.complete:
            completeAblationTests(batch_conditions = batch_conditions, 
                            learning_rates = learning_rates, 
                            metrics_log_file = metrics_log_file, 
                            train_dataset = train_dataset, 
                            test_dataset = test_dataset)

    except Exception as e:
        exception_message = f"Exception catched: {e}: \n{traceback.format_exc()}"
        exception_messages += f"{exception_message}\n"
        log(f"[ERROR] --- \n[ERROR] --- \n[ERROR] CATCHED EXCEPTION: {exception_message}. \n", color=bcolors.ERROR)
        log(F"[ERROR] --- \n[ERROR] --- \n", color=bcolors.ERROR)

    if exception_messages != "": log(f"[ERROR] Exceptions catched during execution: {exception_messages}", color=bcolors.ERROR)