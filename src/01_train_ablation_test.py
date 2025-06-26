#!/usr/bin/env python3
# encoding: utf-8

"""
    Evaluates a model under different initialization conditions. For each condition a hiper-parameter is changed, so that
    under the same condition one parameter changed to check if distances are invariant between different initializations.
    CNN_14L model is used and batch size and learning rate is modified from the default version.
"""

import os
import sys 
import re
import random
import traceback 
import copy
import itertools

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.log_utils import log, logTable
from utils.yaml_utils import updateMetricsLogFile, getMetricsLogFile
from utils.set_seed import set_seed
from models import CNN_14L

# Dummy class for better logging
class CNN_14L_Ablation(CNN_14L):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_14L_Ablation, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)
        
# How many train loops are executed to study its variance
current_dir_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir_path,"../output_data")
total_iterations = 1

# General configuration
input_size = 28 * 28  # Size of each image flattened
num_classes = 10  # Numbers from 0 to 9
num_epochs = 500
patience=10

# Default has batch_size = 64
batch_conditions = [10,40,70]
learning_rates = [0.001,0.005]

if __name__ == "__main__":
    seed = 0
    metrics = []
    
    updated = True # Flag to control out of loop
    exception_messages = ""

    metrics_log_file = os.path.join(output_path, 'CNN_14L_Ablation', f"training_metrics.yaml")
    metrics = getMetricsLogFile(metrics_log_file)
            
    # Load MNIST dataset just once
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    while updated:
        try: 
            seed = random.randint(0, 2**32 - 1)  # Random 32 bits number
            set_seed(seed)

            # Init model, loss and optimizer
            model = CNN_14L_Ablation(input_size=input_size, num_classes=num_classes, learning_rate=learning_rates[0], patience=patience, seed=seed, output_path=output_path)
            model.save_architecture()

            updated = False
            iteration = sum(1 for key in metrics.keys() if 'iteration_' in key)
            if not iteration < total_iterations:
                break

            for batch_size, learning_rate in itertools.product(batch_conditions, learning_rates):
                train_model = copy.deepcopy(model)

                train_model.batch_size = batch_size  
                train_model.learning_rate = learning_rate  
                train_model.patience = patience  
                
                train_loader = DataLoader(dataset=train_dataset, batch_size=train_model.batch_size, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=train_model.batch_size, shuffle=False) 
                
                log(f"[CNN_14L_Ablation] Iteration: {iteration}/{total_iterations}")
                
                metrics = train_model.spinTrainEval(train_loader, test_loader, num_epochs = num_epochs)

                metrics.update({'batch_size': batch_size})
                metrics = { f'Key format': '[batch_size, learning_rate, patience]',
                            f'iteration_{iteration}': {[batch_size,learning_rate,patience]: metrics}}
                updateMetricsLogFile(metrics, metrics_log_file)
                updated = True

        except Exception as e:
            exception_message = f"Exception catched: {e}: \n{traceback.format_exc()}"
            exception_messages += f"{exception_message}\n"
            print(f"[ERROR] --- \n[ERROR] --- \n[ERROR] CATCHED EXCEPTION: {exception_message}. \n")

            # traceback.print_exception() 
            # traceback.print_exception(*sys.exc_info()) 
            print(F"[ERROR] --- \n[ERROR] --- \n")

    print(f"Finished all iterations configured for all models")
    if exception_messages != "": print(f"[ERROR] Exceptions catched during execution: {exception_messages}")