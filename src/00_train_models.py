#!/usr/bin/env python3
# encoding: utf-8

"""
    Evaluates a set of models for a given number of train trials
"""

import os
import sys 
import re
import random
import traceback 

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from utils.log_utils import log, logTable
from utils.yaml_utils import updateMetricsLogFile, getMetricsLogFile
from utils.set_seed import set_seed
from models import SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_14L, CNN_3L, CNN_4L, CNN_5L, BatchNormMaxoutNetInNet
from models.BatchSizeStudy import CNN_14L_B10, CNN_14L_B25, CNN_14L_B50, CNN_14L_B80

# How many train loops are executed to study its variance
train_loops = 800
sameseed = False
current_dir_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir_path,"../output_data")

# General configuration
input_size = 28 * 28  # Size of each image flattened
num_classes = 10  # Numbers from 0 to 9
learning_rate = 0.001
patience = 10
num_epochs = 500


# Dict with how many iterations to be performed with each model
# The number is the total iterations to have stored on each CFG file
model_iterations = {BatchNormMaxoutNetInNet: 400,
                    CNN_14L_B10: 400, 
                    CNN_14L_B25: 400, 
                    CNN_14L_B50: 400,
                    CNN_14L_B80: 400,
                    CNN_3L: 400,
                    CNN_4L: 400,
                    CNN_5L: 400,
                    CNN_14L: 400,             # no-dropout -> https://arxiv.org/pdf/1608.06037
                    DNN_6L: 400,
                    HiddenLayerPerceptron: 400,
                    SimplePerceptron: 400}


if __name__ == "__main__":
    seed = 0
    metrics = []
    
    updated = True # Flag to control out of loop
    exception_messages = ""
    while updated:
        updated = False
        for ModelClass, num_iter in model_iterations.items():
            try:                
                metrics_log_file = os.path.join(output_path, ModelClass.__name__, f"{'sameseed_' if sameseed else 'randomseed_'}training_metrics.yaml")
                metrics = getMetricsLogFile(metrics_log_file)
                date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3}$') 
                iteration = sum(1 for key in metrics.keys() if date_pattern.match(key))
                if iteration < num_iter:
    
                    if not sameseed:
                        seed = random.randint(0, 2**32 - 1)  # Random 32 bits number

                    log(f"[{ModelClass.__name__}] Iteration: {iteration}/{num_iter}")
                    set_seed(seed)

                    # Load MNIST dataset
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
                    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

                    # Init model, loss and optimizer
                    model = ModelClass(input_size=input_size, num_classes=num_classes, learning_rate=learning_rate, patience=patience, seed=seed, output_path=output_path)
                    model.save_architecture()

                    train_loader = DataLoader(dataset=train_dataset, batch_size=model.batch_size, shuffle=True)
                    test_loader = DataLoader(dataset=test_dataset, batch_size=model.batch_size, shuffle=False)

                    metrics = model.spinTrainEval(train_loader, test_loader, num_epochs = num_epochs)
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