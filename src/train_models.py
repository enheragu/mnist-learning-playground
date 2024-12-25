#!/usr/bin/env python3
# encoding: utf-8

import os
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from utils.log_utils import log
from utils.yaml_utils import updateMetricsLogFile
from models import SimplePerceptron, HiddenLayerPerceptron, DNN_6L, CNN_13L, CNN_2L, CNN_4L, CNN_5L
from models.BatchSizeStudy import CNN_13L_B10, CNN_13L_B25, CNN_13L_B50, CNN_13L_B80

# How many train loops are executed to study its variance
train_loops = 800
sameseed = False
current_dir_path = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir_path,"../output_data")

# General configuration
input_size = 28 * 28  # Tamaño de cada imagen aplanada
num_classes = 10  # Números del 0 al 9
learning_rate = 0.001
patience = 10
num_epochs = 500


# Dict with how many iterations to be performed with each model
model_iterations = {CNN_2L: 15,
                    CNN_4L: 15,
                    CNN_5L: 15,
                    CNN_13L: 15,             # no-dropout -> https://arxiv.org/pdf/1608.06037
                    DNN_6L: 15,
                    HiddenLayerPerceptron: 15,
                    SimplePerceptron: 15,
                    CNN_13L_B10: 130, 
                    CNN_13L_B25: 130, 
                    CNN_13L_B50: 130}



def set_seed(seed=42):
    """Establece la semilla para todas las librerías relevantes."""
    torch.manual_seed(seed)  # Para PyTorch
    np.random.seed(seed)  # Para NumPy
    torch.cuda.manual_seed(seed)  # Para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Para todos los dispositivos GPU
    torch.backends.cudnn.deterministic = True  # Garantiza la determinación del algoritmo de cuDNN
    torch.backends.cudnn.benchmark = False  # No optimiza los algoritmos si las dimensiones no cambian



if __name__ == "__main__":
    seed = 0
    metrics = []


    max_iterations = max(model_iterations.values())
    
    for iteration in range(max_iterations):
       for ModelClass, num_iter in model_iterations.items():
            if iteration < num_iter:
 
                if not sameseed:
                    seed = random.randint(0, 2**32 - 1)  # Random 32 bits number

                log(f"[{ModelClass.__name__}] Iteration: {iteration}/{num_iter}")
                set_seed(seed)

                # Cargar el dataset MNIST
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
                test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

                # Init model, loss and optimizer
                model = ModelClass(input_size=input_size, num_classes=num_classes, learning_rate=learning_rate, patience=patience, seed=seed, output_path=output_path)
                model.save_architecture()

                train_loader = DataLoader(dataset=train_dataset, batch_size=model.batch_size, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=model.batch_size, shuffle=False)

                metrics = model.spinTrainEval(train_loader, test_loader, num_epochs = num_epochs)
                updateMetricsLogFile(metrics, os.path.join(model.output_data_path,f"{'sameseed_' if sameseed else 'randomseed_'}training_metrics.yaml"))
            