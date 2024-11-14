#!/usr/bin/env python3
# encoding: utf-8

import os
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from utils.yaml_utils import updateMetricsLogFile
from models.SimplePerceptron import SimplePerceptron
from models.HiddenLayerPerceptron import HiddenLayerPerceptron
from models.DNN_6L import DNN_6L
from models.CNN_13L import CNN_13L
from models.CNN_2L import CNN_2L
from models.CNN_5L import CNN_5L

# How many train loops are executed to study its variance
train_loops = 800
sameseed = False
output_path = "./output_data"

# General configuration
input_size = 28 * 28  # Tamaño de cada imagen aplanada
num_classes = 10  # Números del 0 al 9
batch_size = 64
learning_rate = 0.001
patience = 10
num_epochs = 500


# Dict with how many iterations to be performed with each model
model_iterations = {CNN_2L: 197,                # https://github.com/Coderx7/SimpleNet/blob/master/SimpNet_V1/Models/NoDropout/train_test.prototxt
              CNN_5L: 200,                # https://github.com/Coderx7/SimpleNet/blob/master/SimpNet_V1/Models/NoDropout/train_test.prototxt
              CNN_13L: 188,             # no-dropout -> https://arxiv.org/pdf/1608.06037
              DNN_6L: 188,
              HiddenLayerPerceptron: 139,
              SimplePerceptron: 0}



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

                print(f"[{type(ModelClass).__name__}] Iteration: {iteration}/{num_iter}")
                set_seed(seed)

                # Cargar el dataset MNIST
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
                test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

                # Init model, loss and optimizer
                model = ModelClass(input_size=input_size, num_classes=num_classes, learning_rate=learning_rate, patience=patience, seed=seed, output_path=output_path)
                model.save_architecture()

                metrics = model.spinTrainEval(train_loader, test_loader, num_epochs = num_epochs)
                updateMetricsLogFile(metrics, os.path.join(model.output_data_path,f"{'sameseed_' if sameseed else 'randomseed_'}training_metrics.yaml"))
            