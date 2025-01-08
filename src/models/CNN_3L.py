#!/usr/bin/env python3
# encoding: utf-8
 
import torch
import torch.nn as nn
from models.BaseModel import BaseModelTrainer

from utils.log_utils import log, logTable

# Model from: https://arxiv.org/pdf/2008.10400
# C1 from Figure 8
class CNN_3L(BaseModelTrainer):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_3L, self).__init__(input_size=None, num_classes=num_classes, learning_rate=learning_rate, patience=patience, output_path=output_path)
        
        # Definir las capas convolucionales
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=1),   # Output size: 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # Output size: 14x14

            nn.Conv2d(64, 128, kernel_size=5, padding=1),  # Output size: 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output size: 7x7

            # Flatten output for fully connected layer
            nn.Flatten(),
        ).to(self.device)

        # Fully connected layer for clasification
        self.fc_layers = nn.Sequential(
            nn.Linear(3200, num_classes),
            nn.BatchNorm1d(10)
        ).to(self.device)

        self._initialize_weights(seed)
        log(f"[{self.model_name}] Initialization of model complete, get into training process.")

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x