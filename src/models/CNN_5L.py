#!/usr/bin/env python3
# encoding: utf-8
 
import torch
import torch.nn as nn
from models.BaseModel import BaseModelTrainer

# Model from: https://github.com/Coderx7/SimpleNet/blob/master/SimpNet_V1/Models/NoDropout/train_test.prototxt
# C3 from Figure 8
class CNN_5L(BaseModelTrainer):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_5L, self).__init__(input_size=None, num_classes=num_classes, learning_rate=learning_rate, patience=patience, output_path=output_path)
        
        # Definir las capas convolucionales
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(96, 128, kernel_size=5, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten output for fully connected layer
            nn.Flatten(),
        ).to(self.device)

        # Fully connected layer for clasification
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, num_classes),
            nn.BatchNorm1d(10)
        ).to(self.device)

        self._initialize_weights(seed)
        print(f"[{self.model_name}] Initialization of model complete, get into training process.")

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self, seed):
        """Inicializa los pesos de manera fija."""
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
