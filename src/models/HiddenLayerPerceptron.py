#!/usr/bin/env python3
# encoding: utf-8
 
import torch
import torch.nn as nn

from models.BaseModel import BaseModelTrainer

from utils.log_utils import log, logTable

class HiddenLayerPerceptron(BaseModelTrainer):
    def __init__(self, input_size, num_classes, learning_rate=0.001, patience=10, seed=42, output_path=""):
        super(HiddenLayerPerceptron, self).__init__(input_size, num_classes, learning_rate, patience, output_path=output_path)
        
        self.fc1 = nn.Linear(input_size, 800).to(self.device) # Hidden layer: 800 nodes
        self.fc2 = nn.Linear(800, num_classes).to(self.device)
        
        self._initialize_weights(seed)
        log(f"[{self.model_name}] Initialization of model complete, get into training process.")

    def forward(self, x):
        # Aplanar la imagen 28x28 a un vector de tamaño 784
        x = x.view(-1, 28 * 28)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def _initialize_weights(self, seed):
        """Inicializa los pesos de manera fija."""
        torch.manual_seed(seed)  # Asegura que la inicialización de pesos sea consistente
        
        # Inicializar pesos de fc1
        nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        # Inicializar pesos de fc2
        nn.init.uniform_(self.fc2.weight, a=-0.1, b=0.1)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
