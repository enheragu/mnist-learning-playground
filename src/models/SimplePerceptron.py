#!/usr/bin/env python3
# encoding: utf-8
 
import torch
import torch.nn as nn

from models.BaseModel import BaseModelTrainer

from utils.log_utils import log, logTable

class SimplePerceptron(BaseModelTrainer):
    def __init__(self, input_size, num_classes, learning_rate=0.001, patience=10, seed=42, output_path=""):
        super(SimplePerceptron, self).__init__(input_size, num_classes, learning_rate, patience, output_path=output_path)
        
        self.fc1 = nn.Linear(input_size, num_classes).to(self.device)
        self._initialize_weights(seed)

        log(f"[{self.model_name}] Initialization of model complete, get into training process.")

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanamos la imagen 28x28 a un vector de tamaño 784
        return self.fc1(x)

    def _initialize_weights(self,seed):
        """Inicializa los pesos de manera fija."""
        self.seed = seed
        torch.manual_seed(seed)  # Asegura que la inicialización de pesos sea consistente
        nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1)  # Ejemplo de inicialización uniforme
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)  # Inicia el sesgo en 0