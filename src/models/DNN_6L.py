#!/usr/bin/env python3
# encoding: utf-8
 
import torch
import torch.nn as nn
from models.BaseModel import BaseModelTrainer

from utils.log_utils import log

class DNN_6L(BaseModelTrainer):
    def __init__(self, input_size, num_classes, learning_rate=0.001, patience=10, seed=42, output_path=""):
        super(DNN_6L, self).__init__(input_size, num_classes, learning_rate, patience, output_path=output_path)

        # Definir las capas de la red
        self.fc1 = nn.Linear(input_size, 2500).to(self.device)   # Hidden layer 1
        self.fc2 = nn.Linear(2500, 2000).to(self.device)         # Hidden layer 2
        self.fc3 = nn.Linear(2000, 1500).to(self.device)         # Hidden layer 3
        self.fc4 = nn.Linear(1500, 1000).to(self.device)         # Hidden layer 4
        self.fc5 = nn.Linear(1000, 500).to(self.device)          # Hidden layer 5
        self.fc6 = nn.Linear(500, num_classes).to(self.device)   # Output layer

        self._initialize_weights(seed)
        log(f"[{self.model_name}] Initialization of model complete, get into training process.")

    def forward(self, x):
        # Aplanar la entrada (784)
        x = x.view(-1, 28 * 28)
        
        x = torch.relu(self.fc1(x))  # ReLu for layer 1
        x = torch.relu(self.fc2(x))  # ReLu for layer 2
        x = torch.relu(self.fc3(x))  # ReLu for layer 3
        x = torch.relu(self.fc4(x))  # ReLu for layer 4
        x = torch.relu(self.fc5(x))  # ReLu for layer 5
        
        # Output layer (CrossEntropyLoss will be used)
        x = self.fc6(x)
        
        return x

    def _initialize_weights(self, seed):
        """Inicializa los pesos de cada capa de manera uniforme."""
        torch.manual_seed(seed)  # Asegura que la inicialización de pesos sea consistente
        
        # Inicializar pesos para cada capa
        nn.init.uniform_(self.fc1.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc2.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc3.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc4.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc5.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc6.weight, a=-0.1, b=0.1)
        
        # Inicializar sesgos
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.fc3.bias)
            nn.init.zeros_(self.fc4.bias)
            nn.init.zeros_(self.fc5.bias)
            nn.init.zeros_(self.fc6.bias)
