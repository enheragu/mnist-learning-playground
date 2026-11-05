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
        # Flatten the 28x28 image into a vector of size 784
        x = x.view(-1, 28 * 28)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def _initialize_weights(self, seed):
        self.seed = seed
        """Initializes the weights in a fixed way."""
        torch.manual_seed(seed)  # Ensures the weight initialization is consistent
        
        # Initialize the fc1 weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
