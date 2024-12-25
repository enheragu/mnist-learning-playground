#!/usr/bin/env python3
# encoding: utf-8
 
import torch
import torch.nn as nn
from models.CNN_13L import CNN_13L


class CNN_13L_B10(CNN_13L):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_13L, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)
        
        self.batch_size = 10

class CNN_13L_B25(CNN_13L):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_13L, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)
        
        self.batch_size = 25
        

class CNN_13L_B50(CNN_13L):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_13L, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)
        
        self.batch_size = 50


class CNN_13L_B80(CNN_13L):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_13L, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)
        
        self.batch_size = 80