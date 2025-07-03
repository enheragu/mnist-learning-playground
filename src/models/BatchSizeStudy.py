#!/usr/bin/env python3
# encoding: utf-8
 
import torch
import torch.nn as nn
from models.CNN_14L import CNN_14L


class CNN_14L_B10(CNN_14L):
    batch_size = 10
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_14L_B10, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)


class CNN_14L_B25(CNN_14L):
    batch_size = 25
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_14L_B25, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)

        

class CNN_14L_B50(CNN_14L):
    batch_size = 50
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_14L_B50, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)



class CNN_14L_B80(CNN_14L):
    batch_size = 80
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(CNN_14L_B80, self).__init__(input_size, num_classes, learning_rate, patience, seed, output_path, input_channels)
