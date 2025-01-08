#!/usr/bin/env python3
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModelTrainer

from utils.log_utils import log, logTable

# Modified From: https://github.com/xbeat/Machine-Learning/blob/main/Constructing%20MaxOut%20Neural%20Networks%20in%20Python.md
class MaxOutLayer(nn.Module):
    def __init__(self, in_features, out_features, num_pieces):
        super(MaxOutLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        
        # Changing from a Linear layer to a 1x1 convolution to obtain the pieces.
        # We use a 1x1 convolution to produce "num_pieces" pieces from each input channel.
        self.conv = nn.Conv2d(in_features, out_features * num_pieces, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Apply the 1x1 convolution to get the output pieces.
        x = self.conv(x)  # [batch_size, out_features * num_pieces, H, W]

        # Reshape to [batch_size, out_features, num_pieces, H, W]
        x = x.view(x.size(0), self.out_features, self.num_pieces, x.size(2), x.size(3))

        # Apply max pooling across the pieces (dim=2)
        x = torch.max(x, dim=2)[0]  # Max pooling along the pieces
        return x


class MINBlock(nn.Module):
    """Un bloque MIN (Convolución + MLP Maxout + Pooling + Dropout)."""
    def __init__(self, in_channels, conv_out_channels, mlp_out1, mlp_out2, kernel_size, stride, padding, k, pool_kernel, pool_stride, dropout_rate):
        super(MINBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, conv_out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(conv_out_channels)
        self.mlp1 = nn.Sequential(
            # MaxoutHiddenLayer modeled with Conv2d in pytorch for better optimization ¿?
            nn.Conv2d(conv_out_channels, mlp_out1, kernel_size=1, stride=1, padding=0),
            MaxOutLayer(mlp_out1, mlp_out1, k),
            nn.BatchNorm2d(mlp_out1)
        )
        self.mlp2 = nn.Sequential(
            # MaxoutHiddenLayer modeled with Conv2d in pytorch for better optimization ¿?
            nn.Conv2d(mlp_out1, mlp_out2, kernel_size=1, stride=1, padding=0),
            MaxOutLayer(mlp_out2, mlp_out2, k),
            nn.BatchNorm2d(mlp_out2)
        )
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        print(f"[MINBlock::forward] Input: {x.shape}")
        x = (self.bn(self.conv(x)))
        print(f"[MINBlock::forward] After conv and bn: {x.shape}")
        x = self.mlp1(x)
        print(f"[MINBlock::forward] After mlp1: {x.shape}")
        x = self.mlp2(x)
        print(f"[MINBlock::forward] After mlp2: {x.shape}")
        x = self.pool(x)
        print(f"[MINBlock::forward] After pool: {x.shape}")
        x = self.dropout(x)
        print(f"[MINBlock::forward] After dropout: {x.shape}")
        return x
    
# Approximation from https://arxiv.org/pdf/1511.02583
class BatchNormMaxoutNetInNet(BaseModelTrainer):
    def __init__(self, input_size, num_classes=10, learning_rate=0.001, patience=10, seed=42, output_path="", input_channels=1):
        super(BatchNormMaxoutNetInNet, self).__init__(input_size=None, num_classes=num_classes, learning_rate=learning_rate, patience=patience, output_path=output_path)
        
        self.block1 = MINBlock(
            in_channels=input_channels, conv_out_channels=128, kernel_size=5, stride=1, padding=0,
            mlp_out1=96, mlp_out2=48, k=5, 
            pool_kernel=3, pool_stride=2, dropout_rate=0.5
        )
        self.block2 = MINBlock(
            in_channels=48, conv_out_channels=128, kernel_size=5, stride=1, padding=2,
            mlp_out1=96, mlp_out2=48, k=5, 
            pool_kernel=3, pool_stride=2, dropout_rate=0.5
        )
        self.block3 = MINBlock(
            in_channels=48, conv_out_channels=128, kernel_size=3, stride=1, padding=1,
            mlp_out1=96, mlp_out2=10,  k=5, 
            pool_kernel=1, pool_stride=1, dropout_rate=0.0
        )
        # self.fc = nn.Linear(128 * 24 * 24, num_classes)  # 128 * 24 * 24 es el tamaño de salida después de la convolución

        self._initialize_weights(seed)
        log(f"[{self.model_name}] Initialization of model complete, get into training process.")

    def forward(self, x):
        print(f"[BatchNormMaxoutNetInNet::forward] Input: {x.shape}")
        x = self.block1(x)
        print(f"[BatchNormMaxoutNetInNet::forward] After first block: {x.shape}")
        x = self.block2(x)
        print(f"[BatchNormMaxoutNetInNet::forward] After second block: {x.shape}")
        x = self.block3(x)
        print(f"[BatchNormMaxoutNetInNet::forward] Before flattening: {x.shape}")  # Debugging line: Check shape before flattening
        x = x.view(x.size(0), -1)  # Aplana para softmax
        print(f"[BatchNormMaxoutNetInNet::forward] After flattening: {x.shape}")  # Debugging line: Check shape after flattening
        # x = self.fc(x)
        return F.log_softmax(x, dim=1)

