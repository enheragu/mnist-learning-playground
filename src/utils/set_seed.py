
#!/usr/bin/env python3
# encoding: utf-8

import random
import numpy as np
import torch

def set_seed(seed=42):
    """Establece la semilla para todas las librerías relevantes."""
    random.seed(seed)  # Para el módulo random de Python
    np.random.seed(seed)  # Para NumPy
    torch.manual_seed(seed)  # Para PyTorch
    torch.cuda.manual_seed(seed)  # Para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Para todos los dispositivos GPU
    torch.backends.cudnn.deterministic = True  # Garantiza la determinación del algoritmo de cuDNN
    torch.backends.cudnn.benchmark = False  # No optimiza los algoritmos si las dimensiones no cambian

