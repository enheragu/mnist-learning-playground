

import numpy as np
import torch

def set_seed(seed=42):
    """Establece la semilla para todas las librerías relevantes."""
    torch.manual_seed(seed)  # Para PyTorch
    np.random.seed(seed)  # Para NumPy
    torch.cuda.manual_seed(seed)  # Para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Para todos los dispositivos GPU
    torch.backends.cudnn.deterministic = True  # Garantiza la determinación del algoritmo de cuDNN
    torch.backends.cudnn.benchmark = False  # No optimiza los algoritmos si las dimensiones no cambian

