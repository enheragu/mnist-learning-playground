
#!/usr/bin/env python3
# encoding: utf-8

import random
import numpy as np
import torch

def set_seed(seed=42):
    """Sets the seed for all the relevant libraries."""
    random.seed(seed)  # For the Python random module
    np.random.seed(seed)  # For NumPy
    torch.manual_seed(seed)  # For PyTorch
    torch.cuda.manual_seed(seed)  # For PyTorch on GPU
    torch.cuda.manual_seed_all(seed)  # For all GPU devices
    torch.backends.cudnn.deterministic = True  # Makes the cuDNN algorithm selection deterministic
    torch.backends.cudnn.benchmark = False  # Does not optimize the algorithms if the dimensions do not change

