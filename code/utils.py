
import random
import os
import numpy as np
import torch


##### global dictionary for SG Map class colors
map_colors = {
    'af1': '#636566', 
    'Qal': '#fdf5a4', 
    'Qaf': '#ffa1db', 
    'Qat': '#f9e465', 
    'Qc': '#d6c9a7', 
    'Qca': '#c49d83', 
    'Qr': '#b0acd6'
    }



def set_seed(seed: int):
    """
    Set seed for reproducible training and inference across Python, NumPy, and PyTorch (CPU & GPU).

    Parameters
    ----------
    seed : int
        The random seed to use.
    """
    
    # set python, numpy, pytorch seeds...
    random.seed(seed)                          # set Python built-in RNG
    # os.environ["PYTHONHASHSEED"] = str(seed)   # hash-based ops (DataLoader, dict ordering)
    np.random.seed(seed)                       # set NumPy RNG
    torch.manual_seed(seed)                    # set PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)           # set PyTorch GPU RNG across all CUDA devices

    # make cudnn deterministic (may slow down training)...
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False