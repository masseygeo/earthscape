
import random
import pathlib
import yaml
import numpy as np
import torch



def set_seed(seed):
    """
    Set seed for reproducible training and inference across Python, NumPy, and PyTorch (CPU & GPU).

    Parameters
    ----------
    seed : int
        The random seed to use.

    Return
    ------
    None
    """
    
    # set python, numpy, pytorch seeds...
    random.seed(seed)                          # set Python built-in RNG
    np.random.seed(seed)                       # set NumPy RNG
    torch.manual_seed(seed)                    # set PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)           # set PyTorch GPU RNG across all CUDA devices

    # make cudnn deterministic (may slow down training)...
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg



def save_config_snapshot(cfg, run_dir, config_path):
    """
    Save the resolved config to run_dir/config_used.yml for reproducibility.
    """
    cfg_copy = cfg.copy()
    cfg_copy["experiment"]["config_path"] = str(Path(config_path).resolve())
    out_path = run_dir / "config_used.yml"
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg_copy, f)