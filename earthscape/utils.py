
from earthscape.constants import DATASET_DIR, MODALITIES
import random
from pathlib import Path
import os
import yaml
import glob
import pandas as pd
import numpy as np
import rasterio
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




def calculate_dataset_stats(data_dir=DATASET_DIR, patch_ids=None):

    # find directories containing GeoTIFF files...
    patch_dirs = []
    for current_dir, subdirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.tif'):
                patch_dirs.append(current_dir)
                break


    # iterate through modalities->channels->single image paths...
    df_stats = pd.DataFrame()
    for mod_name, channels in MODALITIES.items():

        # skip categorical channels...
        if mod_name in ['osm', 'nhd', 'mask']:
            continue
        
        # iterate through moddality channels...
        for c in channels:
            
            # find path for single image...
            img_paths = []
            for pdir in patch_dirs:
                if not patch_ids:
                    img_paths.extend(glob.glob(f"{pdir}/*_{c}"))
                else:
                    for id in patch_ids:
                        img_paths.extend(glob.glob(f"{pdir}/{id}_{c}"))
            img_paths = list(set(img_paths))

            # iterate through image channel paths & collect image stats...
            pixel_count = 0.0
            pixel_sum = 0.0
            pixel_sum2 = 0.0
            for ip in img_paths:
                with rasterio.open(ip) as src:
                    data = src.read(1, masked=True)
                    vals = data.compressed()
                    pixel_count += vals.size
                    pixel_sum += vals.sum()
                    pixel_sum2 += (vals**2).sum()

            # calculate global stats (mean & sample var/sd)...
            mean = pixel_sum / pixel_count
            var = (pixel_sum2 - (pixel_sum**2) / pixel_count) / (pixel_count - 1)
            sd = np.sqrt(var)

            # save to df...
            df_stats.loc[c, 'mean'] = mean
            df_stats.loc[c, 'sd'] = sd

    return df_stats