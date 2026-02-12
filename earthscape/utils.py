
from earthscape.constants import DATASET_DIR, MODALITIES
import random
from pathlib import Path
import os
import yaml
import glob
import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.windows import from_bounds
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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




# def calculate_dataset_stats(data_dir=DATASET_DIR, patch_ids=None):

#     # find directories containing GeoTIFF files...
#     patch_dirs = []
#     for current_dir, _, files in os.walk(data_dir):
#         for file in files:
#             if file.lower().endswith('.tif'):
#                 patch_dirs.append(current_dir)
#                 break

#     # iterate through modalities->channels->single image paths...
#     df_stats = pd.DataFrame()

#     for mod_name, channels in MODALITIES.items():

#         # skip categorical channels...
#         if mod_name in ['osm', 'nhd', 'mask']:
#             continue

#         # iterate through moddality channels...
#         for c in channels:
            
#             # find path for single image...
#             img_paths = []
#             for pdir in patch_dirs:
#                 if patch_ids is None:
#                     img_paths.extend(glob.glob(f"{pdir}/*_{c}"))
#                 else:
#                     for id in patch_ids:
#                         img_paths.extend(glob.glob(f"{pdir}/{id}_{c}"))
#             img_paths = list(set(img_paths))

#             # iterate through image channel paths & collect image stats...
#             pixel_count = 0
#             nodata_count = 0
#             pixel_sum = 0.0
#             pixel_sum2 = 0.0
#             global_min = np.inf
#             global_max = -np.inf

#             for ip in img_paths:
#                 with rasterio.open(ip) as src:
#                     data = src.read(1, masked=True)
#                     total_pixels = data.size
#                     vals = data.compressed()

#                     pixel_count += vals.size
#                     nodata_count += total_pixels - vals.size
#                     pixel_sum += vals.sum()
#                     pixel_sum2 += (vals**2).sum()

#                     if vals.min() < global_min:
#                         global_min = vals.min()
#                     if vals.max() > global_max:
#                         global_max = vals.max()

#             # calculate global stats (mean & sample var/sd)...
#             mean = pixel_sum / pixel_count
#             var = (pixel_sum2 - (pixel_sum**2) / pixel_count) / (pixel_count - 1)
#             sd = np.sqrt(var)

#             # save to df...
#             df_stats.loc[c, 'mean'] = mean
#             df_stats.loc[c, 'sd'] = sd
#             df_stats.loc[c, 'min'] = global_min
#             df_stats.loc[c, 'max'] = global_max
#             df_stats.loc[c, 'nodata_count'] = nodata_count

#     return df_stats




def plot_multi_terrain_features(mdhs_path, terrain_paths, bounds, cmap, title):
    """
    Function to plot six terrain features from the same defined area. Terrain features have 50% transparency overlaying a multi-directional hillshade image.

    Parameters
    ----------
    mdhs_path : str
        Path to multi-directional hillshade GeoTIFF.
    terrain_paths : iterable
        List or tuple of paths terrain features at multiple resolutions
    bounds : iterable
        List or tuple of bounding coordinates (left, bottom, right, top) of area of interest.
    cmap : str or variable
        Name of Matplotlib colormap or custom colormap.
    title : str
        Title of terrain feature plot.

    Returns
    -------
    None.
    """

    # set up plot assuming six scales/terrain features
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ax = ax.ravel()

    with rasterio.open(mdhs_path) as mdhs:

        # iterate through each terrain feature (six total)
        for idx, path in enumerate(terrain_paths):
            with rasterio.open(path) as src:

                # set up window for feature, get transform, and data
                window = from_bounds(*bounds, src.transform)
                transform = src.window_transform(window)
                data = src.read(1, window=window)
                min_val = np.min(data)
                max_val = np.max(data)

                # plot feature; this will be hidden and is only for colorbar
                hidden = ax[idx].imshow(data, cmap=cmap)

                # plot multi-directional hillshade as base layer (on top of hidden)
                mdhs_window = from_bounds(*bounds, mdhs.transform)
                mdhs_data = mdhs.read(1, window=mdhs_window)
                mdhs_transform = mdhs.window_transform(mdhs_window)
                show(mdhs_data, ax=ax[idx], cmap='binary_r', transform=mdhs_transform)

                # plot terrain feature with transparency (to overlay on hillshade)
                show(data, ax=ax[idx], cmap=cmap, transform=transform, alpha=0.5)

                # plot custom color bar
                cax = inset_axes(ax[idx], width='5%', height='40%', loc='lower right')
                fig.colorbar(hidden, cax=cax, ticks=[min_val, max_val])
                cax.yaxis.set_ticks_position('left')

                # customize plot elements
                ax[idx].tick_params(axis='both', which='major', labelsize=8)
                ax[idx].tick_params(axis='x', labelrotation=60)
                ax[idx].ticklabel_format(style='plain')
                ax[idx].set_title(os.path.basename(path), style='italic', fontsize=10)

    plt.suptitle(title, y=0.96)
    plt.show()





def plot_training_curves(df):
    """
    Plot training and validation loss and accuracy over epochs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns ``train loss``, ``val loss``,
        ``train accuracy``, and ``val accuracy`` ordered by epoch.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the loss and accuracy curves.
    """

    # setup figure and axes for two subplots
    fig, ax = plt.subplots(ncols=2, figsize=(10,6))

    # create generator for epochs
    epochs = range(1, len(df)+1)

    # plot loss subplot...
    ax[0].plot(epochs, df['train loss'], lw=0.75, label='Train',)
    ax[0].plot(epochs, df['val loss'], lw=0.75, label='Validation')
    ax[0].set_ylabel('Loss')

    # plot micro-averaged accuracy
    ax[1].plot(epochs, df['train accuracy'], lw=0.75, label='Train')
    ax[1].plot(epochs, df['val accuracy'], lw=0.75, label='Validation')
    ax[1].set_ylabel('Micro-accuracy (%)')

    # plot selected model at correct epoch
    for axes in ax:
        axes.axvline(x=df['val loss'].argmin()+1, linestyle='--', color='darkred', label='Selected')
        axes.legend(frameon=False)
        axes.set_xticks(epochs)
        axes.set_xticklabels([str(x) if x%5==0 else '' for x in epochs])
        axes.set_xlabel('Epochs')

    plt.suptitle(f"Training and Validation Curves", y=0.92)

    return fig







def plot_pr_roc_curves(targets, predictions, class_cols):
    """
    Plot per-class precision-recall and ROC curves.

    Parameters
    ----------
    targets : array-like of shape (n_samples, n_classes)
        Ground-truth binary labels.
    predictions : array-like of shape (n_samples, n_classes)
        Predicted scores or probabilities for each class.
    class_cols : array-like of str of shape (n_classes)
        Class names corresponding to each column.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing PR and ROC subplots with one curve per class.
    """

    # initialize figure and axes objects for two subplots 
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))

    # initialize list for skipped classes
    skipped = []

    # iterate through classes...
    for idx, unit in enumerate(class_cols):

        # get ground truth & predicted labels...
        Y_true = targets[:, idx]
        y_pred = predictions[:, idx]

        # if ground truth has no variance (same labels) then skip plots...
        if Y_true.max() == Y_true.min():
            skipped.append(unit)
            continue
        
        # get precision & recall for all thresholds for each class...
        p, r, _ = precision_recall_curve(Y_true, y_pred)

        # get FPR & TPR (recall/sensitivity) for all thresholds for each class...
        fpr, tpr, _ = roc_curve(Y_true, y_pred)


        # plot P-R curve & ROC for each class...
        ax[0].plot(r, p, linewidth=0.75, color=SG_MAPPING[unit], label=class_cols[idx])
        ax[1].plot(fpr, tpr, linewidth=0.75, color=SG_MAPPING[unit], label=class_cols[idx])
    
    # customize plots...
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_title('Precision-Recall Curve', style='italic')

    ax[1].plot([0,1], [0,1], color='k', linestyle='--', lw=1)
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Curve', style='italic')
    
    for axes in ax:
        axes.set_xlim(0,1)
        axes.set_ylim(0,1)
    
    # add legend
    ax[0].legend(loc='upper center', bbox_to_anchor=(1.15, -0.15), ncols=len(class_cols), frameon=False, fontsize=8)

    # add note for any skipped classes...
    if skipped:
        fig.subplots_adjust(bottom=0.28)
        note = "*Not shown - no ground-truth label variation: " + ", ".join(skipped)
        fig.text(0.5, 0.03, note, ha='center', va='bottom', fontsize=8, fontstyle='italic')

    return fig
