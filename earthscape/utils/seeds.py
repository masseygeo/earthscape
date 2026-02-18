
import os
from pathlib import Path
import yaml
import random
import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.windows import from_bounds
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes




def set_seed(seed, strict=False):
    """
    Set random seeds for reproducible behavior in Python, NumPy, and PyTorch.

    Seeds the Python built-in RNG, NumPy RNG, and PyTorch RNG for both
    CPU and all available CUDA devices. When ``strict=True``, enables
    deterministic cuDNN behavior and enforces deterministic PyTorch
    algorithms, which may reduce performance.

    Parameters
    ----------
    seed : int
        Random seed value.
    strict : bool, default=False
        If True, enforce deterministic cuDNN behavior and PyTorch
        deterministic algorithms.

    Returns
    -------
    None
    """

    # set python, numpy, pytorch seeds...
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # optionally set pytorch deterministic settings (may slow training)...
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)





def set_worker_seed(worker_id):
    """
    Initialize random seeds for a PyTorch DataLoader worker.

    Derives a worker-specific seed from ``torch.initial_seed()`` and
    uses it to seed NumPy and Python's built-in random module. Intended
    for use as the ``worker_init_fn`` argument in ``torch.utils.data.DataLoader``
    to ensure reproducible data loading when ``num_workers > 0``.

    Parameters
    ----------
    worker_id : int
        Worker process identifier assigned by the DataLoader.

    Returns
    -------
    None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




# def config_load(path):
#     with open(path, "r") as f:
#         cfg = yaml.safe_load(f)
#     return cfg




# def config_save(cfg, run_dir, config_path):
#     cfg_copy = cfg.copy()
#     cfg_copy["experiment"]["config_path"] = str(Path(config_path).resolve())
#     out_path = run_dir / "config_used.yml"
#     with open(out_path, "w") as f:
#         yaml.safe_dump(cfg_copy, f)




# def plot_multi_terrain_features(mdhs_path, terrain_paths, bounds, cmap, title):
#     """
#     Plot multiple terrain feature images over a multi-directional hillshade 
#     background. 
    
#     Creates a 2x3 panel figure and, for each terrain raster, 
#     plots a cropped multi-directional hillshade as the base layer with the 
#     terrain raster overlaid at 50% transparency. Each panel includes a 
#     small colorbar scaled to the raster's min/max within the requested bounds.

#     Parameters
#     ----------
#     mdhs_path : str or os.PathLike
#         Path to a multi-directional hillshade GeoTIFF.
#     terrain_paths : sequence of str or os.PathLike
#         Paths to terrain-feature rasters to plot (expected length 6).
#     bounds : sequence of float
#         Bounding box (left, bottom, right, top) in the rasters' coordinate
#         reference system.
#     cmap : str or matplotlib.colors.Colormap
#         Colormap for the terrain-feature overlay.
#     title : str
#         Suptitle for the figure.

#     Returns
#     -------
#     None
#     """

#     # set up plot assuming six scales/terrain features
#     fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8), sharex=True, sharey=True)
#     fig.subplots_adjust(wspace=0.1, hspace=0.1)
#     ax = ax.ravel()

#     with rasterio.open(mdhs_path) as mdhs:

#         # iterate through each terrain feature (six total)
#         for idx, path in enumerate(terrain_paths):
#             with rasterio.open(path) as src:

#                 # set up window for feature, get transform, and data
#                 window = from_bounds(*bounds, src.transform)
#                 transform = src.window_transform(window)
#                 data = src.read(1, window=window)
#                 min_val = np.min(data)
#                 max_val = np.max(data)

#                 # plot feature; this will be hidden and is only for colorbar
#                 hidden = ax[idx].imshow(data, cmap=cmap)

#                 # plot multi-directional hillshade as base layer (on top of hidden)
#                 mdhs_window = from_bounds(*bounds, mdhs.transform)
#                 mdhs_data = mdhs.read(1, window=mdhs_window)
#                 mdhs_transform = mdhs.window_transform(mdhs_window)
#                 show(mdhs_data, ax=ax[idx], cmap='binary_r', transform=mdhs_transform)

#                 # plot terrain feature with transparency (to overlay on hillshade)
#                 show(data, ax=ax[idx], cmap=cmap, transform=transform, alpha=0.5)

#                 # plot custom color bar
#                 cax = inset_axes(ax[idx], width='5%', height='40%', loc='lower right')
#                 fig.colorbar(hidden, cax=cax, ticks=[min_val, max_val])
#                 cax.yaxis.set_ticks_position('left')

#                 # customize plot elements
#                 ax[idx].tick_params(axis='both', which='major', labelsize=8)
#                 ax[idx].tick_params(axis='x', labelrotation=60)
#                 ax[idx].ticklabel_format(style='plain')
#                 ax[idx].set_title(os.path.basename(path), style='italic', fontsize=10)

#     plt.suptitle(title, y=0.96)
#     plt.show()




# def plot_training_curves(df):
#     """
#     Plot training and validation loss and accuracy over 
#     epochs. 
    
#     Generates a two-panel figure showing loss and micro-accuracy 
#     for training and validation sets across epochs. The epoch with the
#     minimum validation loss is marked with a vertical dashed line.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         DataFrame ordered by epoch containing columns for ``train loss``, 
#         ``val loss``, ``train accuracy``, and ``val accuracy``.

#     Returns
#     -------
#     matplotlib.figure.Figure
#         Figure containing the loss and accuracy subplots.
#     """

#     # setup figure and axes for two subplots
#     fig, ax = plt.subplots(ncols=2, figsize=(10,6))

#     # create generator for epochs
#     epochs = range(1, len(df)+1)

#     # plot loss subplot...
#     ax[0].plot(epochs, df['train loss'], lw=0.75, label='Train',)
#     ax[0].plot(epochs, df['val loss'], lw=0.75, label='Validation')
#     ax[0].set_ylabel('Loss')

#     # plot micro-averaged accuracy
#     ax[1].plot(epochs, df['train accuracy'], lw=0.75, label='Train')
#     ax[1].plot(epochs, df['val accuracy'], lw=0.75, label='Validation')
#     ax[1].set_ylabel('Micro-accuracy (%)')

#     # plot selected model at correct epoch
#     for axes in ax:
#         axes.axvline(x=df['val loss'].values.argmin()+1, linestyle='--', color='darkred', label='Selected')
#         axes.legend(frameon=False)
#         axes.set_xticks(epochs)
#         axes.set_xticklabels([str(x) if x%5==0 else '' for x in epochs])
#         axes.set_xlabel('Epochs')

#     plt.suptitle(f"Training and Validation Curves", y=0.92)

#     return fig




# def plot_pr_roc_curves(targets, predictions, class_cols):
#     """
#     Plot per-class precision-recall and receiver operating curves (ROC).

#     Generates a two-panel figure showing precision-recall curves (left panel) 
#     and ROC curves (right panel) for model experiment.

#     Parameters
#     ----------
#     targets : ndarray of shape (n_samples, n_classes)
#         Ground-truth binary labels.
#     predictions : ndarray of shape (n_samples, n_classes)
#         Predicted scores or probabilities for each class.
#     class_cols : sequence of str, len n_classes
#         Class names corresponding to each column.

#     Returns
#     -------
#     matplotlib.figure.Figure
#         Figure containing PR and ROC subplots with one curve per class.
#     """

#     # initialize figure and axes objects for two subplots 
#     fig, ax = plt.subplots(ncols=2, figsize=(10,5))

#     # initialize list for skipped classes
#     skipped = []

#     # iterate through classes...
#     for idx, unit in enumerate(class_cols):

#         # get ground truth & predicted labels...
#         Y_true = targets[:, idx]
#         y_pred = predictions[:, idx]

#         # if ground truth has no variance (same labels) then skip plots...
#         if Y_true.max() == Y_true.min():
#             skipped.append(unit)
#             continue
        
#         # get precision & recall for all thresholds for each class...
#         p, r, _ = precision_recall_curve(Y_true, y_pred)

#         # get FPR & TPR (recall/sensitivity) for all thresholds for each class...
#         fpr, tpr, _ = roc_curve(Y_true, y_pred)

#         # plot P-R curve & ROC for each class...
#         ax[0].plot(r, p, linewidth=0.75, label=class_cols[idx])
#         ax[1].plot(fpr, tpr, linewidth=0.75, label=class_cols[idx])
    
#     # customize plots...
#     ax[0].set_xlabel('Recall')
#     ax[0].set_ylabel('Precision')
#     ax[0].set_title('Precision-Recall Curve', style='italic')

#     ax[1].plot([0,1], [0,1], color='k', linestyle='--', lw=1)
#     ax[1].set_xlabel('False Positive Rate')
#     ax[1].set_ylabel('True Positive Rate')
#     ax[1].set_title('Receiver Operating Curve', style='italic')
    
#     for axes in ax:
#         axes.set_xlim(0,1)
#         axes.set_ylim(0,1)
    
#     # add legend
#     ax[0].legend(loc='upper center', bbox_to_anchor=(1.15, -0.15), ncols=len(class_cols), frameon=False, fontsize=8)

#     # add note for any skipped classes...
#     if skipped:
#         fig.subplots_adjust(bottom=0.28)
#         note = "*Not shown - no ground-truth label variation: " + ", ".join(skipped)
#         fig.text(0.5, 0.03, note, ha='center', va='bottom', fontsize=8, fontstyle='italic')

#     return fig
