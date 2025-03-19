
#######################################################################################
# PLOTTING FUNCTIONS
# ------------------
# Author: Matt Massey
# Last updated: 9/16/2024
# Purpose: Functions for customized plotting of geospatial images from surficial geologic mapping dataset.
#######################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import rasterio
from rasterio.plot import show
from rasterio.windows import from_bounds



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
    