
from earthscape.utils.constants import SRC_URLS, VERSION

import os
from datetime import datetime
import json
import pandas as pd
import numpy as np
import rasterio

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.plot import show
from rasterio.windows import from_bounds



def qc_coalignment(input_paths, target_path, atol=1e-9, rtol=0.0):
    """
    Check image grid alignment relative to a target image. 
    
    For each raster in `input_paths`, this function collects quality control (QC)
    metadata including dtype, nodata value, resolution, dimensions, and bounds.
    It then determines whether each image is co-aligned with `target_path`
    based on the QC metrics. Alignment is defined as:
        - Exact match for integer fields (`width`, `height`)
        - Tolerant match for floating-point fields (`res_x`, `res_y`,
        `left`, `bottom`, `right`, `top`) using `numpy.isclose`
        with specified `atol` and `rtol`.

    The output DataFrame includes all QC metrics, a boolean `aligned`
    column indicating whether each image matches the target grid, and 
    general image information (informal image name, path, dtype, and nodata value).

    Parameters
    ----------
    input_paths : sequence of str or os.PathLike
        Paths to raster files to evaluate.
    target_path : str or os.PathLike
        Path to the target raster defining the reference grid. 
        Assumed to also be included in `input_paths`.
    atol : float, default=1e-9
        Absolute tolerance for floating-point grid comparisons.
    rtol : float, default=0.0
        Relative tolerance for floating-point grid comparisons.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per input image and all associated 
        information described above.
    """
   
    # get basenames of images...
    image_names = []
    for path in input_paths:
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        image_names.append(name)
    
    # initialize new df with basenames, their paths, and alignment QC columns...
    df = pd.DataFrame({'image':image_names, 'path': input_paths})

    # add columns for quality control metrics & additional information...
    additional_cols = ['dtype', 'nodata']                            # additional user information
    exact_cols = ['width', 'height']                                 # exact (integers)
    tol_cols = ['res_x', 'res_y', 'left', 'bottom', 'right', 'top']   # tolerant (floating point) 
    df[additional_cols + exact_cols + tol_cols] = pd.NA

    # iterate through image paths & get QC values...
    for path in input_paths:
        with rasterio.open(path) as src:
            df.loc[df['path'] == path, 'dtype'] = src.meta['dtype']
            df.loc[df['path'] == path, 'nodata'] = src.nodata
            df.loc[df['path'] == path, 'res_x'] = src.res[0]
            df.loc[df['path'] == path, 'res_y'] = src.res[1]
            df.loc[df['path'] == path, 'width'] = src.width
            df.loc[df['path'] == path, 'height'] = src.height
            df.loc[df['path'] == path, 'left'] = src.bounds.left
            df.loc[df['path'] == path, 'bottom'] = src.bounds.bottom
            df.loc[df['path'] == path, 'right'] = src.bounds.right
            df.loc[df['path'] == path, 'top'] = src.bounds.top

    # get series of QC metrics from target
    target = df.loc[df['path']==target_path, exact_cols+tol_cols].iloc[0]

    # determine if images are co-aligned with target using QC metrics...
    exact_ok = (df[exact_cols].astype(int) == target[exact_cols].astype(int)).all(axis=1)
    tol_ok = np.isclose(df[tol_cols].astype(float), target[tol_cols].astype(float), atol=atol, rtol=rtol).all(axis=1)

    # provide bool of co-alignment check into df
    df.insert(loc=2, column='aligned', value=exact_ok & tol_ok)
        
    return df




def qc_patch_size(patch_path, patch_size):
    """
    Validate an image patch by size and nodata content. 

    This function checks whether a patch image matches the expected
    square dimensions (`patch_size`, `patch_size`) and contains no
    nodata pixels. If the patch fails validation, its file path is
    returned.

    Parameters
    ----------
    patch_path : str or os.PathLike
        Path to the raster patch to validate.
    patch_size : int
        Expected patch height and width in pixels.

    Returns
    -------
    str or None
        Returns `patch_path` if the patch has incorrect dimensions or
        contains any nodata pixels; otherwise returns None.
    """


    # open image patch; check patch size & nodata pixel count...
    with rasterio.open(patch_path) as src:
        data = src.read(1, masked=True)
        h = src.height
        w = src.width
        nd_count = data.mask.sum()

        # return path IF image does not equal given patch size or has any nodata values
        if (h != patch_size) or (w != patch_size) or (nd_count > 0):
            return patch_path
        



def create_metadata(area_name, label_space, num_patches, num_imgs, img_meta, patch_size, overlap, output_path, sources=SRC_URLS, version=VERSION):
    """
    Create and write a dataset metadata JSON file. 
    
    This function compiles dataset-level metadata for an EarthScape subset,
    including version information, patch configuration, channel count,
    geospatial properties, and selected data sources. The metadata is written
    to `output_path` as a formatted JSON file.

    Parameters
    ----------
    area_name : str
        Name of the study area.
    label_space : sequence of str
        List of class labels included in the dataset.
    num_patches : int
        Total number of image patches.
    num_imgs : int
        Total number of images in the dataset.
    img_meta : dict
        Raster metadata dictionary from `rasterio.open(...).meta`
        used to extract CRS, resolution, and nodata information.
    patch_size : int
        Patch size in pixels.
    overlap : float
        Patch overlap proportion (e.g., 0.25 for 25%).
    output_path : str or os.PathLike
        Path to write the metadata JSON file.
    sources : dict
        Dictionary of data source names and descriptions. 
    version : str
        Dataset version identifier..

    Returns
    -------
    None
    """
    dataset = {}
    dataset['EarthScape'] = {}
    dataset['EarthScape']['version'] = version
    dataset['EarthScape']['created'] = datetime.now().strftime('%Y-%m-%d, %H:%M')
    dataset['EarthScape']['patch size'] = patch_size
    dataset['EarthScape']['overlap'] = str(int(overlap * 100))+'%'
    dataset['EarthScape']['num patches'] = int(num_patches)
    dataset['EarthScape']['num channels'] = int(num_imgs / num_patches)
    dataset['EarthScape']['total images'] = num_imgs
    dataset['EarthScape']['label space'] = label_space

    dataset['Geospatial'] = {}
    dataset['Geospatial']['crs'] = img_meta['crs'].to_string()
    dataset['Geospatial']['units'] = img_meta['crs'].to_dict()['units']
    dataset['Geospatial']['resolution'] = img_meta['transform'][0]
    dataset['Geospatial']['nodata'] = img_meta['nodata']

    dataset['Data Sources'] = {}
    for k,v in sources.items():
        if any(s in k for s in ['kyfromabove', 'nhd', 'osm', area_name.replace(' ','').lower()]):
            dataset['Data Sources'][k] = v

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)




def create_release_metadata(dicts_by_name):
    """
    Recursively merge metadata from multiple subsets.

    For each key across sources:
    - If all present values are equal, the scalar value is retained.
    - If values differ, a mapping of {source: value} is returned.
    - Nested dictionaries are merged recursively.

    Parameters
    ----------
    dicts_by_name : dict of str to dict
        Mapping of source name to its metadata dictionary.

    Returns
    -------
    dict
        Merged dictionary with shared values collapsed and differing
        values represented by source-keyed mappings.
    """
    MISSING = object()

    all_keys = set()
    for d in dicts_by_name.values():
        if isinstance(d, dict):
            all_keys |= set(d.keys())

    merged = {}
    for key in all_keys:
        vals = {
            src: (d.get(key, MISSING) if isinstance(d, dict) else MISSING)
            for src, d in dicts_by_name.items()
        }
        present = {src: v for src, v in vals.items() if v is not MISSING}
        if not present:
            continue

        # recurse if all present values are dicts...
        if all(isinstance(v, dict) for v in present.values()):
            merged[key] = create_release_metadata({src: v for src, v in present.items()})
            continue

        first = next(iter(present.values()))
        all_equal_among_present = all(v == first for v in present.values())

        if all_equal_among_present:
            merged[key] = first    # collapse even if some sources are missing this key
        else:
            merged[key] = present  # provenance mapping at the leaf

    return merged




def plot_multi_terrain_features(mdhs_path, terrain_paths, bounds, cmap, title):
    """
    Plot multiple terrain feature images over a multi-directional hillshade 
    background. 
    
    Creates a 2x3 panel figure and, for each terrain raster, 
    plots a cropped multi-directional hillshade as the base layer with the 
    terrain raster overlaid at 50% transparency. Each panel includes a 
    small colorbar scaled to the raster's min/max within the requested bounds.

    Parameters
    ----------
    mdhs_path : str or os.PathLike
        Path to a multi-directional hillshade GeoTIFF.
    terrain_paths : sequence of str or os.PathLike
        Paths to terrain-feature rasters to plot (expected length 6).
    bounds : sequence of float
        Bounding box (left, bottom, right, top) in the rasters' coordinate
        reference system.
    cmap : str or matplotlib.colors.Colormap
        Colormap for the terrain-feature overlay.
    title : str
        Suptitle for the figure.

    Returns
    -------
    None
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
