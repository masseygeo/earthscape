
import numpy as np
import pandas as pd
import geopandas as gpd



def create_smoke(areas_path, patches_path, split_size, min_n_classes=1, area_threshold=0.2, seed=111):
    """
    Generate a small "smokeset" of patch polygons for lightweight exploration
    and quick modeling experiments.

    Patch geometries are joined with per-class attributes. For each class, 
    ``min_n_classes`` patches with patch footprints greater than ``area_threshold`` are 
    sampled without replacement. If fewer than ``split_size`` patches are selected, 
    additional patches are sampled uniformly from the remaining pool to reach ``split_size``.

    Parameters
    ----------
    areas_path : str or os.PathLike
        Path to class areas CSV containing ``patch_id`` and per-class columns.
    patches_path : str or os.PathLike
        Path to GeoJSON file containing patch geometries and ``patch_id``.
    split_size : int
        Number of patches for the output.
    min_n_classes : int, default=1
        Minimum number of patches of each class for the output; may contain more, 
        but this is the minimum.
    area_threshold : float, default=0.2
        Minimum patch footprint (proportional area) required for eligibility.
    seed : int, default=111
        Seed controlling reproducible sampling.

    Returns
    -------
    selected : geopandas.GeoDataFrame
        GeoDataFrame of selected patches with columns ``patch_id`` and ``geometry``.
    """

    # set seed
    rng = np.random.RandomState(seed)

    # read CSV containing rows of patch_id & columns of class labels
    classes = pd.read_csv(areas_path)

    # read GeoJSON of patches
    patches = gpd.read_file(patches_path)

    # join class labels with patches using patch_id (gdf must be in left argument, df in right)
    gdf = pd.merge(left=patches, right=classes, how='left', on='patch_id')

    # set up variables for smokeset selection...
    class_list = list(gdf.columns[2:])    # list of class labels
    remaining = gdf.copy()                # copy of merged gdf
    initial_picks = []                    # list to hold selected patches

    # select samples from each class for smokeset...
    for c in class_list:

        # samples of class c with area greater than provided threshold
        candidates = remaining.loc[remaining[c] > area_threshold]

        # randomly choose 'min_n_classes' from candidates
        choice = candidates.sample(n=min_n_classes, replace=False, random_state=rng)

        # append selections to list
        initial_picks.append(choice)

        # drop those samples so that they will not be selected again
        remaining.drop(choice.index, inplace=True)
    
    # concatenate selected patches from each class into one df
    selected = pd.concat(initial_picks, axis=0)

    # optional - select additional samples to fill desired split_size...
    need = max(0, split_size - len(selected))
    if need > 0:
        filler = remaining.sample(n=need, replace=False, random_state=rng)
        selected = pd.concat([selected, filler], axis=0)

    # return final gdf...
    selected = selected.loc[:, ["patch_id", "geometry"]]
    selected.reset_index(drop=True, inplace=True)
    
    return selected

