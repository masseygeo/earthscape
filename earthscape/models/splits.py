
import numpy as np
import pandas as pd
import geopandas as gpd


def select_indpendent_patches(gdf, n_patches, seed=111):
    """
    Randomly select a subset of patch polygons, then remove from the remainder any patches that *geographically overlap* the selected ones. By design, selected patches may overlap each other; the constraint is only enforced between `selected` and `remaining`.


    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame of patch polygons to split.
    n_patches : int
        Number of patches to randomly select into `selected`.
    seed : int
        RNG seed for reproducibility of the random selection. Defaults to 111.

    Returns
    -------
    selected : GeoDataFrame
        Randomly selected patches.
    remaining : GeoDataFrame
        All patches excluding `selected` AND those that overlap with `selected`.

    Notes
    ------
    - Patches within a split may overlap each other.
    - Patches between splits are geographically isolated and may NOT overlap one another.
    - Patches between splits may touch one another's boundary edges.

    """

    # select random patches...
    rng = np.random.default_rng(seed=seed)
    random_patches_idx = rng.choice(gdf.index, size=n_patches, replace=False)
    selected = gdf.loc[random_patches_idx].copy()

    # select remaining & non-overlapping patches 
    remaining = gdf.drop(index=selected.index).copy()

    # identify remaining patches that overlap with selected patches
    overlapping = remaining.sjoin(selected, how='inner', predicate='overlaps')

    # remove overlapping patches from remaining
    remaining = remaining.drop(index=overlapping.index)      

    # reset gdf indices...
    selected.reset_index(drop=True, inplace=True)
    remaining.reset_index(drop=True, inplace=True)

    return selected, remaining




def make_smoke_set(classes_path, patches_path, split_size, area_threshold=0.2, seed=111):

    rng = np.random.RandomState(seed)
    classes = pd.read_csv(classes_path)
    patches = gpd.read_file(patches_path)
    gdf = pd.merge(left=patches, right=classes, how='left', on='patch_id')
    class_list = list(gdf.columns[2:])

    remaining = gdf.copy()
    initial_picks = []

    for c in class_list:
        candidates = remaining.loc[remaining[c] > area_threshold]
        choice = candidates.sample(n=1, replace=False, random_state=rng)
        initial_picks.append(choice)
        remaining.drop(choice.index, inplace=True)
    
    selected = pd.concat(initial_picks, axis=0)

    need = max(0, split_size - len(selected))
    if need > 0:
        filler = remaining.sample(n=need, replace=False, random_state=rng)
        selected = pd.concat([selected, filler], axis=0)
        selected.reset_index(drop=True, inplace=True)
    
    return selected
