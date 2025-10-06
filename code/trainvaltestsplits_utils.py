
import numpy as np
import geopandas as gpd


def select_indpendent_patches(gdf, n_patches, seed=111):
    """
    Randomly select a subset of patch polygons, then remove from the remainder any patches that **geographically overlap** the selected ones. By design, selected patches may overlap each other; the constraint is only enforced between `selected` and `remaining`.


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


    ##### 1. select random patches...
    # create numpy RNG using seed
    rng = np.random.default_rng(seed=seed)

    # choose random patches
    random_patches_idx = rng.choice(gdf.index, size=n_patches, replace=False)

    # isolate selected patches
    selected = gdf.loc[random_patches_idx].copy()


    ##### 2. select remaining & non-overlapping patches 
    # separate selected from remaining patches
    remaining = gdf[~gdf.index.isin(selected.index)].copy()
    
    # identify remaining patches that overlap with selected patches
    overlapping = remaining.sjoin(selected, how='inner', predicate='overlaps')

    # remove overlapping patches from remaining
    remaining = remaining[~remaining.index.isin(overlapping.index)]        


    ##### 3. reset gdf indices...
    selected.reset_index(drop=True, inplace=True)
    remaining.reset_index(drop=True, inplace=True)

    return selected, remaining

