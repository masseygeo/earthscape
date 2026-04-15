
import numpy as np
import pandas as pd
import geopandas as gpd




def splits_select_independent(gdf, n_patches, seed=111):
    """
    Split patch polygons into two spatially independent subsets. 
    
    A sample of ``n_patches`` polygons is randomly selected (without 
    replacement) to form the first subset (``selected``). 
    The second subset (``remaining``) contains all original polygons minus
    ``selected`` and those that spatially overlap any polygon in ``selected``. 
    Overlap is determined using the GeoPandas ``predicate="overlaps"`` relation, 
    where polygons share a positive-area intersection, but neither polygon completely 
    contains the other.

    Spatial independence is enforced only between ``selected`` and ``remaining``, but
    no constraint is imposed within each subset. Boundary-only touches between 
    subsets are permitted. 
    

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing patch polygons. Patches are assumed to never 
        completely contain another patch.
    n_patches : int
        Number of patches that will be randomly selected without replacement.
    seed : int, default=111
        Seed passed to ``numpy.random.default_rng`` for reproducibility.

    Returns
    -------
    selected : geopandas.GeoDataFrame
        Randomly selected ``n_patches`` patches.
    remaining : geopandas.GeoDataFrame
        Remaining patches excluding ``selected`` and any patches that overlap them.
    """
    # select random patches...
    rng = np.random.default_rng(seed=seed)
    random_patches_idx = rng.choice(gdf.index, size=n_patches, replace=False)
    selected = gdf.loc[random_patches_idx].copy()

    # select remaining & non-overlapping patches 
    remaining = gdf.drop(index=selected.index)

    # identify remaining patches that overlap with selected patches
    overlapping = remaining.sjoin(selected, how='inner', predicate='overlaps')

    # remove overlapping patches from remaining
    remaining = remaining.drop(index=overlapping.index)      

    # reset gdf indices...
    selected.reset_index(drop=True, inplace=True)
    remaining.reset_index(drop=True, inplace=True)

    return selected, remaining
