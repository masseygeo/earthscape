
import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.mask import mask



def patches_create(reference_path, patch_size, patch_overlap, boundary_path, output_path, name_prefix=None):
    """
    Create square patch polygons aligned to a reference raster grid over an AOI. 
    
    Patch polygons are generated on the pixel grid of `reference_path` so that patch
    edges align exactly with raster pixel boundaries. The raster provides the CRS,
    resolution, and raster-aligned bounds used to step a regular patch grid. The
    AOI provided by `boundary_path` is expected to represent the same area as the
    reference raster (vector vs. raster representation), and is used to retain only
    patches that are fully contained within the AOI geometry.

    Patch size is specified in pixels and converted to map units using the raster
    pixel size. Adjacent patches are spaced according to `patch_overlap`. Each
    retained patch is assigned a unique `patch_id` derived from patch size, overlap
    percentage, and a sequential index (optionally prefixed).

    Parameters
    ----------
    reference_path : str or os.PathLike
        Path to a reference GeoTIFF used to define CRS, pixel resolution, and
        raster-aligned bounds for patch generation.
    patch_size : int
        Size of each square patch in pixels.
    patch_overlap : float
        Proportion of overlap between adjacent patches (e.g., 0.25 for 25% overlap).
    boundary_path : str or os.PathLike
        Path to the AOI boundary GeoJSON. Patches are kept only if fully contained
        within the AOI geometry. The AOI is assumed to correspond to the same area
        as `reference_path` (vector vs. raster extent differences).
    output_path : str or os.PathLike
        Destination path for the output patch polygon GeoJSON.
    name_prefix : str or None, default=None
        Optional prefix to prepend to generated patch IDs.

    Returns
    -------
    None
    """
    # read AOI boundary as gdf & get boundary geometry...
    boundary = gpd.read_file(boundary_path)
    boundary_union = boundary.geometry.union_all()

    # get bounding coordinates, resolution, & CRS of reference target image...
    with rasterio.open(reference_path) as src:
        bounds = src.bounds
        res = src.res[0]
        crs = src.crs
    
    # calculate patch size in CRS units
    patch_size_units = patch_size * res

    # calculate distance from patch1 edge to adjacent patch2 edge in CRS units
    overlap_start_units = patch_size_units * (1 - patch_overlap)

    # initialize list to hold patch polygon geometries
    patches = []

    # initialize E-W starting point
    x = bounds.left

    # create patches while x is in AOI...
    while x < bounds.right:

        # initialize N-S starting point
        y = bounds.bottom

        # create patches while y is in AOI...
        while y < bounds.top:

            # create patch polygon geometry using x,y coordinates (lower left, SE)
            patch = box(x, y, x+patch_size_units, y+patch_size_units)

            # append patch ONLY if patch is fully contained withint AOI...
            if boundary_union.contains(patch):
                patches.append(patch)

        # update lower left coordinates...
            y += overlap_start_units
        x += overlap_start_units
    
    # create gdf of final patches...
    gdf = gpd.GeoDataFrame(geometry=patches, crs=crs)

    # create unique patch ID for each patch...
    if not name_prefix:
        gdf['patch_id'] = [f"{patch_size}_{int(patch_overlap*100)}_{i}" for i in range(1, len(gdf)+1)]
    else:
        gdf['patch_id'] = [f"{name_prefix}_{patch_size}_{int(patch_overlap*100)}_{i}" for i in range(1, len(gdf)+1)]
    
    # save as GeoJSON
    gdf.to_file(output_path, driver='GeoJSON')




def patches_get_stats(data_dir, modalities, patch_ids=None, cat_chans=['osm', 'nhd', 'mask']):
    """
    Compute global summary statistics for patch GeoTIFF channels within a dataset. 
    
    This function searches `data_dir` recursively for patch subdirectories
    containing GeoTIFF files, then aggregates per-channel statistics across all
    matching patch images for each modality listed in `modalities`. Modalities
    listed in `cat_chans` are skipped (assumed categorical). Statistics are
    computed over valid (unmasked) pixels only.

    Parameters
    ----------
    data_dir : str or os.PathLike
        Root directory containing patch subdirectories with GeoTIFF files.
    modalities : dict[str, list[str]]
        Mapping from modality name to a list of channel filename suffixes
        (e.g., {"dem": ["dem.tif"], "ep": ["ep_5x5.tif", "ep_11x11.tif"]}).
    patch_ids : sequence of str or None, default=None
        If provided, restricts computation to the specified patch IDs (prefixes).
    cat_chans : sequence of str or None, default=["osm", "nhd", "mask"]
        Modality names to skip (treated as categorical).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by modality channel suffix containing global mean, standard
        deviation, min, max, and nodata pixel counts aggregated across all images.
    """
    # find all sub-directories containing GeoTIFF files...
    patch_dirs = []
    for current_dir, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.tif'):
                patch_dirs.append(current_dir)
                break

    # initialize df to hold statistics
    df_stats = pd.DataFrame()

    # iterate through dict of modality names (keys) & list of channel path suffixes (values)...
    # EXAMPLE: {'dem': ['dem.tif'], 'ep': ['ep_5x5.tif', 'ep_11x11.tif']}
    for mod_name, channels in modalities.items():

        # skip categorical channels...
        if mod_name in cat_chans:
            continue

        # iterate through modality channels...
        for c in channels:
            
            # get image paths from all sub-directories found above...
            img_paths = []
            for pdir in patch_dirs:

                # for all images...
                if patch_ids is None:
                    img_paths.extend(glob.glob(f"{pdir}/*_{c}"))
                
                # for specified images only
                else:
                    for id in patch_ids:
                        img_paths.extend(glob.glob(f"{pdir}/{id}_{c}"))
            
            # remove duplicates of empty globs 
            img_paths = list(set(img_paths))

            # iterate through image channel paths & collect image stats...
            pixel_count = 0
            nodata_count = 0
            pixel_sum = 0.0
            pixel_sum2 = 0.0
            global_min = np.inf
            global_max = -np.inf

            for ip in img_paths:
                with rasterio.open(ip) as src:
                    data = src.read(1, masked=True)
                    total_pixels = data.size
                    vals = data.compressed()

                    pixel_count += vals.size
                    nodata_count += total_pixels - vals.size
                    pixel_sum += vals.sum()
                    pixel_sum2 += (vals**2).sum()

                    if vals.min() < global_min:
                        global_min = vals.min()
                    if vals.max() > global_max:
                        global_max = vals.max()

            # calculate global stats (mean & sample var/sd)...
            mean = pixel_sum / pixel_count
            var = (pixel_sum2 - (pixel_sum**2) / pixel_count) / (pixel_count - 1)
            sd = np.sqrt(np.float32(max(var, 0.0)))

            # save to df...
            df_stats.loc[c, 'mean'] = np.float32(mean)
            df_stats.loc[c, 'sd'] = np.float32(sd)
            df_stats.loc[c, 'min'] = np.float32(global_min)
            df_stats.loc[c, 'max'] = np.float32(global_max)
            df_stats.loc[c, 'nodata_count'] = np.float32(nodata_count)

    return df_stats




def patches_get_areas(patches_path, mask_path, label_space):
    """
    Compute per-patch class area proportions from a vector mask layer.

    Patch polygon layer and a categorical mask layer are intersected using
    `geopandas.overlay`, and the area of each mask class within each patch
    is computed and aggregated. Areas are normalized to proportions per patch.

    The output `pandas.DataFrame` contains one row per patch (`patch_id`) and 
    one column per class defined in `label_space`. The `label_space` argument 
    is used to enforce a consistent schema across runs: classes not present in 
    a given patch (or dataset) are included with proportion 0 to ensure stable 
    column structure.

    Parameters
    ----------
    patches_path : str or os.PathLike
        Path to a patch polygon GeoJSON (must include a `patch_id` column).
    mask_path : str or os.PathLike
        Path to a categorical mask GeoJSON (must include a `Symbol` column).
    label_space : sequence of str
        Ordered list of class labels to include as output columns. Ensures
        consistent column schema even if some classes are absent.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by patch with class area proportions as columns.
    """
    # read patch polygons & target mask GeoJSON version) as gdf's...
    patches = gpd.read_file(patches_path)
    mask = gpd.read_file(mask_path)

    # spatial overlay of mask intersecting each patch
    overlay = gpd.overlay(mask, patches, how='intersection')

    # calculate area of each class in mask for each patch
    overlay['area_in_patch'] = overlay.geometry.area           

    # convert areas to within-patch proportions...
    for patch in overlay['patch_id'].unique():

        # get array of areas for each class
        geology_areas = overlay.loc[overlay['patch_id'] == patch, 'area_in_patch'].values

        # get total area of patch (should all be same)
        total_area = overlay.loc[overlay['patch_id'] == patch, 'area_in_patch'].sum(axis=0)

        # calculate proportions of each class in each patch
        overlay.loc[overlay['patch_id'] == patch, 'area_in_patch'] = geology_areas / total_area

    # convert to float32 for consistency with all other data
    overlay['area_in_patch'] = overlay['area_in_patch'].astype(np.float32)                         

    # create new df for patches (rows) and class area proportions (columns)...
    areas_dict = {'patch_id': patches['patch_id']}                     # create dict of patch IDs
    df_areas = pd.DataFrame(areas_dict)                                # create df of patch IDs
    df_areas[label_space] = 0                                          # create cols for class labels
    df_areas[label_space] = df_areas[label_space].astype(np.float32)   # convert to float

    # group overlay gdf by patch and geologic map unit type
    # NOTE: there can be multiple instances of same class within each patch -> need to also group by Symbol for overall area
    grouped = overlay.groupby(['patch_id', 'Symbol']).agg({'area_in_patch':'sum'})

    # iterate over grouped patches...
    for (patch, symbol), row in grouped.iterrows():
        area = row['area_in_patch'].item()                           # get total area of class
        df_areas.loc[df_areas['patch_id']==patch, symbol] = area     # insert into new df

    return df_areas




def patches_get_labels(areas_path, threshold=None):
    """
    Calculate one-hot class labels using per-patch class area proportions CSV file. 
    
    This function reads a CSV of per-patch class area proportions (e.g.,
    output from `patch_get_areas`) and converts the class proportion columns
    into binary labels. By default, a class is labeled as present if its
    proportion is greater than 0. If `threshold` is provided, a class is
    labeled as present only if its proportion is greater than or equal to
    that threshold. The output preserves the original column schema (including 
    `patch_id`) and returns integer binary labels (0 or 1) for each class.

    Parameters
    ----------
    areas_path : str or os.PathLike
        Path to CSV file containing per-patch class area proportions; first 
        column assumed to be a non-class column (e.g., `patch_id`).
    threshold : float or None, default=None
        Minimum proportion required to mark a class as present. If None,
        any proportion > 0 is considered present.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same schema as the input, where class columns
        contain binary (0/1) labels.
    """


    # read per-patch class areas CSV into df
    areas = pd.read_csv(areas_path)

    # copy areas for consistent schema
    labels = areas.copy()

    # cast class columns to float
    class_cols = labels.columns[1:]
    labels[class_cols] = labels[class_cols].astype(np.float32)

    # calculate one-hot labels (using threshold if given)...
    # labels indicate class presence in patch (1 or more pixels of exposure)
    if threshold is None:                                       
        labels[class_cols] = labels[class_cols] > 0

    # labels indicate presence & area proportion threshold in patch
    else:                                                       
        labels[class_cols] = labels[class_cols] >= threshold
    
    # cast binary labels to int
    labels[class_cols] = labels[class_cols].astype(int)

    return labels




def img_to_patch(image_path, patches_gdf, output_dir):
    """
    Extract 1-channel images defined by patch polygon geometries and write 
    them as GeoTIFFs. 
    
    For each polygon in `patches_gdf`, this function crops 
    `image_path` to the polygon extent using `rasterio.mask.mask` and writes 
    the cropped raster to `output_dir`. Output filenames are prefixed with the 
    polygon's `patch_id` and the source image base name.

    Parameters
    ----------
    image_path : str or os.PathLike
        Path to the source raster from which patches are extracted.
    patches_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing patch polygons in a `geometry` column and a
        `patch_id` column used for naming outputs.
    output_dir : str or os.PathLike
        Output directory where patch GeoTIFFs will be written.

    Returns
    -------
    None
    """
    # get image name...
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    # open large source image to extract the smaller patch image from...
    with rasterio.open(image_path) as src:
        src_nodata = src.nodata

        # iterate through geodataframe of patch polygons...
        for _, row in patches_gdf.iterrows():
            
            # get spatial geometry of patch
            geom = row['geometry']
            
            # mask source image using current patch geometry to get patch image
            dst_image, dst_transform = mask(src, shapes=[geom], crop=True, filled=True, nodata=src_nodata)

            # copy source metadata & update for patch image...
            dst_meta = src.meta.copy()
            dst_meta.update({'driver':'GTiff', 
                             'height': dst_image.shape[1], 
                             'width': dst_image.shape[2], 
                             'transform': dst_transform,
                             'nodata': src_nodata})

            # save patch image using unique patch ID in gdf & source image name...
            output_path = f"{output_dir}/{row['patch_id']}_{image_name}.tif"
            with rasterio.open(output_path, 'w', **dst_meta) as dst:
                dst.write(dst_image)
