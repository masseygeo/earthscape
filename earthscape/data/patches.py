
import os
import pandas as pd
import numpy as np
import geopandas as gpd
# import fiona
from shapely.geometry import box
import rasterio
from rasterio.mask import mask



def create_image_patches(reference_path, patch_size, patch_overlap, boundary_path, output_path, name_prefix=None):
    """
    Function to create geospatial polygons that represent square image patch locations saved as a GeoJSON. The size of the image patches (assumed to be square) and the proportion of overlap between adjacent patches is specified. Each patch will have a unique id created from the patch_size, patch_overlap, and a unique number.

    Parameters
    ----------
    reference_path : str
        Path to a reference GeoTIFF image that represents the area where patches will be created.
    patch_size : int or float
        Size of the square patch in pixels.
    patch_overlap : float
        Proportion of overlap between adjacent patches.
    boundary_path : str
        Path to area of interest boundary GeoJSON file (should be aligned with boundaries of reference_path image) to ensure patch polygons intersect.
    output_path : str
        Path for output patch polygon GeoJSON file.

    Returns
    -------
    None.
    """

    boundary = gpd.read_file(boundary_path)

    with rasterio.open(reference_path) as src:
        bounds = src.bounds
        res = src.res[0]
        crs = src.crs
        
    patch_size_units = patch_size * res
    overlap_start_units = patch_size_units * (1 - patch_overlap)

    patches = []
    x = bounds.left
    while x < bounds.right:
        y = bounds.bottom
        while y < bounds.top:
            patch = box(x, y, x+patch_size_units, y+patch_size_units)

            if patch.within(boundary.geometry).any():
                patches.append(patch)
            y += overlap_start_units
        x += overlap_start_units
    
    gdf = gpd.GeoDataFrame(geometry=patches, crs=crs)

    if not name_prefix:
        gdf['patch_id'] = [f"{patch_size}_{int(patch_overlap*100)}_{i}" for i in range(1, len(gdf)+1)]
    else:
        gdf['patch_id'] = [f"{name_prefix}_{patch_size}_{int(patch_overlap*100)}_{i}" for i in range(1, len(gdf)+1)]
        
    gdf.to_file(output_path, driver='GeoJSON')




def extract_patch(image_path, patches_gdf, output_dir):
    """
    Function to use extract image patches from a geodataframe of patch polygyons.

    Parameters
    ----------
    image_path : str
        Path to image to extract patch.
    patches_gdf : geodataframe
        Geodataframe of patch polygons.
    output_dir : str
        Path for output image patch. Unique patch id from geodataframe will be used for prefix filename.

    Returns
    -------
    None
    """

    # get image name...
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    with rasterio.open(image_path) as src:
        src_nodata = src.nodata

        for _, row in patches_gdf.iterrows():

            geom = row['geometry']

            dst_image, dst_transform = mask(src, shapes=[geom], crop=True, filled=True, nodata=src_nodata)

            dst_meta = src.meta.copy()
            dst_meta.update({'driver':'GTiff', 
                             'height': dst_image.shape[1], 
                             'width': dst_image.shape[2], 
                             'transform': dst_transform,
                             'nodata': src_nodata})
        
            output_path = f"{output_dir}/{row['patch_id']}_{image_name}.tif"
    
            with rasterio.open(output_path, 'w', **dst_meta) as dst:
                dst.write(dst_image)




def calculate_patch_areas(patches_path, mask_path, label_space):
    
    ##### read data as dataframes...
    patches = gpd.read_file(patches_path)
    mask = gpd.read_file(mask_path)


    ##### intersect map units with patches, then compute areas...
    overlay = gpd.overlay(mask, patches, how='intersection')   # spatial overlay of geologic map units intersecting each patch
    overlay['area_in_patch'] = overlay.geometry.area           # calculate area of each geologic map unit in each area


    ##### convert to within-patch proportions...
    for patch in overlay['patch_id'].unique():
        geology_areas = overlay.loc[overlay['patch_id'] == patch, 'area_in_patch'].values          # get array of geologic map unit areas 
        total_area = overlay.loc[overlay['patch_id'] == patch, 'area_in_patch'].sum(axis=0)        # get total area covered by map unit in each patch
        overlay.loc[overlay['patch_id'] == patch, 'area_in_patch'] = geology_areas / total_area    # calculate proportions of each map unit in each patch
    overlay['area_in_patch'] = overlay['area_in_patch'].astype(np.float32)                         # convert to float32 for consistency with all other data


    ##### create new dataframe for patches (rows) and class area proportions (columns)...
    initialize_dict = {'patch_id': patches['patch_id']}
    df_areas = pd.DataFrame(initialize_dict)
    df_areas[label_space] = 0
    df_areas[label_space] = df_areas[label_space].astype(np.float32)


    ##### group overlay and insert areas into df...
    # group overlay gdf by patch and geologic map unit type
    # NOTE: there can be multiple map units of same type within each patch, so need to also group by Symbol for overall area
    grouped = overlay.groupby(['patch_id', 'Symbol']).agg({'area_in_patch':'sum'})
    for (patch, symbol), row in grouped.iterrows():
        area = row['area_in_patch'].item()                           # get area of unique map unit in specific patch
        df_areas.loc[df_areas['patch_id']==patch, symbol] = area     # insert into dataframe

    return df_areas




def calculate_one_hots(areas_path, threshold=None):

    ##### read areas into df...
    areas = pd.read_csv(areas_path)
    labels = areas.copy()
    class_cols = labels.columns[1:]
    labels[class_cols] = labels[class_cols].astype(np.float32)


    ##### calculate one-hot labels using threshold (if given)...
    if threshold is None:                                       # labels for presence (1 or more pixels) of class in patch 
        labels[class_cols] = labels[class_cols] > 0
    else:                                                       # labels for presence & footprint of class in patch
        labels[class_cols] = labels[class_cols] >= threshold
    labels[class_cols] = labels[class_cols].astype(int)         # cast bool to int

    return labels