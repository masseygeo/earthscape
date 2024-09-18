
#######################################################################################
# VECTOR GIS MANIPULATION FUNCTIONS
# ------------------
# Author: Matt Massey
# Last updated: 9/16/2024
# Purpose: Custom functions for manipulating vector GIS data (shapefiles or GeoJSONs), specific to surficial geologic map dataset curation.
#######################################################################################

import os
import glob
import json
import numpy as np
from math import ceil
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import box



def gis_to_image(input_path, output_path, output_resolution, attribute):
    """
    Function to convert vector geospatial file to GeoTIFF image file with a given resolution and categorical attribute. Output GeoTIFF file is of float32 dtype with NaN representing nodata values.

    Parameters
    ----------
    input_path : str
        Path to input GeoJSON or Shapefile.
    output_path : str
        Path for output GeoTIFF.
    output_resolution : int
        Resolution of GeoTIFF in native spatial units of input GIS file.
    attribute : str
        Name of categorical attribute in GIS file for assigning pixel values.

    Returns
    -------
    None
    """
    # read input GIS file as geodataframe
    gdf = gpd.read_file(input_path)

    # if input is polygon or multipolygon, then apply 0 buffer to mitigate potential geometry errors
    if gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
        gdf['geometry'] = gdf['geometry'].buffer(0)
    
    # get bounding coordinates & output width and height (using desired resolution)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = ceil((maxx - minx) / output_resolution)
    height = ceil((maxy - miny) / output_resolution)

    # calculate transform for output image
    transform = from_origin(west=minx, north=maxy, xsize=output_resolution, ysize=output_resolution)

    # get attribute values for pixel value assignments
    values = gdf[attribute].unique()

    # assign each category to integer
    mapper = {key:value for value, key in enumerate(values, start=1)}

    # create new geodataframe attribute of categorical integer assignments
    gdf[f"{attribute}_int"] = gdf[attribute].apply(lambda x: mapper.get(x, np.nan))


    # get list of geometries and associated values
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[f"{attribute}_int"])]
    
    # rasterize shapes using output height, width, and transform
    output_image = rasterize(shapes = shapes, 
                             out_shape = (height, width), 
                             transform = transform, 
                             all_touched = True, 
                             fill = np.nan, 
                             dtype = rasterio.float32)
    
    # create metadata for output image
    output_meta = {'driver': 'GTiff', 
                   'height': height, 
                   'width': width, 
                   'transform': transform, 
                   'count': 1, 
                   'dtype': output_image.dtype, 
                   'nodata': np.nan, 
                   'crs': gdf.crs.to_string()}
    
    # write image and metadata to GeoTIFF
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        dst.write(output_image, 1)
    
    # write mapping dictionary of integers and categories to JSON
    output_json_path = output_path.replace('.tif', '.json')
    with open(output_json_path, 'w') as file:
        json.dump(mapper, file, indent=4)




def multiple_gis_to_reference_image(input_paths, reference_path, output_path):
    """
    Function to combine multiple geospatial vector GIS features into a new GeoTIFF image aligned with a reference image. In the case of overlapping features, priority for pixel values in the final image will be given to the last feature. Background space will be given a value of 0 and additional features will be given sequential integers in increments of 1.

    Parameters
    ----------
    input_paths : list or tuple
        List of path to vector GIS features in GeoJSON(s) and/or Shapefile(s).
    reference_path : str
        Path to reference GeoTIFF image.
    output_path : str
        Path to output GeoTIFF image.
    
    Returns
    -------
    None
    """
    with rasterio.open(reference_path) as src:

        shapes_all = []
        features = ['background']

        for val, path in enumerate(input_paths, start=1):
            
            feature = os.path.basename(path)
            feature = os.path.splitext(feature)[0]
            features.append(feature)

            gdf = gpd.read_file(path)

            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            shapes = [(geom, val) for geom in gdf.geometry]
            shapes_all.extend(shapes)

        output_image = rasterize(shapes=shapes_all, 
                                 out_shape=(src.height, src.width), 
                                 transform=src.transform, 
                                 fill=0, 
                                 all_touched=True, 
                                 dtype=rasterio.float32)
        
        mask = src.dataset_mask()
        output_image = np.where(mask, output_image, src.nodata)

        output_meta = src.meta.copy()
        output_meta.update({'driver': 'GTiff', 
                            'count': 1, 
                            'dtype':rasterio.float32})
        
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(output_image.astype(rasterio.float32), 1)
        
        mapper = {k:v for v,k in enumerate(features)}
        output_json_path = output_path.replace('.tif', '.json')
        with open(output_json_path, 'w') as meta:
            json.dump(mapper, meta, indent=4)









def get_intersecting_index_tiles(input_path, boundary_path, output_path):
    """
    Function to extract tile index polygons from an input GeoJSON intersecting an area specified by another GeoJSON, and then saving the subset tile index polygons as a new GeoJSON.
    
    Parameters
    ----------
    geojson_path : str
        Path to input tile index GeoJSON.
    boundary_path : str
        Path to area of interest GeoJSON.
    output_geojson_path : str
        Path for output GeoJSON.
    
    Returns
    -------
    None
    """
    gdf_geojson = gpd.read_file(input_path)
    gdf_boundary = gpd.read_file(boundary_path)
    if gdf_boundary.crs != gdf_geojson.crs:
        gdf_geojson = gdf_geojson.to_crs(gdf_boundary.crs)
    gdf_intersect = gpd.sjoin(left_df=gdf_geojson, right_df=gdf_boundary, how='inner')
    gdf_intersect.to_file(output_path, driver='GeoJSON')



def get_contained_and_edge_tile_paths(index_path, boundary_path, data_dir):
    """
    Function to get lists of aerial imagery tile paths that are completely contained or intersecting the edge of the boundary area.

    Parameters
    ----------
    index_path : str
        Path to GeoJSON of aerial imagery tile index polygons for the area of interest.
    boundary_path : str
        Path to GeoJSON of the area of interest polygon.
    data_dir : str
        Directory path containing the aerial imagery tile data. Directory must have only GeoTIFF files.

    Returns
    -------
    within_poly_paths : list
        List of paths of tiles completely contained within the area of interest.
    edge_poly_paths : list
        List of paths of tiles intersecting the boundary of the area of interest.
    """
    gdf_index = gpd.read_file(index_path)
    gdf_boundary = gpd.read_file(boundary_path)

    if gdf_boundary.crs != gdf_index.crs:
        gdf_index = gdf_index.to_crs(gdf_boundary.crs)
    
    boundary = gdf_boundary.iloc[0].geometry
    within_polygons = gdf_index[gdf_index.geometry.within(boundary)]
    edge_polygons = gdf_index[~gdf_index.index.isin(within_polygons.index)]

    within_poly_paths = []
    for _, row in within_polygons.iterrows():
        tile = row['TileName']
        path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
        within_poly_paths.append(path)

    edge_poly_paths = []
    for _, row in edge_polygons.iterrows():
        tile = row['TileName']
        path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
        edge_poly_paths.append(path)

    return within_poly_paths, edge_poly_paths



def clip_gis_to_boundary(input_path, boundary_path, output_path, gdb_layer=None):
    """
    Function to clip GIS spatial data to the extent of an area of interest polygon and save the clipped feature(s) as a new GeoJSON file.

    Parameters
    ----------
    input_path : str
        Path to GIS spatial input file. If this is a geodatabase (.gdb), then the gdb_layer argument must be specified.
    boundary_path : str
        Path to area of interest polygon.
    output_path : str
        Path for output GeoJSON.
    gdb_layer : str (optional)
        Name of geodatabase layer to be clipped. Default is None.

    Returns
    -------
    None
    """
    if not gdb_layer:
        gdf_input = gpd.read_file(input_path)
    else:
        gdf_input = gpd.read_file(input_path, layer=gdb_layer)
    gdf_input = gdf_input.explode(ignore_index=True, index_parts=False)
    gdf_boundary = gpd.read_file(boundary_path)

    if gdf_input.crs != gdf_boundary.crs:
        gdf_input = gdf_input.to_crs(gdf_boundary.crs)

    gdf_output = gpd.clip(gdf_input, mask=gdf_boundary)
    gdf_output.to_file(output_path, driver='GeoJSON')



def create_image_patches(reference_path, patch_size, patch_overlap, boundary_path, output_path):
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
            if patch.intersects(boundary.geometry).any():
                patches.append(patch)
            y += overlap_start_units
        x += overlap_start_units
    
    gdf = gpd.GeoDataFrame(geometry=patches, crs=crs)
    gdf['patch_id'] = [f"{patch_size}_{int(patch_overlap*100)}_{i}" for i in range(1, len(gdf)+1)]
    gdf.to_file(output_path, driver='GeoJSON')






