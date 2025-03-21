
#######################################################################################
# VECTOR GIS MANIPULATION FUNCTIONS
# ------------------
# Author: Matt Massey
# Last updated: 9/21/2024
# Purpose: Custom functions for manipulating vector GIS data (shapefiles or GeoJSONs), specific to surficial geologic map dataset curation.
#######################################################################################

import os
import json
import numpy as np
from math import ceil
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from shapely.geometry import box
import fiona



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
    # dictionary of values and mapped 

    # read input GIS file as geodataframe
    gdf = gpd.read_file(input_path)

    # if input is polygon or multipolygon, then apply 0 buffer to mitigate potential geometry errors
    if gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
        gdf['geometry'] = gdf['geometry'].buffer(0.1)
    
    # get bounding coordinates & output width and height (using desired resolution)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = ceil((maxx - minx) / output_resolution)
    height = ceil((maxy - miny) / output_resolution)

    # calculate transform for output image
    transform = from_origin(west=minx, north=maxy, xsize=output_resolution, ysize=output_resolution)

    # # get attribute values for pixel value assignments
    # values = gdf[attribute].unique()

    # # assign each category to integer
    # mapper = {key:value for value, key in enumerate(values, start=1)}

    mapper = {'af1': 1, 'Qal': 2, 'Qaf': 3, 'Qat': 4, 'Qc': 5, 'Qca': 6, 'Qr': 7}
    # mapper = {key:val for key, val in mapper.items() if key in gdf['Symbol'].unique()}

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



def multiple_gis_to_reference_image(input_paths, reference_path, output_path, binary=True):
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
            


            if not binary:
                shapes = [(geom, val) for geom in gdf.geometry]
            
            else:
                shapes = [(geom, 1) for geom in gdf.geometry]



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
        
        if not binary:
            mapper = {k:v for v,k in enumerate(features)}
            output_json_path = output_path.replace('.tif', '.json')
            with open(output_json_path, 'w') as meta:
                json.dump(mapper, meta, indent=4)



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
            # if patch.intersects(boundary.geometry).any():
            #     patches.append(patch)
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





def reassign_mapunit_symbol(input_path, map_units, output_path=None):

    # read geology GeoJSON to geodataframe
    gdf = gpd.read_file(input_path)

    # apply small buffer to fill any slivers or gaps
    gdf['geometry'] = gdf['geometry'].buffer(0.1)

    # separate selected map units & remaining units
    gdf_subset = gdf[gdf['Symbol'].isin(map_units)].copy()
    gdf_remaining = gdf.loc[~gdf.index.isin(gdf_subset.index)].copy()

    # iterate through selected polygons to find neighbor that shares longest border
    for idx, subset_row in gdf_subset.iterrows():

        # isolate neighbor polygons
        neighbors = gdf_remaining[gdf_remaining.touches(subset_row.geometry)]

        # calculate shared borders of all neighbors (if valid geometries or not empty)
        if neighbors.geometry.is_valid.all():
            shared_lengths = neighbors.geometry.apply(lambda x: subset_row.geometry.intersection(x).length)

        # calculate index of neighbor with longest border
        if len(shared_lengths) > 0:
            max_length_idx = shared_lengths.idxmax()

        # replace selected polygon with adjacent neighbor
        gdf.loc[idx, 'Symbol'] = gdf.loc[max_length_idx, 'Symbol']

    # save new GeoJSON
    if not output_path:
        output_path = input_path
    gdf.to_file(output_path, driver='GeoJSON')




def get_aoi_index_polygons(input_path, boundary_path, output_dir):

    # read buffered boundary into geodataframe
    boundary = gpd.read_file(boundary_path)

    # get list of layers in index geodatabase
    index_layers = fiona.listlayers(input_path)

    # iterate through layers
    for index in index_layers:
        
        # extract dem index
        if 'dem' in index.lower():

            # read dem index as geodataframe
            dem_index = gpd.read_file(input_path, layer=index)

            # perform spatial join between buffered boundary & statewide index (only tiles that intersect index)
            intersect = gpd.sjoin(left_df=dem_index, right_df=boundary, how='inner')

            # define output path for dem index
            output_path = f"{output_dir}/dem_index.geojson"

            # write selected tiles to GeoJSON
            if not os.path.isfile(output_path):
                intersect.to_file(output_path, driver='GeoJSON')
        
        # extract aerial imagery index
        elif 'aerial' in index.lower():
            aerial_index = gpd.read_file(input_path, layer=index)
            intersect = gpd.sjoin(left_df=aerial_index, right_df=boundary, how='inner')
            output_path = f"{output_dir}/aerial_index.geojson"
            if not os.path.isfile(output_path):
                intersect.to_file(output_path, driver='GeoJSON')
