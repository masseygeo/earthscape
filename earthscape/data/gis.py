

from earthscape.constants import SG_MAPPING
import os
import json
import numpy as np
from math import ceil
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
import shapely
from shapely.geometry import Polygon



def gis_to_image(input_path, output_path, output_resolution, multiclass_col=None, multiclass_map=SG_MAPPING):
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
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf['geometry'] = gdf['geometry'].buffer(0.1)
    
    # get bounding coordinates & output width and height (using desired resolution)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = ceil((maxx - minx) / output_resolution)
    height = ceil((maxy - miny) / output_resolution)

    # calculate transform for output image
    transform = from_origin(west=minx, north=maxy, xsize=output_resolution, ysize=output_resolution)

    if multiclass_col is None:
        shapes = [(geom, 1) for geom in gdf.geometry]

    else:
        gdf[f"{multiclass_col}_int"] = gdf[multiclass_col].apply(lambda x: multiclass_map.get(x, np.nan))
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[f"{multiclass_col}_int"])]
    
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



def create_aoi_polygon(input_path, output_path):
    gdf = gpd.read_file(input_path)
    u = shapely.union_all(gdf.geometry.values, grid_size=0.2)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda p: p.area)
    aoi = Polygon(u.exterior)
    gdf_aoi = gpd.GeoDataFrame(geometry=[aoi], crs=gdf.crs)
    gdf_aoi.to_file(output_path, driver='GeoJSON')