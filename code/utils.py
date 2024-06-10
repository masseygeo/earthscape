import os
from osgeo import gdal

import rasterio
from rasterio.features import rasterize

import geopandas as gpd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import json


def multi_tif_to_single_tif_conversion(input_multi_tif_path, output_single_tif_path):
    """
    Function to convert a GeoTIFF with multiple associated metadata files (.tfw, .xml, .ovr) into one single consolidated GeoTIFF file.
    
    Parameters:
    input_multi_tif_path (str): Path to the input GeoTIFF file with multiple associated files.
    output_single_tif_path (str): Path to the single consolidated output GeoTIFF file.
    
    Returns:
    None
    """
    # handle existing output file...
    if os.path.isfile(output_single_tif_path):
        print('Output file already exists...')

    # proceed if output file does not exist...
    else:
        # open the input geotiff with multiple metadata files in 'read only' mode
        input_dataset = gdal.Open(input_multi_tif_path, gdal.GA_ReadOnly)
        
        # instantiate driver for geotiff files
        driver = gdal.GetDriverByName('GTiff')

        # create a copy of the input geotiff with multiple metadata files to the output path
        driver.CreateCopy(output_single_tif_path, input_dataset, 0)
        
        # close the dataset
        input_dataset = None



def shapefile_to_tif_conversion(input_shapefile_path, shapefile_field, reference_tif_path, output_tif_path, dtype='uint8'):
    """
    Convert a GIS shapefile of categorical polygons to a GeoTIFF using the same geospatial metadata as a reference GeoTIFF.
    
    Parameters:
    shapefile_path (str): Path to the input shapefile.
    shapefile_field (str): Field name containing value of interest.
    reference_tif_path (str): Path to the reference GeoTIFF.
    output_tif_path (str): Path to the output GeoTIFF.
    dtype (str): Data type of output GeoTIFF. Default is uint8.
    
    Returns:
    None    
    """
    # handle existing output file...
    if os.path.isfile(output_tif_path):
        print('Output file already exists...')

    else:
        ##### get geospatial metadata...
        with rasterio.open(reference_tif_path) as ref:
            meta = ref.meta.copy()      # get copy metadata (don't overwrite existing)

        ##### shapefile...
        # read shapefile as geodataframe (each GIS polygon has geometry and geologic label)
        gdf = gpd.read_file(input_shapefile_path)

        # ensure shapefile is using same coordinate reference system as reference tif
        gdf = gdf.to_crs(meta['crs'])

        # encode field strings to integers...
        geologic_symbols = gdf[shapefile_field].unique()   # get array of unique field names
        geologic_symbol_to_int = {symbol:idx for idx, symbol in enumerate(geologic_symbols, start=1)}   # create dictionary mapper
        new_field_name = f"{shapefile_field}_int"   # new integer field name
        gdf[new_field_name] = gdf[shapefile_field].map(geologic_symbol_to_int)   # add new field to gdf and map to integers
        
        # get tuples of polygons and geologic map unit labels for rasterization
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[new_field_name]))

        ##### convert shapefile to geotiff...
        # create the output GeoTIFF from the shapefile
        with rasterio.open(
            output_tif_path,              # path to save new geotiff
            'w',                          # designate write mode
            driver='GTiff',               # use GeoTIFF driver
            height=meta['height'],        # use height of reference tif
            width=meta['width'],          # use width of reference tif
            count=1,                      # number of bands/channels (1 for geologic map units)
            dtype=dtype,                  # dtype for categorical geologic map raster
            crs=meta['crs'],              # use coordinate reference system of reference tif
            transform=meta['transform'],  # use affine transformation of reference tif
            nodata=0                      # no-data value to 0 after integer-symbol mapping
        ) as output:

            # convert vector polygons to raster polygons using reference tif metadata
            rasterized = rasterize(
                shapes,                                      # tuple of polygons and labels
                out_shape=(meta['height'], meta['width']),   # height and width of reference tif
                transform=meta['transform'],                 # affine transformation of reference tif
                dtype=dtype)                                 # dtype for categorical field (integers)

            # write the rasterized polygons to the new output GeoTIFF on band 1
            output.write(rasterized, 1)

        ##### write json file with symbol to integer mapping metadata...
        output_dir = os.path.splitext(output_tif_path)[0]       # get directory and output tif basename (no file extension)
        output_dictionary_path = output_dir + '.json'           # create metadata filename with .json extension
        with open(output_dictionary_path, 'w') as file:         # open the output json file to write
            json.dump(geologic_symbol_to_int, file, indent=4)   # write the mapping dictioanry to the json



def geotiff_linear_units(src):
    """Function to return geographic linear units from an opened Rasterio dataset image object.
    
    Parameters:
    src (object): Opened Rasterio image object.
    
    Returns:
    units (str): Geographic linear units of image
    """
    if src.crs.is_geographic:
            units = 'degrees'
    elif src.crs.is_projected:
        units = src.crs.linear_units
    else:
        units = 'unknown'
    
    return units


