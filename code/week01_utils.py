from osgeo import gdal
import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import matplotlib.pyplot as plt


def tif_to_tif_conversion(input_multi_tif_path, output_single_tif_path):
    """
    Function to convert a GeoTIFF with multiple associated metadata files into one single consolidated GeoTIFF file.
    
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


def shapefile_to_tif_conversion(input_shapefile_path, shapefile_field, reference_tif_path, output_tif_path):
    """
    Convert a GIS shapefile of categorical polygons to a GeoTIFF using the same geospatial metadata as a reference GeoTIFF.
    
    Parameters:
    shapefile_path (str): Path to the input shapefile.
    shapefile_field (str): Field name containing value of interest.
    reference_tif_path (str): Path to the reference GeoTIFF.
    output_tif_path (str): Path to the output GeoTIFF.
    
    Returns:
    None    
    """
    # handle existing output file...
    if os.path.isfile(output_tif_path):
        print('Output file already exists...')
    
    # proceed if output file does not exist...
    else:
        
        ##### get geospatial metadata information for file conversion...
        with rasterio.open(reference_tif_path) as ref:

            meta = ref.meta.copy()      # get copy metadata (don't overwrite existing)
            transform = ref.transform   # get transformation (pixels to geo. coordinates)
            width = ref.width           # get width
            height = ref.height         # get height

            # define dtype (8 bit unsigned) for categorical geologic map (not part of ref)
            dtype = 'uint8'


        ##### prepare shapefile...
        # read shapefile as geodataframe (each GIS polygon has geometry and geologic label)
        gdf = gpd.read_file(input_shapefile_path)

        # ensure shapefile is using same coordinate reference system as reference tif
        gdf = gdf.to_crs(meta['crs'])

        # get array of unique field names
        geologic_symbols = gdf[shapefile_field].unique()

        # create dictionary for unique field names to integers (starting at 1)
        geologic_symbol_to_int = {symbol:idx for idx, symbol in enumerate(geologic_symbols, start=1)}

        # create new field mapped to integers
        new_field_name = f"{shapefile_field}_int"
        gdf[new_field_name] = gdf[shapefile_field].map(geologic_symbol_to_int)
        
        # get tuples of polygons and geologic map unit labels for rasterization
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[new_field_name]))


        ##### convert shapefile to geotiff...
        # create the output GeoTIFF from the shapefile
        with rasterio.open(
            output_tif_path,     # path to save new geotiff
            'w',                 # designate write mode
            driver='GTiff',      # use GeoTIFF driver
            height=height,       # use height of reference tif
            width=width,         # use width of reference tif
            count=1,             # number of bands/channels (1 for geologic map units)
            dtype=dtype,         # dtype for categorical geologic map raster
            crs=meta['crs'],     # use coordinate reference system of reference tif
            transform=transform, # use affine transformation of reference tif
            nodata=0             # update no-data value to 0 after integer-symbol mapping
        ) as output:

            # convert vector polygons to raster polygons using reference tif metadata
            rasterized = rasterize(
                shapes,                      # tuple of polygons and labels
                out_shape=(height, width),   # height and width of reference tif
                transform=transform,         # affine transformation of reference tif
                dtype=dtype)                 # dtype for categorical geologic map raster

            # write the rasterized polygons to the new output GeoTIFF on band 1
            output.write(rasterized, 1)


def plot_raster(input_raster_path):
    """
    Function to display GeoTIFF image and mask regions of 'no data'.

    Parameters:
    raster_path (str):

    Returns:
    None
    """
    # open the raster
    with rasterio.open(input_raster_path) as src:

        # read the raster data for first band
        data = src.read(1)
        
        # mask no-data values
        no_data_value = src.nodata

        if no_data_value is not None:
            data = np.ma.masked_equal(data, no_data_value)

        # plot raster...
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='gist_earth')
        plt.axis('off')
        plt.colorbar(shrink=0.5)
        plt.show()


def raster_stats(input_raster_path):
    """
    Function to calculate and print basic summary statistics of GeoTIFF raster.

    Parameters:
    input_raster_path (str): Path to raster.

    Returns:
    None
    """
    # open the raster and calculate the statistics...
    with rasterio.open(input_raster_path) as src:
        data = src.read(1, masked=True)
        print('Min: ', np.min(data))
        print('Max: ', np.max(data))
        print('Mean: ', np.mean(data))
        print('Standard Deviation: ', np.std(data))
        print('No Data value (not included in stats): ', src.nodata)
        print('Raster width (pixels): ', src.width)
        print('Raster height (pixels): ', src.height)
        print('Raster resolution: ', src.res)

        if src.crs.is_geographic:
            units = 'degrees'
        elif src.crs.is_projected:
            units = src.crs.linear_units
        else:
            units = 'unknown'

        print('Raster geographic units: ', units)


