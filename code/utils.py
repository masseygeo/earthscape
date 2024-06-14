import os
from osgeo import gdal
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
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

    else:
        # open the input geotiff with multiple metadata files in 'read only' mode
        input_dataset = gdal.Open(input_multi_tif_path, gdal.GA_ReadOnly)
        
        # instantiate driver for geotiff file
        driver = gdal.GetDriverByName('GTiff')

        # create a copy of the input geotiff with multiple metadata files to the output path
        driver.CreateCopy(output_single_tif_path, input_dataset, 0)
        
        # close the dataset
        input_dataset = None



def shapefile_to_tif_conversion(input_shapefile_path, shapefile_field, reference_tif_path, output_tif_path, geo_dict=None):
    """
    Convert a GIS shapefile of categorical polygons to a GeoTIFF using the same geospatial metadata as a reference GeoTIFF. The shapefile value field to use for the output GeoTIFF will be encoded as integers, either as a new set of integers for each unique value (default) or for a subset of values using a provided mapping dictionary.
    
    Parameters:
    shapefile_path (str): Path to the input shapefile.
    shapefile_field (str): Field name containing value of interest.
    reference_tif_path (str): Path to the reference GeoTIFF.
    output_tif_path (str): Path to the output GeoTIFF.
    geo_dict (dict): Dictionary of field names and encoded integers to use as values in output GeoTIFF.
    
    Returns:
    None    
    """
    # handle existing output file...
    if os.path.isfile(output_tif_path):
        print('Output file already exists...')

    else:
        ##### reference tif...
        # copy metadata from reference tif
        with rasterio.open(reference_tif_path) as ref:
            meta = ref.meta.copy() 

        ##### shapefile...
        # read shapefile as geodataframe (each GIS polygon has geometry and geologic label)
        gdf = gpd.read_file(input_shapefile_path)

        # ensure shapefile has same coordinate reference system as reference tif
        gdf = gdf.to_crs(meta['crs'])

        # encode field strings to integers...
        geologic_symbols = gdf[shapefile_field].unique()   # array of unique values in field
        new_field_name = f"{shapefile_field}_int"          # new feature name for encoded values
        gdf[new_field_name] = 0                            # create new feature; initialize as 0 for "no data"\
        gdf[new_field_name] = gdf[new_field_name].astype(int)

        # handle if string-to-integer mapper is not provided in parameters
        if not geo_dict:
            # create new dictionary mapper for all unique values
            geo_dict = {symbol:idx for idx, symbol in enumerate(geologic_symbols, start=1)}

            # use map to update attributes in new encoded feature field
            gdf[new_field_name] = gdf[shapefile_field].apply(lambda x: geo_dict.get(x, 0))
       
        else:
            # use provied dictionary mapper to update attributes in new encoded feature field
            gdf[new_field_name] = gdf[shapefile_field].apply(lambda x: geo_dict.get(x, 0))

        # get tuples of polygons and encoded integers for rasterization
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[new_field_name]))
        
        ##### convert shapefile to geotiff...
        with rasterio.open(
            output_tif_path,              # path to save new geotiff output
            'w',                          # write mode
            driver='GTiff',               # use GeoTIFF driver
            height=meta['height'],        # same height as reference tif
            width=meta['width'],          # same width as reference tif
            count=1,                      # 1 band/channel for encoded values
            dtype='uint8',                # unsigned 8-bit integer dtype for encoded integers
            crs=meta['crs'],              # same coordinate reference system as reference tif
            transform=meta['transform'],  # same affine transformation as reference tif
            nodata=0                      # set no-data value to 0 for integer-symbol mapping
        ) as output:

            # convert vector polygons in tuple to raster polygons using reference tif metadata
            rasterized = rasterize(
                shapes,                                      # tuple of polygons and labels
                out_shape=(meta['height'], meta['width']),   # height and width of reference tif
                transform=meta['transform'],                 # affine transformation of reference tif
                dtype='uint8')                               # unsigned 8-bit integer dtype

            # write the rasterized polygons to the new output GeoTIFF on band 1
            output.write(rasterized, 1)

        ##### write json file with string-to-integer mapping metadata...
        output_metadata_path = output_tif_path.replace('.tif', '.json')   # same output path but json file
        with open(output_metadata_path, 'w') as file:                     # open output json file to write
            json.dump(geo_dict, file, indent=4)                           # write mapping dictioanry to json
   


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


