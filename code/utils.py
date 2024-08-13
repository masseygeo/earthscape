import os
import json
import numpy as np
import geopandas as gpd
from osgeo import gdal
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
import matplotlib.pyplot as plt

####################
# FILE CONVERSION
####################

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
    
    Parameters
    ----------
    shapefile_path (str): Path to the input shapefile.
    shapefile_field (str): Field name containing value of interest.
    reference_tif_path (str): Path to the reference GeoTIFF.
    output_tif_path (str): Path to the output GeoTIFF.
    geo_dict (dict): Dictionary of field names and encoded integers to use as values in output GeoTIFF.
    
    Returns
    -------
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
   

####################
# DATA UTILITY
####################

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


def total_class_area(input_geo_path, geo_codes_to_names_dict):
    """
    Function to return dictionary of geologic map symbol names (keys) and total geographic area of unique map unit classes:

    Parameters:
    input_geo_path (str): Path to geologic map image.
    geo_codes_to_names_dict (dict): Dictionary of encoded integers (keys) to geologic map names (values).

    Returns:
    names_areas (dict): Dictionary of geologic map names (keys) to total geographic areas (values).
    """
    
    # open geologic map image...
    with rasterio.open(input_geo_path) as geo:

        # read data as masked array (no data masked); compress to 1D array
        data = geo.read(1, masked=True).compressed()

        # get x and y pixel resolutions to calculate geographic area
        x_res, y_res = geo.res

        # get unique values (encoded integers) and count of pixels for each code
        codes, counts = np.unique(data, return_counts=True)

        # create dictionary of codes (keys) and pixel counts (values)
        codes_counts_dict = dict(zip(codes, counts))

        # calculate total geographic area for each unique map unit, then sort by area
        codes_areas = {int(key):(value * x_res * y_res) for key, value in codes_counts_dict.items()}
        codes_areas = dict(sorted(codes_areas.items(), key=lambda item: item[1], reverse=True))

        # ensure code-to-name keys are int dtype; then map areas to key instead of code
        geo_codes_to_names_dict = {int(key):value for key, value in geo_codes_to_names_dict.items()}
        
        names_areas = {geo_codes_to_names_dict[key]:value for key, value in codes_areas.items() if key in geo_codes_to_names_dict.keys()}

        return names_areas


def get_polygon_metrics(gdf, spatial_resolution=1):
    """Function to calculate area and minimum bounding box width and height. Optionally, can set spatial_resolution to convert geographic units to pixels.

    Parameters:
    gdf (dataframe): Geopandas GeoDataframe of polygons.
    spatial_resolution (int or float type): Spatial resolution of pixels in geographic units; default is 1, which yields the same as geographic units.

    Returns:
    gdf (dataframe): Geopandas GeoDataframe with new columns for area, min_box_width, and min_box_height.
    """
    # calculate area
    gdf['area'] = gdf['geometry'].area

    # get dataframe of minimum bounding box coordinates of each polygon
    bounds = gdf['geometry'].bounds

    # calculate minimum bounding box widths and heights for each polygon; np.abs to handle any c.r.s.
    gdf['min_box_width'] = np.abs(bounds['maxx'] - bounds['minx'])
    gdf['min_box_height'] = np.abs(bounds['maxy'] - bounds['miny'])

    if spatial_resolution != 1:
        gdf['min_box_width'] = gdf['min_box_width'] / spatial_resolution
        gdf['min_box_height'] = gdf['min_box_height'] / spatial_resolution

    return gdf


def extract_patch(geotiff_path, centroid, patch_size):

    with rasterio.open(geotiff_path) as src:
        
        # Convert centroid coordinates to row and column
        row, col = src.index(centroid.x, centroid.y)
        
        # Calculate the window size
        half_size = patch_size // 2
        window = Window(col - half_size, row - half_size, patch_size, patch_size)
        
        # Calculate the full patch with nodata values
        full_patch = np.full((patch_size, patch_size), src.nodata, dtype=src.dtypes[0])
        
        # Calculate the region of interest inside the source image
        left = max(0, col - half_size)
        right = min(src.width, col + half_size)
        top = max(0, row - half_size)
        bottom = min(src.height, row + half_size)
        
        # Calculate the corresponding positions in the full patch
        patch_left = half_size - (col - left)
        patch_right = patch_left + (right - left)
        patch_top = half_size - (row - top)
        patch_bottom = patch_top + (bottom - top)
        
        # Read the valid part of the image
        window = Window(left, top, right - left, bottom - top)
        patch_data = src.read(1, window=window)
        
        # Insert the valid data into the full patch array
        full_patch[patch_top:patch_bottom, patch_left:patch_right] = patch_data
        
        # Mask the nodata values
        full_patch = np.ma.masked_equal(full_patch, src.nodata)

        # Get the transform for the patch
        transform = src.window_transform(window)
        
        return full_patch, transform


def clip_shapefile_to_patch(gdf, centroid, patch_width, patch_height, spatial_resolution=5):
    
    # Define the centroid and dimensions
    centroid_x, centroid_y = centroid.x, centroid.y

    # Calculate the bounding box
    minx = centroid_x - (patch_width * spatial_resolution) / 2
    maxx = centroid_x + (patch_width * spatial_resolution) / 2
    miny = centroid_y - (patch_height * spatial_resolution) / 2
    maxy = centroid_y + (patch_height * spatial_resolution) / 2

    # Create the patch area as a GeoDataFrame
    patch_area = box(minx, miny, maxx, maxy)
    patch_gdf = gpd.GeoDataFrame([1], geometry=[patch_area], crs=gdf.crs)

    # Clip the original GeoDataFrame with the patch area
    clipped_gdf = gpd.clip(gdf, patch_gdf)

    return clipped_gdf


####################
# DATA VIZ
####################
def plot_dem_histogram(input_path, title=None):
    
    fig, ax = plt.subplots(figsize=(4,4))

    with rasterio.open(input_path) as dem:
        data = dem.read(1, masked=True)
        data = data.filled(np.nan)
        ax.hist(data.flatten(), bins=50, density=True, align='mid', linewidth=0.5, edgecolor='k')
        dem_min = str(round(np.nanmin(data), 1))
        dem_median = str(round(np.nanmedian(data), 1))
        dem_mean = str(round(np.nanmean(data), 1))
        dem_max = str(round(np.nanmax(data), 1))
        label=f"Min: {dem_min}\nMedian: {dem_median}\nMean: {dem_mean}\nMax: {dem_max}"
        ax.text(0.99, 0.99, label, ha='right', va='top', transform=ax.transAxes)
        ax.set_xlabel('Elevation (ft)')
        ax.set_ylabel('Density')
        ax.set_title(title, style='italic')
        plt.show()


def geo_symbology_colormap(input_metadata_path):
    """
    Function to create custom Matplotlib color map for integer-encoded string names.

    Parameters:
    input_metadata_path (str): Path to metadata .json with string field names and encoded integer values.

    Returns:
    geo_codes (dict): Dictionary of encoded integers (key) and original string field names (value).
    geo_codes_rgb (dict): Dictionary of encoded integers (key) and custom colors (value).
    """
    # open json metadata for string map unit (key) to integer code (value)
    with open(input_metadata_path, 'r') as meta:
        geo_names = json.load(meta)

    # define custom rgb color (value) mapping for geologic map unit names (key) from KGS standardized colors
    # NOTE: update with full list of colors from kgsmap
    geo_names_rgb = {'Qr':(176,172,214),
                     'af1':(99,101,102), 
                     'Qal':(253,245,164), 
                     'Qaf':(255,161,219), 
                     'Qat':(249,228,101), 
                     'Qc':(214,201,167), 
                     'Qca':(196,157,131)}
    
    # reverse mapping using encoded integer (key) and string map unit (value)
    geo_codes = {value:key for key, value in geo_names.items()}

    # color mapping to integer (key) and rgb percentage (rgb/255) (value)
    geo_codes_rgb = {geo_names[key]:tuple(v/255 for v in value) for key, value in geo_names_rgb.items()}

    return geo_codes, geo_codes_rgb