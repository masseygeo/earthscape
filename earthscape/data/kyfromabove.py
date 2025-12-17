
from earthscape.data.downloads import download_tif, download_zip

import os
import glob
import numpy as np
import geopandas as gpd
import fiona
import rasterio
from rasterio.warp import Resampling
from rasterio.merge import merge



def download_data_tiles(index_path, id_field, url_field, output_dir):
    """
    Function to read KyFromAbove Tile Index GeoJSON, download relevant GeoTIFFs using the download URLs from a specified attribute, and then save each GeoTIFF to the specified output directory.

    Parameters
    ----------
    index_path : str
        Path to GeoJSON.
    id_field : str
        Attribute name of GeoJSON containing unique ID for file naming.
    url_field : str
        Attribute name of GeoJSON containing the download URL.
    output_dir : str
        Directory where TIFF(s) will be downloaded.

    Returns
    -------
    None
    """
    gdf = gpd.read_file(index_path)
    
    for _, tile in gdf.iterrows():
        tile_id = tile[id_field]
        url = tile[url_field]
        content_type = url[-3:]

        if len(glob.glob(f"{output_dir}/*{tile_id}*")) > 0:
            continue

        if content_type == 'tif':
            output_path = f"{output_dir}/{tile_id}.tif"
            download_tif(url, output_path)

        elif content_type == 'zip':
            download_zip(url, output_dir)

        else:
            print('Download is not .tif or .zip...')



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



def mosaic_image_tiles(tile_paths, output_path, band_number, resample=None):
    """
    Function to create a new single GeoTIFF mosaic from multiple smaller image tiles.

    Parameters
    ----------
    tile_paths : str
        List of paths to GeoTIFF tiles.
    output_path : str
        Path for new output mosaic GeoTIFF.
    band_number : int
        Band (channel) to mosaic.
    resample : int (optional)
        Resolution of output image. If not provided, output image will have the same resolution as input image tiles.

    Returns
    -------
    None
    """
    images = [rasterio.open(tile_path) for tile_path in tile_paths]

    if resample:
        # mosaic, mosaic_transform = merge(images, indexes=[band_number], res=resample, resampling=Resampling.bilinear, nodata=np.nan)
        mosaic, mosaic_transform = merge(images, indexes=[band_number], res=resample, resampling=Resampling.bilinear)
    else:
        mosaic, mosaic_transform = merge(images, indexes=[band_number], nodata=np.nan)

    mosaic_meta = images[0].meta.copy()
    mosaic_meta.update({'driver': 'GTiff', 
                        'height': mosaic.shape[1], 
                        'width': mosaic.shape[2], 
                        'transform': mosaic_transform, 
                        'crs': images[0].crs, 
                        'count': mosaic.shape[0]})
    
    with rasterio.open(output_path, 'w', **mosaic_meta) as output:
        for i in range(mosaic.shape[0]):
            output.write(mosaic[i, :, :], i+1)
    
    for src in images:
        src.close()
