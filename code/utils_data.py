
#######################################################################################
# DATA DOWNLOAD FUNCTIONS
# ------------------
# Author: Matt Massey
# Last updated: 9/16/2024
# Purpose: Functions for downloading files from URLs, with emphasis on geospatial data sources.
#######################################################################################

import requests
import os
import glob
import shutil
import zipfile
import pandas as pd
import geopandas as gpd
import fiona
import rasterio



def download_zip(url, output_dir):
    """
    Function to download zip file, extract contents in the specified directory, and delete the zip file.

    Parameters
    ----------
    url : str
        Download URL for zip file.
    output_dir : str
        Directory path to save zip file and extract contents.

    Returns
    -------
    None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        zip_path = os.path.join(output_dir, 'download.zip')
        if response.status_code == 200:
            with open(zip_path, 'wb') as zip:
                zip.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip:
                zip.extractall(output_dir)
            os.remove(zip_path)
        else:
            print('Reponse code not 200 for downloading .zip...')
    except:
        print('Error downloading .zip...')


def download_tif(url, output_path):
    """
    Function to download TIFF file from a specified URL.

    Parameters
    ----------
    url : str
        Download URL for GeoTIFF file.
    output_path : str
        Path to save GeoTIFF.

    Returns
    -------
    None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            with open(output_path, 'wb') as tif:
                tif.write(response.content)
        else:
            print('Reponse code for URL not 200...')
    except:
        print(f"Error connecting to URL...\n{url}")


# def kyfromabove_tile_index(output_dir):
#     """
#     Function to fetch and extract KyFromAbove data tile index geodatabase, and save DEM, Aerial, and Lidar Point Cloud layers as GeoJSON files in specified directory. Same as 
    
#     Parameters
#     ----------
#     output_dir : str
#         Directory path for output files.
    
#     Returns
#     -------
#     None
#     """
#     url = r'https://ky.app.box.com/index.php?rm=box_download_shared_file&vanity_name=kymartian-kyaped-5k-tile-grids&file_id=f_1173114014568'

#     try:
#         response = requests.get(url)
#         response.raise_for_status()
        
#         if response.status_code == 200:

#             output_zip_path = f"{output_dir}/index.gdb.zip"
            
#             with open(output_zip_path, 'wb') as zip:
#                 zip.write(response.content)

#             with zipfile.ZipFile(output_zip_path, 'r') as zip:
#                 zip.extractall(output_dir)

#             gdb_path = glob.glob(f"{output_dir}/*.gdb")[0]
#             gdb_layers = fiona.listlayers(gdb_path)
            
#             for layer in gdb_layers:
#                 gdf = gpd.read_file(gdb_path, layer=layer)
#                 output_path = f"{output_dir}/{layer}.geojson"
#                 gdf.to_file(output_path, driver='GeoJSON')
            
#             os.remove(output_zip_path)
#             shutil.rmtree(gdb_path)
    
#     except:
#         print(f"Did not connect with download URL...\n{url}")


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



def check_image_alignment(input_paths, target='geology'):
    """
    Function to check alignment and registration of images in regards to the target image.

    Parameters
    ----------
    input_paths : list, tuple
        List or tuple of paths to images to check for alignment, including target image.
    target : str
        Name of target image that all other images should be aligned to.

    Returns
    -------
    Dataframe of image names, paths, and alignment metrics.
    """
    
    # names of images
    image_names = []
    for path in input_paths:
        name = os.path.basename(path)
        name = os.path.splitext(name)[0]
        image_names.append(name)
    
    # initialize new dataframe with names and paths and columns associated with alignment
    df = pd.DataFrame({'image':image_names, 'path':input_paths})
    df[['dtype', 'aligned', 'resolution_x', 'resolution_y', 'width', 'height', 'left', 'bottom', 'right', 'top']] = pd.NA

    # iterate through image paths and get values
    for image, path in zip(image_names, input_paths):
        with rasterio.open(path) as src:
            df.loc[df['image'] == image, 'dtype'] = src.meta['dtype']
            df.loc[df['image'] == image, 'resolution_x'] = src.res[0]
            df.loc[df['image'] == image, 'resolution_y'] = src.res[1]
            df.loc[df['image'] == image, 'width'] = src.width
            df.loc[df['image'] == image, 'height'] = src.height
            df.loc[df['image'] == image, 'left'] = src.bounds[0]
            df.loc[df['image'] == image, 'bottom'] = src.bounds[1]
            df.loc[df['image'] == image, 'right'] = src.bounds[2]
            df.loc[df['image'] == image, 'top'] = src.bounds[3]

    # get array of values from target
    target_alignment = df.loc[df['image']==target, 'resolution_x':].values

    # check if other images are aligned to target
    df['aligned'] = (df.loc[:, 'resolution_x':]==target_alignment).all(axis=1)
    
    return df
