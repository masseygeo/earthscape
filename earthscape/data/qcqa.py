
from earthscape.constants import *

import os
from datetime import datetime
import json
import pandas as pd
import numpy as np
import rasterio




def check_image_alignment(input_paths, target='mask'):
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
    df[['dtype', 'nodata', 'aligned', 'res_x', 'res_y', 'width', 'height', 'left', 'bottom', 'right', 'top']] = pd.NA

    # iterate through image paths and get values
    for image, path in zip(image_names, input_paths):
        with rasterio.open(path) as src:
            df.loc[df['image'] == image, 'dtype'] = src.meta['dtype']
            df.loc[df['image'] == image, 'nodata'] = src.nodata
            df.loc[df['image'] == image, 'res_x'] = src.res[0]
            df.loc[df['image'] == image, 'res_y'] = src.res[1]
            df.loc[df['image'] == image, 'width'] = src.width
            df.loc[df['image'] == image, 'height'] = src.height
            df.loc[df['image'] == image, 'left'] = src.bounds[0]
            df.loc[df['image'] == image, 'bottom'] = src.bounds[1]
            df.loc[df['image'] == image, 'right'] = src.bounds[2]
            df.loc[df['image'] == image, 'top'] = src.bounds[3]

    # get array of alignment values from target...
    alignment_cols = ['res_x', 'res_y', 'width', 'height', 'left', 'bottom', 'right', 'top']
    target_alignment = df.loc[df['image']=='mask', alignment_cols].values
    # target_alignment = df.loc[df['image']==target, 'res_x':].values

    # check if other images are aligned to target
    # df['aligned'] = (df.loc[:, 'res_x':]==target_alignment).all(axis=1)
    df['aligned'] = (df.loc[:, alignment_cols] == target_alignment).all(axis=1)
    
    return df



def validate_patches(patch_path, patch_size):

    with rasterio.open(patch_path) as src:
        data = src.read(1)
        h = src.height
        w = src.width
        nd_count = np.isnan(data).sum()

        if (h != patch_size) or (w != patch_size) or (nd_count > 0):
            return patch_path
        


def create_metadata(area_name, num_patches, num_tifs, img_meta, patch_size, overlap, output_path, sources=SRC_URLS):
    dataset = {}
    dataset['EarthScape'] = {}
    dataset['EarthScape']['version'] = VERSION
    dataset['EarthScape']['created'] = datetime.now().strftime('%Y-%m-%d, %H:%M')
    dataset['EarthScape']['patch size'] = patch_size
    dataset['EarthScape']['overlap'] = str(int(overlap * 100))+'%'
    dataset['EarthScape']['num patches'] = int(num_patches)
    dataset['EarthScape']['num channels'] = int(num_tifs / num_patches)
    dataset['EarthScape']['total images'] = num_tifs

    dataset['Geospatial'] = {}
    dataset['Geospatial']['crs'] = img_meta['crs'].to_string()
    dataset['Geospatial']['units'] = img_meta['crs'].to_dict()['units']
    dataset['Geospatial']['resolution'] = img_meta['transform'][0]
    dataset['Geospatial']['nodata'] = img_meta['nodata']

    dataset['Data Sources'] = {}
    for k,v in sources.items():
        if any(s in k for s in ['kyfromabove', 'nhd', 'osm', area_name.replace(' ','').lower()]):
            dataset['Data Sources'][k] = v

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)