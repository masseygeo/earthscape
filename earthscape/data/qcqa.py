
import os
import pandas as pd
import numpy as np
import rasterio



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
    df[['dtype', 'nodata', 'aligned', 'resolution_x', 'resolution_y', 'width', 'height', 'left', 'bottom', 'right', 'top']] = pd.NA

    # iterate through image paths and get values
    for image, path in zip(image_names, input_paths):
        with rasterio.open(path) as src:
            # data = src.read(1, masked=True)
            # n_nodata = int(np.ma.count_masked(data))
            df.loc[df['image'] == image, 'dtype'] = src.meta['dtype']
            df.loc[df['image'] == image, 'nodata_val'] = src.nodata
            # df.loc[df['image'] == image, 'nodata_n'] = int(np.ma.count_masked(data))
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