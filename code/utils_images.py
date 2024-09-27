
#######################################################################################
# GEOSPATIAL IMAGE MANIPULATION FUNCTIONS
# ------------------
# Author: Matt Massey
# Last updated: 9/21/2024
# Purpose: Custom functions for manipulating geospatial images, specific to surficial geologic map dataset curation.
#######################################################################################

import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from scipy.ndimage import gaussian_filter



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
        mosaic, mosaic_transform = merge(images, indexes=[band_number], res=resample, resampling=Resampling.bilinear)
    else:
        mosaic, mosaic_transform = merge(images, indexes=[band_number])

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



def resample_image(input_path, new_resolution, output_path):
    """
    Function to resample a GeoTIFF image to a new resolution and save as a new GeoTIFF.

    Parameters
    ----------
    input_path : str
        Path to the input GeoTIFF image to be resampled.
    new_resolution : int or float
        Resolution for the new, resampled image.
    output_path : str
        Path for the new, resampled GeoTIFF image.

    Returns
    -------
    None
    """

    with rasterio.open(input_path) as src:

        # calculate the new transform and dimensions based on the new resolution
        dst_transform, dst_width, dst_height = calculate_default_transform(src.crs,      # source CRS
                                                                           src.crs,      # destination CRS
                                                                           src.width,    # source width
                                                                           src.height,   # source height
                                                                           *src.bounds,  # source left, bottom, right, top coordinates 
                                                                           resolution=new_resolution)     # destination resolution
        
        # create metadata for new resampled image
        dst_meta = src.meta.copy()
        dst_meta.update({'driver': 'GTiff', 
                         'width': dst_width, 
                         'height': dst_height, 
                         'transform': dst_transform})
        
        # write new image to file with new transform & metadata & resolution
        with rasterio.open(output_path, 'w', **dst_meta) as dst:
            reproject(source=rasterio.band(src, 1), 
                      destination=rasterio.band(dst, 1), 
                      src_transform=src.transform, 
                      src_crs=src.crs, 
                      dst_transform=dst_transform, 
                      dst_crs=src.crs, 
                      resampling=Resampling.cubic)



def filter_image(input_path, sigma):
    """
    Function to apply a Gaussian filter to an input image. See scipy.ndimage.gaussin_filter for more information regarding filter.
    
    Parameters
    ----------
    input_path : str
        Path to input image.
    sigma : int, float
        Standard deviation for Gaussian function.

    Returns
    -------
    None
    """

    with rasterio.open(input_path) as src:
        data = src.read(1, masked=True)
        dst_data = gaussian_filter(input=data, sigma=sigma)
        dst_meta = src.meta.copy()
    
    output_path = input_path

    with rasterio.open(output_path, 'w', **dst_meta) as dst:
        dst.write(dst_data, 1)



def image_to_reference_image(input_path, reference_path, output_path=None):
    """
    Function to register and align an input image to a reference image then save the new aligned GeoTIFF. If the output path is not provided, the original input image is overwritten.

    Parameters
    ----------
    input_path : str
        Path to input image to be reprojected and aligned.
    reference_path : str
        Path to reference image to match alignment.
    output_path : str (optional)
        Path for output GeoTIFF. If not provided, the input image is overwritten.

    Returns
    -------
    None
    """

    with rasterio.open(input_path) as src:
        src_profile = src.profile
        src_data = src.read(1)

    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile
        ref_data = ref.read(1, masked=True)
    
    dst_data = np.empty_like(ref_data)

    reproject(source=src_data, 
              destination=dst_data, 
              src_transform=src_profile['transform'], 
              src_crs=src_profile['crs'], 
              dst_transform=ref_profile['transform'], 
              dst_crs=ref_profile['crs'], 
              dst_res=ref_profile['transform'][0], 
              resampling=Resampling.bilinear)

    dst_meta = ref.meta.copy()

    if not output_path:
        output_path = input_path

    with rasterio.open(output_path, 'w', **dst_meta) as dst:
        dst.write(dst_data, 1)



def extract_patch(image_path, patches_gdf, output_dir):
    """
    Function to use extract image patches from a geodataframe of patch polygyons.

    Parameters
    ----------
    image_path : str
        Path to image to extract patch.
    patches_gdf : geodataframe
        Geodataframe of patch polygons.
    output_dir : str
        Path for output image patch. Unique patch id from geodataframe will be used for prefix filename.

    Returns
    -------
    None
    """
    image_name = os.path.basename(image_path)
    image_name = os.path.splitext(image_name)[0]

    with rasterio.open(image_path) as src:

        for _, row in patches_gdf.iterrows():

            geom = row['geometry']

            dst_image, dst_transform = mask(src, shapes=[geom], crop=True, filled=True, nodata=-999999)

            dst_meta = src.meta.copy()
            dst_meta.update({'driver':'GTiff', 
                             'height':dst_image.shape[1], 
                             'width':dst_image.shape[2], 
                             'transform':dst_transform})
        
            output_path = f"{output_dir}/{row['patch_id']}_{image_name}.tif"
    
            with rasterio.open(output_path, 'w', **dst_meta) as dst:
                dst.write(dst_image)
