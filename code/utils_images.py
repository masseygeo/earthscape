
#######################################################################################
# GEOSPATIAL IMAGE MANIPULATION FUNCTIONS
# ------------------
# Author: Matt Massey
# Last updated: 9/16/2024
# Purpose: Custom functions for manipulating geospatial images, specific to surficial geologic map dataset curation.
#######################################################################################

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from scipy.ndimage import gaussian_filter



def clip_image_to_boundary(input_path, boundary_path):
    """
    Function to to clip an image to an area of interest polygon and save the clipped image as a new GeoTIFF.

    Parameters
    ----------
    input_path : str
        Path to image to be clipped.
    boundary_path : str
        Path to boundary polygon GeoJSON.

    Returns
    -------
    None
    """
    boundary = gpd.read_file(boundary_path)

    with rasterio.Env(CHECK_DISK_FREE_SPACE='FALSE'):
        with rasterio.open(input_path) as src:
            
            if src.nodata is None:
                nodata_value = np.nan
                data_type = 'float32'
            else:
                nodata_value = src.nodata
                data_type = src.meta['dtype']

            out_image, out_transform = mask(src, shapes=boundary.geometry, crop=True, nodata=nodata_value)
            out_meta = src.meta.copy()
            out_meta.update({'driver':'GTiff', 
                            'height':out_image.shape[1], 
                            'width':out_image.shape[2], 
                            'transform':out_transform, 
                            'crs': src.crs, 
                            'nodata': nodata_value, 
                            'dtype': data_type})
            
            new_file_dir = os.path.dirname(input_path)
            new_filename = os.path.splitext(os.path.basename(input_path))[0] + '_clip.tif'
            output_tif_path = f"{new_file_dir}/{new_filename}"

            with rasterio.open(output_tif_path, 'w', **out_meta) as output:
                for i in range(out_image.shape[0]):
                    output.write(out_image[i, :, :], i+1)


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


def convert_image_dtype(input_path):
    """
    Function to convert image to float32 dtype.
    
    Parameters
    ----------
    input_tif_path : string
        Path to GeoTIFF image to be converted.
    
    Returns
    -------
    None
    """
    if '_f32.tif' in input_path:
        return
    
    output_path = input_path[:-4] + '_f32.tif'
    
    with rasterio.Env(CHECK_DISK_FREE_SPACE='FALSE'):
        with rasterio.open(input_path) as src:
            nodata_value = src.nodata
            data = src.read()
            out_data = data.astype(rasterio.float32)
            
            if nodata_value is None:
                nodata_value = np.nan
                out_data[data == src.meta['nodata']] = nodata_value

            out_meta = src.meta.copy()
            out_meta.update({'dtype': rasterio.float32, 
                             'nodata': nodata_value})
        
        with rasterio.open(output_path, 'w', **out_meta) as output:
            for i in range(out_data.shape[0]):
                output.write(out_data[i, :, :], i+1)


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
        dst_transform, dst_width, dst_height = calculate_default_transform(src.crs,           # source CRS
                                                                           src.crs,           # destination CRS
                                                                           src.width,         # source width
                                                                           src.height,        # source height
                                                                           *src.bounds,       # source left, bottom, right, top coordinates 
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
