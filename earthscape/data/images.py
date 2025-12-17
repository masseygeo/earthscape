

import numpy as np
from scipy.ndimage import gaussian_filter

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform


def image_to_reference_grid(input_path, reference_path, output_dtype=np.float32, output_path=None):
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
        src_data = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata

    with rasterio.open(reference_path) as ref:
        dst_meta = ref.meta.copy()
        dst_transform = ref.transform
        dst_crs = ref.crs

    dst_data = np.empty((ref.height, ref.width), dtype=np.float32)
    
    if output_dtype == np.uint8:
        dst_nodata = 0
    else:
        dst_nodata = np.nan

    reproject(source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.bilinear)

    if output_dtype == np.uint8:
        nodata_mask = dst_data == dst_nodata
        dst_data = np.clip(np.round(dst_data), 0, 255).astype(np.uint8)
        dst_data[nodata_mask] = dst_nodata

    dst_meta.update(count=1, 
                    dtype=np.dtype(output_dtype).name,
                    nodata=dst_nodata)
    
    if output_path is None:
        output_path = input_path

    with rasterio.open(output_path, "w", **dst_meta) as dst:
        dst.write(dst_data, 1)




def combine_aligned_images(image_paths, output_path, mode="binary", class_values=None):

    datasets = [rasterio.open(p) for p in image_paths]
    arrays = [ds.read(1).astype(np.float32) for ds in datasets]

    shape = arrays[0].shape
    nodata = np.nan

    if mode == "binary":
        out = np.full(shape, nodata, dtype=np.float32)

        any_valid = np.zeros(shape, dtype=bool)
        any_positive = np.zeros(shape, dtype=bool)

        for arr in arrays:
            valid = ~np.isnan(arr)
            any_valid |= valid
            any_positive |= (valid & (arr != 0))

        out[any_valid] = 0.0
        out[any_positive] = 1.0

    else:
        if class_values is None:
            class_values = list(range(1, len(arrays) + 1))

        out = np.full(shape, nodata, dtype=np.float32)

        any_valid = np.zeros(shape, dtype=bool)
        for arr in arrays:
            any_valid |= ~np.isnan(arr)
        out[any_valid] = 0.0

        # later images overwrite earlier ones
        for arr, cls in zip(arrays, class_values):
            present = (~np.isnan(arr)) & (arr != 0)
            out[present] = float(cls)

    meta = datasets[0].meta.copy()
    meta.update({
        "count": 1,
        "dtype": rasterio.float32,
        "nodata": np.nan
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(out, 1)

    for ds in datasets:
        ds.close()



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
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,                         # source CRS
            src.crs,                         # destination CRS
            src.width,                       # source width
            src.height,                      # source height
            *src.bounds,                     # source left, bottom, right, top coordinates 
            resolution=new_resolution        # destination resolution
            )
        
        # create metadata for new resampled image
        dst_meta = src.meta.copy()
        dst_meta.update({
            'driver': 'GTiff', 
            'width': dst_width, 
            'height': dst_height, 
            'transform': dst_transform
            })
        
        # write new image to file with new transform & metadata & resolution
        with rasterio.open(output_path, 'w', **dst_meta) as dst:
            reproject(
                source=rasterio.band(src, 1), 
                destination=rasterio.band(dst, 1), 
                src_transform=src.transform, 
                src_crs=src.crs, 
                dst_transform=dst_transform, 
                dst_crs=src.crs, 
                resampling=Resampling.bilinear
                )
            


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

