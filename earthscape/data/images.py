
import numpy as np
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.merge import merge



def image_to_reference_grid(input_path, reference_path, output_dtype=np.float32, output_path=None, resampling=Resampling.bilinear):
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

    # read input image to be re-projected & aligned...
    with rasterio.open(input_path) as src:
        src_data = src.read(1).astype(np.float32, copy=False)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
        src_meta = src.meta.copy()

    # convert input nodata sentinel to nan (if it exists & isn't already nan)
    if src_nodata is not None:
        if not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
            src_data[np.isclose(src_data, src_nodata)] = np.nan

    # read reference image...
    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_h = ref.height
        ref_w = ref.width

    # initialize float32 array & nodata=nan in workspace regardless of final output...
    out_data = np.full((ref_h, ref_w), np.nan, dtype=np.float32)

    # reproject input image to reference image...
    reproject(
        source=src_data,
        destination=out_data,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        dst_nodata=np.nan,
        resampling=resampling
        )
    
    # finalize output dtype & nodata...
    if output_dtype == np.uint8:
        nodata_mask = np.isnan(out_data)
        out_data = np.clip(np.round(out_data), 0, 255).astype(np.uint8)
        out_data[nodata_mask] = 0
        out_nodata = 0
    else: 
        out_nodata = np.nan

    # update metadata...
    src_meta.update({
        'driver': 'GTiff',
        'count': 1, 
        'height': ref_h, 
        'width': ref_w, 
        'transform': ref_transform, 
        'crs': ref_crs,
        'dtype': np.dtype(output_dtype).name, 
        'nodata': out_nodata
        })
    
    # define path for output image
    if output_path is None:
        output_path = input_path

    # save output image as GeoTIFF
    with rasterio.open(output_path, "w", **src_meta) as dst:
        dst.write(out_data, 1)




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

    # set up parameters for merging tiles...
    merge_kwargs = {
        'indexes': [band_number], 
        'dtype': np.float32, 
        'nodata': np.nan
        }

    # open tile images
    images = [rasterio.open(tile_path) for tile_path in tile_paths]

    # merge tiles (depending on resampling)...
    if resample:
        mosaic, mosaic_transform = merge(images, res=resample, resampling=Resampling.bilinear, **merge_kwargs)
    else:
        mosaic, mosaic_transform = merge(images, **merge_kwargs)

    # get unique nodata values...
    nodata_values = []
    for src in images:
        if src.nodata is None:
            continue
        if isinstance(src.nodata, float) and np.isnan(src.nodata):
            continue
        nodata_values.append(src.nodata)
    nodata_values = sorted(set(nodata_values))

    # replace nodata values with nan...
    for nd in nodata_values:
        mosaic[np.isclose(mosaic, nd)] = np.nan

    # update metadata....
    mosaic_meta = images[0].meta.copy()
    mosaic_meta.update({
        'driver': 'GTiff', 
        'height': mosaic.shape[1], 
        'width': mosaic.shape[2], 
        'transform': mosaic_transform, 
        'crs': images[0].crs, 
        'count': mosaic.shape[0], 
        'nodata': np.nan, 
        'dtype': np.dtype(np.float32).name
        })
    
    # write output GeoTIFF mosaic image...
    with rasterio.open(output_path, 'w', **mosaic_meta) as output:
        for i in range(mosaic.shape[0]):
            output.write(mosaic[i, :, :], i+1)
    
    # close opened tile images
    for src in images:
        src.close()





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

