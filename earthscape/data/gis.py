
import numpy as np
from math import ceil
import geopandas as gpd
import shapely
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy.ndimage import gaussian_filter


def vec_to_img(input_path, output_path, output_resolution, multiclass_map, multiclass_col=None):
    """
    Rasterize a vector geospatial dataset to a single-band GeoTIFF. 
    
    The output raster grid is defined from the input dataset bounds and
    `output_resolution` (in the dataset CRS units). If `multiclass_col` is
    provided, values are mapped to numeric class codes using `multiclass_map`
    and burned into the raster; otherwise all geometries are burned with value 1.
    The output is written as float32 with NaN used as the fill value for pixels
    not covered by any geometry.

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to an input vector dataset readable by GeoPandas (e.g., GeoJSON,
        Shapefile).
    output_path : str or os.PathLike
        Destination path for the output GeoTIFF.
    output_resolution : int or float
        Pixel size in the native units of the input dataset CRS.
    multiclass_map : dict of {str: int}
        Mapping from class names to numeric codes.
    multiclass_col : str or None, default=None
        Attribute column name to use for multiclass rasterization. If None,
        produces a binary raster (burn value 1).

    Returns
    -------
    None
        Writes the raster to `output_path`.
    """
    # read input GIS file as geodataframe
    gdf = gpd.read_file(input_path)

    # if input is polygon or multipolygon, then apply 0 buffer to mitigate potential geometry errors
    if gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf['geometry'] = gdf['geometry'].buffer(0.1)
    
    # get bounding coordinates & output width and height (using desired resolution)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = ceil((maxx - minx) / output_resolution)
    height = ceil((maxy - miny) / output_resolution)

    # calculate transform for output image
    transform = from_origin(west=minx, north=maxy, xsize=output_resolution, ysize=output_resolution)

    if multiclass_col is None:
        shapes = [(geom, 1) for geom in gdf.geometry]

    else:
        gdf[f"{multiclass_col}_int"] = gdf[multiclass_col].apply(lambda x: multiclass_map.get(x, np.nan))
        shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[f"{multiclass_col}_int"])]
    
    # rasterize shapes using output height, width, and transform
    output_image = rasterize(shapes = shapes, 
                             out_shape = (height, width), 
                             transform = transform, 
                             all_touched = True, 
                             fill = np.nan, 
                             dtype = rasterio.float32)
    
    # create metadata for output image
    output_meta = {'driver': 'GTiff', 
                   'height': height, 
                   'width': width, 
                   'transform': transform, 
                   'count': 1, 
                   'dtype': output_image.dtype, 
                   'nodata': np.nan, 
                   'crs': (gdf.crs.to_string() if gdf.crs is not None else None)
                   }
    
    # write image and metadata to GeoTIFF
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        dst.write(output_image, 1)




def vec_clip(input_path, boundary_path, output_path, gdb_layer=None):
    """
    Function to clip GIS spatial data to the extent of an AOI polygon 
    and save the clipped feature(s) as a new GeoJSON file.

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to GIS spatial input file. If this is a geodatabase (.gdb), then the gdb_layer argument must be specified.
    boundary_path : str or os.PathLike
        Path to area of interest polygon.
    output_path : str or os.PathLike
        Path for output GeoJSON.
    gdb_layer : str or None, default=None
        Name of geodatabase layer to be clipped.

    Returns
    -------
    None
    """
    if gdb_layer is None:
        gdf_input = gpd.read_file(input_path)
    else:
        gdf_input = gpd.read_file(input_path, layer=gdb_layer)
    gdf_input = gdf_input.explode(ignore_index=True, index_parts=False)
    gdf_boundary = gpd.read_file(boundary_path)

    if gdf_input.crs != gdf_boundary.crs:
        gdf_input = gdf_input.to_crs(gdf_boundary.crs)

    gdf_output = gpd.clip(gdf_input, mask=gdf_boundary)
    gdf_output.to_file(output_path, driver='GeoJSON')


    

def vec_to_aoi(input_path, output_path, grid_size=0.2):
    """
    Build an area of interest (AOI) polygon from vector geometries and write it to GeoJSON.
    
    The input dataset is read with GeoPandas and all geometries are unioned into a single
    footprint using `shapely.union_all`. If the union results in a MultiPolygon, the
    largest polygon by area is selected as the AOI. The optional `grid_size` parameter
    controls snapping during the union operation (in the units of the input CRS).

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to an input vector dataset readable by GeoPandas.
    output_path : str or os.PathLike
        Destination path for the output AOI GeoJSON.
    grid_size : float, default=0.2
        Grid size passed to `shapely.union_all` to snap coordinates during union. Units
        are the native units of the input CRS.

    Returns
    -------
    None
        Writes a single-feature GeoJSON to `output_path`.
    """
    gdf = gpd.read_file(input_path)
    u = shapely.union_all(gdf.geometry.values, grid_size=grid_size)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda p: p.area)
    aoi = Polygon(u.exterior)
    gdf_aoi = gpd.GeoDataFrame(geometry=[aoi], crs=gdf.crs)
    gdf_aoi.to_file(output_path, driver='GeoJSON')




def img_to_reference(input_path, reference_path, output_dtype=np.float32, output_path=None, resampling=Resampling.bilinear):
    """
    Reproject and resample a single-band raster to match a reference raster grid. 
    
    The input raster is reprojected to the CRS of `reference_path` and resampled
    onto the reference transform, height, and width. Intermediate processing is
    performed in float32 with NaN used as the internal nodata representation.
    The final output is written as a GeoTIFF either to `output_path` or in-place
    (overwriting `input_path`).

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to the input raster (single band) to be reprojected.
    reference_path : str or os.PathLike
        Path to the reference raster whose CRS, transform, width, and height
        define the output grid.
    output_dtype : numpy.dtype or type, default=np.float32
        Output data type. Must be either `np.float32` or `np.uint8`.
    output_path : str or os.PathLike or None, default=None
        Destination path for the output GeoTIFF. If None, the input file is
        overwritten.
    resampling : rasterio.enums.Resampling, default=Resampling.bilinear
        Resampling method used during reprojection (e.g., bilinear, nearest).

    Returns
    -------
    None
        Writes the aligned raster to disk.

    Notes
    -------
    Currently supported output dtypes are `np.float32` and `np.uint8`. For
    `np.uint8`, values are rounded, clipped to [0, 255], and NaN pixels are set
    to 0 (nodata). For `np.float32`, NaN is preserved as nodata.
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




def img_to_int(src_path, dst_path=None):
    """
    Convert a single-band float raster patch to uint8.

    The function reads a raster, replaces NaN and infinite values
    (NaN -> 0, +inf -> 255, -inf -> 0), clips values to the 0-255 range,
    casts to uint8, removes nodata metadata, and writes the result.

    Parameters
    ----------
    src_path : str
        Path to input single-band raster.
    dst_path : str, optional
        Output path. If None, overwrites `src_path`.

    Notes
    -----
    - Assumes valid data are already in the 0-255 range.
    - Any NaN or infinite values are coerced to valid uint8 values.
    - The output raster has dtype uint8 and no nodata value set.
    """

    # open image & get data array & geospatial metadata...
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_data = np.nan_to_num(src_data, nan=0.0, posinf=255.0, neginf=0.0)
        dst_meta = src.meta.copy()

    # clip to valid uint8 range & cast to uint8
    dst_data = np.clip(src_data, 0, 255).astype(np.uint8)

    # update metadata
    dst_meta.update({
        'nodata': None, 
        'dtype': 'uint8'
        })
    
    # save cast image...
    if not dst_path:
        dst_path = src_path
    
    with rasterio.open(dst_path, 'w', **dst_meta) as dst:
        dst.write(dst_data, 1)




def img_resample(input_path, new_resolution, output_path, resampling=Resampling.bilinear):
    """
    Resample a single-band raster to a new spatial resolution. 
    T
    he raster is resampled in its native CRS to `new_resolution`, preserving
    its spatial extent. Output dimensions and transform are recalculated using
    `calculate_default_transform`, and the resampled raster is written to a
    new GeoTIFF.

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to the input raster (single band).
    new_resolution : int or float or tuple of int or float, len 2
        Target pixel size in the units of the raster CRS. May be a single
        value (square pixels) or (xres, yres).
    output_path : str or os.PathLike
        Destination path for the resampled GeoTIFF.
    resampling : rasterio.enums.Resampling, default=Resampling.bilinear
        Resampling method used during interpolation.

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
                resampling=resampling
                )




def img_filter(input_path, sigma):
    """
    Apply a Gaussian smoothing filter to a single-band raster in-place. 
    
    The input raster is read as a masked array and smoothed using
    `scipy.ndimage.gaussian_filter` with the specified standard deviation.
    Masked (nodata) pixels are preserved and not allowed to contaminate
    valid regions. The filtered raster then overwrites the original file.

    Parameters
    ----------
    input_path : str or os.PathLike
        Path to the input raster (single band).
    sigma : float
        Standard deviation of the Gaussian kernel, in pixel units.

    Returns
    -------
    None
    """

    with rasterio.open(input_path) as src:
        data = src.read(1, masked=True)
        nodata = src.nodata
        mask = data.mask

        # use 0 only as temporary filler for filter...
        filled = data.filled(0)
        filtered = gaussian_filter(input=filled, sigma=sigma)

        # restore nodata pixels and get final filtered data & metadata...
        filtered[mask] = nodata
        dst_data = filtered
        dst_meta = src.meta.copy()

    # overwrite input image with smoothed image...
    output_path = input_path
    with rasterio.open(output_path, 'w', **dst_meta) as dst:
        dst.write(dst_data, 1)




def images_mosaic(tile_paths, output_path, band_number, resample=None):
    """
    Merge multiple raster tiles into a single mosaic GeoTIFF. 
    
    Tiles are opened and merged using `rasterio.merge.merge`.
    A single band is selected from each tile via `band_number` (1-based),
    and the mosaic is written as float32. Any per-tile numeric nodata values
    are replaced with NaN in the output array.

    Parameters
    ----------
    tile_paths : sequence of str or os.PathLike
        Paths to input raster tiles.
    output_path : str or os.PathLike
        Destination path for the output mosaic GeoTIFF.
    band_number : int
        1-based band index to read from each tile.
    resample : int or float or tuple of int or float, len 2
        Target output resolution passed to `merge(..., res=...)`. If provided,
        tiles are resampled using bilinear interpolation.

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




def images_overlay(image_paths, output_path, mode='binary', class_values=None):
    """
    Combine multiple categorical rasters into a single output raster. 
    
    This function is intended for categorical rasters stored as float32,
    where features are represented by nonzero values, background is 0, and
    nodata is NaN.

    In ``mode="binary"``, the output is a union mask, where 1 represents areas 
    where any input raster has a nonzero value, and NaN is essentially the background.

    In non-binary mode, each input is assigned a unique value, and are then combined
    into a single raster with multiple categories. Pixels where one raster has a 
    nonzero value are assigned that raster's class code. If multiple rasters overlap a
    pixel, later rasters in ``image_paths`` overwrite earlier ones (i.e.,
    list order defines priority).

    Parameters
    ----------
    image_paths : sequence of str or os.PathLike
        Paths to input single-band categorical rasters on the same grid.
        Rasters are assumed to be float32 with nonzero = class presence. 
        In non-binary mode, order defines class priority (later = higher).
    output_path : str or os.PathLike
        Destination path for the output GeoTIFF.
    mode : str, default="binary"
        Input "binary" produces a union mask; any other value enables
        priority-based categorical overlay.
    class_values : sequence of int or float or None, default=None
        Class codes corresponding to each input raster in non-binary mode.
        If None, defaults to 1..N for N input rasters.

    Returns
    -------
    None
    """



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
        