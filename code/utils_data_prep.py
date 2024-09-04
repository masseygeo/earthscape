
import requests
import os
import zipfile
import shutil
import json
import glob
import numpy as np
from math import ceil
import geopandas as gpd
import fiona
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.windows import from_bounds
from shapely.geometry import box
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.plot import show
from scipy.ndimage import gaussian_filter




#######################################################################################
# DATA DOWNLOAD FUNCTIONS
#######################################################################################

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


def kyfromabove_tile_index(output_dir):
    """
    Function to fetch and extract KyFromAbove data tile index geodatabase, and save DEM, Aerial, and Lidar Point Cloud layers as GeoJSON files in specified directory. Same as 
    
    Parameters
    ----------
    output_dir : str
        Directory path for output files.
    
    Returns
    -------
    None
    """
    url = r'https://ky.app.box.com/index.php?rm=box_download_shared_file&vanity_name=kymartian-kyaped-5k-tile-grids&file_id=f_1173114014568'

    try:
        response = requests.get(url)
        response.raise_for_status()
        
        if response.status_code == 200:

            output_zip_path = f"{output_dir}/index.gdb.zip"
            
            with open(output_zip_path, 'wb') as zip:
                zip.write(response.content)

            with zipfile.ZipFile(output_zip_path, 'r') as zip:
                zip.extractall(output_dir)

            gdb_path = glob.glob(f"{output_dir}/*.gdb")[0]
            gdb_layers = fiona.listlayers(gdb_path)
            
            for layer in gdb_layers:
                gdf = gpd.read_file(gdb_path, layer=layer)
                output_path = f"{output_dir}/{layer}.geojson"
                gdf.to_file(output_path, driver='GeoJSON')
            
            os.remove(output_zip_path)
            shutil.rmtree(gdb_path)
    
    except:
        print(f"Did not connect with download URL...\n{url}")


def download_data_tiles(index_path, id_field, url_field, output_dir):
    """
    Function to read KyFromAbove Tile Index GeoJSON, download the GeoTIFF using the URL from a specified field, and then save the GeoTIFF to the specified output directory. Similar to download_zip function, but modified for KyFromAbove tile index dataset.

    Parameters
    ----------
    index_path : string
        Path to GeoJSON.
    id_field : string
        Attribute name of GeoJSON containing unique ID for file naming.
    url_field : string
        Attribute name of GeoJSON containing the download URL.
    output_dir : string
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
            # zip_path = os.path.join(output_dir, f"{tile_id}.zip")
            download_zip(url, output_dir)

        else:
            print('Download is not .tif or .zip...')




#######################################################################################
# GEOSPATIAL VECTOR FUNCTIONS
#######################################################################################

def get_intersecting_index_tiles(input_path, boundary_path, output_path):
    """
    Function to extract tile index polygons from an input GeoJSON intersecting an area specified by another GeoJSON, and then saving the subset tile index polygons as a new GeoJSON.
    
    Parameters
    ----------
    geojson_path : str
        Path to input tile index GeoJSON.
    boundary_path : str
        Path to area of interest GeoJSON.
    output_geojson_path : str
        Path for output GeoJSON.
    
    Returns
    -------
    None
    """
    gdf_geojson = gpd.read_file(input_path)
    gdf_boundary = gpd.read_file(boundary_path)
    if gdf_boundary.crs != gdf_geojson.crs:
        gdf_geojson = gdf_geojson.to_crs(gdf_boundary.crs)
    gdf_intersect = gpd.sjoin(left_df=gdf_geojson, right_df=gdf_boundary, how='inner')
    gdf_intersect.to_file(output_path, driver='GeoJSON')


def get_contained_and_edge_tile_paths(index_path, boundary_path, data_dir):
    """
    Function to get lists of aerial imagery tile paths that are completely contained or intersecting the edge of the boundary area.

    Parameters
    ----------
    index_path : str
        Path to GeoJSON of aerial imagery tile index polygons for the area of interest.
    boundary_path : str
        Path to GeoJSON of the area of interest polygon.
    data_dir : str
        Directory path containing the aerial imagery tile data. Directory must have only GeoTIFF files.

    Returns
    -------
    within_poly_paths : list
        List of paths of tiles completely contained within the area of interest.
    edge_poly_paths : list
        List of paths of tiles intersecting the boundary of the area of interest.
    """
    gdf_index = gpd.read_file(index_path)
    gdf_boundary = gpd.read_file(boundary_path)

    if gdf_boundary.crs != gdf_index.crs:
        gdf_index = gdf_index.to_crs(gdf_boundary.crs)
    
    boundary = gdf_boundary.iloc[0].geometry
    within_polygons = gdf_index[gdf_index.geometry.within(boundary)]
    edge_polygons = gdf_index[~gdf_index.index.isin(within_polygons.index)]

    within_poly_paths = []
    for _, row in within_polygons.iterrows():
        tile = row['TileName']
        path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
        within_poly_paths.append(path)

    edge_poly_paths = []
    for _, row in edge_polygons.iterrows():
        tile = row['TileName']
        path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
        edge_poly_paths.append(path)

    return within_poly_paths, edge_poly_paths


def clip_gis_to_boundary(input_path, boundary_path, output_path, gdb_layer=None):
    """
    Function to clip GIS spatial data to the extent of an area of interest polygon and save the clipped feature(s) as a new GeoJSON file.

    Parameters
    ----------
    input_path : str
        Path to GIS spatial input file. If this is a geodatabase (.gdb), then the gdb_layer argument must be specified.
    boundary_path : str
        Path to area of interest polygon.
    output_path : str
        Path for output GeoJSON.
    gdb_layer : str (optional)
        Name of geodatabase layer to be clipped. Default is None.

    Returns
    -------
    None
    """
    if not gdb_layer:
        gdf_input = gpd.read_file(input_path)
    else:
        gdf_input = gpd.read_file(input_path, layer=gdb_layer)
    gdf_input = gdf_input.explode(ignore_index=True, index_parts=False)
    gdf_boundary = gpd.read_file(boundary_path)

    if gdf_input.crs != gdf_boundary.crs:
        gdf_input = gdf_input.to_crs(gdf_boundary.crs)

    gdf_output = gpd.clip(gdf_input, mask=gdf_boundary)
    gdf_output.to_file(output_path, driver='GeoJSON')


def create_image_patches(reference_path, patch_size, patch_overlap, boundary_path, output_path):
    """
    Function to create geospatial polygons that represent square image patch locations saved as a GeoJSON. The size of the image patches (assumed to be square) and the proportion of overlap between adjacent patches is specified. Each patch will have a unique id created from the patch_size, patch_overlap, and a unique number.

    Parameters
    ----------
    reference_path : str
        Path to a reference GeoTIFF image that represents the area where patches will be created.
    patch_size : int or float
        Size of the square patch in pixels.
    patch_overlap : float
        Proportion of overlap between adjacent patches.
    boundary_path : str
        Path to area of interest boundary GeoJSON file (should be aligned with boundaries of reference_path image) to ensure patch polygons intersect.
    output_path : str
        Path for output patch polygon GeoJSON file.

    Returns
    -------
    None.
    """

    boundary = gpd.read_file(boundary_path)

    with rasterio.open(reference_path) as src:
        bounds = src.bounds
        res = src.res[0]
        crs = src.crs
        
    patch_size_units = patch_size * res
    overlap_start_units = patch_size_units * (1 - patch_overlap)

    patches = []
    x = bounds.left
    while x < bounds.right:
        y = bounds.bottom
        while y < bounds.top:
            patch = box(x, y, x+patch_size_units, y+patch_size_units)
            if patch.intersects(boundary.geometry).any():
                patches.append(patch)
            y += overlap_start_units
        x += overlap_start_units
    
    gdf = gpd.GeoDataFrame(geometry=patches, crs=crs)
    gdf['patch_id'] = [f"{patch_size}_{int(patch_overlap*100)}_{i}" for i in range(1, len(gdf)+1)]
    gdf.to_file(output_path, driver='GeoJSON')




#######################################################################################
# GEOSPATIAL IMAGE FUNCTIONS
#######################################################################################

def gis_to_image(input_path, output_path, output_resolution, attribute=None):
    """
    Function to convert vector GIS (GeoJSON or Shapefile) to GeoTIFF image with a given resolution. The image will be binary (default) unless the attribute argument is given. Output GeoTIFF file is of float32 dtype with NaN representing no data values.

    Parameters
    ----------
    input_path : str
        Path to input GeoJSON or Shapefile.
    output_path : str
        Path for output GeoTIFF.
    output_resolution : int
        Resolution of GeoTIFF in native spatial units of input GIS file.
    attribute : str (optional)
        Name of attribute in GIS file for assigning pixel values. If not provided, image will be binary.

    Returns
    -------
    None
    """
    gdf = gpd.read_file(input_path)

    if gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
        gdf['geometry'] = gdf['geometry'].buffer(0)
    
    minx, miny, maxx, maxy = gdf.total_bounds
    width = ceil((maxx - minx) / output_resolution)
    height = ceil((maxy - miny) / output_resolution)

    transform = from_origin(west=minx, north=maxy, xsize=output_resolution, ysize=output_resolution)

    values = gdf[attribute].unique()
    mapper = {key:value for value, key in enumerate(values)}
    gdf[f"{attribute}_int"] = gdf[attribute].apply(lambda x: mapper.get(x, np.nan))
    shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[f"{attribute}_int"])]
    
    output_image = rasterize(shapes = shapes, 
                             out_shape = (height, width), 
                             transform = transform, 
                             all_touched = True, 
                             fill = np.nan, 
                             dtype = rasterio.float32)
    
    output_meta = {'driver': 'GTiff', 
                   'height': height, 
                   'width': width, 
                   'transform': transform, 
                   'count': 1, 
                   'dtype': output_image.dtype, 
                   'nodata': np.nan, 
                   'crs': gdf.crs.to_string()}
    
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        dst.write(output_image, 1)
    
    output_json_path = output_path.replace('.tif', '.json')
    with open(output_json_path, 'w') as file:
        json.dump(mapper, file, indent=4)



def multiple_gis_to_reference_image(input_paths, reference_path, output_path):
    """
    Function to combine multiple geospatial vector GIS features into a new GeoTIFF image aligned with a reference image. In the case of overlapping features, priority for pixel values in the final image will be given to the last feature. Background space will be given a value of 0 and additional features will be given sequential integers in increments of 1.

    Parameters
    ----------
    input_paths : list or tuple
        List of path to vector GIS features in GeoJSON(s) and/or Shapefile(s).
    reference_path : str
        Path to reference GeoTIFF image.
    output_path : str
        Path to output GeoTIFF image.
    
    Returns
    -------
    None
    """
    with rasterio.open(reference_path) as src:

        shapes_all = []
        features = ['background']

        for val, path in enumerate(input_paths, start=1):
            
            feature = os.path.basename(path)
            feature = os.path.splitext(feature)[0]
            features.append(feature)

            gdf = gpd.read_file(path)

            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)
            
            shapes = [(geom, val) for geom in gdf.geometry]
            shapes_all.extend(shapes)

        output_image = rasterize(shapes=shapes_all, 
                                 out_shape=(src.height, src.width), 
                                 transform=src.transform, 
                                 fill=0, 
                                 all_touched=True, 
                                 dtype=rasterio.float32)
        
        mask = src.dataset_mask()
        output_image = np.where(mask, output_image, src.nodata)

        output_meta = src.meta.copy()
        output_meta.update({'driver': 'GTiff', 
                            'count': 1, 
                            'dtype':rasterio.float32})
        
        with rasterio.open(output_path, 'w', **output_meta) as dst:
            dst.write(output_image.astype(rasterio.float32), 1)
        
        mapper = {k:v for v,k in enumerate(features)}
        output_json_path = output_path.replace('.tif', '.json')
        with open(output_json_path, 'w') as meta:
            json.dump(mapper, meta, indent=4)


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




#######################################################################################
# PLOTTING 
#######################################################################################


def plot_multi_terrain_features(mdhs_path, terrain_paths, bounds, cmap, title):
    """
    Function to plot terrain features at six resolutions within a defined area. Terrain features have 50% transparency overlaying a multi-directional hillshade image.

    Parameters
    ----------
    mdhs_path : str
        Path to multi-directional hillshade GeoTIFF.
    terrain_paths : iterable
        List or tuple of paths terrain features at multiple resolutions
    bounds : iterable
        List or tuple of bounding coordinates (left, bottom, right, top) of plot area.
    cmap : str or variable
        Name of Matplotlib colormap, or variable name of custom colormap.
    title : str
        Title of terrain feature plot.

    Returns
    -------
    None.
    """

    # set up plot assuming six scales/terrain features
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,8), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ax = ax.ravel()

    with rasterio.open(mdhs_path) as mdhs:

        # iterate through each terrain feature (six total)
        for idx, path in enumerate(terrain_paths):
            with rasterio.open(path) as src:

                # set up window for feature, get transform, and data
                window = from_bounds(*bounds, src.transform)
                transform = src.window_transform(window)
                data = src.read(1, window=window)
                min_val = np.min(data)
                max_val = np.max(data)

                # plot feature; this will be hidden and is only for colorbar
                hidden = ax[idx].imshow(data, cmap=cmap)

                # plot multi-directional hillshade as base layer (on top of hidden)
                mdhs_window = from_bounds(*bounds, mdhs.transform)
                mdhs_data = mdhs.read(1, window=mdhs_window)
                mdhs_transform = mdhs.window_transform(mdhs_window)
                show(mdhs_data, ax=ax[idx], cmap='binary_r', transform=mdhs_transform)

                # plot terrain feature with transparency (to overlay on hillshade)
                show(data, ax=ax[idx], cmap=cmap, transform=transform, alpha=0.5)

                # plot custom color bar
                cax = inset_axes(ax[idx], width='5%', height='40%', loc='lower right')
                fig.colorbar(hidden, cax=cax, ticks=[min_val, max_val])
                cax.yaxis.set_ticks_position('left')

                # customize plot elements
                ax[idx].tick_params(axis='both', which='major', labelsize=8)
                ax[idx].tick_params(axis='x', labelrotation=60)
                ax[idx].ticklabel_format(style='plain')
                ax[idx].set_title(os.path.basename(path), style='italic', fontsize=10)

    plt.suptitle(f"{title}\n5, 10, 20, 50, 100, & 250 feet per pixel resolutions", y=0.96)
    plt.show()


