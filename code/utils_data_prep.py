
# import matplotlib.pyplot as plt
from shapely.geometry import box
from math import ceil 
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
import numpy as np



import requests
import zipfile
import glob
import fiona
import geopandas as gpd
import os
import shutil
import rasterio
from rasterio.features import rasterize


def get_tile_index(output_zip_path):
    """
    Function to fetch KyFromAbove data tile index geodatabase and save layers (DEM, Aerial, and Lidar Point Cloud) as individual geojson files.
    
    Parameters
    ----------
    output_zip_path : string
        Path for output zip file.
    
    Returns
    -------
    None
    """
    url = r'https://ky.app.box.com/index.php?rm=box_download_shared_file&vanity_name=kymartian-kyaped-5k-tile-grids&file_id=f_1173114014568'

    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            # download zip file
            with open(output_zip_path, 'wb') as zip:
                zip.write(response.content)
            # extract zip file contents
            with zipfile.ZipFile(output_zip_path, 'r') as zip:
                extract_dir = os.path.dirname(output_zip_path)
                zip.extractall(extract_dir)
            # extract geodatabase layers as geojsons
            gdb_path = glob.glob(f"{extract_dir}/*Tile*.gdb")[0]
            # gdb_path = output_zip_path.replace('.zip', '')
            gdb_layers = fiona.listlayers(gdb_path)
            for layer in gdb_layers:
                gdf = gpd.read_file(gdb_path, layer=layer)
                output_filename = f"{layer}.geojson"
                output_dir = os.path.dirname(gdb_path)
                output_path = os.path.join(output_dir, output_filename)
                gdf.to_file(output_path, driver='GeoJSON')
            # delete zip file
            os.remove(output_zip_path)
            # delete geodatabase (directory)
            shutil.rmtree(gdb_path)
    except:
        print('Something went wrong...')



def get_intersecting_index_tiles(geojson_path, boundary_path, output_geojson_path):
    """
    Function to extract polygons from an input geojson file intersecting an area (specified by another geojson or shapefile), and then saving as an output geojson file.
    
    Parameters
    ----------
    geojson_path : string
        Path to input geojson.
    boundary_path : string
        Path to area of interest geojson or shapefile.
    output_geojson_path : string
        Path for output geojson.
    
    Returns
    -------
    None
    """
    gdf_geojson = gpd.read_file(geojson_path)
    gdf_boundary = gpd.read_file(boundary_path)
    if gdf_boundary.crs != gdf_geojson.crs:
        gdf_geojson = gdf_geojson.to_crs(gdf_boundary.crs)
    gdf_intersect = gpd.sjoin(left_df=gdf_geojson, right_df=gdf_boundary, how='inner')
    gdf_intersect.to_file(output_geojson_path, driver='GeoJSON')



def download_tif(url, output_path):
    """
    Function to download TIFF or GeoTIFF file from a specified URL.

    Parameters
    ----------
    url : string
        URL for direct download of TIFF or GeoTIFF.
    output_path : string
        Path to save image file.

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
            print('Reponse code not 200 for downloading .tif...')
    except:
        print('Error downloading .tif...')



def download_zip(url, zip_path):
    """
    Function to download .zip file and extract contents from a specified URL.

    Parameters
    ----------
    url : string
        URL for direct download of TIFF or GeoTIFF.
    zip_path : string
        Path to save .zip file; contents will be extracted to this directory.

    Returns
    -------
    None
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.status_code == 200:
            with open(zip_path, 'wb') as zip:
                zip.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip:
                extract_dir = os.path.dirname(zip_path)
                zip.extractall(extract_dir)
            os.remove(zip_path)
        else:
            print('Reponse code not 200 for downloading .zip...')
    except:
        print('Error downloading .zip...')



def download_data_tiles(index_path, id_field, url_field, output_dir):
    """
    Function to read geojson or shapefile, download .zip or .tif from a given URL field, and save in a specified directory using another ID field.

    Parameters
    ----------
    index_path : string
        Path to geojson or shapefile.
    id_field : string
        Attribute of geojson or shapefile with unique ID.
    url_field : string
        Attribute of geojson or shapefile with download URL.
    output_dir : string
        Directory where file(s) will be downloaded.

    Returns
    -------
    None
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
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
            zip_path = os.path.join(output_dir, f"{tile_id}.zip")
            download_zip(url, zip_path)

        else:
            print('Download is not .tif or .zip...')



def clip_spatial_to_boundary(input_spatial, boundary, output_path, gdb_layer=None):
    """
    Function to clip GIS spatial data from a geodatabase to the extent of a polygon boundary feature and save as a new GeoJSON file.

    Parameters
    ----------
    input_spatial : string
        Path to spatial file (or geodatabase) to be clipped.
    boundary : string
        Path to GeoJSON or Shapefile of boundary mask.
    output_path : string
        Path for output GeoJSON file.
    gdb_layer : strings (optional)
        Name of feature layer in geodatabae to be clipped. Optional. Default is None.

    Returns
    -------
    None
    """
    if not gdb_layer:
        gdf_input = gpd.read_file(input_spatial, layer=gdb_layer)
    else:
        gdf_input = gpd.read_file(input_spatial)

    gdf_input = gdf_input.explode(ignore_index=True, index_parts=False)

    gdf_boundary = gpd.read_file(boundary)

    if gdf_input.crs != gdf_boundary.crs:
        gdf_input = gdf_input.to_crs(gdf_boundary.crs)

    gdf_output = gpd.clip(gdf_input, mask=gdf_boundary)
    gdf_output.to_file(output_path, driver='GeoJSON')



def convert_spatial_to_reference_image(input_spatial, reference_image, output_path, attribute=None):
    """
    Function to convert geospatial vector files (shapefile or GeoJSON) to images with spatial, resolution, and transform properties of a reference image (GeoTIFF).

    Parameter
    ---------
    input_spatial : string
        Path to geospatial file.
    reference_image : string
        Path to reference GeoTIFF image.
    output_path : string
        Path for output image.
    attribute : string (optional)
        Name of geospatial attribute to categorize image values. Default is None, which categorizes output image as binary (1 for spatial features, 0 for background),

    Returns
    -------
    None
    """

    gdf = gpd.read_file(input_spatial)

    with rasterio.open(reference_image) as src:

        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        

        if not attribute:
            shapes = [(geom, 1) for geom in gdf.geometry]
        else:
            shapes = [(geom, value) for geom, value in zip(gdf.geometry, gdf[attribute])]
            
        output_image = rasterize(shapes=shapes, 
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








def convert_image_dtype(input_tif_path, dtype):
    
    if '_f32.tif' in input_tif_path:
        return
    
    with rasterio.open(input_tif_path) as src:
        nodata_value = src.nodata
        data = src.read()
        out_data = data.astype(dtype)
        if nodata_value is None:
            nodata_value = np.nan
            out_data[data == src.meta['nodata']] = nodata_value
        out_meta = src.meta.copy()
        out_meta.update({'dtype': dtype, 'nodata': nodata_value})
    output_path = input_tif_path[:-4] + '_f32.tif'
    
    with rasterio.open(output_path, 'w', **out_meta) as output:
        for i in range(out_data.shape[0]):
            output.write(out_data[i, :, :], i+1)



def get_contained_and_edge_tile_paths(index_path, boundary_path, data_dir, file_suffix=None):
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
        if file_suffix is None:
            path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
        else:
            path = glob.glob(f"{data_dir}/*{tile}*{file_suffix}")[0]
        within_poly_paths.append(path)

    edge_poly_paths = []
    for _, row in edge_polygons.iterrows():
        tile = row['TileName']
        if file_suffix is None:
            path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
        else:
            path = glob.glob(f"{data_dir}/*{tile}*{file_suffix}")[0]
        edge_poly_paths.append(path)

    return within_poly_paths, edge_poly_paths


def clip_image_to_boundary(input_tif_path, boundary_path, output_tif_path=None):
    gdf = gpd.read_file(boundary_path)

    with rasterio.Env(CHECK_DISK_FREE_SPACE='FALSE'):
        with rasterio.open(input_tif_path) as src:
            
            data_type = src.meta['dtype']
            if src.nodata is None:
                nodata_value = np.nan
                data_type = 'float32'
            else:
                nodata_value = src.nodata

            out_image, out_transform = mask(src, shapes=gdf.geometry, crop=True, nodata=nodata_value)
            out_meta = src.meta.copy()
            out_meta.update({'driver':'GTiff', 
                            'height':out_image.shape[1], 
                            'width':out_image.shape[2], 
                            'transform':out_transform, 
                            'crs': src.crs, 
                            'nodata': nodata_value, 
                            'dtype': data_type})
            
            if output_tif_path is None:
                new_file_dir = os.path.dirname(input_tif_path)
                new_filename = os.path.splitext(os.path.basename(input_tif_path))[0] + '_clip.tif'
                output_tif_path = os.path.join(new_file_dir, new_filename)

            with rasterio.open(output_tif_path, 'w', **out_meta) as output:
                for i in range(out_image.shape[0]):
                    output.write(out_image[i, :, :], i+1)
        src.close()



def mosaic_image_tiles(tile_paths_list, output_path, band_number=None, resample=None):
    images = [rasterio.open(tile_path) for tile_path in tile_paths_list]
    if band_number:
        if resample:
            mosaic, mosaic_transform = merge(images, indexes=[band_number], res=resample, resampling=Resampling.bilinear)
        else:
            mosaic, mosaic_transform = merge(images, indexes=[band_number])
    else:
        if resample:
            mosaic, mosaic_transform = merge(images, res=resample, resampling=Resampling.bilinear)
        else:
            mosaic, mosaic_transform = merge(images)
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


















def create_patch_polygons(gdf, max_width, max_height, pixel_width=5, pixel_height=5):

    # get coordinates of bounding box of area of intrest
    minx, miny, maxx, maxy = gdf.total_bounds


    # initialize list to hold individual grid cell polygons
    grid_cells = []

    # initialize current x position with minx
    current_x = minx

    # create grid cells by column starting at lower left corner and increasing y, then increasing x to next column...
    while current_x < maxx:

        # test if grid cell width is less than maximum allowable download width...
        if (maxx - current_x) < max_width:

            # ensure current_width is divisible by 5 AND exceeds boundary edges
            current_width = ceil((maxx - current_x) / pixel_width) * pixel_width

        else:
            current_width = max_width

        # initialize current_y as miny for each new column...
        current_y = miny

        # iterate over all grid cells within column...
        while current_y < maxy:

            # test if grid cell height is less than maximum allowable download height...
            if (maxy - current_y) < max_height:
                current_height = ceil((maxy - current_y) / pixel_height) * pixel_height

            else:
                current_height = max_height

            # create box using grid cell coordinates and sizes, then append to grid_cells list
            grid_cells.append(box(current_x, current_y, current_x + current_width, current_y + current_height))

            # increment current_y to next higher grid cell in column
            current_y += current_height
        
        # increment current_x to next column after finishing all grid cells in one column
        current_x += current_width

    # create geodataframe of grid cell polygons
    gdf_grid = gpd.GeoDataFrame({'geometry':grid_cells}, crs=gdf.crs)

    # add unique id for each grid cell
    gdf_grid['id'] = range(len(gdf_grid))

    return gdf_grid




