from shapely.geometry import box
from math import ceil 


# import matplotlib.pyplot as plt


import fiona
import geopandas as gpd
import os
import requests
import glob
import rasterio
from rasterio.merge import merge
import zipfile





def get_tile_index_geojson(gdb_path, output_dir):
    gdb_layers = fiona.listlayers(gdb_path)
    for layer in gdb_layers:
        gdf = gpd.read_file(gdb_path, layer=layer)
        output_filename = f"{layer}.geojson"
        output_path = os.path.join(output_dir, output_filename)
        gdf.to_file(output_path, driver='GeoJSON')



def get_intersecting_index_tiles(geojson_path, shapefile_path, output_geojson_path):
    gdf_shapefile = gpd.read_file(shapefile_path)
    gdf_shapefile['geometry'] = gdf_shapefile['geometry'].buffer(0)
    gdf_shapefile_union = gpd.GeoDataFrame(geometry=[gdf_shapefile.unary_union], crs=gdf_shapefile.crs)
    gdf_geojson = gpd.read_file(geojson_path)
    if gdf_shapefile_union.crs != gdf_geojson.crs:
        gdf_geojson = gdf_geojson.to_crs(gdf_shapefile_union.crs)
    gdf_intersect = gpd.sjoin(left_df=gdf_geojson, right_df=gdf_shapefile_union, how='inner')
    gdf_intersect.to_file(output_geojson_path, driver='GeoJSON')



# def _check_download_type(url):
#     try:
#         response = requests.head(url, allow_redirects=True)
#         response.raise_for_status()
#         content_type = response.headers.get('Content-Type')
#         if 'zip' in content_type:
#             return 'zip'
#         elif 'tif' in content_type:
#             return 'tif'
#         else:
#             return 'other'
#     except:
#         return 'Error checking download type...'



def _download_tif(url, output_path):
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



def _download_zip(url, zip_path):
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
    except:
        print('Error downloading .zip...')



def download_data_tiles(index_path, id_field, url_field, output_dir):
    gdf = gpd.read_file(index_path)
    for _, tile in gdf.iterrows():
        tile_id = tile[id_field]
        url = tile[url_field]
        # content_type = _check_download_type(url)
        content_type = url[-3:]
        if content_type == 'tif':
            output_path = os.path.join(output_dir, f"{tile_id}.tif")
            _download_tif(url, output_path)
        elif content_type == 'zip':
            zip_path = os.path.join(output_dir, f"{tile_id}.zip")
            _download_zip(url, zip_path)
        else:
            print('Download is not .tif or .zip...')



def mosaic_image_tiles(tile_dir, output_path, file_type='.tif'):
    query = f"{tile_dir}/**/*{file_type}"
    tile_paths_list = glob.glob(query, recursive=True)
    image_tiles = []
    for tile_path in tile_paths_list:
        src = rasterio.open(tile_path)
        image_tiles.append(src)
    mosaic, mosaic_transform = merge(image_tiles)
    mosaic_meta = src.meta.copy()
    mosaic_meta.update({'driver':'GTiff', 
                        'height':mosaic.shape[1], 
                        'width':mosaic.shape[2], 
                        'transform':mosaic_transform, 
                        'crs': src.crs})
    with rasterio.open(output_path, 'w', **mosaic_meta) as output:
        output.write(mosaic)
    for src in image_tiles:
        src.close()
    



from rasterio.mask import mask
from shapely.geometry import Polygon

def clip_image_to_boundary(input_path, boundary_path, output_path):

    gdf = gpd.read_file(boundary_path)
    gdf['geometry'] = gdf['geometry'].buffer(0)
    gdf_boundary = gpd.GeoDataFrame(geometry=[gdf.unary_union], crs=gdf.crs)
    
    # # Process each polygon to keep only the exterior as a new polygon
    # exteriors = [Polygon(geom.exterior.coords) for geom in gdf.geometry if geom.is_valid]
    
    # # Create a single unified polygon from all exterior-only polygons
    # unified_geometry = gpd.GeoSeries(exteriors).unary_union

    # # Create a new GeoDataFrame using the unified polygon (without any holes)
    # gdf_boundary = gpd.GeoDataFrame(geometry=[unified_geometry], crs=gdf.crs)


    with rasterio.open(input_path) as src:
        out_image, out_transform = mask(src, shapes=gdf_boundary['geometry'], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({'driver':'GTiff', 
                        'height':out_image.shape[1], 
                        'width':out_image.shape[2], 
                        'transform':out_transform, 
                        'crs': src.crs})
        
        with rasterio.open(output_path, 'w', **out_meta) as output:
            output.write(out_image)







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














def download_imageservice(gdf_grid, output_dir, pixel_width=5, pixel_height=5, type='dem'):

    for idx, patch in gdf_grid.iterrows():

        minx, miny, maxx, maxy = patch.geometry.bounds

        res_x = (maxx - minx) / pixel_width
        res_y = (maxy - miny) / pixel_height

        # Define the parameters for the download
        bbox = f"{minx}, {miny}, {maxx}, {maxy}"
        bboxSR = '3089'
        size = f"{res_x},{res_y}"
        # format_type = 'tif'

        # Construct the URL
        # url = f'https://kyraster.ky.gov/arcgis/rest/services/ElevationServices/Ky_DEM_KYAPED_5FT/ImageServer/exportImage?bbox={bbox}&bboxSR={bboxSR}&size={size}&format={format_type}&f=image'

        url = f"https://kyraster.ky.gov/arcgis/rest/services/ElevationServices/Ky_DEM_KYAPED_5FT/ImageServer/exportImage?bbox=&bboxSR={bboxSR}&bandIds=1&size={size}&format=image/tiff&f=image"

    # Send the request
    response = requests.get(url)

    filename = f"{type}_patchid_{patch['id']}.tif"

    output_path = os.path.join(output_dir, filename)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the image to disk
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print("Image downloaded successfully.")

    else:
        print("Failed to download the image. Status code:", response.status_code)
    
    return output_path

