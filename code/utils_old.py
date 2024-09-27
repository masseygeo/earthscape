#######################################################################################
# OLD FUNCTIONS FOR DATASET PROCESSING
# ------------------
# Author: Matt Massey
# Last updated: 9/21/2024
# Purpose: Archive of functions originally created for dataset compilation, but no longer used.
#######################################################################################



# def kyfromabove_tile_index(output_dir):
#     """
#     Function to fetch and extract KyFromAbove data tile index geodatabase, and save DEM, Aerial, and Lidar Point Cloud layers as GeoJSON files in specified directory. Same as 
    
#     Parameters
#     ----------
#     output_dir : str
#         Directory path for output files.
    
#     Returns
#     -------
#     None
#     """
#     url = r'https://ky.app.box.com/index.php?rm=box_download_shared_file&vanity_name=kymartian-kyaped-5k-tile-grids&file_id=f_1173114014568'

#     try:
#         response = requests.get(url)
#         response.raise_for_status()
        
#         if response.status_code == 200:

#             output_zip_path = f"{output_dir}/index.gdb.zip"
            
#             with open(output_zip_path, 'wb') as zip:
#                 zip.write(response.content)

#             with zipfile.ZipFile(output_zip_path, 'r') as zip:
#                 zip.extractall(output_dir)

#             gdb_path = glob.glob(f"{output_dir}/*.gdb")[0]
#             gdb_layers = fiona.listlayers(gdb_path)
            
#             for layer in gdb_layers:
#                 gdf = gpd.read_file(gdb_path, layer=layer)
#                 output_path = f"{output_dir}/{layer}.geojson"
#                 gdf.to_file(output_path, driver='GeoJSON')
            
#             os.remove(output_zip_path)
#             shutil.rmtree(gdb_path)
    
#     except:
#         print(f"Did not connect with download URL...\n{url}")






# def clip_image_to_boundary(input_path, boundary_path):
#     """
#     Function to to clip an image to an area of interest polygon and save the clipped image as a new GeoTIFF.

#     Parameters
#     ----------
#     input_path : str
#         Path to image to be clipped.
#     boundary_path : str
#         Path to boundary polygon GeoJSON.

#     Returns
#     -------
#     None
#     """
#     boundary = gpd.read_file(boundary_path)

#     with rasterio.Env(CHECK_DISK_FREE_SPACE='FALSE'):
#         with rasterio.open(input_path) as src:
            
#             if src.nodata is None:
#                 nodata_value = np.nan
#                 data_type = 'float32'
#             else:
#                 nodata_value = src.nodata
#                 data_type = src.meta['dtype']

#             out_image, out_transform = mask(src, shapes=boundary.geometry, crop=True, nodata=nodata_value)
#             out_meta = src.meta.copy()
#             out_meta.update({'driver':'GTiff', 
#                             'height':out_image.shape[1], 
#                             'width':out_image.shape[2], 
#                             'transform':out_transform, 
#                             'crs': src.crs, 
#                             'nodata': nodata_value, 
#                             'dtype': data_type})
            
#             new_file_dir = os.path.dirname(input_path)
#             new_filename = os.path.splitext(os.path.basename(input_path))[0] + '_clip.tif'
#             output_tif_path = f"{new_file_dir}/{new_filename}"

#             with rasterio.open(output_tif_path, 'w', **out_meta) as output:
#                 for i in range(out_image.shape[0]):
#                     output.write(out_image[i, :, :], i+1)




# def convert_image_dtype(input_path):
#     """
#     Function to convert image to float32 dtype.
    
#     Parameters
#     ----------
#     input_tif_path : string
#         Path to GeoTIFF image to be converted.
    
#     Returns
#     -------
#     None
#     """
#     if '_f32.tif' in input_path:
#         return
    
#     output_path = input_path[:-4] + '_f32.tif'
    
#     with rasterio.Env(CHECK_DISK_FREE_SPACE='FALSE'):
#         with rasterio.open(input_path) as src:
#             nodata_value = src.nodata
#             data = src.read()
#             out_data = data.astype(rasterio.float32)
            
#             if nodata_value is None:
#                 nodata_value = np.nan
#                 out_data[data == src.meta['nodata']] = nodata_value

#             out_meta = src.meta.copy()
#             out_meta.update({'dtype': rasterio.float32, 
#                              'nodata': nodata_value})
        
#         with rasterio.open(output_path, 'w', **out_meta) as output:
#             for i in range(out_data.shape[0]):
#                 output.write(out_data[i, :, :], i+1)



# def image_to_reference_image(input_path, reference_path, output_path = None):

#     with rasterio.open(reference_path) as ref:
#         ref_transform = ref.transform
#         ref_crs = ref.crs
#         ref_width = ref.width
#         ref_height = ref.height
#         ref_dtype = ref.meta['dtype']
#         dst_meta = ref.meta.copy()
        
#         with rasterio.open(input_path) as src:
#             src_transform = src.transform
#             src_crs = src.crs
#             src_count = src.count
#             src_nodata = src.nodata
#             src_dtype = src.meta['dtype']
        

#             dst_meta.update({'driver': 'GTiff', 
#                             'height': ref_height, 
#                             'width': ref_width, 
#                             'count': src_count, 
#                             'crs': ref_crs, 
#                             'transform': ref_transform,
#                             'nodata': src_nodata})
            
#             dst_data = np.empty((src_count, ref_height, ref_width), dtype=src_dtype)

#             if not output_path:
#                 output_path = input_path

#             with rasterio.open(output_path, 'w', **dst_meta) as dst:
#                 for i in range(1, src_count+1):
#                     reproject(source = rasterio.band(src, i), 
#                                 destination = dst_data[i-1, :, :], 
#                                 src_transform = src_transform, 
#                                 src_crs = src_crs, 
#                                 dst_transform = ref_transform, 
#                                 dst_crs = ref_crs, 
#                                 resampling = Resampling.bilinear)

#                     dst.write(dst_data[i-1, :, :], i)




# def elevation_percentile(window):
#     """
#     Function to...
    
#     Parameters
#     ----------
#     window : array
#         Flattened 1D array of values in the window returned by generic_filter function.
    
#     Returns
#     -------
#     window_percentile : float
#         Percentile value of center pixel within the window.
#     """

#     # get the center pixel index & value
#     # NOTE: assumes square window with odd dimensions
#     center_idx = len(window) // 2
#     center_value = window[center_idx]

#     # skips windows with NaN values (windows situated on edge or completely off valid DEM)
#     if np.isnan(window).any():
#         return np.nan
    
#     # rank center pixel in context of all values in window; uusing 'average' method to handle ties
#     ranks = rankdata(window, method='average')
#     center_rank = ranks[center_idx]
#     n = len(window)

#     # calculate percentile
#     percentile = ((center_rank - 1) / (n - 1)) * 100
#     return percentile



# def image_window_calculation(input_path, output_path, func, window_size):
#     """
#     Function to apply another function to a GeoTIFF using a sliding window.
    
#     Parameters
#     ----------
#     input_path : str
#         Path to input image.
#     output_path : str
#         Path for output image.
#     func : function or class
#         Function that applies some calculation to input 1D array (window).
#     window_size : int
#         Width and height, in pixels, for sliding window.

#     Returns
#     -------
#     None
#     """

#     with rasterio.open(input_path) as src:
        
#         # read dem as masked array; convert masked array to regular array with NaN values
#         data = src.read(1, masked=True).filled(np.nan)
    
#         # apply custom function using a sliding window
#         result = generic_filter(input=data, function=func, size=window_size, mode='constant', cval=np.nan)
        
#         # update metadata for the output file
#         meta = src.meta.copy()
#         meta.update(dtype=rasterio.float32)
        
#         # Write the result to a new GeoTIFF
#         with rasterio.open(output_path, 'w', **meta) as dst:
#             dst.write(result.astype(rasterio.float32), 1)




# def get_intersecting_index_tiles(input_path, boundary_path, output_path):
#     """
#     Function to extract tile index polygons from an input GeoJSON intersecting an area specified by another GeoJSON, and then saving the subset tile index polygons as a new GeoJSON.
    
#     Parameters
#     ----------
#     geojson_path : str
#         Path to input tile index GeoJSON.
#     boundary_path : str
#         Path to area of interest GeoJSON.
#     output_geojson_path : str
#         Path for output GeoJSON.
    
#     Returns
#     -------
#     None
#     """
#     gdf_geojson = gpd.read_file(input_path)
#     gdf_boundary = gpd.read_file(boundary_path)
#     if gdf_boundary.crs != gdf_geojson.crs:
#         gdf_geojson = gdf_geojson.to_crs(gdf_boundary.crs)
#     gdf_intersect = gpd.sjoin(left_df=gdf_geojson, right_df=gdf_boundary, how='inner')
#     gdf_intersect.to_file(output_path, driver='GeoJSON')






# def get_contained_and_edge_tile_paths(index_path, boundary_path, data_dir):
#     """
#     Function to get lists of aerial imagery tile paths that are completely contained or intersecting the edge of the boundary area.

#     Parameters
#     ----------
#     index_path : str
#         Path to GeoJSON of aerial imagery tile index polygons for the area of interest.
#     boundary_path : str
#         Path to GeoJSON of the area of interest polygon.
#     data_dir : str
#         Directory path containing the aerial imagery tile data. Directory must have only GeoTIFF files.

#     Returns
#     -------
#     within_poly_paths : list
#         List of paths of tiles completely contained within the area of interest.
#     edge_poly_paths : list
#         List of paths of tiles intersecting the boundary of the area of interest.
#     """
#     gdf_index = gpd.read_file(index_path)
#     gdf_boundary = gpd.read_file(boundary_path)

#     if gdf_boundary.crs != gdf_index.crs:
#         gdf_index = gdf_index.to_crs(gdf_boundary.crs)
    
#     boundary = gdf_boundary.iloc[0].geometry
#     within_polygons = gdf_index[gdf_index.geometry.within(boundary)]
#     edge_polygons = gdf_index[~gdf_index.index.isin(within_polygons.index)]

#     within_poly_paths = []
#     for _, row in within_polygons.iterrows():
#         tile = row['TileName']
#         path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
#         within_poly_paths.append(path)

#     edge_poly_paths = []
#     for _, row in edge_polygons.iterrows():
#         tile = row['TileName']
#         path = glob.glob(f"{data_dir}/*{tile}*.tif")[0]
#         edge_poly_paths.append(path)

#     return within_poly_paths, edge_poly_paths