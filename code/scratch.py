import glob
import os
from utils_datadownloads import *


#############################################
# Mosaic DEM tiles into single DEM GeoTIFF
#############################################
os.chdir(r'/Users/matthew/GitHub/cs612/code')

# path to geodatabase
gdb_path = r'../data/geology.gdb'

# layer name of boundary feature class in geodatabase used to clip dem 
boundary_layer = r'warren_geo_boundary'

# path to geojson containing tile polygons
geojson_path = r'../data/warren/warren_KYAPED_Aerial_Tile_Index.geojson'

# directory containing dem tiles
aerial_tile_dir = r'../data/warren/aerial_tiles'

# path for output dem
output_aerial_path = r'../data/warren/aerial.tif'

# get lists of paths of edge tiles & contained tiles
within_tile_paths, edge_tile_paths = get_contained_and_edge_tile_paths(gdb_path, boundary_layer, geojson_path, aerial_tile_dir)
print('got paths...')
# iterate through edge tiles and clip to dataset boundary
for tile_path in edge_tile_paths:
    clip_image_to_boundary(tile_path, gdb_path, boundary_layer, output_tif_path=None)
print('clipped tiles...')
# get list of clipped edge tile paths & combine with contained tile paths
clipped_edge_tile_paths = glob.glob(f"{aerial_tile_dir}/*clip.tif")
tile_paths_list = within_tile_paths + clipped_edge_tile_paths

# mosaic clipped edge and contained tiles into single dem and save
mosaic_image_tiles(tile_paths_list, output_aerial_path)

# clean up clipped tiles (keep original full tiles)
for path in clipped_edge_tile_paths:
    os.remove(path)
