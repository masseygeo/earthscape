import geopandas as gpd
import glob



# identify main tile (use polygon patch centroid)
# get surrounding eight neighbor tiles

def get_main_tile_for_patch(patch_path, index_path):
    patches = gpd.read_file(patch_path)
    index = gpd.read_file(index_path)

    tiles = []
    for _, row in patches.iterrows():

        patch_centroid = row.geometry.centroid
        tile = index[index.geometry.contains(patch_centroid)]
        tile_names = list(tile['TileName'].values)

        neighbors = index[index.geometry.touches(tile.geometry.iloc[0], align=False)]
        neighbor_names = list(neighbors['TileName'])

        tile_names = tile_names + neighbor_names
        tiles.append(tile_names)
    patches['main_tile'] = tiles

    return patches



# get paths to 9 tiles

# merge tiles

# resample tiles

# extract patch image

################################################################
patch_path = glob.glob(r'../data/warren/patches*.geojson')[0]
index_path = glob.glob(r'../data/warren/*Aerial*.geojson')[0]

patches = get_main_tile_for_patch(patch_path, index_path)
patches.head()

# gpd.read_file(index_path).loc[0,'TileName'].values()