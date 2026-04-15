
#############################################
# EarthScape Metadata
#############################################

VERSION = '1.1'
STR_VERSION = f"esv{VERSION.replace('.','p')}"
ES_REPO_URL = r'https://github.com/masseygeo/earthscape'
ES_DATA_URL = r'https://doi.org/10.13023/kgs.data.05.01.2025'
ES_PAPER_URL = r'https://doi.org/10.48550/arXiv.2503.15625'


#############################################
# Modalities & Channels
#############################################
MODALITIES = {
    "mask": ['mask.tif'],
    "dem": ['dem.tif'],
    "aerial": ['aerialr.tif', 'aerialg.tif', 'aerialb.tif', 'aerialnir.tif'],
    "nhd": ['nhd.tif'],
    "osm": ['osm.tif'],
    "ep": ['ep_5x5.tif', 'ep_11x11.tif', 'ep_21x21.tif', 'ep_51x51.tif', 'ep_101x101.tif', 'ep_201x201.tif'],
    "plc": ['plc.tif', 'plc_10.tif', 'plc_20.tif', 'plc_50.tif', 'plc_100.tif', 'plc_200.tif'],
    "prc": ['prc.tif', 'prc_10.tif', 'prc_20.tif', 'prc_50.tif', 'prc_100.tif', 'prc_200.tif'],
    "s": ['s.tif', 's_10.tif', 's_20.tif', 's_50.tif', 's_100.tif', 's_200.tif'],
    "sds": ['sds_5x5.tif', 'sds_11x11.tif', 'sds_21x21.tif', 'sds_51x51.tif', 'sds_101x101.tif', 'sds_201x201.tif'],
    }


#############################################
# Labels (Map Units) & Colors (for visualizations)
#############################################

SG_MAPPING = {
    'af1': 1, 
    'Qal': 2, 
    'Qaf': 3, 
    'Qat': 4, 
    'Qc': 5, 
    'Qca': 6, 
    'Qr': 7
    }

SG_COLORS = {
    'af1': '#636566', 
    'Qal': '#fdf5a4', 
    'Qaf': '#ffa1db', 
    'Qat': '#f9e465', 
    'Qc': '#d6c9a7', 
    'Qca': '#c49d83', 
    'Qr': '#b0acd6'
    }


#############################################
# Default Paths
#############################################

##### source code directory
SRC_DIR = r'../earthscape'


##### dataset directory
DATASET_DIR = r'../data'
SMOKE_DIR = f"{DATASET_DIR}/smoke"


##### assets directory (figures, csv files, etc)
ASSETS_DIR = r'../assets'


##### training/validation/testing split directories
SPLIT_DIR = r'../splits'
ES_SPLIT_DIR = f"{SPLIT_DIR}/{STR_VERSION}_splits"
# SMOKE_VERSION_SPLIT_DIR = f"{SPLIT_DIR}/es{STR_VERSION}_smoke"


##### model output directories
MODEL_DIR = r'../experiments'


#############################################
# Data Sources
#############################################
SRC_URLS = {

    # kentucky DEM tile index - ESRI API
    'kyfromabove_dem_index': r'https://kygisserver.ky.gov/arcgis/rest/services/WGS84WM_Services/KY_Data_Tiles_DEM_WGS84WM/MapServer/0',

    # kentucky aerial RGB+NIR tile index - ESRI API
    'kyfromabove_aerial_index': r'https://services3.arcgis.com/ghsX9CKghMvyYjBU/arcgis/rest/services/Ky_KYAPED_Aerial_Tile_Index_WM_gdb/FeatureServer/0',

    # National Hydrography Dataset Plus HR - geodatabase download
    'nhd': r'https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/VPU/Current/GDB/NHDPLUS_H_0511_HU4_GDB.zip',

    # Open Street Maps - shapefile download
    'osm': r'http://download.geofabrik.de/north-america/us/kentucky-latest-free.shp.zip',

    # Kentucky Surficial Geologic Maps - geodatabase downloads...
    # Hardin County subset
    'howevalley': r'https://ngmdb.usgs.gov/ngm-bin/gems_download.pl?id=1631&pid=111985',
    'sonora': r'https://ngmdb.usgs.gov/ngm-bin/gems_download.pl?id=1630&pid=111983',

    # Warren County subset
    'warren': r'https://uknowledge.uky.edu/context/kgs_data/article/1014/type/native/viewcontent',
    }
