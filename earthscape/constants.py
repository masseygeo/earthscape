
##############################
# EarthScape Metadata
##############################

VERSION = '1.1'
STR_VERSION = f"v{VERSION.replace('.','-')}"
ES_REPO_URL = r'https://github.com/masseygeo/earthscape'
ES_DATA_URL = r'https://doi.org/10.13023/kgs.data.05.01.2025'
ES_PAPER_URL = r'https://doi.org/10.48550/arXiv.2503.15625'


##############################
# EarthScape 1.x Modalities & Channels
##############################
MODALITIES = {
    "mask": ['geology.tif'],
    "dem": ['dem.tif'],
    "aerial": ['aerialr.tif', 'aerialg.tif', 'aerialb.tif', 'aerialnir.tif'],
    "nhd": ['nhd.tif'],
    "osm": ['osm.tif'],
    "ep": ['ep_5x5.tif', 'ep_11x11.tif', 'ep_21x21.tif', 'ep_51x51.tif', 'ep_101x101.tif', 'ep_201x201.tif'],
    "plc": ['plancurv.tif', 'plancurv_10.tif', 'plancurv_20.tif', 'plancurv_50.tif', 'plancurv_100.tif', 'plancurv_200.tif'],
    "prc": ['procurv.tif', 'procurv_10.tif', 'procurv_20.tif', 'procurv_50.tif', 'procurv_100.tif', 'procurv_200.tif'],
    "slope": ['slope.tif', 'slope_10.tif', 'slope_20.tif', 'slope_50.tif', 'slope_100.tif', 'slope_200.tif'],
    "sds": ['stdslope_5x5.tif', 'stdslope_11x11.tif', 'stdslope_21x21.tif', 'stdslope_51x51.tif', 'stdslope_101x101.tif', 'stdslope_201x201.tif'],
    }


##############################
# SG Map Units & Colors (for visualizations)
##############################

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


##############################
# Default Filenames
##############################

##### dataset directory & files
DATASET_DIR = r'../data'

GLOBAL_AREAS_PATH = f"{DATASET_DIR}/earthscape_{STR_VERSION}_areas.csv"
GLOBAL_MAPPING_PATH = f"{DATASET_DIR}/earthscape_{STR_VERSION}_mapping.json"
GLOBAL_PATCHES_PATH = f"{DATASET_DIR}/earthscape_{STR_VERSION}_patches.geojson"
GLOBAL_STATS_PATH = f"{DATASET_DIR}/earthscape_{STR_VERSION}_stats.csv"

# LOCAL_AREAS_BASE = f"{STR_VERSION}_areas.csv"
# LOCAL_MAPPING_BASE = f"{STR_VERSION}_mapping.json"
# LOCAL_PATCHES_BASE = f"{STR_VERSION}_patches.geojson"
# LOCAL_STATS_BASE = f"{STR_VERSION}_stats.csv"


##### model output directory & files
MODEL_DIR = r'../models'
SPLIT_DIR = f"{MODEL_DIR}/splits"
MODEL_CLF_DIR = f"{MODEL_DIR}/classification"

# SPLIT_TRAIN_PATH = f"{SPLIT_DIR}/{STR_VERSION}_train.geojson"
# SPLIT_VAL_PATH = f"{SPLIT_DIR}/{STR_VERSION}_val.geojson"
# SPLIT_TEST_PATH = f"{SPLIT_DIR}/{STR_VERSION}_test.geojson"
# SPLIT_CROSS_PATH = f"{SPLIT_DIR}/{STR_VERSION}_cross.geojson"


##### smoke set directory & files
SMOKE_DIR = r'../smoke'
SMOKE_SPLITS_DIR = f"{SMOKE_DIR}/splits"
SMOKE_PATCHES_DIR = f"{SMOKE_DIR}/patches"


##############################
# Data Sources
##############################
SGMAP_URLS = []
DEM_BASE_URLS = []
AERIAL_BASE_URLS = []
AERIAL_TILE_INDEX_URL = []
NHD_URL = []
OSM_URL = []
