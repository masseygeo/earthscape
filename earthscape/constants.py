
##############################
# EarthScape Metadata
##############################

VERSION = '1.1'
REPO_URL = r'https://github.com/masseygeo/earthscape'
DATA_URL = r'https://doi.org/10.13023/kgs.data.05.01.2025'
PAPER_URL = r'https://doi.org/10.48550/arXiv.2503.15625'


##############################
# EarthScape 1.0 Modalities
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
    "sdss": ['stdslope_5x5.tif', 'stdslope_11x11.tif', 'stdslope_21x21.tif', 'stdslope_51x51.tif', 'stdslope_101x101.tif', 'stdslope_201x201.tif'],
    }


##############################
# Default Filenames
##############################

##### dataset directory & files
DATASET_DIR = r'../data'

# global files...
GLOBAL_AREAS_BASE = f"earthscape_v_{VERSION.replace('.', '_')}_areas.csv"
# GLOBAL_LABELS_BASE = f"earthscape_v_{VERSION.replace('.', '_')}_labels.csv"
GLOBAL_MAPPING_BASE = f"earthscape_v_{VERSION.replace('.', '_')}_mapping.json"
GLOBAL_PATCHES_BASE = f"earthscape_v_{VERSION.replace('.', '_')}_patches.geojson"
GLOBAL_STATS_BASE = f"earthscape_v_{VERSION.replace('.', '_')}_stats.csv"

# map area local files...
LOCAL_AREAS_BASE = f"v_{VERSION.replace('.', '_')}_areas.csv"
# LOCAL_LABELS_BASE = f"v_{VERSION.replace('.', '_')}_labels.csv"
LOCAL_MAPPING_BASE = f"v_{VERSION.replace('.', '_')}_mapping.json"
LOCAL_PATCHES_BASE = f"v_{VERSION.replace('.', '_')}_patches.geojson"
LOCAL_STATS_BASE = f"v_{VERSION.replace('.', '_')}_stats.csv"


##### model output directory & files
MODEL_OUTPUT_DIR = r'../models'
SPLIT_DIR = f"../models/v_{VERSION.replace('.', '_')}_splits"


##### smoke set directory & files
SMOKE_DIR = r'../smoke'
SMOKE_SPLITS_DIR = f"{SMOKE_DIR}/smoke_v_{VERSION.replace('.', '_')}_splits"
SMOKE_PATCHES_DIR = f"{SMOKE_DIR}/smoke_v_{VERSION.replace('.', '_')}_patches"
SMOKE_GLOBAL_AREAS_BASE = f"smoke_v_{VERSION.replace('.', '_')}_areas.csv"
SMOKE_GLOBAL_MAPPING_BASE = f"smoke_v_{VERSION.replace('.', '_')}_mapping.json"
SMOKE_GLOBAL_PATCHES_BASE = f"smoke_v_{VERSION.replace('.', '_')}_patches.geojson"
SMOKE_GLOBAL_STATS_BASE = f"smoke_v_{VERSION.replace('.', '_')}_stats.csv"



##############################
# Data Sources
##############################
SGMAP_URLS = []
DEM_BASE_URLS = []
AERIAL_BASE_URLS = []
AERIAL_TILE_INDEX_URL = []
NHD_URL = []
OSM_URL = []


##############################
# SG Map Unit Colors from Kentucky Geological Survey
##############################
MAP_COLORS = {
    'af1': '#636566', 
    'Qal': '#fdf5a4', 
    'Qaf': '#ffa1db', 
    'Qat': '#f9e465', 
    'Qc': '#d6c9a7', 
    'Qca': '#c49d83', 
    'Qr': '#b0acd6'
    }
