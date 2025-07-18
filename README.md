# EarthScape

[![Paper](https://img.shields.io/badge/Paper-10.48550%2FarXiv.2503.15625-BB3E00)](https://doi.org/10.48550/arXiv.2503.15625)
[![Dataset](https://img.shields.io/badge/Dataset-10.13023%2Fkgs.data.05.01.2025-FFA55D)](https://uknowledge.uky.edu/kgs_data/16/)
[![Python](https://img.shields.io/badge/Python-3.10+-FFDF88)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-BBD8A3)](https://creativecommons.org/licenses/by/4.0/)


***EarthScape*** is a living, open-source, AI-ready geospatial dataset for surficial geologic mapping and Earth surface analysis, and includes:

- Expert-labeled surficial geologic masks and labels
- LiDAR-derived DEMs and geomorphometric terrain features at multiple spatial resolutions  
- High-resolution aerial RGB+NIR imagery  
- Hydrography and infrastructure vector overlays  
- Baseline models for multilabel classification


## Navigating the Repository
- 📁 [**../code**](https://github.com/masseygeo/earthscape/tree/main/code) – Directory containing all code used for dataset curation pipeline, dataloaders, and models.
  
  - *Dataset preparation (notebooks and utility functions for source downloads, data manipulation, GIS, and visualizations.)*
    - [**../code/data_prep_howe_valley.ipynb**](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_howevalley.ipynb)
    - [**../code/data_prep_sonora.ipynb**](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_sonora.ipynb)
    - [**../code/data_prep_warren.ipynb**](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_warren.ipynb)
    - [**../code/utils_data.py**](https://github.com/masseygeo/earthscape/blob/main/code/utils_data.py)
      
  - *Modeling (notebooks, utility functions, and scripts for patch selection, dataloader, focal loss, visualizations, and training.)*
    - [**../code/model_dataselection.ipynb**](https://github.com/masseygeo/earthscape/blob/main/code/model_dataselection.ipynb)
    - [**../code/utils_model_dataloader.py**](https://github.com/masseygeo/earthscape/blob/main/code/utils_model_dataloader.py)
    - [**../code/utils_model_training.py**](https://github.com/masseygeo/earthscape/blob/main/code/utils_model_training.py)
    - [**../code/model_classification.ipynb**](https://github.com/masseygeo/earthscape/blob/main/code/model_classification.ipynb)
    - [**../code/run.bat**](https://github.com/masseygeo/earthscape/blob/main/code/run.bat)
    - [**../code/run.sh**](https://github.com/masseygeo/earthscape/blob/main/code/run.sh)
      
- 📁 [**../data**](https://github.com/masseygeo/earthscape/tree/main/data) – Directory containing all data, including location GeoJSONs, label CSVs, and 1-channel GeoTIFF images.
  
  - *Class labels (all), class areas (all), class encoding key, patch locations (GIS), and example visualization images.*
    - [**../data/earthscape_areas.csv**](https://github.com/masseygeo/earthscape/blob/main/data/earthscape_areas.csv)
    - [**../data/earthscape_labels.csv**](https://github.com/masseygeo/earthscape/blob/main/data/earthscape_labels.csv)
    - [**../data/earthscape_locations.geojson**](https://github.com/masseygeo/earthscape/blob/main/data/earthscape_locations.geojson)
    - [**../data/hardin_sonora_256_50_2950.png**](https://github.com/masseygeo/earthscape/blob/main/data/hardin_sonora_256_50_2950_modalities.jpg)
    - [**../data/warren_256_50_21983.png**](https://github.com/masseygeo/earthscape/blob/main/data/warren_256_50_21983_modalities.jpg)
      
  - *GeoTIFF images and per-patch labels (not saved in GitHub; see download links given in the "Exploring the Dataset" section below).*
    - [**../data/patches_warren**](https://github.com/masseygeo/earthscape/tree/main/data/patches_warren)
      - **../data/patches_warren/*.tif**
      - **../data/patches_warren/*.csv**
    - [**../data/patches_hardin**](https://github.com/masseygeo/earthscape/tree/main/data/patches_hardin)
      - **../data/patches_hardin/*.tif**
      - **../data/patches_hardin/*.csv**
        
- 📁 [**../models**](https://github.com/masseygeo/earthscape/tree/main/models) – Directory containing patch locations used for all training, validation, and testing, and model results.
  
  - In the process of cleaning and organizing...


## Exploring the Dataset
[![Available](https://img.shields.io/badge/Available%20Patches-31%2c066-BB3E00)](#)
[![Patch Size](https://img.shields.io/badge/Patch%20Size-256x256-FFA55D)](#)
[![Patch Overlap](https://img.shields.io/badge/Patch%20Overlap-50%25-FFDF88)](#)
[![Modalities](https://img.shields.io/badge/Channels-37-BBD8A3)](#)
[![Classes](https://img.shields.io/badge/Classes-7-F0F1C5)](#)

- The data pre-processing pipeline can be explored with the following notebooks:
  - [*Warren County (six quadrangles)*](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_warren.ipynb)
  - [*Sonora Quadrangle*](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_sonora.ipynb)
  - [*Howe Valley Quadrangle*](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_howevalley.ipynb)

- Metadata, segmentation masks, vector labels, and images can be downloaded here: https://uknowledge.uky.edu/kgs_data/16/
  - The ***README*** and ***DataDictionary*** contain basic metadata and file structure information.
    - These are versioned and all modifications will be captured here.
  - A ***small example .zip file (15.1 MB)*** is available for exploring the available information for two patch locations (see the main "DOWNLOAD" link).
    - It is strongly recommended to inspect this first before downloading the full dataset packages!
  - The full datasets for each quadrangle may be downloaded from their respective links (~26-32 GB each).


## Exploring Multilabel Classification Results
Selected SGMap-Net results...


## Future Work
- Adding additional 1:24,000-scale surficial geologic quadrangle maps
- Updating the unique patch ID grid for intuitive geospatially aware selection
- Testing additional modalities
  - New terrain features
  - Datasets with broader coverage (e.g., 1/3-arc-second DEM, Sentinel-1, Sentinel-2, etc.)
- Segmentation tests
