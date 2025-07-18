# EarthScape

[![Paper](https://img.shields.io/badge/Paper-10.48550%2FarXiv.2503.15625-BB3E00)](https://doi.org/10.48550/arXiv.2503.15625)
[![Dataset](https://img.shields.io/badge/Dataset-10.13023%2Fkgs.data.05.01.2025-FFA55D)](https://uknowledge.uky.edu/kgs_data/16/)
[![Python](https://img.shields.io/badge/Python-3.10+-FFDF88)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-5E936C)](https://creativecommons.org/licenses/by/4.0/)

***EarthScape*** is a living, open-source, AI-ready geospatial dataset for surficial geologic mapping and Earth surface analysis, and includes:

- Expert-labeled surficial geologic masks and labels
- LiDAR-derived DEMs and geomorphometric terrain features at multiple spatial resolutions  
- High-resolution aerial RGB+NIR imagery  
- Hydrography and infrastructure vector overlays  
- Baseline models for multilabel classification

![logo](https://github.com/masseygeo/earthscape/blob/main/data/warren_256_50_21983_modalities.jpg)


## Navigating the Repository
- 📁 **[../code](https://github.com/masseygeo/earthscape/tree/main/code) – Directory containing all code used for dataset curation pipeline, dataloaders, and models.**
  
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
    - [**../data/visualizations.ipynb**](https://github.com/masseygeo/earthscape/blob/main/code/visualizations.ipynb)
      
- 📁 **[../data](https://github.com/masseygeo/earthscape/tree/main/data) – Directory containing all data, including location GeoJSONs, label and area CSVs, and GeoTIFF images.**
  
  - *Class labels (all), class areas (all), class encoding key, patch locations (GIS), and example visualization images.*
    - [**../data/earthscape_areas.csv**](https://github.com/masseygeo/earthscape/blob/main/data/earthscape_areas.csv)
    - [**../data/earthscape_labels.csv**](https://github.com/masseygeo/earthscape/blob/main/data/earthscape_labels.csv)
    - [**../data/earthscape_locations.geojson**](https://github.com/masseygeo/earthscape/blob/main/data/earthscape_locations.geojson)
    - [**../data/hardin_sonora_256_50_2950.png**](https://github.com/masseygeo/earthscape/blob/main/data/hardin_sonora_256_50_2950_modalities.jpg)
    - [**../data/warren_256_50_21983.png**](https://github.com/masseygeo/earthscape/blob/main/data/warren_256_50_21983_modalities.jpg)
      
  - *GeoTIFF images and per-patch labels (not saved in GitHub; see download links given in the "Exploring the Dataset" section below).*
    - [**../data/patches_warren**](https://github.com/masseygeo/earthscape/tree/main/data/patches_warren)
      - ../data/patches_warren/*.tif
      - ../data/patches_warren/*.csv
    - [**../data/patches_hardin**](https://github.com/masseygeo/earthscape/tree/main/data/patches_hardin)
      - ../data/patches_hardin/*.tif
      - ../data/patches_hardin/*.csv
        
- 📁 **[../models**](https://github.com/masseygeo/earthscape/tree/main/models) – Directory containing selected training patches and model results.**
  
  - *Selected training, validation, testing, and cross-domain testing patches (GIS).*
    - [**../models/patches**](https://github.com/masseygeo/earthscape/tree/main/models/patches)
      - [**../models/patches/warren_patches_train.geojson**](https://github.com/masseygeo/earthscape/blob/main/models/patches/warren_patches_train.geojson)
      - [**../models/patches/warren_patches_val.geojson**](https://github.com/masseygeo/earthscape/blob/main/models/patches/warren_patches_val.geojson)
      - [**../models/patches/warren_patches_test.geojson**](https://github.com/masseygeo/earthscape/blob/main/models/patches/warren_patches_test.geojson)
      - [**../models/patches/hardin_patches_test.geojson**](https://github.com/masseygeo/earthscape/blob/main/models/patches/hardin_patches_test.geojson)
     
  - *Unimodal and multimodal model checkpoints, results, and visualizations.*
    - [**../models/classification**](https://github.com/masseygeo/earthscape/tree/main/models/classification) 


## Exploring the Dataset
[![Version](https://img.shields.io/badge/Version-1.0.1-BB3E00)](#)
[![Available](https://img.shields.io/badge/Available%20Patches-31%2c066-FFA55D)](#)
[![Patch Size](https://img.shields.io/badge/Patch%20Size-256x256-FFDF88)](#)
[![Patch Overlap](https://img.shields.io/badge/Patch%20Overlap-50%25-5E936C)](#)
[![Modalities](https://img.shields.io/badge/Channels-37-BBD8A3)](#)
[![Classes](https://img.shields.io/badge/Classes-7-F0F1C5)](#)

- **Metadata, segmentation masks, vector labels, and images can be downloaded here: https://uknowledge.uky.edu/kgs_data/16/**
  - The *README* and *DataDictionary* contain basic metadata and file structure information.
  - A *small example .zip file (15.1 MB)* available for easy exploration of the available data for two patch locations (see the main "DOWNLOAD" link on the landing page).
    - It is strongly recommended to inspect this first before downloading the full dataset packages!
  - The full datasets for each quadrangle may be downloaded from their respective links (~26-32 GB each) shown on the landing page.
  - ***NOTE: This dataset is versioned. All updates and modifications will be reflected in the README. Individual quadrangle datasets should be re-downloaded for the current version.***


- **Individual images are in GeoTIFF format, but can easily be inspected with GIS software (QGIS, ArcGIS) or Python. For Python users, we recommend [Rasterio](https://rasterio.readthedocs.io/en/stable/).**
  ```Python
  import rasterio
  from rasterio.plot import show
  
  with rasterio.open("PATH TO GEOTIFF") as src:
    show(src)
  ```

- **The data pre-processing pipeline can be explored with the following notebooks:**
  - [*Warren County (six quadrangles)*](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_warren.ipynb)
  - [*Sonora Quadrangle*](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_sonora.ipynb)
  - [*Howe Valley Quadrangle*](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_howevalley.ipynb)



## Ongoing Work
- Adding additional 1:24,000-scale surficial geologic quadrangle maps
- Updating the unique patch ID grid for intuitive geospatially aware selection
- Testing additional modalities
  - New terrain features
  - Datasets with broader coverage (e.g., 1/3-arc-second DEM, Sentinel-1, Sentinel-2, etc.)
- Segmentation tests

## Citations
- The dataset:
  - @article{masseyearthscape, title={EarthScape AI Dataset}, author={Massey, Matthew and Imran, Abdullah-Al-Zubaer and others}, publisher={University of Kentucky Libraries}}
  - 
- The manuscript descibing the dataset processing and initial modeling:
  - @article{massey2025earthscape, title={EarthScape: A Multimodal Dataset for Surficial Geologic Mapping and Earth Surface Analysis}, author={Massey, Matthew and Imran, Abdullah-Al-Zubaer}, journal={arXiv preprint arXiv:2503.15625}, year={2025}}
