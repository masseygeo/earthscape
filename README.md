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
- 📁 [*code/*](https://github.com/masseygeo/earthscape/tree/main/code) – Directory containing all code used for dataset curation pipeline, dataloaders, and models
  - In the process of cleaning and organizing...
- 📁 [*data/*](https://github.com/masseygeo/earthscape/tree/main/data) – Directory containing all data, including location GeoJSONs, label CSVs, and 1-channel GeoTIFF images
  - In the process of cleaning and organizing...
- 📁 [*models/*](https://github.com/masseygeo/earthscape/tree/main/models) – Directory containing patch locations used for all training, validation, and testing, and model results
  - In the process of cleaning and organizing...

## Exploring the Dataset
[![Available](https://img.shields.io/badge/Available%20Patches-31%2c066-BB3E00)](#)
[![Patch Size](https://img.shields.io/badge/Patch%20Size-256x256-FFA55D)](#)
[![Patch Overlap](https://img.shields.io/badge/Patch%20Overlap-50%25-FFDF88)](#)
[![Modalities](https://img.shields.io/badge/Channels-37-BBD8A3)](#)
[![Classes](https://img.shields.io/badge/Classes-7-F0F1C5)](#)

The data pre-processing pipeline can be explored with the following notebooks:
- [![Warren](https://github.com/masseygeo/earthscape/blob/main/code/data_prep_warren.ipynb) County (contains six 1:24,000-scale geologic quadrangle maps)

## Exploring Multilabel Classification Results
Selected SGMap-Net results...

## Future Work
- Adding additional 1:24,000-scale surficial geologic quadrangle maps
- Updating the unique patch ID grid for intuitive geospatially aware selection
- Testing additional modalities
  - New terrain features
  - Datasets with broader coverage (e.g., 1/3-arc-second DEM, Sentinel-1, Sentinel-2, etc.)
- Segmentation tests
