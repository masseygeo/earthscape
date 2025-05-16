# EarthScape

[![Paper](https://img.shields.io/badge/Paper-10.48550%2FarXiv.2503.15625-BB3E00)](https://doi.org/10.48550/arXiv.2503.15625)
[![Dataset](https://img.shields.io/badge/Dataset-10.13023%2Fkgs.data.05.01.2025-F7AD45)](https://uknowledge.uky.edu/kgs_data/16/)
[![Python](https://img.shields.io/badge/Python-3.10+-B6B09F)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-EAE4D5)](https://creativecommons.org/licenses/by/4.0/)


## Overview
***EarthScape*** is a living, open-source, AI-ready geospatial dataset for surficial geologic mapping and Earth surface analysis, and includes:

- Expert-labeled surficial geologic masks and labels
- LiDAR-derived DEMs and geomorphometric terrain features at multiple spatial resolutions  
- High-resolution aerial RGB+NIR imagery  
- Hydrography and infrastructure vector overlays  
- Baseline models for multilabel classification

## Navigating the Repository
- 📁 *code/* – Directory containing all code used for dataset curation pipeline, dataloaders, and models
  - In the process of cleaning and organizing...
- 📁 *data/* – Directory containing all data, including location GeoJSONs, label CSVs, and 1-channel GeoTIFF images
  - In the process of cleaning and organizing...
- 📁 *models/* – Directory containing patch locations used for all training, validation, and testing, and model results
  - In the process of cleaning and organizing...

## Exploring the Dataset
[![Images](https://img.shields.io/badge/Images-31,066-B6B09F)]
[![Patch Size](https://img.shields.io/badge/Patch%20Size-256x256-B6B09F)]
[![Patch Overlap](https://img.shields.io/badge/Patch%20Overlap-50%25-B6B09F)]
[![Modalities](https://img.shields.io/badge/Channels-37-B6B09F)]
[![Classes](https://img.shields.io/badge/Classes-7-B6B09F)]


## Exploring Multilabel Classification Results
Selected SGMap-Net results...

## Future Work
- Adding X additional 1:24,000-scale surficial geologic quadrangle maps (~X more patches!)
- Updating the unique patch ID grid for intuitive geospatially aware selection
- Testing additional modalities
  - New terrain features
  - Using datasets with broader coverage (e.g., 1/3-arc-second DEM, Sentinel-1, Sentinel-2, etc.)
- Segmentation tests
