# EarthScape: A Multimodal Dataset and Benchmark for Surficial Geologic Mapping and Earth Surface Analysis

![logo](https://github.com/masseygeo/earthscape/blob/v1.1/assets/data_eda/esv1p1_warren_modalities.png)


## Introduction
[![Paper](https://img.shields.io/badge/Paper-10.48550%2FarXiv.2503.15625-BB3E00)](https://doi.org/10.48550/arXiv.2503.15625)
[![Dataset](https://img.shields.io/badge/Dataset-10.13023%2Fkgs.data.05.01.2025-FFA55D)](https://uknowledge.uky.edu/kgs_data/16/)
[![Python](https://img.shields.io/badge/Python-3.12+-FFDF88)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

EarthScape is a living, open-source, AI-ready geospatial dataset for surficial geologic mapping and Earth surface analysis. It includes:

- Expert-labeled surficial geology masks and annotations
- DEM-derived terrain features computed across multiple spatial resolutions
- High-resolution optical imagery (RGB + NIR)
- Hydrography and infrastructure vector layers
- Seven classes that represent common geomorphic processes and capturing long-tail distribution and spatial complexity.
- Multiple geographic areas within the same label space intentionally designed to support covariate shift studies
- Baseline benchmarks for multilabel classification and semantic segmentation


## Table of Contents
- [Intallation and Quickstart](#installation-and-quickstart)
- [Navigating the Repository](#navigating-the-repository)
- [Exploring the Dataset](#exploring-the-dataset)
- [Roadmap](#roadmap)
- [Citations](#citations)


## Installation and Quickstart

1. **Install [Conda](https://anaconda.org/channels/anaconda/packages/conda/overview) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main).**
2. **Create the environment.**
    - *If using a CPU-only machine, remove 'pytorch-gpu' from environment.yml.*
```Bash
conda env create -f environment.yml
```
3. **Activate the environment.**
```Bash
conda activate earthscape
```
4. **Install the earthscape package.**
```Bash
pip install -e .
```

You can now use EarthScape to reproduce the dataset, train and evaluate models, and run analyses. While the full data pipeline is available, most users will prefer downloading the precompiled dataset.

5. **Download the [dataset and metadata](https://uknowledge.uky.edu/kgs_data/16/).**
6. **Extract all archives into the [data](https://github.com/masseygeo/earthscape/tree/v1.1/data) directory.**


## Navigating the Repository
- :file_folder: [assets](https://github.com/masseygeo/earthscape/tree/main/assets) - Figures and tables from dataset and experiment analyses.
- :file_folder: [data](https://github.com/masseygeo/earthscape/tree/main/data) - Dataset and associated metadata; frozen for each minor version (v1.1.x)
- :file_folder: [earthscape](https://github.com/masseygeo/earthscape/tree/main/earthscape) - Core source code.
- :file_folder: [experiments](https://github.com/masseygeo/earthscape/tree/main/experiments) - Multilabel classification and segmentation experiments with configurations for reproducible workflows.
- :file_folder: [notebooks](https://github.com/masseygeo/earthscape/tree/main/notebooks) - Jupyter notebooks for dataset generation, statistics, splits, and analysis.
- :file_folder: [scripts](https://github.com/masseygeo/earthscape/tree/main/scripts) - Experiment orchestration scripts for bulk hyperparameter sweeps.
- :file_folder: [splits](https://github.com/masseygeo/earthscape/tree/main/splits) - Train, validation, in-domain, and cross-domain test splits; frozen for each major version (v1.x).
- :page_facing_up: [CHANGELOG.md](https://github.com/masseygeo/earthscape/blob/v1.1/CHANGELOG.md) - Version history.
- :page_facing_up: [environment.yml](https://github.com/masseygeo/earthscape/blob/v1.1/environment.yml) - Reproducible environment specification.
- :page_facing_up: [pyproject.yml](https://github.com/masseygeo/earthscape/blob/v1.1/pyproject.toml) - Python package configuration.


## Exploring the Dataset
[![Version](https://img.shields.io/badge/Version-1.1-BB3E00)](#)
[![Available](https://img.shields.io/badge/Available%20Patches-31%2c066-FFA55D)](#)
[![Patch Size](https://img.shields.io/badge/Patch%20Size-256x256-FFDF88)](#)
[![Patch Overlap](https://img.shields.io/badge/Patch%20Overlap-50%25-5E936C)](#)
[![Modalities](https://img.shields.io/badge/Channels-38-BBD8A3)](#)
[![Classes](https://img.shields.io/badge/Classes-7-F0F1C5)](#)


### *Where to get it?*
Metadata, segmentation masks, vector labels, and features can be reproduced with this repository or directly downloaded [here](https://uknowledge.uky.edu/kgs_data/16/).


### *What's included?*
#### Image Patches
- Overview
  - 31,066 patches (256 x 256 pixels)
  - Each patch covers ~1,280 x 1,280 ft at 5 ft (~1.5 m) GSD
  - 50% overlap between adjacent patches
- Geospatial
  - Coordinate reference system: EPSG:3089
  - Two study areas separated by ~77 km
- Data Layers
  - Segmentation masks
  - Aerial optical imagery (RGB+NIR)
  - LiDAR DEM
  - OpenStreetMap road and railway centerlines
  - U.S. Geological Survey National Hydrography Dataset stream flowlines and water body polygons
  - DEM-derived terrain features calculated at six spatial resolutions:
    - Elevation Percentile
    - Planform Curvature
    - Profile Curvature
    - Slope
    - Standard Deviation of Slope
  - Filenames that use a unique patch ID and modality/scale
    - *{patch_id}_{modality/scale}.tif*

#### Metadata Files
- [areas.csv](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_areas.csv) - Per-patch class area proportions.
- [labels.csv](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_labels.csv) - Binary class presence labels (≥1 pixel).
- [patches.geojson](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_patches.geojson) - Patch geometries.
- [metadata.json](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_metadata.json) - Dataset-level metadata.
- [stats.csv](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_stats.csv) - Per-modality summary statistics.
- Common keys (*patch_id*, *modality*) enable joins across files and data layers.


### *How was the dataset prepared?*


Check out the dataset compilation pipeline notebooks to see how each area was compiled:
- [Hardin County, Howe Valley Quadrangle](https://github.com/masseygeo/earthscape/blob/v1.1/notebooks/data_area_hardin_howevalley.ipynb)
- [Hardin County, Sonora Quadrangle](https://github.com/masseygeo/earthscape/blob/v1.1/notebooks/data_area_hardin_sonora.ipynb)
- [Warren County quadrangles](https://github.com/masseygeo/earthscape/blob/v1.1/notebooks/data_area_warren.ipynb)



### *How to explore?*

#### Dataset compilation notebooks
Check out the dataset compilation pipeline notebooks to see how each area was compiled:
- [](https://github.com/masseygeo/earthscape/blob/v1.1/notebooks/data_area_hardin_howevalley.ipynb)


#### Self-exploration
Individual images are in GeoTIFF format, and can easily be inspected with GIS software (QGIS, ArcGIS) or Python. For Python users, we recommend [Rasterio](https://rasterio.readthedocs.io/en/stable/).

  ```Python
  import rasterio
  from rasterio.plot import show
  
  with rasterio.open("PATH TO GEOTIFF") as src:
    show(src)
  ```





## Roadmap
- Adding additional 1:24,000-scale surficial geologic quadrangle maps
- Updating the unique patch ID grid for intuitive geospatially aware selection
- Additional modalities
  - New terrain features
  - Datasets with broader coverage (e.g., 1/3-arc-second DEM, Sentinel-1, Sentinel-2, etc.)



## Citations
```
# dataset download
@article{
  masseyearthscape, title={EarthScape AI Dataset},
  author={Massey, Matthew and Imran, Abdullah-Al-Zubaer and others},
  publisher={University of Kentucky Libraries}
  }
    
# manuscript descibing processing and benchmarks
@article{
  massey2025earthscape,
  title={EarthScape: A Multimodal Dataset for Surficial Geologic Mapping and Earth Surface Analysis},
  author={Massey, Matthew and Imran, Abdullah-Al-Zubaer},
  journal={arXiv preprint arXiv:2503.15625},
  year={2025}
  }
```
