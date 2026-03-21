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
- [Baseline Benchmarks](#baseline-benchmarks)
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
- :file_folder: [experiments](https://github.com/masseygeo/earthscape/tree/main/experiments) - Multilabel classification and segmentation experiments with configurations for reproducibility.
- :file_folder: [notebooks](https://github.com/masseygeo/earthscape/tree/main/notebooks) - Jupyter notebooks for dataset generation, statistics, splits, and analysis.
- :file_folder: [scripts](https://github.com/masseygeo/earthscape/tree/main/scripts) - Experiment orchestration scripts for bulk hyperparameter sweeps.
- :file_folder: [splits](https://github.com/masseygeo/earthscape/tree/main/splits) - Train, validation, in-domain, and cross-domain test splits; frozen for each major version (v1.x).
- :page_facing_up: [CHANGELOG.md](https://github.com/masseygeo/earthscape/blob/v1.1/CHANGELOG.md) - Version history.
- :page_facing_up: [environment.yml](https://github.com/masseygeo/earthscape/blob/v1.1/environment.yml) - Reproducible environment specification.
- :page_facing_up: [pyproject.yml](https://github.com/masseygeo/earthscape/blob/v1.1/pyproject.toml) - Python package configuration.


## Exploring the Dataset
[![Version](https://img.shields.io/badge/Current%20Version-1.1-BB3E00)](#)
[![Patches](https://img.shields.io/badge/Available%20Patches-31%2c066-FFA55D)](#)
[![Size](https://img.shields.io/badge/Patch%20Size-256x256-FFDF88)](#)
[![Modalities](https://img.shields.io/badge/Channels-38-5E936C)](#)
[![Classes](https://img.shields.io/badge/Classes-7-BBD8A3)](#)
[![EPSG](https://img.shields.io/badge/CRS-EPSG:3089-F0F1C5)](#)


### *Where to get it?*
Metadata, segmentation masks, vector labels, and features can be reproduced with this repository or directly downloaded [here](https://uknowledge.uky.edu/kgs_data/16/).


### *What's included?*

#### Image Patches
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
- Filenames that use a unique patch ID and modality/scale: *{patch_id}_{modality/scale}.tif*

#### Metadata Files
- [areas.csv](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_areas.csv) - Per-patch class area proportions.
- [labels.csv](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_labels.csv) - Binary class presence labels (≥1 pixel).
- [patches.geojson](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_patches.geojson) - Patch geometries.
- [metadata.json](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_metadata.json) - Dataset-level metadata.
- [stats.csv](https://github.com/masseygeo/earthscape/blob/v1.1/data/esv1p1_stats.csv) - Per-modality summary statistics.
- Common keys (*patch_id*, *modality*) enable joins across files and data layers.


### *How was the dataset prepared?*

EarthScape is derived from open-access geospatial data sources, compiled into a co-registered stack at a native resolution of 5 ft (~1.5 m) GSD. Terrain features are computed across multiple spatial scales. Slope and curvatures are calculated using 5x5 windows on DEMs resampled to 5, 10, 20, 50, 100, and 200 ft resolutions. Elevation percentile and slope standard deviation are computed from only the 5 ft DEM using variable kernel sizes (5x5, 11x11, 21x21, 51x51, 101x101, 201x201) to ensure a consistent effective spatial footprint across features. Patches are generated to lie entirely within mapped surficial geologic units, resulting in no background class and no nodata pixels. All layers are validated for spatial alignment and co-registration prior to patch extraction. Final patches are clipped to a common extent and verified for consistent resolution and dimensions.

![pipeline](https://github.com/masseygeo/earthscape/blob/v1.1/assets/data_eda/pipeline.png)

Check out these notebooks to see how each map area was compiled and processed:
- [Hardin County, Howe Valley Quadrangle](https://github.com/masseygeo/earthscape/blob/v1.1/notebooks/data_area_hardin_howevalley.ipynb)
- [Hardin County, Sonora Quadrangle](https://github.com/masseygeo/earthscape/blob/v1.1/notebooks/data_area_hardin_sonora.ipynb)
- [Warren County quadrangles](https://github.com/masseygeo/earthscape/blob/v1.1/notebooks/data_area_warren.ipynb)


### *What do the class labels represent?*
EarthScape defines seven surficial geologic (SG) units that form a mutually exclusive representation of surface cover within each study area. These units correspond to five process-based environments:
- Fluvial deposits — Qal (alluvium), Qat (terrace deposits)
- Debris-flow deposits — Qaf (alluvial fans)
- Hillslope materials — Qc (colluvium), Qca (colluvial aprons)
- In situ weathering products — Qr (residuum)
- Anthropogenic modification — af1 (artificial fill)
Although geographically limited, the represented surface processes are broadly applicable across many landscapes.


### *What are the main characteristics of the dataset?*

#### Class Imbalance
EarthScape exhibits a pronounced long-tailed distribution across its seven classes. Qr appears in 94.4% of patches, whereas the rarest units occur in only 4.6% (Qat) and 0.9% (Qaf) of patches. Effective number of samples ranges from 9,464 (Qr) to 266 (Qaf), and the imbalance ratio per label spans more than two orders of magnitude (1.0-108.4), reflecting strong label-level complexity driven by frequency skew. Beyond global frequencies, EarthScape exhibits marked intra-patch complexity. Mean and standard-deviation class-area proportions show that most patches contain multiple SG units with uneven contributions, and the majority-area rate indicates that Qr dominates more than 70\% of patches while rare units almost never occupy the largest fraction. Patch-level class counts vary widely across the regions, reflecting strong geospatial complexity in how classes co-occur and mix spatially.

| Class | Freq. (n) | Freq. (%) | IRLbl | N Eff. | Mean Patch Area | SD Patch Area | Dominant Class Rate (%)
| :--- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| Qr | 29271 | 94.4 | 1.0 | 9464.6 | 0.651 | 0.358 | 0.701 |
| Qal | 18802 | 60.6 | 1.56 | 8474.5 | 0.089 | 0.168 | 0.058 |
| Qc | 13768 | 44.4 | 2.13 | 7476.3 | 0.142 | 0.242 | 0.148 |
| af1 | 10908 | 35.2 | 2.68 | 6640.7 | 	0.051 | 0.161 | 0.035 |
| Qca | 7666 | 24.7 | 3.82 | 5354.3 | 0.061 | 0.154 | 0.054 |
| Qat | 1435 | 4.6 | 20.40 | 1336.9 | 0.006 | 0.045 | 0.004 |
| Qaf | 270 | 0.9 | 108.41 | 266.4 | 0.0002 | 0.003 | 0.0 |

![class_dist](https://github.com/masseygeo/earthscape/blob/v1.1/assets/data_eda/class_dist.png)

#### Domain Shift
EarthScape spans two disjoint regions in Kentucky, USA, consisting of 23,566 patches from Warren County and 7,452 patches from Hardin County, separated by ~77 km. This structure provides a natural geographic partition for analyzing cross-region variation. Maximum mean discrepancy (MMD) is used to quantify distributional differences between patch-level feature summaries (P5, P10, P25, P50, P75, P90, P95) of selected input modalities from each region. We observe measurable domain shift, including MMD values of 0.365 for RGB, 0.832 for DEM, and 0.164 for a multi-scale terrain stack (EP+S+SDS). Although both regions share the same label set, their input feature distributions differ, reflecting geographic variation and providing a clean, geographically partitioned setting for studying domain shift in multimodal geospatial learning.

| Modality | MMD | | Modality | MMD |
| :-- | :-- | :--  | :-- | :-- |
| DEM | 0.2773 | | - | - |
| RGB | 0.1396 | | - | - |
| - | - | | - | - |


## Baseline Benchmarks

### *Multilabel Classification*

### *Semantic Segmentation*


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
