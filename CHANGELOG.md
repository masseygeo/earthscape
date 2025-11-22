# **EarthScape CHANGELOG**

---

## **v1.1 — Improved Reproducibility, Organization, and Baseline Pipelines**
**Release:** -  

### New & Enhanced
- **Config System**
  - Added `configs.yaml` for full experiment reproducibility (paths, hyperparameters, modality definitions, model settings).
  - Training scripts now load configs and store a copy inside each run directory.

- **Reproducibility & Seed Control**
  - Added comprehensive seed function covering PyTorch CPU/GPU, numpy, Python, dataloaders, and augmentations.

- **Repository Reorganization**
  - Split codebase into a clearer structure:
    - `earthscape/` — dataset, preprocessing, metadata, utilities.
    - `models/` — architecture definitions, SSL modules, training utilities.
    - `results/` — experiment logs, plots, configs, checkpoints.
  - Improves clarity, modularity, and maintainability.

- **Training & Inference Scripts**
  - Unified training script using configs.
  - Added inference pipeline for computing metrics, ROC/PR curves, and writing outputs.

- **Improved Baseline Implementations**
  - Updated and documented baseline SGMap-Net classification models.
  - Standardized forward passes, output channels, and normalization steps.

- **Dataset EDA Notebook**
  - Added notebook for exploring dataset characteristics:
    - Class imbalance  
    - Patch visualization  
    - Per-modality preview  
    - Global statistics

- **Documentation Updates**
  - Expanded README with dataset overview, folder structure, example usage.
  - Added this changelog.

---

## v1.0 — Initial Release  
**Release:** July 18, 2025  

### Dataset Features
- ~31,000 geospatial patches (256×256 px, 50% overlap).
- 38 aligned channels, including:
  - RGB + NIR aerial imagery  
  - DEM  
  - Multi-scale terrain derivatives (elevation percentile, profile curvature, planform curvature, slope, standard deviation of slope)  
  - OpenStreetMap road and rail centerlines  
  - National Hydrography Dataset High Resolution streams and waterbodies  
  - Detailed surficial geologic masks and proportional map-unit areas  
- Supports classification, segmentation, and regression

### Pipeline & Processing
- Extracted AOIs, downloaded and mosaicked KyFromAbove + USGS tiles.
- Rasterized surficial geologic maps to 5-ft (~1.5 GSD) resolution.
- Computed terrain metrics at six spatial scales.
- Generated patch grid with 50% overlap and computed one-hot + area labels.
- Ensured geospatial alignment of all channels.

### Baseline Experiments
- Multilabel classification using RGB-only, DEM-only, and concatenated modalities.
- Initial findings:
  - Terrain shape derivatives show strong predictive power.  
  - Generalization across counties/regions remains challenging.

### Documentation & Distribution
- Initial README with usage instructions and dataset description.
- Example PyTorch dataset loader included.
