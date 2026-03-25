# **EarthScape CHANGELOG**

All notable changes to EarthScape will be documented in this file.

---

## [v1.1.0] - 2026-03-24

This release updates the dataset within the existing scope, improves consistency of derived features across geographic areas, reorganizes the repository around reproducibility, includes new benchmarks, and expands documentation.

### Added
- A formal Python package layout under `earthscape/`
- `pyproject.toml` for package installation and project configuration
- New `environment.yml` for reproducible environment setup
- Configuration files in `experiments/*` for reproducible baseline experiments
- Experiment orchestration scripts in `scripts/`
- Dataset and experiment figures/tables used in documentation and analysis in `assets/`
- Explicit release tracking in `CHANGELOG.md `
- Support and documentation for semantic segmentation in addition to multilabel classification

### Changed
- Repository structure was refactored
- README scope was expanded
- The following raster products were standardized to `uint8`:
  - RGB+NIR imagery
  - Segmentation masks
  - OSM infrastructure layers
  - NHD hydrography layers
- Dataset class was updated to support:
  - Semantic segmentation
  - One-hot label vectors based on user-defined threshold of patch-level class-area proportion
  - Class-area proportions as target vectors
- Corrected inconsistent computation of profile and planform curvatures across available geographic regions
- Modified focal loss class to support binary cross-entropy, positive weighting, class weighting, and focal parameter.

---

## [v1.0.1] - 2025-07-18

Initial release of the EarthScape dataset and repository.
- Public release of the EarthScape dataset with 31,066 patches, 38 channels, and 7 surficial geologic classes
- Benchmark support for multilabel classification
- Dataset download and exploratory usage documentation in the README
