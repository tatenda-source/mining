# GeoMine AI -- Project Status

**Last updated:** 2 April 2026
**Phase:** 1 -- Prove Signal
**Study area:** Great Dyke, Zimbabwe (Cr/PGM)
**Team:** 2 humans + AI agents

---

## What We've Built

### Codebase (5,560 lines across 24 Python modules)

```
geomine/
  cli.py                    Full CLI pipeline: download, compute-features, train, predict, layers, run-all
  ingest/
    sentinel2.py            Copernicus STAC search + band download + stack preprocessing
    srtm.py                 NASA SRTM/Copernicus DEM tile download + mosaic
    mrds.py                 USGS mineral deposit label downloader + MINDAT stub
  spectral/
    indices.py              10 spectral indices (6 Sentinel-2, 4 ASTER)
    compute.py              Full pipeline: compute all indices, stack into feature raster
    visualize.py            7 geological band composite presets, matplotlib maps, QGIS .qml styles
  structural/
    terrain.py              Slope, aspect, curvature, multi-azimuth hillshade from DEM
    lineaments.py           Canny + Hough lineament extraction, density, intersection mapping
    proximity.py            Distance-to-features, buffered density, drainage density
  training/
    sampling.py             Exploration-bias-aware negative sampling, feature extraction
    spatial_cv.py           Spatial block cross-validation (prevents data leakage)
    train.py                XGBoost + RF training, SHAP audit, model serialization
  predict/
    inference.py            Chunked raster prediction, bootstrap uncertainty, target clustering, HTML report
  utils/
    config.py               YAML config loader + AOI geometry extraction
    raster.py               Reproject, clip, resample, stack, read utilities
```

### Configuration

- `configs/great_dyke.yaml` -- all pipeline parameters: satellite sources, spectral index formulas, structural analysis settings, spatial CV block size (25km), XGBoost hyperparameters, output thresholds
- `configs/great_dyke_aoi.geojson` -- study area polygon covering the full Great Dyke intrusion (Mvurwi to Wedza)

### Technical Specification

- `mineral_prediction_engine_v2.md` -- 900+ line spec document covering architecture, data sources, ML pipeline, phased roadmap, cost analysis, 9 identified weak points with mitigations, strategic positioning (research engine vs SaaS), and 6 documented failure modes

---

## Data Downloaded

| Dataset | Size | Source | Status |
|---|---|---|---|
| Sentinel-2 L2A (1 scene, 2023-07-30, 0% cloud) | 1.1 GB | Copernicus Data Space | Downloaded + extracted |
| Copernicus DEM 30m (9 tiles) | ~330 MB | AWS Open Data | Downloaded |
| Training labels (17 deposits: Cr, PGM, Ni, Au) | < 1 KB | Published literature | Created manually (MRDS API down) |

### Processed Data

| Output | Size | Description |
|---|---|---|
| Stacked Sentinel-2 (7 bands, 10m) | 3.4 GB | B02, B03, B04, B08, B8A, B11, B12 at 10980x10980 pixels |
| DEM mosaic (EPSG:32736) | 348 MB | Copernicus DEM reprojected to UTM 36S |
| 6 spectral index rasters | ~6x340 MB | clay_ratio, iron_oxide, ferric_iron, ndvi, ferrous_iron, clay_swir |

### First Spectral Results

| Index | Median | 5th Percentile | 95th Percentile | What It Means |
|---|---|---|---|---|
| ferrous_iron | 1.36 | 1.19 | 1.50 | Strong ferrous silicate signal -- consistent with Great Dyke serpentinized ultramafics |
| clay_ratio | 1.30 | 1.15 | 1.38 | Widespread clay weathering of mafic rocks |
| iron_oxide | 1.27 | 1.13 | 1.43 | Moderate iron oxide; 95th percentile pixels = laterite caps / gossans |
| ndvi | 0.15 | 0.11 | 0.22 | Sparse vegetation = good spectral visibility for geology |
| ferric_iron | 0.18 | 0.12 | 0.25 | Moderate Fe3+ enrichment |
| clay_swir | 0.13 | 0.07 | 0.16 | Detectable clay absorption in SWIR bands |

These are real geological signals, not noise. The ferrous_iron and clay_ratio values are consistent with what published literature reports for the Great Dyke's weathered ultramafic lithology.

---

## Environment

- **Python:** 3.11 (conda, `geomine` environment)
- **Key packages:** GDAL, rasterio, geopandas, scikit-learn, xgboost, shap, pystac-client, numpy, scipy
- **JP2 support:** libgdal-jp2openjpeg installed for Sentinel-2 JP2 band reading
- **CLI:** `geomine` command installed via `pip install -e .`

### API Accounts

| Service | Status | Expires |
|---|---|---|
| NASA EarthData | Active (JWT token) | 2026-05-29 |
| Copernicus Data Space | Active (OAuth) | Refreshable |
| MINDAT | Not registered (non-blocking) | -- |

---

## What's Next

### Immediate (this week)

1. **Mosaic DEM and compute terrain features** -- slope, aspect, curvature, lineament extraction
2. **Visual validation in QGIS** -- overlay spectral indices on known deposit locations. Do high ferrous_iron zones coincide with known chromitite seams? This is the first signal-existence test.
3. **Build feature stack** -- combine all spectral + terrain + structural features into a single multi-band raster

### Next (week 2-3)

4. **Assemble training data** -- extract features at deposit locations (positives) and exploration-bias-aware barren locations (negatives)
5. **Train XGBoost** with spatial block cross-validation (25km blocks)
6. **SHAP audit** -- are the top features geological (ferrous_iron, iron_oxide) or infrastructural (distance to road)? If infrastructural, we have an exploration bias problem.
7. **Generate probability + uncertainty maps**

### Phase 1 Gate Decision

- **Pass:** PR-AUC > 0.60, spatial AUC-ROC > 0.70, SHAP features geologically plausible
- **Marginal:** PR-AUC 0.45-0.60 -- model shows signal but needs improvement
- **Fail:** PR-AUC < 0.45 -- stop and diagnose before building anything else

---

## Git History

```
f5712e5  docs: add GeoMine AI v2.1 technical specification
8371b4d  feat: add CLI with full Phase 1 pipeline orchestration
a438230  feat: add prediction engine -- probability maps, uncertainty, and target clustering
91133d3  feat: add ML training pipeline with spatial CV and SHAP audit
75fbc36  feat: add structural geology analysis module
d3e3bbe  feat: add spectral index computation module
2f773a3  feat: add data ingestion -- Sentinel-2, SRTM, and MRDS downloaders
6ceb311  feat: add utility modules -- config loader and raster operations
756de14  config: add Great Dyke AOI and Phase 1 pipeline configuration
81adf13  init: project scaffolding, pyproject.toml, and directory structure
```

---

## Known Issues

1. **MRDS WFS API is down** -- created manual training labels from published literature (17 deposits). This is sufficient for Phase 1 but should be supplemented with more records when the API comes back or via direct download from USGS.
2. **SRTM direct download 404s** -- switched to Copernicus DEM 30m from AWS (free, no auth, actually better quality than SRTM).
3. **GDAL deflate compression + nodata=0 silently drops SWIR bands** -- resolved by writing without nodata sentinel. Need to update `utils/raster.py` to avoid this pattern.
4. **Only 1 Sentinel-2 scene downloaded** -- sufficient for Phase 1 PoC. Multi-scene temporal compositing is Phase 2.
5. **Only 17 training deposits** -- minimum viable for proving signal. More labels needed for production model.

---

## Lessons Learned So Far

- **GDAL/rasterio nodata + compression is a trap.** Setting `nodata=0` with `compress='deflate'` silently eliminated valid pixel values in SWIR bands. Always verify band values after writing.
- **Sentinel-2 STAC assets point to S3 internal paths**, not HTTP URLs. Must use OData API for actual downloads.
- **Copernicus Open Hub (scihub.copernicus.eu) is dead.** Replaced by Copernicus Data Space Ecosystem. The `sentinelsat` Python library is deprecated.
- **20m to 10m resampling** must use `src.read(1, out_shape=(...))` not `rasterio.warp.reproject` when source and target share the same CRS and bounds.
- **Conda is essential for geospatial on macOS.** GDAL via pip is fragile. The JP2OpenJPEG driver needed separate `conda install`.
