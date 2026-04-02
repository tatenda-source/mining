# GeoMine AI -- Project Status

**Last updated:** 2 April 2026
**Phase:** 1 -- Prove Signal
**Study area:** Great Dyke, Zimbabwe (Cr/PGM)
**Team:** 2 humans + AI agents

---

## What We've Built

### Codebase (~6,000 lines across 24 Python modules)

```
geomine/
  cli.py                    CLI: download, compute-features, train, predict, layers, run-all
  ingest/
    sentinel2.py            Copernicus STAC search + OData download + stack preprocessing
    srtm.py                 Copernicus DEM 30m tile download + mosaic
    mrds.py                 USGS mineral deposit labels + MINDAT stub
  spectral/
    indices.py              10 spectral indices (6 Sentinel-2, 4 ASTER)
    compute.py              Compute all indices, stack into multi-band feature raster
    visualize.py            7 geological band composite presets + QGIS .qml styles
  structural/
    terrain.py              Slope, aspect, curvature, multi-azimuth hillshade
    lineaments.py           Canny + Hough lineament extraction, density, intersections
    proximity.py            Distance-to-features, buffered density, drainage density
  training/
    sampling.py             Exploration-bias-aware negative sampling
    spatial_cv.py           Spatial block CV + along-strike CV for linear intrusions
    train.py                XGBoost/RF + baselines + SHAP + lithology-vs-prospectivity guard
  predict/
    inference.py            Chunked raster prediction, bootstrap uncertainty, target clustering
  utils/
    config.py               YAML config loader + AOI geometry
    raster.py               Reproject, clip, resample, stack, read utilities
```

### Key Design Decisions (from reviewer feedback)

| Concern | Guard |
|---|---|
| Model maps lithology instead of prospectivity | `check_lithology_vs_prospectivity()` -- warns if SHAP top features are bulk lithology discriminators |
| XGBoost overfits with only 17 deposits | `run_baselines()` -- logistic regression + single-feature threshold runs first to calibrate expectations |
| Linear intrusion leaks between CV folds | `create_along_strike_blocks()` -- blocks along Great Dyke's NNE strike (15 degrees) |
| Exploration bias in training labels | Exploration intensity proxy + SHAP audit for distance-to-road features |

---

## Data on Disk

### Raw Downloads

| Dataset | Size | Source | Coverage |
|---|---|---|---|
| Sentinel-2 L2A scene (2023-07-30, 0% cloud) | 1.1 GB | Copernicus Data Space | Tile T36KTD |
| Copernicus DEM 30m (9 tiles) | ~330 MB | AWS Open Data | Full Great Dyke |
| Training labels (17 deposits) | < 1 KB | Published geological literature | Cr, PGM, Ni, Au |

### Processed Data

| Output | Size | Description |
|---|---|---|
| Stacked Sentinel-2 (7 bands, 10m) | 3.4 GB | B02, B03, B04, B08, B8A, B11, B12 -- 10980x10980 px |
| DEM mosaic (EPSG:32736) | 348 MB | Copernicus DEM reprojected to UTM Zone 36S |
| 6 spectral index rasters | ~2 GB total | clay_ratio, iron_oxide, ferric_iron, ndvi, ferrous_iron, clay_swir |

### Spectral Index Statistics

| Index | Median | 5th %ile | 95th %ile | Geological Meaning |
|---|---|---|---|---|
| ferrous_iron | 1.36 | 1.19 | 1.50 | Serpentinite/ultramafic indicator (hosts chromitite) |
| clay_ratio | 1.30 | 1.15 | 1.38 | Weathered alteration of mafic rocks |
| iron_oxide | 1.27 | 1.13 | 1.43 | Laterite caps, gossans, ferricrete |
| ndvi | 0.15 | 0.11 | 0.22 | Sparse vegetation = good spectral visibility |
| ferric_iron | 0.18 | 0.12 | 0.25 | Fe3+ enrichment (weathering indicator) |
| clay_swir | 0.13 | 0.07 | 0.16 | SWIR clay absorption feature |

---

## Signal Check Results (2 April 2026)

**Coverage:** Only 4 of 17 deposits fall within the downloaded Sentinel-2 tile (T36KTD). The remaining 13 are on adjacent tiles not yet downloaded. This is a subset test.

**Deposits in tile:** Selous PGM, Darwendale Chrome, Mutorashanga Chrome, Trojan Nickel

| Index | Deposit Mean | Background Mean | Z-score | Percentile | Signal |
|---|---|---|---|---|---|
| ferrous_iron | 1.193 | 1.354 | -1.45 | 6th | **MODERATE** (deposits *lower* than background) |
| ferric_iron | 0.233 | 0.181 | +1.15 | 88th | **MODERATE** (deposits higher than 88% of scene) |
| iron_oxide | 1.230 | 1.270 | -0.44 | 32nd | WEAK |
| ndvi | 0.176 | 0.156 | +0.48 | 75th | WEAK |
| clay_ratio | 1.300 | 1.291 | +0.12 | 47th | NONE |
| clay_swir | 0.130 | 0.126 | +0.16 | 46th | NONE |

### Interpretation

**Two signals detected with n=4 deposits:**

1. **ferric_iron (Z=+1.15):** Deposits sit at the 88th percentile -- elevated Fe3+ consistent with weathered chromitite and sulphide-bearing horizons. This is a geologically plausible pathfinder.

2. **ferrous_iron (Z=-1.45):** Deposits are *lower* than background. This is counterintuitive -- ultramafic-hosted deposits should have high ferrous silicates. But it may indicate that the deposit horizons (chromitite seams, MSZ) are less serpentinized than the surrounding dunite/harzburgite. Chromitite is chromite + intercumulus silicate, not pure serpentinite. This could actually be a real discriminator *within* the ultramafic sequence.

3. **clay_ratio and clay_swir show no signal** -- clay weathering is uniform across the Dyke, doesn't discriminate deposit vs barren ultramafic. This confirms the reviewer's concern: these indices map lithology, not prospectivity.

### Verdict: MODERATE -- Proceed with Caution

Signal exists in ferric_iron and (inverted) ferrous_iron. But n=4 is dangerously small. Need to download adjacent tiles to get the remaining 13 deposits into the analysis before training.

### What this tells us about the lithology question

The reviewer was right to worry. Clay and iron oxide indices don't discriminate. The discriminating features are **ferric_iron** (oxidation state) and **inverted ferrous_iron** (chromitite/MSZ has different spectral character than surrounding serpentinite). This is a subtle within-ultramafic signal, not a simple lithology boundary.

---

## Environment

- **Python:** 3.11 (conda `geomine` environment)
- **Key packages:** GDAL, rasterio, geopandas, scikit-learn, xgboost, shap, pystac-client
- **JP2 support:** libgdal-jp2openjpeg for Sentinel-2 band reading
- **CLI:** `geomine` command via `pip install -e .`
- **API accounts:** NASA EarthData (active, expires 2026-05-29), Copernicus Data Space (active)

---

## Scientific Claim Scope

**What Phase 1 can claim (if gate passes):**
> "Sentinel-2 spectral and terrain features can discriminate known Cr/PGM deposit locations from barren ground within the Great Dyke layered intrusion, as measured by spatial block cross-validation."

**What Phase 1 cannot claim:**
- Generalization to other geological terranes
- Discovery of unknown deposits (only correlates with known ones)
- Performance in covered terranes (thick regolith, dense vegetation)
- Temporal robustness (single dry-season scene)
- Prospectivity vs lithology mapping (must be verified by SHAP audit)

### The Lithology vs Prospectivity Question

The Great Dyke is ultramafic along its entire 550km strike. A model that learns "ultramafic = deposit" is doing lithology mapping, not prospectivity mapping. Deposits occur at specific horizons within the ultramafic sequence -- not everywhere.

The pipeline guards against this with:
1. Automated SHAP feature audit for lithology vs structural/alteration features
2. Baseline comparison (if ferrous_iron alone gives PR-AUC 0.55, XGBoost must exceed meaningfully)
3. Along-strike CV blocking to prevent adjacent similar segments leaking

### Known Limitations

1. **Single scene** -- models one dry-season day, not persistent geology
2. **20m upsampling** -- SWIR bands fabricate spatial detail at 10m
3. **17 deposits** -- high-variance CV estimates (~12 train / 5 test per fold)
4. **Approximate coordinates** -- some older deposit locations may have km-scale error

---

## What's Next

### Immediate
1. Signal check -- extract spectral values at deposit locations vs random background
2. Terrain features -- slope, aspect, curvature, lineaments from DEM
3. Visual overlay in QGIS -- do anomalies coincide with known deposits?

### Week 2-3
4. Feature stack assembly
5. Training with baselines + XGBoost + spatial block CV
6. SHAP audit + lithology-vs-prospectivity check
7. Probability + uncertainty maps

### Phase 1 Gate

| Outcome | Criteria |
|---|---|
| **Pass** | PR-AUC > 0.60, spatial AUC-ROC > 0.70, SHAP features geologically plausible |
| **Marginal** | PR-AUC 0.45-0.60, model shows signal but needs refinement |
| **Fail** | PR-AUC < 0.45, stop and diagnose before building further |

---

## Git History (15 commits)

```
f0247de  docs: add scientific claim scope, lithology guard, and data limitations
f76da76  feat: add baseline models, lithology-vs-prospectivity guard, along-strike CV
28a0833  docs: add project status report
0482b39  fix: correct setuptools build backend path
b7c95ff  feat: add geological band composite layer presets and rendering
f5712e5  docs: add GeoMine AI v2.1 technical specification
8371b4d  feat: add CLI with full Phase 1 pipeline orchestration
a438230  feat: add prediction engine
91133d3  feat: add ML training pipeline with spatial CV and SHAP audit
75fbc36  feat: add structural geology analysis module
d3e3bbe  feat: add spectral index computation module
2f773a3  feat: add data ingestion
6ceb311  feat: add utility modules
756de14  config: add Great Dyke AOI and Phase 1 pipeline configuration
81adf13  init: project scaffolding
```

---

## Known Issues

1. **MRDS WFS API is down** -- manual labels from published literature
2. **SRTM direct download 404** -- using Copernicus DEM 30m instead (better quality)
3. **GDAL deflate + nodata=0 silently drops SWIR bands** -- write without nodata sentinel
4. **Copernicus STAC assets are S3 internal paths** -- must use OData API for downloads
5. **sentinelsat is deprecated** -- use cdsetool or OData API

---

## Lessons Learned

- Always verify band values after writing rasters (GDAL nodata + compression traps)
- Use `src.read(out_shape=...)` not `rasterio.warp.reproject` when CRS matches
- Conda is essential for GDAL on macOS -- pip installs are fragile
- Copernicus Open Hub is dead -- use Copernicus Data Space Ecosystem
- Run baselines before fancy models -- if logistic regression gets 0.55, XGBoost may only add marginal lift
