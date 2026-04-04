# GeoMine AI -- Project Status

**Last updated:** 4 April 2026
**Phase:** 1 COMPLETE -- transitioning to Phase 2
**Study area:** Great Dyke, Zimbabwe (Cr/PGM)

---

## Phase 1 Final Verdict

### The Numbers

| Test | PR-AUC | What It Means |
|---|---|---|
| Random CV | 0.841 | Inflated by spatial leakage |
| **Leave-one-tile-out CV** | **0.228** | **Signal does not generalize across tiles** |
| Along-strike CV | 0.599 | Mixed (0.11 to 0.95 per segment) |
| Random baseline | 0.226 | Class prior |
| Bootstrap coef CI | [0.29, 1.68] | Direction stable, magnitude uncertain |

### What We Proved

1. **Ferric iron (Fe3+) correlates with Cr/PGM deposits within T35KRU.** The signal is real within its geographic anchor.
2. **The coefficient direction is stable** (bootstrap CI does not cross zero). Higher Fe3+ = more prospective is robust.
3. **The pipeline works end-to-end.** Download -> process -> compute indices -> train -> predict -> validate.
4. **Spatial leakage is the primary failure mode.** 10 of 12 deposits cluster in one tile. The model cannot predict deposits it hasn't geographically "seen."

### What We Did NOT Prove

- That ferric_iron generalizes across tiles (it doesn't -- LOTO PR-AUC = 0.228)
- That this is a commercially deployable prediction engine
- That the signal captures geological structure rather than local spectral conditions

### Epistemic Status

**"Preliminary local signal detected. Not yet generalizable."**

This is a valuable negative result. The failure mode (domain shift across tiles) is the moat -- whoever solves cross-tile invariance owns the first-mover advantage in satellite mineral exploration.

---

## What Was Built

### Codebase (~6,500 lines)

```
geomine/
  cli.py              CLI: download, compute-features, train, predict, layers
  ingest/             Sentinel-2 (Copernicus), DEM (AWS), MRDS labels
  spectral/           10 spectral indices, 7 band composite presets
  structural/         Slope, aspect, curvature, lineaments, proximity, drainage
  training/           Spatial CV, along-strike CV, baselines, SHAP, lithology guard
  predict/            Raster prediction, uncertainty, target clustering
  utils/              Config, raster operations
```

### Data Processed

| Data | Size | Tiles |
|---|---|---|
| Sentinel-2 L2A (4 tiles, dry season 2023) | 4.5 GB raw, 13.5 GB stacked | T35KRU, T36KTC, T36KTD, T36KTE |
| Copernicus DEM 30m (9 tiles) | 330 MB | Full Great Dyke coverage |
| Spectral indices (6 per tile) | ~8 GB | ferric_iron, ferrous_iron, clay_ratio, iron_oxide, ndvi, clay_swir |
| Terrain features | 5.5 GB | slope, aspect, curvature, hillshade x8, lineaments, drainage |
| Training labels | 17 deposits | Cr (6), PGM (5), Ni (3), Au (3) |

### Model Outputs

| Output | Description |
|---|---|
| probability_*.tif (4 tiles) | LR(ferric_iron) probability maps |
| uncertainty_*.tif (4 tiles) | Fold variance uncertainty maps |
| top5_targets.tif | Top 5% target mask (603 km2) |
| shap_summary.png | Feature importance |
| calibration.png | Model calibration curve |
| spatial_stress_test.json | LOTO + strike CV + bootstrap results |

---

## Phase 2 Plan: Bridge from 0.228 to 0.55+

### Thread A: Scene Normalization + Physics-Informed Features

The core problem: ferric_iron values are not comparable across tiles from different dates, atmospheric conditions, and sun angles. A deposit in T35KRU has a different raw spectral value than the same mineral in T36KTE.

**Approach:**
1. **Z-score normalization per tile** -- subtract tile mean, divide by tile std. Makes features comparable across scenes.
2. **First-derivative spectral ratios** -- use band ratios and normalized differences rather than raw reflectance. More invariant to illumination.
3. **Pseudo-invariant feature calibration** -- identify stable targets (deep water, bare rock) present in multiple tiles and use them as cross-calibration anchors.
4. **Continuum removal** -- normalize spectral curves by their hull, isolating absorption features from background reflectance.

**Validation:** Re-run LOTO CV after normalization. Target: PR-AUC >= 0.45.

### Thread B: Label Expansion (parallel)

12 deposits across 4 tiles is not enough. Need 40-50 spread geographically.

**Sources:**
- Zimbabwe Geological Survey records
- Published academic literature on Great Dyke chromitite seams
- Mining company annual reports (Zimplats, Unki, Mimosa publish deposit locations)
- MINDAT API (when access granted)
- Manual digitization from geological maps

**Target:** At least 10 deposits per tile, 40+ total.

### Thread C: ASTER SWIR Integration

Sentinel-2 has 2 SWIR bands. ASTER has 6 SWIR bands (1.6-2.43 um) with far better mineral discrimination. Phase 2 adds ASTER for Al-OH, Mg-OH, and carbonate indices that can distinguish specific alteration assemblages.

### Success Criteria for Phase 2

| Metric | Target | What It Proves |
|---|---|---|
| LOTO CV PR-AUC | >= 0.55 | Signal generalizes across tiles |
| Along-strike CV PR-AUC | >= 0.65 | Signal generalizes along the Dyke |
| Number of deposits | >= 40 | Statistically meaningful |
| SHAP top feature | ferric_iron or derivative | Geologically interpretable |

---

## Git History (18 commits)

```
dad7fc2  data: spatial stress test -- tile CV collapses to 0.228
b66d897  feat: Phase 1 complete -- LR(ferric_iron) model, probability maps
a6d88d7  feat: add multi-tile processing and full signal check script
4e594dc  data: add signal check results -- ferric_iron and inverted ferrous_iron
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

## Lessons Learned

### Technical
- GDAL deflate + nodata=0 silently drops valid SWIR data
- Sentinel-2 STAC assets are S3 internal paths, use OData for downloads
- `src.read(out_shape=...)` for same-CRS resampling, not `rasterio.warp.reproject`
- Conda is essential for GDAL on macOS

### Scientific
- **Single-feature models beat complex models at small N.** LR(ferric_iron) outperformed XGBoost, RF, and multi-feature LR. Occam's razor is not optional.
- **Spatial leakage is the silent killer.** Random CV PR-AUC 0.841 collapsed to 0.228 under tile-based CV. Never trust random CV on geospatial data.
- **Weak univariate signals (Z=0.7) do not survive cross-geography tests.** The signal was real but local.
- **Bootstrap the coefficient.** A single point estimate (0.84) means nothing without CI ([0.29, 1.68]). The width matters.
- **The failure mode IS the IP.** Solving cross-tile spectral normalization is the moat, not the ML model.

### Process
- Small frequent commits beat large batches
- Run baselines before fancy models -- always
- The visual QGIS check is more important than any metric
- Build the pipeline end-to-end first, then validate science
- Honest negative results are more valuable than inflated positive ones
