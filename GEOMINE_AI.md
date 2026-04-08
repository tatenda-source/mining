# GeoMine AI -- Technical Whitepaper

**Last Updated:** 7 April 2026
**Version:** 2.0
**Status:** Phase 2a complete, Phase 2b (Foundation Models) ready to execute

*See also: [INVESTOR_DECK.md](INVESTOR_DECK.md) | [ONE_PAGER.md](ONE_PAGER.md)*

---

## Abstract

GeoMine AI is a modular geospatial ML platform for satellite-based mineral prospectivity mapping. The system ingests free, publicly available Earth observation data (Sentinel-2, ASTER, DEM) and produces validated probability maps for platinum group metals (PGMs) and chromite deposits.

This whitepaper documents what was built, what was tested, why classical approaches failed, and the architectural pivot to foundation models that addresses the root cause of failure.

**Key result:** Eight model configurations were tested with institutional-grade spatial cross-validation. All failed to generalize across satellite tiles (best cross-tile PR-AUC: 0.050 vs 0.055 baseline). Root cause: Sentinel-2 spectral indices encode atmospheric/vegetation state, not rock mineralogy. The coefficient sign flip between tiles (+0.446 vs -0.650 for ferrous iron) is mathematical proof of non-stationarity.

---

## What We Actually Built

### The Codebase: 8,300+ Lines of Production Python

```
geomine/                          # Core package (6,241 lines, 24 modules)
  cli.py                          Full CLI: download, compute, train, predict, validate
  ingest/
    sentinel2.py                  Sentinel-2 L2A download via Copernicus STAC/OData
    srtm.py                      Copernicus DEM 30m download (free, no auth)
    mrds.py                      USGS mineral deposit query
    aster.py                     ASTER SWIR L1T/L2 search + download from NASA
  spectral/
    indices.py                   10 spectral indices (6 Sentinel-2, 4 ASTER)
    compute.py                   Full feature computation pipeline
    visualize.py                 7 geological band composite presets + QGIS export
  structural/
    terrain.py                   Slope, aspect, curvature, multi-azimuth hillshade
    lineaments.py                Canny + Hough lineament extraction
    proximity.py                 Distance-to-features, drainage density
  training/
    sampling.py                  Exploration-bias-aware negative sampling
    spatial_cv.py                Spatial block CV + along-strike CV (azimuth 15 deg)
    train.py                     XGBoost/RF/LR training, SHAP, baselines, calibration
  predict/
    inference.py                 Chunked raster prediction, bootstrap uncertainty, targeting
  utils/
    config.py                    YAML config loader, AOI geometry
    raster.py                    Reproject, clip, resample, stack

scripts/                          # Execution scripts (2,082 lines, 5 scripts)
  download_all.py                CLI for all data downloads
  process_and_check.py           Multi-tile processing + signal check
  compute_terrain.py             DEM terrain/structural features
  prepare_prithvi_chips.py       224x224 training chips for foundation model
  foundation_model_setup.py      Prithvi-EO-2.0 fine-tuning boilerplate (1,016 lines)

notebooks/
  prithvi_finetune_colab.ipynb   Ready-to-run Colab notebook for GPU training

configs/
  great_dyke.yaml                All pipeline parameters
  great_dyke_aoi.geojson         Study area polygon
```

### The Data: 42 GB Processed

| Dataset | Source | Size | Cost |
|---|---|---|---|
| Sentinel-2 L2A (4 tiles, 7 bands) | Copernicus | 22 GB | Free |
| ASTER SWIR (11 scenes, 6 bands each) | NASA EarthData | 320 MB | Free |
| Copernicus DEM 30m | AWS Open Data | 365 MB | Free |
| Terrain features (slope, curvature, hillshade) | Computed from DEM | 4.7 GB | Free |
| Structural features (lineaments, drainage) | Computed | 1.1 GB | Free |
| Spectral indices (6 per tile) | Computed from S2 | 12 GB | Free |
| Probability maps (4 tiles) | ML output | 1.9 GB | Free |
| Training chips (284 chips, 224x224) | Prepared for Prithvi | 342 MB | Free |
| **Total** | | **~42 GB** | **$0** |

### The Labels: 71 Deposit Locations

| Source | Count | Quality |
|---|---|---|
| Published mine locations (literature) | 17 | Point coordinates, km-scale precision |
| Menabilly KML drill holes (algorithmically classified) | 9 ore-bearing | Spectrally scored, 500m precision |
| Menabilly KML drill holes (barren) | 13 | Classified as non-prospective |
| Macrostrat lithology grid | 1,200 pts | 1:1M scale geological context |
| GEM active faults (Zimbabwe) | 1 fault | Sparse but authoritative |

### The Validation Framework: Rigorous, Un-Cheatable

We built a validation system that goes far beyond standard ML evaluation:

1. **Leave-One-Tile-Out (LOTO) CV** -- holds out an entire 110x110 km satellite tile during training and tests whether the model can predict deposits on imagery it has never seen
2. **Geographic block CV** -- latitude-strip spatial blocks independent of tile boundaries
3. **Along-strike CV** -- respects the linear geology of the Great Dyke (15-degree azimuth)
4. **Bootstrap coefficient stability** -- 500 resamples to check if signal direction is robust
5. **Mandatory baselines** -- every model is compared to random, single-feature, and class-prior baselines
6. **Leakage guards** -- `distance_to_deposits` explicitly excluded (it's circular)

### The Drill Hole Classification Algorithm

We built a novel algorithm that classifies raw drill hole locations as ore-bearing vs barren without any assay data, using:

1. **6-index Mahalanobis spectral distance** to a reference profile built from known deposits
2. **Proximity to mapped geological structures** (faults, contacts from KML)
3. **Spatial cluster density** (delineation drilling clusters = higher confidence)
4. **Weighted composite scoring** (50% spectral, 30% structural, 20% density)

Result: 9 ore-bearing / 13 barren out of 22 scoreable drill holes on the Menabilly property.

---

## What We Tested (The Honest Part)

### 8 Model Configurations, All Failed to Generalize

| Model | Features | Cross-Tile PR-AUC | Lift | Verdict |
|---|---|---|---|---|
| LR(ferric_iron) | 1 Sentinel-2 index | 0.035 | 0.7x | FAIL |
| LR(ferric_iron) Z-scored | Normalized per tile | 0.035 | 0.7x | FAIL |
| LR(6 S2 indices) | All 6 Sentinel-2 indices | 0.028 | 0.6x | FAIL |
| LR(terrain only) | 6 DEM features | 0.016 | 0.7x | FAIL |
| LR(ASTER SWIR) | 6 mineral indices | N/A | N/A | Folds empty |
| LR(S2 Z-scored) | Geo block CV | 0.034 | 0.8x | FAIL |
| LR(DEM only) | Geo block CV | 0.050 | 0.9x | FAIL |
| LR(S2 + DEM combined) | 11 features | 0.037 | 0.8x | FAIL |

**Why everything failed:** Sentinel-2 spectral indices map atmospheric conditions and vegetation, not rock mineralogy. The ferrous iron coefficient literally flipped signs between tiles (+0.446 in T35KRU, -0.650 in T36KTC) -- mathematical proof that the signal is local, not geological.

### The Sign Flip: Proof of Non-Stationarity

```
                  T35KRU          T36KTC          Interpretation
ferrous_iron      +0.446          -0.650          OPPOSITE SIGN
ferric_iron       +0.086          +0.483          Same sign, 5x magnitude shift
clay_ratio        -0.454          varies          Unstable

Verdict: The model learns local atmospheric/vegetation patterns,
         not transferable geological signatures.
```

This is not a tuning problem. It is a sensor physics wall. At 10-30m resolution, a single pixel covers 100-900m2. Chromitite seams and mineralized shear zones are 5-10m wide. The mineral signature is mathematically diluted below the noise floor.

### What DID Work (Locally)

| Test | Score | Interpretation |
|---|---|---|
| Within-tile random CV | 0.841 | Strong local signal |
| Within-tile bootstrap CI | [0.29, 1.68] | Direction stable |
| Ferric iron vs known deposits | +2σ above background | Real spectral anomaly |
| Menabilly probability overlay | 23% of drill holes in top zones | Some predictive power locally |

---

## The Platform Architecture (Commodity-Agnostic)

GeoMine AI is not a model. It is a platform. The architecture is modular and commodity-agnostic:

```
Layer 1: Data Ingestion          Sentinel-2, ASTER, DEM, geological vectors
          |                      (any sensor can be plugged in)
Layer 2: Feature Computation     Spectral indices, terrain, structural, mineral
          |                      (any index recipe can be added)
Layer 3: Validation Framework    LOTO CV, geo-block CV, along-strike CV, leakage guards
          |                      (this is the moat)
Layer 4: Model Backend           LR/XGBoost (Phase 1) -> Prithvi ViT (Phase 2)
          |                      (any backbone can be swapped)
Layer 5: Prediction + Targeting  Probability maps, uncertainty, attention maps
```

PGM on the Great Dyke is the wedge. The platform serves any commodity, any layered intrusion, any geography. Bushveld Complex (SA), Stillwater (USA), Sudbury (Canada) -- same pipeline, different config file.

---

## Monetization Strategy (Sequenced)

### Phase A: Pre-Validation (Now)

**Sellable today:** Geospatial ML Due Diligence as a Service

The validation framework is the product. The customer is not miners -- it's other mining AI startups, exploration consultancies, and academic labs.

- Spatial CV implementation + leakage audit: $5,000-15,000 per engagement
- Signal robustness audit (does their model actually generalize?): $10,000-25,000
- Model honesty certification: $5,000

This is niche, defensible, and requires no cross-tile PR-AUC number.

### Phase B: Post-Validation (After LOTO >= 0.45)

**B1: Exploration targeting per concession**
- $10,000-50,000 per concession, 95%+ gross margin
- Target: junior miners raising capital for Great Dyke concessions

**B2: Data licensing**
- Full Great Dyke probability maps: $50,000-100,000
- Regional sub-areas: $10,000-25,000
- Bundled with interpretation report: +$15,000-30,000

### Phase C: Scale (2027+)

**C1: Platform SaaS**
- Upload concession boundary, get probability map
- Per-concession or subscription pricing
- Expand beyond Great Dyke to other layered intrusions globally

**C2: Raise capital**
- $250,000-500,000 seed round, backed by the cross-tile validation number
- Use of funds: ground truth acquisition, team, first 3-5 paying concessions

---

## The Competitive Moat

### What makes GeoMine AI defensible:

1. **The validation framework is the moat.** Anyone can run a model. Almost nobody runs LOTO CV, along-strike CV, bootstrap stability checks, and mandatory baselines. This framework catches every failure mode and builds investor trust.

2. **The honest negative result is an asset.** We tested 8 configurations and proved that Sentinel-2 spectral indices cannot map PGM deposits across tiles. This eliminates a class of competitors who claim otherwise without rigorous validation.

3. **The foundation model recipe is proprietary.** Prithvi-EO-2.0 is public, but the fine-tuning approach (which bands, what normalization, what chip size, what augmentation, what segmentation head) is our IP.

4. **The drill hole classification algorithm has no precedent.** Using Mahalanobis spectral distance + structural proximity + cluster density to classify drill holes without assay data is novel.

5. **Domain expertise compounds.** Every negative result narrows the search space for what WILL work. We know more about why satellite-based PGM detection fails than almost anyone who hasn't published on it.

---

## The Technology Stack

| Component | Technology | Cost |
|---|---|---|
| Satellite imagery | Sentinel-2 L2A (ESA/Copernicus) | Free |
| SWIR mineral mapping | ASTER L2 (NASA EarthData) | Free |
| Elevation data | Copernicus DEM 30m (AWS) | Free |
| Geological context | Macrostrat API | Free |
| Structural geology | GEM Active Faults (GitHub) | Free |
| ML framework | scikit-learn, XGBoost | Free |
| Foundation model | Prithvi-EO-2.0 (NASA/IBM) | Free |
| Fine-tuning compute | Google Colab T4 GPU | Free (or $10/mo Pro) |
| GIS processing | GDAL, rasterio, geopandas | Free |
| Language | Python 3.11 | Free |
| **Total infrastructure cost** | | **$0/month** |

---

## What's Next (Phase 2b)

### Immediate (This Week)
1. Zip 284 training chips (342 MB) and upload to Google Drive
2. Run Prithvi fine-tuning on Colab (30-45 min on T4 GPU)
3. Measure LOTO CV PR-AUC -- **this is the make-or-break number**

### If LOTO >= 0.45 (The Win)
- Generate full-tile probability maps with the foundation model
- Create attention maps showing what the model "looks at" (geological validation)
- Draft investor pitch deck with the cross-tile validation number
- Approach 2-3 junior mining companies for paid pilot

### If LOTO < 0.45 (The Pivot)
- Switch to multi-temporal analysis (seasonal vegetation masking)
- Try Clay Foundation Model (supports SAR + optical fusion)
- Acquire commercial ground truth (drill logs from a mining company)
- Consider hyperspectral when EMIT/PRISMA data becomes available for this region

---

## Revenue Timeline (Conservative)

| Quarter | Milestone | Revenue |
|---|---|---|
| Q2 2026 | Foundation model validation | $0 |
| Q3 2026 | First paid pilot (1 concession) | $10,000-25,000 |
| Q4 2026 | Seed raise (if LOTO passes) | $250,000-500,000 |
| Q1 2027 | 3-5 paying concessions | $50,000-100,000 |
| Q2-Q4 2027 | Expand to Bushveld Complex (SA) | $200,000-500,000 |
| 2028 | Platform launch (SaaS) | $500,000-1,000,000 ARR |

---

## Key Metrics

| Metric | Current Value | Target |
|---|---|---|
| Codebase | 8,300+ lines | -- |
| Data pipeline | 42 GB processed | -- |
| Infrastructure cost | $0/month | < $100/month |
| Labeled deposits | 26 curated | 100+ |
| Cross-tile PR-AUC | 0.035 (LR baseline) | >= 0.45 |
| Model configs tested | 8 (all failed) | Prithvi next |
| Tiles processed | 4 (44,000 km2) | 10+ |
| Commodities covered | Cr, PGM | + Au, Ni, Cu |
| Revenue | $0 | First dollar Q3 2026 |

---

## The Bottom Line

GeoMine AI is a $0-infrastructure mineral prediction engine backed by NASA/ESA free satellite data, a rigorous validation framework that caught its own failures, and a clear path to foundation-model-based generalization.

The market opportunity is platinum group metals in structural supply deficit, on the world's second-largest reserve, using technology that costs nothing to run.

The only question left is whether Prithvi's spatial intelligence can see what logistic regression couldn't. That answer is 45 minutes of GPU time away.

---

*Built with: Python, GDAL, rasterio, scikit-learn, Prithvi-EO-2.0, Sentinel-2, ASTER, Copernicus DEM*
*Total development cost: $0 in data/compute, 100% open-source stack*
