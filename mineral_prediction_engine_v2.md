# GeoMine AI -- Satellite-Powered Mineral Prediction Engine
> *A cost-efficient, open-source-first platform for mineral exploration using remote sensing, machine learning, and geospatial intelligence.*

---

## Table of Contents
1. [Project Vision](#1-project-vision)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Data Sources -- Free Satellite & Geospatial Data](#3-data-sources--free-satellite--geospatial-data)
4. [Core Modules](#4-core-modules)
5. [Technology Stack (Open Source First)](#5-technology-stack-open-source-first)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Data Flow Diagram](#7-data-flow-diagram)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Phased Roadmap](#9-phased-roadmap)
10. [Cost Breakdown vs Traditional Exploration](#10-cost-breakdown-vs-traditional-exploration)
11. [Key Risks & Mitigations](#11-key-risks--mitigations)
12. [Team & Roles Required](#12-team--roles-required)
13. [Open Source Licensing Strategy](#13-open-source-licensing-strategy)
14. [Next Steps (Start Tomorrow)](#14-next-steps-start-tomorrow)
15. [Weak Point Analysis & Design Decisions](#15-weak-point-analysis--design-decisions)

---

## 1. Project Vision

Traditional mineral exploration is expensive -- drilling campaigns, geophysical surveys, and field teams can cost **$2M--$50M+** before a single significant deposit is confirmed.

**GeoMine AI** cuts this cost by up to **80%** in the early-stage prospecting phase by:
- Using **free satellite imagery** (multispectral, hyperspectral, radar) to detect surface mineralogy indicators
- Applying **machine learning** to correlate remote sensing data with known deposit signatures
- Ranking exploration targets by **probability score** before any ground-truthing begins
- Leveraging **100% open-source tools** to eliminate software licensing costs

**Target Users:**
- Junior mining companies with limited exploration budgets
- National geological surveys (Zimbabwe, DRC, Zambia, Mozambique, Tanzania)
- Academic geology departments conducting research
- Government resource ministries needing resource inventories
- Environmental consultancies assessing mining impact zones

---

## 2. System Architecture Overview

```
+---------------------------------------------------------------------------+
|                          GeoMine AI Platform                              |
|                                                                           |
|  +----------------+    +----------------+    +------------------------+   |
|  |  DATA          |    |  PROCESSING    |    |  PREDICTION ENGINE     |   |
|  |  INGESTION     |--->|  PIPELINE      |--->|  & SCORING             |   |
|  |  LAYER         |    |  LAYER         |    |                        |   |
|  +----------------+    +----------------+    +------------------------+   |
|         |                     |                        |                  |
|         v                     v                        v                  |
|  +----------------+    +----------------+    +------------------------+   |
|  |  STAC Catalog  |    |  Feature       |    |  Mineral Probability   |   |
|  |  + Satellite   |    |  Extraction    |    |  Heat Maps             |   |
|  |  APIs          |    |  (Spectral +   |    |  + Uncertainty Maps    |   |
|  |  (Copernicus   |    |  Hyperspectral |    |  + Target Reports      |   |
|  |  Data Space,   |    |  Unmixing +    |    |                        |   |
|  |  NASA, USGS)   |    |  Geology ML)   |    |                        |   |
|  +----------------+    +----------------+    +------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |   SPECTRAL REFERENCE ENGINE (USGS splib07 + ECOSTRESS + MINDAT)   |   |
|  +-------------------------------------------------------------------+   |
|                                                                           |
|  +-------------------------------------------------------------------+   |
|  |          WEB DASHBOARD (Map + Reports + Spectral Viewer)          |   |
|  +-------------------------------------------------------------------+   |
+---------------------------------------------------------------------------+
```

**Key architectural change from v1.0:** Added a dedicated Spectral Reference Engine layer. Mineral identification without spectral library matching is like pattern recognition without a dictionary -- the model can learn correlations but cannot physically validate what mineral it's detecting. This layer bridges physics-based remote sensing with data-driven ML.

---

## 3. Data Sources -- Free Satellite & Geospatial Data

### 3.1 Primary Satellite Sources

| Satellite / Dataset | Provider | Resolution | Key Use | Access |
|---|---|---|---|---|
| **Sentinel-2 MSI** | ESA Copernicus | 10--60m | Multispectral mineralogy mapping | Free via Copernicus Data Space Ecosystem |
| **Sentinel-1 SAR** | ESA Copernicus | 5--20m | Structural geology, fault mapping | Free via Copernicus Data Space Ecosystem |
| **Landsat 8/9 OLI** | USGS | 15--30m | Iron oxide, clay, carbonate mapping | Free via USGS EarthExplorer |
| **ASTER** | NASA/METI | 15--90m | Hydrothermal alteration mapping (best for minerals) | Free via NASA EarthData |
| **SRTM / NASADEM** | NASA | 30m | Digital Elevation Model, terrain analysis | Free via NASA EarthData |
| **ALOS PALSAR** | JAXA | 10--100m | SAR for structural lineaments | Free via JAXA G-Portal |
| **PRISMA** | ASI (Italy) | 30m | Hyperspectral (advanced mineral discrimination) | Free registration |
| **EMIT** | NASA/JPL (ISS) | 60m | Imaging spectrometer -- 285 bands, purpose-built for mineral mapping | Free via NASA EarthData |
| **EnMAP** | DLR (Germany) | 30m | Hyperspectral (242 bands), excellent SWIR coverage for clays/carbonates | Free via EnMAP portal |
| **Harmonized Landsat Sentinel (HLS)** | NASA | 30m | Consistent multi-sensor time series for change detection | Free via NASA EarthData |

> **Note on Copernicus:** The old Copernicus Open Access Hub (scihub.copernicus.eu) was decommissioned in October 2023. All Sentinel data is now accessed through the **Copernicus Data Space Ecosystem** (dataspace.copernicus.eu). The `sentinelsat` Python library is deprecated -- use `cdsetool` or the OData/STAC APIs instead.

### 3.2 Supporting Ground Truth & Geological Databases

| Dataset | Source | Use |
|---|---|---|
| Global Mineral Deposit Database | USGS MRDS | Training data for known deposits |
| CGMW Geological Maps | Commission for the Geological Map of the World | Base geology layers |
| Africa Mining Atlas | Various national surveys | Africa-specific deposit locations |
| **MINDAT** | mindat.org (API available) | World's largest mineral occurrence database -- 400k+ localities |
| **Macrostrat** | macrostrat.org (REST API) | Geological maps and stratigraphic columns with API access |
| **GEOROC** | georoc.eu | Geochemistry of rocks -- major/trace element and isotope data |
| **EarthChem** | earthchem.org | Geochemical data with REST API |
| OpenStreetMap | OSM | Infrastructure, access roads |
| GEBCO Bathymetry | GEBCO | Offshore extension |
| World Magnetic Model | NOAA | Geomagnetic anomaly reference |
| **USGS Spectral Library v7** | USGS | ~2500 reference spectra for spectral matching (SAM/SID) |
| **ECOSTRESS Spectral Library** | NASA/JPL | Comprehensive mineral, rock, and soil spectra |

### 3.3 Data Catalog Strategy

All downloaded satellite data should be indexed using the **STAC (SpatioTemporal Asset Catalog)** standard. This provides:
- Standardized metadata across all satellite sources
- Programmatic search and access via `pystac-client`
- Compatibility with Microsoft Planetary Computer, Element84, and other STAC endpoints
- Future-proofing if any single satellite API changes or shuts down

**Tools:** `pystac`, `pystac-client`, `stackstac`, `stactools`

---

## 4. Core Modules

### Module 1: Data Ingestion & Preprocessing
**Purpose:** Automatically download, clip, and preprocess satellite imagery for any Area of Interest (AOI).

**Functions:**
- AOI definition (draw polygon on map or upload GeoJSON/shapefile)
- Automated scene search via STAC catalog and Copernicus Data Space APIs
- Cloud masking and atmospheric correction (Sen2Cor for Sentinel-2)
- Image mosaicking and temporal compositing (cloud-free best-pixel)
- Radiometric calibration to surface reflectance
- Multi-resolution data fusion (resampling 20m SWIR bands to 10m using super-resolution or bicubic)
- Storage as Cloud Optimized GeoTIFFs (COG) for efficient streaming access

**Key Tools:** `cdsetool`, `pystac-client`, `rasterio`, `GDAL`, `SNAP (ESA)`, `rioxarray`, `Dask`

> **Why COG?** Cloud Optimized GeoTIFFs allow range-request access -- the dashboard can stream just the visible tiles instead of loading entire rasters. This is critical for performance at scale.

---

### Module 2: Spectral Feature Extraction
**Purpose:** Extract mineralogical and geological signals from satellite bands.

**Key Spectral Indices to Compute:**

| Index | Formula | Target Minerals |
|---|---|---|
| **Clay Ratio** | SWIR1/SWIR2 (Bands 11/12) | Kaolinite, illite, smectite |
| **Iron Oxide Ratio** | Red/Blue (Bands 4/2) | Hematite, goethite |
| **Ferric Iron** | (B8A - B4) / (B8A + B4) | Fe3+ anomalies |
| **Carbonate Index** | B13/B14 (ASTER TIR) | Limestone, dolomite, skarns |
| **Al-OH Minerals** | B5/B7 (ASTER SWIR) | Alunite, kaolinite (Au pathfinder) |
| **Mg-OH Minerals** | B7/B8 (ASTER SWIR) | Chlorite, serpentine (Ni, Cr) |
| **Silica Index** | B13/B10 (ASTER TIR) | Quartz-rich zones |
| **NDVI** | (NIR - Red)/(NIR + Red) | Vegetation cover (geobotany anomalies) |
| **Lineament Density** | Edge detection on DEM/SAR | Fault/fracture hosting |

**Hyperspectral Mineral Mapping (EMIT, PRISMA, EnMAP):**
- Spectral Angle Mapper (SAM) -- match pixel spectra against USGS reference library
- Spectral Information Divergence (SID) -- probabilistic spectral matching
- Linear Spectral Unmixing -- estimate sub-pixel mineral abundances
- Endmember extraction (N-FINDR, VCA, ATGP algorithms)
- Principal Component Analysis (PCA) / Minimum Noise Fraction (MNF) for dimensionality reduction
- Continuum removal for absorption feature analysis

**Tools:** `NumPy`, `rasterio`, `scikit-image`, `Spectral Python (SPy)`, `PySptools`, `HyperSpy`, `SNAP Graph Builder`

> **Why hyperspectral matters:** Sentinel-2 has only 13 bands -- enough to detect broad mineral groups (clays vs iron oxides) but not to distinguish kaolinite from montmorillonite, or alunite from illite. EMIT's 285 bands can discriminate individual mineral species, which is the difference between "there might be alteration here" and "this is an epithermal gold pathfinder signature."

---

### Module 3: Structural Geology Analysis
**Purpose:** Map faults, lineaments, and structural traps that host ore deposits.

**Functions:**
- DEM-based hillshade analysis (multi-azimuth: 0, 45, 90, 135, 180, 225, 270, 315 degrees)
- Automated lineament extraction (Canny edge detection + Probabilistic Hough Transform)
- Lineament density mapping (kernel density estimation)
- Fault intersection analysis (structural trap identification)
- Drainage anomaly detection (structural control on drainage = mineralization indicator)
- Rose diagram generation for lineament orientation analysis
- Curvature analysis (plan + profile curvature from DEM)

**Tools:** `QGIS`, `WhiteboxTools`, `OpenCV` (edge detection), `GeoPandas`, `Shapely`, `PyGMT` (publication-quality maps)

---

### Module 4: Machine Learning Prediction Engine
**Purpose:** The core AI that scores every pixel/polygon for mineral potential.

**Approach:** Hybrid supervised + physics-informed learning using known deposits as positive training samples.

**Input Features per Training Sample:**
- All spectral indices (Module 2 outputs)
- Hyperspectral mineral abundances from unmixing (where available)
- Structural proximity scores (Module 3 outputs)
- Elevation, slope, aspect, curvature (terrain)
- Lithology type (from geological maps via Macrostrat API)
- Magnetic anomaly value (if available)
- Distance to nearest known deposit (proximity feature)
- Distance to major faults and fault intersections
- Lineament density within 1km, 5km, 10km buffers
- Soil/stream geochemistry (from GEOROC/EarthChem where available)
- Drainage density and anomalous drainage patterns

**Training Data Strategy:**

The biggest methodological risk in mineral prospectivity mapping is **how you sample negatives.** Random barren sampling introduces spatial bias. Instead:

1. **Positive samples:** Known mineral deposits from USGS MRDS + national survey databases + MINDAT localities. Buffer by deposit footprint size.
2. **Negative samples:** Use a stratified approach:
   - Sample from each lithology type proportionally
   - Exclude buffer zones around known deposits (2--5km depending on deposit type)
   - Ensure negatives cover the full range of spectral/terrain feature space
   - Include "difficult negatives" -- locations with high spectral alteration but no known deposit (reduces false positives)
3. **Class imbalance handling:** Mineral deposits are rare events. Use:
   - SMOTE (Synthetic Minority Oversampling) or ADASYN for tree models
   - Weighted loss functions for deep learning models
   - Precision-Recall curves instead of ROC alone (ROC can be misleading with severe imbalance)

**Spatial Cross-Validation:**

Standard k-fold cross-validation is invalid for geospatial data because nearby points are autocorrelated. A model that memorizes "deposits near the Great Dyke" will score well on random splits but fail elsewhere.

Use **Spatial Block Cross-Validation:**
- Divide AOI into spatial blocks (10--50km depending on geology)
- Ensure train/test splits have no spatial overlap
- Report performance per geological terrane, not just globally

**Models to Train:**

| Model | Use Case | Library |
|---|---|---|
| Random Forest | Baseline classifier, feature importance ranking | `scikit-learn` |
| XGBoost | Primary tabular model, handles missing data natively | `xgboost` |
| U-Net (spatial CNN) | Pixel-level spatial pattern recognition on imagery | `TorchGeo` / `segmentation_models_pytorch` |
| Vision Transformer | Fine-tuned geospatial foundation model | `TorchGeo` + `Prithvi` or `Clay` |
| Gaussian Process | Uncertainty quantification on final predictions | `scikit-learn` / `GPyTorch` |
| **Ensemble** | Weighted average of above models | Custom |

**Geospatial Foundation Models:**

Recent open-source foundation models pre-trained on massive satellite datasets can be fine-tuned for mineral mapping with relatively little labeled data:

- **Prithvi** (IBM/NASA) -- pre-trained on Harmonized Landsat-Sentinel data. Fine-tune for mineral alteration detection.
- **Clay Foundation Model** -- pre-trained on diverse EO data. Strong for feature extraction.

These models have already learned general remote sensing features (vegetation patterns, soil types, terrain). Fine-tuning for mineral mapping requires only hundreds of labeled samples instead of tens of thousands.

**Output:**
- Probability score (0--100%) per grid cell for each mineral target
- **Uncertainty map** (epistemic + aleatoric) per prediction -- critical for clients to gauge confidence
- Feature importance map showing which inputs drove each prediction (SHAP values)
- Anomaly score for "novel" signatures not seen in training data

---

### Module 5: Visualization & Reporting Dashboard
**Purpose:** Present results as interactive maps and downloadable reports for geologists.

**Features:**
- Interactive web map with probability heat maps overlaid on satellite imagery (COG streaming)
- **Uncertainty overlay toggle** -- show confidence alongside predictions
- Target polygon ranking table (top N targets with scores, areas, and access info)
- Spectral signature viewer (click any point to see full spectrum + reference library match)
- **Mineral deposit type classification** (not just "high probability" but "likely porphyry Cu" vs "likely orogenic Au")
- Exportable PDF reports per target area
- GeoJSON / Shapefile / GeoPackage export for use in QGIS
- Historical analysis timeline (multi-year change detection)
- SHAP feature importance viewer per target (why did the model flag this area?)

**Tools:** `Leaflet.js` / `Deck.gl`, `FastAPI` (backend), `React` (frontend), `TiTiler` (COG tile serving), `Folium` (Python maps), `WeasyPrint` (PDF)

> **Why TiTiler over GeoServer?** TiTiler is a lightweight, Python-native COG tile server that integrates naturally with the FastAPI backend. GeoServer is Java-based, heavier to deploy, and overkill for serving COG rasters. TiTiler also supports dynamic band math and rescaling, which is perfect for on-the-fly spectral index visualization.

---

### Module 6: Ground Truth Feedback Loop
**Purpose:** Allow field geologists to feed back real results to continuously improve the model.

**Functions:**
- Upload field sampling results (GPS point + assay data + photos)
- Mark predictions as confirmed/refuted with geological notes
- Trigger model retraining with new data (automated via Airflow)
- Model version control and performance tracking
- A/B comparison between model versions on same AOI
- **Active learning:** System suggests which locations to sample next for maximum model improvement

**Tools:** `MLflow`, `DVC`, `PostgreSQL + PostGIS`, `Label Studio` (optional, for image annotation)

---

### Module 7: Geophysical Data Integration (Phase 2+)
**Purpose:** Incorporate gravity, magnetic, and electromagnetic data to see beneath surface cover.

Remote sensing is limited to surface expression. In areas with thick regolith, alluvial cover, or dense vegetation, the mineral signature may be invisible from satellite data alone. Geophysical data "sees" subsurface structures.

**Functions:**
- Import airborne magnetic survey data (if available from national surveys)
- Compute magnetic derivatives (total horizontal gradient, analytic signal)
- Import Bouguer gravity data
- Gridding and interpolation of irregularly sampled geophysical data
- Merge geophysical features into the ML feature matrix

**Tools:** `Fatiando a Terra` (verde for gridding, harmonica for gravity/magnetics), `SimPEG` (forward modeling/inversion), `PyGMT` (visualization)

> **Data availability:** Many African countries have national airborne geophysical surveys (e.g., Zimbabwe's airborne magnetic survey). These are often available free or at low cost from geological surveys. They dramatically improve predictions in covered terranes.

---

### Module 8: 3D Geological Modeling (Phase 3+)
**Purpose:** Build implicit 3D geological models to understand subsurface deposit geometry.

**Functions:**
- Build 3D lithological models from surface geology + drill hole data
- Implicit surface modeling (no manual wireframing)
- Uncertainty quantification on geological boundaries
- Export block models for resource estimation

**Tools:** `GemPy` (implicit 3D modeling), `LoopStructural` (structural geology modeling), `PyVista` (3D visualization), `OMF` (Open Mining Format for data interchange)

---

## 5. Technology Stack (Open Source First)

### Backend & Processing
```
Language:           Python 3.11+
Geospatial Core:    GDAL, Rasterio, Fiona, rioxarray
Vector Analysis:    GeoPandas, Shapely, PyProj
Data Cubes:         Xarray + Dask (out-of-core processing for large rasters)
Satellite Access:   pystac-client, cdsetool, EODAG, Google Earth Engine Python API
Hyperspectral:      Spectral Python (SPy), PySptools, HyperSpy
Image Processing:   OpenCV, scikit-image, NumPy, SciPy
Machine Learning:   scikit-learn, XGBoost, TorchGeo, segmentation_models_pytorch
Foundation Models:  Prithvi (IBM/NASA), Clay Foundation Model, TerraTorch
Explainability:     SHAP, Captum (PyTorch)
Geophysics:         Fatiando a Terra (verde, harmonica), SimPEG
3D Modeling:        GemPy, LoopStructural, PyVista
Workflow Engine:    Apache Airflow (pipeline orchestration)
Database:           PostgreSQL + PostGIS (spatial queries)
Object Storage:     MinIO (S3-compatible, self-hosted)
API Framework:      FastAPI
Data Catalog:       STAC (pystac, stactools)
```

### Frontend & Visualization
```
Map Engine:         Leaflet.js + GeoTIFF.js (COG support)
Tile Server:        TiTiler (dynamic COG tile serving)
UI Framework:       React + TypeScript
Charting:           Recharts / Plotly.js
3D Visualization:   Deck.gl (for 3D terrain overlays)
Report Generation:  WeasyPrint (PDF from HTML)
Desktop GIS:        QGIS + EnMAP-Box plugin (hyperspectral workflows)
```

### Infrastructure
```
Containerization:   Docker + Docker Compose
Orchestration:      Kubernetes (production) / Docker Compose (dev)
CI/CD:              GitHub Actions
Monitoring:         Grafana + Prometheus
Cloud (optional):   AWS / Azure / GCP -- or fully on-premise
Satellite Proc.:    Google Earth Engine (free) or OpenDataCube (self-hosted)
```

### GIS & Remote Sensing Tools (Open Source Desktop)
```
QGIS             -- Desktop GIS + EnMAP-Box plugin for hyperspectral
SNAP (ESA)       -- Sentinel satellite data processing
SAGA GIS         -- Terrain analysis, geomorphometry
WhiteboxTools    -- DEM analysis, lineament extraction
ORFEO Toolbox    -- Remote sensing image processing
EnMAP-Box        -- QGIS plugin for imaging spectroscopy / mineral mapping
```

---

## 6. Machine Learning Pipeline

```
RAW SATELLITE DATA (Sentinel-2, ASTER, EMIT, SRTM, SAR)
        |
        v
+-------------------+
|  Preprocessing    |  <-- Cloud masking, atmospheric correction, reprojection
|                   |      Store as COG in MinIO, index in STAC catalog
+--------+----------+
         |
         v
+-------------------+
| Feature           |  <-- Spectral indices, hyperspectral unmixing,
| Engineering       |      terrain metrics, structural proximity,
|                   |      geophysical derivatives (if available)
+--------+----------+
         |
         v
+-------------------+
| Training Data     |  <-- Known deposits (positive) + stratified barren (negative)
| Preparation       |      Spatial block split: 70% train / 15% val / 15% test
|                   |      SMOTE/ADASYN for class imbalance
+--------+----------+
         |
         v
+-------------------+
| Model Training    |  <-- XGBoost (tabular) + U-Net (spatial) + Foundation Model
| & Evaluation      |      Spatial cross-validation
|                   |      Metrics: PR-AUC, Spatial AUC-ROC, F1, top-k recall
+--------+----------+
         |
         v
+-------------------+
| Ensemble &        |  <-- Weighted model combination
| Uncertainty       |      Epistemic uncertainty via MC Dropout / GP
+--------+----------+
         |
         v
+-------------------+
| Prediction &      |  <-- Run on new AOI, generate probability + uncertainty rasters
| Scoring           |      SHAP feature importance per prediction
+--------+----------+
         |
         v
+-------------------+
| Post-processing   |  <-- Threshold, cluster targets, rank, generate polygons
|                   |      Filter by minimum area, merge adjacent clusters
+--------+----------+
         |
         v
+-------------------+
| Report Output     |  <-- Heat maps, uncertainty maps, ranked target list, PDF
+-------------------+
         |
         v (Field verification)
+-------------------+
| Feedback Loop     |  <-- Field results --> retrain --> active learning
+-------------------+
```

---

## 7. Data Flow Diagram

```
[User defines AOI on web map]
        |
        v
[System searches STAC catalogs + Copernicus/USGS/NASA APIs]
        |
        v
[Downloads Sentinel-2, ASTER, EMIT, SRTM tiles --> COG storage]
        |
        +--------------------------------------------+
        v                                            v
[Spectral Processing]                    [Terrain Processing]
 - Band ratios (multispectral)            - Slope, aspect, curvature
 - SAM/SID mineral matching               - Lineament extraction
 - Spectral unmixing (hyperspectral)       - Fault proximity
 - NDVI geobotany anomalies               - Drainage analysis
        |                                            |
        +-------------------+------------------------+
                            v
                   [Feature Matrix Assembly]
                    (per 30m grid cell, ~40+ features)
                            |
                            v
                   [ML Ensemble Inference]
                    (XGBoost + U-Net + Foundation Model)
                            |
                            v
                   [Probability + Uncertainty Rasters]
                    0-100% per pixel, with confidence bands
                            |
                            v
                   [Target Clustering & Ranking]
                    (contiguous high-prob, low-uncertainty zones)
                            |
                            v
                   [Dashboard + Reports + SHAP Explanations]
```

---

## 8. Deployment Architecture

### Option A: Cloud (Low Cost Start)
```
Google Earth Engine (free)           <-- Satellite data processing
        +
AWS EC2 g4dn.xlarge (~$380/mo)      <-- GPU for ML training + inference
        +
AWS RDS PostgreSQL (~$50/mo)         <-- PostGIS database
        +
AWS S3 (~$20/mo)                     <-- COG raster storage
        +
Cloudflare (free tier)               <-- CDN + DNS
--------------------------------------------------------------
Total estimated cloud cost:  ~$450/month (realistic with GPU)
```

> **Note:** The original $220/mo estimate used a t3.xlarge (CPU only). ML training and inference on satellite imagery requires GPU. A g4dn.xlarge with T4 GPU is the minimum viable option. For Phase 1 prototyping, you can use spot instances (~$115/mo) or Google Colab Pro ($10/mo) to reduce costs.

### Option B: On-Premise (Zero Cloud Cost)
```
Workstation (RTX 4090 GPU, 64GB RAM) <-- ML training
        +
NAS Storage (8TB+)                    <-- Satellite data archive (grows fast)
        +
Local PostgreSQL + PostGIS            <-- Database
        +
Nginx + FastAPI + TiTiler             <-- Internal web + tile server
--------------------------------------------------------------
Hardware cost: ~$4,000-$6,000 one-time, then ~$30/month (electricity)
```

> **Storage warning:** A single Sentinel-2 tile is ~800MB. ASTER scenes are ~4GB. A regional study (e.g., Great Dyke) with multi-year temporal compositing can easily consume 500GB--2TB. Budget 8TB+ storage, not 4TB.

### Option C: Hybrid (Recommended)
```
Google Earth Engine         <-- Heavy preprocessing (free, handles petabytes)
        +
On-premise GPU              <-- ML training (one-time hardware cost)
        +
Hetzner/OVH VPS ($20/mo)   <-- Web dashboard (European hosting, better for African clients)
        +
MinIO on-premise            <-- COG raster archive
--------------------------------------------------------------
Ongoing cost: ~$20-50/month after hardware purchase
```

---

## 9. Phased Roadmap

### Phase 1: Proof of Concept (Months 1--3)
**Goal:** Demonstrate that satellite data can discriminate known deposits from barren ground in a single well-studied region.

- [ ] Set up development environment (Docker, Python, GDAL, conda)
- [ ] Register for Copernicus Data Space, NASA EarthData, USGS EarthExplorer
- [ ] Download Sentinel-2 and ASTER data for 1 known mining region (Great Dyke, Zimbabwe)
- [ ] Compute all spectral indices (clay ratio, iron oxide, ferric iron, Al-OH, Mg-OH)
- [ ] Download USGS MRDS + MINDAT data for training labels in that region
- [ ] Implement stratified negative sampling strategy (not random barren)
- [ ] Train Random Forest + XGBoost classifiers with **spatial block cross-validation**
- [ ] Compute SHAP feature importances to validate geological plausibility
- [ ] Produce first probability heat map + uncertainty map
- [ ] Validate against known deposit locations AND known barren zones
- [ ] **Success metric:** PR-AUC > 0.60 AND spatial AUC-ROC > 0.70 on spatially held-out test blocks
- [ ] **Geological validation:** Top-20 predicted targets include >50% of known deposits in test region

**Team needed:** 1 geologist + 1 data scientist (can be same person with both skills)

> **Why PR-AUC > 0.60 instead of ROC-AUC > 0.75?** Mineral deposits are rare (~0.1% of area). ROC-AUC can be misleadingly high even with poor precision because it's dominated by true negatives. PR-AUC directly measures the precision-recall tradeoff that matters: "of the areas I flag, how many are real?"

---

### Phase 2: Full Pipeline + Hyperspectral (Months 4--6)
**Goal:** Automate the full pipeline, add structural geology and hyperspectral capability.

- [ ] Build Apache Airflow pipeline (auto-download -> preprocess -> predict)
- [ ] Integrate DEM lineament extraction (WhiteboxTools + OpenCV Hough Transform)
- [ ] Add EMIT/PRISMA hyperspectral processing (SPy + PySptools for SAM/unmixing)
- [ ] Integrate USGS Spectral Library for automated mineral matching
- [ ] Fine-tune Prithvi or Clay foundation model on mineral alteration detection
- [ ] Build ensemble (XGBoost + U-Net + foundation model)
- [ ] Implement uncertainty quantification (MC Dropout + Gaussian Process)
- [ ] Add PostGIS database + STAC catalog for results storage
- [ ] Build FastAPI backend + TiTiler for COG tile serving
- [ ] Build Leaflet.js map dashboard (probability maps, uncertainty toggle, spectral viewer)
- [ ] Test on 3 additional AOIs in different geological settings (e.g., Zambian Copperbelt, Limpopo Belt, Kaapvaal Craton)
- [ ] **Success metric:** System runs end-to-end with <2 hours of human input per new AOI

**Team needed:** +1 backend developer +1 frontend developer

---

### Phase 3: Product Release (Months 7--12)
**Goal:** Launch as a usable product for external clients.

- [ ] Add user authentication, project management, and multi-tenancy
- [ ] Build PDF report generator with SHAP explanations per target
- [ ] QGIS plugin for direct integration (+ EnMAP-Box workflows)
- [ ] Ground truth feedback loop (field data upload -> model retraining -> active learning)
- [ ] Add more mineral targets (extend beyond initial: Au, Cu, Li, Cr, PGMs, Ni, REE)
- [ ] Integrate geophysical data module (Fatiando for magnetics/gravity)
- [ ] Security audit and penetration testing
- [ ] Beta release with 3--5 pilot clients
- [ ] **Success metric:** First paying client, model validated by drill results

---

### Phase 4: Scale & Expand (Year 2+)
- [ ] Drone LiDAR integration for high-resolution ground surveys (PDAL for point cloud processing)
- [ ] 3D geological modeling (GemPy) integrated into dashboard
- [ ] Automated anomaly alerting (new satellite pass triggers new prediction)
- [ ] Mobile field app for geologists (React Native + offline maps)
- [ ] Partnership with national geological surveys (Zimbabwe, Zambia, DRC, Mozambique)
- [ ] Expand training database to global coverage
- [ ] InSAR ground deformation monitoring for active mines (MintPy)

---

## 10. Cost Breakdown vs Traditional Exploration

| Activity | Traditional Cost | GeoMine AI Cost | Savings |
|---|---|---|---|
| Regional reconnaissance (50,000 km2) | $500,000--$2M | $5,000--$20,000 | **95--99%** |
| Target identification | $200,000--$500,000 | $2,000--$10,000 | **95--98%** |
| Desk study & data purchase | $50,000--$200,000 | ~$0 (open data) | **100%** |
| Geophysical survey | $1M--$5M | Partially replaced | **50--70%** |
| **First-pass prospecting total** | **$1.75M--$7.7M** | **$7,000--$30,000** | **~97%** |

> **Honest caveat:** GeoMine AI does NOT replace drilling, and the 97% savings applies only to the first-pass prospecting phase. Total exploration cost (including drilling, feasibility studies, environmental assessment) is typically $10M--$50M. GeoMine AI reduces the number of targets that need to be drilled, potentially saving 40--60% on drilling costs by improving target quality. A more conservative total savings estimate is **30--50% of total pre-feasibility exploration cost.**

---

## 11. Key Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| Cloud cover obscures satellite imagery | High | High (tropical Africa) | Multi-temporal compositing; use SAR (cloud-penetrating); prioritize dry season imagery |
| Insufficient training data for rare deposits | High | High | Data augmentation; transfer learning from analogous deposit types; use Prithvi foundation model pre-training |
| Model overfits to training region geology | High | High | **Spatial block cross-validation**; train across multiple geological terranes; test on geologically distinct holdout regions |
| Copernicus/NASA APIs change or become restricted | Medium | Low | STAC-based data catalog with multiple redundant sources; local data archive |
| False positives waste client drilling budget | High | Medium | **Uncertainty maps** alongside predictions; mandate ground truth before drilling; report precision at client's chosen threshold |
| Predictions fail in covered terranes (regolith, vegetation) | High | High (tropical) | Integrate SAR (structure under cover) + geophysical data (Phase 2); clearly communicate coverage limitations per AOI |
| Regulatory / data sovereignty issues | Medium | Medium | Deploy on-premise option for government clients; use region-specific open data; comply with local mining laws |
| Negative sampling bias distorts model | High | High (if not handled) | Stratified negative sampling strategy (see Module 4); validate predictions against geological knowledge |
| Class imbalance causes high false positive rate | High | High | PR-AUC as primary metric; SMOTE/weighted loss; threshold calibration per mineral type |
| GDAL/rasterio dependency conflicts | Low | High (for new devs) | Docker-based dev environment; conda-forge for geospatial dependencies |

---

## 12. Team & Roles Required

| Role | Phase | Key Skills |
|---|---|---|
| **Economic Geologist** | All | Mineral systems knowledge, field validation, geological plausibility review |
| **Remote Sensing Specialist** | 1--4 | ASTER/Sentinel/EMIT processing, spectral analysis, hyperspectral unmixing |
| **ML / Data Scientist** | 1--4 | scikit-learn, PyTorch, TorchGeo, geospatial ML, spatial statistics |
| **Backend Developer** | 2--4 | Python, FastAPI, PostgreSQL, Docker, Airflow |
| **Frontend Developer** | 2--4 | React, Leaflet.js, Deck.gl, data visualization |
| **GIS Analyst** | 1--4 | QGIS, GDAL, PostGIS, EnMAP-Box |
| **DevOps Engineer** | 3--4 | Kubernetes, CI/CD, monitoring |

**Minimum viable team for Phase 1:** 1 geologist-data scientist hybrid (1 person who can do both)

> **Realistic note:** Finding someone with both economic geology expertise AND ML/remote sensing skills is rare. More likely: 1 geologist who understands mineral systems + 1 data scientist willing to learn basic geology. The geologist's role in Phase 1 is critical for **validating that model outputs make geological sense** -- a model that flags every ironstone outcrop as a gold target is useless regardless of its AUC score.

---

## 13. Open Source Licensing Strategy

- **Core platform:** Release under **Apache 2.0** -- allows commercial use, encourages adoption
- **Trained models:** Distribute weights under **CC BY 4.0** with attribution
- **Training datasets:** Publish augmented geological datasets under **ODbL** (Open Database License)
- **Proprietary layer (monetization):**
  - Hosted SaaS with managed infrastructure
  - Premium support and SLA guarantees
  - Custom model training for specific regions/minerals/clients
  - Priority access to new satellite data integrations
  - Consulting services for ground truth campaign design

**Why open source the core?**
- Builds trust with geological community (peer review of methodology)
- Attracts contributors (universities, geological surveys already doing this work manually)
- Viral distribution -- geologists share tools that work
- Creates a moat through data and model quality, not software lock-in
- Enables national geological surveys to self-host (builds government partnerships)

---

## 14. Next Steps (Start Tomorrow)

### Day 1 -- Environment Setup (4 hours)
```bash
# 1. Install core dependencies
conda create -n geomine python=3.11
conda activate geomine
conda install -c conda-forge gdal rasterio fiona geopandas numpy scipy

# Install ML and remote sensing packages via pip
pip install scikit-learn xgboost torch torchgeo spectral pysptools
pip install pystac-client stackstac rioxarray xarray dask
pip install fastapi uvicorn folium shap mlflow

# 2. Install SNAP (ESA Sentinel Application Platform)
# Download from: https://step.esa.int/main/download/snap-download/

# 3. Install QGIS + EnMAP-Box plugin
# Download QGIS from: https://qgis.org/en/site/forusers/download.html
# Install EnMAP-Box via QGIS Plugin Manager

# 4. Set up PostgreSQL + PostGIS
docker run -d --name postgis -e POSTGRES_PASSWORD=geomine \
  -p 5432:5432 postgis/postgis:16-3.4

# 5. Register free accounts (do this today):
#    - https://dataspace.copernicus.eu/ (Sentinel data - NEW portal, replaces scihub)
#    - https://earthexplorer.usgs.gov/ (Landsat + ASTER)
#    - https://earthengine.google.com/ (Google Earth Engine)
#    - https://earthdata.nasa.gov/ (NASA EarthData - for EMIT, SRTM)
#    - https://www.mindat.org/ (MINDAT mineral database - register for API)
```

### Day 2--3 -- First Data Download
```bash
# Download Sentinel-2 data via STAC API (replaces deprecated sentinelsat)
python scripts/download_sentinel2.py --aoi great_dyke_zimbabwe.geojson \
  --start 2024-01-01 --end 2024-12-31 --cloud-cover 20

# Download ASTER data from NASA EarthData
python scripts/download_aster.py --aoi great_dyke_zimbabwe.geojson

# Download EMIT hyperspectral data (if available for AOI)
python scripts/download_emit.py --aoi great_dyke_zimbabwe.geojson

# Download USGS MRDS training labels
python scripts/download_mrds.py --bbox 29.0,-21.5,31.0,-19.0
```

### Day 4--7 -- First Spectral Analysis
- Compute clay ratio, iron oxide ratio, ferric iron index on Sentinel-2 imagery
- Run SAM mineral matching on ASTER SWIR bands against USGS spectral library
- Visualize in QGIS with known deposit overlay
- Compare spectral anomalies with MRDS + MINDAT deposit locations
- Document which indices correlate with which deposit types in the Great Dyke context

### Week 2 -- First ML Model
- Assemble training dataset with stratified negative sampling
- Train Random Forest + XGBoost with spatial block CV
- Compute SHAP values -- do the important features make geological sense?
- Generate first probability heat map + uncertainty map
- Review with geologist: are the high-probability zones geologically plausible?

### Resources to Bookmark
- Copernicus Data Space: https://dataspace.copernicus.eu/
- ESA SNAP Tutorials: https://step.esa.int/main/doc/tutorials/
- Google Earth Engine Guides: https://developers.google.com/earth-engine/guides
- USGS Mineral Resources: https://mrdata.usgs.gov/
- EMIT Data Portal: https://earth.jpl.nasa.gov/emit/
- MINDAT API: https://api.mindat.org/
- Macrostrat API: https://macrostrat.org/api
- USGS Spectral Library: https://www.usgs.gov/labs/spectroscopy-lab/science/spectral-library
- TorchGeo Documentation: https://torchgeo.readthedocs.io/
- Spectral Python Documentation: https://www.spectralpython.net/
- Fatiando a Terra: https://www.fatiando.org/

---

## 15. Weak Point Analysis & Design Decisions

This section documents known weaknesses in the approach and explicit design decisions made to address them. This is not a sales document -- it's an engineering document, and honest assessment of limitations is more valuable than optimism.

### WP1: Surface-Only Detection
**Problem:** Satellite remote sensing only detects surface mineralogy. In areas with thick soil cover, vegetation, or alluvial deposits, the mineral signature may be completely masked.

**Impact:** Much of tropical Africa (DRC, Mozambique, Tanzania) has thick regolith and dense vegetation. The Great Dyke itself has relatively good exposure, but extending to equatorial regions will degrade performance significantly.

**Mitigation:**
- SAR data penetrates vegetation and detects structural geology beneath cover
- NDVI anomalies (stressed vegetation over mineralized ground) provide indirect signals
- Geophysical data integration (Phase 2+) provides subsurface information
- **Clearly communicate per-AOI coverage quality to clients** -- a "low confidence due to cover" warning is more valuable than a false prediction

### WP2: Class Imbalance & Spatial Autocorrelation
**Problem:** Mineral deposits occupy <0.1% of any study area. Standard ML approaches will either predict "no deposit" everywhere (high accuracy, useless) or flag everything as a deposit (low precision, also useless). Additionally, nearby pixels are correlated, so random train/test splits overestimate performance.

**Mitigation:** Spatial block cross-validation, SMOTE/ADASYN oversampling, PR-AUC as primary metric, and weighted loss functions. This is the single most important methodological decision in the project.

### WP3: Geological Domain Shift
**Problem:** A model trained on the Great Dyke (layered mafic intrusion, Cr-PGM deposits) will not generalize to the Zambian Copperbelt (sediment-hosted Cu) or Limpopo Belt (orogenic Au) without retraining. Mineral deposit types have fundamentally different signatures.

**Mitigation:** Train separate models per deposit type, or use a multi-task model with deposit-type-specific heads. Foundation models (Prithvi/Clay) help by providing general EO features that transfer across geologies. Phase 2 tests on 3 diverse AOIs specifically to quantify domain shift.

### WP4: Validation Without Drilling
**Problem:** Until predictions are validated by drilling, we're measuring correlation with known deposits, not predictive power on unknown deposits. A model that rediscovers known deposits is a literature review, not a prediction engine.

**Mitigation:** True validation requires a "double-blind" test -- predict on an area where deposits exist but were withheld from training. This requires careful data management and ideally partnership with a mining company willing to share proprietary drill results.

### WP5: Spectral Index Ambiguity
**Problem:** Iron oxide ratios flag laterite, ferricrete, and red soils -- not just mineralization. Clay ratios flag agricultural soil, weathered rock, and sedimentary sequences. Without hyperspectral discrimination, many spectral "anomalies" are false positives.

**Mitigation:** Hyperspectral data (EMIT, PRISMA, EnMAP) with SAM matching against reference libraries dramatically reduces ambiguity. The ensemble approach uses structural geology and multi-feature context to filter false positives. SHAP explanations let geologists see why an area was flagged.

### WP6: Dependency on Free Data Continuity
**Problem:** The project depends entirely on free satellite data. ESA, NASA, and USGS have historically maintained free access, but policy can change.

**Mitigation:** STAC-based data catalog with multiple redundant sources. Local data archive for critical regions. OpenDataCube for self-hosted satellite data management. The most critical data (ASTER, SRTM) has already been fully collected and archived -- it won't disappear.

---

## Appendix A: Key Spectral Band Reference

### Sentinel-2 Bands for Mineral Mapping
| Band | Wavelength | Resolution | Mineral Use |
|---|---|---|---|
| B2 (Blue) | 490 nm | 10m | Iron oxide ratio denominator |
| B3 (Green) | 560 nm | 10m | Vegetation health |
| B4 (Red) | 665 nm | 10m | Iron oxide, ferricrete |
| B8 (NIR) | 842 nm | 10m | NDVI, biomass |
| B8A (NIR narrow) | 865 nm | 20m | Improved vegetation/mineral discrimination |
| B11 (SWIR1) | 1610 nm | 20m | Clay minerals, ferrous iron |
| B12 (SWIR2) | 2190 nm | 20m | Clay/carbonate ratio |

### ASTER Bands for Mineral Mapping
| Band | Wavelength | Mineral Target |
|---|---|---|
| B1--3 (VNIR) | 0.52--0.86 um | Iron oxides, ferric minerals |
| B4--9 (SWIR) | 1.6--2.43 um | Clay, carbonate, sulphate, Al-OH, Mg-OH |
| B10--14 (TIR) | 8.1--11.6 um | Silica, carbonate, quartz, feldspar |

### EMIT Bands for Mineral Mapping
| Range | Wavelength | Mineral Target |
|---|---|---|
| VNIR (59 bands) | 381--1000 nm | Iron oxides, Fe2+/Fe3+ discrimination |
| SWIR (226 bands) | 1000--2500 nm | Individual clay species, carbonates, sulfates, micas |

> **EMIT advantage:** With 285 contiguous bands at ~7.4nm spectral resolution, EMIT can discriminate individual mineral species (kaolinite vs montmorillonite vs illite) that are indistinguishable with Sentinel-2's 13 broad bands. This is transformative for identifying specific alteration assemblages tied to deposit types.

---

## Appendix B: Additional Open-Source Tools Reference

| Tool | Purpose | URL |
|---|---|---|
| **TorchGeo** | Deep learning on satellite imagery (datasets, samplers, pre-trained models) | github.com/microsoft/torchgeo |
| **Spectral Python** | Hyperspectral image I/O, display, classification, spectral matching | github.com/spectralpython/spectral |
| **PySptools** | Spectral unmixing, endmember extraction, SAM/SID matching | github.com/ctherien/pysptools |
| **HyperSpy** | Multi-dimensional data analysis (PCA, NMF, ICA on spectral data) | github.com/hyperspy/hyperspy |
| **Prithvi** | NASA/IBM geospatial foundation model (fine-tunable) | github.com/NASA-IMPACT/Prithvi-EO-2.0 |
| **Clay** | Open-source EO foundation model | github.com/Clay-foundation/model |
| **TerraTorch** | Fine-tuning toolkit for geospatial foundation models | github.com/IBM/terratorch |
| **GemPy** | Implicit 3D geological modeling | github.com/cgre-aachen/gempy |
| **LoopStructural** | 3D structural geology modeling with uncertainty | github.com/Loop3D/LoopStructural |
| **Fatiando a Terra** | Geophysics (gravity, magnetics gridding and modeling) | github.com/fatiando |
| **SimPEG** | Geophysical forward modeling and inversion | github.com/simpeg/simpeg |
| **PyGMT** | Publication-quality geological maps | github.com/GenericMappingTools/pygmt |
| **PyVista** | 3D visualization (drill holes, block models, surfaces) | github.com/pyvista/pyvista |
| **OpenDataCube** | Self-hosted satellite data management (STAC-compatible) | github.com/opendatacube/datacube-core |
| **TiTiler** | Dynamic COG tile server (Python/FastAPI-native) | github.com/developmentseed/titiler |
| **PDAL** | Point cloud processing (LiDAR/drone data) | pdal.io |
| **MintPy** | InSAR time-series analysis (ground deformation) | github.com/insarlab/MintPy |
| **OMF** | Open Mining Format for data interchange | github.com/gmggroup/omf |
| **EnMAP-Box** | QGIS plugin for imaging spectroscopy workflows | github.com/EnMAP-Box/enmap-box |
| **leafmap** | Interactive geospatial analysis in Jupyter notebooks | github.com/opengeos/leafmap |
| **SHAP** | Model explainability (feature importance per prediction) | github.com/shap/shap |

---

*Document Version: 2.0 | Created: April 2026 | Updated: April 2026 | Project: GeoMine AI*
*License: CC BY 4.0 -- Attribution required for derivative works*
