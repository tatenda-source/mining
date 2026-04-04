# GeoMine AI -- What We're Building

---

## The One-Liner

**A system that reads satellite imagery and tells you where to dig for minerals -- before spending millions on exploration.**

---

## What We're Building

A satellite-powered mineral prediction engine. It takes free satellite data from space agencies, runs spectral and terrain analysis to detect mineral signatures, trains machine learning models on known deposit locations, and outputs ranked target maps showing where to explore next.

The first target: **chromite and platinum group metals (PGMs) on the Great Dyke, Zimbabwe** -- a 550km layered intrusion that hosts some of Africa's largest chrome and platinum deposits.

### How It Works (in plain terms)

1. **Download satellite imagery** from ESA and NASA (free)
2. **Compute spectral indices** -- different minerals reflect light differently. Iron oxides glow in red/blue ratios. Clays absorb in shortwave infrared. Serpentinite (the rock that hosts chromitite) has a distinct ferrous iron signature.
3. **Extract terrain features** from elevation data -- faults, lineaments, drainage patterns that control where minerals concentrate
4. **Train a model** using known mine locations as "positive" examples and barren ground as "negative" examples
5. **Predict** a probability map over the entire study area -- every 10m pixel gets a score from 0-100%
6. **Rank targets** -- cluster high-probability zones into exploration targets, sorted by confidence

### Who It's For

- Junior mining companies that can't afford $2M+ reconnaissance campaigns
- National geological surveys (Zimbabwe, Zambia, DRC, Mozambique)
- Academic geology departments
- Government resource ministries doing mineral inventories

---

## What We've Built (so far)

| Component | Status | Description |
|---|---|---|
| **CLI pipeline** | Done | `geomine download / compute-features / train / predict / layers` |
| **Satellite downloader** | Done | Copernicus Data Space (Sentinel-2) + NASA EarthData (DEM) |
| **Spectral index engine** | Done | 6 indices: clay ratio, iron oxide, ferric iron, NDVI, ferrous iron, clay SWIR |
| **Band composite layers** | Done | 7 geological visualization presets (true color, SWIR, geology combos) |
| **Structural geology module** | Done (code) | Slope, aspect, curvature, lineament extraction, drainage density |
| **ML training pipeline** | Done (code) | XGBoost + baselines + spatial block CV + SHAP explainability |
| **Prediction engine** | Done (code) | Chunked raster prediction + bootstrap uncertainty + target clustering |
| **Exploration bias guard** | Done | Detects if model learns "near roads" instead of "geological signal" |
| **Lithology vs prospectivity guard** | Done | Detects if model maps rock type instead of deposit-bearing horizons |
| **Along-strike CV** | Done | Cross-validation that respects the Great Dyke's linear geometry |
| **Data downloaded** | Done | 4 Sentinel-2 tiles (4.5 GB), 9 DEM tiles (330 MB), 17 deposit labels |
| **First signal check** | Done | ferric_iron and inverted ferrous_iron show signal (n=4, need n=17) |

---

## What We Still Need to Build

| Component | Phase | Priority | Description |
|---|---|---|---|
| **Full signal check (n=17)** | 1 | NOW | Process remaining 3 tiles, test all 17 deposits |
| **Terrain feature computation** | 1 | This week | Run slope/aspect/lineament extraction on DEM |
| **Feature stack assembly** | 1 | This week | Combine all spectral + terrain into single raster |
| **Model training + SHAP audit** | 1 | Week 2 | XGBoost with spatial CV, verify features are geological |
| **Probability + uncertainty maps** | 1 | Week 2 | Heat maps of deposit probability and model confidence |
| **Transfer test** | 1 | Week 3 | Test model on a geologically different area |
| **Airflow pipeline automation** | 2 | Month 2 | Auto-download → preprocess → predict for new AOIs |
| **Hyperspectral processing** | 2 | Month 2 | EMIT/PRISMA SAM matching for mineral species discrimination |
| **FastAPI backend + TiTiler** | 2 | Month 3 | Web API and COG tile serving |
| **Leaflet.js dashboard** | 2 | Month 3 | Interactive map with probability overlays |
| **Foundation model fine-tuning** | 2 | Month 3 | Prithvi/Clay for transfer learning |
| **Ground truth feedback loop** | 3 | Month 4+ | Field data → model retraining → active learning |
| **QGIS plugin** | 3 | Month 5+ | Desktop integration for geologists |
| **Multi-tenancy + auth** | 3 | Month 6+ | SaaS product features |

---

## Free Technology We're Using

### Free Satellite Data (the foundation -- all $0)

| Source | Data | Provider | Value If Purchased |
|---|---|---|---|
| **Copernicus Data Space** | Sentinel-2 multispectral imagery (10-60m) | ESA (European Space Agency) | ~$1,000-5,000/scene equivalent |
| **NASA EarthData** | SRTM/Copernicus DEM 30m elevation | NASA / DLR | ~$500-2,000/region |
| **NASA EarthData** | ASTER multispectral (best for minerals) | NASA/METI | ~$2,000-10,000/region |
| **NASA EarthData** | EMIT hyperspectral (285 bands, Phase 2) | NASA/JPL | Priceless -- no commercial equivalent |
| **USGS EarthExplorer** | Landsat 8/9 (backup multispectral) | USGS | ~$500-1,500/scene |
| **AWS Open Data** | Copernicus DEM 30m tiles | Copernicus/AWS | Free hosting of free data |
| **USGS MRDS** | Global mineral deposit database | USGS | Would cost $50k+ to compile manually |
| **MINDAT** | 400k+ mineral occurrence localities | Community | Irreplaceable community resource |

**Total value of free data we're using: ~$50,000-200,000 per study region** if purchased commercially.

### Free Software Stack (all open source, $0 licensing)

| Tool | What It Does | Commercial Equivalent | Commercial Cost |
|---|---|---|---|
| **Python 3.11** | Core language | MATLAB | ~$2,000/yr |
| **GDAL + Rasterio** | Geospatial raster processing | ERDAS IMAGINE | ~$5,000-15,000/yr |
| **GeoPandas + Shapely** | Vector spatial analysis | ArcGIS Pro | ~$1,500-7,000/yr |
| **scikit-learn + XGBoost** | Machine learning | SAS / SPSS | ~$5,000-50,000/yr |
| **SHAP** | Model explainability | Proprietary consulting | ~$10,000+/engagement |
| **QGIS** | Desktop GIS visualization | ArcGIS Desktop | ~$1,500-7,000/yr |
| **SNAP (ESA)** | Sentinel satellite processing | ENVI | ~$5,000-15,000/yr |
| **WhiteboxTools** | DEM analysis + lineaments | PCI Geomatica | ~$5,000-10,000/yr |
| **Docker** | Containerization | -- | Free |
| **PostgreSQL + PostGIS** | Spatial database | Oracle Spatial | ~$10,000-50,000/yr |
| **FastAPI** | Web API framework | -- | Free |
| **Leaflet.js / React** | Web dashboard | -- | Free |
| **MLflow / DVC** | ML experiment tracking | Weights & Biases (paid tier) | ~$1,000-5,000/yr |
| **TorchGeo** (Phase 2) | Deep learning on satellite imagery | Google Earth Engine Pro | ~$2,000-10,000/yr |
| **Spectral Python / PySptools** (Phase 2) | Hyperspectral mineral mapping | ENVI + Spectral Analysis | ~$10,000-20,000/yr |
| **GemPy** (Phase 3) | 3D geological modeling | Leapfrog Geo | ~$15,000-50,000/yr |
| **Fatiando a Terra** (Phase 2+) | Geophysical data processing | Oasis Montaj | ~$10,000-30,000/yr |

**Total commercial software equivalent: ~$70,000-270,000/year.** We pay $0.

### Free Compute (Phase 1)

| Resource | What It Does | Cost |
|---|---|---|
| **Google Earth Engine** | Petabyte-scale satellite processing | Free (research tier) |
| **Google Colab** | GPU for model training (fallback) | Free tier available |
| **Local machine** | All Phase 1 processing | $0 (your Mac) |

---

## What We'd Pay For (if we scale)

### Phase 1: $0

Everything runs on your Mac with free data and free software. The only "cost" is your internet bandwidth to download ~5-10 GB of satellite data.

### Phase 2 (if signal proves out): ~$50-450/month

| Item | Cost | Why |
|---|---|---|
| **VPS for web dashboard** | $20-50/mo | Hetzner/OVH for hosting the map dashboard |
| **GPU cloud for training** | $0-400/mo | Only needed for deep learning (U-Net, foundation models). Phase 1 XGBoost runs on CPU. Spot instances or Colab Pro ($10/mo) for occasional use. |
| **Domain name** | ~$12/yr | For the web dashboard |

### Phase 3 (product/SaaS): ~$500-2,000/month

| Item | Cost | Why |
|---|---|---|
| **Cloud GPU (persistent)** | $300-500/mo | AWS g4dn or equivalent for continuous inference |
| **Managed PostgreSQL** | $50-100/mo | PostGIS database for multi-tenant data |
| **S3/MinIO storage** | $20-50/mo | COG raster storage grows with clients |
| **Monitoring (Grafana)** | $0-50/mo | Free self-hosted, or Grafana Cloud |
| **Auth provider** | $0-25/mo | Auth0 free tier or self-hosted |
| **CI/CD** | $0 | GitHub Actions free tier |

### Things We Never Pay For

| Item | Why Free |
|---|---|
| **Satellite data** | ESA, NASA, USGS provide it free forever (mandated by policy) |
| **Core software** | Apache 2.0 / MIT licensed open source |
| **Training labels** | USGS MRDS, MINDAT, published geological literature |
| **Spectral reference libraries** | USGS splib07, ECOSTRESS -- government-funded, permanently free |

---

## The Commercial Comparison

A traditional mining company doing the same work pays:

| Traditional Approach | Cost | GeoMine AI Approach | Cost |
|---|---|---|---|
| License ENVI + ArcGIS + Leapfrog | $50-100k/yr | Open source stack | $0 |
| Purchase satellite imagery | $10-50k/region | Free Sentinel-2/ASTER/EMIT | $0 |
| Hire remote sensing consultant | $100-300k/yr | Automated pipeline | $0 |
| Regional reconnaissance field team | $500k-2M | Satellite-first targeting | $5-20k |
| Geophysical survey | $1-5M | Partially replaced (Phase 2) | $0 |
| **Total first-pass prospecting** | **$1.7-7.7M** | **GeoMine AI** | **$5-30k** |

The savings are real, but come with a caveat: **GeoMine AI does not replace drilling.** It replaces the expensive first-pass reconnaissance that decides *where* to drill. Better targeting means fewer wasted drill holes, which is where the 40-60% drilling cost reduction comes from.

---

## The Honest Assessment

### What's working
- Real spectral signal detected (ferric_iron, inverted ferrous_iron)
- The signal is geologically plausible (oxidation state and chromitite vs serpentinite)
- Pipeline is built end-to-end (download → process → train → predict)
- Multiple safety guards (exploration bias, lithology check, spatial CV)

### What's uncertain
- Only n=4 deposits tested so far (need n=17 from additional tiles)
- Single satellite scene (one dry-season day, not persistent geology)
- 17 training labels total -- high variance in any ML model
- Haven't proven the model finds anything a geologist wouldn't already know

### What could kill this
- Signal doesn't hold with more deposits (n=4 → n=17 might collapse)
- Model maps lithology not prospectivity (clay/iron oxide already show no discrimination)
- Training labels are too noisy (km-scale coordinate errors on older records)
- Can't generalize beyond the Great Dyke (different geology = different model)

---

*Last updated: 4 April 2026*
