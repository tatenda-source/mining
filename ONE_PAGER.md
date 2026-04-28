# GeoMine AI

**Audit-grade mineral targeting. Find PGM deposits 100x cheaper than drilling -- with a verifiable hash, not a marketing claim.**

---

## Headline Number

**Cross-tile PR-AUC: 0.453** (leave-one-tile-out CV, Great Dyke, Zimbabwe)

Class prior: 0.226. Random CV: 0.612. Leakage gap: 0.159. Grade A on the 5-test audit protocol.

Reproduce it: `geomine audit data/benchmark/dataset.parquet data/benchmark/model.joblib` -- the certificate hash will match if the run is real.

## What It Does

Two products built on the same pipeline:

**1. Targeting.** `POST /v1/score` takes a concession boundary, returns ranked prospectivity zones with confidence scores. Built from free Sentinel-2 + ASTER + DEM, no fieldwork.

**2. Audit.** `POST /v1/audit` (or `geomine audit` CLI) takes any binary classifier + labelled dataset and runs the GeoMine validation protocol: spatial leakage, class-prior baseline, bootstrap stability, calibration, feature-label leakage. Returns a content-addressed certificate. Same data + same model = same hash.

## Who Pays For This

| Segment | Product | Why |
|---|---|---|
| Mining-AI startups | Audit-as-a-service | Their LPs want the model graded by an independent protocol before the next round |
| Junior miners | Targeting reports | Reduce $2-50M exploration cost to a $25K starter map |
| Banks / sovereign funds underwriting mining loans | Audit certificate | "Show me the LOTO PR-AUC, not the random-CV number" -- compliance-grade |
| Geological surveys | API access to Great Dyke probability map | Replaces $1M+ regional studies |

## Pricing

| Product | Price |
|---|---|
| Per-audit certificate (single model) | $5,000 |
| Per-concession targeting report | $25,000 |
| Audit subscription (unlimited runs, 1 org) | $50,000/year |
| Great Dyke probability map license | $100,000 |

## Why It's Credible

We tested 8 model configurations. Phase 1 collapsed under leave-one-tile-out CV (PR-AUC 0.228, random baseline 0.226 -- no skill). We documented this publicly. Phase 2 closed the gap: 0.453 LOTO, audited, signed.

**Most mining AI hides failure. We shipped it as a product feature.**

The audit module that caught our own failure is the same module we sell to grade other people's models. The moat is the protocol, not the prediction.

## The Technology

| Layer | What |
|---|---|
| Data | ESA Sentinel-2 + NASA ASTER + Copernicus DEM (free) |
| Processing | 7,000 lines Python (GDAL, rasterio, geopandas) |
| ML backend | Logistic regression baseline + Prithvi-EO-2.0 ViT (NASA/IBM) |
| Validation | `geomine.audit` -- LOTO CV + bootstrap + leakage detection + content-addressed certificates |
| API | FastAPI (`/v1/score`, `/v1/audit`, `/v1/benchmark`) |

## The Market

- PGMs in structural supply deficit, $1,500-2,000+/oz
- Chromite has no substitute for stainless steel
- Zimbabwe Great Dyke = 2nd largest PGM reserve globally
- 300+ active concessions on the Great Dyke alone
- Every mining-AI startup competing for capital is a candidate audit customer

## Run It

```bash
pip install -e ".[api,audit]"
uvicorn geomine.api.main:app --port 8000
curl http://localhost:8000/v1/benchmark
```

## Contact

GeoMine AI Project
tatendawalter62@gmail.com

---

*Built on $0 infrastructure. Validated by failure. Now auditable by anyone.*
