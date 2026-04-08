# GeoMine AI

**Satellite-powered mineral targeting. Zero data cost. Institutional-grade validation.**

---

## What It Does

GeoMine AI takes free satellite imagery (Sentinel-2, ASTER, DEM) and produces probability maps showing where PGM and chromite deposits are most likely located -- without fieldwork, without drilling, without lab analysis.

Upload a concession boundary. Get back ranked prospectivity zones with confidence scores.

## Who It's For

- **Junior mining companies** raising capital for Great Dyke (Zimbabwe) concessions
- **Exploration consultancies** seeking ML-augmented targeting
- **Mining AI startups** needing rigorous model validation (due diligence)

## What It Costs

| Service | Price |
|---|---|
| ML Due Diligence Audit (validate your model) | $5,000-25,000 |
| Per-Concession Targeting Report | $10,000-50,000 |
| Full Great Dyke Probability Map License | $50,000-100,000 |

95%+ gross margin. Zero marginal data cost.

## Why It's Credible

We tested 8 model configurations with Leave-One-Tile-Out cross-validation. All failed to generalize. We documented this publicly.

**Most mining AI hides failure. We publish it.**

That validation discipline is what makes the next result trustworthy. When the cross-tile number passes 0.45, it means something -- because we proved we aren't gaming the metric.

## The Technology

| Layer | What | Cost |
|---|---|---|
| Data | ESA Sentinel-2 + NASA ASTER + AWS DEM | Free |
| Processing | 8,300 lines Python (GDAL, rasterio) | Free |
| ML Backend | Prithvi-EO-2.0 (NASA/IBM foundation model) | Free |
| Validation | LOTO CV + bootstrap + leakage detection | Proprietary |
| Compute | Google Colab T4 GPU | Free |

## The Market

- PGMs in structural supply deficit, $1,500-2,000+/oz
- Chromite has no substitute for stainless steel
- Zimbabwe's Great Dyke = 2nd largest PGM reserve globally
- 300+ active concessions on the Great Dyke alone

## Contact

GeoMine AI Project
tatendawalter62@gmail.com

---

*Built on $0 infrastructure. Validated by failure. Ready for the number that matters.*
