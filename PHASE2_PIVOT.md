# Phase 2 Pivot -- From Spectral Regression to Foundation Models

**Date:** 4 April 2026
**Status:** Pivoting after comprehensive negative result
**Study area:** Great Dyke, Zimbabwe (Cr/PGM)

---

## What We Tested (Exhaustive)

| Model | Features | CV Method | PR-AUC | Baseline | Lift | Result |
|---|---|---|---|---|---|---|
| LR(ferric_iron) | 1 S2 index | Tile LOTO | 0.035 | 0.054 | 0.7x | FAIL |
| LR(ferric_iron) Z-scored | 1 S2 index normalized | Tile LOTO | 0.035 | 0.054 | 0.7x | FAIL |
| LR(6 S2 indices) | 6 S2 indices | Tile LOTO | 0.028 | 0.045 | 0.6x | FAIL |
| LR(terrain) | 6 DEM features | Tile LOTO | 0.016 | 0.024 | 0.7x | FAIL |
| LR(6 ASTER SWIR) | Mg-OH, Al-OH, etc. | Tile LOTO | N/A | N/A | N/A | Folds empty |
| LR(S2 Z-scored) | 6 S2 indices | Geo block | 0.034 | 0.045 | 0.8x | FAIL |
| LR(DEM only) | 5 terrain features | Geo block | 0.050 | 0.055 | 0.9x | FAIL |
| LR(S2 + DEM) | 11 combined | Geo block | 0.037 | 0.045 | 0.8x | FAIL |

**Every configuration produced lift <= 1.0x.** The model is no better than random.

---

## Why It Failed (Root Causes)

### 1. Mixed Pixel Problem (Sensor Physics)
At 10-30m resolution, a single pixel covers 100-900 m2. Chromitite seams and mineralized shear zones are 5-10m wide. The mineral signature is mathematically diluted by surrounding vegetation, soil, and barren rock. No band ratio can extract a signal that isn't there at the pixel level.

### 2. Atmospheric Non-Stationarity (Sentinel-2)
Ferrous iron coefficient flipped sign between tiles (+0.446 in T35KRU, -0.650 in T36KTC). This mathematically proves the model learned atmospheric/vegetation patterns, not rock mineralogy. The signal was real within one scene but non-transferable.

### 3. Label Insufficiency
26 point labels with km-scale coordinate uncertainty over a 550km linear feature. The model was underdetermined -- not enough data to learn any meaningful spatial pattern, let alone a transferable one.

### 4. Feature Space Collapse
All tested features (spectral, terrain, SWIR mineral indices) occupy the same value range at deposit and background locations. There is no separating hyperplane in the current feature space.

---

## What Still Works

1. **The pipeline is production-grade.** Download -> process -> compute -> train -> predict -> validate runs end-to-end.
2. **The validation framework is rigorous.** LOTO CV, geographic block CV, bootstrap CIs, coefficient stability checks -- these correctly identified every failure mode.
3. **The data infrastructure is built.** 4 Sentinel-2 tiles, 11 ASTER scenes, DEM, lineaments, 26+ labeled deposits, Menabilly drill hole KML.
4. **The negative result is publishable.** Proving that Sentinel-2 spectral indices cannot transfer across tiles for PGM/Cr targeting is a genuine scientific contribution.

---

## The Pivot: Two-Track Strategy

### Track 1: Geology-Constrained Model (Immediate)

**Concept:** Stop asking the satellite to discover geology. Instead, use mapped geology as the primary input and satellite imagery as an anomaly detector within known rock units.

**Data needed:**
- Zimbabwe Geological Survey lithology polygons (mapped rock units)
- Structural geology vectors (faults, shear zones)
- Known mine boundaries (polygon, not point)

**Model approach:**
- Mask by lithology: only score pixels within pyroxenite/dunite units
- Within those units, look for spectral anomalies (deviation from unit mean)
- Use structural proximity as a secondary weighting factor

**Expected impact:** Dramatically reduces false positives by constraining the search space to geologically plausible rock types. Even a weak spectral signal becomes useful when you're only looking at the right rocks.

### Track 2: Foundation Model Fine-Tuning (The Moat)

**Concept:** Use pre-trained Earth observation Vision Transformers (Prithvi-EO-2.0 or Clay) that have learned spatial patterns from petabytes of global satellite imagery. Fine-tune on our local dataset for binary segmentation (deposit vs non-deposit).

**Why this could work where LR failed:**
1. **Spatial context.** LR sees one pixel. A ViT sees a 224x224 patch (~2.2km x 2.2km at 10m). It can learn spatial patterns (texture, edge relationships, neighborhood context) that single-pixel models cannot.
2. **Pre-trained invariance.** These models have seen millions of scenes across all atmospheric conditions, seasons, and geographies. They've already learned to factor out illumination, vegetation seasonality, and atmospheric effects.
3. **Transfer learning.** Instead of training from 26 labels, we're fine-tuning from billions of pre-trained parameters. The model starts with a rich understanding of Earth surface processes.

**Compute requirements:**
- Fine-tuning: 1x A100 GPU for 2-4 hours (Google Colab Pro or ~$5 on Lambda/RunPod)
- Inference: CPU possible, GPU preferred
- Storage: ~2GB model weights

**Target validation:**
- LOTO CV PR-AUC >= 0.45 (the defensible threshold)
- Coefficient stability is replaced by attention map analysis (which patches drive the prediction)

---

## Success Criteria for Phase 2b

| Milestone | Target | What It Proves |
|---|---|---|
| Geology mask reduces search space | > 80% of deposits in < 20% of area | Lithology constrains correctly |
| Foundation model LOTO PR-AUC | >= 0.45 | Spatial features transfer across tiles |
| Attention maps highlight geological contacts | Visual inspection | Model learned geology, not artifacts |
| Blind test on held-out deposits | >= 3/5 in top 20% zones | Practical targeting value |

---

## What We're Building Toward

The pitch is no longer "our spectral index found deposits." It's:

> "We combined NASA's foundation model with mapped geology to build a spatial targeting system that identifies prospective zones within the right rock units. The model generalizes across 100km of the Great Dyke with a cross-tile validation score of X.XX, reducing reconnaissance search space by 60-80%."

This is a defensible technical moat:
1. Foundation model weights are public, but the fine-tuning recipe is proprietary
2. Geology-constrained masking requires domain expertise
3. The validation framework proves it works (not just claims)

---

## Honest Assessment

| Dimension | Phase 1 | Phase 2a (current) | Phase 2b (target) |
|---|---|---|---|
| Research value | 8/10 | 9/10 (clean negative) | TBD |
| Commercial readiness | 3/10 | 2/10 | 5-6/10 |
| Technical foundation | High | Very high | Very high |
| Signal reliability | Local only | None cross-tile | TBD |

The engineering is ahead of the science. The pivot to foundation models + geology constraints is the most capital-efficient path to a defensible result.

---

*Phase 2a completed: 4 April 2026*
*8 model configurations tested. 11 ASTER scenes downloaded. 54 drill holes classified. 0 configurations beat baseline.*
*The honest answer: Sentinel-2 spectral indices cannot map PGM/Cr deposits across geography at this scale.*
