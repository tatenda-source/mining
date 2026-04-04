# Phase 1 Postmortem -- Honest Assessment

**Date:** 4 April 2026
**Model:** LR(ferric_iron), single feature logistic regression
**Study area:** Great Dyke, Zimbabwe

---

## The Three Numbers That Matter

| Test | PR-AUC | Interpretation |
|---|---|---|
| Random CV | 0.841 | Inflated. Spatial leakage. |
| **Leave-one-tile-out CV** | **0.228** | **The real number. Signal does not cross tile boundaries.** |
| Along-strike CV | 0.599 | Partial. Works in some segments (0.95), fails in others (0.11). |

---

## What Is Genuinely Correct

### Simplicity winning is a green flag

With N=53 (12 positive, 41 negative):
- Logistic regression has low variance
- RF/XGBoost overfit (confirmed: XGBoost PR-AUC = 0.715 < LR's 0.841)
- Single feature reduces hypothesis space

The fact that the simplest model won is not embarrassing. It is statistically mature. This is textbook bias-variance tradeoff behavior.

### PR-AUC 0.841 is non-trivial (within its context)

With 12 positives, 41 negatives, and imbalanced data, a PR-AUC of 0.841 is meaningfully above baseline (0.226). That is real separation.

But **separation does not equal generalization.** That is the nuance.

### Spatial autocorrelation is the main risk (confirmed)

This was identified as the critical risk before training. The stress test confirmed it:
- 10/12 deposits cluster in tile T35KRU
- Model trained without that tile scores PR-AUC = 0.228
- The random CV was testing on deposits geographically near training deposits

**This is the only thing that truly invalidated Phase 1 as a generalizable result.**

---

## Where the Analysis Overreaches

### "Fundamental proxy for underlying geological variance" -- too strong

We proved: ferric iron correlates with deposits in this dataset.

We did NOT prove:
- It is causal
- It is fundamental
- It generalizes beyond the study area

Correlation under small N must stay humble.

### "High-confidence linear separator" -- unverified

We reported coefficient = +0.8385. But:
- Bootstrap 95% CI: [0.29, 1.68] -- wide
- Direction is stable (doesn't cross zero)
- Magnitude is uncertain (could be 0.3 or 1.7)

With 12 positive samples, coefficient stability is fragile. The direction is robust. The magnitude is not.

### "Paradigm shift" language -- premature

This is investor language. Academically, this is:

> "A promising spectral proxy under constrained conditions."

Scientists love replication. Mixing disruption narratives with preliminary results destroys credibility.

---

## Calibrated Assessment

### What was achieved

- A statistically detectable local signal
- A geologically interpretable model (Fe3+ = oxidation of sulphide-bearing horizons)
- A ranked targeting output (probability maps)
- A replicable ML pipeline (end-to-end, config-driven)
- A rigorous validation framework (LOTO, along-strike, bootstrap)

### Epistemic status

**"Preliminary but promising."**

Not: "Geological truth discovered."

### Rating

| Dimension | Score |
|---|---|
| Research milestone | 8/10 |
| Commercial-ready discovery engine | 3/10 |
| Foundation for something serious | Very high potential |

---

## The Fork in the Road

There is only one test that determines if this becomes real:

**Does performance hold under spatial block cross-validation?**

| Outcome | Implication |
|---|---|
| Random CV = 0.841, Spatial CV >= 0.70 | Captured geological structure |
| Random CV = 0.841, Spatial CV = 0.40-0.55 | Mostly spatial leakage |
| Random CV = 0.841, **Spatial CV = 0.228** | **Phase 1 was spatial leakage** |

We landed in the worst case. But this is diagnostic, not terminal.

---

## Why It Failed (Mechanistically)

### 1. Deposit clustering

10 of 12 deposits are in tile T35KRU. The model learns the spectral statistics of that tile, not the geology. When tested on T36KTC or T36KTE (different atmospheric conditions, different scene date), the raw ferric_iron values don't transfer.

### 2. No cross-tile normalization

Ferric iron = (B8A - B04) / (B8A + B04). This ratio is affected by:
- Atmospheric water vapor (varies between scenes)
- Sun angle (different dates = different illumination)
- Aerosol optical depth
- Sensor calibration drift

Two identical chromitite outcrops in different tiles will have different ferric_iron values. The model sees "high value in T35KRU" not "chromitite signature."

### 3. Insufficient label diversity

12 deposits, 10 in one tile. Even a perfect model cannot learn cross-tile patterns from 2 out-of-tile examples.

---

## What Would Fix It

Three things, in order of impact:

### 1. Scene normalization (highest priority)

**What:** Make spectral features comparable across tiles.

**How:**
- Z-score per tile (subtract mean, divide by std)
- First-derivative spectral ratios (more invariant to illumination)
- Pseudo-invariant feature calibration (use stable targets across tiles as anchors)
- Continuum removal (normalize spectra by their hull)

**Expected impact:** LOTO PR-AUC from 0.228 to 0.40-0.50

### 2. Label expansion (parallel effort)

**What:** Go from 12 to 40-50 deposits spread across tiles.

**How:**
- Zimbabwe Geological Survey records
- Published literature on Great Dyke chromitite seams
- Mining company annual reports (Zimplats, Unki, Mimosa)
- MINDAT API
- Manual digitization from geological maps

**Expected impact:** Reduces CV variance, enables meaningful LOTO

### 3. ASTER SWIR integration

**What:** Add 6-band SWIR data (1.6-2.43 um) with better mineral discrimination.

**How:** Download ASTER L1T from NASA EarthData, compute Al-OH, Mg-OH, carbonate indices.

**Expected impact:** Features that are more geologically specific and potentially more invariant across scenes.

---

## What Would Turn This Into a Breakthrough

Three milestones:

1. **Spatial block CV >= 0.55 after normalization** -- signal survives domain shift
2. **Hold out an entire tile, train on others, predict successfully** -- cross-geography transfer
3. **Apply same model to a second craton without tuning** -- true generalization

If #3 holds, the "fundamental proxy" language becomes justified.

Until then: promising local signal, not yet generalizable.

---

## The Business Angle

### Current signal
Valuable in situ (T35KRU), not deployable elsewhere.

### The real product
Solving domain shift / tile-to-tile calibration. This is the competitive moat -- not the ML model itself.

### If cross-tile invariance is cracked
First-mover advantage in low-cost AI exploration in Zimbabwe (and potentially Africa-wide).

### What we're selling
Search space reduction, not gold. A tool that says "look here first" and saves 60-80% of reconnaissance cost. Not a tool that says "there is gold here."

---

## Where We Are

We are not in the Valley of Failure. We are in the **Valley of Disillusionment** -- exactly where breakthrough IP is forged.

The pipeline works. The signal exists locally. The failure mode is identified and understood. The path forward is clear: normalization + labels + ASTER.

Phase 2 is a bridge from 0.228 to 0.55+. If we cross it, this becomes publishable and potentially commercial. If we don't, we have a clean negative result and a reusable pipeline.

Either way, the engineering investment is not wasted.

---

*Phase 1 completed: 4 April 2026*
*18 commits, ~6,500 lines of code, 4 Sentinel-2 tiles, 17,045 lineaments, 1 honest answer.*
