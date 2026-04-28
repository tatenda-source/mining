# GeoMine Benchmark

Headline numbers for the GeoMine model, pinned and verifiable. Same data + same code = same certificate hash.

## Current Model: Phase 2 (ViT cross-tile)

| Metric | Score |
|---|---|
| PR-AUC, random CV (5-fold) | 0.612 |
| **PR-AUC, leave-one-tile-out CV** | **0.453** |
| Spatial leakage gap | 0.159 |
| Class prior baseline | 0.226 |
| Bootstrap stable feature fraction | 0.83 |
| Expected calibration error | < 0.10 |
| Tests passed | 5/5 |
| Grade | A |

**Geographic extent:** Great Dyke, Zimbabwe -- 4 Sentinel-2 tiles (T35KRU, T36KTC/D/E)
**Deposits:** 17 labelled (Cr, PGM, Ni, Au)
**Last updated:** 2026-04-28

## Why These Numbers Matter

Most mining-AI benchmarks report random-CV PR-AUC. We report both random and leave-one-tile-out (LOTO) CV. **The gap is the truth.**

- Random CV inflates by capturing spatial autocorrelation -- positives near other positives.
- LOTO CV holds out an entire tile, forcing the model to predict deposits in geography it has never seen.

Phase 1 random CV was 0.841. LOTO CV was 0.228. **A 0.613 gap.** Phase 2 closed that gap to 0.159 with a 0.453 LOTO score. Reproducible, audited, signed.

## How to Verify

```bash
geomine audit data/benchmark/dataset.parquet data/benchmark/model.joblib \
    --output your_audit.md \
    --json-output your_audit.json
```

Compare the `certificate` field in your output to the certificate published here. Match = your local run reproduced ours bit-for-bit.

## Audit Protocol (`geomine.audit`)

Every benchmark run executes five tests:

1. **Spatial leakage** -- gap between random CV and spatial block CV must be < 0.30.
2. **Class prior** -- spatial CV PR-AUC must exceed the rate of positives in the dataset.
3. **Bootstrap stability** -- coefficient signs must be stable across 200 bootstrap resamples.
4. **Calibration** -- ECE on a held-out fold must be < 0.10.
5. **Feature-label leakage** -- no feature with |corr| ≥ 0.95 against the label.

Failing any test fails the audit. The protocol is published; the thresholds are public.

## Benchmark History

| Date | Model | LOTO PR-AUC | Grade | Notes |
|---|---|---|---|---|
| 2026-04-28 | ViT cross-tile (Phase 2) | 0.453 | A | Phase 2 close-out |
| 2026-04-22 | Z-score normalization | 0.228 | F | Did not improve over baseline |
| 2026-04-15 | LR(ferric_iron) baseline | 0.228 | F | Phase 1 close-out |
| 2026-04-08 | LR(ferric_iron) random CV | 0.841 | -- | Inflated by spatial leakage |

## What This Number Is Not

- **Not a guarantee of mineral discovery.** PR-AUC measures rank quality on labelled deposits, not exploration outcome.
- **Not transferable to other geologies.** This model was trained on the Great Dyke. A different orogen needs a different model and a different audit.
- **Not a substitute for fieldwork.** Probability maps narrow targets; drilling confirms them.

## What This Number Is

- The first cross-tile generalizable mineral prospectivity score we have produced.
- Verifiable by anyone with the published dataset.
- Auditable by `geomine.audit` against the public protocol.

---

*Run the audit yourself. Match the hash. The number is what it claims to be.*
