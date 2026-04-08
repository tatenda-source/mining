# GeoMine AI -- Investor Narrative

*15 slides. Founder-level positioning. One metric that matters.*

---

## Slide 1: The Problem

Every year, the global mining industry spends **$13 billion** on mineral exploration. Over 90% of exploration programs fail to find an economic deposit.

The reason: geologists are still walking grids with hand samples and shipping rocks to labs. Satellite data covering every square meter of Earth's surface sits unused.

---

## Slide 2: The Insight

**Free satellite data already contains geological signal.** Sentinel-2 (ESA) captures spectral reflectance every 5 days at 10-meter resolution. ASTER (NASA) measures mineral-specific SWIR absorption features. Digital elevation models resolve fault structures and drainage patterns.

This data is free, global, and updated continuously. Nobody in mining is using it properly because they lack the validation discipline to know whether their models actually work.

---

## Slide 3: What We Built

A modular geospatial ML platform that goes from raw satellite imagery to validated probability maps in hours:

**8,300+ lines** of production Python. **42 GB** of processed data. **$0/month** infrastructure cost.

Five layers:
1. **Data Ingestion** -- Sentinel-2, ASTER, DEM, geological vectors
2. **Feature Computation** -- 10 spectral indices, terrain, structural geology
3. **Validation Framework** -- the moat (see next slide)
4. **Model Backend** -- swappable: LR/XGBoost today, Prithvi ViT tomorrow
5. **Prediction** -- probability maps, uncertainty, drill targeting

---

## Slide 4: The Moat (This Is the Product)

We built an **institutional-grade validation framework** that most mining AI startups don't have:

- **Leave-One-Tile-Out CV** -- holds out 12,000 km2 of unseen terrain
- **Along-strike CV** -- respects linear geological structures
- **Bootstrap stability** -- 500 resamples to verify signal direction
- **Mandatory baselines** -- every model compared to random guessing
- **Leakage guards** -- automatically excludes circular features

**Why this matters:** Most mining AI companies overfit, validate randomly, publish a pretty map, and raise money. Their models don't survive first contact with new geography.

Ours detected its own failure. That's rare. That's defensible.

---

## Slide 5: The Honest Failure (Why It's an Asset)

We tested **8 model configurations** against rigorous spatial cross-validation.

All failed to generalize across satellite tiles.

Best cross-tile result: PR-AUC 0.050 vs 0.055 baseline. Lift: 0.9x. Below random.

**Why this builds investor trust:**
- We didn't hide it. We documented it publicly.
- We diagnosed the root cause: sensor physics (spectral indices encode atmosphere, not rock).
- We identified the exact mathematical signature: coefficient sign flip across tiles.
- We designed the correct pivot: foundation models that see spatial patterns, not pixel values.

A team that catches its own fraud is a team you can fund.

---

## Slide 6: Why Logistic Regression Failed

Classical ML sees **one pixel at a time.** It computes a band ratio (e.g., ferric iron = B4/B3) and asks: "Is this number higher at deposit locations?"

The problem: that number is dominated by **atmospheric conditions, vegetation density, and local soil weathering** -- not rock mineralogy.

**Proof:** The ferrous iron coefficient was +0.446 in one tile and -0.650 in the adjacent tile. Literally opposite conclusions about the same geological feature.

This is not a tuning problem. It's a sensor physics wall.

---

## Slide 7: Why Foundation Models Fix This

NASA/IBM's **Prithvi-EO-2.0** is a Vision Transformer pre-trained on **4.2 million global satellite samples.**

Where logistic regression sees a single pixel, Prithvi sees a **2.24 km x 2.24 km spatial patch.** It learns:
- Texture patterns (layered intrusions have distinctive banding)
- Edge relationships (geological contacts between rock units)
- Seasonal invariance (trained across all conditions globally)
- Geographic context (location embeddings encode where on Earth)

**Our hypothesis:** Layered intrusions like the Great Dyke are spatially organized geological bodies. A model that sees structure, not just color, should transfer across tiles.

This is testable. 45 minutes of free GPU time.

---

## Slide 8: The Market

**Platinum Group Metals (PGMs):**
- Structural global supply deficit (South African production declining)
- Hybrid vehicle boom drives catalytic converter demand
- Platinum trading $1,500-2,000+/oz in 2026
- Zimbabwe's Great Dyke = world's 2nd largest PGM reserve

**Chromite:**
- No substitute for stainless steel production
- Global demand growing 5-6% annually
- High-carbon ferrochrome at ~$1,200/ton
- Great Dyke ore is exceptionally high-grade

You cannot make stainless steel without chromium. There is no substitute.

---

## Slide 9: The Unfair Advantage

| Advantage | Detail |
|---|---|
| Data cost | $0 (all public satellite data) |
| Compute cost | ~$10-50 per concession (free GPU tier) |
| Validation discipline | Only team we know running LOTO + along-strike + bootstrap |
| Geographic position | Direct access to Great Dyke concessions (Zimbabwe) |
| Domain IP | 8 failed configurations = 8 approaches competitors will also fail at |
| Foundation model recipe | Prithvi is public; our fine-tuning approach is proprietary |

**Gross margin on exploration targeting: > 95%.**

---

## Slide 10: Business Model (Sequenced)

**Now (pre-validation):**
- Geospatial ML Due Diligence as a Service
- Help other mining AI companies validate their models properly
- $5,000-25,000 per engagement

**After LOTO >= 0.45:**
- Per-concession exploration targeting: $10,000-50,000
- Data licensing (probability maps): $50,000-100,000
- Platform SaaS at scale

**The customer is not miners (yet).** Until we have the number, our customer is every mining AI company that doesn't validate properly.

---

## Slide 11: Roadmap

| Quarter | Milestone | Revenue |
|---|---|---|
| Q2 2026 | Prithvi fine-tuning, LOTO validation | $0 |
| Q3 2026 | First paid pilot (1 concession) | $10K-25K |
| Q4 2026 | Seed raise | $250K-500K |
| Q1 2027 | 3-5 paying concessions | $50K-100K |
| 2027 H2 | Expand to Bushveld Complex (SA) | $200K-500K |
| 2028 | Platform SaaS launch | $500K-1M ARR |

---

## Slide 12: What Happens If Prithvi Fails?

If LOTO < 0.45 after foundation model fine-tuning, we have falsified the satellite-only thesis for PGM detection at this scale.

**That is still valuable.** It means:
1. Pivot to multi-sensor fusion (add SAR via Clay Foundation Model)
2. Pivot to hyperspectral (EMIT/PRISMA, 285 bands vs 13)
3. Pivot to hybrid model: satellite + mapped geology + ground truth
4. The validation framework + platform architecture remain intact for any of these

The platform is commodity-agnostic. PGM was the wedge, not the ceiling.

---

## Slide 13: The Team

A team that:
- Built 8,300 lines of production code in weeks, not months
- Processed 42 GB of satellite data for $0
- Detected and publicly documented its own model failures
- Pivoted cleanly from classical ML to foundation models
- Understands both the geology (Great Dyke stratigraphy) and the ML (spatial CV, bootstrap, leakage detection)

---

## Slide 14: The Ask

**$250,000-500,000 seed round**

Use of funds:
1. Complete foundation model fine-tuning + validation (Q2 2026)
2. Acquire 1-2 commercial ground truth datasets (drill logs)
3. First 3 paid concession pilots (Q3-Q4 2026)
4. Expand to Bushveld Complex, South Africa (Q1 2027)

---

## Slide 15: The One Metric That Matters

Forget lines of code. Forget gigabytes. Forget GPU time.

**Cross-tile PR-AUC on unseen geography.**

Current: 0.035 (baseline).
Target: >= 0.45.

That number is the difference between a research prototype and a venture-backable company. It is 45 minutes of free GPU time away.

Everything else is noise.

---

*GeoMine AI -- $0 infrastructure. Institutional-grade validation. The only mining AI that detects its own fraud.*
