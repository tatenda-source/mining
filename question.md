# Questions for Architecture Review

## For the Solutions Architect

**Data pipeline & scale:**

1. "We expect 200-500GB of satellite rasters per region, with maybe 5-10 regions in Phase 1. The pipeline runs weekly: download -> preprocess -> store as COG -> index in STAC. Is a full orchestration framework (Airflow/Prefect/Dagster) justified, or would a well-structured Python CLI + cron be enough until we hit scale?"

2. "If 10 geologists are simultaneously browsing prediction layers at zoom levels 8-14 via TiTiler, what part fails first -- CPU, I/O, or storage? How would we detect that early?"

3. "If we're under 10 million vector rows in PostGIS and mostly doing bounding-box queries + spatial joins on training labels and predictions, is managed PostGIS overkill for Phase 1? Or is Dockerized local fine?"

**Architecture decisions:**

4. "What concrete pain would force us to split ingest, ML, and API into separate services -- scaling independently, team parallelism, or failure isolation? We're a monorepo Python package right now."

5. "The ML inference runs on GPU but the web dashboard is lightweight. How would you deploy these on different compute tiers without overcomplicating Phase 1?"

---

## For Senior Engineers

**Geospatial-specific traps:**

6. "If you were reviewing a PR that processes satellite rasters end-to-end, what bugs would immediately scare you -- CRS mismatch, nodata propagation, resampling artifacts, reprojection distortion?"

7. "We're targeting mineral systems with ~5-20km spatial continuity. For spatial block cross-validation, does 10km blocking make sense, or should block size relate to spatial autocorrelation length?"

8. "If we upsample 20m/60m Sentinel-2 bands to 10m for feature stacking, are we fabricating signal that the model will overfit to? What resampling strategy preserves signal integrity?"

**ML pipeline:**

9. "If we compute SHAP on ~5 million pixels from XGBoost predictions, do we need to move to approximate TreeSHAP, downsample explanation grids, or pre-aggregate to tiles?"

10. "For uncertainty quantification, would quantile regression or Monte Carlo dropout analogs be cheaper than our current plan of 20 bootstrap XGBoost ensembles? What gives honest uncertainty without 20x inference cost?"

---

## The Question That Matters Most

11. "If we had 3 weeks and no production infra, what would a technically honest mineral prospectivity demo look like that wouldn't embarrass us in front of a geologist? Jupyter notebook + QGIS files -- not a product."

> This forces brutal scope-cutting. Their answer reveals what's scientifically essential vs engineering polish vs premature optimization.

---

## The Question You Should Not Skip

12. "Where are we most likely to be scientifically wrong, even if the engineering works perfectly?"

> In mineral prediction: engineering failure is fixable. Scientific misinterpretation is catastrophic. If they struggle to answer this, that's a signal.
