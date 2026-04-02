"""Prediction, target clustering, and reporting for mineral prospectivity.

This module applies a trained prospectivity model to raster feature stacks,
identifies high-probability exploration targets, and generates an HTML report
summarising the results.
"""

from __future__ import annotations

import base64
import json
import logging
from html import escape
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rasterio_shapes
from rasterio.windows import Window
from scipy.ndimage import label as ndimage_label
from shapely.geometry import Polygon, shape

logger = logging.getLogger(__name__)

# Maximum number of rows to process per chunk when predicting on large rasters
_DEFAULT_CHUNK_ROWS = 512


# ---------------------------------------------------------------------------
# Raster prediction
# ---------------------------------------------------------------------------

def predict_raster(
    model: Any,
    feature_stack_path: str | Path,
    output_probability_path: str | Path,
    output_uncertainty_path: str | Path,
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """Run model prediction on every pixel of a stacked feature raster.

    The raster is read in horizontal strips (windows) so that arbitrarily
    large files can be processed without exhausting memory.

    Uncertainty is estimated via bootstrap: *N* models are trained on
    resampled versions of the original training data, and the standard
    deviation of their probability predictions is taken as pixel-level
    uncertainty.

    Parameters
    ----------
    model : fitted estimator
        Must expose ``.predict_proba()``.
    feature_stack_path : str or Path
        Multi-band GeoTIFF where each band is one feature (order must
        match the model's training features).
    output_probability_path : str or Path
        Destination GeoTIFF for the probability surface (0 -- 100).
    output_uncertainty_path : str or Path
        Destination GeoTIFF for the uncertainty surface (0 -- 100).
    config : dict
        Full project config.

    Returns
    -------
    tuple of Path
        ``(probability_path, uncertainty_path)``
    """
    feature_stack_path = Path(feature_stack_path)
    output_probability_path = Path(output_probability_path)
    output_uncertainty_path = Path(output_uncertainty_path)
    output_probability_path.parent.mkdir(parents=True, exist_ok=True)
    output_uncertainty_path.parent.mkdir(parents=True, exist_ok=True)

    n_bootstrap = config.get("model", {}).get("uncertainty", {}).get("n_estimators", 20)

    with rasterio.open(feature_stack_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        n_bands = src.count
        nodata_in = src.nodata
        transform = src.transform

    logger.info(
        "Predicting on raster %s (%d x %d, %d bands)",
        feature_stack_path.name, width, height, n_bands,
    )

    nodata_out = -9999.0
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=nodata_out,
        compress="lzw",
    )

    # --- Bootstrap models for uncertainty ---
    bootstrap_models = _build_bootstrap_models(model, n_bootstrap)

    # --- Chunked prediction ---
    chunk_rows = _DEFAULT_CHUNK_ROWS

    with (
        rasterio.open(feature_stack_path) as src,
        rasterio.open(output_probability_path, "w", **out_profile) as dst_prob,
        rasterio.open(output_uncertainty_path, "w", **out_profile) as dst_unc,
    ):
        for row_start in range(0, height, chunk_rows):
            n_rows = min(chunk_rows, height - row_start)
            window = Window(0, row_start, width, n_rows)

            # Read all bands for this strip: shape (n_bands, n_rows, width)
            data = src.read(window=window).astype(np.float32)

            # Reshape to (n_pixels, n_bands)
            pixels = data.reshape(n_bands, -1).T  # (n_pixels, n_bands)

            # Build nodata mask
            nodata_mask = np.zeros(pixels.shape[0], dtype=bool)
            if nodata_in is not None:
                nodata_mask |= np.any(pixels == nodata_in, axis=1)
            nodata_mask |= np.any(np.isnan(pixels), axis=1)

            prob_strip = np.full(pixels.shape[0], nodata_out, dtype=np.float32)
            unc_strip = np.full(pixels.shape[0], nodata_out, dtype=np.float32)

            valid_idx = ~nodata_mask
            if valid_idx.sum() > 0:
                valid_pixels = pixels[valid_idx]

                # Primary probability
                prob_pred = model.predict_proba(valid_pixels)[:, 1]
                prob_strip[valid_idx] = (prob_pred * 100.0).astype(np.float32)

                # Bootstrap uncertainty
                if bootstrap_models:
                    boot_preds = np.column_stack([
                        bm.predict_proba(valid_pixels)[:, 1]
                        for bm in bootstrap_models
                    ])
                    unc_pred = boot_preds.std(axis=1)
                    unc_strip[valid_idx] = (unc_pred * 100.0).astype(np.float32)
                else:
                    unc_strip[valid_idx] = 0.0

            dst_prob.write(
                prob_strip.reshape(n_rows, width), 1, window=window
            )
            dst_unc.write(
                unc_strip.reshape(n_rows, width), 1, window=window
            )

    logger.info("Probability raster saved: %s", output_probability_path)
    logger.info("Uncertainty raster saved: %s", output_uncertainty_path)
    return output_probability_path, output_uncertainty_path


def _build_bootstrap_models(
    model: Any,
    n_bootstrap: int,
) -> list[Any]:
    """Train bootstrap resampled copies of the model for uncertainty estimation.

    If the model does not have stored training data (``_train_X``, ``_train_y``
    attributes), we cannot build bootstraps and return an empty list.  The
    calling pipeline should attach these attributes after training if
    bootstrap uncertainty is desired.
    """
    from sklearn.base import clone

    X_train = getattr(model, "_train_X", None)
    y_train = getattr(model, "_train_y", None)

    if X_train is None or y_train is None:
        logger.warning(
            "Model lacks _train_X / _train_y attributes; "
            "skipping bootstrap uncertainty (set them after training)"
        )
        return []

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n_samples = len(y_train)

    logger.info("Training %d bootstrap models for uncertainty estimation", n_bootstrap)
    rng = np.random.default_rng(42)
    models: list[Any] = []

    for i in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        bm = clone(model)
        # Suppress verbose for bootstrap runs
        if hasattr(bm, "verbose"):
            bm.set_params(verbose=0)
        if hasattr(bm, "early_stopping_rounds"):
            bm.set_params(early_stopping_rounds=None)
        bm.fit(X_train[idx], y_train[idx])
        models.append(bm)

    logger.info("Bootstrap model training complete")
    return models


# ---------------------------------------------------------------------------
# Target clustering
# ---------------------------------------------------------------------------

def cluster_targets(
    probability_path: str | Path,
    uncertainty_path: str | Path,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Identify contiguous high-probability target zones.

    Parameters
    ----------
    probability_path : str or Path
        Probability raster (values 0 -- 100).
    uncertainty_path : str or Path
        Uncertainty raster (values 0 -- 100).
    config : dict
        Full project config.  Uses ``output.threshold`` and
        ``output.min_cluster_area_km2``.

    Returns
    -------
    GeoDataFrame
        One row per target zone with columns ``[geometry, rank, area_km2,
        mean_probability, mean_uncertainty, centroid_lat, centroid_lon]``.
        Also saved as GeoJSON to ``config["output"]["targets_geojson"]``.
    """
    probability_path = Path(probability_path)
    uncertainty_path = Path(uncertainty_path)

    threshold = config["output"].get("threshold", 0.7) * 100.0  # scale to raster range
    min_area_km2 = config["output"].get("min_cluster_area_km2", 0.5)

    with rasterio.open(probability_path) as src:
        prob = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        pixel_size_m = abs(transform.a)

    with rasterio.open(uncertainty_path) as src:
        unc = src.read(1)

    logger.info(
        "Clustering targets: threshold=%.0f%%, min_area=%.1f km2",
        threshold, min_area_km2,
    )

    # Binary mask: above threshold and not nodata
    mask = prob >= threshold
    if nodata is not None:
        mask &= prob != nodata

    if not mask.any():
        logger.warning("No pixels above threshold -- returning empty GeoDataFrame")
        return gpd.GeoDataFrame(
            columns=[
                "geometry", "rank", "area_km2", "mean_probability",
                "mean_uncertainty", "centroid_lat", "centroid_lon",
            ],
            crs=crs,
        )

    # Label contiguous regions
    labeled, n_clusters = ndimage_label(mask)
    logger.info("Found %d raw contiguous regions", n_clusters)

    pixel_area_km2 = (pixel_size_m ** 2) / 1e6
    records: list[dict[str, Any]] = []

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled == cluster_id
        n_pixels = int(cluster_mask.sum())
        area_km2 = n_pixels * pixel_area_km2

        if area_km2 < min_area_km2:
            continue

        mean_prob = float(prob[cluster_mask].mean())
        mean_unc = float(unc[cluster_mask].mean()) if nodata is not None else 0.0

        # Extract polygon geometry via rasterio.features.shapes
        cluster_binary = (cluster_mask).astype(np.uint8)
        geom_gen = rasterio_shapes(cluster_binary, mask=cluster_mask, transform=transform)
        polygons = [shape(geom) for geom, val in geom_gen if val == 1]

        if not polygons:
            continue

        from shapely.ops import unary_union
        merged = unary_union(polygons)

        # Compute centroid in WGS84 for human-readable lat/lon
        centroid_proj = merged.centroid
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(centroid_proj.x, centroid_proj.y)
        except Exception:
            lon, lat = centroid_proj.x, centroid_proj.y

        records.append({
            "geometry": merged,
            "area_km2": round(area_km2, 2),
            "mean_probability": round(mean_prob, 1),
            "mean_uncertainty": round(mean_unc, 1),
            "centroid_lat": round(lat, 5),
            "centroid_lon": round(lon, 5),
        })

    if not records:
        logger.warning("No target clusters meet minimum area requirement")
        return gpd.GeoDataFrame(
            columns=[
                "geometry", "rank", "area_km2", "mean_probability",
                "mean_uncertainty", "centroid_lat", "centroid_lon",
            ],
            crs=crs,
        )

    # Rank by mean probability descending
    records.sort(key=lambda r: r["mean_probability"], reverse=True)
    for rank, rec in enumerate(records, start=1):
        rec["rank"] = rank

    targets_gdf = gpd.GeoDataFrame(records, crs=crs)

    # Reorder columns
    col_order = [
        "geometry", "rank", "area_km2", "mean_probability",
        "mean_uncertainty", "centroid_lat", "centroid_lon",
    ]
    targets_gdf = targets_gdf[col_order]

    # Save as GeoJSON
    base_dir = Path(config.get("_base_dir", "."))
    geojson_path = base_dir / config["output"].get("targets_geojson", "data/outputs/targets.geojson")
    geojson_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to WGS84 for GeoJSON standard
    targets_wgs84 = targets_gdf.to_crs("EPSG:4326")
    targets_wgs84.to_file(geojson_path, driver="GeoJSON")
    logger.info("Target clusters saved: %s (%d targets)", geojson_path, len(targets_gdf))

    return targets_gdf


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_report(
    targets_gdf: gpd.GeoDataFrame,
    shap_results: tuple[np.ndarray, list[tuple[str, float]]] | None,
    cv_metrics: dict[str, Any],
    config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Generate a self-contained HTML report summarising the analysis.

    Parameters
    ----------
    targets_gdf : GeoDataFrame
        Target zones from :func:`cluster_targets`.
    shap_results : tuple or None
        ``(shap_values, feature_importance)`` from
        :func:`~geomine.training.train.compute_shap_analysis`.
    cv_metrics : dict
        Cross-validation results dict.
    config : dict
        Full project config.
    output_path : str or Path
        Destination HTML file path.

    Returns
    -------
    Path
        The written HTML file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_name = config.get("project", {}).get("name", "GeoMine AI")
    description = config.get("project", {}).get("description", "")

    # --- Metrics summary ---
    mean_m = cv_metrics.get("mean", {})
    pr_auc = mean_m.get("pr_auc", float("nan"))
    roc_auc = mean_m.get("roc_auc", float("nan"))
    f1 = mean_m.get("f1_at_0.5", float("nan"))

    # --- Feature importance ---
    feat_rows_html = ""
    if shap_results is not None:
        _, importance = shap_results
        for rank, (name, val) in enumerate(importance[:15], start=1):
            feat_rows_html += (
                f"<tr><td>{rank}</td><td>{escape(name)}</td>"
                f"<td>{val:.4f}</td></tr>\n"
            )

    # --- Targets table ---
    target_rows_html = ""
    if not targets_gdf.empty:
        for _, row in targets_gdf.iterrows():
            target_rows_html += (
                f"<tr>"
                f"<td>{row.get('rank', '')}</td>"
                f"<td>{row.get('area_km2', '')}</td>"
                f"<td>{row.get('mean_probability', '')}</td>"
                f"<td>{row.get('mean_uncertainty', '')}</td>"
                f"<td>{row.get('centroid_lat', '')}</td>"
                f"<td>{row.get('centroid_lon', '')}</td>"
                f"</tr>\n"
            )

    # --- Embedded SHAP plot ---
    shap_img_html = ""
    base_dir = Path(config.get("_base_dir", "."))
    shap_png = base_dir / config.get("output", {}).get("shap_summary", "data/outputs/shap_summary.png")
    if shap_png.exists():
        with open(shap_png, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        shap_img_html = (
            f'<h2>SHAP Feature Importance</h2>\n'
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="SHAP Summary" style="max-width:100%;"/>\n'
        )

    # --- GeoJSON link ---
    geojson_rel = config.get("output", {}).get("targets_geojson", "data/outputs/targets.geojson")
    geojson_note = (
        f'<p>Target polygons saved as <code>{escape(geojson_rel)}</code> '
        f'&mdash; open in QGIS or <a href="https://geojson.io">geojson.io</a> '
        f'for interactive viewing.</p>'
    )

    # --- Assemble HTML ---
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>GeoMine AI Report &mdash; {escape(project_name)}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2em; color: #222; }}
  h1 {{ color: #1a5276; }}
  h2 {{ color: #2e86c1; border-bottom: 1px solid #aed6f1; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; }}
  th, td {{ border: 1px solid #d5dbdb; padding: 6px 10px; text-align: left; }}
  th {{ background: #eaf2f8; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .metric {{ display: inline-block; margin: 0 1.5em 1em 0; padding: 0.8em 1.2em;
             background: #eaf2f8; border-radius: 6px; text-align: center; }}
  .metric .value {{ font-size: 1.6em; font-weight: bold; color: #1a5276; }}
  .metric .label {{ font-size: 0.85em; color: #555; }}
  .warn {{ color: #b7950b; font-weight: bold; }}
</style>
</head>
<body>
<h1>GeoMine AI &mdash; Mineral Prospectivity Report</h1>
<p><strong>Project:</strong> {escape(project_name)}<br/>
   <strong>Description:</strong> {escape(description)}</p>

<h2>Model Performance (Spatial Block CV)</h2>
<div>
  <div class="metric"><div class="value">{pr_auc:.3f}</div><div class="label">PR-AUC</div></div>
  <div class="metric"><div class="value">{roc_auc:.3f}</div><div class="label">ROC-AUC</div></div>
  <div class="metric"><div class="value">{f1:.3f}</div><div class="label">F1 @ 0.5</div></div>
</div>

{shap_img_html}

{"<h2>Top Features (SHAP)</h2>" if feat_rows_html else ""}
{"<table><tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th></tr>" + feat_rows_html + "</table>" if feat_rows_html else ""}

<h2>Exploration Targets</h2>
{geojson_note}
<table>
<tr><th>Rank</th><th>Area (km&sup2;)</th><th>Mean Prob (%)</th><th>Mean Unc (%)</th><th>Lat</th><th>Lon</th></tr>
{target_rows_html if target_rows_html else "<tr><td colspan='6'>No targets identified above threshold.</td></tr>"}
</table>

<hr/>
<p style="font-size:0.8em; color:#888;">
  Generated by GeoMine AI v0.1.0
</p>
</body>
</html>
"""

    with open(output_path, "w") as fh:
        fh.write(html)

    logger.info("Report saved: %s", output_path)
    return output_path
