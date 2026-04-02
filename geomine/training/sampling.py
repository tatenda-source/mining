"""Training data assembly with exploration-bias-aware sampling.

This module handles the critical step of constructing training datasets for
mineral prospectivity modelling.  It addresses *exploration bias* -- the
tendency for known deposits to cluster in well-explored areas -- by weighting
negative sample selection toward regions that have genuinely been explored and
found barren, rather than simply under-explored.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import rowcol
from scipy.ndimage import uniform_filter
from shapely.geometry import Point, box, mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_at_points(
    points_gdf: gpd.GeoDataFrame,
    feature_rasters: list[str | Path],
    feature_names: list[str],
) -> pd.DataFrame:
    """Sample all feature rasters at the locations given by *points_gdf*.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Point geometries.  Must have a ``geometry`` column.
    feature_rasters : list of str or Path
        Ordered list of single-band raster paths (one per feature).
    feature_names : list of str
        Human-readable feature names corresponding to *feature_rasters*.

    Returns
    -------
    DataFrame
        Columns ``[x, y, <feature_names...>]``.  Rows where **any** critical
        feature is nodata are dropped.
    """
    if len(feature_rasters) != len(feature_names):
        raise ValueError(
            f"feature_rasters ({len(feature_rasters)}) and feature_names "
            f"({len(feature_names)}) must have the same length"
        )

    coords = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
    result: dict[str, np.ndarray] = {
        "x": coords[:, 0],
        "y": coords[:, 1],
    }

    for raster_path, name in zip(feature_rasters, feature_names):
        raster_path = Path(raster_path)
        logger.debug("Sampling %s at %d points", raster_path.name, len(coords))

        with rasterio.open(raster_path) as src:
            nodata = src.nodata
            sampled = np.array(
                list(src.sample(coords, indexes=1)),
                dtype=np.float64,
            ).ravel()
            if nodata is not None:
                sampled[sampled == nodata] = np.nan

        result[name] = sampled

    df = pd.DataFrame(result)
    n_before = len(df)
    df = df.dropna(subset=feature_names).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "Dropped %d / %d samples with nodata features", n_dropped, n_before
        )

    return df


# ---------------------------------------------------------------------------
# Exploration intensity
# ---------------------------------------------------------------------------

def compute_exploration_intensity(
    aoi_geometry: Any,
    roads_gdf: gpd.GeoDataFrame | None,
    deposits_gdf: gpd.GeoDataFrame,
    raster_template_path: str | Path,
    output_path: str | Path,
    radius_km: float = 50.0,
) -> Path:
    """Compute a per-pixel exploration-intensity proxy and save as GeoTIFF.

    The intensity is based on the density of known deposit records within a
    given radius.  If a roads GeoDataFrame is supplied, distance-to-nearest-
    road is also incorporated (lower distance = higher intensity).

    The result is normalised to the 0 -- 1 range.

    Parameters
    ----------
    aoi_geometry : shapely geometry
        Area of interest polygon.
    roads_gdf : GeoDataFrame or None
        Road network lines (optional).
    deposits_gdf : GeoDataFrame
        Known deposit point locations (projected CRS).
    raster_template_path : str or Path
        Any raster whose grid (shape, transform, CRS) should be reused for
        the output.
    output_path : str or Path
        Destination GeoTIFF path.
    radius_km : float
        Search radius in kilometres for deposit density calculation.

    Returns
    -------
    Path
        The *output_path* that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_template_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

    logger.info(
        "Computing exploration intensity (%d x %d grid, radius=%.0f km)",
        width, height, radius_km,
    )

    # --- Deposit density layer ---
    # Create a binary raster of deposit presence, then smooth with a
    # uniform filter whose kernel approximates the search radius.
    deposit_coords = np.column_stack([
        deposits_gdf.geometry.x, deposits_gdf.geometry.y
    ])

    deposit_raster = np.zeros((height, width), dtype=np.float32)
    for x, y in deposit_coords:
        try:
            r, c = rowcol(transform, x, y)
        except Exception:
            continue
        if 0 <= r < height and 0 <= c < width:
            deposit_raster[r, c] += 1.0

    # Pixel size in metres (assume square pixels)
    pixel_size_m = abs(transform.a)
    kernel_pixels = max(1, int((radius_km * 1000) / pixel_size_m))
    density = uniform_filter(deposit_raster, size=kernel_pixels * 2 + 1)

    # Normalise density to 0-1
    d_min, d_max = float(np.nanmin(density)), float(np.nanmax(density))
    if d_max > d_min:
        intensity = (density - d_min) / (d_max - d_min)
    else:
        intensity = np.ones_like(density, dtype=np.float32)

    # --- Road proximity (optional) ---
    if roads_gdf is not None and not roads_gdf.empty:
        logger.info("Incorporating road proximity into exploration intensity")
        from rasterio.features import rasterize
        from shapely.ops import unary_union

        road_geom = unary_union(roads_gdf.geometry)
        # Rasterise distance (approximate: use buffered road binary)
        road_mask = rasterize(
            [(mapping(road_geom), 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        from scipy.ndimage import distance_transform_edt

        distance_to_road = distance_transform_edt(road_mask == 0) * pixel_size_m
        # Normalise: closer to road = higher intensity
        max_dist = float(np.nanmax(distance_to_road))
        if max_dist > 0:
            road_intensity = 1.0 - (distance_to_road / max_dist)
        else:
            road_intensity = np.ones_like(distance_to_road)

        # Combine: average of deposit density and road proximity
        intensity = (intensity.astype(np.float32) + road_intensity.astype(np.float32)) / 2.0

    intensity = intensity.astype(np.float32)

    # Mask pixels outside AOI
    aoi_mask = geometry_mask(
        [mapping(aoi_geometry)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )
    nodata_val = -9999.0
    intensity[~aoi_mask] = nodata_val

    # Write output
    out_profile = profile.copy()
    out_profile.update(
        dtype="float32",
        count=1,
        nodata=nodata_val,
        compress="lzw",
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(intensity, 1)

    logger.info("Exploration intensity raster saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Negative sample generation
# ---------------------------------------------------------------------------

def generate_negative_samples(
    deposits_gdf: gpd.GeoDataFrame,
    aoi_geometry: Any,
    config: dict[str, Any],
    exploration_intensity_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Generate exploration-bias-aware negative (barren) training samples.

    Parameters
    ----------
    deposits_gdf : GeoDataFrame
        Known deposit locations.
    aoi_geometry : shapely geometry
        Area of interest polygon (projected CRS).
    config : dict
        Full project config dictionary.
    exploration_intensity_path : str, Path, or None
        Path to the exploration-intensity raster.  If ``None``, falls back to
        random stratified sampling within the AOI.

    Returns
    -------
    GeoDataFrame
        Negative sample points with a ``label`` column set to ``0``.
    """
    neg_cfg = config["training"]["negative_sampling"]
    buffer_m = neg_cfg["deposit_buffer_km"] * 1000.0
    ratio = neg_cfg["neg_to_pos_ratio"]
    n_positives = len(deposits_gdf)
    n_negatives = n_positives * ratio

    logger.info(
        "Generating %d negative samples (ratio %d:1, buffer %.0f m)",
        n_negatives, ratio, buffer_m,
    )

    # Build exclusion zone around known deposits
    if not deposits_gdf.empty:
        exclusion_zone = deposits_gdf.geometry.buffer(buffer_m).union_all()
        sampling_area = aoi_geometry.difference(exclusion_zone)
    else:
        sampling_area = aoi_geometry

    if sampling_area.is_empty:
        logger.warning(
            "Exclusion buffers cover the entire AOI; reducing buffer to 1 km"
        )
        exclusion_zone = deposits_gdf.geometry.buffer(1000.0).union_all()
        sampling_area = aoi_geometry.difference(exclusion_zone)

    # --- Exploration-aware sampling ---
    if exploration_intensity_path is not None:
        exploration_intensity_path = Path(exploration_intensity_path)
        if exploration_intensity_path.exists():
            logger.info(
                "Using exploration intensity raster for bias-aware sampling"
            )
            return _sample_with_intensity(
                sampling_area,
                exploration_intensity_path,
                n_negatives,
                deposits_gdf.crs,
            )

    # --- Fallback: random stratified sampling ---
    logger.info("Falling back to random stratified sampling within AOI")
    return _random_stratified_sample(
        sampling_area, n_negatives, deposits_gdf.crs
    )


def _sample_with_intensity(
    sampling_area: Any,
    intensity_path: Path,
    n_samples: int,
    crs: Any,
    intensity_threshold: float = 0.2,
) -> gpd.GeoDataFrame:
    """Sample negative points weighted by exploration intensity."""
    with rasterio.open(intensity_path) as src:
        intensity = src.read(1)
        nodata = src.nodata
        transform = src.transform

    # Build candidate mask: inside sampling area + intensity > threshold
    height, width = intensity.shape
    aoi_mask = geometry_mask(
        [mapping(sampling_area)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
    )

    valid = aoi_mask.copy()
    if nodata is not None:
        valid &= intensity != nodata
    valid &= intensity >= intensity_threshold

    rows, cols = np.where(valid)
    if len(rows) == 0:
        logger.warning(
            "No pixels above intensity threshold; sampling from all AOI pixels"
        )
        rows, cols = np.where(aoi_mask)

    if len(rows) == 0:
        raise ValueError("No valid pixels found for negative sampling")

    # Weight by intensity
    weights = intensity[rows, cols].astype(np.float64)
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total > 0:
        probs = weights / total
    else:
        probs = np.ones(len(weights)) / len(weights)

    n_samples = min(n_samples, len(rows))
    chosen = np.random.default_rng(42).choice(
        len(rows), size=n_samples, replace=False, p=probs
    )

    xs, ys = rasterio.transform.xy(transform, rows[chosen], cols[chosen])
    points = [Point(x, y) for x, y in zip(xs, ys)]

    gdf = gpd.GeoDataFrame(
        {"label": np.zeros(len(points), dtype=int)},
        geometry=points,
        crs=crs,
    )
    logger.info("Generated %d exploration-bias-aware negative samples", len(gdf))
    return gdf


def _random_stratified_sample(
    sampling_area: Any,
    n_samples: int,
    crs: Any,
) -> gpd.GeoDataFrame:
    """Random stratified sampling within a polygon geometry."""
    rng = np.random.default_rng(42)
    minx, miny, maxx, maxy = sampling_area.bounds

    points: list[Point] = []
    attempts = 0
    max_attempts = n_samples * 50

    while len(points) < n_samples and attempts < max_attempts:
        batch_size = min((n_samples - len(points)) * 5, 10000)
        xs = rng.uniform(minx, maxx, size=batch_size)
        ys = rng.uniform(miny, maxy, size=batch_size)

        for x, y in zip(xs, ys):
            pt = Point(x, y)
            if sampling_area.contains(pt):
                points.append(pt)
                if len(points) >= n_samples:
                    break
        attempts += batch_size

    if len(points) < n_samples:
        logger.warning(
            "Could only generate %d / %d negative samples within AOI",
            len(points), n_samples,
        )

    gdf = gpd.GeoDataFrame(
        {"label": np.zeros(len(points), dtype=int)},
        geometry=points,
        crs=crs,
    )
    logger.info("Generated %d random stratified negative samples", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Training data preparation
# ---------------------------------------------------------------------------

def prepare_training_data(
    deposits_gdf: gpd.GeoDataFrame,
    negative_samples_gdf: gpd.GeoDataFrame,
    feature_rasters: list[str | Path],
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Assemble the complete training dataset.

    Parameters
    ----------
    deposits_gdf : GeoDataFrame
        Known deposit locations (positive class).
    negative_samples_gdf : GeoDataFrame
        Barren sample locations (negative class), e.g. from
        :func:`generate_negative_samples`.
    feature_rasters : list of str or Path
        Ordered feature raster paths.
    feature_names : list of str
        Feature names matching *feature_rasters*.
    config : dict
        Full project config dictionary.

    Returns
    -------
    X : DataFrame
        Feature matrix (rows = samples, columns = feature names).
    y : ndarray
        Label vector (1 = deposit, 0 = barren).
    metadata : dict
        Coordinates and sample provenance information.
    """
    # Tag positives
    positives = deposits_gdf.copy()
    positives["label"] = 1

    # Ensure negatives have label column
    negatives = negative_samples_gdf.copy()
    if "label" not in negatives.columns:
        negatives["label"] = 0

    combined = gpd.GeoDataFrame(
        pd.concat([positives[["geometry", "label"]], negatives[["geometry", "label"]]],
                  ignore_index=True),
        crs=deposits_gdf.crs,
    )

    logger.info(
        "Combined training set: %d positives, %d negatives",
        int((combined["label"] == 1).sum()),
        int((combined["label"] == 0).sum()),
    )

    df = extract_features_at_points(combined, feature_rasters, feature_names)

    if df.empty:
        raise ValueError(
            "Training dataset is empty after dropping nodata rows. "
            "Check that feature rasters cover the sample locations."
        )

    # Attach labels -- extract_features_at_points returns rows aligned to
    # the input (after nodata drops), but we need the label from the
    # combined GDF.  Since extract drops rows, we must track indices.
    # Rebuild: re-extract with labels attached.
    combined_with_coords = combined.copy()
    combined_with_coords["x"] = combined.geometry.x
    combined_with_coords["y"] = combined.geometry.y

    # Merge labels back by coordinate match
    df = df.merge(
        combined_with_coords[["x", "y", "label"]],
        on=["x", "y"],
        how="left",
    )
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    X = df[feature_names]
    y = df["label"].astype(int).values

    metadata = {
        "x": df["x"].values,
        "y": df["y"].values,
        "n_positives": int((y == 1).sum()),
        "n_negatives": int((y == 0).sum()),
        "feature_names": feature_names,
        "crs": str(deposits_gdf.crs),
    }

    logger.info(
        "Training data ready: %d samples, %d features, %d positives, %d negatives",
        len(X), len(feature_names), metadata["n_positives"], metadata["n_negatives"],
    )

    return X, y, metadata
