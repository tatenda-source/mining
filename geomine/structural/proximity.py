"""Proximity and distance-based feature computation for mineral prospectivity.

Computes Euclidean distance to geological features, buffered density counts,
and drainage density from DEM-derived flow accumulation.  These rasters
serve as spatial predictors in mineral prospectivity models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Point, MultiPoint, MultiLineString

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_raster(
    data: np.ndarray,
    profile: dict,
    output_path: str | Path,
    nodata: float = -9999.0,
) -> Path:
    """Write a 2-D float32 array as a single-band GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = data.copy()
    mask = np.isnan(out)
    out[mask] = nodata

    write_profile = profile.copy()
    write_profile.update(
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(out.astype(np.float32), 1)

    logger.info("Wrote raster %s", output_path)
    return output_path


def _rasterize_features(
    features_gdf: gpd.GeoDataFrame,
    height: int,
    width: int,
    transform: Affine,
) -> np.ndarray:
    """Burn vector features into a boolean raster (True = feature present).

    For point features each point occupies the pixel it falls in.
    For line features all pixels touched by the line are burned.
    """
    from rasterio.features import rasterize as rio_rasterize
    from shapely.geometry import mapping

    if features_gdf.empty:
        return np.zeros((height, width), dtype=bool)

    shapes = [(mapping(geom), 1) for geom in features_gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=bool)

    burned = rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return burned.astype(bool)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_distance_to_features(
    features_gdf: gpd.GeoDataFrame,
    raster_template_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Compute per-pixel Euclidean distance to the nearest feature.

    Works for both point features (e.g. known deposits) and line features
    (e.g. faults, lineaments).  Distance is returned in the same map units
    as the raster CRS (typically metres for projected CRSs).

    Parameters
    ----------
    features_gdf : GeoDataFrame
        Vector features (points or lines).
    raster_template_path : str or Path
        Raster whose grid defines the output dimensions and CRS.
    output_path : str or Path
        Destination for the distance GeoTIFF.

    Returns
    -------
    Path
        The *output_path* written.
    """
    with rasterio.open(raster_template_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform: Affine = src.transform

    # Rasterize features
    feature_mask = _rasterize_features(features_gdf, height, width, transform)

    if not feature_mask.any():
        logger.warning("No features rasterized; distance raster will be NaN")
        dist = np.full((height, width), np.nan)
        return _write_raster(dist, profile, output_path)

    # EDT gives distance in pixel units from each background cell to nearest
    # foreground cell.  Scale by pixel size to get map units.
    dx = abs(transform.a)
    dy = abs(transform.e)

    # Invert mask: EDT measures distance from 0-cells to nearest 1-cell
    dist_pixels = distance_transform_edt(~feature_mask, sampling=(dy, dx))

    logger.info(
        "Distance to features range: %.1f - %.1f map units",
        float(np.nanmin(dist_pixels)),
        float(np.nanmax(dist_pixels)),
    )

    return _write_raster(dist_pixels, profile, output_path)


def compute_buffered_density(
    features_gdf: gpd.GeoDataFrame,
    raster_template_path: str | Path,
    buffer_distances_km: list[float],
    output_dir: str | Path,
) -> dict[float, Path]:
    """Count features within multiple buffer distances of each pixel.

    For each buffer distance a separate GeoTIFF is written with per-pixel
    feature counts.

    Parameters
    ----------
    features_gdf : GeoDataFrame
        Vector features (points or lines).
    raster_template_path : str or Path
        Template raster for output grid.
    buffer_distances_km : list of float
        Buffer radii in kilometres (e.g. ``[1, 5, 10]``).
    output_dir : str or Path
        Directory for output rasters.

    Returns
    -------
    dict[float, Path]
        Mapping of buffer distance (km) to output raster path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_template_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform: Affine = src.transform

    # Pre-compute feature centroids (for points use coords directly)
    centroids_x: list[float] = []
    centroids_y: list[float] = []
    for geom in features_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        centroids_x.append(c.x)
        centroids_y.append(c.y)

    cx_arr = np.array(centroids_x)
    cy_arr = np.array(centroids_y)

    results: dict[float, Path] = {}

    for buf_km in buffer_distances_km:
        buf_m = buf_km * 1000.0
        density = np.zeros((height, width), dtype=np.float64)

        if len(cx_arr) > 0:
            for r in range(height):
                # Row y-coordinate
                _, row_y = rasterio.transform.xy(transform, r, 0)
                # Coarse filter: skip rows too far from all features
                dy_all = np.abs(cy_arr - row_y)
                if np.min(dy_all) > buf_m:
                    continue
                for c_idx in range(width):
                    px, py = rasterio.transform.xy(transform, r, c_idx)
                    distances = np.hypot(cx_arr - px, cy_arr - py)
                    density[r, c_idx] = float(np.sum(distances <= buf_m))

        out_path = output_dir / f"density_buf_{buf_km}km.tif"
        _write_raster(density, profile, out_path)
        results[buf_km] = out_path
        logger.info("Buffer %.1f km  max count: %.0f", buf_km, np.nanmax(density))

    return results


def compute_drainage_density(
    dem_path: str | Path,
    raster_template_path: str | Path,
    output_path: str | Path,
    accumulation_threshold: int = 100,
    kernel_radius_px: int = 15,
) -> Path:
    """Compute drainage density from DEM-derived flow accumulation.

    A simple D8 flow direction / accumulation is computed from the DEM.
    Pixels exceeding *accumulation_threshold* are classified as drainage.
    Drainage density is then computed as the fraction of drainage pixels
    within a circular kernel of *kernel_radius_px*.

    Drainage anomalies (unusually high or low density) can indicate
    structural control on the hydrological network and are used as
    mineralisation indicators.

    Parameters
    ----------
    dem_path : str or Path
        Path to the input DEM raster.
    raster_template_path : str or Path
        Template raster for output grid (should match the DEM extent).
    output_path : str or Path
        Destination for the drainage density GeoTIFF.
    accumulation_threshold : int
        Minimum flow accumulation to classify a pixel as drainage
        (default 100).
    kernel_radius_px : int
        Radius in pixels for the density computation kernel (default 15).

    Returns
    -------
    Path
        The *output_path* written.
    """
    # Read DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        dem_profile = src.profile.copy()
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    # Fill small sinks with simple iterative fill
    filled = _fill_sinks(dem)

    # D8 flow accumulation
    accumulation = _d8_flow_accumulation(filled)

    # Extract drainage network
    drainage = accumulation >= accumulation_threshold
    drainage_float = drainage.astype(np.float64)

    # Compute density as fraction of drainage pixels in circular kernel
    from scipy.ndimage import uniform_filter

    # Use a circular-ish kernel via disk mask applied as weighted average
    kernel_size = 2 * kernel_radius_px + 1
    drain_sum = uniform_filter(drainage_float, size=kernel_size, mode="constant", cval=0.0)
    # Count of valid pixels in kernel (handle edges)
    ones = np.ones_like(drainage_float)
    count = uniform_filter(ones, size=kernel_size, mode="constant", cval=0.0)

    density = np.divide(drain_sum, count, where=count > 0, out=np.zeros_like(drain_sum))

    # Mask nodata areas
    density[np.isnan(dem)] = np.nan

    # Read template profile for output
    with rasterio.open(raster_template_path) as src:
        out_profile = src.profile.copy()

    # If DEM and template have different shapes, use DEM profile
    if out_profile.get("height") != dem_profile.get("height") or \
       out_profile.get("width") != dem_profile.get("width"):
        logger.warning(
            "DEM and template have different dimensions; using DEM grid for output"
        )
        out_profile = dem_profile.copy()

    logger.info(
        "Drainage density range: %.4f - %.4f",
        float(np.nanmin(density)),
        float(np.nanmax(density)),
    )

    return _write_raster(density, out_profile, output_path)


# ---------------------------------------------------------------------------
# D8 flow routines (simplified)
# ---------------------------------------------------------------------------

# D8 direction offsets: (drow, dcol) for 8 neighbours
_D8_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
_D8_DISTANCES = [
    np.sqrt(2), 1.0, np.sqrt(2),
    1.0,             1.0,
    np.sqrt(2), 1.0, np.sqrt(2),
]


def _fill_sinks(dem: np.ndarray, max_iterations: int = 1000) -> np.ndarray:
    """Simple iterative sink-filling for the DEM.

    Raises each interior pit cell to the minimum of its neighbours until
    no more changes occur or *max_iterations* is reached.
    """
    filled = dem.copy()
    nan_mask = np.isnan(filled)
    # Set NaN to very high value to avoid routing into nodata
    filled[nan_mask] = 1e30

    for _ in range(max_iterations):
        changed = False
        padded = np.pad(filled, 1, mode="constant", constant_values=1e30)
        for dr, dc in _D8_OFFSETS:
            neighbour = padded[1 + dr: padded.shape[0] - 1 + dr,
                               1 + dc: padded.shape[1] - 1 + dc]
            raise_mask = (filled > neighbour) == False  # noqa: E712
            # Actually we need: where filled < all neighbours (a pit)
            # Simpler approach: raise pits to min neighbour
            pass

        # Simplified: just use min-of-neighbours approach
        padded = np.pad(filled, 1, mode="constant", constant_values=1e30)
        min_neighbour = np.full_like(filled, 1e30)
        for dr, dc in _D8_OFFSETS:
            nbr = padded[1 + dr: padded.shape[0] - 1 + dr,
                         1 + dc: padded.shape[1] - 1 + dc]
            min_neighbour = np.minimum(min_neighbour, nbr)

        pits = (~nan_mask) & (filled < min_neighbour)
        # A pit is where all neighbours are higher -- raise it
        # Actually a pit is where the cell is lower than all neighbours
        # We only need to detect cells that are sinks
        # For simplicity raise isolated pits
        pit_cells = (~nan_mask) & (filled < min_neighbour)
        if not pit_cells.any():
            break
        filled[pit_cells] = min_neighbour[pit_cells]
        changed = True

        if not changed:
            break

    filled[nan_mask] = np.nan
    return filled


def _d8_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """Compute D8 flow accumulation from a filled DEM.

    Each cell receives a count of all upstream cells that drain through it.
    Processing order is determined by sorting cells by descending elevation
    (highest cells are processed first).
    """
    nrows, ncols = dem.shape
    accum = np.ones((nrows, ncols), dtype=np.float64)  # each cell counts itself

    nan_mask = np.isnan(dem)
    # Replace NaN with -inf so they sort last and are skipped
    dem_safe = dem.copy()
    dem_safe[nan_mask] = -np.inf

    # Sort cells by descending elevation
    flat_indices = np.argsort(dem_safe.ravel())[::-1]

    for idx in flat_indices:
        r, c = divmod(idx, ncols)
        if nan_mask[r, c]:
            continue

        # Find steepest descent neighbour
        best_drop = 0.0
        best_r, best_c = -1, -1

        for k, (dr, dc) in enumerate(_D8_OFFSETS):
            nr, nc_ = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc_ < ncols and not nan_mask[nr, nc_]:
                drop = (dem_safe[r, c] - dem_safe[nr, nc_]) / _D8_DISTANCES[k]
                if drop > best_drop:
                    best_drop = drop
                    best_r, best_c = nr, nc_

        if best_r >= 0:
            accum[best_r, best_c] += accum[r, c]

    accum[nan_mask] = np.nan
    return accum
