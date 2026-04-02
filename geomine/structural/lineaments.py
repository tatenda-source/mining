"""Lineament extraction from DEM-derived hillshade imagery.

Detects linear geological structures (faults, fractures, contacts) by
combining multi-azimuth hillshade analysis with edge detection and the
probabilistic Hough line transform.  Results are returned as georeferenced
GeoDataFrames suitable for density and intersection analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter, sobel
from shapely.geometry import LineString, Point
from skimage.transform import probabilistic_hough_line

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a single-band raster, replacing nodata with NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr, profile


def _pixel_to_geo(
    row: float, col: float, transform: Affine
) -> tuple[float, float]:
    """Convert (row, col) pixel coordinates to (x, y) geographic coordinates."""
    x, y = rasterio.transform.xy(transform, row, col)
    return x, y


def _canny_edges(image: np.ndarray, sigma: float = 1.5,
                 low_threshold: float = 0.05,
                 high_threshold: float = 0.15) -> np.ndarray:
    """Simple Canny-style edge detection using scipy.ndimage.

    Steps:
    1. Gaussian smoothing
    2. Sobel gradient magnitude
    3. Hysteresis thresholding (simplified -- no non-max suppression,
       which is acceptable for lineament detection where the Hough
       transform handles line fitting).
    """
    # Handle NaN: replace with local mean via gaussian blur
    nan_mask = np.isnan(image)
    clean = image.copy()
    if nan_mask.any():
        clean[nan_mask] = 0.0

    smoothed = gaussian_filter(clean, sigma=sigma)

    # Sobel gradients
    gx = sobel(smoothed, axis=1)  # horizontal gradient
    gy = sobel(smoothed, axis=0)  # vertical gradient
    magnitude = np.hypot(gx, gy)

    # Normalise to [0, 1]
    mag_max = np.nanmax(magnitude)
    if mag_max > 0:
        magnitude /= mag_max

    # Hysteresis thresholding
    strong = magnitude >= high_threshold
    weak = (magnitude >= low_threshold) & ~strong

    # Simple dilation of strong edges to absorb connected weak edges
    from scipy.ndimage import binary_dilation

    edges = binary_dilation(strong, iterations=1) & (strong | weak)

    # Mask out nodata areas
    edges[nan_mask] = False

    return edges.astype(np.uint8)


def _line_azimuth(x0: float, y0: float, x1: float, y1: float) -> float:
    """Compute geographic azimuth (0-180) of a line segment.

    Azimuths are folded into [0, 180) because lineament orientation is
    bidirectional.
    """
    dx = x1 - x0
    dy = y1 - y0
    angle = np.degrees(np.arctan2(dx, dy))  # bearing from north
    angle %= 360.0
    if angle >= 180.0:
        angle -= 180.0
    return angle


def _line_length(x0: float, y0: float, x1: float, y1: float) -> float:
    """Euclidean distance between two points in map units."""
    return float(np.hypot(x1 - x0, y1 - y0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_lineaments(
    dem_path: str | Path,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Extract lineaments from a DEM using multi-azimuth hillshade + Hough transform.

    Parameters
    ----------
    dem_path : str or Path
        Path to the input DEM raster.
    config : dict
        Configuration dictionary.  Relevant keys under ``structural``:

        - ``hillshade_azimuths`` (list[int]): azimuths for hillshade
          (default ``[0, 45, 90, 135, 180, 225, 270, 315]``).
        - ``canny_sigma`` (float): Gaussian sigma for edge detection
          (default 1.5).
        - ``canny_low`` (float): low hysteresis threshold (default 0.05).
        - ``canny_high`` (float): high hysteresis threshold (default 0.15).
        - ``hough_threshold`` (int): accumulator threshold (default 10).
        - ``hough_min_line_length`` (int): min length in pixels (default 30).
        - ``hough_line_gap`` (int): max gap in pixels (default 5).
        - ``min_length_m`` (float): discard lineaments shorter than this
          (default 1000).

    Returns
    -------
    GeoDataFrame
        Columns: ``geometry`` (LineString), ``azimuth`` (float 0-180),
        ``length_m`` (float).  CRS matches the input DEM.
    """
    from geomine.structural.terrain import compute_hillshade

    dem_path = Path(dem_path)
    struct_cfg = config.get("structural", {})

    azimuths = struct_cfg.get("hillshade_azimuths", [0, 45, 90, 135, 180, 225, 270, 315])
    sigma = struct_cfg.get("canny_sigma", 1.5)
    low_t = struct_cfg.get("canny_low", 0.05)
    high_t = struct_cfg.get("canny_high", 0.15)
    hough_threshold = struct_cfg.get("hough_threshold", 10)
    min_line_px = struct_cfg.get("hough_min_line_length", 30)
    line_gap = struct_cfg.get("hough_line_gap", 5)
    min_length_m = struct_cfg.get("min_length_m", 1000.0)

    # Read DEM metadata for coordinate transforms
    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs

    # Accumulate edge maps across azimuths
    import tempfile
    combined_edges: Optional[np.ndarray] = None

    for az in azimuths:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
            compute_hillshade(dem_path, tmp.name, azimuth=az)
            hs_arr, _ = _read_raster(tmp.name)

        edges = _canny_edges(hs_arr, sigma=sigma, low_threshold=low_t,
                             high_threshold=high_t)
        if combined_edges is None:
            combined_edges = edges.astype(np.int32)
        else:
            combined_edges += edges.astype(np.int32)

    # Threshold: a pixel is an edge if detected in at least 2 azimuths
    if combined_edges is None:
        logger.warning("No hillshade azimuths processed; returning empty GeoDataFrame")
        return gpd.GeoDataFrame(
            columns=["geometry", "azimuth", "length_m"],
            geometry="geometry",
            crs=crs,
        )

    edge_binary = (combined_edges >= 2).astype(np.uint8)
    logger.info("Combined edge pixels: %d", int(edge_binary.sum()))

    # Probabilistic Hough transform
    lines = probabilistic_hough_line(
        edge_binary,
        threshold=hough_threshold,
        line_length=min_line_px,
        line_gap=line_gap,
    )
    logger.info("Hough lines detected: %d", len(lines))

    # Convert pixel lines to geographic coordinates and filter by length
    records: list[dict[str, Any]] = []
    for (col0, row0), (col1, row1) in lines:
        x0, y0 = _pixel_to_geo(row0, col0, transform)
        x1, y1 = _pixel_to_geo(row1, col1, transform)

        length = _line_length(x0, y0, x1, y1)
        if length < min_length_m:
            continue

        azimuth = _line_azimuth(x0, y0, x1, y1)
        geom = LineString([(x0, y0), (x1, y1)])
        records.append({"geometry": geom, "azimuth": azimuth, "length_m": length})

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    logger.info("Lineaments after length filter (>%.0f m): %d", min_length_m, len(gdf))
    return gdf


def compute_lineament_density(
    lineaments_gdf: gpd.GeoDataFrame,
    raster_template_path: str | Path,
    bandwidth_m: float = 5000.0,
    output_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Compute kernel density of lineaments on a raster grid.

    For each grid cell the function counts lineament segments whose
    midpoint falls within *bandwidth_m* and normalises to lines per km^2.

    Parameters
    ----------
    lineaments_gdf : GeoDataFrame
        Lineaments with LineString geometries.
    raster_template_path : str or Path
        Raster whose grid dimensions, CRS, and transform define the output.
    bandwidth_m : float
        Search radius in map units (metres) for the kernel (default 5000).
    output_path : str or Path, optional
        If provided, write the result as a GeoTIFF.

    Returns
    -------
    np.ndarray
        2-D array of lineament density (lines per km^2).
    """
    with rasterio.open(raster_template_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform: Affine = src.transform

    density = np.zeros((height, width), dtype=np.float64)

    if lineaments_gdf.empty:
        logger.warning("Empty lineament GeoDataFrame; density is zero everywhere")
        if output_path:
            _write_density(density, profile, output_path)
        return density

    # Pre-compute lineament midpoints as arrays for vectorised distance calc
    midpoints_x = np.array([g.interpolate(0.5, normalized=True).x for g in lineaments_gdf.geometry])
    midpoints_y = np.array([g.interpolate(0.5, normalized=True).y for g in lineaments_gdf.geometry])

    # Cell area in km^2 for normalisation
    dx = abs(transform.a)
    dy = abs(transform.e)
    cell_area_km2 = (dx * dy) / 1e6  # assuming map units are metres

    # Iterate over rows to keep memory bounded
    for r in range(height):
        for c in range(width):
            cx, cy = rasterio.transform.xy(transform, r, c)
            distances = np.hypot(midpoints_x - cx, midpoints_y - cy)
            count = np.sum(distances <= bandwidth_m)
            # Normalise by kernel area in km^2
            kernel_area_km2 = (np.pi * bandwidth_m ** 2) / 1e6
            density[r, c] = count / kernel_area_km2 if kernel_area_km2 > 0 else 0.0

    logger.info("Lineament density range: %.4f - %.4f lines/km^2",
                np.nanmin(density), np.nanmax(density))

    if output_path:
        _write_density(density, profile, output_path)

    return density


def compute_lineament_intersections(
    lineaments_gdf: gpd.GeoDataFrame,
    raster_template_path: str | Path,
    search_radius_m: float = 500.0,
    output_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Compute intersection density of lineaments (structural trap indicator).

    Finds all pairwise intersection points between lineaments and produces
    a density raster counting intersections within *search_radius_m* of
    each pixel centre.

    Parameters
    ----------
    lineaments_gdf : GeoDataFrame
        Lineaments with LineString geometries.
    raster_template_path : str or Path
        Template raster for output grid.
    search_radius_m : float
        Radius in map units for density counting (default 500).
    output_path : str or Path, optional
        If provided, write result as GeoTIFF.

    Returns
    -------
    np.ndarray
        2-D intersection density raster.
    """
    with rasterio.open(raster_template_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        transform: Affine = src.transform

    # Find pairwise intersections
    intersection_points: list[tuple[float, float]] = []
    geoms = lineaments_gdf.geometry.tolist()
    n = len(geoms)
    for i in range(n):
        for j in range(i + 1, n):
            if geoms[i].intersects(geoms[j]):
                pt = geoms[i].intersection(geoms[j])
                if pt.is_empty:
                    continue
                if pt.geom_type == "Point":
                    intersection_points.append((pt.x, pt.y))
                elif pt.geom_type == "MultiPoint":
                    for p in pt.geoms:
                        intersection_points.append((p.x, p.y))

    logger.info("Lineament intersections found: %d", len(intersection_points))

    density = np.zeros((height, width), dtype=np.float64)

    if not intersection_points:
        if output_path:
            _write_density(density, profile, output_path)
        return density

    ix = np.array([p[0] for p in intersection_points])
    iy = np.array([p[1] for p in intersection_points])

    for r in range(height):
        for c in range(width):
            cx, cy = rasterio.transform.xy(transform, r, c)
            distances = np.hypot(ix - cx, iy - cy)
            density[r, c] = float(np.sum(distances <= search_radius_m))

    logger.info("Intersection density range: %.0f - %.0f",
                np.nanmin(density), np.nanmax(density))

    if output_path:
        _write_density(density, profile, output_path)

    return density


# ---------------------------------------------------------------------------
# Internal writer
# ---------------------------------------------------------------------------


def _write_density(data: np.ndarray, profile: dict, output_path: str | Path) -> Path:
    """Write a density array as a single-band GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodata = -9999.0
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

    logger.info("Wrote density raster %s", output_path)
    return output_path
