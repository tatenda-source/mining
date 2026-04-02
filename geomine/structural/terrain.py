"""DEM-derived terrain feature computation for structural geology analysis.

Computes slope, aspect, curvature, and hillshade products from Digital
Elevation Models using finite-difference methods.  All outputs are written
as GeoTIFF rasters preserving the source CRS and affine transform.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_dem(dem_path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a single-band DEM and return the array + profile metadata.

    Nodata pixels are replaced with ``np.nan`` so downstream maths
    propagate missing values naturally.
    """
    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        dem[dem == nodata] = np.nan

    logger.info("Read DEM %s  shape=%s  CRS=%s", dem_path.name, dem.shape, profile.get("crs"))
    return dem, profile


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


def _cell_size(profile: dict) -> tuple[float, float]:
    """Return (cellsize_x, cellsize_y) in map units from the profile transform."""
    transform: Affine = profile["transform"]
    dx = abs(transform.a)
    dy = abs(transform.e)
    return dx, dy


def _gradient_3x3(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute dz/dx and dz/dy using the Horn (1981) 3x3 finite-difference kernel.

    Returns arrays the same shape as *dem* with NaN where computation is
    not possible (edges and nodata neighbours).
    """
    # Pad with NaN so edges produce NaN naturally
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)

    # Neighbour references (row, col offsets from centre)
    a = padded[:-2, :-2]   # top-left
    b = padded[:-2, 1:-1]  # top-centre
    c = padded[:-2, 2:]    # top-right
    d = padded[1:-1, :-2]  # mid-left
    f = padded[1:-1, 2:]   # mid-right
    g = padded[2:, :-2]    # bottom-left
    h = padded[2:, 1:-1]   # bottom-centre
    i = padded[2:, 2:]     # bottom-right

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)

    return dz_dx, dz_dy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_slope(dem_path: str | Path, output_path: str | Path) -> Path:
    """Compute slope in degrees from a DEM and save as GeoTIFF.

    Uses the Horn (1981) finite-difference method on a 3x3 kernel:

        slope = arctan(sqrt((dz/dx)^2 + (dz/dy)^2))

    Parameters
    ----------
    dem_path : str or Path
        Path to the input DEM raster (single band).
    output_path : str or Path
        Destination for the slope GeoTIFF.

    Returns
    -------
    Path
        The *output_path* written.
    """
    dem, profile = _read_dem(dem_path)
    dx, dy = _cell_size(profile)

    dz_dx, dz_dy = _gradient_3x3(dem, dx, dy)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    return _write_raster(slope_deg, profile, output_path)


def compute_aspect(dem_path: str | Path, output_path: str | Path) -> Path:
    """Compute aspect in degrees (0-360, clockwise from north) from a DEM.

    Uses ``arctan2(-dz/dy, dz/dx)`` then converts the mathematical angle
    (counter-clockwise from east) to a compass bearing.

    Parameters
    ----------
    dem_path : str or Path
        Path to the input DEM raster.
    output_path : str or Path
        Destination for the aspect GeoTIFF.

    Returns
    -------
    Path
        The *output_path* written.
    """
    dem, profile = _read_dem(dem_path)
    dx, dy = _cell_size(profile)

    dz_dx, dz_dy = _gradient_3x3(dem, dx, dy)

    # Mathematical angle in radians (counter-clockwise from east)
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)

    # Convert to compass bearing: 0=North, 90=East, 180=South, 270=West
    # compass = 90 - math_angle, wrapped to [0, 360)
    compass = (90.0 - aspect_deg) % 360.0

    # Flat areas (slope == 0) get -1 by convention
    flat_mask = (dz_dx == 0) & (dz_dy == 0)
    compass[flat_mask] = -1.0

    return _write_raster(compass, profile, output_path)


def compute_curvature(
    dem_path: str | Path,
    output_path_plan: str | Path,
    output_path_profile: str | Path,
) -> tuple[Path, Path]:
    """Compute plan and profile curvature from a DEM.

    Plan curvature measures the rate of change of aspect along a contour
    (convergence / divergence of flow).  Profile curvature measures the
    rate of change of slope along the steepest descent direction
    (acceleration / deceleration of flow).

    Parameters
    ----------
    dem_path : str or Path
        Path to the input DEM raster.
    output_path_plan : str or Path
        Destination for plan curvature GeoTIFF.
    output_path_profile : str or Path
        Destination for profile curvature GeoTIFF.

    Returns
    -------
    tuple[Path, Path]
        (*output_path_plan*, *output_path_profile*).
    """
    dem, profile = _read_dem(dem_path)
    dx, dy = _cell_size(profile)

    # Pad for second-order finite differences
    padded = np.pad(dem, 1, mode="constant", constant_values=np.nan)

    # First derivatives (Horn method)
    dz_dx, dz_dy = _gradient_3x3(dem, dx, dy)

    # Second derivatives via central differences on the padded grid
    # d2z/dx2
    d2z_dx2 = (
        padded[1:-1, 2:] - 2.0 * padded[1:-1, 1:-1] + padded[1:-1, :-2]
    ) / (dx ** 2)
    # d2z/dy2
    d2z_dy2 = (
        padded[2:, 1:-1] - 2.0 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]
    ) / (dy ** 2)
    # d2z/dxdy
    d2z_dxdy = (
        (padded[:-2, 2:] - padded[:-2, :-2]) - (padded[2:, 2:] - padded[2:, :-2])
    ) / (4.0 * dx * dy)

    p = dz_dx ** 2
    q = dz_dy ** 2
    pq = p + q

    # Avoid division by zero on flat terrain
    safe_pq = np.where(pq == 0, np.nan, pq)

    plan_curv = -(q * d2z_dx2 - 2.0 * dz_dx * dz_dy * d2z_dxdy + p * d2z_dy2) / (
        safe_pq ** 1.5
    )
    profile_curv = -(p * d2z_dx2 + 2.0 * dz_dx * dz_dy * d2z_dxdy + q * d2z_dy2) / (
        safe_pq * np.sqrt(1.0 + safe_pq)
    )

    p_plan = _write_raster(plan_curv, profile, output_path_plan)
    p_prof = _write_raster(profile_curv, profile, output_path_profile)
    return p_plan, p_prof


def compute_hillshade(
    dem_path: str | Path,
    output_path: str | Path,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> Path:
    """Compute an analytical hillshade from a DEM.

    Parameters
    ----------
    dem_path : str or Path
        Path to the input DEM raster.
    output_path : str or Path
        Destination for the hillshade GeoTIFF (values 0-255).
    azimuth : float
        Sun azimuth in degrees clockwise from north (default 315 = NW).
    altitude : float
        Sun altitude above the horizon in degrees (default 45).

    Returns
    -------
    Path
        The *output_path* written.
    """
    dem, profile = _read_dem(dem_path)
    dx, dy = _cell_size(profile)

    dz_dx, dz_dy = _gradient_3x3(dem, dx, dy)

    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect_rad = np.arctan2(-dz_dy, dz_dx)

    # Convert illumination angles
    azimuth_rad = np.radians(360.0 - azimuth + 90.0)  # to math angle
    altitude_rad = np.radians(altitude)

    hillshade = (
        np.sin(altitude_rad) * np.cos(slope_rad)
        + np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )

    # Clamp and scale to 0-255
    hillshade = np.clip(hillshade, 0, 1) * 255.0

    return _write_raster(hillshade, profile, output_path)


def compute_multi_hillshade(
    dem_path: str | Path,
    output_dir: str | Path,
    azimuths: Optional[Sequence[float]] = None,
) -> list[Path]:
    """Compute hillshades at multiple azimuths for visual lineament detection.

    Parameters
    ----------
    dem_path : str or Path
        Path to the input DEM raster.
    output_dir : str or Path
        Directory to write individual hillshade GeoTIFFs.
    azimuths : sequence of float, optional
        Sun azimuths in degrees.  Defaults to eight cardinal/intercardinal
        directions: ``[0, 45, 90, 135, 180, 225, 270, 315]``.

    Returns
    -------
    list[Path]
        Paths to the written hillshade rasters, one per azimuth.
    """
    if azimuths is None:
        azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for az in azimuths:
        out_path = output_dir / f"hillshade_az{int(az):03d}.tif"
        compute_hillshade(dem_path, out_path, azimuth=az)
        paths.append(out_path)
        logger.info("Hillshade azimuth=%d -> %s", az, out_path)

    return paths
