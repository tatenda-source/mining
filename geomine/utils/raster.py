"""Raster I/O and processing utilities for GeoMine AI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rasterio_mask
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject
from shapely.geometry import mapping

logger = logging.getLogger(__name__)


def reproject_raster(
    src_path: str | Path,
    dst_path: str | Path,
    dst_crs: str,
) -> Path:
    """Reproject a raster file to a new coordinate reference system.

    Parameters
    ----------
    src_path : str or Path
        Input raster file path.
    dst_path : str or Path
        Output raster file path.
    dst_crs : str
        Target CRS (e.g. ``"EPSG:32736"``).

    Returns
    -------
    Path
        The output file path.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reprojecting %s -> %s (CRS: %s)", src_path.name, dst_path.name, dst_crs)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst_profile = src.profile.copy()
        dst_profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
        )

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

    logger.info("Reprojection complete: %s", dst_path)
    return dst_path


def clip_raster_to_aoi(
    src_path: str | Path,
    dst_path: str | Path,
    aoi_geometry: Any,
    dst_crs: str | None = None,
) -> Path:
    """Clip a raster to an Area of Interest polygon.

    Parameters
    ----------
    src_path : str or Path
        Input raster file path.
    dst_path : str or Path
        Output raster file path.
    aoi_geometry : shapely.geometry.BaseGeometry
        AOI polygon in the same CRS as the source raster,
        or in WGS84 if ``dst_crs`` triggers a reproject first.
    dst_crs : str, optional
        If provided and different from the source CRS, the raster is
        reprojected before clipping.

    Returns
    -------
    Path
        The output file path.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Clipping %s to AOI -> %s", src_path.name, dst_path.name)

    with rasterio.open(src_path) as src:
        geojson_geom = [mapping(aoi_geometry)]

        out_image, out_transform = rasterio_mask(
            src,
            geojson_geom,
            crop=True,
            nodata=src.nodata if src.nodata is not None else -9999,
        )

        out_profile = src.profile.copy()
        out_profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
            nodata=src.nodata if src.nodata is not None else -9999,
        )

        if dst_crs is not None:
            out_profile["crs"] = dst_crs

        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(out_image)

    logger.info("Clipping complete: %s", dst_path)
    return dst_path


def resample_to_target(
    src_path: str | Path,
    dst_path: str | Path,
    target_resolution: float,
) -> Path:
    """Resample a raster to a target resolution (in CRS units, typically meters).

    Parameters
    ----------
    src_path : str or Path
        Input raster file path.
    dst_path : str or Path
        Output raster file path.
    target_resolution : float
        Desired pixel size in the raster's CRS units.

    Returns
    -------
    Path
        The output file path.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Resampling %s to %.1f m resolution -> %s",
        src_path.name,
        target_resolution,
        dst_path.name,
    )

    with rasterio.open(src_path) as src:
        # Compute new dimensions
        left, bottom, right, top = src.bounds
        new_width = max(1, int((right - left) / target_resolution))
        new_height = max(1, int((top - bottom) / target_resolution))
        new_transform = from_bounds(left, bottom, right, top, new_width, new_height)

        dst_profile = src.profile.copy()
        dst_profile.update(
            width=new_width,
            height=new_height,
            transform=new_transform,
        )

        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear,
                )

    logger.info("Resampling complete: %s", dst_path)
    return dst_path


def stack_bands(
    band_paths: list[str | Path],
    output_path: str | Path,
) -> Path:
    """Stack multiple single-band rasters into one multi-band GeoTIFF.

    All input rasters must share the same CRS, transform, and dimensions.

    Parameters
    ----------
    band_paths : list of str or Path
        Ordered list of single-band raster file paths.
    output_path : str or Path
        Output multi-band GeoTIFF path.

    Returns
    -------
    Path
        The output file path.
    """
    if not band_paths:
        raise ValueError("band_paths must not be empty")

    band_paths = [Path(p) for p in band_paths]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Stacking %d bands -> %s", len(band_paths), output_path.name)

    # Read the first band to get the reference profile
    with rasterio.open(band_paths[0]) as ref:
        profile = ref.profile.copy()
        profile.update(
            count=len(band_paths),
            driver="GTiff",
            compress="lzw",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for idx, bp in enumerate(band_paths, start=1):
                with rasterio.open(bp) as src:
                    data = src.read(1)
                    dst.write(data, idx)
                    dst.set_band_description(idx, bp.stem)

    logger.info("Band stacking complete: %s", output_path)
    return output_path


def read_band(
    raster_path: str | Path,
    band_index: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a single band from a raster as a NumPy array.

    Parameters
    ----------
    raster_path : str or Path
        Path to the raster file.
    band_index : int
        1-based band index to read (default: 1).

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple of (data_array, profile_dict). The data array has nodata
        pixels replaced with ``np.nan`` for float dtypes.
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        data = src.read(band_index).astype(np.float32)
        profile = src.profile.copy()

        # Replace nodata with NaN
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    return data, profile
