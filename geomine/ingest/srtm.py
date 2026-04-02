"""SRTM 30m DEM download and processing."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from rasterio.merge import merge
from shapely.geometry import mapping

from geomine.utils.raster import clip_raster_to_aoi, reproject_raster

logger = logging.getLogger(__name__)

SRTM_BASE_URL = (
    "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11"
)


def _tile_name(lat: int, lon: int) -> str:
    """Generate the SRTM tile filename for a given integer lat/lon corner.

    SRTM tiles are named by their south-west corner, e.g. ``N19E030``.

    Parameters
    ----------
    lat : int
        Latitude of the tile's south-west corner.
    lon : int
        Longitude of the tile's south-west corner.
    """
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"


def _tiles_for_bbox(bbox: list[float]) -> list[str]:
    """Compute the list of SRTM tile names covering a bounding box.

    Parameters
    ----------
    bbox : list[float]
        [west, south, east, north] in WGS84 degrees.

    Returns
    -------
    list[str]
        Tile names like ``["N19E029", "N19E030", ...]``.
    """
    west, south, east, north = bbox

    lat_min = math.floor(south)
    lat_max = math.floor(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)

    tiles: list[str] = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            tiles.append(_tile_name(lat, lon))

    return tiles


def download_srtm(
    config: dict[str, Any],
    earthdata_token: str | None = None,
) -> list[Path]:
    """Download SRTM 30m tiles covering the project AOI.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.
    earthdata_token : str, optional
        NASA EarthData bearer token. Required for authenticated downloads.
        Can also be set via the ``EARTHDATA_TOKEN`` environment variable.

    Returns
    -------
    list[Path]
        Paths to the downloaded HGT/TIFF tile files.
    """
    import os

    bbox = config["aoi"]["bbox"]
    base_dir = Path(config.get("_base_dir", "."))
    raw_dir = base_dir / config["data"]["raw_dir"] / "srtm"
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = earthdata_token or os.environ.get("EARTHDATA_TOKEN")

    tile_names = _tiles_for_bbox(bbox)
    logger.info("SRTM tiles needed for bbox %s: %s", bbox, tile_names)

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    downloaded: list[Path] = []

    for tile_name in tile_names:
        filename = f"{tile_name}.SRTMGL1.hgt.zip"
        url = f"{SRTM_BASE_URL}/{filename}"
        dst_file = raw_dir / filename

        if dst_file.exists():
            logger.info("SRTM tile %s already downloaded, skipping", tile_name)
            downloaded.append(dst_file)
            continue

        logger.info("Downloading SRTM tile %s from %s", tile_name, url)
        try:
            response = requests.get(
                url, headers=headers, stream=True, timeout=300, allow_redirects=True
            )
            response.raise_for_status()

            with open(dst_file, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)

            downloaded.append(dst_file)
            logger.info("Saved SRTM tile: %s (%d bytes)", dst_file.name, dst_file.stat().st_size)

        except requests.RequestException as exc:
            logger.error("Failed to download SRTM tile %s: %s", tile_name, exc)

    logger.info("Downloaded %d / %d SRTM tiles", len(downloaded), len(tile_names))
    return downloaded


def mosaic_and_clip(
    tile_paths: list[str | Path],
    output_path: str | Path,
    aoi_geometry: Any,
    dst_crs: str,
) -> Path:
    """Mosaic SRTM tiles and clip to the AOI.

    Parameters
    ----------
    tile_paths : list of str or Path
        Paths to individual SRTM raster tiles.
    output_path : str or Path
        Output clipped DEM file path.
    aoi_geometry : shapely.geometry.BaseGeometry
        AOI polygon (in WGS84).
    dst_crs : str
        Target CRS for the output (e.g. ``"EPSG:32736"``).

    Returns
    -------
    Path
        Path to the clipped, reprojected DEM.
    """
    tile_paths = [Path(p) for p in tile_paths]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tile_paths:
        raise ValueError("No SRTM tile paths provided")

    logger.info("Mosaicking %d SRTM tiles", len(tile_paths))

    # Open all tiles
    src_files = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic_data, mosaic_transform = merge(src_files, nodata=-9999)
    finally:
        for src in src_files:
            src.close()

    # Write mosaic to a temporary file
    mosaic_path = output_path.parent / "srtm_mosaic_tmp.tif"
    profile = {
        "driver": "GTiff",
        "dtype": mosaic_data.dtype,
        "width": mosaic_data.shape[2],
        "height": mosaic_data.shape[1],
        "count": mosaic_data.shape[0],
        "crs": "EPSG:4326",
        "transform": mosaic_transform,
        "nodata": -9999,
        "compress": "lzw",
    }

    with rasterio.open(mosaic_path, "w", **profile) as dst:
        dst.write(mosaic_data)

    logger.info("Mosaic written: %s", mosaic_path)

    # Reproject to target CRS
    reproj_path = output_path.parent / "srtm_reproj_tmp.tif"
    reproject_raster(mosaic_path, reproj_path, dst_crs)

    # Clip to AOI
    clip_raster_to_aoi(reproj_path, output_path, aoi_geometry, dst_crs)

    # Clean up intermediate files
    mosaic_path.unlink(missing_ok=True)
    reproj_path.unlink(missing_ok=True)

    logger.info("SRTM DEM clipped and reprojected: %s", output_path)
    return output_path
