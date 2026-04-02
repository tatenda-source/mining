"""Sentinel-2 L2A scene search and download via Copernicus Data Space STAC."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests
from pystac_client import Client as STACClient

from geomine.utils.raster import (
    clip_raster_to_aoi,
    reproject_raster,
    stack_bands,
)

logger = logging.getLogger(__name__)

COPERNICUS_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"


def search_scenes(config: dict[str, Any]) -> list[Any]:
    """Search the Copernicus STAC catalog for Sentinel-2 L2A scenes.

    Parameters
    ----------
    config : dict
        Project configuration dictionary. Expected keys::

            aoi.bbox            -- [west, south, east, north] in WGS84
            data.sentinel2.collection
            data.sentinel2.date_range  -- [start, end] ISO date strings
            data.sentinel2.max_cloud_cover

    Returns
    -------
    list[pystac.Item]
        Matching STAC items sorted by cloud cover (ascending).
    """
    s2_cfg = config["data"]["sentinel2"]
    bbox = config["aoi"]["bbox"]
    date_start, date_end = s2_cfg["date_range"]
    max_cloud = s2_cfg["max_cloud_cover"]
    collection = s2_cfg["collection"]

    logger.info(
        "Searching Sentinel-2 scenes: bbox=%s, dates=%s/%s, cloud<=%d%%",
        bbox, date_start, date_end, max_cloud,
    )

    catalog = STACClient.open(COPERNICUS_STAC_URL)

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lte": max_cloud}},
        max_items=100,
    )

    items = list(search.items())
    # Sort by cloud cover ascending
    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))

    logger.info("Found %d Sentinel-2 scenes matching criteria", len(items))
    return items


def download_scene(
    scene_item: Any,
    bands: list[str],
    output_dir: str | Path,
    auth_token: str | None = None,
) -> dict[str, Path]:
    """Download specific bands from a Sentinel-2 STAC scene item.

    Parameters
    ----------
    scene_item : pystac.Item
        A STAC item representing a single Sentinel-2 scene.
    bands : list[str]
        Band identifiers to download (e.g. ``["B02", "B03", "B04"]``).
    output_dir : str or Path
        Directory to save downloaded band files.
    auth_token : str, optional
        Bearer token for authenticated downloads from Copernicus Data Space.

    Returns
    -------
    dict[str, Path]
        Mapping of band name to downloaded file path.
    """
    output_dir = Path(output_dir)
    scene_id = scene_item.id
    scene_dir = output_dir / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading scene %s, bands: %s", scene_id, bands)

    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    downloaded: dict[str, Path] = {}

    for band_name in bands:
        # STAC assets may use different key conventions
        asset_key = _resolve_asset_key(scene_item, band_name)
        if asset_key is None:
            logger.warning(
                "Band %s not found in scene %s assets, skipping", band_name, scene_id
            )
            continue

        asset = scene_item.assets[asset_key]
        url = asset.href
        dst_file = scene_dir / f"{band_name}.tif"

        if dst_file.exists():
            logger.info("Band %s already downloaded, skipping", band_name)
            downloaded[band_name] = dst_file
            continue

        logger.info("Downloading %s from %s", band_name, url)
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=300)
            response.raise_for_status()

            with open(dst_file, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)

            downloaded[band_name] = dst_file
            logger.info("Saved %s (%d bytes)", dst_file.name, dst_file.stat().st_size)

        except requests.RequestException as exc:
            logger.error("Failed to download %s: %s", band_name, exc)
            # Clean up partial file
            if dst_file.exists():
                dst_file.unlink()

    return downloaded


def _resolve_asset_key(scene_item: Any, band_name: str) -> str | None:
    """Resolve the STAC asset key for a given band name.

    Different STAC catalogs may store band assets under varying key names
    (e.g. ``"B02"``, ``"b02"``, ``"B02_10m"``).
    """
    assets = scene_item.assets
    # Try exact match first
    if band_name in assets:
        return band_name
    # Try lowercase
    if band_name.lower() in assets:
        return band_name.lower()
    # Try partial match
    for key in assets:
        if band_name.lower() in key.lower():
            return key
    return None


def preprocess_sentinel2(
    raw_dir: str | Path,
    processed_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Preprocess downloaded Sentinel-2 scenes.

    For each scene directory in ``raw_dir``:
    1. Reproject each band to the project CRS.
    2. Clip each band to the AOI.
    3. Stack all bands into a single multi-band GeoTIFF.

    Parameters
    ----------
    raw_dir : str or Path
        Directory containing raw scene subdirectories.
    processed_dir : str or Path
        Directory for processed output files.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    list[Path]
        Paths to the stacked multi-band GeoTIFFs.
    """
    from geomine.utils.config import get_aoi_geometry

    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dst_crs = config["project"]["crs"]
    bands = config["data"]["sentinel2"]["bands"]
    aoi_geometry = get_aoi_geometry(config)

    stacked_outputs: list[Path] = []

    scene_dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not scene_dirs:
        logger.warning("No scene directories found in %s", raw_dir)
        return stacked_outputs

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        logger.info("Processing scene: %s", scene_id)

        proc_scene_dir = processed_dir / scene_id
        proc_scene_dir.mkdir(parents=True, exist_ok=True)

        reprojected_bands: list[Path] = []

        for band_name in bands:
            raw_band = scene_dir / f"{band_name}.tif"
            if not raw_band.exists():
                logger.warning("Band file missing: %s", raw_band)
                continue

            # Step 1: Reproject
            reproj_path = proc_scene_dir / f"{band_name}_reproj.tif"
            reproject_raster(raw_band, reproj_path, dst_crs)

            # Step 2: Clip to AOI
            clipped_path = proc_scene_dir / f"{band_name}_clipped.tif"
            clip_raster_to_aoi(reproj_path, clipped_path, aoi_geometry, dst_crs)

            reprojected_bands.append(clipped_path)

            # Remove intermediate reprojected file to save space
            reproj_path.unlink(missing_ok=True)

        if reprojected_bands:
            # Step 3: Stack bands
            stacked_path = proc_scene_dir / f"{scene_id}_stacked.tif"
            stack_bands(reprojected_bands, stacked_path)
            stacked_outputs.append(stacked_path)
            logger.info("Stacked scene saved: %s", stacked_path)
        else:
            logger.warning("No bands processed for scene %s", scene_id)

    logger.info("Preprocessing complete: %d scenes processed", len(stacked_outputs))
    return stacked_outputs
