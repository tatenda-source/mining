#!/usr/bin/env python3
"""CLI script to download all data sources for a GeoMine AI project.

Usage::

    python scripts/download_all.py configs/great_dyke.yaml
    python scripts/download_all.py --skip-sentinel configs/great_dyke.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from tqdm import tqdm

# Ensure the project root is on sys.path when running as a script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from geomine.utils.config import get_aoi_geometry, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("geomine.download_all")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--skip-sentinel", is_flag=True, help="Skip Sentinel-2 download")
@click.option("--skip-aster", is_flag=True, help="Skip ASTER L1T download")
@click.option("--skip-srtm", is_flag=True, help="Skip SRTM DEM download")
@click.option("--skip-mrds", is_flag=True, help="Skip MRDS label download")
@click.option(
    "--earthdata-token",
    envvar="EARTHDATA_TOKEN",
    default=None,
    help="NASA EarthData bearer token for SRTM downloads",
)
@click.option(
    "--copernicus-token",
    envvar="COPERNICUS_TOKEN",
    default=None,
    help="Copernicus Data Space bearer token for Sentinel-2 downloads",
)
def main(
    config_path: str,
    skip_sentinel: bool,
    skip_aster: bool,
    skip_srtm: bool,
    skip_mrds: bool,
    earthdata_token: str | None,
    copernicus_token: str | None,
) -> None:
    """Download all satellite and label data for a GeoMine project.

    CONFIG_PATH is the path to the project YAML configuration file.
    """
    config = load_config(config_path)
    base_dir = Path(config["_base_dir"])

    project_name = config["project"]["name"]
    logger.info("Starting data download for project: %s", project_name)

    steps = []
    if not skip_sentinel:
        steps.append("Sentinel-2")
    if not skip_aster:
        steps.append("ASTER")
    if not skip_srtm:
        steps.append("SRTM")
    if not skip_mrds:
        steps.append("MRDS")

    if not steps:
        logger.warning("All download steps skipped. Nothing to do.")
        return

    progress = tqdm(steps, desc="Downloading data", unit="source")
    errors: list[str] = []

    for step in progress:
        progress.set_postfix_str(step)

        if step == "Sentinel-2":
            try:
                _download_sentinel2(config, base_dir, copernicus_token)
            except Exception as exc:
                logger.error("Sentinel-2 download failed: %s", exc, exc_info=True)
                errors.append(f"Sentinel-2: {exc}")

        elif step == "ASTER":
            try:
                _download_aster(config, earthdata_token)
            except Exception as exc:
                logger.error("ASTER download failed: %s", exc, exc_info=True)
                errors.append(f"ASTER: {exc}")

        elif step == "SRTM":
            try:
                _download_srtm(config, base_dir, earthdata_token)
            except Exception as exc:
                logger.error("SRTM download failed: %s", exc, exc_info=True)
                errors.append(f"SRTM: {exc}")

        elif step == "MRDS":
            try:
                _download_mrds(config)
            except Exception as exc:
                logger.error("MRDS download failed: %s", exc, exc_info=True)
                errors.append(f"MRDS: {exc}")

    progress.close()

    if errors:
        logger.warning("Completed with %d error(s):", len(errors))
        for err in errors:
            logger.warning("  - %s", err)
    else:
        logger.info("All downloads completed successfully.")


def _download_sentinel2(
    config: dict,
    base_dir: Path,
    auth_token: str | None,
) -> None:
    """Search and download Sentinel-2 scenes, then preprocess."""
    from geomine.ingest.sentinel2 import (
        download_scene,
        preprocess_sentinel2,
        search_scenes,
    )

    raw_dir = base_dir / config["data"]["raw_dir"] / "sentinel2"
    processed_dir = base_dir / config["data"]["processed_dir"] / "sentinel2"
    bands = config["data"]["sentinel2"]["bands"]

    logger.info("--- Sentinel-2 Download ---")
    scenes = search_scenes(config)

    if not scenes:
        logger.warning("No Sentinel-2 scenes found matching criteria")
        return

    logger.info("Downloading %d Sentinel-2 scenes", len(scenes))
    for scene in tqdm(scenes, desc="Sentinel-2 scenes", unit="scene", leave=False):
        download_scene(scene, bands, raw_dir, auth_token=auth_token)

    logger.info("Preprocessing Sentinel-2 scenes")
    stacked = preprocess_sentinel2(raw_dir, processed_dir, config)
    logger.info("Produced %d stacked multi-band GeoTIFFs", len(stacked))


def _download_aster(
    config: dict,
    earthdata_token: str | None,
) -> None:
    """Download ASTER L1T scenes (pre-2008 for SWIR bands)."""
    from geomine.ingest.aster import download_aster

    logger.info("--- ASTER L1T Download ---")
    logger.info(
        "NOTE: SWIR bands only available in pre-April-2008 data "
        "(detector failure 2008-04-23)"
    )

    files = download_aster(
        config,
        earthdata_token=earthdata_token,
        method="earthaccess",  # try earthaccess first
        max_granules=50,
    )

    if not files:
        logger.warning("No ASTER L1T files downloaded")
        return

    logger.info("Downloaded %d ASTER L1T files", len(files))


def _download_srtm(
    config: dict,
    base_dir: Path,
    earthdata_token: str | None,
) -> None:
    """Download and process SRTM DEM."""
    from geomine.ingest.srtm import download_srtm, mosaic_and_clip

    logger.info("--- SRTM DEM Download ---")
    tile_paths = download_srtm(config, earthdata_token=earthdata_token)

    if not tile_paths:
        logger.warning("No SRTM tiles downloaded")
        return

    aoi_geometry = get_aoi_geometry(config)
    dst_crs = config["project"]["crs"]
    processed_dir = base_dir / config["data"]["processed_dir"] / "srtm"
    output_path = processed_dir / "srtm_dem.tif"

    mosaic_and_clip(tile_paths, output_path, aoi_geometry, dst_crs)
    logger.info("SRTM DEM saved: %s", output_path)


def _download_mrds(config: dict) -> None:
    """Download MRDS and MINDAT mineral deposit labels."""
    from geomine.ingest.mrds import download_mindat, download_mrds, merge_labels

    logger.info("--- MRDS Label Download ---")
    mrds_gdf = download_mrds(config)
    mindat_gdf = download_mindat(config)

    merged = merge_labels(mrds_gdf, mindat_gdf)
    logger.info("Total deposit labels: %d", len(merged))

    if not merged.empty:
        base_dir = Path(config["_base_dir"])
        training_dir = base_dir / config["data"]["training_dir"]
        training_dir.mkdir(parents=True, exist_ok=True)
        output_path = training_dir / "deposit_labels.geojson"
        merged.to_file(output_path, driver="GeoJSON")
        logger.info("Merged labels saved: %s", output_path)


if __name__ == "__main__":
    main()
