"""Configuration loader for GeoMine AI projects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from shapely.geometry import shape

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return it as a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.info("Loading config from %s", path)
    with open(path, "r") as fh:
        config = yaml.safe_load(fh)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    # Resolve relative paths against the config file's parent directory
    config["_base_dir"] = str(path.parent.parent)
    return config


def get_aoi_geometry(config: dict[str, Any]) -> Any:
    """Load the AOI GeoJSON file referenced in the config and return a shapely geometry.

    The ``aoi.geojson`` value in the config is resolved relative to the
    project base directory (parent of the ``configs/`` folder).

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary (from :func:`load_config`).

    Returns
    -------
    shapely.geometry.BaseGeometry
        The AOI as a shapely geometry object.
    """
    base_dir = Path(config.get("_base_dir", "."))
    geojson_rel = config["aoi"]["geojson"]
    geojson_path = base_dir / geojson_rel

    if not geojson_path.exists():
        raise FileNotFoundError(f"AOI GeoJSON not found: {geojson_path}")

    logger.info("Loading AOI geometry from %s", geojson_path)
    with open(geojson_path, "r") as fh:
        geojson_data = json.load(fh)

    # Handle both FeatureCollection and single Feature/Geometry
    if geojson_data.get("type") == "FeatureCollection":
        features = geojson_data["features"]
        if len(features) == 0:
            raise ValueError("AOI GeoJSON FeatureCollection is empty")
        geometry = shape(features[0]["geometry"])
    elif geojson_data.get("type") == "Feature":
        geometry = shape(geojson_data["geometry"])
    else:
        # Assume raw geometry
        geometry = shape(geojson_data)

    logger.info("AOI geometry type: %s, bounds: %s", geometry.geom_type, geometry.bounds)
    return geometry
