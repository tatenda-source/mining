"""USGS MRDS and MINDAT mineral deposit label downloaders."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Point

logger = logging.getLogger(__name__)

MRDS_WFS_URL = "https://mrdata.usgs.gov/mrds/wfs"


def download_mrds(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Download mineral deposit records from the USGS MRDS WFS service.

    Queries deposits within the configured bounding box, filtered by
    the commodity types listed in the config.

    Parameters
    ----------
    config : dict
        Project configuration dictionary. Expected keys::

            data.mrds.bbox         -- [west, south, east, north] in WGS84
            data.mrds.commodities  -- list of commodity strings

    Returns
    -------
    GeoDataFrame
        Mineral deposit records with columns:
        [name, commodity, latitude, longitude, deposit_type, geometry].
    """
    mrds_cfg = config["data"]["mrds"]
    bbox = mrds_cfg["bbox"]
    commodities = mrds_cfg.get("commodities", [])

    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"

    logger.info("Querying MRDS WFS: bbox=%s, commodities=%s", bbox_str, commodities)

    params: dict[str, str] = {
        "service": "WFS",
        "version": "1.1.0",
        "request": "GetFeature",
        "typeName": "mrds",
        "bbox": bbox_str,
        "outputFormat": "application/json",
        "maxFeatures": "5000",
        "srsName": "EPSG:4326",
    }

    try:
        response = requests.get(MRDS_WFS_URL, params=params, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("MRDS WFS request failed: %s", exc)
        return gpd.GeoDataFrame(
            columns=["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]
        )

    content_type = response.headers.get("Content-Type", "")

    if "json" in content_type or response.text.strip().startswith("{"):
        gdf = _parse_geojson_response(response)
    else:
        gdf = _parse_wfs_xml_response(response)

    if gdf.empty:
        logger.warning("No MRDS records returned for bbox %s", bbox_str)
        return gdf

    # Filter by commodity if specified
    if commodities:
        commodity_pattern = "|".join(commodities)
        mask = gdf["commodity"].str.contains(commodity_pattern, case=False, na=False)
        gdf = gdf[mask].copy()
        logger.info("Filtered to %d records matching commodities: %s", len(gdf), commodities)

    # Save to the raw data directory
    base_dir = Path(config.get("_base_dir", "."))
    raw_dir = base_dir / config["data"]["raw_dir"] / "mrds"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "mrds_deposits.geojson"
    gdf.to_file(output_path, driver="GeoJSON")
    logger.info("Saved %d MRDS records to %s", len(gdf), output_path)

    return gdf


def _parse_geojson_response(response: requests.Response) -> gpd.GeoDataFrame:
    """Parse a GeoJSON WFS response into a GeoDataFrame."""
    data = response.json()
    features = data.get("features", [])

    if not features:
        return gpd.GeoDataFrame(
            columns=["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]
        )

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    # Normalize column names -- MRDS WFS may use varying field names
    column_map: dict[str, str] = {}
    for col in gdf.columns:
        col_lower = col.lower()
        if "name" in col_lower and "name" not in column_map.values():
            column_map[col] = "name"
        elif "commod" in col_lower:
            column_map[col] = "commodity"
        elif "dep_type" in col_lower or "deposit" in col_lower:
            column_map[col] = "deposit_type"

    gdf = gdf.rename(columns=column_map)

    # Ensure required columns exist
    for required_col in ["name", "commodity", "deposit_type"]:
        if required_col not in gdf.columns:
            gdf[required_col] = np.nan

    # Extract coordinates
    gdf["longitude"] = gdf.geometry.x
    gdf["latitude"] = gdf.geometry.y

    return gdf[["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]]


def _parse_wfs_xml_response(response: requests.Response) -> gpd.GeoDataFrame:
    """Parse a WFS XML response into a GeoDataFrame."""
    try:
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError as exc:
        logger.error("Failed to parse MRDS WFS XML response: %s", exc)
        return gpd.GeoDataFrame(
            columns=["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]
        )

    # Define common GML namespaces
    ns = {
        "gml": "http://www.opengis.net/gml",
        "wfs": "http://www.opengis.net/wfs",
        "mrds": "https://mrdata.usgs.gov/mrds",
    }

    records: list[dict[str, Any]] = []

    for member in root.iter("{http://www.opengis.net/gml}featureMember"):
        record: dict[str, Any] = {}
        for elem in member.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            tag_lower = tag.lower()
            if "name" in tag_lower and "name" not in record:
                record["name"] = elem.text
            elif "commod" in tag_lower:
                record["commodity"] = elem.text
            elif "dep_type" in tag_lower or "deposit" in tag_lower:
                record["deposit_type"] = elem.text
            elif tag_lower in ("latitude", "lat"):
                record["latitude"] = float(elem.text) if elem.text else None
            elif tag_lower in ("longitude", "lon", "long"):
                record["longitude"] = float(elem.text) if elem.text else None

        # Try extracting coords from GML point
        if "latitude" not in record or "longitude" not in record:
            pos = member.find(".//{http://www.opengis.net/gml}pos")
            if pos is not None and pos.text:
                parts = pos.text.strip().split()
                if len(parts) >= 2:
                    record["latitude"] = float(parts[0])
                    record["longitude"] = float(parts[1])

        if record.get("latitude") is not None and record.get("longitude") is not None:
            record["geometry"] = Point(record["longitude"], record["latitude"])
            records.append(record)

    if not records:
        return gpd.GeoDataFrame(
            columns=["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    for col in ["name", "commodity", "deposit_type"]:
        if col not in gdf.columns:
            gdf[col] = np.nan

    return gdf[["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]]


def parse_mrds_response(response: requests.Response) -> gpd.GeoDataFrame:
    """Parse a raw MRDS WFS response into a normalized GeoDataFrame.

    Handles both GeoJSON and XML response formats.

    Parameters
    ----------
    response : requests.Response
        Raw HTTP response from the MRDS WFS endpoint.

    Returns
    -------
    GeoDataFrame
        Columns: [name, commodity, latitude, longitude, deposit_type, geometry].
    """
    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type or response.text.strip().startswith("{"):
        return _parse_geojson_response(response)
    return _parse_wfs_xml_response(response)


def download_mindat(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Download mineral occurrence data from the MINDAT API.

    .. note::
        TODO: Implement MINDAT API integration. Requires an API key
        obtained from https://www.mindat.org/. The API provides mineral
        locality data that can supplement MRDS for training labels.

    Parameters
    ----------
    config : dict
        Project configuration dictionary.

    Returns
    -------
    GeoDataFrame
        Empty GeoDataFrame with the expected schema (placeholder).
    """
    # TODO: Implement MINDAT API download.
    #   1. Register for an API key at https://www.mindat.org/
    #   2. Set the key in config or environment variable MINDAT_API_KEY
    #   3. Query endpoint: https://api.mindat.org/geomaterials/
    #   4. Filter by locality bounding box and mineral species
    #   5. Parse JSON response into GeoDataFrame
    logger.warning(
        "download_mindat is a placeholder -- MINDAT API key required. "
        "Returning empty GeoDataFrame."
    )
    return gpd.GeoDataFrame(
        columns=["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]
    )


def merge_labels(
    mrds_gdf: gpd.GeoDataFrame,
    mindat_gdf: gpd.GeoDataFrame,
    dedup_distance_m: float = 1000.0,
) -> gpd.GeoDataFrame:
    """Merge and deduplicate mineral deposit records from multiple sources.

    Points within ``dedup_distance_m`` meters of each other are considered
    duplicates; the MRDS record is kept in such cases.

    Parameters
    ----------
    mrds_gdf : GeoDataFrame
        Deposit records from MRDS.
    mindat_gdf : GeoDataFrame
        Deposit records from MINDAT.
    dedup_distance_m : float
        Distance threshold in meters for deduplication (default: 1000).

    Returns
    -------
    GeoDataFrame
        Merged and deduplicated deposit records.
    """
    expected_cols = ["name", "commodity", "latitude", "longitude", "deposit_type", "geometry"]

    # Handle empty inputs
    if mrds_gdf.empty and mindat_gdf.empty:
        return gpd.GeoDataFrame(columns=expected_cols)
    if mindat_gdf.empty:
        return mrds_gdf.copy()
    if mrds_gdf.empty:
        return mindat_gdf.copy()

    # Tag the source
    mrds_gdf = mrds_gdf.copy()
    mindat_gdf = mindat_gdf.copy()
    mrds_gdf["source"] = "mrds"
    mindat_gdf["source"] = "mindat"

    # Concatenate
    merged = gpd.GeoDataFrame(
        gpd.pd.concat([mrds_gdf, mindat_gdf], ignore_index=True),
        crs=mrds_gdf.crs,
    )

    # Deduplicate: project to a metric CRS for distance calculation
    merged_proj = merged.to_crs("EPSG:3857")

    keep_mask = [True] * len(merged_proj)
    coords = np.array(
        [(geom.x, geom.y) for geom in merged_proj.geometry]
    )

    for i in range(len(merged_proj)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(merged_proj)):
            if not keep_mask[j]:
                continue
            dist = np.sqrt(
                (coords[i, 0] - coords[j, 0]) ** 2
                + (coords[i, 1] - coords[j, 1]) ** 2
            )
            if dist < dedup_distance_m:
                # Keep MRDS record, drop MINDAT duplicate
                if merged.iloc[j]["source"] == "mindat":
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False

    result = merged[keep_mask].copy()
    result = result.drop(columns=["source"], errors="ignore")

    # Ensure expected columns
    for col in expected_cols:
        if col not in result.columns:
            result[col] = np.nan

    result = result[expected_cols].reset_index(drop=True)

    logger.info(
        "Merged labels: %d MRDS + %d MINDAT -> %d unique deposits",
        len(mrds_gdf), len(mindat_gdf), len(result),
    )
    return result
