"""ASTER L1T scene search and download via NASA EarthData (CMR API).

CRITICAL NOTE: ASTER SWIR bands (4-9) failed on 2008-04-23 due to detector
overheating. Only data acquired BEFORE April 2008 has usable SWIR bands.
Post-2008 SWIR data is saturated and unusable.

For mineral discrimination (Mg-OH, Al-OH, carbonate indices), you MUST
constrain searches to the 2000-03-01 to 2008-04-01 temporal window.

This module supports two download methods:
1. earthaccess library (preferred -- handles auth, cloud-hosted data)
2. Direct CMR REST API + HTTPS download (fallback)

Product: AST_L1T Version 004 (Cloud Optimized GeoTIFF)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CMR granule search endpoint
CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# ASTER L1T product identifiers
ASTER_SHORT_NAME = "AST_L1T"
ASTER_VERSION_V004 = "004"
ASTER_VERSION_V031 = "031"  # fallback if V004 not yet available for a scene

# SWIR detector failure date -- ONLY search before this for SWIR bands
SWIR_CUTOFF_DATE = "2008-04-01"

# ASTER band subsystems
ASTER_SUBSYSTEMS = {
    "vnir": {"bands": ["B01", "B02", "B3N"], "resolution_m": 15},
    "swir": {"bands": ["B04", "B05", "B06", "B07", "B08", "B09"], "resolution_m": 30},
    "tir": {"bands": ["B10", "B11", "B12", "B13", "B14"], "resolution_m": 90},
}


# ---------------------------------------------------------------------------
# Search via CMR REST API
# ---------------------------------------------------------------------------


def search_aster_cmr(
    bbox: list[float],
    date_start: str = "2000-03-01",
    date_end: str = "2008-04-01",
    version: str = ASTER_VERSION_V004,
    max_results: int = 200,
    earthdata_token: str | None = None,
) -> list[dict]:
    """Search for ASTER L1T granules via the CMR REST API.

    Parameters
    ----------
    bbox : list[float]
        [west, south, east, north] in WGS84 degrees.
    date_start : str
        Start date (ISO format). Default is ASTER launch.
    date_end : str
        End date (ISO format). Default is SWIR failure cutoff.
    version : str
        Product version. Use "004" for the latest reprocessed data.
    max_results : int
        Maximum number of granules to return.
    earthdata_token : str, optional
        NASA EarthData bearer token.

    Returns
    -------
    list[dict]
        List of granule metadata dictionaries from CMR.
    """
    token = earthdata_token or os.environ.get("EARTHDATA_TOKEN")

    west, south, east, north = bbox
    params = {
        "short_name": ASTER_SHORT_NAME,
        "version": version,
        "bounding_box": f"{west},{south},{east},{north}",
        "temporal": f"{date_start}T00:00:00Z,{date_end}T23:59:59Z",
        "page_size": min(max_results, 2000),
        "sort_key": "-start_date",  # newest first
    }

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    logger.info(
        "Searching CMR for %s v%s: bbox=%s, temporal=%s/%s",
        ASTER_SHORT_NAME, version, bbox, date_start, date_end,
    )

    all_granules: list[dict] = []
    page = 1

    while len(all_granules) < max_results:
        params["page_num"] = page
        resp = requests.get(
            CMR_GRANULE_URL, params=params, headers=headers, timeout=60,
        )
        resp.raise_for_status()

        data = resp.json()
        feed = data.get("feed", {})
        entries = feed.get("entry", [])

        if not entries:
            break

        all_granules.extend(entries)
        logger.info("CMR page %d: %d granules (total: %d)", page, len(entries), len(all_granules))

        # CMR returns fewer than page_size when no more results
        if len(entries) < params["page_size"]:
            break
        page += 1

    logger.info("Found %d ASTER L1T granules total", len(all_granules))
    return all_granules[:max_results]


def _extract_download_links(granule: dict) -> list[str]:
    """Extract HTTPS download URLs from a CMR granule entry.

    Parameters
    ----------
    granule : dict
        A single granule entry from CMR search results.

    Returns
    -------
    list[str]
        Download URLs (typically HDF or COG files).
    """
    links = granule.get("links", [])
    download_urls = []
    for link in links:
        href = link.get("href", "")
        rel = link.get("rel", "")
        # Look for data download links (not metadata, not browse images)
        if (
            "http" in href
            and rel in ("http://esipfed.org/ns/fedsearch/1.1/data#", "enclosure")
            and not href.endswith(".xml")
            and not href.endswith(".jpg")
            and not href.endswith(".png")
        ):
            download_urls.append(href)
    return download_urls


def download_aster_granules(
    granules: list[dict],
    output_dir: str | Path,
    earthdata_token: str | None = None,
    max_granules: int | None = None,
) -> list[Path]:
    """Download ASTER L1T granules from NASA EarthData.

    Parameters
    ----------
    granules : list[dict]
        Granule metadata from :func:`search_aster_cmr`.
    output_dir : str or Path
        Directory to save downloaded files.
    earthdata_token : str, optional
        NASA EarthData bearer token.
    max_granules : int, optional
        Limit the number of granules to download.

    Returns
    -------
    list[Path]
        Paths to downloaded granule files.
    """
    token = earthdata_token or os.environ.get("EARTHDATA_TOKEN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    to_download = granules[:max_granules] if max_granules else granules
    downloaded: list[Path] = []

    for i, granule in enumerate(to_download, 1):
        granule_id = granule.get("producer_granule_id", granule.get("title", f"granule_{i}"))
        urls = _extract_download_links(granule)

        if not urls:
            logger.warning("No download URLs for granule %s, skipping", granule_id)
            continue

        for url in urls:
            filename = url.split("/")[-1].split("?")[0]
            dst_file = output_dir / filename

            if dst_file.exists():
                logger.info("Already downloaded: %s", filename)
                downloaded.append(dst_file)
                continue

            logger.info("[%d/%d] Downloading %s", i, len(to_download), filename)
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=600,
                    allow_redirects=True,
                )
                resp.raise_for_status()

                with open(dst_file, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8192):
                        fh.write(chunk)

                size_mb = dst_file.stat().st_size / (1024 * 1024)
                downloaded.append(dst_file)
                logger.info("Saved: %s (%.1f MB)", filename, size_mb)

            except requests.RequestException as exc:
                logger.error("Failed to download %s: %s", filename, exc)
                if dst_file.exists():
                    dst_file.unlink()

    logger.info("Downloaded %d files from %d granules", len(downloaded), len(to_download))
    return downloaded


# ---------------------------------------------------------------------------
# Search + download via earthaccess (preferred method)
# ---------------------------------------------------------------------------


def search_and_download_earthaccess(
    bbox: list[float],
    output_dir: str | Path,
    date_start: str = "2000-03-01",
    date_end: str = "2008-04-01",
    max_results: int = 50,
) -> list[Path]:
    """Search and download ASTER L1T using the earthaccess library.

    This is the recommended method. It handles NASA EarthData authentication
    and supports cloud-hosted data access.

    Requires: ``pip install earthaccess``

    Parameters
    ----------
    bbox : list[float]
        [west, south, east, north] in WGS84 degrees.
    output_dir : str or Path
        Directory to save downloaded files.
    date_start : str
        Start date (ISO format).
    date_end : str
        End date (ISO format). Defaults to SWIR cutoff.
    max_results : int
        Maximum granules to return.

    Returns
    -------
    list[Path]
        Paths to downloaded files.
    """
    try:
        import earthaccess
    except ImportError:
        raise ImportError(
            "earthaccess is required for this method. "
            "Install with: pip install earthaccess"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # earthaccess will use EARTHDATA_TOKEN env var or prompt for login
    earthaccess.login()

    logger.info(
        "Searching ASTER L1T via earthaccess: bbox=%s, dates=%s/%s",
        bbox, date_start, date_end,
    )

    results = earthaccess.search_data(
        short_name=ASTER_SHORT_NAME,
        version=ASTER_VERSION_V004,
        bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3]),
        temporal=(date_start, date_end),
        count=max_results,
    )

    logger.info("earthaccess found %d ASTER L1T granules", len(results))

    if not results:
        # Fallback to V031 if V004 reprocessing not yet complete for region
        logger.info("Trying V031 as fallback...")
        results = earthaccess.search_data(
            short_name=ASTER_SHORT_NAME,
            version=ASTER_VERSION_V031,
            bounding_box=(bbox[0], bbox[1], bbox[2], bbox[3]),
            temporal=(date_start, date_end),
            count=max_results,
        )
        logger.info("earthaccess found %d V031 granules", len(results))

    if not results:
        logger.warning("No ASTER L1T granules found for the specified parameters")
        return []

    downloaded = earthaccess.download(results, str(output_dir))
    logger.info("Downloaded %d files to %s", len(downloaded), output_dir)
    return [Path(f) for f in downloaded]


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def download_aster(
    config: dict[str, Any],
    earthdata_token: str | None = None,
    method: str = "earthaccess",
    max_granules: int = 50,
) -> list[Path]:
    """Download ASTER L1T data for the project AOI.

    Parameters
    ----------
    config : dict
        Project configuration (must have ``aoi.bbox`` and ``data.raw_dir``).
    earthdata_token : str, optional
        NASA EarthData bearer token.
    method : str
        "earthaccess" (recommended) or "cmr" (direct CMR API).
    max_granules : int
        Maximum number of granules to download.

    Returns
    -------
    list[Path]
        Paths to downloaded ASTER files.
    """
    bbox = config["aoi"]["bbox"]
    base_dir = Path(config.get("_base_dir", "."))
    raw_dir = base_dir / config["data"]["raw_dir"] / "aster"

    # SWIR data only available before April 2008
    date_start = "2000-03-01"
    date_end = SWIR_CUTOFF_DATE

    aster_cfg = config.get("data", {}).get("aster", {})
    if aster_cfg:
        # Allow config override, but warn if post-2008
        cfg_end = aster_cfg.get("date_end", date_end)
        if cfg_end > SWIR_CUTOFF_DATE:
            logger.warning(
                "Config date_end=%s is after SWIR failure (%s). "
                "SWIR bands will be unusable for post-April-2008 data. "
                "Clamping to %s.",
                cfg_end, SWIR_CUTOFF_DATE, SWIR_CUTOFF_DATE,
            )
            cfg_end = SWIR_CUTOFF_DATE
        date_end = cfg_end
        date_start = aster_cfg.get("date_start", date_start)

    logger.info("ASTER download: method=%s, bbox=%s, dates=%s/%s", method, bbox, date_start, date_end)

    if method == "earthaccess":
        return search_and_download_earthaccess(
            bbox=bbox,
            output_dir=raw_dir,
            date_start=date_start,
            date_end=date_end,
            max_results=max_granules,
        )
    elif method == "cmr":
        token = earthdata_token or os.environ.get("EARTHDATA_TOKEN")
        granules = search_aster_cmr(
            bbox=bbox,
            date_start=date_start,
            date_end=date_end,
            earthdata_token=token,
        )
        return download_aster_granules(
            granules=granules,
            output_dir=raw_dir,
            earthdata_token=token,
            max_granules=max_granules,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'earthaccess' or 'cmr'.")
