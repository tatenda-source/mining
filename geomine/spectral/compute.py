"""Pipeline for computing spectral indices from satellite imagery.

Reads Sentinel-2 and ASTER rasters, computes all relevant spectral indices,
and writes each index as a single-band GeoTIFF.  A final stacking step
combines all indices into a multi-band feature raster for ML ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject

from geomine.spectral import indices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_index_raster(
    data: np.ndarray,
    profile: dict[str, Any],
    output_path: Path,
    band_name: str,
) -> Path:
    """Write a single-band float32 index raster to disk.

    The output GeoTIFF preserves the CRS and transform from *profile* and
    uses ``np.nan`` as the nodata sentinel.  LZW compression is applied.
    """
    profile = profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
        driver="GTiff",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)
        dst.update_tags(band_name=band_name)
        dst.set_band_description(1, band_name)

    logger.info("Wrote index %s -> %s", band_name, output_path)
    return output_path


def _read_band(src: rasterio.DatasetReader, band_idx: int) -> np.ndarray:
    """Read a single band as float32, converting nodata pixels to np.nan."""
    data = src.read(band_idx).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        data[data == np.float32(nodata)] = np.nan
    return data


# ===================================================================
# Sentinel-2 pipeline
# ===================================================================

# Band order in the expected stacked raster.
_S2_BAND_ORDER = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
_S2_BAND_INDEX = {name: idx + 1 for idx, name in enumerate(_S2_BAND_ORDER)}

# Index definitions: (function, required_bands, display_name)
_S2_INDEX_DEFS: list[tuple[str, Any, list[str]]] = [
    ("clay_ratio",    indices.clay_ratio,    ["B11", "B12"]),
    ("iron_oxide",    indices.iron_oxide_ratio, ["B04", "B02"]),
    ("ferric_iron",   indices.ferric_iron,   ["B8A", "B04"]),
    ("ndvi",          indices.ndvi,           ["B08", "B04"]),
    ("ferrous_iron",  indices.ferrous_iron,  ["B11", "B8A"]),
    ("clay_swir",     indices.clay_swir,     ["B11", "B12"]),
]


def compute_sentinel2_indices(
    stacked_raster_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Compute all Sentinel-2 spectral indices from a stacked raster.

    Parameters
    ----------
    stacked_raster_path : path-like
        Path to a 7-band stacked Sentinel-2 GeoTIFF.  Band order must be:
        B02, B03, B04, B08, B8A, B11, B12.
    output_dir : path-like
        Directory where individual index GeoTIFFs will be written.
    config : dict, optional
        Project configuration dictionary (currently unused; reserved for
        future threshold / masking options).

    Returns
    -------
    dict[str, Path]
        Mapping from index name to the path of the output GeoTIFF.
    """
    stacked_raster_path = Path(stacked_raster_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading stacked Sentinel-2 raster: %s", stacked_raster_path)

    with rasterio.open(stacked_raster_path) as src:
        if src.count < len(_S2_BAND_ORDER):
            raise ValueError(
                f"Expected at least {len(_S2_BAND_ORDER)} bands, "
                f"got {src.count} in {stacked_raster_path}"
            )

        # Read all required bands into a dict keyed by band name.
        bands: dict[str, np.ndarray] = {}
        for name, idx in _S2_BAND_INDEX.items():
            bands[name] = _read_band(src, idx)

        profile = src.profile.copy()

    results: dict[str, Path] = {}

    for index_name, func, band_names in _S2_INDEX_DEFS:
        logger.info("Computing Sentinel-2 index: %s", index_name)
        args = [bands[b] for b in band_names]
        data = func(*args)
        out_path = output_dir / f"s2_{index_name}.tif"
        _write_index_raster(data, profile, out_path, index_name)
        results[index_name] = out_path

    logger.info("Computed %d Sentinel-2 indices", len(results))
    return results


# ===================================================================
# ASTER pipeline
# ===================================================================

# ASTER band file naming convention: the directory is expected to contain
# individual GeoTIFFs named like ``ASTER_B05.tif``, ``ASTER_B07.tif``, etc.
# The prefix is configurable via the *config* dict key ``aster.band_prefix``
# (default ``ASTER_``).

_ASTER_INDEX_DEFS: list[tuple[str, Any, list[str]]] = [
    ("aloh_minerals",   indices.aloh_minerals,   ["B05", "B07"]),
    ("mgoh_minerals",   indices.mgoh_minerals,   ["B07", "B08"]),
    ("silica_index",    indices.silica_index,     ["B13", "B10"]),
    ("carbonate_index", indices.carbonate_index,  ["B13", "B14"]),
]


def _find_aster_band(aster_dir: Path, band_name: str,
                     prefix: str) -> Path:
    """Locate an ASTER band file inside *aster_dir*.

    Tries ``<prefix><band_name>.tif`` first, then a case-insensitive glob.
    """
    candidate = aster_dir / f"{prefix}{band_name}.tif"
    if candidate.exists():
        return candidate

    # Fallback: case-insensitive glob
    pattern = f"*{band_name}*"
    matches = list(aster_dir.glob(pattern))
    tif_matches = [m for m in matches if m.suffix.lower() in (".tif", ".tiff")]
    if tif_matches:
        return tif_matches[0]

    raise FileNotFoundError(
        f"Cannot find ASTER {band_name} in {aster_dir} "
        f"(tried {candidate.name} and glob {pattern})"
    )


def compute_aster_indices(
    aster_dir: str | Path,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Compute all ASTER spectral indices.

    Parameters
    ----------
    aster_dir : path-like
        Directory containing individual ASTER band GeoTIFFs.
    output_dir : path-like
        Directory where index GeoTIFFs will be written.
    config : dict, optional
        Project configuration.  Recognised keys:

        - ``aster.band_prefix`` (str): filename prefix before the band
          number (default ``"ASTER_"``).

    Returns
    -------
    dict[str, Path]
        Mapping from index name to output path.
    """
    aster_dir = Path(aster_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = "ASTER_"
    if config and "aster" in config:
        prefix = config["aster"].get("band_prefix", prefix)

    # Determine the set of unique bands we need.
    required_bands: set[str] = set()
    for _, _, band_names in _ASTER_INDEX_DEFS:
        required_bands.update(band_names)

    # Read bands.
    bands: dict[str, np.ndarray] = {}
    profile: dict[str, Any] | None = None

    for band_name in sorted(required_bands):
        band_path = _find_aster_band(aster_dir, band_name, prefix)
        logger.info("Reading ASTER %s from %s", band_name, band_path)
        with rasterio.open(band_path) as src:
            bands[band_name] = _read_band(src, 1)
            if profile is None:
                profile = src.profile.copy()

    if profile is None:
        raise RuntimeError("No ASTER bands found.")

    results: dict[str, Path] = {}

    for index_name, func, band_names in _ASTER_INDEX_DEFS:
        logger.info("Computing ASTER index: %s", index_name)
        args = [bands[b] for b in band_names]
        data = func(*args)
        out_path = output_dir / f"aster_{index_name}.tif"
        _write_index_raster(data, profile, out_path, index_name)
        results[index_name] = out_path

    logger.info("Computed %d ASTER indices", len(results))
    return results


# ===================================================================
# Feature stacking
# ===================================================================

def stack_all_features(
    index_paths: dict[str, Path | str],
    output_path: str | Path,
    target_resolution: float = 30.0,
) -> Path:
    """Stack all index rasters into a single multi-band feature raster.

    Each input index raster is reprojected / resampled to a common 30 m grid
    (matching the coarser ASTER TIR resolution) and written as one band of the
    output GeoTIFF.  Band descriptions and a ``band_names`` tag are set so
    downstream consumers can identify each feature.

    Parameters
    ----------
    index_paths : dict[str, Path]
        Mapping from index name to its single-band GeoTIFF path.
    output_path : path-like
        Path for the output stacked GeoTIFF.
    target_resolution : float
        Target pixel size in the CRS units (default 30 m).

    Returns
    -------
    Path
        The *output_path* as a :class:`pathlib.Path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not index_paths:
        raise ValueError("index_paths is empty; nothing to stack.")

    ordered_names = sorted(index_paths.keys())

    # Use the first raster to establish the target CRS and bounds.
    first_path = Path(index_paths[ordered_names[0]])
    with rasterio.open(first_path) as ref:
        target_crs = ref.crs
        # Compute a union of all bounds so we don't clip any raster.
        left, bottom, right, top = ref.bounds

    for name in ordered_names[1:]:
        with rasterio.open(Path(index_paths[name])) as src:
            b = src.bounds
            left = min(left, b.left)
            bottom = min(bottom, b.bottom)
            right = max(right, b.right)
            top = max(top, b.top)

    # Compute the target transform and dimensions.
    dst_transform = from_bounds(left, bottom, right, top,
                                width=int((right - left) / target_resolution),
                                height=int((top - bottom) / target_resolution))
    dst_width = int((right - left) / target_resolution)
    dst_height = int((top - bottom) / target_resolution)

    if dst_width <= 0 or dst_height <= 0:
        raise ValueError(
            f"Computed invalid raster dimensions ({dst_width}x{dst_height}). "
            "Check that input rasters have valid extents."
        )

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": dst_width,
        "height": dst_height,
        "count": len(ordered_names),
        "crs": target_crs,
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "lzw",
    }

    logger.info(
        "Stacking %d indices into %s  (%dx%d @ %.1f m)",
        len(ordered_names), output_path, dst_width, dst_height,
        target_resolution,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx, name in enumerate(ordered_names, start=1):
            src_path = Path(index_paths[name])
            with rasterio.open(src_path) as src:
                dest_array = np.full((dst_height, dst_width), np.nan,
                                    dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dest_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
                dst.write(dest_array, band_idx)
                dst.set_band_description(band_idx, name)

        dst.update_tags(band_names=",".join(ordered_names))

    logger.info("Feature stack written to %s", output_path)
    return output_path
