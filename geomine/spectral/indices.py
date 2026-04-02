"""Core spectral index calculations for mineral exploration.

All functions accept numpy float32 arrays representing satellite band data
and return float32 arrays.  Nodata (np.nan) is propagated: any computation
involving a nodata pixel produces nodata in the output.  Division by zero
is masked to np.nan.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: npt.NDArray[np.float32],
                denominator: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Element-wise ratio with division-by-zero protection.

    Where *denominator* is zero the result is set to ``np.nan``.  NaN values
    already present in either operand are preserved automatically by IEEE-754
    arithmetic.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator == 0.0, np.nan, numerator / denominator)
    return result.astype(np.float32)


def _safe_normalized_diff(a: npt.NDArray[np.float32],
                          b: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """(A - B) / (A + B) with division-by-zero protection."""
    total = a + b
    diff = a - b
    return _safe_ratio(diff, total)


# ===================================================================
# Sentinel-2 indices
# ===================================================================

def clay_ratio(b11: npt.NDArray[np.float32],
               b12: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Clay mineral ratio (B11 / B12).

    High values indicate the presence of clay minerals such as kaolinite,
    illite and smectite.

    Parameters
    ----------
    b11 : ndarray
        Sentinel-2 Band 11 (SWIR-1, 1610 nm).
    b12 : ndarray
        Sentinel-2 Band 12 (SWIR-2, 2190 nm).

    Returns
    -------
    ndarray
        Float32 array of clay ratio values.
    """
    return _safe_ratio(b11, b12)


def iron_oxide_ratio(b04: npt.NDArray[np.float32],
                     b02: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Iron oxide ratio (B04 / B02).

    Highlights hematite and goethite.  High values suggest ferric iron
    enrichment at the surface.

    Parameters
    ----------
    b04 : ndarray
        Sentinel-2 Band 4 (Red, 665 nm).
    b02 : ndarray
        Sentinel-2 Band 2 (Blue, 490 nm).
    """
    return _safe_ratio(b04, b02)


def ferric_iron(b8a: npt.NDArray[np.float32],
                b04: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Ferric iron index (B8A - B04) / (B8A + B04).

    Normalized difference sensitive to Fe3+ absorption features.

    Parameters
    ----------
    b8a : ndarray
        Sentinel-2 Band 8A (Vegetation Red Edge, 865 nm).
    b04 : ndarray
        Sentinel-2 Band 4 (Red, 665 nm).
    """
    return _safe_normalized_diff(b8a, b04)


def ndvi(b08: npt.NDArray[np.float32],
         b04: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalized Difference Vegetation Index (B08 - B04) / (B08 + B04).

    Used to mask vegetated areas that may obscure mineral signatures.

    Parameters
    ----------
    b08 : ndarray
        Sentinel-2 Band 8 (NIR, 842 nm).
    b04 : ndarray
        Sentinel-2 Band 4 (Red, 665 nm).
    """
    return _safe_normalized_diff(b08, b04)


def ferrous_iron(b11: npt.NDArray[np.float32],
                 b8a: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Ferrous iron ratio (B11 / B8A).

    Sensitive to ferrous silicates such as chlorite and biotite.

    Parameters
    ----------
    b11 : ndarray
        Sentinel-2 Band 11 (SWIR-1, 1610 nm).
    b8a : ndarray
        Sentinel-2 Band 8A (Vegetation Red Edge, 865 nm).
    """
    return _safe_ratio(b11, b8a)


def clay_swir(b11: npt.NDArray[np.float32],
              b12: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalized clay SWIR index (B11 - B12) / (B11 + B12).

    A normalized variant of the clay ratio that is more robust to
    illumination differences.

    Parameters
    ----------
    b11 : ndarray
        Sentinel-2 Band 11 (SWIR-1, 1610 nm).
    b12 : ndarray
        Sentinel-2 Band 12 (SWIR-2, 2190 nm).
    """
    return _safe_normalized_diff(b11, b12)


# ===================================================================
# ASTER indices
# ===================================================================

def aloh_minerals(b05: npt.NDArray[np.float32],
                  b07: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Al-OH mineral index (B05 / B07).

    Detects aluminium-hydroxyl minerals (muscovite, kaolinite).  Used as
    a gold pathfinder in orogenic gold systems.

    Parameters
    ----------
    b05 : ndarray
        ASTER Band 5 (SWIR, 2167 nm).
    b07 : ndarray
        ASTER Band 7 (SWIR, 2262 nm).
    """
    return _safe_ratio(b05, b07)


def mgoh_minerals(b07: npt.NDArray[np.float32],
                  b08: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Mg-OH mineral index (B07 / B08).

    Detects magnesium-hydroxyl minerals (serpentine, talc, chlorite).
    Used as a pathfinder for nickel and chromium deposits.

    Parameters
    ----------
    b07 : ndarray
        ASTER Band 7 (SWIR, 2262 nm).
    b08 : ndarray
        ASTER Band 8 (SWIR, 2336 nm).
    """
    return _safe_ratio(b07, b08)


def silica_index(b13: npt.NDArray[np.float32],
                 b10: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Silica index (B13 / B10).

    High values indicate quartz-rich zones associated with silicification
    and hydrothermal alteration.

    Parameters
    ----------
    b13 : ndarray
        ASTER Band 13 (TIR, 10.66 um).
    b10 : ndarray
        ASTER Band 10 (TIR, 8.29 um).
    """
    return _safe_ratio(b13, b10)


def carbonate_index(b13: npt.NDArray[np.float32],
                    b14: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Carbonate index (B13 / B14).

    Highlights limestone and dolomite occurrences.

    Parameters
    ----------
    b13 : ndarray
        ASTER Band 13 (TIR, 10.66 um).
    b14 : ndarray
        ASTER Band 14 (TIR, 11.32 um).
    """
    return _safe_ratio(b13, b14)
