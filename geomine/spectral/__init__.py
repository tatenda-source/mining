"""Spectral index computation for mineral exploration."""

from geomine.spectral.indices import (
    aloh_minerals,
    carbonate_index,
    clay_ratio,
    clay_swir,
    ferric_iron,
    ferrous_iron,
    iron_oxide_ratio,
    mgoh_minerals,
    ndvi,
    silica_index,
)

__all__ = [
    # Sentinel-2
    "clay_ratio",
    "iron_oxide_ratio",
    "ferric_iron",
    "ndvi",
    "ferrous_iron",
    "clay_swir",
    # ASTER
    "aloh_minerals",
    "mgoh_minerals",
    "silica_index",
    "carbonate_index",
]
