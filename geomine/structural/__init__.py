"""Structural geology and terrain analysis for mineral prospectivity mapping.

Submodules
----------
terrain
    DEM-derived terrain features (slope, aspect, curvature, hillshade).
lineaments
    Lineament extraction, density, and intersection analysis.
proximity
    Distance-to-feature, buffered density, and drainage density.
"""

from geomine.structural.terrain import (
    compute_aspect,
    compute_curvature,
    compute_hillshade,
    compute_multi_hillshade,
    compute_slope,
)
from geomine.structural.lineaments import (
    compute_lineament_density,
    compute_lineament_intersections,
    extract_lineaments,
)
from geomine.structural.proximity import (
    compute_buffered_density,
    compute_distance_to_features,
    compute_drainage_density,
)

__all__ = [
    "compute_slope",
    "compute_aspect",
    "compute_curvature",
    "compute_hillshade",
    "compute_multi_hillshade",
    "extract_lineaments",
    "compute_lineament_density",
    "compute_lineament_intersections",
    "compute_distance_to_features",
    "compute_buffered_density",
    "compute_drainage_density",
]
