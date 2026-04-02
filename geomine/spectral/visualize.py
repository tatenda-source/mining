"""Visualization helpers for spectral index rasters.

Provides matplotlib plotting and QGIS .qml style generation for quick
inspection of computed indices.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band composite layer presets (Sentinel-2 stacked raster)
# Band order in stacked raster: B02, B03, B04, B08, B8A, B11, B12
#                                 1     2     3     4     5     6     7
# ---------------------------------------------------------------------------

LAYER_PRESETS: dict[str, dict[str, Any]] = {
    "true_color": {
        "name": "True Color",
        "bands": (3, 2, 1),       # B04, B03, B02 (Red, Green, Blue)
        "description": "Natural color as the human eye sees it",
        "use": "Baseline context, vegetation, water bodies, urbanization",
    },
    "false_color": {
        "name": "False Color (Vegetation)",
        "bands": (4, 3, 2),       # B08, B04, B03 (NIR, Red, Green)
        "description": "Vegetation appears bright red, bare soil cyan/brown",
        "use": "Vegetation health, geobotany anomalies over mineralized ground",
    },
    "geology_12_8_2": {
        "name": "Geology (12, 8, 2)",
        "bands": (7, 4, 1),       # B12, B08, B02 (SWIR2, NIR, Blue)
        "description": "Clay minerals appear pink/magenta, iron oxides orange/brown, vegetation green",
        "use": "Hydrothermal alteration mapping, lithology discrimination",
    },
    "geology_8_11_12": {
        "name": "Geology (8, 11, 12)",
        "bands": (4, 6, 7),       # B08, B11, B12 (NIR, SWIR1, SWIR2)
        "description": "Discriminates clay types, ferrous vs ferric iron, carbonate outcrops",
        "use": "Mineral group discrimination, alteration zonation",
    },
    "swir": {
        "name": "SWIR Composite",
        "bands": (7, 5, 3),       # B12, B8A, B04 (SWIR2, NIR narrow, Red)
        "description": "Enhanced clay and iron oxide discrimination",
        "use": "Best single composite for mineral exploration -- highlights alteration halos",
    },
    "iron_detection": {
        "name": "Iron Detection",
        "bands": (7, 3, 1),       # B12, B04, B02 (SWIR2, Red, Blue)
        "description": "Iron-bearing minerals appear bright red/orange",
        "use": "Ferricrete, laterite, gossans, iron oxide caps over sulphide deposits",
    },
    "clay_discrimination": {
        "name": "Clay Discrimination",
        "bands": (7, 6, 4),       # B12, B11, B08 (SWIR2, SWIR1, NIR)
        "description": "Different clay types show as distinct colors",
        "use": "Kaolinite vs illite vs smectite zones, weathering profiles",
    },
}


def render_composite(
    stacked_raster_path: str | Path,
    preset: str | tuple[int, int, int],
    output_png: str | Path | None = None,
    title: str | None = None,
    percentile_stretch: tuple[float, float] = (2.0, 98.0),
    figsize: tuple[float, float] = (12, 10),
    gamma: float = 1.0,
) -> None:
    """Render an RGB band composite from a stacked Sentinel-2 raster.

    Parameters
    ----------
    stacked_raster_path : path-like
        Path to the 7-band stacked Sentinel-2 GeoTIFF
        (band order: B02, B03, B04, B08, B8A, B11, B12).
    preset : str or tuple of 3 ints
        Either a preset name from LAYER_PRESETS (e.g. ``"geology_12_8_2"``)
        or a custom tuple of 1-based band indices for (R, G, B).
    output_png : path-like, optional
        Save figure to this path. If None, calls plt.show().
    title : str, optional
        Plot title. Auto-generated from preset if None.
    percentile_stretch : tuple
        Low/high percentiles for contrast stretching (default 2nd-98th).
    figsize : tuple
        Figure size in inches.
    gamma : float
        Gamma correction (< 1 brightens shadows, > 1 darkens).
    """
    stacked_raster_path = Path(stacked_raster_path)

    # Resolve preset
    if isinstance(preset, str):
        if preset not in LAYER_PRESETS:
            available = ", ".join(LAYER_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")
        info = LAYER_PRESETS[preset]
        band_indices = info["bands"]
        if title is None:
            title = f"{info['name']}\n{info['description']}"
    else:
        band_indices = preset
        if title is None:
            title = f"Custom Composite (bands {band_indices[0]}, {band_indices[1]}, {band_indices[2]})"

    # Read bands
    with rasterio.open(stacked_raster_path) as src:
        if max(band_indices) > src.count:
            raise ValueError(
                f"Band index {max(band_indices)} exceeds raster band count ({src.count})"
            )

        rgb = np.zeros((src.height, src.width, 3), dtype=np.float32)
        for i, band_idx in enumerate(band_indices):
            data = src.read(band_idx).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == np.float32(nodata)] = np.nan
            rgb[:, :, i] = data

    # Percentile stretch per channel
    for i in range(3):
        channel = rgb[:, :, i]
        valid = channel[np.isfinite(channel)]
        if valid.size == 0:
            continue
        lo = np.percentile(valid, percentile_stretch[0])
        hi = np.percentile(valid, percentile_stretch[1])
        if hi > lo:
            channel = (channel - lo) / (hi - lo)
        else:
            channel = np.zeros_like(channel)
        channel = np.clip(channel, 0, 1)
        rgb[:, :, i] = channel

    # Gamma correction
    if gamma != 1.0:
        valid_mask = np.all(np.isfinite(rgb), axis=2)
        rgb[valid_mask] = np.power(rgb[valid_mask], 1.0 / gamma)

    # Set nodata pixels to white
    nodata_mask = ~np.all(np.isfinite(rgb), axis=2)
    rgb[nodata_mask] = 1.0

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_aspect("equal")
    plt.tight_layout()

    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved composite to %s", output_png)
    else:
        plt.show()


def render_all_presets(
    stacked_raster_path: str | Path,
    output_dir: str | Path,
    gamma: float = 1.2,
) -> dict[str, Path]:
    """Render all geological band composite presets and save as PNGs.

    Returns a dict mapping preset name to output PNG path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for preset_name, info in LAYER_PRESETS.items():
        output_png = output_dir / f"{preset_name}.png"
        try:
            render_composite(
                stacked_raster_path,
                preset=preset_name,
                output_png=output_png,
                gamma=gamma,
            )
            results[preset_name] = output_png
        except Exception as e:
            logger.warning("Failed to render %s: %s", preset_name, e)

    logger.info("Rendered %d/%d presets to %s", len(results), len(LAYER_PRESETS), output_dir)
    return results


# Default colormaps per index family.
_INDEX_CMAPS: dict[str, str] = {
    "ndvi":            "RdYlGn",
    "clay_ratio":      "YlOrBr",
    "clay_swir":       "YlOrBr",
    "iron_oxide":      "hot",
    "ferric_iron":     "hot",
    "ferrous_iron":    "hot",
    "aloh_minerals":   "copper",
    "mgoh_minerals":   "BuGn",
    "silica_index":    "inferno",
    "carbonate_index": "cool",
}


def _guess_cmap(name: str) -> str:
    """Return a sensible default colormap for a given index name."""
    for key, cmap in _INDEX_CMAPS.items():
        if key in name.lower():
            return cmap
    return "viridis"


# ===================================================================
# Single-index map
# ===================================================================

def plot_index_map(
    raster_path: str | Path,
    title: str,
    cmap: str | None = None,
    output_png: str | Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """Read a single-band index raster and render it as a map.

    Parameters
    ----------
    raster_path : path-like
        Path to a single-band GeoTIFF.
    title : str
        Plot title.
    cmap : str, optional
        Matplotlib colormap name.  If *None* a default is chosen based on
        the raster's ``band_name`` tag.
    output_png : path-like, optional
        If given the figure is saved here; otherwise ``plt.show()`` is called.
    vmin, vmax : float, optional
        Colour-scale limits.  If *None* they are derived from the 2nd and
        98th percentiles of valid pixels.
    figsize : tuple
        Figure size in inches.
    """
    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        tags = src.tags()

    if nodata is not None:
        data[data == np.float32(nodata)] = np.nan

    valid = data[np.isfinite(data)]
    if valid.size == 0:
        logger.warning("Raster %s has no valid pixels; skipping plot.", raster_path)
        return

    if vmin is None:
        vmin = float(np.percentile(valid, 2))
    if vmax is None:
        vmax = float(np.percentile(valid, 98))

    if cmap is None:
        band_name = tags.get("band_name", raster_path.stem)
        cmap = _guess_cmap(band_name)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Index value")
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved index map to %s", output_png)
    else:
        plt.show()


# ===================================================================
# Grid of all indices
# ===================================================================

def plot_index_grid(
    index_paths: dict[str, str | Path],
    output_png: str | Path | None = None,
    figsize: tuple[float, float] = (18, 14),
) -> None:
    """Create a grid of all spectral index maps for quick comparison.

    Parameters
    ----------
    index_paths : dict[str, path-like]
        Mapping from index name to single-band GeoTIFF path.
    output_png : path-like, optional
        If given the figure is saved; otherwise shown interactively.
    figsize : tuple
        Figure size in inches.
    """
    names = sorted(index_paths.keys())
    n = len(names)
    if n == 0:
        logger.warning("No indices to plot.")
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)

    for idx, name in enumerate(names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        path = Path(index_paths[name])
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            data[data == np.float32(nodata)] = np.nan

        valid = data[np.isfinite(data)]
        if valid.size > 0:
            vmin = float(np.percentile(valid, 2))
            vmax = float(np.percentile(valid, 98))
        else:
            vmin, vmax = 0.0, 1.0

        cmap = _guess_cmap(name)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplot cells.
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Spectral Indices Overview", fontsize=16, y=1.01)
    plt.tight_layout()

    if output_png is not None:
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved index grid to %s", output_png)
    else:
        plt.show()


# ===================================================================
# QGIS style generation
# ===================================================================

def _mpl_cmap_to_rgba(cmap_name: str, n_stops: int = 10) -> list[tuple[float, int, int, int, int]]:
    """Sample a matplotlib colormap and return (fraction, r, g, b, a) stops."""
    cmap = plt.get_cmap(cmap_name)
    stops: list[tuple[float, int, int, int, int]] = []
    for i in range(n_stops):
        frac = i / (n_stops - 1)
        r, g, b, a = cmap(frac)
        stops.append((frac, int(r * 255), int(g * 255), int(b * 255), int(a * 255)))
    return stops


def generate_qgis_style(
    index_name: str,
    vmin: float,
    vmax: float,
    cmap: str | None = None,
    output_path: str | Path | None = None,
    n_classes: int = 10,
) -> str:
    """Generate a QGIS ``.qml`` style file for a spectral index raster.

    Parameters
    ----------
    index_name : str
        Name of the spectral index (used in labels).
    vmin, vmax : float
        Value range for the graduated colour ramp.
    cmap : str, optional
        Matplotlib colormap name.  If *None* a default is chosen.
    output_path : path-like, optional
        If given the QML is written to this path.
    n_classes : int
        Number of colour-ramp classes (default 10).

    Returns
    -------
    str
        The QML XML as a string.
    """
    if cmap is None:
        cmap = _guess_cmap(index_name)

    stops = _mpl_cmap_to_rgba(cmap, n_classes)

    # Build QML XML tree.
    qgis = ET.Element("qgis", attrib={"version": "3.34"})
    pipe = ET.SubElement(qgis, "pipe")
    renderer = ET.SubElement(pipe, "rasterrenderer", attrib={
        "type": "singlebandpseudocolor",
        "band": "1",
        "opacity": "1",
        "alphaBand": "-1",
        "classificationMin": str(vmin),
        "classificationMax": str(vmax),
    })

    shader = ET.SubElement(renderer, "rastershader")
    color_ramp_shader = ET.SubElement(shader, "colorrampshader", attrib={
        "colorRampType": "INTERPOLATED",
        "classificationMode": "1",
        "minimumValue": str(vmin),
        "maximumValue": str(vmax),
        "labelPrecision": "4",
        "clip": "0",
    })

    for frac, r, g, b, a in stops:
        value = vmin + frac * (vmax - vmin)
        ET.SubElement(color_ramp_shader, "item", attrib={
            "value": f"{value:.6f}",
            "color": f"#{r:02x}{g:02x}{b:02x}",
            "alpha": str(a),
            "label": f"{value:.4f}",
        })

    # Add a colour ramp definition for QGIS interoperability.
    colorramp = ET.SubElement(color_ramp_shader, "colorramp", attrib={
        "name": f"[{index_name}]",
        "type": "gradient",
    })
    first = stops[0]
    last = stops[-1]
    ET.SubElement(colorramp, "prop", attrib={
        "k": "color1", "v": f"{first[1]},{first[2]},{first[3]},{first[4]}",
    })
    ET.SubElement(colorramp, "prop", attrib={
        "k": "color2", "v": f"{last[1]},{last[2]},{last[3]},{last[4]}",
    })

    # Intermediate stops.
    gradient_stops = []
    for frac, r, g, b, a in stops[1:-1]:
        gradient_stops.append(f"{frac:.6f};{r},{g},{b},{a}")
    if gradient_stops:
        ET.SubElement(colorramp, "prop", attrib={
            "k": "stops", "v": ":".join(gradient_stops),
        })

    ET.indent(qgis, space="  ")
    xml_str = ET.tostring(qgis, encoding="unicode", xml_declaration=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_str, encoding="utf-8")
        logger.info("Wrote QGIS style to %s", output_path)

    return xml_str
