"""Process all downloaded Sentinel-2 tiles and run signal check across all deposits.

Usage:
    conda activate geomine
    python scripts/process_and_check.py
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling


RAW_DIR = Path("data/raw/sentinel2")
PROCESSED_DIR = Path("data/processed/sentinel2")
FEATURE_DIR = Path("data/processed/features/spectral")
OUTPUT_DIR = Path("data/outputs")

BAND_ORDER = [
    ("B02", "B02_10m"),
    ("B03", "B03_10m"),
    ("B04", "B04_10m"),
    ("B08", "B08_10m"),
    ("B8A", "B8A_20m"),
    ("B11", "B11_20m"),
    ("B12", "B12_20m"),
]


def extract_and_stack(zip_path: Path) -> Path | None:
    """Extract needed bands from a Sentinel-2 zip and stack into a GeoTIFF."""
    tile_id = zip_path.stem.split("_")[5]
    stack_path = PROCESSED_DIR / f"{tile_id}_stacked.tif"

    if stack_path.exists():
        print(f"  {tile_id}: already stacked")
        return stack_path

    extract_dir = RAW_DIR / "extracted" / tile_id
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {tile_id}: extracting bands...", end=" ", flush=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_files = zf.namelist()
        band_paths: dict[str, Path] = {}

        for band_name, band_key in BAND_ORDER:
            matching = [f for f in all_files if band_key in f and f.endswith(".jp2")]
            if not matching:
                print(f"\n    WARNING: {band_key} not found in {zip_path.name}")
                return None
            member = matching[0]
            out_name = member.split("/")[-1]
            out_path = extract_dir / out_name
            if not out_path.exists():
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            band_paths[band_name] = out_path

    # Get target shape from 10m band
    with rasterio.open(band_paths["B02"]) as ref:
        profile = ref.profile.copy()
        target_h, target_w = ref.height, ref.width

    profile.update(driver="GTiff", count=7, dtype="float32", tiled=True, blockxsize=512, blockysize=512)
    profile.pop("nodata", None)

    print("stacking...", end=" ", flush=True)

    with rasterio.open(str(stack_path), "w", **profile) as dst:
        for i, (band_name, _) in enumerate(BAND_ORDER, 1):
            with rasterio.open(band_paths[band_name]) as src:
                data = src.read(
                    1, out_shape=(target_h, target_w), resampling=Resampling.bilinear
                ).astype(np.float32)
            data = data / 10000.0
            dst.write(data, i)
            dst.set_band_description(i, band_name)

    print(f"done ({stack_path.stat().st_size / (1024**2):.0f} MB)")
    return stack_path


def compute_indices(stack_path: Path) -> dict[str, Path]:
    """Compute spectral indices from a stacked raster."""
    tile_id = stack_path.stem.replace("_stacked", "")
    out_dir = FEATURE_DIR / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, dtype="float32")
        b02 = src.read(1).astype(np.float32)
        b04 = src.read(3).astype(np.float32)
        b08 = src.read(4).astype(np.float32)
        b8a = src.read(5).astype(np.float32)
        b11 = src.read(6).astype(np.float32)
        b12 = src.read(7).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        indices = {
            "clay_ratio": b11 / np.where(b12 > 0, b12, np.float32(1e-6)),
            "iron_oxide": b04 / np.where(b02 > 0, b02, np.float32(1e-6)),
            "ferric_iron": (b8a - b04) / np.where((b8a + b04) > 0, b8a + b04, np.float32(1e-6)),
            "ndvi": (b08 - b04) / np.where((b08 + b04) > 0, b08 + b04, np.float32(1e-6)),
            "ferrous_iron": b11 / np.where(b8a > 0, b8a, np.float32(1e-6)),
            "clay_swir": (b11 - b12) / np.where((b11 + b12) > 0, b11 + b12, np.float32(1e-6)),
        }

    paths = {}
    for name, data in indices.items():
        data = np.clip(data, -10, 10).astype(np.float32)
        out_path = out_dir / f"{name}.tif"
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, name)
        paths[name] = out_path

    return paths


def sample_index_at_points(
    index_path: Path, coords: list[tuple[float, float]]
) -> list[float]:
    """Sample an index raster at given coordinates, returning valid values."""
    values = []
    with rasterio.open(index_path) as src:
        for val in src.sample(coords):
            v = float(val[0])
            if np.isfinite(v) and abs(v) < 100 and v != 0:
                values.append(v)
            else:
                values.append(np.nan)
    return values


def run_signal_check() -> None:
    """Run signal check across ALL tiles and ALL deposits."""
    print("\n" + "=" * 70)
    print("FULL SIGNAL CHECK: All tiles, all deposits")
    print("=" * 70)

    deposits = gpd.read_file("data/training/deposits.geojson")
    index_names = ["clay_ratio", "iron_oxide", "ferric_iron", "ndvi", "ferrous_iron", "clay_swir"]

    # Collect all index directories (per-tile + original flat)
    index_dirs = list(FEATURE_DIR.iterdir())
    index_dirs = [d for d in index_dirs if d.is_dir()] + [FEATURE_DIR]

    # For each deposit, try to sample from every tile's indices
    deposit_values: dict[str, dict[str, float]] = defaultdict(dict)
    bg_values: dict[str, list[float]] = defaultdict(list)

    for idx_dir in index_dirs:
        for idx_name in index_names:
            idx_path = idx_dir / f"{idx_name}.tif"
            if not idx_path.exists():
                continue

            with rasterio.open(idx_path) as src:
                crs = src.crs
                bounds = src.bounds

            # Reproject deposits to this raster's CRS
            deps_proj = deposits.to_crs(crs)
            coords = [(g.x, g.y) for g in deps_proj.geometry]
            vals = sample_index_at_points(idx_path, coords)

            for j, (_, row) in enumerate(deposits.iterrows()):
                if np.isfinite(vals[j]):
                    deposit_values[row["name"]][idx_name] = vals[j]

            # Background sample
            with rasterio.open(idx_path) as src:
                data = src.read(1)
                valid = data[(data != 0) & np.isfinite(data) & (np.abs(data) < 100)]
                if len(valid) > 2000:
                    rng = np.random.RandomState(42)
                    bg_values[idx_name].extend(rng.choice(valid, 2000, replace=False).tolist())
                else:
                    bg_values[idx_name].extend(valid.tolist())

    # Compute statistics
    print(f"\nDeposits with spectral data: {len(deposit_values)} / {len(deposits)}")
    for name, vals in sorted(deposit_values.items()):
        n_indices = len(vals)
        comm = deposits[deposits["name"] == name].iloc[0]["commodity"]
        print(f"  {name:30s} [{comm:8s}]  {n_indices} indices sampled")

    print("\n" + "=" * 70)
    print("SIGNAL SUMMARY (all deposits, all tiles)")
    print("=" * 70)
    print(
        f"\n{'Index':<20s}  {'Dep Mean':>10s}  {'BG Mean':>10s}  "
        f"{'Z-score':>8s}  {'%ile':>6s}  {'n':>3s}  {'Signal':>8s}"
    )
    print("-" * 75)

    results: dict[str, dict] = {}
    signal_count = 0

    for idx_name in index_names:
        dep_vals = [
            deposit_values[name].get(idx_name)
            for name in deposit_values
            if idx_name in deposit_values[name]
        ]
        dep_vals = [v for v in dep_vals if v is not None and np.isfinite(v)]
        bg = np.array(bg_values.get(idx_name, []))

        if len(dep_vals) < 2 or len(bg) < 100:
            print(f"{idx_name:<20s}  {'--':>10s}  {'--':>10s}  {'--':>8s}  {'--':>6s}  {len(dep_vals):>3d}  {'NO DATA':>8s}")
            continue

        dep_arr = np.array(dep_vals)
        dep_mean = float(np.mean(dep_arr))
        bg_mean = float(np.mean(bg))
        bg_std = float(np.std(bg))
        z = (dep_mean - bg_mean) / bg_std if bg_std > 0 else 0
        pct = float(np.mean(bg < dep_mean) * 100)

        sig = (
            "STRONG" if abs(z) > 1.5
            else "MODERATE" if abs(z) > 0.8
            else "WEAK" if abs(z) > 0.3
            else "NONE"
        )
        if sig in ("STRONG", "MODERATE"):
            signal_count += 1

        results[idx_name] = {
            "deposit_mean": dep_mean,
            "background_mean": bg_mean,
            "z_score": z,
            "percentile": pct,
            "n_deposits": len(dep_vals),
            "signal": sig,
        }
        print(
            f"{idx_name:<20s}  {dep_mean:10.4f}  {bg_mean:10.4f}  "
            f"{z:+8.2f}  {pct:5.0f}%  {len(dep_vals):3d}  {sig:>8s}"
        )

    # Per-deposit detail
    print("\n" + "-" * 70)
    print("PER-DEPOSIT VALUES")
    print("-" * 70)
    for name in sorted(deposit_values.keys()):
        vals = deposit_values[name]
        comm = deposits[deposits["name"] == name].iloc[0]["commodity"]
        vals_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(vals.items()))
        print(f"  {name:30s} [{comm:6s}]  {vals_str}")

    # Verdict
    print("\n" + "=" * 70)
    n_total = len([v for v in results.values() if v["n_deposits"] >= 2])
    if signal_count >= 3:
        verdict = "STRONG signal. Multiple indices discriminate deposits from background."
    elif signal_count >= 1:
        verdict = "MODERATE signal. Some indices discriminate. Proceed with caution."
    else:
        verdict = "WEAK/NO signal. Investigate before proceeding to ML."
    print(f"VERDICT: {verdict}")
    print(f"Signal detected in {signal_count}/{n_total} indices")
    print("=" * 70)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "signal_check_full.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "summary": results,
                "per_deposit": dict(deposit_values),
                "verdict": verdict,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved to {out_path}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Process all zip files
    zip_files = sorted(RAW_DIR.glob("*.zip"))
    print(f"Found {len(zip_files)} Sentinel-2 zip files\n")

    for zip_path in zip_files:
        print(f"Processing {zip_path.name}...")
        stack_path = extract_and_stack(zip_path)
        if stack_path:
            print(f"  Computing spectral indices...")
            compute_indices(stack_path)

    # Also process existing stacked rasters that may not have per-tile indices
    existing = PROCESSED_DIR / "20230730_T36KTD_stacked.tif"
    if existing.exists():
        t36ktd_dir = FEATURE_DIR / "T36KTD"
        if not t36ktd_dir.exists():
            print(f"\nComputing indices for existing T36KTD stack...")
            # Use the flat spectral dir as T36KTD
            # (indices already computed there)

    run_signal_check()


if __name__ == "__main__":
    main()
