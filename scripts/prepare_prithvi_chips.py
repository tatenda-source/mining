#!/usr/bin/env python3
"""
Prepare 224x224 training chips from Sentinel-2 tiles for Prithvi-EO-2.0.

Creates:
  data/processed/chips/       -- 6-band GeoTIFF chips (B02, B03, B04, B8A, B11, B12)
  data/processed/chip_labels/ -- binary mask chips (1=deposit buffer, 0=background)
  data/processed/chip_meta.json -- metadata for each chip (tile, coords, label)

Chip strategy:
  - Positive chips: centered on each deposit with 500m buffer circle
  - Hard negative chips: random locations in same tile, > 2km from any deposit
  - Augmentation: 4 rotations per positive chip (0, 90, 180, 270)
  - Tile assignment preserved for LOTO CV

Prithvi input format:
  - 6 bands: B02(Blue), B03(Green), B04(Red), B8A(NIR), B11(SWIR1), B12(SWIR2)
  - Shape: (6, 224, 224)
  - Values: surface reflectance (0-1 scale from our stacks)
  - Normalization happens at training time using HLS mean/std
"""
from __future__ import annotations

import json
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import from_bounds
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer

# ── Configuration ──
CHIP_SIZE = 224  # pixels (224 x 10m = 2.24km per side)
BUFFER_M = 500   # deposit buffer radius in meters
MIN_NEG_DIST_M = 2000  # minimum distance from deposit for negatives
NEG_PER_TILE = 50  # negative chips per tile
AUGMENT_ROTATIONS = [0, 90, 180, 270]  # rotation augmentation for positives

# Band mapping: our stacks are [B02, B03, B04, B08, B8A, B11, B12]
# Prithvi needs [B02, B03, B04, B8A, B11, B12] = indices [1, 2, 3, 5, 6, 7] (1-indexed)
PRITHVI_BANDS = [1, 2, 3, 5, 6, 7]  # rasterio is 1-indexed

PROCESSED_DIR = Path("data/processed/sentinel2")
CHIPS_DIR = Path("data/processed/chips")
LABELS_DIR = Path("data/processed/chip_labels")
DEPOSITS_PATH = Path("data/training/deposits_curated.geojson")

np.random.seed(42)


def create_deposit_mask(shape, transform, deposits_utm, buffer_px):
    """Create binary mask with 1s in circles around deposits."""
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y in deposits_utm:
        col = int((x - transform[2]) / transform[0])
        row = int((y - transform[5]) / transform[4])
        # Draw filled circle
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        dist = np.sqrt((xx - col)**2 + (yy - row)**2)
        mask[dist <= buffer_px] = 1
    return mask


def extract_chip(src, row_off, col_off, chip_size, bands):
    """Extract a chip from a rasterio dataset."""
    window = Window(col_off, row_off, chip_size, chip_size)
    data = src.read(bands, window=window).astype(np.float32)
    win_transform = src.window_transform(window)
    return data, win_transform


def rotate_chip(data, mask, angle):
    """Rotate chip and mask by angle (0, 90, 180, 270)."""
    k = angle // 90
    if k == 0:
        return data, mask
    return np.rot90(data, k, axes=(1, 2)).copy(), np.rot90(mask, k).copy()


def main():
    CHIPS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRITHVI CHIP PREPARATION")
    print("=" * 70)

    # Load deposits
    deposits = gpd.read_file(DEPOSITS_PATH)
    print(f"Deposits: {len(deposits)}")

    # Find available stacked tiles
    stack_files = sorted(PROCESSED_DIR.glob("*_stacked.tif"))
    print(f"Stacked tiles: {len(stack_files)}")

    chip_meta = []
    chip_id = 0

    for stack_path in stack_files:
        tile_name = stack_path.stem.split("_")[0]
        # Handle naming variations
        if tile_name.startswith("20"):
            tile_name = stack_path.stem.split("_")[1]

        print(f"\n{'─' * 50}")
        print(f"Processing tile: {tile_name} ({stack_path.name})")

        with rasterio.open(stack_path) as src:
            crs = src.crs
            transform = src.transform
            width = src.width
            height = src.height
            res = abs(transform[0])
            n_bands = src.count

            print(f"  Shape: {width}x{height}, Bands: {n_bands}, Res: {res}m, CRS: {crs}")

            if n_bands < 7:
                print(f"  SKIP: need 7 bands, got {n_bands}")
                continue

            # Project deposits to tile CRS
            deps_proj = deposits.to_crs(crs)
            bounds = src.bounds

            # Filter deposits within this tile
            tile_deposits = []
            for idx, row in deps_proj.iterrows():
                x, y = row.geometry.x, row.geometry.y
                if bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top:
                    tile_deposits.append((x, y, deposits.iloc[idx]['name']))

            print(f"  Deposits in tile: {len(tile_deposits)}")

            buffer_px = int(BUFFER_M / res)
            half_chip = CHIP_SIZE // 2
            neg_min_px = int(MIN_NEG_DIST_M / res)

            # ── Positive chips: centered on deposits ──
            pos_count = 0
            for dep_x, dep_y, dep_name in tile_deposits:
                col_center = int((dep_x - transform[2]) / transform[0])
                row_center = int((dep_y - transform[5]) / transform[4])

                col_off = col_center - half_chip
                row_off = row_center - half_chip

                # Bounds check
                if col_off < 0 or row_off < 0:
                    continue
                if col_off + CHIP_SIZE > width or row_off + CHIP_SIZE > height:
                    continue

                # Extract chip
                try:
                    data, win_transform = extract_chip(src, row_off, col_off, CHIP_SIZE, PRITHVI_BANDS)
                except Exception as e:
                    print(f"    Error extracting chip for {dep_name}: {e}")
                    continue

                # Check data quality
                valid_frac = np.mean(data > 0)
                if valid_frac < 0.8:
                    print(f"    {dep_name}: too many zeros ({valid_frac:.0%} valid), skipping")
                    continue

                # Create mask (deposit buffer circle in center)
                mask = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
                yy, xx = np.ogrid[:CHIP_SIZE, :CHIP_SIZE]
                dist = np.sqrt((xx - half_chip)**2 + (yy - half_chip)**2)
                mask[dist <= buffer_px] = 1

                # Save with augmentation
                for angle in AUGMENT_ROTATIONS:
                    rot_data, rot_mask = rotate_chip(data, mask, angle)

                    chip_name = f"chip_{chip_id:05d}"
                    chip_path = CHIPS_DIR / f"{chip_name}.tif"
                    label_path = LABELS_DIR / f"{chip_name}.tif"

                    # Write chip
                    chip_profile = {
                        'driver': 'GTiff',
                        'dtype': 'float32',
                        'width': CHIP_SIZE,
                        'height': CHIP_SIZE,
                        'count': 6,
                        'crs': crs,
                        'transform': win_transform,
                    }
                    with rasterio.open(str(chip_path), 'w', **chip_profile) as dst:
                        dst.write(rot_data)

                    # Write label
                    label_profile = chip_profile.copy()
                    label_profile.update(count=1, dtype='uint8')
                    with rasterio.open(str(label_path), 'w', **label_profile) as dst:
                        dst.write(rot_mask[np.newaxis])

                    chip_meta.append({
                        'chip_id': chip_name,
                        'tile': tile_name,
                        'deposit': dep_name,
                        'label': 'positive',
                        'rotation': angle,
                        'center_x': float(dep_x),
                        'center_y': float(dep_y),
                        'valid_fraction': float(valid_frac),
                        'mask_pixels': int(rot_mask.sum()),
                    })
                    chip_id += 1
                    pos_count += 1

            print(f"  Positive chips (with augmentation): {pos_count}")

            # ── Negative chips: random locations far from deposits ──
            neg_count = 0
            rng = np.random.RandomState(42 + hash(tile_name) % 1000)
            attempts = 0

            while neg_count < NEG_PER_TILE and attempts < NEG_PER_TILE * 10:
                attempts += 1

                col_off = rng.randint(0, max(1, width - CHIP_SIZE))
                row_off = rng.randint(0, max(1, height - CHIP_SIZE))

                # Check distance from all deposits
                chip_cx = transform[2] + (col_off + half_chip) * transform[0]
                chip_cy = transform[5] + (row_off + half_chip) * transform[4]

                too_close = False
                for dep_x, dep_y, _ in tile_deposits:
                    dist_px = np.sqrt(((dep_x - chip_cx)/res)**2 + ((dep_y - chip_cy)/res)**2)
                    if dist_px < neg_min_px:
                        too_close = True
                        break

                if too_close:
                    continue

                # Extract chip
                try:
                    data, win_transform = extract_chip(src, row_off, col_off, CHIP_SIZE, PRITHVI_BANDS)
                except:
                    continue

                valid_frac = np.mean(data > 0)
                if valid_frac < 0.8:
                    continue

                mask = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)

                chip_name = f"chip_{chip_id:05d}"
                chip_path = CHIPS_DIR / f"{chip_name}.tif"
                label_path = LABELS_DIR / f"{chip_name}.tif"

                chip_profile = {
                    'driver': 'GTiff',
                    'dtype': 'float32',
                    'width': CHIP_SIZE,
                    'height': CHIP_SIZE,
                    'count': 6,
                    'crs': crs,
                    'transform': win_transform,
                }
                with rasterio.open(str(chip_path), 'w', **chip_profile) as dst:
                    dst.write(data)

                label_profile = chip_profile.copy()
                label_profile.update(count=1, dtype='uint8')
                with rasterio.open(str(label_path), 'w', **label_profile) as dst:
                    dst.write(mask[np.newaxis])

                chip_meta.append({
                    'chip_id': chip_name,
                    'tile': tile_name,
                    'deposit': None,
                    'label': 'negative',
                    'rotation': 0,
                    'center_x': float(chip_cx),
                    'center_y': float(chip_cy),
                    'valid_fraction': float(valid_frac),
                    'mask_pixels': 0,
                })
                chip_id += 1
                neg_count += 1

            print(f"  Negative chips: {neg_count}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("CHIP PREPARATION COMPLETE")
    print(f"{'=' * 70}")

    n_pos = sum(1 for m in chip_meta if m['label'] == 'positive')
    n_neg = sum(1 for m in chip_meta if m['label'] == 'negative')
    print(f"Total chips: {len(chip_meta)} ({n_pos} positive, {n_neg} negative)")

    # Per-tile breakdown
    from collections import Counter
    tile_counts = Counter()
    for m in chip_meta:
        tile_counts[f"{m['tile']}_{m['label']}"] += 1
    for key in sorted(tile_counts.keys()):
        print(f"  {key}: {tile_counts[key]}")

    # LOTO split info
    print(f"\nLOTO CV splits:")
    tiles = sorted(set(m['tile'] for m in chip_meta))
    for held_out in tiles:
        train_pos = sum(1 for m in chip_meta if m['tile'] != held_out and m['label'] == 'positive')
        train_neg = sum(1 for m in chip_meta if m['tile'] != held_out and m['label'] == 'negative')
        test_pos = sum(1 for m in chip_meta if m['tile'] == held_out and m['label'] == 'positive')
        test_neg = sum(1 for m in chip_meta if m['tile'] == held_out and m['label'] == 'negative')
        print(f"  Hold out {held_out}: train={train_pos}+/{train_neg}-, test={test_pos}+/{test_neg}-")

    # Save metadata
    meta_path = Path("data/processed/chip_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(chip_meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    # Disk usage
    chip_bytes = sum(f.stat().st_size for f in CHIPS_DIR.glob("*.tif"))
    label_bytes = sum(f.stat().st_size for f in LABELS_DIR.glob("*.tif"))
    total_mb = (chip_bytes + label_bytes) / (1024**2)
    print(f"Disk usage: {total_mb:.1f} MB ({chip_bytes/(1024**2):.1f} MB chips + {label_bytes/(1024**2):.1f} MB labels)")


if __name__ == "__main__":
    main()
