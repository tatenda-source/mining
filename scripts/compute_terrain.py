"""Compute all terrain + structural features from DEM."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from skimage.transform import probabilistic_hough_line
from shapely.geometry import LineString

dem_path = "data/processed/dem_32736.tif"
out_dir = Path("data/processed/features/terrain")
out_dir.mkdir(parents=True, exist_ok=True)
struct_dir = Path("data/processed/features/structural")
struct_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TERRAIN + STRUCTURAL FEATURE COMPUTATION")
print("=" * 60)

with rasterio.open(dem_path) as src:
    dem = src.read(1).astype(np.float32)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    res_x = abs(transform[0])
    res_y = abs(transform[4])

profile.update(count=1, dtype="float32")
print(f"DEM: {dem.shape[1]}x{dem.shape[0]}, res={res_x:.1f}m, CRS={crs}")

nodata_mask = (dem == 0) | ~np.isfinite(dem)
dem[nodata_mask] = np.nan

# ---- SLOPE ----
print("\n1. Computing slope...")
dy, dx = np.gradient(dem, res_y, res_x)
slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
slope_deg = np.degrees(slope_rad).astype(np.float32)
with rasterio.open(str(out_dir / "slope.tif"), "w", **profile) as dst:
    dst.write(slope_deg, 1)
    dst.set_band_description(1, "slope_degrees")
print(f"  median={np.nanmedian(slope_deg):.2f} deg")

# ---- ASPECT ----
print("2. Computing aspect...")
aspect = np.degrees(np.arctan2(-dy, dx)).astype(np.float32)
aspect = np.mod(aspect, 360)
with rasterio.open(str(out_dir / "aspect.tif"), "w", **profile) as dst:
    dst.write(aspect, 1)
    dst.set_band_description(1, "aspect_degrees")
print("  done")

# ---- CURVATURE ----
print("3. Computing curvature...")
dxx = np.gradient(dx, res_x, axis=1)
dyy = np.gradient(dy, res_y, axis=0)
dxy = np.gradient(dx, res_y, axis=0)

denom = np.power(dx**2 + dy**2 + 1e-10, 1.5)
plan_curv = np.clip(-((dxx * dy**2 - 2 * dxy * dx * dy + dyy * dx**2) / denom), -1, 1).astype(np.float32)
prof_curv = np.clip(-((dxx * dx**2 + 2 * dxy * dx * dy + dyy * dy**2) / denom), -1, 1).astype(np.float32)

for name, data in [("plan_curvature", plan_curv), ("profile_curvature", prof_curv)]:
    with rasterio.open(str(out_dir / f"{name}.tif"), "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, name)
print("  done")

# ---- HILLSHADE ----
print("4. Computing hillshade (8 azimuths)...")
altitude = np.radians(45)
for az_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
    azimuth = np.radians(az_deg)
    hs = np.cos(altitude) * np.cos(slope_rad) + np.sin(altitude) * np.sin(slope_rad) * np.cos(azimuth - np.arctan2(-dy, dx))
    hs = np.clip(hs * 255, 0, 255).astype(np.float32)
    with rasterio.open(str(out_dir / f"hillshade_{az_deg}.tif"), "w", **profile) as dst:
        dst.write(hs, 1)
print("  done")

# ---- LINEAMENT EXTRACTION ----
print("5. Extracting lineaments...")
hs_norm = np.nan_to_num((hs - np.nanmin(hs)) / (np.nanmax(hs) - np.nanmin(hs) + 1e-10), nan=0)
hs_uint8 = (hs_norm * 255).astype(np.uint8)

smoothed = ndimage.gaussian_filter(hs_uint8.astype(float), sigma=2.0)
sx = ndimage.sobel(smoothed, axis=1)
sy = ndimage.sobel(smoothed, axis=0)
edge_mag = np.hypot(sx, sy)
threshold = np.percentile(edge_mag[edge_mag > 0], 85)
edges = (edge_mag > threshold).astype(np.uint8)

scale = 4
edges_small = edges[::scale, ::scale]
print(f"  Edge detection done ({edges_small.shape}), running Hough...")

lines = probabilistic_hough_line(edges_small, threshold=30, line_length=50, line_gap=10)
print(f"  Raw lines: {len(lines)}")

lineament_geoms = []
lineament_data = []
for (x0, y0), (x1, y1) in lines:
    px0, py0 = x0 * scale, y0 * scale
    px1, py1 = x1 * scale, y1 * scale
    geo_x0 = transform[2] + px0 * transform[0]
    geo_y0 = transform[5] + py0 * transform[4]
    geo_x1 = transform[2] + px1 * transform[0]
    geo_y1 = transform[5] + py1 * transform[4]
    line = LineString([(geo_x0, geo_y0), (geo_x1, geo_y1)])
    length_m = line.length
    if length_m > 500:
        dx_l = geo_x1 - geo_x0
        dy_l = geo_y1 - geo_y0
        azimuth = np.degrees(np.arctan2(dx_l, dy_l)) % 180
        lineament_geoms.append(line)
        lineament_data.append({"azimuth": azimuth, "length_m": length_m})

lineaments_gdf = gpd.GeoDataFrame(lineament_data, geometry=lineament_geoms, crs=crs)
lineaments_gdf.to_file(str(struct_dir / "lineaments.geojson"), driver="GeoJSON")
print(f"  {len(lineaments_gdf)} lineaments (>500m)")

# ---- LINEAMENT DENSITY ----
print("6. Computing lineament density...")
lin_raster = np.zeros_like(dem)
for _, row in lineaments_gdf.iterrows():
    mid = row.geometry.centroid
    col = int((mid.x - transform[2]) / transform[0])
    r = int((mid.y - transform[5]) / transform[4])
    if 0 <= r < dem.shape[0] and 0 <= col < dem.shape[1]:
        lin_raster[r, col] += 1

kernel_px = max(1, int(5000 / res_x))
lin_density = ndimage.uniform_filter(lin_raster, size=kernel_px).astype(np.float32)
with rasterio.open(str(struct_dir / "lineament_density.tif"), "w", **profile) as dst:
    dst.write(lin_density, 1)
    dst.set_band_description(1, "lineament_density")
print("  done")

# ---- DISTANCE TO DEPOSITS ----
print("7. Distance to known deposits...")
deposits = gpd.read_file("data/training/deposits.geojson").to_crs(crs)
dep_raster = np.ones(dem.shape, dtype=int)
for _, row in deposits.iterrows():
    col = int((row.geometry.x - transform[2]) / transform[0])
    r = int((row.geometry.y - transform[5]) / transform[4])
    if 0 <= r < dem.shape[0] and 0 <= col < dem.shape[1]:
        dep_raster[r, col] = 0

dist_dep = (ndimage.distance_transform_edt(dep_raster) * res_x).astype(np.float32)
with rasterio.open(str(struct_dir / "distance_to_deposits.tif"), "w", **profile) as dst:
    dst.write(dist_dep, 1)
    dst.set_band_description(1, "distance_to_deposits_m")
print("  done")

# ---- DISTANCE TO LINEAMENTS ----
print("8. Distance to lineaments...")
lin_inv = (1 - (lin_raster > 0).astype(int))
dist_lin = (ndimage.distance_transform_edt(lin_inv) * res_x).astype(np.float32)
with rasterio.open(str(struct_dir / "distance_to_lineaments.tif"), "w", **profile) as dst:
    dst.write(dist_lin, 1)
    dst.set_band_description(1, "distance_to_lineaments_m")
print("  done")

# ---- DRAINAGE DENSITY ----
print("9. Drainage density (gradient proxy)...")
flow = np.sqrt(dx**2 + dy**2)
flow_smooth = ndimage.uniform_filter(np.nan_to_num(flow, nan=0), size=kernel_px).astype(np.float32)
with rasterio.open(str(struct_dir / "drainage_density.tif"), "w", **profile) as dst:
    dst.write(flow_smooth, 1)
    dst.set_band_description(1, "drainage_density")
print("  done")

print("\n" + "=" * 60)
print("ALL TERRAIN + STRUCTURAL FEATURES COMPLETE")
print("=" * 60)
all_files = sorted(list(out_dir.glob("*.tif")) + list(struct_dir.glob("*.tif")) + list(struct_dir.glob("*.geojson")))
total_mb = sum(f.stat().st_size for f in all_files) / (1024**2)
print(f"{len(all_files)} files, {total_mb:.0f} MB total")
for f in all_files:
    print(f"  {f.relative_to('data/processed')}: {f.stat().st_size/(1024**2):.0f} MB")
