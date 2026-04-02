"""GeoMine AI -- Command-line interface for the mineral prediction pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from geomine.utils.config import load_config, get_aoi_geometry

logger = logging.getLogger("geomine")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def main(verbose: bool) -> None:
    """GeoMine AI -- Satellite-powered mineral prediction engine."""
    setup_logging(verbose)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--skip-sentinel", is_flag=True, help="Skip Sentinel-2 download")
@click.option("--skip-srtm", is_flag=True, help="Skip SRTM download")
@click.option("--skip-mrds", is_flag=True, help="Skip MRDS download")
@click.option("--earthdata-token", envvar="EARTHDATA_TOKEN", default=None)
@click.option("--copernicus-token", envvar="COPERNICUS_TOKEN", default=None)
def download(
    config_path: str,
    skip_sentinel: bool,
    skip_srtm: bool,
    skip_mrds: bool,
    earthdata_token: str | None,
    copernicus_token: str | None,
) -> None:
    """Download satellite data and training labels for an AOI."""
    config = load_config(config_path)
    errors: list[str] = []

    if not skip_sentinel:
        try:
            from geomine.ingest.sentinel2 import search_scenes, download_scene, preprocess_sentinel2

            logger.info("Searching for Sentinel-2 scenes...")
            scenes = search_scenes(config)
            logger.info(f"Found {len(scenes)} scenes")

            raw_dir = Path(config["data"]["raw_dir"]) / "sentinel2"
            raw_dir.mkdir(parents=True, exist_ok=True)

            for i, scene in enumerate(scenes[:10]):  # limit to 10 best scenes
                logger.info(f"Downloading scene {i + 1}/{min(len(scenes), 10)}")
                download_scene(scene, config["data"]["sentinel2"]["bands"], raw_dir, token=copernicus_token)

            processed_dir = Path(config["data"]["processed_dir"]) / "sentinel2"
            processed_dir.mkdir(parents=True, exist_ok=True)
            preprocess_sentinel2(raw_dir, processed_dir, config)
        except Exception as e:
            logger.error(f"Sentinel-2 download failed: {e}")
            errors.append(f"Sentinel-2: {e}")

    if not skip_srtm:
        try:
            from geomine.ingest.srtm import download_srtm, mosaic_and_clip

            logger.info("Downloading SRTM DEM tiles...")
            tile_paths = download_srtm(config, token=earthdata_token)

            processed_dir = Path(config["data"]["processed_dir"])
            processed_dir.mkdir(parents=True, exist_ok=True)
            dem_path = processed_dir / "srtm_dem.tif"

            aoi_geom = get_aoi_geometry(config)
            mosaic_and_clip(tile_paths, dem_path, aoi_geom, config.get("project", {}).get("crs", "EPSG:32736"))
            logger.info(f"DEM saved to {dem_path}")
        except Exception as e:
            logger.error(f"SRTM download failed: {e}")
            errors.append(f"SRTM: {e}")

    if not skip_mrds:
        try:
            from geomine.ingest.mrds import download_mrds

            logger.info("Downloading MRDS deposit records...")
            deposits_path = download_mrds(config)
            logger.info(f"Deposits saved to {deposits_path}")
        except Exception as e:
            logger.error(f"MRDS download failed: {e}")
            errors.append(f"MRDS: {e}")

    if errors:
        logger.warning(f"Completed with {len(errors)} error(s): {errors}")
    else:
        logger.info("All downloads completed successfully.")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def compute_features(config_path: str) -> None:
    """Compute spectral indices and structural features from downloaded data."""
    config = load_config(config_path)

    from geomine.spectral.compute import compute_sentinel2_indices, stack_all_features
    from geomine.structural.terrain import compute_slope, compute_aspect, compute_curvature
    from geomine.structural.lineaments import extract_lineaments, compute_lineament_density
    from geomine.structural.proximity import compute_distance_to_features, compute_drainage_density

    processed_dir = Path(config["data"]["processed_dir"])
    output_dir = Path(config["data"]["processed_dir"]) / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Spectral indices ---
    logger.info("Computing spectral indices...")
    sentinel_dir = processed_dir / "sentinel2"
    stacked_rasters = list(sentinel_dir.glob("*_stacked.tif"))
    if not stacked_rasters:
        logger.error("No stacked Sentinel-2 rasters found. Run 'download' first.")
        return

    spectral_output = output_dir / "spectral"
    spectral_output.mkdir(exist_ok=True)
    index_paths = compute_sentinel2_indices(str(stacked_rasters[0]), str(spectral_output), config)
    logger.info(f"Computed {len(index_paths)} spectral indices")

    # --- Terrain features ---
    logger.info("Computing terrain features...")
    dem_path = processed_dir / "srtm_dem.tif"
    if not dem_path.exists():
        logger.error("No DEM found. Run 'download' first.")
        return

    terrain_output = output_dir / "terrain"
    terrain_output.mkdir(exist_ok=True)

    slope_path = str(terrain_output / "slope.tif")
    aspect_path = str(terrain_output / "aspect.tif")
    plan_curv_path = str(terrain_output / "plan_curvature.tif")
    prof_curv_path = str(terrain_output / "profile_curvature.tif")

    compute_slope(str(dem_path), slope_path)
    compute_aspect(str(dem_path), aspect_path)
    compute_curvature(str(dem_path), plan_curv_path, prof_curv_path)

    # --- Lineament extraction ---
    logger.info("Extracting lineaments...")
    structural_output = output_dir / "structural"
    structural_output.mkdir(exist_ok=True)

    lineaments_gdf = extract_lineaments(str(dem_path), config)
    lineaments_path = structural_output / "lineaments.geojson"
    lineaments_gdf.to_file(str(lineaments_path), driver="GeoJSON")
    logger.info(f"Extracted {len(lineaments_gdf)} lineaments")

    lineament_density_path = str(structural_output / "lineament_density.tif")
    compute_lineament_density(lineaments_gdf, str(dem_path), output_path=lineament_density_path)

    # --- Distance features ---
    logger.info("Computing distance features...")
    training_dir = Path(config["data"]["training_dir"])
    deposits_file = training_dir / "mrds_deposits.geojson"
    if not deposits_file.exists():
        deposits_file = list(training_dir.glob("*deposits*.geojson"))
        if deposits_file:
            deposits_file = deposits_file[0]
        else:
            logger.warning("No deposit file found, skipping distance-to-deposit feature")
            deposits_file = None

    if deposits_file:
        import geopandas as gpd
        deposits_gdf = gpd.read_file(str(deposits_file))
        dist_deposit_path = str(structural_output / "distance_to_deposits.tif")
        compute_distance_to_features(deposits_gdf, str(dem_path), dist_deposit_path)

    dist_lineament_path = str(structural_output / "distance_to_lineaments.tif")
    compute_distance_to_features(lineaments_gdf, str(dem_path), dist_lineament_path)

    drainage_path = str(structural_output / "drainage_density.tif")
    compute_drainage_density(str(dem_path), str(dem_path), drainage_path)

    # --- Stack all features ---
    logger.info("Stacking all features into single raster...")
    all_feature_paths = {}

    # Spectral
    for name, path in index_paths.items():
        all_feature_paths[f"spectral_{name}"] = path

    # Terrain
    all_feature_paths["slope"] = slope_path
    all_feature_paths["aspect"] = aspect_path
    all_feature_paths["plan_curvature"] = plan_curv_path
    all_feature_paths["profile_curvature"] = prof_curv_path

    # Structural
    all_feature_paths["lineament_density"] = lineament_density_path
    all_feature_paths["distance_to_lineaments"] = dist_lineament_path
    all_feature_paths["drainage_density"] = drainage_path
    if deposits_file:
        all_feature_paths["distance_to_deposits"] = dist_deposit_path

    feature_stack_path = str(output_dir / "feature_stack.tif")
    stack_all_features(all_feature_paths, feature_stack_path)
    logger.info(f"Feature stack saved: {feature_stack_path} ({len(all_feature_paths)} bands)")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    """Train mineral prospectivity model with spatial cross-validation."""
    config = load_config(config_path)

    import geopandas as gpd
    from geomine.training.sampling import (
        generate_negative_samples,
        prepare_training_data,
        compute_exploration_intensity,
    )
    from geomine.training.train import (
        train_with_spatial_cv, compute_shap_analysis, save_model,
        run_baselines, check_lithology_vs_prospectivity,
    )
    from geomine.training.spatial_cv import create_spatial_blocks, create_along_strike_blocks

    processed_dir = Path(config["data"]["processed_dir"])
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load deposits
    training_dir = Path(config["data"]["training_dir"])
    deposits_files = list(training_dir.glob("*deposits*.geojson"))
    if not deposits_files:
        logger.error("No deposit GeoJSON found in training dir. Run 'download' first.")
        return
    deposits_gdf = gpd.read_file(str(deposits_files[0]))
    logger.info(f"Loaded {len(deposits_gdf)} deposit records")

    # Load AOI
    aoi_geom = get_aoi_geometry(config)

    # Compute exploration intensity
    logger.info("Computing exploration intensity proxy...")
    feature_dir = processed_dir / "features"
    dem_path = str(processed_dir / "srtm_dem.tif")
    exploration_path = str(feature_dir / "exploration_intensity.tif")
    compute_exploration_intensity(aoi_geom, None, deposits_gdf, dem_path, exploration_path)

    # Generate negative samples
    logger.info("Generating negative samples (exploration-bias aware)...")
    negatives_gdf = generate_negative_samples(deposits_gdf, aoi_geom, config, exploration_path)
    logger.info(f"Generated {len(negatives_gdf)} negative samples")

    # Collect feature rasters
    feature_stack = str(feature_dir / "feature_stack.tif")
    if not Path(feature_stack).exists():
        logger.error("Feature stack not found. Run 'compute-features' first.")
        return

    import rasterio
    with rasterio.open(feature_stack) as src:
        feature_names = [src.descriptions[i] or f"band_{i+1}" for i in range(src.count)]
        feature_rasters = {name: feature_stack for name in feature_names}

    # Prepare training data
    logger.info("Preparing training data...")
    X, y, metadata = prepare_training_data(
        deposits_gdf, negatives_gdf, feature_rasters, feature_names, config
    )
    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features, "
                f"{y.sum()} positive, {(~y.astype(bool)).sum()} negative")

    # --- Run baselines FIRST (calibrate expectations) ---
    logger.info("Running baseline models before XGBoost...")
    coords_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(metadata["x"], metadata["y"]),
        crs=config["project"]["crs"],
    )
    block_ids = create_spatial_blocks(coords_gdf, config["training"]["spatial_cv"]["block_size_km"])

    # Also create along-strike blocks for the Great Dyke (linear intrusion)
    strike_block_ids = create_along_strike_blocks(coords_gdf, n_segments=5, strike_azimuth=15.0)

    baseline_results = run_baselines(X, y, block_ids, metadata["feature_names"])

    # --- Train with spatial CV ---
    logger.info("Training XGBoost with spatial block cross-validation...")
    model, cv_results, trained_feature_names = train_with_spatial_cv(X, y, metadata, config)

    logger.info(f"Spatial CV results:")
    mean_metrics = cv_results.get("mean", cv_results)
    pr_auc_key = "pr_auc" if "pr_auc" in mean_metrics else "mean_pr_auc"
    roc_auc_key = "roc_auc" if "roc_auc" in mean_metrics else "mean_roc_auc"
    logger.info(f"  PR-AUC:  {mean_metrics.get(pr_auc_key, 0):.3f}")
    logger.info(f"  ROC-AUC: {mean_metrics.get(roc_auc_key, 0):.3f}")

    # --- SHAP analysis ---
    logger.info("Computing SHAP analysis...")
    shap_values, shap_importance = compute_shap_analysis(model, X, trained_feature_names, str(output_dir))

    # --- Lithology vs prospectivity check ---
    diagnosis = check_lithology_vs_prospectivity(shap_importance, trained_feature_names)

    # --- Save model ---
    model_path = str(output_dir / "model_xgboost.joblib")
    save_model(model, {
        "features": trained_feature_names,
        "cv_results": cv_results,
        "baseline_results": baseline_results,
        "lithology_diagnosis": diagnosis,
        "config_path": config_path,
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    # --- Gate check ---
    pr_auc = mean_metrics.get(pr_auc_key, 0)
    if pr_auc < 0.45:
        logger.warning("=" * 60)
        logger.warning(f"GATE FAILURE: PR-AUC = {pr_auc:.3f} < 0.45")
        logger.warning("Signal may be too weak. Investigate before proceeding.")
        logger.warning("Possible causes: noisy labels, insufficient spectral resolution,")
        logger.warning("exploration bias, or deposit type not detectable at satellite scale.")
        logger.warning("=" * 60)
    elif pr_auc < 0.60:
        logger.warning(f"PR-AUC = {pr_auc:.3f} -- below target (0.60). Model may need improvement.")
    else:
        logger.info(f"PR-AUC = {pr_auc:.3f} -- meets Phase 1 target (>= 0.60)")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--model-path", default=None, help="Path to trained model (default: auto-detect)")
def predict(config_path: str, model_path: str | None) -> None:
    """Run prediction on the AOI and generate target maps."""
    config = load_config(config_path)

    import joblib
    from geomine.predict.inference import predict_raster, cluster_targets, generate_report

    output_dir = Path(config["data"]["output_dir"])
    processed_dir = Path(config["data"]["processed_dir"])

    # Load model
    if model_path is None:
        model_path = str(output_dir / "model_xgboost.joblib")
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Run 'train' first.")
        return

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Feature stack
    feature_stack = str(processed_dir / "features" / "feature_stack.tif")
    if not Path(feature_stack).exists():
        logger.error("Feature stack not found. Run 'compute-features' first.")
        return

    # Predict
    prob_path = config["output"]["probability_raster"]
    uncert_path = config["output"]["uncertainty_raster"]
    Path(prob_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running prediction on full AOI...")
    predict_raster(model, feature_stack, prob_path, uncert_path, config)
    logger.info(f"Probability raster: {prob_path}")
    logger.info(f"Uncertainty raster: {uncert_path}")

    # Cluster targets
    logger.info("Clustering high-probability targets...")
    targets_path = config["output"]["targets_geojson"]
    targets_gdf = cluster_targets(prob_path, uncert_path, config)
    logger.info(f"Found {len(targets_gdf)} target zones above threshold")

    if len(targets_gdf) > 0:
        logger.info("Top 5 targets:")
        for _, row in targets_gdf.head(5).iterrows():
            logger.info(f"  Rank {row.get('rank', '?')}: "
                        f"prob={row.get('mean_probability', 0):.1f}%, "
                        f"uncert={row.get('mean_uncertainty', 0):.1f}%, "
                        f"area={row.get('area_km2', 0):.2f} km2")

    # Generate report
    report_path = config["output"]["report"]
    logger.info(f"Generating report: {report_path}")

    # Load CV metrics from model metadata
    meta_path = Path(model_path).with_suffix(".json")
    cv_metrics = {}
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
            cv_metrics = meta.get("cv_results", {})

    generate_report(targets_gdf, {}, cv_metrics, config, report_path)
    logger.info("Done. Open the following in QGIS:")
    logger.info(f"  - Probability: {prob_path}")
    logger.info(f"  - Uncertainty: {uncert_path}")
    logger.info(f"  - Targets: {targets_path}")
    logger.info(f"  - Report: {report_path}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--preset", default=None, help="Preset name (e.g. geology_12_8_2, swir, iron_detection)")
@click.option("--all-presets", is_flag=True, help="Render all geological presets")
@click.option("--bands", default=None, help="Custom band indices as R,G,B (e.g. 7,4,1)")
@click.option("--gamma", default=1.2, help="Gamma correction (default 1.2)")
def layers(config_path: str, preset: str | None, all_presets: bool, bands: str | None, gamma: float) -> None:
    """Render RGB band composite layers for geological interpretation."""
    config = load_config(config_path)

    from geomine.spectral.visualize import render_composite, render_all_presets, LAYER_PRESETS

    processed_dir = Path(config["data"]["processed_dir"])
    output_dir = Path(config["data"]["output_dir"]) / "layers"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find stacked raster
    sentinel_dir = processed_dir / "sentinel2"
    stacked = list(sentinel_dir.glob("*_stacked.tif"))
    if not stacked:
        logger.error("No stacked Sentinel-2 raster found. Run 'download' first.")
        return
    raster_path = str(stacked[0])

    if all_presets:
        results = render_all_presets(raster_path, str(output_dir), gamma=gamma)
        logger.info(f"Rendered {len(results)} layer presets to {output_dir}")
        for name, path in results.items():
            info = LAYER_PRESETS[name]
            logger.info(f"  {info['name']:30s} -> {path}")
        return

    if bands:
        custom = tuple(int(b.strip()) for b in bands.split(","))
        if len(custom) != 3:
            logger.error("--bands must be 3 comma-separated indices (e.g. 7,4,1)")
            return
        out = output_dir / f"custom_{'_'.join(str(b) for b in custom)}.png"
        render_composite(raster_path, preset=custom, output_png=str(out), gamma=gamma)
        logger.info(f"Custom composite saved: {out}")
        return

    if preset:
        if preset not in LAYER_PRESETS:
            logger.error(f"Unknown preset '{preset}'. Available: {', '.join(LAYER_PRESETS.keys())}")
            return
        out = output_dir / f"{preset}.png"
        render_composite(raster_path, preset=preset, output_png=str(out), gamma=gamma)
        logger.info(f"Layer saved: {out}")
        return

    # No args: list available presets
    logger.info("Available layer presets:")
    logger.info("")
    for name, info in LAYER_PRESETS.items():
        logger.info(f"  {name:25s}  {info['name']}")
        logger.info(f"  {'':25s}  Bands: {info['bands']}  |  {info['use']}")
        logger.info("")
    logger.info("Usage: geomine layers CONFIG --preset geology_12_8_2")
    logger.info("       geomine layers CONFIG --all-presets")
    logger.info("       geomine layers CONFIG --bands 7,4,1")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def run_all(config_path: str) -> None:
    """Run the full Phase 1 pipeline: download -> features -> train -> predict."""
    ctx = click.get_current_context()

    logger.info("=" * 60)
    logger.info("GeoMine AI -- Full Phase 1 Pipeline")
    logger.info("=" * 60)

    logger.info("\n[1/4] Downloading data...")
    ctx.invoke(download, config_path=config_path)

    logger.info("\n[2/4] Computing features...")
    ctx.invoke(compute_features, config_path=config_path)

    logger.info("\n[3/4] Training model...")
    ctx.invoke(train, config_path=config_path)

    logger.info("\n[4/4] Running prediction...")
    ctx.invoke(predict, config_path=config_path)

    logger.info("\n" + "=" * 60)
    logger.info("Phase 1 pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
