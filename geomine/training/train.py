"""Model training, evaluation, and persistence for mineral prospectivity.

Supports XGBoost and Random Forest classifiers with spatial block
cross-validation for unbiased performance estimation and SHAP-based
interpretability analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from geomine.training.spatial_cv import (
    create_spatial_blocks,
    evaluate_spatial_cv,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray | None,
    y_val: np.ndarray | None,
    config: dict[str, Any],
) -> XGBClassifier:
    """Train an XGBoost classifier with optional early stopping.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and labels.
    X_val, y_val : array-like or None
        Validation set for early stopping.  If ``None``, early stopping is
        disabled.
    config : dict
        Full project config (``config["model"]["xgboost"]`` is used).

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    xgb_cfg = config["model"]["xgboost"]

    model = XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 500),
        max_depth=xgb_cfg.get("max_depth", 8),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        eval_metric=xgb_cfg.get("eval_metric", "aucpr"),
        early_stopping_rounds=(
            xgb_cfg.get("early_stopping_rounds", 50) if X_val is not None else None
        ),
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    fit_kwargs: dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

    logger.info("Training XGBoost (%d estimators, max_depth=%d)", model.n_estimators, model.max_depth)
    model.fit(X_train, y_train, **fit_kwargs)

    best_iter = getattr(model, "best_iteration", model.n_estimators)
    logger.info("XGBoost training complete (best iteration: %d)", best_iter)
    return model


def train_random_forest(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Parameters
    ----------
    X_train, y_train : array-like
        Training features and labels.
    config : dict
        Full project config (``config["model"]["random_forest"]`` is used).

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    rf_cfg = config["model"]["random_forest"]

    model = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 500),
        max_depth=rf_cfg.get("max_depth", 20),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
        random_state=42,
        n_jobs=-1,
    )

    logger.info(
        "Training Random Forest (%d estimators, max_depth=%s)",
        model.n_estimators, model.max_depth,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete")
    return model


# ---------------------------------------------------------------------------
# Full training pipeline with spatial CV
# ---------------------------------------------------------------------------

def train_with_spatial_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any], list[str]]:
    """Full training pipeline: spatial CV evaluation then final model fit.

    Steps
    -----
    1. Build spatial blocks from sample coordinates in *metadata*.
    2. Run spatial block cross-validation for unbiased metrics.
    3. Train a final model on **all** data.
    4. Log summary metrics.

    Parameters
    ----------
    X : DataFrame
        Feature matrix.
    y : ndarray
        Label vector.
    metadata : dict
        Must contain ``"x"`` and ``"y"`` coordinate arrays and
        ``"feature_names"`` list.
    config : dict
        Full project config.

    Returns
    -------
    model : fitted estimator
        The final model trained on all data.
    cv_results : dict
        Spatial CV performance metrics.
    feature_names : list of str
        Ordered feature names.
    """
    feature_names = metadata["feature_names"]
    cv_cfg = config["training"]["spatial_cv"]
    block_size_km = cv_cfg.get("block_size_km", 25)
    n_folds = cv_cfg.get("n_folds", 5)

    # 1. Create spatial blocks
    coords_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(metadata["x"], metadata["y"]),
        crs=metadata.get("crs", config["project"]["crs"]),
    )
    block_ids = create_spatial_blocks(coords_gdf, block_size_km)

    # 2. Spatial CV
    primary_model_name = config["model"].get("primary", "xgboost")
    logger.info("Running spatial CV with %s model (%d folds)", primary_model_name, n_folds)

    if primary_model_name == "xgboost":
        cv_model = XGBClassifier(
            n_estimators=config["model"]["xgboost"].get("n_estimators", 500),
            max_depth=config["model"]["xgboost"].get("max_depth", 8),
            learning_rate=config["model"]["xgboost"].get("learning_rate", 0.05),
            subsample=config["model"]["xgboost"].get("subsample", 0.8),
            colsample_bytree=config["model"]["xgboost"].get("colsample_bytree", 0.8),
            eval_metric=config["model"]["xgboost"].get("eval_metric", "aucpr"),
            early_stopping_rounds=config["model"]["xgboost"].get("early_stopping_rounds", 50),
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )
    else:
        cv_model = RandomForestClassifier(
            n_estimators=config["model"]["random_forest"].get("n_estimators", 500),
            max_depth=config["model"]["random_forest"].get("max_depth", 20),
            min_samples_leaf=config["model"]["random_forest"].get("min_samples_leaf", 5),
            random_state=42,
            n_jobs=-1,
        )

    cv_results = evaluate_spatial_cv(cv_model, X, y, block_ids, n_folds)

    logger.info(
        "Spatial CV results: PR-AUC=%.3f, ROC-AUC=%.3f, F1@0.5=%.3f",
        cv_results["mean"].get("pr_auc", float("nan")),
        cv_results["mean"].get("roc_auc", float("nan")),
        cv_results["mean"].get("f1_at_0.5", float("nan")),
    )

    # 3. Train final model on all data
    logger.info("Training final %s model on all %d samples", primary_model_name, len(y))
    if primary_model_name == "xgboost":
        final_model = train_xgboost(X, y, None, None, config)
    else:
        final_model = train_random_forest(X, y, config)

    return final_model, cv_results, feature_names


# ---------------------------------------------------------------------------
# Baseline models (run BEFORE XGBoost to calibrate expectations)
# ---------------------------------------------------------------------------

def run_baselines(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run simple baseline models to calibrate XGBoost expectations.

    Three baselines:
    1. Single-feature threshold on each feature (best univariate predictor)
    2. Logistic regression on all features
    3. Random baseline (shuffled labels)

    If logistic regression already achieves PR-AUC ~0.55, XGBoost may only
    add marginal lift -- useful knowledge for interpreting results.

    Returns a dict with baseline results and the best single-feature name.
    """
    X_arr = np.asarray(X)
    results: dict[str, Any] = {}

    # --- Baseline 1: Best single-feature threshold ---
    best_feat_prauc = 0.0
    best_feat_name = ""
    for i, name in enumerate(feature_names):
        feat = X_arr[:, i]
        valid = np.isfinite(feat)
        if valid.sum() < 10:
            continue
        try:
            prauc = average_precision_score(y[valid], feat[valid])
            # Also try inverse (lower = more prospective)
            prauc_inv = average_precision_score(y[valid], -feat[valid])
            if prauc_inv > prauc:
                prauc = prauc_inv
                direction = "inverse"
            else:
                direction = "direct"
            if prauc > best_feat_prauc:
                best_feat_prauc = prauc
                best_feat_name = f"{name} ({direction})"
        except Exception:
            continue

    results["best_single_feature"] = best_feat_name
    results["best_single_feature_prauc"] = best_feat_prauc
    logger.info(
        "Baseline 1 -- Best single feature: %s (PR-AUC=%.3f)",
        best_feat_name, best_feat_prauc,
    )

    # --- Baseline 2: Logistic regression with spatial CV ---
    from geomine.training.spatial_cv import evaluate_spatial_cv
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_results = evaluate_spatial_cv(lr, X, y, block_ids, n_folds)
    results["logistic_regression"] = lr_results.get("mean", {})
    lr_prauc = lr_results.get("mean", {}).get("pr_auc", float("nan"))
    logger.info("Baseline 2 -- Logistic Regression: PR-AUC=%.3f", lr_prauc)

    # --- Baseline 3: Random (shuffled labels) ---
    rng = np.random.RandomState(42)
    y_shuffled = rng.permutation(y)
    random_prauc = average_precision_score(y, y_shuffled.astype(float))
    results["random_prauc"] = random_prauc
    results["class_prior"] = float(y.mean())
    logger.info(
        "Baseline 3 -- Random: PR-AUC=%.3f (class prior=%.3f)",
        random_prauc, y.mean(),
    )

    logger.info("=" * 50)
    logger.info("BASELINE SUMMARY:")
    logger.info("  Random:            PR-AUC=%.3f", random_prauc)
    logger.info("  Best single feat:  PR-AUC=%.3f (%s)", best_feat_prauc, best_feat_name)
    logger.info("  Logistic Regr:     PR-AUC=%.3f", lr_prauc)
    logger.info("  XGBoost target:    PR-AUC>0.60")
    logger.info("=" * 50)

    return results


def check_lithology_vs_prospectivity(
    shap_importance: list[tuple[str, float]],
    feature_names: list[str],
) -> str:
    """Check if the model is doing prospectivity mapping or lithology mapping.

    If spectral features that simply identify ultramafic rock dominate
    (ferrous_iron, clay_ratio, ndvi), the model may be mapping lithology
    rather than discriminating deposit-bearing horizons within the intrusion.

    Returns a diagnostic string: 'prospectivity', 'lithology', or 'ambiguous'.
    """
    # Features that primarily identify ultramafic lithology
    lithology_markers = {"ferrous_iron", "spectral_ferrous_iron", "clay_ratio",
                         "spectral_clay_ratio", "ndvi", "spectral_ndvi"}
    # Features that discriminate within ultramafics (structural controls, specific alteration)
    prospectivity_markers = {"lineament_density", "distance_to_lineaments",
                             "fault_intersection_density", "slope", "aspect",
                             "iron_oxide", "spectral_iron_oxide", "ferric_iron",
                             "spectral_ferric_iron", "clay_swir", "spectral_clay_swir",
                             "plan_curvature", "profile_curvature", "drainage_density"}

    top_5 = [name for name, _ in shap_importance[:5]]
    lith_count = sum(1 for f in top_5 if any(m in f for m in lithology_markers))
    prosp_count = sum(1 for f in top_5 if any(m in f for m in prospectivity_markers))

    if lith_count >= 4:
        logger.warning(
            "LITHOLOGY WARNING: Top SHAP features (%s) are primarily lithology "
            "discriminators. The model may be mapping ultramafic rock rather than "
            "predicting deposit-bearing horizons within the intrusion. Consider:\n"
            "  1. Restrict training to WITHIN the ultramafic unit only\n"
            "  2. Add structural features (lineaments, fault intersections)\n"
            "  3. Add proximity-to-contact features",
            top_5,
        )
        return "lithology"
    elif prosp_count >= 3:
        logger.info(
            "PROSPECTIVITY CONFIRMED: Top features (%s) include structural "
            "and specific alteration indicators, not just bulk lithology.",
            top_5,
        )
        return "prospectivity"
    else:
        logger.info(
            "AMBIGUOUS: Top features (%s) mix lithology and prospectivity "
            "indicators. Review SHAP plots manually.", top_5,
        )
        return "ambiguous"


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------

def compute_shap_analysis(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Compute SHAP values and generate interpretability plots.

    Uses ``TreeExplainer`` for tree-based models (XGBoost, Random Forest)
    for fast, exact Shapley value computation.

    Parameters
    ----------
    model : fitted estimator
        Must be a tree-based sklearn or XGBoost model.
    X : DataFrame or ndarray
        Feature matrix (a representative subset is fine for speed).
    feature_names : list of str
        Ordered feature names matching columns of *X*.
    output_dir : str or Path
        Directory to save SHAP plots.

    Returns
    -------
    shap_values : ndarray
        SHAP value matrix (n_samples x n_features).
    feature_importance : list of (name, mean_abs_shap)
        Sorted descending by importance.
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Computing SHAP values for %d samples, %d features", len(X), len(feature_names))

    X_df = pd.DataFrame(X, columns=feature_names) if not isinstance(X, pd.DataFrame) else X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    shap_values = np.asarray(shap_values)

    # Feature importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_pairs = sorted(
        zip(feature_names, mean_abs_shap.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info("Top 5 SHAP features:")
    for name, val in importance_pairs[:5]:
        logger.info("  %s: %.4f", name, val)

    # Check for exploration bias in top features
    bias_features = {"distance_to_road", "deposit_density_50km"}
    top_5_names = {name for name, _ in importance_pairs[:5]}
    bias_overlap = bias_features & top_5_names
    if bias_overlap:
        logger.warning(
            "EXPLORATION BIAS WARNING: %s in top-5 SHAP features. "
            "The model may be learning exploration patterns rather than "
            "geological prospectivity signals.",
            ", ".join(sorted(bias_overlap)),
        )

    # --- Beeswarm summary plot ---
    summary_path = output_dir / "shap_summary.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary plot saved: %s", summary_path)

    # --- Bar chart ---
    bar_path = output_dir / "shap_importance.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP importance bar chart saved: %s", bar_path)

    return shap_values, importance_pairs


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    metadata: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save a trained model and its metadata to disk.

    The model is serialised with :mod:`joblib`.  A companion JSON file
    with the same stem is written alongside containing feature names,
    training configuration, and CV metrics.

    Parameters
    ----------
    model : fitted estimator
        The trained model.
    metadata : dict
        Must include ``"feature_names"``, ``"cv_metrics"``, and optionally
        ``"config"`` and ``"shap_importance"``.
    output_path : str or Path
        Destination ``.joblib`` file path.

    Returns
    -------
    Path
        The model file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, output_path)
    logger.info("Model saved: %s", output_path)

    # Save metadata as JSON alongside
    meta_path = output_path.with_suffix(".json")

    # Make metadata JSON-serialisable
    serialisable = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            serialisable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serialisable[k] = float(v)
        else:
            try:
                json.dumps(v)
                serialisable[k] = v
            except (TypeError, ValueError):
                serialisable[k] = str(v)

    with open(meta_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Model metadata saved: %s", meta_path)

    return output_path
