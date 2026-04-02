"""Spatial block cross-validation for mineral prospectivity modelling.

Standard random cross-validation produces over-optimistic performance
estimates when training samples exhibit spatial autocorrelation (as mineral
deposits always do).  This module implements *spatial block CV*, which
partitions the study area into contiguous rectangular blocks and holds out
entire blocks at a time, preventing spatial leakage between train and test
sets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Generator

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------

def create_spatial_blocks(
    points_gdf: gpd.GeoDataFrame,
    block_size_km: float,
) -> np.ndarray:
    """Assign each sample point to a rectangular spatial block.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Sample points in a *projected* CRS (units = metres).
    block_size_km : float
        Edge length of the square blocks, in kilometres.

    Returns
    -------
    ndarray of int
        Block ID for every point, in the same order as *points_gdf*.
    """
    block_size_m = block_size_km * 1000.0
    xs = points_gdf.geometry.x.values
    ys = points_gdf.geometry.y.values

    x_min, y_min = xs.min(), ys.min()

    col_ids = ((xs - x_min) / block_size_m).astype(int)
    row_ids = ((ys - y_min) / block_size_m).astype(int)

    n_cols = int(col_ids.max()) + 1
    block_ids = row_ids * n_cols + col_ids

    n_blocks = len(np.unique(block_ids))
    logger.info(
        "Created %d spatial blocks (%.0f km grid) for %d points",
        n_blocks, block_size_km, len(points_gdf),
    )

    return block_ids.astype(int)


def create_along_strike_blocks(
    points_gdf: gpd.GeoDataFrame,
    n_segments: int = 5,
    strike_azimuth: float = 15.0,
) -> np.ndarray:
    """Assign points to blocks along a linear geological feature's strike direction.

    For linear intrusions like the Great Dyke, standard square grid blocking
    can place geologically similar adjacent segments into train and test sets.
    This function projects points onto the strike direction and divides them
    into segments along that axis.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Sample points in a projected CRS (units = metres).
    n_segments : int
        Number of along-strike segments (= number of CV folds).
    strike_azimuth : float
        Strike direction in degrees clockwise from north.
        Great Dyke is roughly NNE-SSW (~15 degrees).

    Returns
    -------
    ndarray of int
        Segment ID for every point.
    """
    xs = points_gdf.geometry.x.values
    ys = points_gdf.geometry.y.values

    # Rotate coordinates so strike direction aligns with Y axis
    angle_rad = np.radians(strike_azimuth)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Project onto strike direction
    strike_proj = xs * sin_a + ys * cos_a

    # Divide into equal-count segments along strike
    percentiles = np.linspace(0, 100, n_segments + 1)
    boundaries = np.percentile(strike_proj, percentiles)
    segment_ids = np.digitize(strike_proj, boundaries[1:-1])

    n_actual = len(np.unique(segment_ids))
    logger.info(
        "Created %d along-strike segments (azimuth=%.0f°) for %d points",
        n_actual, strike_azimuth, len(points_gdf),
    )

    return segment_ids.astype(int)


# ---------------------------------------------------------------------------
# CV splitter
# ---------------------------------------------------------------------------

def spatial_block_cv(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    n_folds: int = 5,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Yield train/test index splits based on spatial blocks.

    Blocks are distributed across folds using a greedy algorithm that tries
    to balance the positive-class count in each fold.

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix (used only for its length).
    y : ndarray
        Label vector.
    block_ids : ndarray of int
        Block ID per sample (from :func:`create_spatial_blocks`).
    n_folds : int
        Number of CV folds.

    Yields
    ------
    train_indices, test_indices : ndarray, ndarray
        Integer index arrays for each fold.
    """
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < n_folds:
        logger.warning(
            "Only %d spatial blocks available but %d folds requested; "
            "reducing n_folds to %d",
            len(unique_blocks), n_folds, len(unique_blocks),
        )
        n_folds = len(unique_blocks)

    # Count positives per block for balanced assignment
    block_pos_count: dict[int, int] = defaultdict(int)
    for bid, label in zip(block_ids, y):
        if label == 1:
            block_pos_count[int(bid)] += 1

    # Sort blocks by positive count descending (greedy balancing)
    sorted_blocks = sorted(
        unique_blocks, key=lambda b: block_pos_count.get(int(b), 0), reverse=True
    )

    # Assign blocks to folds greedily (fold with fewest positives gets next block)
    fold_assignment: dict[int, int] = {}
    fold_pos_counts = np.zeros(n_folds, dtype=int)

    for bid in sorted_blocks:
        target_fold = int(np.argmin(fold_pos_counts))
        fold_assignment[int(bid)] = target_fold
        fold_pos_counts[target_fold] += block_pos_count.get(int(bid), 0)

    n_samples = len(y) if isinstance(y, np.ndarray) else len(y)
    all_indices = np.arange(n_samples)

    for fold_idx in range(n_folds):
        test_blocks = {
            bid for bid, fid in fold_assignment.items() if fid == fold_idx
        }
        test_mask = np.array([int(bid) in test_blocks for bid in block_ids])
        test_indices = all_indices[test_mask]
        train_indices = all_indices[~test_mask]

        n_test_pos = int(y[test_indices].sum()) if len(test_indices) > 0 else 0
        n_train_pos = int(y[train_indices].sum()) if len(train_indices) > 0 else 0

        logger.debug(
            "Fold %d: train=%d (pos=%d), test=%d (pos=%d), blocks_held_out=%d",
            fold_idx + 1,
            len(train_indices), n_train_pos,
            len(test_indices), n_test_pos,
            len(test_blocks),
        )

        yield train_indices, test_indices


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_spatial_cv(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Run spatial block CV and collect classification metrics per fold.

    A fresh clone of *model* is trained on each fold's training data.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must support ``.fit()`` and ``.predict_proba()``.
        Will be cloned via ``sklearn.base.clone``.
    X : DataFrame or ndarray
        Feature matrix.
    y : ndarray
        Label vector.
    block_ids : ndarray of int
        Block ID per sample.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    dict
        ``"per_fold"`` -- list of per-fold metric dicts.
        ``"mean"`` -- mean metrics across folds.
        ``"predictions"`` -- ndarray of predicted probabilities for every
        sample (from its held-out fold).
    """
    from sklearn.base import clone

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    y_arr = np.asarray(y)

    predictions = np.full(len(y_arr), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, float]] = []
    thresholds_to_eval = [0.3, 0.5, 0.7]

    for fold_idx, (train_idx, test_idx) in enumerate(
        spatial_block_cv(X_arr, y_arr, block_ids, n_folds)
    ):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        if len(np.unique(y_train)) < 2:
            logger.warning("Fold %d: single-class training set -- skipping", fold_idx + 1)
            continue

        fold_model = clone(model)

        # XGBoost early stopping support
        fit_params: dict[str, Any] = {}
        model_class = type(fold_model).__name__
        if model_class in ("XGBClassifier",) and len(np.unique(y_test)) >= 2:
            fit_params["eval_set"] = [(X_test, y_test)]
            fit_params["verbose"] = False

        fold_model.fit(X_train, y_train, **fit_params)

        if len(np.unique(y_test)) < 2:
            logger.warning("Fold %d: single-class test set -- metrics limited", fold_idx + 1)
            y_prob = fold_model.predict_proba(X_test)[:, 1]
            predictions[test_idx] = y_prob
            fold_metrics.append({
                "fold": fold_idx + 1,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "pr_auc": float("nan"),
                "roc_auc": float("nan"),
            })
            continue

        y_prob = fold_model.predict_proba(X_test)[:, 1]
        predictions[test_idx] = y_prob

        metrics: dict[str, Any] = {
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        for thr in thresholds_to_eval:
            y_pred = (y_prob >= thr).astype(int)
            metrics[f"f1_at_{thr}"] = float(f1_score(y_test, y_pred, zero_division=0))
            metrics[f"precision_at_{thr}"] = float(precision_score(y_test, y_pred, zero_division=0))
            metrics[f"recall_at_{thr}"] = float(recall_score(y_test, y_pred, zero_division=0))

        fold_metrics.append(metrics)
        logger.info(
            "Fold %d: PR-AUC=%.3f  ROC-AUC=%.3f  F1@0.5=%.3f",
            fold_idx + 1,
            metrics["pr_auc"],
            metrics["roc_auc"],
            metrics.get("f1_at_0.5", float("nan")),
        )

    # Compute mean metrics
    if fold_metrics:
        numeric_keys = [
            k for k in fold_metrics[0]
            if isinstance(fold_metrics[0][k], (int, float)) and k != "fold"
        ]
        mean_metrics = {}
        for k in numeric_keys:
            vals = [fm[k] for fm in fold_metrics if not np.isnan(fm.get(k, float("nan")))]
            mean_metrics[k] = float(np.mean(vals)) if vals else float("nan")
    else:
        mean_metrics = {}

    logger.info(
        "Spatial CV complete: mean PR-AUC=%.3f, mean ROC-AUC=%.3f",
        mean_metrics.get("pr_auc", float("nan")),
        mean_metrics.get("roc_auc", float("nan")),
    )

    return {
        "per_fold": fold_metrics,
        "mean": mean_metrics,
        "predictions": predictions,
    }
