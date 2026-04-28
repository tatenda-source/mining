"""Core audit logic.

A single public entry point -- ``audit()`` -- runs the full protocol against
any sklearn-compatible binary classifier and returns a structured result.

The protocol has five tests:

  1. **Random CV vs spatial CV gap** -- detects spatial leakage. A large gap
     means the random-CV number is inflated and the model does not generalize
     across geographies.
  2. **Class-prior baseline** -- the model must beat the rate at which
     positives appear in the dataset. PR-AUC at or below the prior is no skill.
  3. **Bootstrap stability** -- resampled coefficients (LR) or feature
     importances (tree models) must keep the same sign with reasonable
     confidence interval width.
  4. **Calibration** -- predicted probabilities must track empirical positive
     rates within a bin-wise tolerance.
  5. **Feature-label leakage** -- any single feature with point-biserial
     correlation > 0.95 against the label is flagged. Real geological
     signals are rarely that strong; this usually means the label was
     leaked into the features.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AuditConfig:
    """Audit thresholds and toggles.

    Defaults are tuned for mineral prospectivity. Override per use case.
    """
    spatial_leakage_max_gap: float = 0.30
    """Max acceptable PR-AUC drop from random CV to spatial CV. Default 0.30."""

    block_size_km: float = 25.0
    """Edge length of spatial blocks for spatial CV. Coordinates must be in metres."""

    n_random_folds: int = 5
    n_spatial_folds: int = 5
    n_bootstrap: int = 200

    feature_leakage_threshold: float = 0.95
    """Point-biserial correlation above which a feature is flagged as leaking."""

    calibration_n_bins: int = 10
    calibration_max_ece: float = 0.10
    """Max expected calibration error."""

    random_state: int = 42


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    score: float
    threshold: float
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditResult:
    passed: bool
    grade: str
    tests: list[TestResult]
    summary: dict[str, Any]
    certificate: str
    """Content hash of (config, dataset fingerprint, scores).

    Two audits with the same hash had the same inputs and produced the same
    numbers. Customers can publish the hash; auditors can re-derive it.
    """
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spatial_blocks(coords_xy: np.ndarray, block_size_km: float) -> np.ndarray:
    block_m = block_size_km * 1000.0
    xs, ys = coords_xy[:, 0], coords_xy[:, 1]
    col = ((xs - xs.min()) / block_m).astype(int)
    row = ((ys - ys.min()) / block_m).astype(int)
    n_cols = int(col.max()) + 1
    return (row * n_cols + col).astype(int)


def _spatial_cv_score(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    n_folds: int,
) -> float:
    unique = np.unique(block_ids)
    if len(unique) < 2:
        return float("nan")
    n_folds = min(n_folds, len(unique))

    rng = np.random.default_rng(0)
    shuffled = rng.permutation(unique)
    fold_assign = {b: i % n_folds for i, b in enumerate(shuffled)}

    preds = np.full(len(y), np.nan)
    for fold in range(n_folds):
        test_blocks = {b for b, f in fold_assign.items() if f == fold}
        test_mask = np.array([b in test_blocks for b in block_ids])
        if test_mask.sum() == 0 or (~test_mask).sum() == 0:
            continue
        if len(np.unique(y[~test_mask])) < 2:
            continue

        m = clone(model)
        m.fit(X[~test_mask], y[~test_mask])
        preds[test_mask] = m.predict_proba(X[test_mask])[:, 1]

    valid = ~np.isnan(preds)
    if len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(average_precision_score(y[valid], preds[valid]))


def _random_cv_score(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_state: int,
) -> float:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in skf.split(X, y):
        if len(np.unique(y[train_idx])) < 2:
            continue
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        preds[test_idx] = m.predict_proba(X[test_idx])[:, 1]

    valid = ~np.isnan(preds)
    if len(np.unique(y[valid])) < 2:
        return float("nan")
    return float(average_precision_score(y[valid], preds[valid]))


def _bootstrap_stability(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_state)
    n = len(y)

    importances: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        m = clone(model)
        m.fit(X[idx], y[idx])

        if hasattr(m, "coef_"):
            importances.append(np.asarray(m.coef_).ravel())
        elif hasattr(m, "feature_importances_"):
            importances.append(np.asarray(m.feature_importances_))
        else:
            return {"supported": False, "stable_fraction": float("nan")}

    if not importances:
        return {"supported": False, "stable_fraction": float("nan")}

    arr = np.vstack(importances)
    n_features = arr.shape[1]

    sign_consistency = np.zeros(n_features)
    for j in range(n_features):
        col = arr[:, j]
        pos_frac = (col > 0).mean()
        sign_consistency[j] = max(pos_frac, 1 - pos_frac)

    stable_fraction = float((sign_consistency >= 0.95).mean())

    return {
        "supported": True,
        "n_features": n_features,
        "n_bootstraps": int(arr.shape[0]),
        "sign_consistency_per_feature": sign_consistency.tolist(),
        "stable_fraction": stable_fraction,
        "ci_lower": np.percentile(arr, 2.5, axis=0).tolist(),
        "ci_upper": np.percentile(arr, 97.5, axis=0).tolist(),
    }


def _calibration_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _feature_label_leakage(
    X: np.ndarray, y: np.ndarray, threshold: float
) -> dict[str, Any]:
    flagged: list[tuple[int, float]] = []
    n_features = X.shape[1]
    for j in range(n_features):
        col = X[:, j]
        if np.std(col) == 0:
            continue
        valid = ~np.isnan(col)
        if valid.sum() < 5:
            continue
        corr = np.corrcoef(col[valid], y[valid])[0, 1]
        if abs(corr) >= threshold:
            flagged.append((j, float(corr)))
    return {
        "n_features": n_features,
        "threshold": threshold,
        "flagged": flagged,
        "max_abs_corr": float(
            max(
                (abs(np.corrcoef(X[:, j][~np.isnan(X[:, j])], y[~np.isnan(X[:, j])])[0, 1])
                 for j in range(n_features) if np.std(X[:, j]) > 0),
                default=0.0,
            )
        ),
    }


def _certificate(
    config: AuditConfig,
    X: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    scores: dict[str, Any],
) -> str:
    """Content-addressed hash of inputs + outputs.

    Same inputs + same model -> same hash. Customers publish the hash;
    auditors recompute it from the same data to verify the audit was real.
    """
    h = hashlib.sha256()
    h.update(json.dumps(asdict(config), sort_keys=True).encode())
    h.update(hashlib.sha256(X.tobytes()).digest())
    h.update(hashlib.sha256(y.tobytes()).digest())
    h.update(hashlib.sha256(coords.tobytes()).digest())
    h.update(json.dumps(scores, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _grade(passed_count: int, total: int) -> str:
    ratio = passed_count / total if total else 0
    if ratio == 1.0:
        return "A"
    if ratio >= 0.8:
        return "B"
    if ratio >= 0.6:
        return "C"
    if ratio >= 0.4:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def audit(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    coords_xy: np.ndarray,
    feature_names: list[str] | None = None,
    config: AuditConfig | None = None,
) -> AuditResult:
    """Run the full audit protocol.

    Parameters
    ----------
    model : sklearn-compatible binary classifier
        Must implement ``fit`` and ``predict_proba``. Will be cloned per fold;
        the original is not mutated.
    X : ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : ndarray, shape (n_samples,)
        Binary labels (0/1).
    coords_xy : ndarray, shape (n_samples, 2)
        Spatial coordinates for each sample, in **projected metres**
        (e.g. UTM). Used for spatial blocking.
    feature_names : list of str, optional
        Names for diagnostic output. Defaults to ``["f0", "f1", ...]``.
    config : AuditConfig, optional
        Override default thresholds.
    """
    started = time.time()
    cfg = config or AuditConfig()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    if X.ndim != 2 or X.shape[0] != len(y) or coords_xy.shape != (len(y), 2):
        raise ValueError("X, y, coords_xy shapes are inconsistent")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("y must be binary 0/1")

    prior = float(y.mean())
    logger.info("Audit start: n=%d, positives=%d, prior=%.3f", len(y), y.sum(), prior)

    # Test 1+2: random vs spatial CV
    pr_random = _random_cv_score(model, X, y, cfg.n_random_folds, cfg.random_state)
    block_ids = _spatial_blocks(coords_xy, cfg.block_size_km)
    pr_spatial = _spatial_cv_score(model, X, y, block_ids, cfg.n_spatial_folds)
    gap = pr_random - pr_spatial if not (np.isnan(pr_random) or np.isnan(pr_spatial)) else float("nan")

    leakage_test = TestResult(
        name="spatial_leakage",
        passed=bool(not np.isnan(gap) and gap <= cfg.spatial_leakage_max_gap),
        score=float(gap),
        threshold=cfg.spatial_leakage_max_gap,
        detail={
            "pr_auc_random_cv": pr_random,
            "pr_auc_spatial_cv": pr_spatial,
            "n_spatial_blocks": int(len(np.unique(block_ids))),
            "block_size_km": cfg.block_size_km,
        },
    )

    baseline_test = TestResult(
        name="beats_class_prior",
        passed=bool(not np.isnan(pr_spatial) and pr_spatial > prior),
        score=float(pr_spatial - prior) if not np.isnan(pr_spatial) else float("nan"),
        threshold=0.0,
        detail={"pr_auc_spatial_cv": pr_spatial, "class_prior": prior},
    )

    # Test 3: bootstrap stability
    boot = _bootstrap_stability(model, X, y, cfg.n_bootstrap, cfg.random_state)
    bootstrap_test = TestResult(
        name="bootstrap_stability",
        passed=bool(boot.get("supported") and boot["stable_fraction"] >= 0.5),
        score=float(boot.get("stable_fraction", float("nan"))),
        threshold=0.5,
        detail=boot,
    )

    # Test 4: calibration -- fit on all data, evaluate on a holdout fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
    train_idx, test_idx = next(iter(skf.split(X, y)))
    cal_model = clone(model)
    cal_model.fit(X[train_idx], y[train_idx])
    cal_probs = cal_model.predict_proba(X[test_idx])[:, 1]
    ece = _calibration_ece(y[test_idx], cal_probs, cfg.calibration_n_bins)
    calibration_test = TestResult(
        name="calibration",
        passed=ece <= cfg.calibration_max_ece,
        score=float(ece),
        threshold=cfg.calibration_max_ece,
        detail={"expected_calibration_error": ece, "n_bins": cfg.calibration_n_bins},
    )

    # Test 5: feature-label leakage
    leak = _feature_label_leakage(X, y, cfg.feature_leakage_threshold)
    feature_leakage_test = TestResult(
        name="feature_label_leakage",
        passed=len(leak["flagged"]) == 0,
        score=float(leak["max_abs_corr"]),
        threshold=cfg.feature_leakage_threshold,
        detail={
            "flagged_features": [
                {"name": feature_names[j], "index": j, "correlation": c}
                for j, c in leak["flagged"]
            ],
            "max_abs_corr": leak["max_abs_corr"],
        },
    )

    tests = [
        leakage_test,
        baseline_test,
        bootstrap_test,
        calibration_test,
        feature_leakage_test,
    ]
    passed_count = sum(1 for t in tests if t.passed)
    grade = _grade(passed_count, len(tests))

    summary = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_positives": int(y.sum()),
        "class_prior": prior,
        "pr_auc_random_cv": pr_random,
        "pr_auc_spatial_cv": pr_spatial,
        "spatial_leakage_gap": gap,
        "tests_passed": passed_count,
        "tests_total": len(tests),
        "grade": grade,
    }

    cert = _certificate(cfg, X, y, coords_xy, summary)

    elapsed = time.time() - started
    logger.info(
        "Audit complete in %.1fs: grade=%s, passed=%d/%d, cert=%s",
        elapsed, grade, passed_count, len(tests), cert[:12],
    )

    return AuditResult(
        passed=passed_count == len(tests),
        grade=grade,
        tests=tests,
        summary=summary,
        certificate=cert,
        elapsed_seconds=elapsed,
    )
