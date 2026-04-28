"""FastAPI application for GeoMine.

This is a deliberately thin layer. The interesting work lives in
``geomine.audit`` and ``geomine.predict``. The API exists to give the product
a surface a customer can hit, and to make the validation discipline (the
moat) machine-checkable, not just a claim in a README.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from geomine.audit import AuditConfig, audit as run_audit

logger = logging.getLogger("geomine.api")

app = FastAPI(
    title="GeoMine AI",
    description=(
        "Satellite-powered mineral targeting + model audit authority. "
        "Built on $0 infrastructure. Validated by failure."
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GeoJSONPolygon(BaseModel):
    type: str = Field(..., examples=["Polygon"])
    coordinates: list[list[list[float]]]


class ScoreRequest(BaseModel):
    boundary: GeoJSONPolygon
    commodity: str = Field("PGM", examples=["PGM", "Cr"])


class ScoreZone(BaseModel):
    rank: int
    centroid_lonlat: tuple[float, float]
    area_km2: float
    mean_probability: float
    confidence: float


class ScoreResponse(BaseModel):
    commodity: str
    n_zones: int
    top_zones: list[ScoreZone]
    benchmark_certificate: str
    notes: str


class AuditRequest(BaseModel):
    """Inline payload audit. Suitable for small-N exploration datasets.

    For larger datasets, upload via the (planned) /v1/audit/upload endpoint.
    """
    feature_names: list[str]
    X: list[list[float]]
    y: list[int]
    coords_xy: list[list[float]] = Field(
        ..., description="Projected metres (e.g. UTM). NOT lon/lat."
    )
    block_size_km: float = 25.0
    n_bootstrap: int = 200


class AuditTest(BaseModel):
    name: str
    passed: bool
    score: float
    threshold: float


class AuditResponse(BaseModel):
    grade: str
    passed: bool
    tests: list[AuditTest]
    summary: dict[str, Any]
    certificate: str
    elapsed_seconds: float


class BenchmarkResponse(BaseModel):
    """Pinned headline numbers for the GeoMine model. Verifiable.

    Customers can re-derive the certificate by running
    ``geomine audit data/benchmark/dataset.parquet data/benchmark/model.joblib``.
    """
    model_name: str
    pr_auc_random_cv: float
    pr_auc_spatial_cv: float
    spatial_leakage_gap: float
    n_deposits: int
    geographic_extent: str
    certificate: str
    last_updated: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> dict[str, Any]:
    return {
        "service": "GeoMine AI",
        "version": "0.1.0",
        "tagline": "Validated by failure. Audit-grade mineral targeting.",
        "docs": "/docs",
        "benchmark": "/v1/benchmark",
    }


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/benchmark", response_model=BenchmarkResponse)
def benchmark() -> BenchmarkResponse:
    """Current pinned benchmark for the GeoMine model.

    These numbers are reproduced by the audit module on the published
    dataset. Same inputs -> same certificate hash.
    """
    return BenchmarkResponse(
        model_name="ViT cross-tile (Phase 2)",
        pr_auc_random_cv=0.612,
        pr_auc_spatial_cv=0.453,
        spatial_leakage_gap=0.159,
        n_deposits=17,
        geographic_extent="Great Dyke, Zimbabwe (4 Sentinel-2 tiles)",
        certificate="pending-recompute",
        last_updated="2026-04-28",
    )


@app.post("/v1/audit", response_model=AuditResponse)
def audit_endpoint(req: AuditRequest) -> AuditResponse:
    """Run the GeoMine audit protocol on a customer-supplied dataset.

    The customer ships their own model? Not yet -- this endpoint runs a
    standard logistic regression baseline against their data and reports
    whether the *data itself* exhibits leakage. To audit a specific model,
    use the CLI: ``geomine audit data.parquet model.joblib``.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(req.X, dtype=np.float64)
    y = np.asarray(req.y, dtype=int)
    coords = np.asarray(req.coords_xy, dtype=np.float64)

    if X.ndim != 2 or X.shape[0] != len(y) or coords.shape != (len(y), 2):
        raise HTTPException(400, "X, y, coords_xy shape mismatch")
    if X.shape[1] != len(req.feature_names):
        raise HTTPException(400, "feature_names length must match X columns")

    cfg = AuditConfig(
        block_size_km=req.block_size_km,
        n_bootstrap=req.n_bootstrap,
    )
    result = run_audit(
        LogisticRegression(max_iter=1000),
        X, y, coords,
        feature_names=req.feature_names,
        config=cfg,
    )

    return AuditResponse(
        grade=result.grade,
        passed=result.passed,
        tests=[
            AuditTest(
                name=t.name, passed=t.passed,
                score=float(t.score) if t.score == t.score else 0.0,
                threshold=t.threshold,
            )
            for t in result.tests
        ],
        summary=result.summary,
        certificate=result.certificate,
        elapsed_seconds=result.elapsed_seconds,
    )


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Score a concession boundary against the GeoMine model.

    Stub: full implementation requires the trained model + cached feature
    rasters at deploy time. Returns the structure customers will receive.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "Scoring requires deployed model + feature rasters. "
            "Currently delivered as a per-engagement service. "
            "Contact tatendawalter62@gmail.com for access."
        ),
    )
