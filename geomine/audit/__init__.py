"""GeoMine Audit -- model-agnostic validation for mineral prospectivity classifiers.

The audit module productizes the validation discipline that GeoMine has applied
to its own work: detecting spatial leakage, bootstrapping coefficients, comparing
against the class-prior baseline, and grading calibration.

It is independent of the rest of the GeoMine pipeline. Anyone with a binary
classifier and a labelled point dataset (with coordinates) can run it.
"""

from geomine.audit.core import audit, AuditConfig, AuditResult
from geomine.audit.report import to_markdown, to_json

__all__ = ["audit", "AuditConfig", "AuditResult", "to_markdown", "to_json"]
