"""Formatters for AuditResult.

JSON for machine consumption; Markdown for humans and pitch decks.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from geomine.audit.core import AuditResult


def to_json(result: AuditResult, indent: int = 2) -> str:
    return json.dumps(asdict(result), indent=indent, default=str)


def to_markdown(result: AuditResult, model_name: str = "model") -> str:
    s = result.summary
    pass_emoji = lambda b: "PASS" if b else "FAIL"  # noqa: E731

    lines = [
        f"# GeoMine Audit Report: {model_name}",
        "",
        f"**Grade:** {result.grade}  ({s['tests_passed']}/{s['tests_total']} tests passed)",
        f"**Certificate:** `{result.certificate}`",
        f"**Elapsed:** {result.elapsed_seconds:.1f}s",
        "",
        "## Dataset",
        "",
        f"- Samples: {s['n_samples']:,}",
        f"- Features: {s['n_features']}",
        f"- Positives: {s['n_positives']:,} (prior = {s['class_prior']:.3f})",
        "",
        "## Headline Numbers",
        "",
        f"- PR-AUC (random CV):  **{s['pr_auc_random_cv']:.3f}**",
        f"- PR-AUC (spatial CV): **{s['pr_auc_spatial_cv']:.3f}**",
        f"- Spatial leakage gap: **{s['spatial_leakage_gap']:.3f}**",
        "",
        "## Tests",
        "",
        "| Test | Result | Score | Threshold |",
        "|---|---|---|---|",
    ]
    for t in result.tests:
        lines.append(
            f"| {t.name} | {pass_emoji(t.passed)} | {t.score:.3f} | {t.threshold:.3f} |"
        )

    lines += ["", "## Test Details", ""]
    for t in result.tests:
        lines += [
            f"### {t.name} -- {pass_emoji(t.passed)}",
            "",
            f"- Score: {t.score:.3f} (threshold: {t.threshold:.3f})",
        ]
        if t.detail:
            lines.append("- Details:")
            for k, v in t.detail.items():
                if isinstance(v, list) and len(v) > 8:
                    v = f"[{len(v)} items]"
                lines.append(f"    - {k}: {v}")
        lines.append("")

    lines += [
        "---",
        "",
        "*Audit run by `geomine.audit` v0.1. Hash is content-addressed: same data + same"
        " model -> same hash. Independent verifiers can re-derive it.*",
    ]
    return "\n".join(lines)
