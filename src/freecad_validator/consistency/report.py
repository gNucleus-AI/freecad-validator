"""Consistency-report data models.

Carries the fields a consumer needs to interpret a check result:
per-bucket findings (consistent / inconsistent / not_found), an
aggregate summary, and an optional top-level error.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParamFinding(BaseModel):
    param: str
    spec_value: Any                              # original spec value (any type)
    measured_value: Optional[Any] = None         # CAD-side value, or None if not_found
    unit: str = ""
    feature: Optional[str] = None                # feature back-reference (e.g., "Pad.Length")
    rel_diff: Optional[float] = None             # |spec − measured| / max(|spec|, |measured|)
    reason: Optional[str] = None                 # short explanation when bucket = inconsistent / not_found


class ReportSummary(BaseModel):
    total_params: int
    consistent: int
    inconsistent: int
    not_found: int
    unexpected_features: int
    consistency_rate: float                      # consistent / total_params (0.0–1.0)
    measurable_rate: float                       # (consistent + inconsistent) / total_params


class ConsistencyReport(BaseModel):
    spec_name: str
    fcstd_path: str
    summary: Optional[ReportSummary] = None      # filled by the orchestrator before return
    consistent: List[ParamFinding] = Field(default_factory=list)
    inconsistent: List[ParamFinding] = Field(default_factory=list)
    not_found: List[ParamFinding] = Field(default_factory=list)
    unexpected_features: List[str] = Field(default_factory=list)
    feature_health: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


def compute_summary(report: ConsistencyReport) -> ReportSummary:
    """Fill-in helper for the `summary` field. Called once per report,
    after all checks and categories have run."""
    n_ok = len(report.consistent)
    n_bad = len(report.inconsistent)
    n_miss = len(report.not_found)
    total = n_ok + n_bad + n_miss
    return ReportSummary(
        total_params=total,
        consistent=n_ok,
        inconsistent=n_bad,
        not_found=n_miss,
        unexpected_features=len(report.unexpected_features),
        consistency_rate=round(n_ok / total, 4) if total else 0.0,
        measurable_rate=round((n_ok + n_bad) / total, 4) if total else 0.0,
    )
