"""Apply trusted preprocessing geometry results to a static FEM report."""

import math
from copy import deepcopy

from freecad_validator.fem.pre_process.errors import EvaluationError
from freecad_validator.fem.schema import ScoringReport, grade_from_score


def apply_preprocessing_score(report: ScoringReport, geometry_score: float | None) -> ScoringReport:
    """Multiply the gated FEM total by geometry credit, without changing its inputs.

    None means no applicable edits, or a non-preprocessing task. Missing required
    evaluation data must raise EvaluationError at the caller, not become None.
    """
    if geometry_score is None:
        return report
    if (
        isinstance(geometry_score, bool)
        or not isinstance(geometry_score, (int, float))
        or not math.isfinite(geometry_score)
        or not 0 <= geometry_score <= 1
    ):
        raise EvaluationError("Preprocessing score must be finite and in [0, 1]")
    if "preprocessing" in report.subscores_details:
        raise EvaluationError("Preprocessing multiplier has already been applied")
    combined = deepcopy(report)
    combined.overall_score = report.overall_score * geometry_score
    combined.grade = grade_from_score(combined.overall_score)
    combined.subscores["preprocessing"] = 100 * geometry_score
    combined.subscores_details["preprocessing"] = {
        "category": "preprocessing",
        "raw_score": 100 * geometry_score,
        "weight": 0.0,
        "weighted_points": 0.0,
        "findings": [],
        "aggregation": "multiplier",
        "multiplier": geometry_score,
        "fem_score_before_preprocessing": report.overall_score,
        "deducted_points": report.overall_score - combined.overall_score,
    }
    combined.pass_fail_flags["preprocessing_correct"] = geometry_score == 1
    combined.evidence.append(
        f"preprocessing: FEM {report.overall_score:.6f} × geometry {geometry_score:.6f}"
        f" = {combined.overall_score:.6f}/100"
    )
    return combined
