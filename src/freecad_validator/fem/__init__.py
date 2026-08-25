"""Deterministic validation of solved FreeCAD/CalculiX FEM analyses.

The scorer compares a candidate FCStd with an engineer-generated solved
reference and can replay the candidate solve before accepting its stored result
fields.
"""

from freecad_validator.fem.schema import (
    CATEGORIES,
    CRITICAL_GATES,
    DEFAULT_WEIGHTS,
    CaseDefinition,
    FailureMode,
    Finding,
    ScoringReport,
    Submission,
    SubScore,
    grade_from_score,
)
from freecad_validator.fem.scorer import score_result
from freecad_validator.fem.step_interface import (
    ExtractionError,
    RuntimeEnvironmentError,
    build_case,
    score_step_fcstd,
    score_trusted_payloads,
)
from freecad_validator.fem.validator import FEMValidator

FEMScoringReport = ScoringReport

__all__ = [
    "CATEGORIES",
    "CRITICAL_GATES",
    "DEFAULT_WEIGHTS",
    "CaseDefinition",
    "ExtractionError",
    "FEMScoringReport",
    "FEMValidator",
    "FailureMode",
    "Finding",
    "RuntimeEnvironmentError",
    "ScoringReport",
    "Submission",
    "SubScore",
    "build_case",
    "grade_from_score",
    "score_result",
    "score_step_fcstd",
    "score_trusted_payloads",
]
