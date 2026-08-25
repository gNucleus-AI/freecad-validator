"""Data contract for deterministic FEA/FEM validation.

Three objects flow through the scorer:

* ``CaseDefinition``  - the validation case: geometry, material, expected setup,
  reference solution (analytical / numerical / empirical / expert), physical
  bounds, mesh expectations, and a reporting rubric.
* ``Submission``      - the candidate problem setup, mesh statistics,
  solver outputs, result quantities, the written report and the artifacts it
  claims to have generated.
* ``ScoringReport``   - the scorer's output: overall 0-100 score, flat
  per-category subscores, detailed subscore breakdowns, pass/fail flags,
  detected failure modes, numerical comparisons, engineering feedback,
  confidence and reproducibility status.

Everything is plain dataclasses with ``from_dict`` / ``to_dict`` so cases and
submissions can live as JSON on disk and the report can be serialised verbatim.
The scorer core depends only on the standard library.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# --------------------------------------------------------------------------- #
# Scoring categories and the default weighting scheme (documented in report)   #
# --------------------------------------------------------------------------- #
CATEGORIES = [
    "problem_setup",
    "mesh_quality",
    "mesh_budget",
    "numerical_reliability",
    "physical_validity",
    "accuracy_vs_reference",
    "engineering_reporting",
]

DEFAULT_WEIGHTS: dict[str, float] = {
    "problem_setup": 0.20,
    "mesh_quality": 0.15,
    "mesh_budget": 0.00,          # off by default; activated by reference-based scoring
    "numerical_reliability": 0.20,
    "physical_validity": 0.20,
    "accuracy_vs_reference": 0.15,
    "engineering_reporting": 0.10,
}

SEVERITIES = ("info", "minor", "major", "critical")

# Per-severity default deduction applied to a category's 0-100 raw score.
SEVERITY_PENALTY = {"info": 0.0, "minor": 8.0, "major": 30.0, "critical": 80.0}


# --------------------------------------------------------------------------- #
# Tolerance / accuracy-band configuration (correlated: change one, rest follow)#
# --------------------------------------------------------------------------- #
# Relative-error tolerances for comparing a candidate result to the reference.
DISP_TOL = 0.05            # displacement / natural frequency: <= 5% rel. err = full credit
STRESS_TOL = 0.08          # stresses (mesh-sensitive): <= 8% rel. err = full credit

# Accuracy band shape used by metrics.tolerance_band_score:
#   error <= TOL_BAND_LO * tol  -> 100  (full credit)
#   error >= TOL_BAND_HI * tol  ->   0  (no credit; TOL_BAND_HI is the "fade")
# linear in between, over [TOL_BAND_LO*tol, TOL_BAND_HI*tol]. With TOL_BAND_HI=2
# the displacement sub-score fades 5%->10% and stress fades 8%->16%.
TOL_BAND_LO = 1.0
TOL_BAND_HI = 2.0

# Gross-miss GATE for a critical quantity: off by more than GROSS_TOL is a gross
# failure to reproduce the reference and GATES the overall score to 0. Deliberately
# WIDER than the sub-score band fade (a 10-20% displacement miss earns no accuracy
# credit but is not yet invalid), so the gate keeps its own multiple of DISP_TOL
# rather than tracking TOL_BAND_HI (default = 4 * 0.05 = 0.20).
GROSS_GATE_MULT = 4.0
GROSS_TOL = GROSS_GATE_MULT * DISP_TOL

# Mesh-budget: candidate element count vs the reference baseline. Full credit at or
# below the baseline; zero credit at this multiple of it.
MESH_BUDGET_ZERO_RATIO = 1.3

# Lower floor for the mesh-budget credit. A mesh below this fraction of the reference
# baseline is too coarse to have resolved the field and earns ZERO mesh-budget
# credit (not full), preventing a degenerate mesh from receiving efficiency
# credit. Deliberately small so honest
# coarse-but-valid meshes are spared while degenerate ones are excluded.
MESH_BUDGET_FLOOR_RATIO = 0.05

# Load match: how closely the candidate's applied loading must reproduce the reference.
# A load is an INPUT (given in the prompt or derived from the environment), so it
# should match tightly; a gross magnitude miss or a reversed force means a DIFFERENT
# problem was solved and GATES the score (like a wrong material), even if the result
# numbers happen to land near the reference (the compensating-error blind spot).
LOAD_MAG_TOL = 0.05         # force/pressure magnitude rel-error before a (minor) penalty
LOAD_MAG_GROSS_TOL = 0.20   # ... beyond this it is the wrong load -> critical gate
# Force DIRECTION: a load is an INPUT that must point the way the reference specifies, so the
# candidate's net force at each loaded face is compared to the reference's by angle. Within a
# few degrees it is the same load; past a tight bound it is a different load (a reversal is
# just the extreme - and it even fools the accuracy check, which sees identical magnitudes).
LOAD_DIR_ALIGNED_DEG = 6.0   # <= this: aligned, no finding
LOAD_DIR_GATE_DEG = 15.0     # >= this: wrong load direction -> critical gate (between: penalty)
LOAD_LOC_TOL = 0.15          # centroid offset / characteristic length before a penalty


# --------------------------------------------------------------------------- #
# Failure-mode codes                                                           #
# --------------------------------------------------------------------------- #
class FailureMode:
    WRONG_ANALYSIS_TYPE = "WRONG_ANALYSIS_TYPE"
    WRONG_MATERIAL = "WRONG_MATERIAL"
    UNIT_INCONSISTENCY = "UNIT_INCONSISTENCY"
    MISSING_BOUNDARY_CONDITION = "MISSING_BOUNDARY_CONDITION"
    MISSING_LOAD = "MISSING_LOAD"
    WRONG_LOAD = "WRONG_LOAD"
    UNJUSTIFIED_MESH = "UNJUSTIFIED_MESH"
    NO_CONVERGENCE_STUDY = "NO_CONVERGENCE_STUDY"
    INVERTED_ELEMENTS = "INVERTED_ELEMENTS"
    POOR_MESH_QUALITY = "POOR_MESH_QUALITY"
    SOLVER_NOT_CONVERGED = "SOLVER_NOT_CONVERGED"
    HALLUCINATED_CONVERGENCE = "HALLUCINATED_CONVERGENCE"
    HALLUCINATED_SOLVER_OUTPUT = "HALLUCINATED_SOLVER_OUTPUT"
    UNVERIFIED_SOLVER_OUTPUT = "UNVERIFIED_SOLVER_OUTPUT"
    REACTION_IMBALANCE = "REACTION_IMBALANCE"
    ENERGY_IMBALANCE = "ENERGY_IMBALANCE"
    INTERNAL_INCONSISTENCY = "INTERNAL_INCONSISTENCY"
    PHYSICALLY_IMPOSSIBLE = "PHYSICALLY_IMPOSSIBLE"
    SINGULARITY_MISINTERPRETED = "SINGULARITY_MISINTERPRETED"
    FREQ_NONPHYSICAL = "FREQ_NONPHYSICAL"
    BUCKLING_NONPHYSICAL = "BUCKLING_NONPHYSICAL"
    ACCURACY_GROSS_ERROR = "ACCURACY_GROSS_ERROR"
    NON_REPRODUCIBLE = "NON_REPRODUCIBLE"
    CASE_MISMATCH = "CASE_MISMATCH"
    MISSING_RESULTS = "MISSING_RESULTS"
    GEOMETRY_MISMATCH = "GEOMETRY_MISMATCH"
    PREPROCESSING_NOT_PERFORMED = "PREPROCESSING_NOT_PERFORMED"
    BOOLEAN_NOT_PERFORMED = "BOOLEAN_NOT_PERFORMED"


# Critical (gate) failure modes. The scorer is a VALIDITY GATE: if any of these
# is raised at severity "critical", the overall score is 0 (invalid) regardless
# of how well the weighted categories scored; otherwise the weighted sum stands.
# This makes a physically impossible / hallucinated / wrong-part result fail
# outright, no matter how polished. ACCURACY_GROSS_ERROR gates a candidate that
# misses the reference's critical quantity by more than GROSS_TOL.
CRITICAL_GATES: frozenset = frozenset({
    FailureMode.PHYSICALLY_IMPOSSIBLE,
    FailureMode.GEOMETRY_MISMATCH,
    FailureMode.PREPROCESSING_NOT_PERFORMED,
    FailureMode.BOOLEAN_NOT_PERFORMED,
    FailureMode.HALLUCINATED_SOLVER_OUTPUT,
    FailureMode.UNVERIFIED_SOLVER_OUTPUT,
    FailureMode.MISSING_RESULTS,
    FailureMode.SOLVER_NOT_CONVERGED,
    FailureMode.MISSING_BOUNDARY_CONDITION,
    FailureMode.WRONG_LOAD,
    FailureMode.INVERTED_ELEMENTS,
    FailureMode.HALLUCINATED_CONVERGENCE,
    FailureMode.SINGULARITY_MISINTERPRETED,
    FailureMode.REACTION_IMBALANCE,
    FailureMode.NON_REPRODUCIBLE,
    FailureMode.WRONG_ANALYSIS_TYPE,
    FailureMode.INTERNAL_INCONSISTENCY,
    FailureMode.FREQ_NONPHYSICAL,
    FailureMode.BUCKLING_NONPHYSICAL,
    FailureMode.WRONG_MATERIAL,
    FailureMode.ACCURACY_GROSS_ERROR,
})


@dataclass
class Finding:
    """A single observation the scorer made about a submission."""
    category: str
    code: str
    severity: str
    message: str
    evidence: str = ""
    penalty: float = 0.0  # points subtracted from the category's 0-100 score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Case definition                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceQuantity:
    name: str
    value: float
    unit: str = ""
    tol_rel: float = 0.10
    tol_abs: float | None = None
    critical: bool = False
    note: str = ""


@dataclass
class CaseDefinition:
    case_id: str
    title: str
    category: str = "base"          # caller-defined grouping, e.g. "base" or "fem"
    subtype: str = ""
    analysis_type: str = "static"   # static|modal|buckling|thermal|thermal_mechanical|nonlinear_material|large_deformation|contact|transient
    description: str = ""
    units_expected: dict[str, str] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    material: dict[str, Any] = field(default_factory=dict)
    expected_bcs: list[dict[str, Any]] = field(default_factory=list)
    expected_loads: list[dict[str, Any]] = field(default_factory=list)
    reference: dict[str, Any] = field(default_factory=dict)         # {method, source, quantities:{...}}
    physical_bounds: dict[str, Any] = field(default_factory=dict)
    mesh_expectations: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=dict)
    weights_override: dict[str, float] | None = None

    # convenience -----------------------------------------------------------
    def reference_quantities(self) -> dict[str, ReferenceQuantity]:
        out: dict[str, ReferenceQuantity] = {}
        for name, spec in self.reference.get("quantities", {}).items():
            out[name] = ReferenceQuantity(name=name, **spec)
        return out

    def reference_method(self) -> str:
        return self.reference.get("method", "none")

    def weights(self) -> dict[str, float]:
        if self.weights_override:
            w = dict(DEFAULT_WEIGHTS)
            w.update(self.weights_override)
            total = sum(w.values())
            return {k: v / total for k, v in w.items()}
        return dict(DEFAULT_WEIGHTS)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CaseDefinition:
        known = CaseDefinition.__dataclass_fields__.keys()
        return CaseDefinition(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def load(path: str) -> CaseDefinition:
        with open(path, encoding="utf-8") as fh:
            return CaseDefinition.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# --------------------------------------------------------------------------- #
# Submission                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Submission:
    case_id: str
    analysis_type: str = ""
    units: dict[str, str] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    material: dict[str, Any] = field(default_factory=dict)
    boundary_conditions: list[dict[str, Any]] = field(default_factory=list)
    loads: list[dict[str, Any]] = field(default_factory=list)
    mesh: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    reported_values: dict[str, Any] = field(default_factory=dict)   # {name:{text,table,plot}}
    report: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)         # {name: path_or_null}

    # Caller metadata (not read by scoring logic).
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Submission:
        known = Submission.__dataclass_fields__.keys()
        return Submission(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def load(path: str) -> Submission:
        with open(path, encoding="utf-8") as fh:
            return Submission.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# --------------------------------------------------------------------------- #
# Scoring report                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class SubScore:
    category: str
    raw_score: float            # 0-100 before weighting
    weight: float
    weighted_points: float      # raw_score * weight (contribution to 0-100)
    findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ScoringReport:
    case_id: str
    overall_score: float
    grade: str
    subscores: dict[str, float] = field(default_factory=dict)
    subscores_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    pass_fail_flags: dict[str, bool | None] = field(default_factory=dict)
    failure_modes_detected: list[dict[str, Any]] = field(default_factory=list)
    gates_triggered: list[dict[str, Any]] = field(default_factory=list)
    numerical_comparisons: list[dict[str, Any]] = field(default_factory=list)
    engineering_feedback: list[str] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)
    confidence: float = 1.0
    reproducibility_status: str = "unknown"
    evidence: list[str] = field(default_factory=list)
    runtime_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def summary_line(self) -> str:
        fm = ",".join(sorted({f["code"] for f in self.failure_modes_detected})) or "none"
        return f"{self.case_id}: {self.overall_score:.1f}/100 [{self.grade}] failures={fm}"


def grade_from_score(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 55:
        return "acceptable"
    if score >= 35:
        return "poor"
    return "invalid"
