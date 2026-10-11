"""Weighted V1 scoring of trusted saved-analysis facts using shared setup checks."""

import math
from dataclasses import asdict

from freecad_validator.fem import reference_compare, validators
from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.required_setup import required_setup_findings
from freecad_validator.fem.schema import (
    Finding,
    ScoringReport,
    Submission,
    SubScore,
    grade_from_score,
)
from freecad_validator.fem.setup_similarity import _number, _setup
from freecad_validator.fem.static_validity import evaluate_validity


def validate_reference(reference):
    if validators._norm_analysis(reference.get("analysis_type", "")) != "static":
        raise EvaluationError("A supported static reference analysis is required")
    results = reference.get("results", {})
    if any(
        not _number(results.get(key)) or results[key] < 0
        for key in ("max_displacement_mm", "max_von_mises_MPa")
    ):
        raise EvaluationError("Reference requires finite nonnegative displacement and stress")
    if any(not _number(value) or value < 0 for value in results.values()):
        raise EvaluationError("Reference result quantities must be finite and nonnegative")
    for material in reference.get("materials") or [reference.get("material", {})]:
        if (
            not _number(material.get("E_MPa"))
            or material["E_MPa"] <= 0
            or not _number(material.get("nu"))
            or not -1 < material["nu"] < 0.5
        ):
            raise EvaluationError("Reference requires physically admissible elastic properties")
        if "rho_kg_m3" in material and (
            not _number(material["rho_kg_m3"]) or material["rho_kg_m3"] <= 0
        ):
            raise EvaluationError("Reference density must be finite and positive")
    if not reference.get("boundary_conditions"):
        raise EvaluationError("Static reference requires restraints")
    for contact in reference.get("contacts", []):
        if contact.get("type") not in ("contact", "tie") or not all(
            contact.get(side) for side in ("master", "slave")
        ):
            raise EvaluationError("Reference Contact/Tie requires a type and geometric surfaces")
    try:
        _setup(reference.get("geometry", {}), reference, reference)
    except ValueError as exc:
        raise EvaluationError(str(exc)) from exc


def score_label_submission(case, reference, candidate):
    """Score candidate facts against the already validated reference."""
    sub = Submission.from_dict({**candidate, "case_id": case.case_id})
    setup, components, geometry = _setup(reference["geometry"], reference, candidate)
    physical, numerical, findings, invalid_values = evaluate_validity(case, sub)
    budget, budget_findings = validators.evaluate_mesh_budget(case, sub)
    findings.extend(budget_findings)
    if invalid_values:
        accuracy, accuracy_findings, comparisons = 0.0, [], []
    else:
        accuracy, accuracy_findings, comparisons = reference_compare.compare_to_reference(case, sub)
    findings.extend(accuracy_findings)
    required = required_setup_findings(reference, candidate)
    findings.extend(required)
    if setup < 1:
        findings.append(
            Finding(
                "problem_setup",
                "SETUP_MISMATCH",
                "info",
                "Setup agreement receives continuous partial credit.",
                str(components),
            )
        )
    gates = [finding for finding in findings if finding.severity == "critical"]
    critical_codes = {finding.code for finding in gates}
    reproducibility = validators.reproducibility_status(sub)
    raw = {
        "accuracy_vs_reference": accuracy or 0.0,
        "mesh_budget": budget or 0.0,
        "problem_setup": 100 * setup,
        "physical_validity": physical,
        "numerical_reliability": numerical,
    }
    weights = case.weights()
    details = {
        category: asdict(
            SubScore(
                category,
                value,
                weights[category],
                value * weights[category],
                [f.to_dict() for f in findings if f.category == category],
            )
        )
        for category, value in raw.items()
    }
    details["problem_setup"].update(components=components, geometry=geometry)
    total = 0.0 if gates else sum(item["weighted_points"] for item in details.values())
    return ScoringReport(
        case_id=case.case_id,
        overall_score=total,
        grade=grade_from_score(total),
        subscores=raw,
        subscores_details=details,
        pass_fail_flags={
            "valid": not gates,
            "setup_correct": math.isclose(setup, 1, rel_tol=1e-9),
            "required_setup_present": not required,
            "not_hallucinated": not critical_codes.intersection(
                {
                    "UNVERIFIED_SOLVER_OUTPUT",
                    "HALLUCINATED_SOLVER_OUTPUT",
                    "HALLUCINATED_CONVERGENCE",
                    "INTERNAL_INCONSISTENCY",
                }
            ),
            "physically_valid": not any(f.category == "physical_validity" for f in gates)
            and not critical_codes.intersection({"INVALID_MATERIAL", "INVALID_RESULTS"}),
            "numerically_reliable": not any(f.category == "numerical_reliability" for f in gates)
            and not critical_codes.intersection(
                {"SOLVER_NOT_CONVERGED", "INVALID_SOLVER_EVIDENCE"}
            ),
            "mesh_adequate": "INVERTED_ELEMENTS" not in critical_codes and budget == 100.0,
            "reproducible": reproducibility == "reproducible",
            "mesh_within_budget": budget == 100.0,
            "accurate_vs_reference": bool(comparisons)
            and all(row["within_tol"] for row in comparisons if row["critical"]),
        },
        gates_triggered=[{"reason": f.code, "evidence": f.evidence} for f in gates],
        failure_modes_detected=[f.to_dict() for f in findings],
        numerical_comparisons=comparisons,
        engineering_feedback=[f.message for f in findings],
        reproducibility_status=reproducibility,
        evidence=[
            "setup_policy: continuous agreement; required setup checked separately",
            "geometry_source: reference FCStd geometry",
        ],
    )
