"""Shared physical and numerical checks for saved static analyses."""

from freecad_validator.fem import metrics, validators
from freecad_validator.fem.schema import Finding
from freecad_validator.fem.setup_similarity import _number


def evaluate_validity(case, sub):
    physical, physical_findings = validators.evaluate_physical(case, sub)
    convergence = metrics.convergence_from_study(sub.mesh.get("convergence_study") or [])
    numerical, numerical_findings = validators.evaluate_numerical(case, sub, convergence)
    cross_findings = validators.detect_submission_failures(case, sub)
    numerical = max(
        0.0,
        numerical - sum(f.penalty for f in cross_findings if f.category == "numerical_reliability"),
    )
    findings = physical_findings + numerical_findings + cross_findings
    if sub.solver.get("converged") is not True:
        findings.append(
            Finding(
                "validity",
                "SOLVER_NOT_CONVERGED",
                "critical",
                "A successful solve must be established.",
            )
        )
    if validators._norm_analysis(sub.analysis_type) != "static":
        findings.append(
            Finding(
                "validity",
                "UNSUPPORTED_ANALYSIS",
                "critical",
                "Static scoring requires a supported static candidate analysis.",
            )
        )
    residual = sub.solver.get("residual")
    if residual is not None and (not _number(residual) or residual < 0):
        findings.append(
            Finding(
                "validity",
                "INVALID_SOLVER_EVIDENCE",
                "critical",
                "Solver residual must be finite and nonnegative.",
            )
        )
    if not sub.boundary_conditions:
        findings.append(
            Finding(
                "validity",
                "MISSING_BOUNDARY_CONDITION",
                "critical",
                "A supported static analysis requires restraints.",
            )
        )
    invalid_values = [key for key, value in sub.results.items() if not _number(value) or value < 0]
    if invalid_values or any(
        key not in sub.results for key in ("max_displacement_mm", "max_von_mises_MPa")
    ):
        findings.append(
            Finding(
                "validity",
                "INVALID_RESULTS",
                "critical",
                "Displacement and stress must be present, finite and nonnegative.",
            )
        )
    for material in sub.materials or [sub.material]:
        if not _number(material.get("E_MPa")) or material["E_MPa"] <= 0:
            findings.append(
                Finding(
                    "validity",
                    "INVALID_MATERIAL",
                    "critical",
                    "Every material requires a finite positive Young's modulus.",
                )
            )
        nu = material.get("nu")
        if not _number(nu) or not -1 < nu < 0.5:
            findings.append(
                Finding(
                    "validity",
                    "INVALID_MATERIAL",
                    "critical",
                    "Linear isotropic elasticity requires -1 < nu < 0.5.",
                )
            )
    jacobian = sub.mesh.get("quality", {}).get("min_jacobian")
    if jacobian is not None and (not _number(jacobian) or jacobian <= 0):
        findings.append(
            Finding(
                "validity",
                "INVERTED_ELEMENTS",
                "critical",
                "Nonpositive or nonfinite element Jacobian.",
            )
        )
    return physical, numerical, findings, invalid_values
