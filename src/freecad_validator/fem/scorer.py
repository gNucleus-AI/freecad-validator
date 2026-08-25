"""Top-level scorer: ``score_result(case, submission) -> ScoringReport``.

Pipeline
--------
1. Run the six category evaluators (problem setup, mesh, numerical, physical,
   accuracy, reporting) plus cross-cutting submission validation.
2. Aggregate category raw scores with the (case-overridable) weights into a
   0-100 weighted score. If the case has no reference solution, the accuracy
   weight is redistributed proportionally to the other categories.
3. Apply the VALIDITY GATE: if any critical failure mode in ``CRITICAL_GATES``
   is raised, the overall score is 0 (invalid); otherwise the weighted sum
   stands. This keeps a physically impossible / hallucinated / non-reproducible /
   wrong-part result from passing no matter how polished it looks.
4. Emit pass/fail flags, detected failure modes, numerical comparisons,
   actionable feedback, suggested fixes, confidence and reproducibility status.

Deterministic and offline by construction.
"""

from __future__ import annotations

from freecad_validator.fem import reference_compare, rubric, validators
from freecad_validator.fem.schema import (
    CATEGORIES,
    CRITICAL_GATES,
    CaseDefinition,
    Finding,
    ScoringReport,
    Submission,
    SubScore,
    grade_from_score,
)

# Actionable fix suggestions keyed by failure-mode / finding code.
SUGGESTED_FIXES = {
    "WRONG_ANALYSIS_TYPE": "Re-run with the analysis type the case requires (e.g. modal/buckling/thermal), not a static stress pass.",
    "UNIT_INCONSISTENCY": "Make the unit system self-consistent (e.g. N-mm-MPa) and state it explicitly.",
    "WRONG_MATERIAL": "Use the specified material properties (E, nu, density) from the case definition.",
    "MISSING_BOUNDARY_CONDITION": "Add restraints that remove all six rigid-body modes before solving.",
    "MISSING_LOAD": "Apply the load defined by the case before solving.",
    "UNJUSTIFIED_MESH": "Report element type and quality metrics, and justify the mesh density.",
    "NO_CONVERGENCE_STUDY": "Run >=3 systematically refined meshes and report the GCI / successive change.",
    "INVERTED_ELEMENTS": "Fix the mesh: negative Jacobians mean inverted elements; remesh or improve geometry healing.",
    "POOR_MESH_QUALITY": "Improve worst-element aspect ratio/skewness, especially in high-gradient regions.",
    "SOLVER_NOT_CONVERGED": "Resolve the non-convergence (step size, contact, stabilization) before trusting results.",
    "HALLUCINATED_CONVERGENCE": "Do not claim mesh independence without a refinement study that demonstrates it.",
    "HALLUCINATED_SOLVER_OUTPUT": "Provide the solver log, result file and reaction balance that produced these numbers.",
    "UNVERIFIED_SOLVER_OUTPUT": "Save a complete, re-runnable FEM analysis whose stored fields are reproduced by a fresh CalculiX solve of the same mesh.",
    "REACTION_IMBALANCE": "Check equilibrium: total reaction must balance the applied load.",
    "ENERGY_IMBALANCE": "Reduce artificial/hourglass energy below ~5% of internal energy (better elements/mesh).",
    "INTERNAL_INCONSISTENCY": "Make every reported value identical across text, tables and plots.",
    "PHYSICALLY_IMPOSSIBLE": "Re-examine the model: the result violates basic physics (sign, magnitude, admissibility).",
    "SINGULARITY_MISINTERPRETED": "Treat the sharp feature as a singularity (add a fillet or report a notch-stress/away-from-singularity value), not a converged peak.",
    "FREQ_NONPHYSICAL": "Negative/zero constrained frequencies imply rigid-body modes; fix constraints.",
    "BUCKLING_NONPHYSICAL": "A non-positive buckling factor is invalid; check the eigenvalue extraction and preload.",
    "ACCURACY_GROSS_ERROR": "Reconcile the result with the analytical/reference value; a large gap signals a setup error.",
    "NON_REPRODUCIBLE": "Ship the input deck/script and result file so the run can be reproduced and audited.",
    "PARTIAL_REPRODUCIBILITY": "Include both the input deck/script and the result file.",
    "CASE_MISMATCH": "Write the report against THIS geometry/loading, not boilerplate from another case.",
    "MISSING_RESULTS": "Report the quantities of interest the case asks for.",
    "INCOMPLETE_REPORT": "Add the missing report sections (assumptions, interpretation, limitations, etc.).",
    "MESH_NOT_CONVERGED": "Refine until the quantity of interest changes < ~3% between meshes.",
    "LARGE_DEFLECTION": "Switch to geometrically nonlinear analysis; linear small-strain theory is invalid here.",
    "MESH_BUDGET_EXCEEDED": "Reduce the element count to within the reference mesh budget (the sub-score is 0 at/over the budget ceiling); coarsen away from high-gradient regions or use local refinement only where needed.",
    "MESH_BUDGET_OVER": "Trim the element count toward the reference baseline to recover mesh-budget points.",
    "MESH_BUDGET_UNDERRESOLVED": "Report the mesh you actually solved on; a mesh far below the reference baseline is under-resolved and earns no mesh-budget credit. Refine until the field is resolved, then trim toward (not below) the baseline.",
}


def _subscore(category: str, raw: float, weight: float, findings: list[Finding]) -> SubScore:
    return SubScore(category=category, raw_score=round(raw, 2), weight=round(weight, 4),
                    weighted_points=round(raw * weight, 3),
                    findings=[f.to_dict() for f in findings])


def score_result(case: CaseDefinition, submission: Submission) -> ScoringReport:
    """Score one submission against one case definition."""
    if submission.case_id and case.case_id and submission.case_id != case.case_id:
        # Not fatal because a caller may remap case identifiers.
        pass

    # ---- run evaluators --------------------------------------------------
    ps_score, ps_find = validators.validate_problem_setup(case, submission)
    mq_score, mq_find, conv = validators.evaluate_mesh(case, submission)
    nr_score, nr_find = validators.evaluate_numerical(case, submission, conv)
    pv_score, pv_find = validators.evaluate_physical(case, submission)
    acc_score, acc_find, comparisons = reference_compare.compare_to_reference(case, submission)
    rep_score, rep_find = rubric.evaluate_reporting(case, submission)
    mb_score, mb_find = validators.evaluate_mesh_budget(case, submission)

    # Cross-cutting submission checks and geometry fidelity.
    submission_findings = validators.detect_submission_failures(case, submission)
    submission_findings += validators.validate_geometry_fidelity(case, submission)

    cat_findings: dict[str, list[Finding]] = {c: [] for c in CATEGORIES}
    for f in ps_find:
        cat_findings["problem_setup"].append(f)
    for f in mq_find:
        cat_findings["mesh_quality"].append(f)
    for f in nr_find:
        cat_findings["numerical_reliability"].append(f)
    for f in pv_find:
        cat_findings["physical_validity"].append(f)
    for f in acc_find:
        cat_findings["accuracy_vs_reference"].append(f)
    for f in rep_find:
        cat_findings["engineering_reporting"].append(f)
    for f in mb_find:
        cat_findings["mesh_budget"].append(f)
    for f in submission_findings:
        cat_findings.setdefault(f.category, []).append(f)

    # Each category's raw score = the evaluator's base score MINUS only the
    # penalties from cross-cutting submission findings routed into it. The
    # penalty-based evaluators already fold their own findings into their base
    # (100 - own penalties); the score-based ones (accuracy, reporting) return a
    # computed score, so we must not re-derive it from penalties.
    cross_pen: dict[str, float] = {}
    for f in submission_findings:
        cross_pen[f.category] = cross_pen.get(f.category, 0.0) + f.penalty

    base_scores = {
        "problem_setup": ps_score,
        "mesh_quality": mq_score,
        "numerical_reliability": nr_score,
        "physical_validity": pv_score,
        "engineering_reporting": rep_score,
    }
    raw_scores = {cat: max(0.0, base - cross_pen.get(cat, 0.0))
                  for cat, base in base_scores.items()}

    # ---- weighting: score-based categories that can be N/A (accuracy, mesh_budget)
    # have their weight redistributed across the rest when not applicable ----
    weights = case.weights()
    optional_scores = {"accuracy_vs_reference": acc_score, "mesh_budget": mb_score}
    applicable: dict[str, bool] = {}
    na_weight = 0.0
    for cat, sc in optional_scores.items():
        if sc is None:
            na_weight += weights.pop(cat, 0.0)
            raw_scores[cat] = 0.0
            applicable[cat] = False
        else:
            raw_scores[cat] = max(0.0, sc - cross_pen.get(cat, 0.0))
            applicable[cat] = True
    if na_weight > 0.0:
        tot = sum(weights.values())
        if tot > 0.0:
            weights = {k: v + v / tot * na_weight for k, v in weights.items()}
    accuracy_applicable = applicable["accuracy_vs_reference"]

    subscores: dict[str, float] = {}
    subscores_details: dict[str, dict] = {}
    weighted_total = 0.0
    for cat in CATEGORIES:
        w = weights.get(cat, 0.0)
        raw = raw_scores.get(cat, 0.0)
        ss = _subscore(cat, raw, w, cat_findings.get(cat, []))
        subscores[cat] = ss.raw_score
        subscores_details[cat] = ss.__dict__
        weighted_total += ss.weighted_points

    # ---- validity gate: any critical gate failure -> overall 0 -----------
    all_findings = [f for fs in cat_findings.values() for f in fs]
    failure_modes: list[dict] = []
    gate_codes: list[str] = []
    for f in all_findings:
        if f.severity in ("major", "critical") and f.code in _KNOWN_FAILURE_CODES:
            failure_modes.append({"code": f.code, "severity": f.severity,
                                  "category": f.category, "evidence": f.message})
        if f.severity == "critical" and f.code in CRITICAL_GATES:
            gate_codes.append(f.code)

    gates_triggered = [{"reason": c} for c in dict.fromkeys(gate_codes)]
    overall = 0.0 if gate_codes else weighted_total
    overall = max(0.0, min(100.0, overall))
    grade = grade_from_score(overall)

    # ---- flags -----------------------------------------------------------
    repro = validators.reproducibility_status(submission)
    crit_codes = {f.code for f in all_findings if f.severity == "critical"}
    flags = {
        "setup_correct": raw_scores["problem_setup"] >= 70 and
                         "WRONG_ANALYSIS_TYPE" not in crit_codes and
                         "MISSING_BOUNDARY_CONDITION" not in crit_codes,
        "mesh_adequate": raw_scores["mesh_quality"] >= 60 and "INVERTED_ELEMENTS" not in crit_codes,
        "numerically_reliable": "SOLVER_NOT_CONVERGED" not in crit_codes and
                                "HALLUCINATED_CONVERGENCE" not in crit_codes and
                                "REACTION_IMBALANCE" not in crit_codes and
                                "INTERNAL_INCONSISTENCY" not in crit_codes,
        "physically_valid": "PHYSICALLY_IMPOSSIBLE" not in crit_codes and
                            "FREQ_NONPHYSICAL" not in crit_codes and
                            "BUCKLING_NONPHYSICAL" not in crit_codes,
        "accurate_vs_reference": accuracy_applicable and all(
            c["within_tol"] for c in comparisons if c["critical"]) if comparisons else None,
        "reproducible": repro == "reproducible",
        "not_hallucinated": not ({"HALLUCINATED_SOLVER_OUTPUT", "UNVERIFIED_SOLVER_OUTPUT",
                                  "HALLUCINATED_CONVERGENCE", "INTERNAL_INCONSISTENCY"}
                                 & crit_codes),
        "mesh_within_budget": (raw_scores["mesh_budget"] > 0.0)
                              if applicable["mesh_budget"] else None,
    }

    # ---- feedback & fixes ------------------------------------------------
    feedback: list[str] = []
    fixes: list[str] = []
    seen_fix = set()
    for f in sorted(all_findings, key=lambda x: {"critical": 0, "major": 1, "minor": 2, "info": 3}[x.severity]):
        if f.severity in ("critical", "major"):
            feedback.append(f"[{f.severity.upper()}/{f.category}] {f.message}")
        if f.code in SUGGESTED_FIXES and f.code not in seen_fix and f.severity in ("critical", "major"):
            fixes.append(SUGGESTED_FIXES[f.code])
            seen_fix.add(f.code)
    if not feedback:
        feedback.append("No major or critical issues detected; result looks sound on the checked axes.")

    # ---- confidence ------------------------------------------------------
    confidence = _confidence(submission, accuracy_applicable, conv)

    # ---- evidence summary ------------------------------------------------
    evidence = _evidence(case, submission, conv, comparisons, accuracy_applicable)

    return ScoringReport(
        case_id=case.case_id,
        overall_score=round(overall, 1),
        grade=grade,
        subscores=subscores,
        subscores_details=subscores_details,
        pass_fail_flags=flags,
        failure_modes_detected=_dedup_failures(failure_modes),
        gates_triggered=gates_triggered,
        numerical_comparisons=comparisons,
        engineering_feedback=feedback,
        suggested_fixes=fixes,
        confidence=round(confidence, 2),
        reproducibility_status=repro,
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
_KNOWN_FAILURE_CODES = set(CRITICAL_GATES) | {
    "POOR_MESH_QUALITY", "NO_CONVERGENCE_STUDY", "MESH_NOT_CONVERGED", "ENERGY_IMBALANCE",
    "ACCURACY_GROSS_ERROR", "LARGE_DEFLECTION", "PARTIAL_REPRODUCIBILITY", "INCOMPLETE_REPORT",
    "HIGH_RESIDUAL", "UNDER_RESOLVED", "MATERIAL_NOT_STATED", "BC_COUNT_LOW",
    "MESH_BUDGET_EXCEEDED", "MESH_BUDGET_OVER", "MESH_BUDGET_UNDERRESOLVED",
}


def _dedup_failures(failures: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for f in failures:
        key = (f["code"], f["category"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def _confidence(submission: Submission, accuracy_applicable: bool, conv: dict) -> float:
    """How much the scorer trusts its own verdict given the evidence available."""
    c = 1.0
    if not submission.solver:
        c -= 0.15
    if not submission.units:
        c -= 0.05
    if not (submission.mesh or {}).get("quality"):
        c -= 0.1
    if not accuracy_applicable:
        c -= 0.1
    if conv.get("n_grids", 0) < 2:
        c -= 0.05
    if not submission.artifacts:
        c -= 0.1
    return max(0.3, c)


def _evidence(case, submission, conv, comparisons, accuracy_applicable) -> list[str]:
    ev = []
    ev.append(f"analysis_type: required={case.analysis_type}, submitted={submission.analysis_type or 'n/a'}")
    if conv.get("n_grids"):
        ev.append(f"convergence: {conv['n_grids']} grids, converged={conv.get('converged')}, "
                  f"last_change={conv.get('last_rel_change')}, GCI={conv.get('gci_fine')}")
    if accuracy_applicable:
        for c in comparisons:
            ev.append(f"ref[{c['quantity']}]: claimed={c['claimed']} vs {c['reference']} {c['unit']} "
                      f"(err={c['rel_error']}, within_tol={c['within_tol']}, {c['method']})")
    repro = (submission.artifacts or {})
    ev.append(f"artifacts: {sorted(k for k, v in repro.items() if v)}")
    return ev
