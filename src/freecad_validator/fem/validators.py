"""Deterministic, rule-based checks - the backbone of the scorer.

Each ``evaluate_*`` / ``validate_*`` function inspects the ``Submission`` against
the ``CaseDefinition`` and returns ``(raw_score_0_100, [Finding, ...])`` for one
scoring category, except ``detect_submission_failures`` which returns cross-cutting
findings (consistency, reproducibility, hallucinated output, case mismatch)
already tagged with the category they should score against.

Critical findings carry a large category penalty AND a code in
``schema.CRITICAL_GATES`` so the orchestrator gates the overall score to 0.
"""

from __future__ import annotations

import math
from typing import Any

from freecad_validator.fem import metrics
from freecad_validator.fem.schema import (
    LOAD_DIR_ALIGNED_DEG,
    LOAD_DIR_GATE_DEG,
    LOAD_LOC_TOL,
    LOAD_MAG_GROSS_TOL,
    LOAD_MAG_TOL,
    MESH_BUDGET_FLOOR_RATIO,
    MESH_BUDGET_ZERO_RATIO,
    CaseDefinition,
    FailureMode,
    Finding,
    Submission,
)

# --------------------------------------------------------------------------- #
# small helpers                                                               #
# --------------------------------------------------------------------------- #
RESTRAINT_TYPES = {
    "fixed",
    "clamped",
    "encastre",
    "pinned",
    "restraint",
    "displacement",
    "fixed_support",
    "support",
}
PARTIAL_RESTRAINT_TYPES = {"symmetry", "symmetric", "roller", "frictionless", "contact"}

SHAPE_WORDS = [
    "cantilever",
    "simply supported",
    "plate with hole",
    "plate-with-hole",
    "thick cylinder",
    "pressure vessel",
    "bracket",
    "shaft",
    "torsion",
    "truss",
    "frame",
    "column",
    "buckling",
    "beam",
    "lug",
    "clevis",
    "hook",
    "flange",
    "nozzle",
    "heat sink",
    "crank",
    "gearbox",
    "housing",
    "implant",
    "disc",
    "disk",
    "sphere",
    "ring",
]


def _num(d: dict[str, Any], *keys: str) -> float | None:
    """First present numeric value among ``keys`` in dict ``d``."""
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def _magnitude(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return abs(float(v))
    if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, float)) for x in v):
        return math.sqrt(sum(float(x) ** 2 for x in v))
    return None


def _score_from(findings: list[Finding]) -> float:
    return max(0.0, 100.0 - sum(f.penalty for f in findings))


def _norm_analysis(a: str) -> str:
    a = (a or "").lower()
    if "modal" in a or "frequency" in a or "eigen" in a or "vibrat" in a:
        return "modal"
    if "buckl" in a:
        return "buckling"
    if "thermal" in a and "mech" in a:
        return "thermal_mechanical"
    if "thermal" in a or "heat" in a:
        return "thermal"
    if "transient" in a or "dynamic" in a or "explicit" in a:
        return "transient"
    if "nonlinear" in a and ("mat" in a or "plast" in a):
        return "nonlinear_material"
    if "large" in a or "geom" in a:
        return "large_deformation"
    if "static" in a or "linear" in a or "stress" in a:
        return "static"
    return a or "static"


def _has_restraint(bcs: list[dict[str, Any]]) -> tuple[bool, bool]:
    """(has_full_restraint, has_partial_restraint)."""
    full = any((b.get("type", "").lower() in RESTRAINT_TYPES) for b in bcs)
    partial = any((b.get("type", "").lower() in PARTIAL_RESTRAINT_TYPES) for b in bcs)
    return full, partial


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _loc(centroid: list[float] | None) -> str:
    return (
        "unlocated"
        if centroid is None
        else f"({centroid[0]:.0f}, {centroid[1]:.0f}, {centroid[2]:.0f}) mm"
    )


def _cluster_loads(
    loads: list[dict[str, Any]], kind: str, mag_keys: tuple[str, ...], cluster_tol: float
) -> list[dict[str, Any]]:
    """Group `kind` loads into per-location clusters (one loaded face/edge), summing the
    component forces at each location into a net vector. Loads with no resolved centroid
    collapse into a single 'unlocated' cluster, so the comparison degrades gracefully to
    the net-aggregate when the adapter could not resolve locations. Each cluster carries
    its centroid, summed magnitude, and net force vector (None if a member lacks direction)."""
    clusters: list[dict[str, Any]] = []
    for ld in loads:
        if ld.get("type") != kind:
            continue
        mag = _num(ld, *mag_keys)
        if mag is None:
            continue
        c = ld.get("centroid")
        c = [float(x) for x in c] if isinstance(c, (list, tuple)) and len(c) == 3 else None
        d = ld.get("direction")
        d = [float(x) for x in d] if isinstance(d, (list, tuple)) and len(d) == 3 else None
        host = next(
            (
                cl
                for cl in clusters
                if (cl["centroid"] is None) == (c is None)
                and (c is None or _dist(cl["centroid"], c) <= cluster_tol)
            ),
            None,
        )
        if host is None:
            host = {"centroid": c, "vec": [0.0, 0.0, 0.0], "mag": 0.0, "_dir": True}
            clusters.append(host)
        host["mag"] += abs(mag)
        if d is None:
            host["_dir"] = False
        else:
            host["vec"] = [host["vec"][i] + mag * d[i] for i in range(3)]
    for cl in clusters:
        if not cl["_dir"]:
            cl["vec"] = None
    return clusters


def _match_clusters(exp: list[dict[str, Any]], got: list[dict[str, Any]], match_tol: float):
    """Greedy nearest-centroid pairing of loaded locations (closest pair first); the
    unlocated clusters match each other. Returns (pairs, unmatched_expected, unmatched_got)."""
    candidates = []
    for i, e in enumerate(exp):
        for j, g in enumerate(got):
            if (e["centroid"] is None) != (g["centroid"] is None):
                continue
            d = 0.0 if e["centroid"] is None else _dist(e["centroid"], g["centroid"])
            if e["centroid"] is None or d <= match_tol:
                candidates.append((d, i, j))
    used_e: set = set()
    used_g: set = set()
    pairs = []
    for _, i, j in sorted(candidates):
        if i not in used_e and j not in used_g:
            used_e.add(i)
            used_g.add(j)
            pairs.append((exp[i], got[j]))
    unmatched_e = [e for i, e in enumerate(exp) if i not in used_e]
    unmatched_g = [g for j, g in enumerate(got) if j not in used_g]
    return pairs, unmatched_e, unmatched_g


def _mag_finding(
    got_m: float, exp_m: float, kind: str, unit: str, loc: str, findings: list[Finding], cat: str
) -> None:
    """Compare two load magnitudes at one location; a gross miss gates."""
    if exp_m <= 0:
        return
    err = metrics.relative_error(got_m, exp_m)
    where = "" if loc == "unlocated" else f" at {loc}"
    if err > LOAD_MAG_GROSS_TOL:
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_LOAD,
                "critical",
                f"Applied {kind} {got_m:g} {unit}{where} differs from the reference's "
                f"{exp_m:g} {unit} by {err * 100:.0f}% - wrong load.",
                penalty=80,
            )
        )
    elif err > LOAD_MAG_TOL:
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_LOAD,
                "major",
                f"Applied {kind} {got_m:g} {unit}{where} differs from the reference's "
                f"{exp_m:g} {unit} by {err * 100:.0f}%.",
                penalty=30,
            )
        )


def _compare_force_pair(
    e: dict[str, Any], g: dict[str, Any], findings: list[Finding], cat: str
) -> None:
    """Compare one matched loaded face: net force magnitude, then direction. Falls back to
    summed magnitude when direction is unavailable or the reference's net force is ~zero."""
    loc = _loc(e["centroid"])
    where = "" if loc == "unlocated" else f" at {loc}"
    if e["vec"] is None or g["vec"] is None or _norm(e["vec"]) < 1e-9:
        _mag_finding(g["mag"], e["mag"], "force", "N", loc, findings, cat)
        return
    _mag_finding(_norm(g["vec"]), _norm(e["vec"]), "force", "N", loc, findings, cat)
    ang = _angle_deg(e["vec"], g["vec"])
    if ang is None:
        return
    if ang >= LOAD_DIR_GATE_DEG:
        # past the tight alignment bound the load points the wrong way -> wrong problem.
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_LOAD,
                "critical",
                f"Applied force direction is {ang:.0f} deg from the reference's{where} - wrong "
                "load direction.",
                penalty=80,
            )
        )
    elif ang > LOAD_DIR_ALIGNED_DEG:
        # small offset: a real direction error, but not yet a different problem.
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_LOAD,
                "major",
                f"Applied force direction is {ang:.0f} deg from the reference's{where}.",
                penalty=30,
            )
        )


def _angle_deg(a: list[float], b: list[float]) -> float | None:
    la = math.sqrt(sum(x * x for x in a))
    lb = math.sqrt(sum(x * x for x in b))
    if la < 1e-12 or lb < 1e-12:
        return None
    cos = sum(a[i] * b[i] for i in range(3)) / (la * lb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _compare_loads(
    expected: list[dict[str, Any]], got: list[dict[str, Any]], cat: str, char_len: float | None
) -> list[Finding]:
    """Check the candidate reproduces the LABEL's loading, MATCHING loads by location
    first: each side's loads are clustered per loaded face (component forces summed into a
    net vector), faces are paired by centroid (comparable on the shared source solid), and
    each pair is compared on net magnitude + direction. An unmatched face is flagged as a
    missing or spurious load. Without resolved centroids this degrades to one net-aggregate
    comparison. Magnitude/direction are compared only when both sides carry the data, so an
    equivalent reformulation (a force modelled as the equivalent pressure) is flagged for
    review, not gated."""
    findings: list[Finding] = []
    cluster_tol = max(1.0, 1e-3 * char_len) if char_len else 1.0
    match_tol = LOAD_LOC_TOL * char_len if char_len else float("inf")

    # forces: match loaded faces, then compare net magnitude + direction per face ----
    exp_fc = _cluster_loads(expected, "force", ("magnitude_N", "force_N", "magnitude"), cluster_tol)
    got_fc = _cluster_loads(got, "force", ("magnitude_N", "force_N", "magnitude"), cluster_tol)
    if exp_fc and not got_fc:
        extra = (
            " (a pressure was used instead - verify equivalence)"
            if any(ld.get("type") == "pressure" for ld in got)
            else ""
        )
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_LOAD,
                "major",
                f"Label applies {sum(c['mag'] for c in exp_fc):g} N of force but the "
                f"candidate applies no force load{extra}.",
                penalty=30,
            )
        )
    elif exp_fc:
        pairs, un_e, un_g = _match_clusters(exp_fc, got_fc, match_tol)
        for e, g in pairs:
            _compare_force_pair(e, g, findings, cat)
        for e in un_e:
            findings.append(
                Finding(
                    cat,
                    FailureMode.WRONG_LOAD,
                    "major",
                    f"Label applies {e['mag']:g} N of force at {_loc(e['centroid'])} but the "
                    "candidate applies none there.",
                    penalty=25,
                )
            )
        for g in un_g:
            findings.append(
                Finding(
                    cat,
                    FailureMode.WRONG_LOAD,
                    "major",
                    f"Candidate applies {g['mag']:g} N of force at {_loc(g['centroid'])} not "
                    "present in the reference.",
                    penalty=25,
                )
            )

    # pressures: match loaded faces, then compare magnitude per face -----------------
    exp_pc = _cluster_loads(expected, "pressure", ("magnitude_Pa",), cluster_tol)
    got_pc = _cluster_loads(got, "pressure", ("magnitude_Pa",), cluster_tol)
    if exp_pc and not got_pc:
        extra = (
            " (a force was used instead - verify equivalence)"
            if any(ld.get("type") == "force" for ld in got)
            else ""
        )
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_LOAD,
                "major",
                f"Label applies {sum(c['mag'] for c in exp_pc):g} Pa pressure but the "
                f"candidate applies none{extra}.",
                penalty=30,
            )
        )
    elif exp_pc:
        pairs, un_e, un_g = _match_clusters(exp_pc, got_pc, match_tol)
        for e, g in pairs:
            _mag_finding(g["mag"], e["mag"], "pressure", "Pa", _loc(e["centroid"]), findings, cat)
        for e in un_e:
            findings.append(
                Finding(
                    cat,
                    FailureMode.WRONG_LOAD,
                    "major",
                    f"Label applies {e['mag']:g} Pa pressure at {_loc(e['centroid'])} but the "
                    "candidate applies none there.",
                    penalty=25,
                )
            )
        for g in un_g:
            findings.append(
                Finding(
                    cat,
                    FailureMode.WRONG_LOAD,
                    "major",
                    f"Candidate applies {g['mag']:g} Pa pressure at {_loc(g['centroid'])} not "
                    "present in the reference.",
                    penalty=25,
                )
            )

    # self-weight / gravity on-off ---------------------------------------------------
    if any(ld.get("type") == "self_weight" for ld in expected) and not any(
        ld.get("type") == "self_weight" for ld in got
    ):
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_LOAD,
                "minor",
                "Label includes self-weight/gravity but the candidate omits it.",
                penalty=15,
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# A. Problem setup correctness                                                #
# --------------------------------------------------------------------------- #
def validate_problem_setup(case: CaseDefinition, sub: Submission) -> tuple[float, list[Finding]]:
    findings: list[Finding] = []
    cat = "problem_setup"

    # analysis type ---------------------------------------------------------
    want = _norm_analysis(case.analysis_type)
    got = _norm_analysis(sub.analysis_type)
    if not sub.analysis_type:
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_ANALYSIS_TYPE,
                "minor",
                "Analysis type not stated.",
                penalty=8,
            )
        )
    elif got != want:
        findings.append(
            Finding(
                cat,
                FailureMode.WRONG_ANALYSIS_TYPE,
                "critical",
                f"Analysis type '{sub.analysis_type}' does not match required "
                f"'{case.analysis_type}'.",
                evidence=f"got={got} want={want}",
                penalty=80,
            )
        )

    # units -----------------------------------------------------------------
    if not sub.units:
        findings.append(
            Finding(cat, FailureMode.UNIT_INCONSISTENCY, "minor", "Units not declared.", penalty=6)
        )
    else:
        consistent = metrics.stress_unit_consistent(sub.units)
        if consistent is False:
            findings.append(
                Finding(
                    cat,
                    FailureMode.UNIT_INCONSISTENCY,
                    "major",
                    "Declared units are dimensionally inconsistent "
                    f"(force={sub.units.get('force')}, length={sub.units.get('length')}, "
                    f"stress={sub.units.get('stress')}).",
                    penalty=30,
                )
            )

    # material (expected vs actual) ----------------------------------------
    exp_E = _num(case.material, "E_MPa", "youngs_modulus_MPa")
    got_E = _num(sub.material, "E_MPa", "youngs_modulus_MPa")
    if exp_E and got_E:
        err = metrics.relative_error(got_E, exp_E)
        if err > 0.20:
            findings.append(
                Finding(
                    cat,
                    FailureMode.WRONG_MATERIAL,
                    "critical",
                    f"Young's modulus {got_E:g} MPa differs from the specified "
                    f"{exp_E:g} MPa by {err * 100:.0f}% - wrong material.",
                    penalty=80,
                )
            )
    elif exp_E and not got_E:
        findings.append(
            Finding(cat, "MATERIAL_NOT_STATED", "minor", "Young's modulus not reported.", penalty=8)
        )

    # boundary conditions ---------------------------------------------------
    full, partial = _has_restraint(sub.boundary_conditions)
    case_full, _ = _has_restraint(case.expected_bcs)
    # transient/explicit dynamics (e.g. a drop test) can be in free flight with
    # no classic restraint - rigid-body motion is physical there - so it is not
    # in the set that requires a fixed support.
    needs_restraint = want in (
        "static",
        "buckling",
        "thermal_mechanical",
        "nonlinear_material",
        "large_deformation",
        "contact",
    )
    expects_bc = len(case.expected_bcs) > 0
    if needs_restraint and expects_bc:
        if case_full and not full:
            # the case calls for a real restraint and the submission lacks one
            if partial:
                findings.append(
                    Finding(
                        cat,
                        FailureMode.MISSING_BOUNDARY_CONDITION,
                        "major",
                        "Only partial/symmetry restraints found where a full restraint "
                        "is required; rigid-body modes may not be removed.",
                        penalty=30,
                    )
                )
            else:
                findings.append(
                    Finding(
                        cat,
                        FailureMode.MISSING_BOUNDARY_CONDITION,
                        "critical",
                        "No restraint/fixed boundary condition - the model is "
                        "under-constrained (rigid-body motion).",
                        penalty=80,
                    )
                )
        elif (not case_full) and not (full or partial):
            # symmetry-reducible case, but the submission has no restraint at all
            findings.append(
                Finding(
                    cat,
                    FailureMode.MISSING_BOUNDARY_CONDITION,
                    "critical",
                    "No restraint at all - even a symmetry model needs its symmetry "
                    "planes constrained to remove rigid-body motion.",
                    penalty=80,
                )
            )
        elif len(sub.boundary_conditions) < len(case.expected_bcs):
            findings.append(
                Finding(
                    cat,
                    "BC_COUNT_LOW",
                    "minor",
                    f"Fewer boundary conditions ({len(sub.boundary_conditions)}) than "
                    f"expected ({len(case.expected_bcs)}).",
                    penalty=10,
                )
            )
    elif expects_bc and len(sub.boundary_conditions) < len(case.expected_bcs):
        findings.append(
            Finding(
                cat,
                "BC_COUNT_LOW",
                "minor",
                f"Fewer boundary conditions ({len(sub.boundary_conditions)}) than "
                f"expected ({len(case.expected_bcs)}).",
                penalty=10,
            )
        )

    # loads -----------------------------------------------------------------
    expects_load = len(case.expected_loads) > 0
    if expects_load and not sub.loads and want not in ("modal",):
        findings.append(
            Finding(
                cat,
                FailureMode.MISSING_LOAD,
                "major",
                "No loads applied although the case requires loading.",
                penalty=30,
            )
        )
    elif expects_load and sub.loads:
        # A load is present; verify that it reproduces the reference loading.
        # (magnitude, direction, location), not just that something was applied.
        findings.extend(
            _compare_loads(
                case.expected_loads, sub.loads, cat, _num(case.geometry, "characteristic_length_mm")
            )
        )

    return _score_from(findings), findings


# --------------------------------------------------------------------------- #
# B. Mesh quality and convergence                                             #
# --------------------------------------------------------------------------- #
def evaluate_mesh(
    case: CaseDefinition, sub: Submission
) -> tuple[float, list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    cat = "mesh_quality"
    mesh = sub.mesh or {}

    if not mesh:
        findings.append(
            Finding(
                cat,
                FailureMode.UNJUSTIFIED_MESH,
                "major",
                "No mesh information provided - mesh quality cannot be assessed.",
                penalty=40,
            )
        )
        return _score_from(findings), findings, {"n_grids": 0}

    quality = mesh.get("quality", {}) or {}

    # inverted elements (negative Jacobian) - fatal
    min_jac = quality.get("min_jacobian")
    if min_jac is not None and min_jac <= 0:
        findings.append(
            Finding(
                cat,
                FailureMode.INVERTED_ELEMENTS,
                "critical",
                f"Negative/zero minimum Jacobian ({min_jac:g}) indicates inverted "
                "elements; the discretisation is invalid.",
                penalty=85,
            )
        )

    q_score, q_notes = metrics.mesh_quality_score(quality)
    if q_score < 100:
        sev = "major" if q_score < 60 else "minor"
        findings.append(
            Finding(
                cat,
                FailureMode.POOR_MESH_QUALITY,
                sev,
                "Element-quality metrics out of band: " + "; ".join(q_notes)
                if q_notes
                else "Element-quality metrics out of band.",
                penalty=min(40.0, 100.0 - q_score),
            )
        )

    # mesh density
    nel = mesh.get("num_elements")
    min_el = case.mesh_expectations.get("min_elements")
    if nel is not None and min_el and nel < min_el:
        findings.append(
            Finding(
                cat,
                "UNDER_RESOLVED",
                "major",
                f"Only {nel} elements; case expects at least ~{min_el} for a resolved field.",
                penalty=22,
            )
        )

    # convergence study
    study = mesh.get("convergence_study", []) or []
    conv = metrics.convergence_from_study(study)
    expect_conv = case.mesh_expectations.get("expect_convergence", True) and _norm_analysis(
        case.analysis_type
    ) in (
        "static",
        "thermal",
        "thermal_mechanical",
        "nonlinear_material",
        "large_deformation",
        "contact",
    )
    singular = bool(case.mesh_expectations.get("singular_features"))

    if expect_conv:
        if conv["n_grids"] < 2:
            findings.append(
                Finding(
                    cat,
                    FailureMode.NO_CONVERGENCE_STUDY,
                    "major",
                    "No mesh-convergence study (need >=2-3 refinements to show "
                    "the result is mesh-independent).",
                    penalty=28,
                )
            )
        elif not conv["converged"]:
            if singular and conv.get("diverging"):
                findings.append(
                    Finding(
                        cat,
                        "SINGULARITY_PRESENT",
                        "info",
                        "Peak quantity diverges with refinement, consistent with a "
                        "known stress singularity at a sharp feature.",
                        penalty=0,
                    )
                )
            else:
                lrc = conv.get("last_rel_change")
                findings.append(
                    Finding(
                        cat,
                        "MESH_NOT_CONVERGED",
                        "major",
                        f"Mesh not converged (last successive change "
                        f"{(lrc or 0) * 100:.1f}% > 3%).",
                        penalty=25,
                    )
                )
        else:
            gci = conv.get("gci_fine")
            if gci is not None and gci > 5.0:
                findings.append(
                    Finding(
                        cat,
                        "HIGH_GCI",
                        "minor",
                        f"Grid Convergence Index {gci:.1f}% > 5% - discretisation "
                        "uncertainty is high.",
                        penalty=8,
                    )
                )

    # local refinement at stress concentrations
    refine_at = case.mesh_expectations.get("refine_at")
    if refine_at and not mesh.get("local_refinement"):
        findings.append(
            Finding(
                cat,
                "NO_LOCAL_REFINEMENT",
                "minor",
                f"No local refinement reported near {', '.join(refine_at)} "
                "where stress gradients are steep.",
                penalty=10,
            )
        )

    return _score_from(findings), findings, conv


# --------------------------------------------------------------------------- #
# C. Numerical reliability                                                    #
# --------------------------------------------------------------------------- #
def evaluate_numerical(
    case: CaseDefinition, sub: Submission, conv: dict[str, Any]
) -> tuple[float, list[Finding]]:
    findings: list[Finding] = []
    cat = "numerical_reliability"
    solver = sub.solver or {}
    want = _norm_analysis(case.analysis_type)

    # solver convergence
    converged = solver.get("converged")
    if converged is False:
        findings.append(
            Finding(
                cat,
                FailureMode.SOLVER_NOT_CONVERGED,
                "critical",
                "Solver reported non-convergence; results are unreliable.",
                penalty=80,
            )
        )
    elif converged is None and not solver:
        findings.append(
            Finding(
                cat,
                "NO_SOLVER_EVIDENCE",
                "minor",
                "No solver convergence/residual evidence provided.",
                penalty=10,
            )
        )

    # residual
    resid = solver.get("residual")
    if isinstance(resid, (int, float)) and resid > 1e-3:
        sev = "major" if resid > 1e-1 else "minor"
        findings.append(
            Finding(
                cat,
                "HIGH_RESIDUAL",
                sev,
                f"Solver residual {resid:g} is large.",
                penalty=20 if sev == "major" else 8,
            )
        )

    # reaction-force equilibrium (static-type analyses with a known applied load)
    if want in ("static", "nonlinear_material", "large_deformation", "contact"):
        applied = _magnitude(solver.get("applied_load_N"))
        reaction = _magnitude(solver.get("reaction_force_N"))
        if applied and reaction is not None and applied > 0:
            imb = abs(reaction - applied) / applied
            if imb > 0.25:
                findings.append(
                    Finding(
                        cat,
                        FailureMode.REACTION_IMBALANCE,
                        "critical",
                        f"Reaction force {reaction:g} N does not balance the applied "
                        f"load {applied:g} N ({imb * 100:.0f}% imbalance) - equilibrium "
                        "is violated.",
                        penalty=80,
                    )
                )
            elif imb > 0.05:
                findings.append(
                    Finding(
                        cat,
                        FailureMode.REACTION_IMBALANCE,
                        "major",
                        f"Reaction/applied imbalance {imb * 100:.1f}% exceeds 5%.",
                        penalty=30,
                    )
                )
        elif sub.loads and not (solver.get("replay_accepted") or solver.get("replay_verified")):
            findings.append(
                Finding(
                    cat,
                    "NO_REACTION_CHECK",
                    "minor",
                    "Trusted applied-load resultant and reaction-force balance are "
                    "not both reported (cannot confirm equilibrium).",
                    penalty=10,
                )
            )

    # energy balance / artificial (hourglass) energy
    energy = solver.get("energy", {}) or {}
    internal = _num(energy, "internal", "ALLIE", "strain")
    artificial = _num(energy, "artificial", "ALLAE", "hourglass")
    if internal and artificial is not None and internal > 0:
        ratio = artificial / internal
        if ratio > 0.5:
            findings.append(
                Finding(
                    cat,
                    FailureMode.ENERGY_IMBALANCE,
                    "critical",
                    f"Artificial/hourglass energy is {ratio * 100:.0f}% of internal "
                    "energy - the solution is dominated by numerical artefacts.",
                    penalty=80,
                )
            )
        elif ratio > 0.10:
            findings.append(
                Finding(
                    cat,
                    FailureMode.ENERGY_IMBALANCE,
                    "major",
                    f"Artificial energy {ratio * 100:.0f}% of internal energy (>10%).",
                    penalty=28,
                )
            )
        elif ratio > 0.05:
            findings.append(
                Finding(
                    cat,
                    FailureMode.ENERGY_IMBALANCE,
                    "minor",
                    f"Artificial energy {ratio * 100:.1f}% of internal energy (>5%).",
                    penalty=8,
                )
            )

    # hallucinated convergence: claims mesh-independence with no/contradictory study
    claim = bool(sub.report.get("convergence_claim")) if sub.report else False
    if claim:
        if conv.get("n_grids", 0) < 2:
            findings.append(
                Finding(
                    cat,
                    FailureMode.HALLUCINATED_CONVERGENCE,
                    "critical",
                    "Report claims a mesh-converged result but provides no "
                    "convergence study to support it.",
                    penalty=80,
                )
            )
        elif not conv.get("converged") and not case.mesh_expectations.get("singular_features"):
            findings.append(
                Finding(
                    cat,
                    FailureMode.HALLUCINATED_CONVERGENCE,
                    "critical",
                    "Report claims convergence but the supplied study has not converged.",
                    penalty=80,
                )
            )

    # singularity misinterpreted: claims a finite converged peak at a singular feature
    if case.mesh_expectations.get("singular_features") and conv.get("diverging"):
        peak = _num(sub.results, "max_von_mises_MPa", "max_stress_MPa")
        if peak is not None and claim:
            findings.append(
                Finding(
                    cat,
                    FailureMode.SINGULARITY_MISINTERPRETED,
                    "critical",
                    "A finite converged peak stress is claimed at a feature that is "
                    "a stress singularity (stress diverges with refinement).",
                    penalty=80,
                )
            )

    # nonlinear / transient step adequacy
    if want in ("nonlinear_material", "large_deformation", "transient"):
        inc_done = solver.get("increments_completed")
        inc_req = solver.get("increments_requested")
        if solver.get("diverged_increments") or (
            isinstance(inc_done, (int, float))
            and isinstance(inc_req, (int, float))
            and inc_done < inc_req
        ):
            findings.append(
                Finding(
                    cat,
                    "INCOMPLETE_STEPPING",
                    "major",
                    "Load/time stepping did not complete - step size likely inadequate.",
                    penalty=28,
                )
            )

    return _score_from(findings), findings


# --------------------------------------------------------------------------- #
# D. Physical validity                                                        #
# --------------------------------------------------------------------------- #
def evaluate_physical(case: CaseDefinition, sub: Submission) -> tuple[float, list[Finding]]:
    findings: list[Finding] = []
    cat = "physical_validity"
    res = sub.results or {}
    want = _norm_analysis(case.analysis_type)

    if not res:
        findings.append(
            Finding(
                cat,
                FailureMode.MISSING_RESULTS,
                "critical",
                "No result quantities reported.",
                penalty=85,
            )
        )
        return _score_from(findings), findings

    # von Mises non-negativity / finiteness
    vm = _num(res, "max_von_mises_MPa", "max_stress_MPa")
    if vm is not None:
        if not math.isfinite(vm) or vm < 0:
            findings.append(
                Finding(
                    cat,
                    FailureMode.PHYSICALLY_IMPOSSIBLE,
                    "critical",
                    f"von Mises stress {vm} is negative or non-finite - impossible.",
                    penalty=85,
                )
            )
        elif vm > 1.0e7:  # > 10 TPa: not a real material stress
            findings.append(
                Finding(
                    cat,
                    FailureMode.PHYSICALLY_IMPOSSIBLE,
                    "critical",
                    f"von Mises stress {vm:g} MPa is physically impossible.",
                    penalty=85,
                )
            )

    # displacement plausibility
    disp = _num(res, "max_displacement_mm", "max_disp_mm")
    char = _num(case.geometry, "characteristic_length_mm", "length_mm", "L_mm")
    if disp is not None and char:
        if not math.isfinite(disp):
            findings.append(
                Finding(
                    cat,
                    FailureMode.PHYSICALLY_IMPOSSIBLE,
                    "critical",
                    "Displacement is non-finite.",
                    penalty=85,
                )
            )
        else:
            ratio = abs(disp) / char
            if ratio > 1.0:
                findings.append(
                    Finding(
                        cat,
                        FailureMode.PHYSICALLY_IMPOSSIBLE,
                        "critical",
                        f"Max displacement {disp:g} mm exceeds the characteristic "
                        f"size {char:g} mm (ratio {ratio:.1f}) - impossible for this "
                        "model.",
                        penalty=85,
                    )
                )
            elif ratio > 0.2 and want == "static":
                findings.append(
                    Finding(
                        cat,
                        "LARGE_DEFLECTION",
                        "major",
                        f"Displacement is {ratio * 100:.0f}% of the characteristic size; "
                        "small-deflection linear static theory is invalid here.",
                        penalty=28,
                    )
                )
    if disp is not None and disp == 0 and sub.loads and want in ("static",):
        findings.append(
            Finding(
                cat,
                FailureMode.PHYSICALLY_IMPOSSIBLE,
                "major",
                "Zero displacement under a non-zero load is implausible.",
                penalty=30,
            )
        )

    # material admissibility
    E = _num(sub.material, "E_MPa", "youngs_modulus_MPa")
    nu = _num(sub.material, "nu", "poisson", "PoissonRatio")
    if E is not None and E <= 0:
        findings.append(
            Finding(
                cat,
                FailureMode.PHYSICALLY_IMPOSSIBLE,
                "critical",
                "Non-positive Young's modulus.",
                penalty=85,
            )
        )
    if nu is not None:
        if nu <= -1.0 or nu > 0.6:
            findings.append(
                Finding(
                    cat,
                    FailureMode.PHYSICALLY_IMPOSSIBLE,
                    "critical",
                    f"Poisson's ratio {nu} is outside the admissible range (-1, 0.5].",
                    penalty=85,
                )
            )
        elif nu >= 0.5:
            findings.append(
                Finding(
                    cat,
                    "NU_INCOMPRESSIBLE",
                    "major",
                    f"Poisson's ratio {nu} >= 0.5 (near-incompressible) needs special "
                    "elements; standard elements lock.",
                    penalty=28,
                )
            )
        elif nu < 0:
            findings.append(
                Finding(
                    cat,
                    "NU_AUXETIC",
                    "minor",
                    f"Negative Poisson's ratio {nu} is rare - confirm it is intended.",
                    penalty=6,
                )
            )

    # modal
    if want == "modal":
        freqs = res.get("natural_frequencies_Hz")
        free_free = not case.expected_bcs
        if not freqs:
            findings.append(
                Finding(
                    cat,
                    FailureMode.MISSING_RESULTS,
                    "critical",
                    "Modal analysis but no natural frequencies reported.",
                    penalty=85,
                )
            )
        else:
            nums = [f for f in freqs if isinstance(f, (int, float))]
            if any((not math.isfinite(f)) or f < -1e-3 for f in nums):
                findings.append(
                    Finding(
                        cat,
                        FailureMode.FREQ_NONPHYSICAL,
                        "critical",
                        "Negative or non-finite natural frequency.",
                        penalty=85,
                    )
                )
            elif not free_free and nums and min(nums) <= 1e-3:
                findings.append(
                    Finding(
                        cat,
                        FailureMode.FREQ_NONPHYSICAL,
                        "critical",
                        "Near-zero fundamental frequency in a constrained model "
                        "indicates an unconstrained rigid-body mode.",
                        penalty=80,
                    )
                )
            if nums and nums != sorted(nums):
                findings.append(
                    Finding(
                        cat,
                        "FREQ_NOT_ASCENDING",
                        "minor",
                        "Natural frequencies are not in ascending order.",
                        penalty=8,
                    )
                )

    # buckling
    if want == "buckling":
        bf = _num(res, "buckling_factor", "load_factor", "buckling_load_factor")
        if bf is None:
            findings.append(
                Finding(
                    cat,
                    FailureMode.MISSING_RESULTS,
                    "critical",
                    "Buckling analysis but no buckling/load factor reported.",
                    penalty=85,
                )
            )
        elif bf <= 0:
            findings.append(
                Finding(
                    cat,
                    FailureMode.BUCKLING_NONPHYSICAL,
                    "critical",
                    f"Non-positive buckling factor {bf} is non-physical.",
                    penalty=85,
                )
            )

    # thermal bounds
    if want in ("thermal", "thermal_mechanical"):
        tmax = _num(res, "max_temperature_C", "Tmax_C")
        bounds = case.physical_bounds.get("temperature_C")
        if (
            tmax is not None
            and bounds
            and (tmax < bounds.get("min", -1e9) or tmax > bounds.get("max", 1e9))
        ):
            findings.append(
                Finding(
                    cat,
                    FailureMode.PHYSICALLY_IMPOSSIBLE,
                    "major",
                    f"Peak temperature {tmax} C is outside the plausible range {bounds}.",
                    penalty=30,
                )
            )

    # explicit physical-bounds from the case (e.g., expected sign of deflection)
    sign = case.physical_bounds.get("displacement_sign")
    got_sign = res.get("displacement_sign")
    if sign and got_sign and sign != got_sign:
        findings.append(
            Finding(
                cat,
                "WRONG_DEFLECTION_DIRECTION",
                "major",
                f"Deflection direction '{got_sign}' contradicts the expected "
                f"'{sign}' for this loading.",
                penalty=28,
            )
        )

    return _score_from(findings), findings


# --------------------------------------------------------------------------- #
# G. Agent-specific failure modes (cross-cutting)                             #
# --------------------------------------------------------------------------- #
def detect_submission_failures(case: CaseDefinition, sub: Submission) -> list[Finding]:
    findings: list[Finding] = []

    # 1) trusted solver replay: a saved result object is not proof that the solver ran.
    replay = (sub.meta or {}).get("solver_replay")
    if isinstance(replay, dict) and not replay.get("passed"):
        status = replay.get("status", "failed")
        failures = replay.get("failures") or []
        detail = "; ".join(str(failure) for failure in failures[:3])
        findings.append(
            Finding(
                "numerical_reliability",
                FailureMode.UNVERIFIED_SOLVER_OUTPUT,
                "critical",
                f"Stored FEM results failed trusted CalculiX replay verification ({status})"
                + (f": {detail}" if detail else "."),
                evidence=str(replay),
                penalty=80,
            )
        )

    # 2) internal consistency: same quantity must agree across text/table/plot/results
    for name, places in (sub.reported_values or {}).items():
        if not isinstance(places, dict):
            continue
        vals = [v for v in places.values() if isinstance(v, (int, float))]
        res_val = sub.results.get(name)
        if isinstance(res_val, (int, float)):
            vals.append(res_val)
        disc = metrics.max_pairwise_discrepancy(vals)
        if disc > 0.20:
            findings.append(
                Finding(
                    "numerical_reliability",
                    FailureMode.INTERNAL_INCONSISTENCY,
                    "critical",
                    f"Reported '{name}' disagrees across text/table/plot/results by "
                    f"{disc * 100:.0f}% - numbers are inconsistent (hallucination "
                    "signal).",
                    evidence=str(places),
                    penalty=70,
                )
            )
        elif disc > 0.05:
            findings.append(
                Finding(
                    "numerical_reliability",
                    FailureMode.INTERNAL_INCONSISTENCY,
                    "major",
                    f"Reported '{name}' varies by {disc * 100:.1f}% across the submission.",
                    evidence=str(places),
                    penalty=28,
                )
            )

    # 3) hallucinated solver output: results exist but no solver evidence at all
    has_results = bool(sub.results) and any(
        isinstance(v, (int, float)) for v in sub.results.values()
    )
    has_solver_evidence = bool(sub.solver) and (
        "converged" in sub.solver
        or "residual" in sub.solver
        or "reaction_force_N" in sub.solver
        or "iterations" in sub.solver
    )
    result_file = (sub.artifacts or {}).get("result_file")
    if has_results and not has_solver_evidence and not result_file:
        findings.append(
            Finding(
                "numerical_reliability",
                FailureMode.HALLUCINATED_SOLVER_OUTPUT,
                "critical",
                "Result quantities are reported with no solver evidence and no "
                "result file - the outputs may be fabricated.",
                penalty=80,
            )
        )

    # 4) reproducibility: need a way to re-run (input deck or script) AND a result file
    arts = sub.artifacts or {}
    replay_accepted = bool(
        (sub.solver or {}).get("replay_accepted") or (sub.solver or {}).get("replay_verified")
    )
    has_setup = bool(arts.get("input_deck") or arts.get("script") or replay_accepted)
    has_result_file = bool(arts.get("result_file"))
    if not has_setup and not has_result_file:
        findings.append(
            Finding(
                "numerical_reliability",
                FailureMode.NON_REPRODUCIBLE,
                "critical",
                "No input deck/script and no result file - the analysis is not "
                "reproducible or verifiable.",
                penalty=70,
            )
        )
    elif not has_setup or not has_result_file:
        missing = "input deck/script" if not has_setup else "result file"
        findings.append(
            Finding(
                "engineering_reporting",
                "PARTIAL_REPRODUCIBILITY",
                "major",
                f"Missing {missing}; reproducibility is only partial.",
                penalty=20,
            )
        )

    # 5) copy-paste / case mismatch: report describes a different geometry
    text = " ".join(
        str(sub.report.get(k, "")) for k in ("interpretation", "assumptions", "summary")
    ).lower()
    if isinstance(sub.report.get("assumptions"), list):
        text += " " + " ".join(str(x) for x in sub.report["assumptions"]).lower()
    correct = (case.subtype or "").lower().replace("_", " ")
    case_words = [w for w in SHAPE_WORDS if w in (correct + " " + case.title.lower())]
    if text:
        mentioned = [w for w in SHAPE_WORDS if w in text]
        wrong = [
            w
            for w in mentioned
            if w not in case_words and not any(cw in w or w in cw for cw in case_words)
        ]
        # only flag if it names a competing shape and never names the right one
        if wrong and case_words and not any(cw in text for cw in case_words):
            findings.append(
                Finding(
                    "engineering_reporting",
                    FailureMode.CASE_MISMATCH,
                    "major",
                    f"The write-up describes '{wrong[0]}' but the case is "
                    f"'{correct or case.title}' - boilerplate/copy-paste suspected.",
                    penalty=30,
                )
            )
    return findings


def validate_geometry_fidelity(case: CaseDefinition, sub: Submission) -> list[Finding]:
    """Did the submission actually analyse the target geometry?

    Compares the target geometry (e.g. from a STEP file, carried on
    ``case.geometry``) against the body the submission analysed
    (``sub.geometry``). Volume is the primary, mesh-independent signal; the
    bounding-box characteristic length is the fallback. A no-op when either side
    lacks the data, so this is safe to call on reference-based cases too.
    """
    cat = "problem_setup"
    sv, fv = _num(case.geometry, "volume_mm3"), _num(sub.geometry, "volume_mm3")
    if sv and fv and sv > 0:
        err = abs(fv - sv) / sv
        if err > 0.25:
            return [
                Finding(
                    cat,
                    FailureMode.GEOMETRY_MISMATCH,
                    "critical",
                    f"Analysed body volume {fv:.0f} mm^3 differs from the target geometry "
                    f"{sv:.0f} mm^3 by {err * 100:.0f}% - a different or incomplete part was "
                    "analysed.",
                    penalty=80,
                )
            ]
        if err > 0.05:
            return [
                Finding(
                    cat,
                    FailureMode.GEOMETRY_MISMATCH,
                    "major",
                    f"Analysed body volume differs from the target geometry by {err * 100:.1f}%.",
                    penalty=30,
                )
            ]
        return []
    sc, fc = (
        _num(case.geometry, "characteristic_length_mm"),
        _num(sub.geometry, "characteristic_length_mm"),
    )
    if sc and fc and sc > 0:
        err = abs(fc - sc) / sc
        if err > 0.25:
            return [
                Finding(
                    cat,
                    FailureMode.GEOMETRY_MISMATCH,
                    "critical",
                    f"Analysed body size differs from the target by {err * 100:.0f}% "
                    "(no volume available to compare).",
                    penalty=80,
                )
            ]
        if err > 0.05:
            return [
                Finding(
                    cat,
                    FailureMode.GEOMETRY_MISMATCH,
                    "major",
                    f"Analysed body size differs from the target by {err * 100:.1f}%.",
                    penalty=30,
                )
            ]
    return []


def evaluate_mesh_budget(
    case: CaseDefinition, sub: Submission
) -> tuple[float | None, list[Finding]]:
    """Mesh-efficiency category: candidate element count vs a baseline.

    The baseline (for example, the reference element count) is carried on
    ``case.mesh_expectations["baseline_num_elements"]``; the ceiling ratio is
    ``budget_zero_ratio`` (default 1.3). Returns ``(score_0_100, findings)`` or
    ``(None, [])`` when no baseline is available (so the weight is redistributed).
    Findings carry penalty 0 because the *score* already encodes the deduction.
    """
    baseline = case.mesh_expectations.get("baseline_num_elements")
    if not baseline:
        return None, []
    cand = (sub.mesh or {}).get("num_elements")
    if not cand:
        return None, []
    zero_at = case.mesh_expectations.get("budget_zero_ratio", MESH_BUDGET_ZERO_RATIO)
    floor_ratio = case.mesh_expectations.get("budget_floor_ratio", MESH_BUDGET_FLOOR_RATIO)
    ratio = cand / baseline
    findings: list[Finding] = []
    # Lower floor: a mesh far below the baseline is too coarse to have resolved the
    # field. Deny the efficiency credit so a real solve cannot be paired with a
    # A trivial mesh must not receive the mesh-budget efficiency share; going
    # coarser is only "efficient" down to a point, below which it is under-resolved.
    if cand < floor_ratio * baseline:
        findings.append(
            Finding(
                "mesh_budget",
                "MESH_BUDGET_UNDERRESOLVED",
                "major",
                f"Candidate mesh has {cand} elements = {ratio * 100:.2f}% of the reference "
                f"baseline ({baseline}); below the {floor_ratio * 100:.0f}% floor a mesh "
                "is too coarse to have resolved the field, so the mesh-budget sub-score "
                "is 0 (a degenerate mesh cannot receive efficiency credit for a real "
                "solve).",
                penalty=0,
            )
        )
        return 0.0, findings
    score = metrics.mesh_budget_score(cand, baseline, zero_at)
    if score <= 0.0:
        findings.append(
            Finding(
                "mesh_budget",
                "MESH_BUDGET_EXCEEDED",
                "major",
                f"Candidate mesh has {cand} elements = {ratio * 100:.0f}% of the reference "
                f"baseline ({baseline}); at/over {zero_at * 100:.0f}% the mesh-budget "
                "sub-score is 0.",
                penalty=0,
            )
        )
    elif score < 100.0:
        findings.append(
            Finding(
                "mesh_budget",
                "MESH_BUDGET_OVER",
                "minor",
                f"Candidate mesh has {cand} elements = {ratio * 100:.0f}% of the reference "
                f"baseline ({baseline}); over budget -> mesh-budget sub-score "
                f"{score:.0f}/100.",
                penalty=0,
            )
        )
    return score, findings


def reproducibility_status(sub: Submission) -> str:
    arts = sub.artifacts or {}
    replay_accepted = bool(
        (sub.solver or {}).get("replay_accepted") or (sub.solver or {}).get("replay_verified")
    )
    has_setup = bool(arts.get("input_deck") or arts.get("script") or replay_accepted)
    has_result_file = bool(arts.get("result_file"))
    if has_setup and has_result_file:
        return "reproducible"
    if has_setup or has_result_file:
        return "partial"
    return "non_reproducible"
