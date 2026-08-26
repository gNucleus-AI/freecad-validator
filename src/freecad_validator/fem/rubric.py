"""Deterministic engineering-reporting quality checks.

The reporting score is a deterministic rubric: did the submission do the things
a competent CAE engineer's report does (state assumptions and units, justify the
mesh, interpret the results, identify limitations, compute a safety factor when
relevant, show convergence evidence)? Cases may override the rubric items and
weights via ``case.rubric``.

"""

from __future__ import annotations

from freecad_validator.fem.schema import CaseDefinition, Finding, Submission

# default rubric: item -> (weight, human label)
DEFAULT_RUBRIC = {
    "assumptions_stated": (0.18, "assumptions stated"),
    "units_stated": (0.12, "units declared"),
    "results_interpreted": (0.20, "results interpreted in engineering terms"),
    "limitations_identified": (0.15, "limitations identified"),
    "convergence_evidence": (0.15, "mesh-convergence evidence shown"),
    "mesh_justification": (0.10, "mesh choice justified"),
    "safety_factor": (0.10, "safety factor computed where relevant"),
}


def _present(case: CaseDefinition, sub: Submission, item: str) -> bool:
    rep = sub.report or {}
    if item == "assumptions_stated":
        a = rep.get("assumptions")
        return bool(a) and (len(a) > 0 if isinstance(a, list) else len(str(a).strip()) > 0)
    if item == "units_stated":
        return bool(sub.units)
    if item == "results_interpreted":
        return len(str(rep.get("interpretation", "")).strip()) >= 40
    if item == "limitations_identified":
        lim = rep.get("limitations")
        return bool(lim) and (len(lim) > 0 if isinstance(lim, list) else len(str(lim).strip()) > 0)
    if item == "convergence_evidence":
        study = (sub.mesh or {}).get("convergence_study") or []
        return len(study) >= 2 or "converg" in str(rep.get("mesh_justification", "")).lower()
    if item == "mesh_justification":
        return len(str(rep.get("mesh_justification", "")).strip()) >= 20
    if item == "safety_factor":
        if not case.rubric.get("requires_safety_factor", False):
            return True  # not required -> counts as satisfied
        has_sf = sub.results.get("safety_factor") is not None
        return has_sf or len(str(rep.get("safety_factor_discussion", "")).strip()) > 0
    return False


def evaluate_reporting(case: CaseDefinition, sub: Submission) -> tuple[float, list[Finding]]:
    findings: list[Finding] = []
    rubric = dict(DEFAULT_RUBRIC)
    # allow the case to override weights / add items
    for item, w in (case.rubric.get("weights", {}) or {}).items():
        label = rubric[item][1] if item in rubric else item.replace("_", " ")
        rubric[item] = (w, label)

    total_w = sum(w for w, _ in rubric.values()) or 1.0
    score = 0.0
    missing: list[str] = []
    for item, (w, label) in rubric.items():
        if _present(case, sub, item):
            score += w
        else:
            missing.append(label)
    score = 100.0 * score / total_w

    # bonus: sensitivity analysis mentioned (does not exceed 100)
    if str((sub.report or {}).get("sensitivity", "")).strip():
        score = min(100.0, score + 5.0)

    if missing:
        sev = "major" if score < 50 else "minor"
        findings.append(
            Finding(
                "engineering_reporting",
                "INCOMPLETE_REPORT",
                sev,
                "Reporting gaps: " + ", ".join(missing) + ".",
                penalty=0,
            )
        )
    return score, findings
