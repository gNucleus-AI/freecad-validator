"""E. Accuracy vs reference.

Compares candidate quantities against the case's reference solution.
The reference may be analytical (closed form), numerical (high-fidelity run),
empirical, or expert-derived; the ``method`` is carried through into every
comparison record so the report can state the provenance of each tolerance.

Two reference styles are supported:

* point values  - ``reference.quantities[name] = {value, unit, tol_rel, critical}``
* ranges        - ``reference.ranges[name]    = {min, max, unit, critical}``
  (used by real-world cases where only a plausible band is known)

Returns ``(score, findings, comparisons)``. ``score`` is ``None`` when the case
declares no reference at all, signalling the orchestrator to redistribute the
accuracy weight across the other categories.
"""

from __future__ import annotations

from typing import Any

from freecad_validator.fem import metrics
from freecad_validator.fem.schema import GROSS_TOL, CaseDefinition, FailureMode, Finding, Submission

# Common quantity aliases allow minor field-naming differences while preserving
# still matched to the reference quantity.
_ALIASES = {
    "max_displacement_mm": ["max_disp_mm", "tip_deflection_mm", "max_deflection_mm", "umax_mm"],
    "max_von_mises_MPa": ["max_stress_MPa", "peak_von_mises_MPa", "sigma_vm_max_MPa"],
    "max_hoop_stress_MPa": ["sigma_theta_max_MPa", "hoop_stress_MPa", "circumferential_stress_MPa"],
    "max_principal_stress_MPa": ["s1_max_MPa", "max_principal_MPa"],
    "first_natural_frequency_Hz": ["f1_Hz", "fundamental_frequency_Hz", "natural_frequency_Hz"],
    "buckling_factor": ["load_factor", "buckling_load_factor", "critical_load_factor"],
    "max_temperature_C": ["Tmax_C", "peak_temperature_C"],
    "reaction_force_N": ["total_reaction_N"],
}


def _lookup(results: dict[str, Any], name: str) -> float | None:
    if name in results and isinstance(results[name], (int, float)):
        return float(results[name])
    for alt in _ALIASES.get(name, []):
        if alt in results and isinstance(results[alt], (int, float)):
            return float(results[alt])
    # special case: first natural frequency from a list
    if name == "first_natural_frequency_Hz":
        fl = results.get("natural_frequencies_Hz")
        if isinstance(fl, list) and fl and isinstance(fl[0], (int, float)):
            return float(fl[0])
    return None


def compare_to_reference(
    case: CaseDefinition, sub: Submission
) -> tuple[float | None, list[Finding], list[dict[str, Any]]]:
    findings: list[Finding] = []
    comparisons: list[dict[str, Any]] = []
    method = case.reference_method()

    quantities = case.reference_quantities()
    ranges = case.reference.get("ranges", {})
    if not quantities and not ranges:
        return None, findings, comparisons

    weighted_scores: list[tuple[float, float]] = []  # (score, weight)
    results = sub.results or {}
    gross_tol = case.reference.get("gross_tol", GROSS_TOL)  # critical miss beyond this gates to 0

    # point-value comparisons
    for name, rq in quantities.items():
        claimed = _lookup(results, name)
        weight = 2.0 if rq.critical else 1.0
        if claimed is None:
            if rq.critical:
                # Omitting a CRITICAL quantity is a gross failure to reproduce the
                # reference - at least as bad as reporting it grossly wrong - so it must
                # raise the same gating finding, not a non-gating MISSING_QUANTITY.
                # Otherwise omitting the hardest datum could outscore reporting it incorrectly.
                findings.append(
                    Finding(
                        "accuracy_vs_reference",
                        FailureMode.ACCURACY_GROSS_ERROR,
                        "critical",
                        f"Critical reference quantity '{name}' ({rq.value:g} {rq.unit}) "
                        "was not reported: gross failure to reproduce the critical "
                        "quantity.",
                        penalty=0,
                    )
                )
            else:
                findings.append(
                    Finding(
                        "accuracy_vs_reference",
                        "MISSING_QUANTITY",
                        "minor",
                        f"Reference quantity '{name}' ({rq.value:g} {rq.unit}) was not reported.",
                        penalty=0,
                    )
                )
            weighted_scores.append((0.0, weight))
            comparisons.append(
                {
                    "quantity": name,
                    "claimed": None,
                    "reference": rq.value,
                    "unit": rq.unit,
                    "rel_error": None,
                    "tol": rq.tol_rel,
                    "within_tol": False,
                    "critical": rq.critical,
                    "method": method,
                }
            )
            continue
        ok, err = metrics.within_tolerance(claimed, rq.value, rq.tol_rel, rq.tol_abs)
        qscore = metrics.tolerance_band_score(err, rq.tol_rel)
        weighted_scores.append((qscore, weight))
        comparisons.append(
            {
                "quantity": name,
                "claimed": claimed,
                "reference": rq.value,
                "unit": rq.unit,
                "rel_error": err,
                "tol": rq.tol_rel,
                "within_tol": ok,
                "critical": rq.critical,
                "method": method,
            }
        )
        if rq.critical and err > gross_tol:
            findings.append(
                Finding(
                    "accuracy_vs_reference",
                    FailureMode.ACCURACY_GROSS_ERROR,
                    "critical",
                    f"'{name}' = {claimed:g} {rq.unit} is {err * 100:.0f}% from the reference "
                    f"{rq.value:g} {rq.unit} (> gross_tol {gross_tol * 100:.0f}%): gross "
                    "failure to reproduce the critical quantity.",
                    penalty=0,
                )
            )

    # range comparisons
    for name, spec in ranges.items():
        claimed = _lookup(results, name)
        critical = bool(spec.get("critical"))
        weight = 2.0 if critical else 1.0
        lo, hi = spec.get("min"), spec.get("max")
        unit = spec.get("unit", "")
        if claimed is None:
            if critical:
                findings.append(
                    Finding(
                        "accuracy_vs_reference",
                        FailureMode.ACCURACY_GROSS_ERROR,
                        "critical",
                        f"Critical reference quantity '{name}' (range [{lo}, {hi}] "
                        f"{unit}) was not reported: gross failure to reproduce it.",
                        penalty=0,
                    )
                )
            weighted_scores.append((0.0, weight))
            comparisons.append(
                {
                    "quantity": name,
                    "claimed": None,
                    "reference": f"[{lo}, {hi}]",
                    "unit": unit,
                    "rel_error": None,
                    "tol": "range",
                    "within_tol": False,
                    "critical": critical,
                    "method": method,
                }
            )
            continue
        if lo is not None and hi is not None:
            if lo <= claimed <= hi:
                qscore = 100.0
                ok = True
            else:
                width = max(hi - lo, 1e-9)
                dist = (lo - claimed) if claimed < lo else (claimed - hi)
                qscore = metrics.tolerance_band_score(dist / width, 0.0 + 1e-9, fade=2.0)
                ok = False
            weighted_scores.append((qscore, weight))
            comparisons.append(
                {
                    "quantity": name,
                    "claimed": claimed,
                    "reference": f"[{lo}, {hi}]",
                    "unit": unit,
                    "rel_error": None,
                    "tol": "range",
                    "within_tol": ok,
                    "critical": critical,
                    "method": method,
                }
            )

    if not weighted_scores:
        return None, findings, comparisons

    total_w = sum(w for _, w in weighted_scores)
    score = sum(s * w for s, w in weighted_scores) / total_w if total_w else 0.0

    # propagate the accuracy shortfall into the category score as a penalty record
    if score < 100:
        worst = [c for c in comparisons if not c["within_tol"]]
        if worst:
            findings.append(
                Finding(
                    "accuracy_vs_reference",
                    "OUT_OF_TOLERANCE",
                    "major" if score < 60 else "minor",
                    f"{len(worst)} reference quantity(ies) outside tolerance.",
                    penalty=0,
                )
            )
    return score, findings, comparisons
