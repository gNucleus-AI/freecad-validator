"""Pure numeric helpers used by the scorer.

No FEA dependencies: these are the maths the validators and the
reference-comparison module call. Side-effect free; depends only on the standard
library and the tolerance constants in ``freecad_validator.fem.schema``, so they
are trivially unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from freecad_validator.fem.schema import MESH_BUDGET_ZERO_RATIO, TOL_BAND_HI, TOL_BAND_LO


# --------------------------------------------------------------------------- #
# Error / tolerance scoring                                                   #
# --------------------------------------------------------------------------- #
def relative_error(claimed: float, reference: float) -> float:
    """|claimed - reference| / |reference| with a safe fallback for ref==0."""
    if reference == 0:
        return abs(claimed)  # absolute when reference is exactly zero
    return abs(claimed - reference) / abs(reference)


def tolerance_band_score(error: float, tol: float, fade: float = TOL_BAND_HI) -> float:
    """Map an error to a 0-100 score over the band [TOL_BAND_LO*tol, fade*tol].

    error <= TOL_BAND_LO*tol   -> 100  (inside the accepted band)
    error >= fade*tol          -> 0    (grossly wrong)
    linear interpolation in between.

    ``fade`` (default TOL_BAND_HI) controls how quickly the score decays once
    outside tolerance; a result a little outside is still partially credited, a
    result several times outside gets nothing.
    """
    if tol <= 0:
        tol = 1e-9
    lo = TOL_BAND_LO * tol
    hi = fade * tol
    if error <= lo:
        return 100.0
    if error >= hi:
        return 0.0
    return 100.0 * (hi - error) / (hi - lo)


def within_tolerance(
    claimed: float, reference: float, tol_rel: float, tol_abs: float | None = None
) -> tuple[bool, float]:
    """Return (within_tol, relative_error)."""
    err = relative_error(claimed, reference)
    ok = err <= tol_rel
    if tol_abs is not None:
        ok = ok or abs(claimed - reference) <= tol_abs
    return ok, err


# --------------------------------------------------------------------------- #
# Mesh quality scoring                                                        #
# --------------------------------------------------------------------------- #
# Acceptance bands collated from Ansys / DIANA mesh-quality documentation and
# FeaGPT's Jacobian-validity emphasis (see sources.json).
ASPECT_GOOD, ASPECT_WARN, ASPECT_BAD = 3.0, 10.0, 20.0
SKEW_GOOD, SKEW_WARN, SKEW_BAD = 0.5, 0.8, 0.95
JAC_IDEAL, JAC_MIN = 1.0, 0.5


def _band_penalty(value: float, good: float, bad: float, max_pen: float) -> float:
    """Linear penalty: 0 at/under ``good``, ``max_pen`` at/above ``bad``."""
    if value <= good:
        return 0.0
    if value >= bad:
        return max_pen
    return max_pen * (value - good) / (bad - good)


def mesh_quality_score(quality: dict[str, float]) -> tuple[float, list[str]]:
    """Score element-quality statistics (0-100) and return human notes.

    Recognised keys (all optional): ``max_aspect_ratio``, ``max_skewness``,
    ``min_jacobian``, ``pct_elements_below_jac``, ``pct_high_aspect``.
    A missing key is simply not penalised (but the caller flags absence).
    """
    score = 100.0
    notes: list[str] = []

    ar = quality.get("max_aspect_ratio")
    if ar is not None:
        pen = _band_penalty(ar, ASPECT_GOOD, ASPECT_BAD, 35.0)
        score -= pen
        if ar > ASPECT_WARN:
            notes.append(f"max aspect ratio {ar:g} exceeds {ASPECT_WARN:g}")

    sk = quality.get("max_skewness")
    if sk is not None:
        pen = _band_penalty(sk, SKEW_GOOD, SKEW_BAD, 35.0)
        score -= pen
        if sk > SKEW_WARN:
            notes.append(f"max skewness {sk:g} exceeds {SKEW_WARN:g}")

    jac = quality.get("min_jacobian")
    if jac is not None and jac < JAC_MIN:
        # negative handled as inverted element by the validator; here penalise low
        pen = _band_penalty(JAC_MIN - max(jac, 0.0), 0.0, JAC_MIN, 30.0)
        score -= pen
        notes.append(f"min Jacobian {jac:g} below recommended {JAC_MIN:g}")

    pct_jac = quality.get("pct_elements_below_jac")
    if pct_jac:
        score -= min(30.0, pct_jac * 2.0)  # 1% bad -> 2 pts, capped
        if pct_jac > 1.0:
            notes.append(f"{pct_jac:g}% of elements below Jacobian threshold")

    pct_ar = quality.get("pct_high_aspect")
    if pct_ar:
        score -= min(20.0, pct_ar * 1.0)

    return max(0.0, score), notes


def mesh_budget_score(
    candidate_elements: float, baseline_elements: float, zero_at: float = MESH_BUDGET_ZERO_RATIO
) -> float:
    """Mesh-efficiency score against a baseline element count from the reference.

    full marks (100) at or below the baseline; linearly decaying to 0 once the
    candidate reaches ``zero_at`` x the baseline (default 130%). Using fewer
    elements than the baseline is never penalised.

        r = candidate / baseline
        r <= 1.0       -> 100
        1.0 < r < z    -> 100 * (z - r) / (z - 1)
        r >= z         -> 0
    """
    if not baseline_elements or baseline_elements <= 0:
        return 100.0
    r = candidate_elements / baseline_elements
    if r <= 1.0:
        return 100.0
    if r >= zero_at:
        return 0.0
    return 100.0 * (zero_at - r) / (zero_at - 1.0)


# --------------------------------------------------------------------------- #
# Mesh convergence (Grid Convergence Index, Roache 1994 / Celik et al. 2008)  #
# --------------------------------------------------------------------------- #
def convergence_from_study(
    study: Sequence[dict[str, float]], rel_tol: float = 0.03
) -> dict[str, Any]:
    """Analyse a mesh-convergence study.

    ``study`` is a list of {"n_elements": N, "value": Q} ordered coarse->fine
    (or any order; we sort by N). Returns a dict with:

      converged        - last successive relative change <= rel_tol
      last_rel_change   - |Q_fine - Q_med| / |Q_fine|
      monotonic         - values change monotonically (asymptotic-like)
      diverging         - |change| grows with refinement (singularity signature)
      gci_fine          - GCI of the finest grid (%), if >=3 grids and feasible
      apparent_order    - observed order p, if computable
      n_grids           - number of grids
    """
    pts = sorted(
        [p for p in study if "value" in p and "n_elements" in p], key=lambda p: p["n_elements"]
    )
    out: dict[str, Any] = {
        "n_grids": len(pts),
        "converged": False,
        "last_rel_change": None,
        "monotonic": None,
        "diverging": False,
        "gci_fine": None,
        "apparent_order": None,
    }
    if len(pts) < 2:
        return out

    vals = [float(p["value"]) for p in pts]
    changes = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    last = abs(vals[-1] - vals[-2]) / (abs(vals[-1]) if vals[-1] else 1.0)
    out["last_rel_change"] = last
    out["converged"] = last <= rel_tol

    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    out["monotonic"] = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
    # diverging: successive changes are NOT shrinking (stress-singularity signature)
    if len(changes) >= 2:
        out["diverging"] = changes[-1] >= changes[-2] * 0.95 and last > rel_tol

    # GCI on the three finest grids with (assumed) constant refinement ratio.
    if len(pts) >= 3:
        f1, f2, f3 = vals[-1], vals[-2], vals[-3]  # fine, medium, coarse
        n1, n2, n3 = pts[-1]["n_elements"], pts[-2]["n_elements"], pts[-3]["n_elements"]
        # representative cell-size ratios in 3D: h ~ N^(-1/3)
        r21 = (n1 / n2) ** (1.0 / 3.0)
        r32 = (n2 / n3) ** (1.0 / 3.0)
        e21, e32 = (f1 - f2), (f2 - f3)
        if e21 != 0 and e32 != 0 and r21 > 1.0 and r32 > 1.0:
            ratio = e32 / e21
            if ratio > 0:  # monotone convergence required for a clean order estimate
                try:
                    p = abs(math.log(abs(ratio))) / math.log(r21)
                    p = max(0.3, min(p, 6.0))
                    out["apparent_order"] = p
                    ext = f1 + (f1 - f2) / (r21**p - 1.0)
                    ea = abs((f1 - f2) / f1) if f1 else abs(f1 - f2)
                    gci = 1.25 * ea / (r21**p - 1.0)
                    out["gci_fine"] = 100.0 * gci
                    out["extrapolated"] = ext
                except (ValueError, ZeroDivisionError):
                    pass
    return out


# --------------------------------------------------------------------------- #
# Internal consistency (hallucination signal)                                 #
# --------------------------------------------------------------------------- #
def values_agree(values: Sequence[float], rel_tol: float = 0.02) -> bool:
    """True if all numbers agree within rel_tol of their mean magnitude."""
    nums = [v for v in values if isinstance(v, (int, float))]
    if len(nums) < 2:
        return True
    lo, hi = min(nums), max(nums)
    scale = max(abs(lo), abs(hi), 1e-12)
    return (hi - lo) / scale <= rel_tol


def max_pairwise_discrepancy(values: Sequence[float]) -> float:
    nums = [v for v in values if isinstance(v, (int, float))]
    if len(nums) < 2:
        return 0.0
    lo, hi = min(nums), max(nums)
    scale = max(abs(lo), abs(hi), 1e-12)
    return (hi - lo) / scale


# --------------------------------------------------------------------------- #
# Unit consistency                                                            #
# --------------------------------------------------------------------------- #
def stress_unit_consistent(units: dict[str, str]) -> bool | None:
    """Cheap dimensional sanity for the common N/mm/MPa and SI systems.

    Returns True/False if it can decide, None if there isn't enough info.
    MPa == N/mm^2 ; Pa == N/m^2. A mix like force=N, length=mm, stress=Pa is
    inconsistent (would be off by 1e6).
    """
    force = (units.get("force") or "").lower()
    length = (units.get("length") or "").lower()
    stress = (units.get("stress") or "").lower()
    if not (force and length and stress):
        return None
    if force in ("n", "newton") and length in ("mm", "millimeter"):
        return stress in ("mpa", "n/mm^2", "n/mm2", "megapascal")
    if force in ("n", "newton") and length in ("m", "meter"):
        return stress in ("pa", "n/m^2", "n/m2", "pascal")
    return None
