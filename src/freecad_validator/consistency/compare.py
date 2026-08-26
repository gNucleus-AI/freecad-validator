"""Comparison and ParamFinding-construction primitives shared by both
`checks/` (ParamCheck ABC tree) and `categories/` (Category ABC tree).

These are the pieces neither axis owns but both need. No inheritance
coupling between the two trees — they simply import from here.
"""

from __future__ import annotations

import math
from typing import Any

from freecad_validator.consistency.report import ParamFinding
from freecad_validator.measurement.schema import MeasurementBank

# A candidate is (measured_value, feature_ref_str). Both scalar and
# vector checks use this shape; the "value" slot is weakly-typed
# because vectors carry tuples.
Candidate = tuple[Any, str]


def rel_err(a: float, b: float, eps: float = 1e-9) -> float:
    """Relative error between two floats, safe against tiny denominators."""
    return abs(a - b) / max(abs(a), abs(b), eps)


def closest_scalar(
    target: float,
    candidates: list[Candidate],
) -> tuple[float, float, str] | None:
    """Return (measured_value, rel_err, feature_ref) of the closest
    candidate by rel_err. Returns None if candidates is empty."""
    if not candidates:
        return None
    best: tuple[float, float, str] | None = None
    for val, feat in candidates:
        err = rel_err(float(target), float(val))
        if best is None or err < best[1]:
            best = (float(val), err, feat)
    return best


def as_display_angle(value_rad: float) -> float:
    """Convert an internal radian value to a display-friendly
    degree number, rounded for readability. Angles are compared
    numerically in radians (rel_err is unit-free), but reports
    present them in degrees to match how specs normally declare them."""
    return round(math.degrees(float(value_rad)), 4)


def obb_diagonal(bank: MeasurementBank) -> float:
    """Diagonal length of the OBB (for scaling position tolerances)."""
    obb = bank.globals.get("obb_sorted")
    if obb is None or not isinstance(obb.value, tuple):
        return 1.0
    return math.sqrt(sum(x * x for x in obb.value))


# ---------------------------------------------------------------------------
# ParamFinding constructors — keep finding-shape details in one place so
# both checks/ and categories/ produce consistent outputs.
# ---------------------------------------------------------------------------


def make_consistent_finding(
    *,
    param: str,
    spec_value: Any,
    measured_value: Any,
    unit: str,
    feature: str | None,
) -> ParamFinding:
    return ParamFinding(
        param=param,
        spec_value=spec_value,
        measured_value=measured_value,
        unit=unit,
        feature=feature,
    )


def make_inconsistent_finding(
    *,
    param: str,
    spec_value: Any,
    measured_value: Any,
    unit: str,
    feature: str | None,
    rel_diff: float,
    reason: str,
) -> ParamFinding:
    return ParamFinding(
        param=param,
        spec_value=spec_value,
        measured_value=measured_value,
        unit=unit,
        feature=feature,
        rel_diff=rel_diff,
        reason=reason,
    )


def make_not_found_finding(
    *,
    param: str,
    spec_value: Any,
    unit: str,
    reason: str,
) -> ParamFinding:
    return ParamFinding(
        param=param,
        spec_value=spec_value,
        unit=unit,
        reason=reason,
    )
