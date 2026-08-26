"""`Category` abstract base class + shared reclassify helper.

One `Category` subclass per domain (gear, spline, keyway, hex, …).
Each subclass lives in `categories/<name>.py`. The base class provides
the reclassify mechanics so subclasses only need to implement
``derived_candidates``.
"""

from __future__ import annotations

import abc
import math
from typing import Any

from freecad_validator.consistency.compare import (
    as_display_angle,
    make_consistent_finding,
    make_inconsistent_finding,
    rel_err,
)
from freecad_validator.consistency.report import (
    ConsistencyReport,
    ParamFinding,
)
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec


class Category(abc.ABC):
    """Optional per-domain refinement applied after generic checks.

    A category's ``derived_candidates(bank, spec)`` returns
    ``{spec_key: (value, feature_ref)}`` for keys it knows how to
    derive. ``apply()`` then walks ``report.not_found`` and
    ``report.inconsistent`` and moves any matching finding into
    ``consistent`` / ``inconsistent`` based on whether the derived
    value agrees with the spec within ``tol_scalar``.

    Subclasses override ``derived_candidates`` only. Override
    ``apply()`` only if the default reclassify mechanics don't fit.
    """

    name: str = ""

    @abc.abstractmethod
    def derived_candidates(
        self,
        bank: MeasurementBank,
        spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]: ...

    def apply(
        self,
        report: ConsistencyReport,
        bank: MeasurementBank,
        spec: StructuredSpec,
        tol_scalar: float,
    ) -> None:
        derived = self.derived_candidates(bank, spec)
        if not derived:
            return
        _reclassify_against(report, derived, tol_scalar, self.name)


def _reclassify_against(
    report: ConsistencyReport,
    derived: dict[str, tuple[float, str]],
    tol_scalar: float,
    category_name: str,
) -> None:
    """Walk `not_found` and `inconsistent` findings; for each whose
    `param` key has a derived candidate, rebuild the finding with the
    derived value and move it to the right bucket."""

    def _reclass(
        findings: list[ParamFinding],
    ) -> tuple[list[ParamFinding], list[ParamFinding], list[ParamFinding]]:
        kept, moved_consistent, moved_inconsistent = [], [], []
        for f in findings:
            if f.param not in derived:
                kept.append(f)
                continue
            val, ref = derived[f.param]
            # Any param whose key ends in `_angle` (or the gear-tradition aliases
            # `pressure_angle` / `alpha`) is treated as an angle. Categories
            # store derived angles in radians; specs declare them in degrees.
            # The `_angle` suffix catches hex_head_angle / hex_nut_half_angle /
            # any future *_angle key without per-category opt-in.
            is_angle = f.param.endswith("_angle") or f.param == "alpha"
            # rel_err is unit-free, so comparison happens in whichever
            # unit both sides carry. For angles we stored deg on the
            # spec_value (display side) earlier, so convert back to rad
            # before comparing.
            spec_numeric = math.radians(float(f.spec_value)) if is_angle else float(f.spec_value)
            err = rel_err(spec_numeric, val)

            if is_angle:
                unit = "deg"
                spec_display: Any = f.spec_value  # already deg
                measured_display: Any = as_display_angle(val)
            elif f.param == "diametral_pitch":
                unit = "1/in"  # inch-system reciprocal
                spec_display = f.spec_value
                measured_display = round(val, 4)
            else:
                unit = f.unit or "mm"
                spec_display = f.spec_value
                measured_display = round(val, 6)

            if err <= tol_scalar:
                moved_consistent.append(
                    make_consistent_finding(
                        param=f.param,
                        spec_value=spec_display,
                        measured_value=measured_display,
                        unit=unit,
                        feature=ref,
                    )
                )
            else:
                moved_inconsistent.append(
                    make_inconsistent_finding(
                        param=f.param,
                        spec_value=spec_display,
                        measured_value=measured_display,
                        unit=unit,
                        feature=ref,
                        rel_diff=err,
                        reason=f"derived {category_name} value disagrees "
                        f"(rel_diff {err:.3f} > tol {tol_scalar})",
                    )
                )
        return kept, moved_consistent, moved_inconsistent

    kept_nf, to_cons_nf, to_inc_nf = _reclass(report.not_found)
    report.not_found = kept_nf
    report.consistent.extend(to_cons_nf)
    report.inconsistent.extend(to_inc_nf)

    kept_inc, to_cons_inc, still_inc = _reclass(report.inconsistent)
    report.inconsistent = kept_inc + still_inc
    report.consistent.extend(to_cons_inc)
