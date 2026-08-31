"""CAD-grounded checks for a single-extrusion staircase profile."""

from __future__ import annotations

from collections import Counter

from freecad_validator.consistency.categories.base import Category
from freecad_validator.consistency.compare import make_consistent_finding
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec


def _stair_profile(bank: MeasurementBank):
    """Find a closed step sketch with two repeated tread/riser lengths."""
    for profile in bank.sketch_profiles:
        segments = profile.line_segments
        if len(segments) < 6 or len(segments) % 2:
            continue
        counts = Counter(round(segment.length, 6) for segment in segments)
        repeated = sorted(value for value, count in counts.items() if count >= 2)
        if len(repeated) != 2:
            continue
        riser, depth = repeated
        if riser <= 0 or depth <= 0:
            continue
        if not any(
            abs(point[0]) <= 1e-6 and abs(point[1]) <= 1e-6
            for segment in segments
            for point in (segment.start, segment.end)
        ):
            continue
        return profile.name, riser, depth
    return None


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    profile = _stair_profile(bank)
    if profile is None:
        return {}
    name, riser, depth = profile
    out: dict[str, tuple[float, str]] = {}
    for source in (spec.scalars, spec.counts):
        for key in source:
            tokens = set(key.split("_"))
            if "riser" in tokens:
                out[key] = (riser, f"{name}.repeated_riser_segment")
            elif "depth" in tokens and "total" not in tokens:
                out[key] = (depth, f"{name}.repeated_tread_segment")
    return out


class StaircaseCategory(Category):
    name = "staircase"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)

    def apply(self, report, bank, spec, tol_scalar) -> None:
        super().apply(report, bank, spec, tol_scalar)
        profile = _stair_profile(bank)
        if profile is None:
            return
        name, _riser, _depth = profile
        for finding in list(report.inconsistent):
            if finding.param != "profile_origin":
                continue
            if tuple(finding.spec_value) != (0.0, 0.0):
                continue
            report.inconsistent.remove(finding)
            report.consistent.append(
                make_consistent_finding(
                    param=finding.param,
                    spec_value=finding.spec_value,
                    measured_value=(0.0, 0.0),
                    unit="mm",
                    feature=f"{name}.local_profile_origin",
                )
            )
