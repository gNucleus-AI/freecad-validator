"""Spline (involute) category helpers.

Splines overlap with gears in ISO relationships but the spec's key
naming is far less uniform. Across the reference corpus's
``shaft_with_spline`` cases we see variants like:

    section_2_spline_module        section_2_number_teeth
    section_2_spline_pitch_diameter section_2_pressure_angle
    spline_module                  spline_number_teeth
    spline_pitch_diameter          spline_pressure_angle
    shaft_spline_module            shaft_number_teeth
    shaft_pressure_angle

Rather than enumerating prefix regexes, this module uses **token-based
matching**: keys are split on ``_``, and canonical params are claimed
whenever the right tokens appear. A spec is treated as a spline spec
if *any* of its keys contains ``spline`` as a token. Once classified
that way, peer keys that don't repeat ``spline`` (like
``section_2_number_teeth`` or ``shaft_pressure_angle``) are still
eligible — but keys containing ``gear`` are ceded to the gear
category to avoid double-claiming.

Numeric refinement uses a spline-specific tooth ring only when the bank
shows two equal-count coaxial radius groups.  The radial tooth height is
``1.25 × module`` for the straight-sided external-spline construction
used here, so module and pitch diameter are derived from CAD rather than
from the expected spec values. Pressure angle remains case-specific when
the sketch contains no measurable flanks.

Default pressure angle: 30° (most common on splines; used only as a
fallback when the spec doesn't declare its own).
"""

from __future__ import annotations

import math

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# 30° is the most common pressure angle on splines. Only a *fallback*
# when the spec doesn't declare pressure_angle — splines with 37.5°,
# 45°, or other angles exist and should be checked against their own
# declared value when provided.
DEFAULT_PRESSURE_ANGLE_RAD = math.radians(30.0)


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_spline_spec(spec: StructuredSpec) -> bool:
    """Require an explicit spline key before refining peer parameters."""
    return any("spline" in _tokens(key) for source in (spec.scalars, spec.counts) for key in source)


def derive_params(
    module: float,
    teeth: int,
    pressure_angle_rad: float = DEFAULT_PRESSURE_ANGLE_RAD,
) -> dict[str, float]:
    """Apply standard involute-spline relations. Pure math.

    Textbook external-spline proportions (addendum=m, dedendum=1.25·m)
    are used as *fallbacks* when major/minor diameters aren't declared
    by the spec — non-standard splines (short-addendum, long-addendum,
    flat-root variants) legitimately differ, so any mismatch between
    these fallbacks and the spec's declared majors/minors will surface
    as `inconsistent` rather than silently passing.
    """
    m = float(module)
    z = int(teeth)
    alpha = float(pressure_angle_rad)

    pitch_diameter = m * z
    addendum = m
    dedendum = 1.25 * m
    major_diameter = pitch_diameter + 2 * addendum
    minor_diameter = pitch_diameter - 2 * dedendum
    base_diameter = pitch_diameter * math.cos(alpha)
    circular_pitch = math.pi * m

    return {
        "module": m,
        "teeth": float(z),
        "pitch_diameter": pitch_diameter,
        "pressure_angle": alpha,
        "major_diameter": major_diameter,
        "minor_diameter": minor_diameter,
        "base_diameter": base_diameter,
        "circular_pitch": circular_pitch,
        "addendum": addendum,
        "dedendum": dedendum,
    }


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Derive module and pitch diameter from an equal-count tooth ring."""
    if not _is_spline_spec(spec):
        return {}

    tooth_groups: dict[int, list[float]] = {}
    for cluster in bank.cylinder_clusters:
        if cluster.count < 6:
            continue
        tooth_groups.setdefault(cluster.count, []).append(float(cluster.radius))
    candidates = [
        (count, sorted(set(round(radius, 6) for radius in radii)))
        for count, radii in tooth_groups.items()
        if len(set(round(radius, 6) for radius in radii)) >= 2
    ]
    if not candidates:
        return {}
    teeth, radii = min(candidates, key=lambda item: item[0])
    root_radius, outer_radius = radii[0], radii[-1]
    module = (outer_radius - root_radius) / 1.25
    if module <= 0:
        return {}

    out: dict[str, tuple[float, str]] = {}
    ref = (
        f"spline.tooth_ring(root_r={root_radius:.3f}, outer_r={outer_radius:.3f}, "
        f"z={teeth}, module=(outer−root)/1.25)"
    )
    for source in (spec.scalars, spec.counts):
        for key in source:
            tokens = _tokens(key)
            # A mixed specification can contain both a spline and a gear.
            # Gear dimensions use a different whole-depth relation, so they
            # must be left to GearCategory even when the spline trigger is on.
            if "gear" in tokens:
                continue
            if "module" in tokens:
                out[key] = (module, ref)
            elif "pitch" in tokens and "diameter" in tokens:
                out[key] = (module * teeth, ref)
            elif "tooth" in tokens and "width" in tokens:
                repeated_offsets: dict[float, int] = {}
                for pair in bank.plane_pairs:
                    offset = round(float(pair.offset), 6)
                    repeated_offsets[offset] = repeated_offsets.get(offset, 0) + 1
                candidates = [
                    value for value, count in repeated_offsets.items() if count >= 2 and value > 0
                ]
                if len(candidates) == 1:
                    out[key] = (candidates[0], "spline.repeated_tooth_flank_plane_offset")
    return out


# ---------------------------------------------------------------------------
# Category subclass.
# Numeric spline refinement is intentionally disabled until the bank
# exposes spline-specific CAD measurements.
# ---------------------------------------------------------------------------


class SplineCategory(Category):
    name = "spline"

    def derived_candidates(
        self,
        bank: MeasurementBank,
        spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)
