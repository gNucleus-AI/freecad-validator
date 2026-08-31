"""Pin category — straight cylindrical pins, dowel pins, coiled spring pins.

A pin is essentially a single elongated cylinder; dimensions worth
deriving are its overall length and shaft diameter. End chamfers and
slot details are feature-specific and require sketch / feature-tree
inspection — those are out of scope for this category.

Trigger: any spec key contains the token ``pin``.

Geometric anchors:
    length / pin_length          → max axis of the global aabb_sorted
                                   (the pin's long axis)
    diameter / pin_diameter      → 2 × radius of the principal convex
                                   cylinder cluster (largest by
                                   convex-radius × axial-extent)
"""

from __future__ import annotations

import math

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_pin_spec(spec: StructuredSpec) -> bool:
    """Trigger on the part name (most reliable — pin specs are usually
    named e.g. ``coiled_spring_pin_heavy_type``); fall back to a key
    token check for cases that mention pin in a key name."""
    if "pin" in (spec.name or "").lower().split("_"):
        return True
    for source in (spec.scalars, spec.counts):
        for key in source:
            if "pin" in _tokens(key):
                return True
    return False


def _classify(key: str) -> str | None:
    """'length' | 'diameter' | 'chamfer_length' | 'rounded_end_height' | None.

    `chamfer_length` is derived from axial-extent diffs; on tapered
    or rounded-end pins, the top/bottom rounding adds a small axial
    extent past the main cone — claim those as `rounded_end_height`.
    Other sub-feature qualifiers (slot_width, head_*, ...) are still
    skipped here — those describe end details and need feature-tree
    analysis."""
    toks = _tokens(key)
    if "chamfer" in toks and "length" in toks:
        return "chamfer_length"
    if "rounded" in toks and ({"end", "tip"} & toks) and "height" in toks:
        return "rounded_end_height"
    if "slot" in toks and "width" in toks:
        return "slot_width"
    if {"chamfer", "rounded", "head"} & toks:
        return None
    if "length" in toks:
        return "length"
    if "diameter" in toks:
        return "diameter"
    return None


def _rounded_end_height(bank: MeasurementBank) -> tuple[float, str] | None:
    """A taper / rounded-end pin's main body is a Cone surface that
    spans most of the part's axial extent; the small remainder
    (aabb_max − cone.axial_extent) splits between the two rounded
    end caps. Returns half that gap.
    """
    g = bank.globals.get("aabb_sorted")
    if g is None or not isinstance(g.value, tuple) or len(g.value) != 3:
        return None
    aabb_max = float(max(g.value))
    if not bank.conic_surfaces:
        return None
    largest_cone = max(bank.conic_surfaces, key=lambda cs: cs.axial_extent)
    gap = aabb_max - largest_cone.axial_extent
    if gap <= 0:
        return None
    return gap / 2.0, f"pin.(aabb[max] − {largest_cone.id}.axial_extent)/2"


def _chamfer_length_from_axial(bank: MeasurementBank) -> tuple[float, str] | None:
    """Derive chamfer_length from axial extents.

    A coiled or chamfered cylindrical pin has TWO distinct axial
    extents in the bank: the full pin length (aabb axial OR the
    un-chamfered inner cylinder's axial_extent) and the chamfered
    outer cylinder's axial_extent, which is shorter by 2× chamfer.
    Compare the aabb axis against each cyl_cluster axial_extent and
    take the largest positive difference / 2.

    Sketch line lengths are deliberately NOT used as candidates here:
    on solid pins (e.g. slotted_headless_cylindrical_pin) the
    chamfer is a separate small Groove sketch whose lines have no
    relation to the pin axial extent, and feeding them in poisons
    the diff. For solid pins use ``_chamfer_length_from_sketch``.
    """
    g = bank.globals.get("aabb_sorted")
    if g is None or not isinstance(g.value, tuple) or len(g.value) != 3:
        return None
    aabb_max = float(max(g.value))
    best: tuple[float, float, str] | None = None  # (chamfer, diff, ref)
    for c in bank.cylinder_clusters:
        ax = float(c.axial_extent)
        diff = aabb_max - ax
        if diff <= 1e-6:
            continue
        # Reasonableness gate: chamfer should be a small fraction of
        # the pin length. Anything more than 40% of the pin is almost
        # certainly the wrong cluster (e.g. a head height being read).
        if diff / aabb_max > 0.40:
            continue
        chamfer = diff / 2.0
        if best is None or diff < best[1]:
            # Prefer the cluster CLOSEST to the full length — that's
            # the one chamfered, not some unrelated short hub.
            best = (chamfer, diff, f"pin.(aabb[max] − {c.id}.axial_extent)/2")
    if best is None:
        return None
    return best[0], best[2]


def _chamfer_length_from_sketch(bank: MeasurementBank) -> tuple[float, str] | None:
    """Solid-pin chamfer detection. The chamfer profile is a revolved
    triangle whose sketch carries a leg length ``L`` (axial leg = the
    chamfer_length) and a hypotenuse ``H``, where ``H² ≈ L_axial² +
    L_radial²``. For symmetric 45° chamfers L_axial = L_radial = L
    so ``H ≈ √2·L`` — verifiable by ``H² ≈ 2·L²``.

    Filters out non-chamfer sketches by REQUIRING this Pythagorean
    signature: at least two line-length groups must satisfy
    ``H² ≈ 2·L²`` to within 5%. A plain rectangle slot (1.0 × 0.2)
    does not satisfy this and is rejected.
    """
    g = bank.globals.get("aabb_sorted")
    aabb_max = (
        float(max(g.value))
        if (g is not None and isinstance(g.value, tuple) and len(g.value) == 3)
        else None
    )
    for sp in bank.sketch_profiles:
        if not sp.line_lengths:
            continue
        # Distinct values (group near-equal).
        unique: list[tuple[float, int]] = []
        for ln in sorted(sp.line_lengths):
            if (
                not unique
                or abs(ln - unique[-1][0]) / max(abs(ln), abs(unique[-1][0]), 1e-9) > 1e-3
            ):
                unique.append((ln, 1))
            else:
                unique[-1] = (unique[-1][0], unique[-1][1] + 1)
        repeated = [(v, n) for (v, n) in unique if n >= 2]
        if len(repeated) < 2:
            continue
        # Find a (leg, hypotenuse) pair that satisfies the 45°-chamfer
        # signature: hypotenuse² ≈ 2·leg² (5% tolerance). The leg is
        # the smaller value.
        for i in range(len(repeated)):
            leg, _ = repeated[i]
            if leg <= 0:
                continue
            for j in range(i + 1, len(repeated)):
                hyp, _ = repeated[j]
                expected_sq = 2.0 * leg * leg
                actual_sq = hyp * hyp
                if abs(actual_sq - expected_sq) / max(expected_sq, 1e-12) > 0.05:
                    continue
                if aabb_max is not None and leg > 0.40 * aabb_max:
                    continue
                return leg, f"pin.{sp.name}.chamfer_leg({leg:.3f}, hyp={hyp:.3f})"
    return None


def _aabb_max(bank: MeasurementBank) -> float | None:
    g = bank.globals.get("aabb_sorted")
    if g is None or not isinstance(g.value, tuple) or len(g.value) != 3:
        return None
    return float(max(g.value))


def _principal_convex_cyl(bank: MeasurementBank):
    convex = [c for c in bank.cylinder_clusters if c.convex]
    if not convex:
        return None
    # The pin's main shaft is the convex cluster with the largest "presence"
    # = radius × axial_extent. End chamfers are smaller convex caps and lose.
    return max(convex, key=lambda c: c.radius * c.axial_extent)


def _declares_head_dimension(spec: StructuredSpec) -> bool:
    """Return whether the spec distinguishes a head from the shaft.

    The current bank has no reliable head-thickness measurement. In that
    case the overall AABB must not be adjusted with an expected spec value
    and presented as a CAD-derived shaft length.
    """
    for source in (spec.scalars, spec.counts):
        for key in source:
            toks = _tokens(key)
            if "head" in toks and toks & {"thickness", "height", "length"}:
                return True
    return False


def _cad_head_thickness(bank: MeasurementBank) -> tuple[float, str] | None:
    """Measure a revolved head from its sketch profile.

    The longest profile segment establishes the revolution-axis direction.
    Among parallel profile segments, the one farthest from that axis is the
    outer head edge; its axial projection is the head thickness.
    """
    profile_by_name = {profile.name: profile for profile in bank.sketch_profiles}
    for feature in bank.feature_tree:
        if "Revolution" not in feature.type_id:
            continue
        for dependency in feature.dependencies:
            profile = profile_by_name.get(dependency)
            if profile is None or len(profile.line_segments) < 3:
                continue
            axis = max(profile.line_segments, key=lambda segment: segment.length)
            axis_vec = tuple(axis.end[i] - axis.start[i] for i in range(3))
            axis_len = math.sqrt(sum(value * value for value in axis_vec))
            if axis_len <= 1e-9:
                continue
            direction = tuple(value / axis_len for value in axis_vec)
            parallel: list[tuple[float, float]] = []  # (distance, axial projection)
            for segment in profile.line_segments:
                if segment.index == axis.index:
                    continue
                delta = tuple(segment.end[i] - segment.start[i] for i in range(3))
                delta_len = math.sqrt(sum(value * value for value in delta))
                if delta_len <= 1e-9:
                    continue
                projection = abs(sum(delta[i] * direction[i] for i in range(3)))
                if projection / delta_len < 0.995:
                    continue
                midpoint = tuple((segment.start[i] + segment.end[i]) / 2 for i in range(3))
                from_axis = tuple(midpoint[i] - axis.start[i] for i in range(3))
                axial_position = sum(from_axis[i] * direction[i] for i in range(3))
                radial = tuple(from_axis[i] - axial_position * direction[i] for i in range(3))
                distance = math.sqrt(sum(value * value for value in radial))
                parallel.append((distance, projection))
            if len(parallel) < 2:
                continue
            parallel.sort(key=lambda item: item[0], reverse=True)
            head_distance, thickness = parallel[0]
            next_distance = parallel[1][0]
            if head_distance > 1.05 * next_distance and 1e-9 < thickness < 0.5 * axis_len:
                return thickness, f"{profile.name}.outer_head_edge"
    return None


def _slot_width(bank: MeasurementBank) -> tuple[float, str] | None:
    """Measure a rectangular slot from the sketch driving a Pocket."""
    profiles = {profile.name: profile for profile in bank.sketch_profiles}
    for feature in bank.feature_tree:
        if feature.type_id != "PartDesign::Pocket":
            continue
        for dependency in feature.dependencies:
            profile = profiles.get(dependency)
            if profile is None or len(profile.line_segments) != 4:
                continue
            lengths = sorted(segment.length for segment in profile.line_segments)
            if lengths[0] <= 0 or abs(lengths[0] - lengths[1]) / lengths[0] > 1e-3:
                continue
            if abs(lengths[2] - lengths[3]) / max(lengths[3], 1e-9) > 1e-3:
                continue
            return lengths[0], f"{profile.name}.rectangular_slot_short_side"
    return None


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    if not _is_pin_spec(spec):
        return {}
    out: dict[str, tuple[float, str]] = {}
    aabb_max = _aabb_max(bank)
    main = _principal_convex_cyl(bank)
    has_head_dimension = _declares_head_dimension(spec)
    head_thickness = _cad_head_thickness(bank) if has_head_dimension else None
    chamfer_axial = _chamfer_length_from_axial(bank)
    chamfer_sketch = _chamfer_length_from_sketch(bank)
    # Prefer axial-diff (more robust on coiled spring pins) when it
    # finds a value; fall back to sketch detection for solid pins where
    # the cylinder cluster has no axial step.
    chamfer_hit = chamfer_axial or chamfer_sketch
    rounded_hit = _rounded_end_height(bank)
    slot_hit = _slot_width(bank)
    for source in (spec.scalars, spec.counts):
        for key, _ in source.items():
            kind = _classify(key)
            toks = _tokens(key)
            if kind == "slot_width" and slot_hit is not None:
                out[key] = slot_hit
            elif kind == "chamfer_length" and chamfer_hit is not None:
                out[key] = chamfer_hit
            elif kind == "rounded_end_height" and rounded_hit is not None:
                out[key] = rounded_hit
            elif kind == "length" and aabb_max is not None:
                # For a headed pin, aabb_max is the overall part length while
                # pin_length commonly means shaft-only length. Subtract only
                # a head thickness recovered from the Revolution profile.
                if "pin" in toks and has_head_dimension:
                    if head_thickness is None:
                        continue
                    thickness, ref = head_thickness
                    if 0 < thickness < aabb_max:
                        out[key] = (aabb_max - thickness, f"pin.aabb[max] − {ref}")
                    continue
                out[key] = (aabb_max, "pin.aabb[max]")
            elif kind == "diameter" and main is not None:
                out[key] = (2 * main.radius, f"pin.cylinder({main.id}.radius × 2)")
    return out


# ---------------------------------------------------------------------------


class PinCategory(Category):
    name = "pin"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)
