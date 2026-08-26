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
    if {"chamfer", "slot", "rounded", "head"} & toks:
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


def _spec_value(spec: StructuredSpec, key: str) -> float | None:
    """Return spec.scalars[key] (or counts[key]) as a float, or None."""
    for source in (spec.scalars, spec.counts):
        if key in source:
            try:
                return float(source[key])
            except (TypeError, ValueError):
                return None
    return None


def _head_thickness(spec: StructuredSpec) -> float | None:
    """Find a head_thickness-like spec value (head_thickness, head_height,
    head_length). Returns the value in mm or None."""
    for source in (spec.scalars, spec.counts):
        for key in source:
            toks = _tokens(key)
            if "head" in toks and toks & {"thickness", "height", "length"}:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    continue
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
    head_t = _head_thickness(spec)
    chamfer_axial = _chamfer_length_from_axial(bank)
    chamfer_sketch = _chamfer_length_from_sketch(bank)
    # Prefer axial-diff (more robust on coiled spring pins) when it
    # finds a value; fall back to sketch detection for solid pins where
    # the cylinder cluster has no axial step.
    chamfer_hit = chamfer_axial or chamfer_sketch
    rounded_hit = _rounded_end_height(bank)
    for source in (spec.scalars, spec.counts):
        for key, _ in source.items():
            kind = _classify(key)
            toks = _tokens(key)
            if kind == "chamfer_length" and chamfer_hit is not None:
                out[key] = chamfer_hit
            elif kind == "rounded_end_height" and rounded_hit is not None:
                out[key] = rounded_hit
            elif kind == "length" and aabb_max is not None:
                # Headed pin: spec's "pin_length" excludes the head — the bank
                # measures the full part (shaft + head) as aabb_max, so
                # subtract head_thickness when both are declared.
                if "pin" in toks and head_t is not None:
                    out[key] = (
                        aabb_max - head_t,
                        f"pin.aabb[max] − spec.head_thickness({head_t})",
                    )
                else:
                    out[key] = (aabb_max, "pin.aabb[max]")
            elif kind == "diameter" and main is not None:
                out[key] = (2 * main.radius, f"pin.cylinder({main.id}.radius × 2)")
    return out


# ---------------------------------------------------------------------------


class PinCategory(Category):
    name = "pin"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)
