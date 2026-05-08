"""Washer category — flat / spring / lock / square-neck washers.

A washer is an annulus (or a helical-cut annulus, for spring washers).
Its "width" is the RADIAL band between the inner and outer edge —
``washer_width = (outer_radius − inner_radius)``. The generic checker
tends to anchor `washer_width` on `Extrude.Length` (the axial
thickness), which is wrong by an order of magnitude.

Triggers when ``spec.name`` contains ``washer`` AND not ``square``
(``square_washer`` is already handled by FlangePlateCategory). Helix-
angle and pitch on spring washers stay unhandled here — they need
helix-feature analysis at the measurement layer.

Geometric anchors:
    washer_width            → outer_r − inner_r
    inner_diameter          → 2 × inner_r
    outer_diameter          → 2 × outer_r
    thickness               → aabb[min] (axial)
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec


def _tokens(key: str) -> FrozenSet[str]:
    return frozenset(key.split("_"))


def _is_washer_spec(spec: StructuredSpec) -> bool:
    """Trigger ONLY on annular washers — name contains ``washer`` AND
    none of {square, rectangular} (those are plate-shaped and handled
    by FlangePlateCategory). Spec keys like ``washer_width`` aren't
    enough to trigger because ``square_washer`` also uses them with a
    different geometric meaning (planar width, not radial band)."""
    name = (spec.name or "").lower()
    name_toks = set(name.split("_")) | set(name.split())
    if "washer" not in name_toks:
        return False
    if name_toks & {"square", "rectangular"}:
        return False
    return True


def _aabb_sorted(bank: MeasurementBank) -> Optional[Tuple[float, float, float]]:
    g = bank.globals.get("aabb_sorted")
    if g is None or not isinstance(g.value, tuple) or len(g.value) != 3:
        return None
    return tuple(sorted(float(x) for x in g.value))  # type: ignore[return-value]


def _outer_inner_radii(bank: MeasurementBank):
    """Return (outer, inner) cylinder clusters by radius. Doesn't trust
    `convex` flag because the bank's convex-detection on annular extrusions
    can flip depending on face-orientation heuristics."""
    if not bank.cylinder_clusters:
        return None, None
    by_r = sorted(bank.cylinder_clusters, key=lambda c: c.radius)
    return by_r[-1], by_r[0]


def _classify(key: str) -> Optional[str]:
    toks = _tokens(key)
    if "wall" in toks and "thickness" in toks:
        return "wall_thickness"
    if "thickness" in toks:
        return "thickness"
    if "width" in toks:
        return "washer_width"
    if "diameter" in toks and ({"inner", "inside", "id", "bore", "hole"} & toks):
        return "inner_diameter"
    if "diameter" in toks and ({"outer", "outside", "od"} & toks):
        return "outer_diameter"
    if "diameter" in toks:
        return "outer_diameter"
    return None


def derived_candidates(
    bank: MeasurementBank, spec: StructuredSpec,
) -> Dict[str, Tuple[float, str]]:
    if not _is_washer_spec(spec):
        return {}
    out: Dict[str, Tuple[float, str]] = {}
    aabb = _aabb_sorted(bank)
    outer, inner = _outer_inner_radii(bank)

    for source in (spec.scalars, spec.counts):
        for key, _ in source.items():
            kind = _classify(key)
            if kind is None:
                continue
            if kind == "washer_width" and outer is not None and inner is not None:
                out[key] = (
                    outer.radius - inner.radius,
                    f"washer.width({outer.id}.r − {inner.id}.r)",
                )
            elif kind == "thickness" and aabb is not None:
                out[key] = (aabb[0], "washer.aabb[min]")
            elif kind == "outer_diameter" and outer is not None:
                out[key] = (2 * outer.radius, f"washer.cylinder({outer.id}.radius × 2)")
            elif kind == "inner_diameter" and inner is not None:
                out[key] = (2 * inner.radius, f"washer.cylinder({inner.id}.radius × 2)")
    return out


# ---------------------------------------------------------------------------

from freecad_validator.consistency.categories.base import Category


class WasherCategory(Category):
    name = "washer"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)
