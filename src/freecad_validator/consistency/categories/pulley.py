"""Pulley category — V-belt and timing-belt pulleys.

A pulley sits axially: the disc face spans X/Y, the axial extent is Z.
The bank gives us:

  * the outer envelope (largest convex cyl-cluster ≈ tip-circle of the
    pulley body, OR the two largest aabb axes when the geometry is so
    toothed that no single convex cylinder captures it)
  * the central bore — the smallest concave cylinder cluster
  * the axial thickness (face_width / pulley_width) — aabb's smallest axis

Tooth-profile params (tooth_height, tooth_bottom_width, groove_angle,
groove_depth, top_width, etc.) genuinely need 2D-sketch analysis of the
tooth profile and stay unhandled here.

Trigger: any spec key contains the token ``pulley``.

Geometric anchors:
    outer_diameter              → 2 × largest convex cylinder radius;
                                  fallback to max(aabb_xy)
    bore_diameter               → 2 × smallest concave cylinder radius
                                  (the central shaft hole)
    width / face_width / thickness
                                → aabb's smallest axis
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec


def _tokens(key: str) -> FrozenSet[str]:
    return frozenset(key.split("_"))


def _is_pulley_spec(spec: StructuredSpec) -> bool:
    """Trigger on part name (most reliable) or description; pulley specs
    rarely put ``pulley`` in their key names."""
    name_toks = (spec.name or "").lower().split("_")
    if "pulley" in name_toks:
        return True
    if "pulley" in (spec.description or "").lower():
        return True
    for source in (spec.scalars, spec.counts):
        for key in source:
            if "pulley" in _tokens(key):
                return True
    return False


def _classify(key: str) -> Optional[str]:
    toks = _tokens(key)
    # Skip tooth-profile and groove-profile sub-dimensions — those need
    # 2D-sketch analysis and would be wildly mis-anchored if we tried
    # to claim them from aabb / cylinder data.
    if {"tooth", "groove"} & toks:
        return None
    if "diameter" in toks and ({"bore", "shaft", "hole", "inner", "id"} & toks):
        return "bore_diameter"
    if "diameter" in toks and ({"outer", "outside", "od"} & toks):
        return "outer_diameter"
    if "diameter" in toks:
        return "outer_diameter"
    # Only claim bare 'width'/'face_width'/'pulley_width' — never 'tooth_width'
    # or 'top_width' which are tooth-profile sub-dims.
    if (toks == {"width"}
        or toks == {"face", "width"}
        or toks == {"pulley", "width"}
        or toks == {"thickness"}
        or toks == {"face", "thickness"}):
        return "width"
    return None


def _aabb_sorted(bank: MeasurementBank) -> Optional[Tuple[float, float, float]]:
    g = bank.globals.get("aabb_sorted")
    if g is None or not isinstance(g.value, tuple) or len(g.value) != 3:
        return None
    return tuple(sorted(float(x) for x in g.value))  # type: ignore[return-value]


def _outer_inner_radii(bank: MeasurementBank):
    convex = [c for c in bank.cylinder_clusters if c.convex]
    concave = [c for c in bank.cylinder_clusters if not c.convex]
    outer = max(convex, key=lambda c: c.radius) if convex else None
    inner = min(concave, key=lambda c: c.radius) if concave else None
    return outer, inner


def derived_candidates(
    bank: MeasurementBank, spec: StructuredSpec,
) -> Dict[str, Tuple[float, str]]:
    if not _is_pulley_spec(spec):
        return {}
    out: Dict[str, Tuple[float, str]] = {}
    aabb = _aabb_sorted(bank)
    outer, inner = _outer_inner_radii(bank)

    for source in (spec.scalars, spec.counts):
        for key, _ in source.items():
            kind = _classify(key)
            if kind is None:
                continue

            if kind == "outer_diameter":
                if outer is not None:
                    out[key] = (2 * outer.radius, f"pulley.cylinder({outer.id}.radius × 2)")
                elif aabb is not None:
                    out[key] = (aabb[2], "pulley.aabb[max]")

            elif kind == "bore_diameter":
                if inner is not None:
                    out[key] = (2 * inner.radius, f"pulley.cylinder({inner.id}.radius × 2)")

            elif kind == "width":
                if aabb is not None:
                    out[key] = (aabb[0], "pulley.aabb[min]")
    return out


# ---------------------------------------------------------------------------

from freecad_validator.consistency.categories.base import Category


class PulleyCategory(Category):
    name = "pulley"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)
