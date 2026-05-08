"""Hex (regular hexagonal profile) category helpers.

Handles hex-nut / hex-head / hex-coupling specs whose key dimensions
are standard hex-profile measurements. Observed in the abundant dataset
across ``coupling_nuts`` and ``flange_nuts``:

    hex_width_across_flats      hex_head_width_across_flats
    hex_nut_width_across_flats  across_flats_width
    hex_width_across_corners    hex_head_width_across_corners
    hex_nut_width_across_corners
    hex_head_angle              hex_nut_half_angle

Regular-hex geometry relations used for derivation:

    across_corners = across_flats · (2 / √3)         (ACF = AF/cos30°)
    half_angle     = 30°                              (axis→flat vs axis→corner)

Candidates sourced from the bank:
    across_flats      → closest `plane_pair.offset`  (the two parallel
                        flats are 3 antiparallel pairs at the same AF)
    across_corners    → derived from the AF plane_pair × 2/√3
    angle / half_angle→ geometric constant (30°)

Trigger: any spec key contains one of the token groups `{hex}`,
`{across, flats}`, or `{across, corners}`. `key` alone (as in spline
specs' `key_*`) doesn't trigger since it has no hex semantic.
"""
from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Tuple

from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# ACF = AF / cos(30°) = AF × 2 / √3
_ACROSS_CORNERS_FACTOR = 2.0 / math.sqrt(3.0)

# The axis-to-flat vs axis-to-corner half-angle of a regular hexagon.
# Most specs write `hex_head_angle = 30°` referring to this value.
_HEX_HALF_ANGLE_RAD = math.radians(30.0)

# Token groups that mark a spec as hex-related. Any one triggers.
_TRIGGER_GROUPS: Tuple[FrozenSet[str], ...] = (
    frozenset({"hex"}),
    frozenset({"across", "flats"}),
    frozenset({"across", "corners"}),
)


def _tokens(key: str) -> FrozenSet[str]:
    return frozenset(key.split("_"))


def _is_hex_spec(spec: StructuredSpec) -> bool:
    for source in (spec.scalars, spec.counts):
        for key in source:
            toks = _tokens(key)
            for group in _TRIGGER_GROUPS:
                if group.issubset(toks):
                    return True
    return False


def _classify_key(key: str) -> Optional[str]:
    """Map a spec key to a canonical hex dimension."""
    toks = _tokens(key)
    # Specific combinations first
    if "across" in toks and "flats" in toks:
        return "across_flats"
    if "across" in toks and "corners" in toks:
        return "across_corners"
    # Any hex-angle key. Specs call the 30° half-angle by many names
    # (`hex_head_angle`, `hex_nut_half_angle`); treat them alike.
    if "hex" in toks and "angle" in toks:
        return "half_angle"
    return None


def derived_candidates(
    bank: MeasurementBank, spec: StructuredSpec,
) -> Dict[str, Tuple[float, str]]:
    """Return ``{spec_key: (value, feature_ref)}`` for every hex-related
    spec param. AF comes from the bank's plane-pair offsets; ACF is
    derived from AF via `2/√3`; angles fall out of regular-hex geometry.
    Returns empty if the spec isn't hex-related.
    """
    if not _is_hex_spec(spec):
        return {}

    plane_pair_cands: List[Tuple[float, str]] = [
        (pp.offset, f"{pp.id}.offset") for pp in bank.plane_pairs
    ]

    out: Dict[str, Tuple[float, str]] = {}
    for source in (spec.scalars, spec.counts):
        for spec_key, spec_val in source.items():
            canonical = _classify_key(spec_key)
            if canonical is None:
                continue

            if canonical == "across_flats":
                if not plane_pair_cands:
                    continue
                # Closest plane_pair.offset to the spec-declared AF.
                best = min(plane_pair_cands, key=lambda c: abs(c[0] - float(spec_val)))
                out[spec_key] = (
                    float(best[0]),
                    f"hex.derived_from_cad({best[1]}, AF)",
                )

            elif canonical == "across_corners":
                if not plane_pair_cands:
                    continue
                # Find the plane_pair.offset that best fits AF = ACF × cos(30°).
                af_expected = float(spec_val) * math.cos(math.radians(30.0))
                best = min(plane_pair_cands, key=lambda c: abs(c[0] - af_expected))
                acf_derived = best[0] * _ACROSS_CORNERS_FACTOR
                out[spec_key] = (
                    float(acf_derived),
                    f"hex.derived_from_cad({best[1]} × 2/√3, ACF)",
                )

            elif canonical == "half_angle":
                # Regular-hex geometric constant — not measured.
                out[spec_key] = (
                    _HEX_HALF_ANGLE_RAD,
                    "hex.geom (half_angle = 30°)",
                )
    return out


# ---------------------------------------------------------------------------
# Category subclass.
# ---------------------------------------------------------------------------

from freecad_validator.consistency.categories.base import Category


class HexCategory(Category):
    name = "hex"

    def derived_candidates(
        self, bank: MeasurementBank, spec: StructuredSpec,
    ) -> Dict[str, Tuple[float, str]]:
        return derived_candidates(bank, spec)
