"""Hex (regular hexagonal profile) category helpers.

Handles hex-nut / hex-head / hex-coupling specs whose key dimensions
are standard hex-profile measurements. Observed in the reference
corpus across ``coupling_nuts`` and ``flange_nuts``:

    hex_width_across_flats      hex_head_width_across_flats
    hex_nut_width_across_flats  across_flats_width
    hex_width_across_corners    hex_head_width_across_corners
    hex_nut_width_across_corners
    hex_head_angle              hex_nut_half_angle

Regular-hex geometry relations used for derivation:

    across_corners = across_flats · (2 / √3)         (ACF = AF/cos30°)
    half_angle     = 30°                              (axis→flat vs axis→corner)

Candidates sourced from the bank:
    across_flats      → repeated `plane_pair.offset` group (multiple
                        non-parallel pairs at the same AF)
    across_corners    → derived from the AF plane_pair × 2/√3
    angle / half_angle→ geometric constant (30°)

Trigger: any spec key contains one of the token groups `{hex}`,
`{across, flats}`, or `{across, corners}`. `key` alone (as in spline
specs' `key_*`) doesn't trigger since it has no hex semantic.
"""

from __future__ import annotations

import math

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# ACF = AF / cos(30°) = AF × 2 / √3
_ACROSS_CORNERS_FACTOR = 2.0 / math.sqrt(3.0)

# The axis-to-flat vs axis-to-corner half-angle of a regular hexagon.
# Most specs write `hex_head_angle = 30°` referring to this value.
_HEX_HALF_ANGLE_RAD = math.radians(30.0)

# Token groups that mark a spec as hex-related. Any one triggers.
_TRIGGER_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"hex"}),
    frozenset({"across", "flats"}),
    frozenset({"across", "corners"}),
)


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_hex_spec(spec: StructuredSpec) -> bool:
    name_tokens = set((spec.name or "").lower().replace("-", " ").replace("_", " ").split())
    if "hex" in name_tokens:
        return True
    for source in (spec.scalars, spec.counts):
        for key in source:
            toks = _tokens(key)
            for group in _TRIGGER_GROUPS:
                if group.issubset(toks):
                    return True
    return False


def _classify_key(key: str) -> str | None:
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
    if {"hub", "width"}.issubset(toks):
        return "across_flats"
    return None


def _regular_hex_across_flats(bank: MeasurementBank) -> tuple[float, str] | None:
    """Find a CAD-confirmed regular-hex across-flats measurement.

    A hex prism exposes parallel face pairs in multiple in-plane normal
    directions, all with the same offset. The detector may collapse one of
    the three directions, so two non-parallel pairs are sufficient. A single
    plane-pair (for example a plate thickness) is not a hex signature.
    """
    groups: list[list] = []
    for pair in sorted(bank.plane_pairs, key=lambda item: item.offset):
        for group in groups:
            anchor = group[0].offset
            if abs(pair.offset - anchor) / max(abs(pair.offset), abs(anchor), 1e-9) <= 1e-3:
                group.append(pair)
                break
        else:
            groups.append([pair])

    matches: list[list] = []
    for group in groups:
        if len(group) < 2:
            continue
        has_hex_normals = any(
            abs(abs(sum(a * b for a, b in zip(left.normal, right.normal, strict=True))) - 0.5)
            <= 0.05
            for i, left in enumerate(group)
            for right in group[i + 1 :]
        )
        if has_hex_normals:
            matches.append(group)

    if len(matches) > 1:
        return None
    plane_across_flats = (
        float(sum(pair.offset for pair in matches[0]) / len(matches[0])) if matches else None
    )

    profile_hits: list[tuple[float, str]] = []
    for profile in bank.sketch_profiles:
        segments = profile.line_segments
        if len(segments) != 6:
            continue
        lengths = [segment.length for segment in segments]
        side = sum(lengths) / len(lengths)
        if any(abs(length - side) / max(abs(length), abs(side), 1e-9) > 1e-3 for length in lengths):
            continue
        vertex_degree: dict[tuple[float, float, float], int] = {}
        for segment in segments:
            for point in (segment.start, segment.end):
                vertex = tuple(round(value, 6) for value in point)
                vertex_degree[vertex] = vertex_degree.get(vertex, 0) + 1
        if len(vertex_degree) != 6 or any(degree != 2 for degree in vertex_degree.values()):
            continue
        profile_hits.append((side * math.sqrt(3.0), profile.name))

    if plane_across_flats is None:
        if len(profile_hits) != 1:
            return None
        across_flats, profile_name = profile_hits[0]
        return across_flats, f"{profile_name}.closed_regular_hex(AF)"

    matching_profiles = [
        (value, name)
        for value, name in profile_hits
        if abs(value - plane_across_flats) / max(abs(value), abs(plane_across_flats), 1e-9) <= 1e-3
    ]
    if not matching_profiles:
        return None
    refs = ",".join([*(pair.id for pair in matches[0]), *(name for _, name in matching_profiles)])
    across_flats = plane_across_flats
    return across_flats, refs


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Return ``{spec_key: (value, feature_ref)}`` for every hex-related
    spec param. AF comes from the bank's plane-pair offsets; ACF is
    derived from AF via `2/√3`; angles fall out of regular-hex geometry.
    Returns empty if the spec isn't hex-related.
    """
    if not _is_hex_spec(spec):
        return {}

    hex_af = _regular_hex_across_flats(bank)
    if hex_af is None:
        return {}
    across_flats, plane_refs = hex_af

    out: dict[str, tuple[float, str]] = {}
    for source in (spec.scalars, spec.counts):
        for spec_key in source:
            canonical = _classify_key(spec_key)
            if canonical is None:
                continue

            if canonical == "across_flats":
                out[spec_key] = (
                    across_flats,
                    f"hex.regular_profile({plane_refs}, AF)",
                )

            elif canonical == "across_corners":
                acf_derived = across_flats * _ACROSS_CORNERS_FACTOR
                out[spec_key] = (
                    float(acf_derived),
                    f"hex.regular_profile({plane_refs}, AF × 2/√3)",
                )

            elif canonical == "half_angle":
                # The constant is valid only after the candidate CAD has
                # supplied a regular-hex face-pair signature.
                out[spec_key] = (
                    _HEX_HALF_ANGLE_RAD,
                    f"hex.regular_profile({plane_refs}, half_angle = 30°)",
                )
    return out


# ---------------------------------------------------------------------------
# Category subclass.
# ---------------------------------------------------------------------------


class HexCategory(Category):
    name = "hex"

    def derived_candidates(
        self,
        bank: MeasurementBank,
        spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)
