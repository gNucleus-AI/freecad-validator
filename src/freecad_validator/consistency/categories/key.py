"""Key category — machine keys (gib_head_key, half_moon_key, round_key,
square_key, etc.).

Keys are typically a small extruded sketch profile with one of two
shape families:
  * Rectangular / gib-head — the profile sketch holds the key's
    side-view, and spec dimensions like ``length`` or
    ``height_with_head`` correspond to specific line segments of that
    sketch.
  * Half-moon / Woodruff — the profile is a circular arc + chord, and
    ``moon_height`` is the chord-to-arc-tip distance (which equals
    the smaller aabb axis perpendicular to the extrusion).

The generic checker tends to anchor on `Extrude.Length` or
`Extrude.Length2`, which are the extrusion depth — NOT what the spec
means by "length" or "height" for a key (those are sketch-side
dimensions). This category re-anchors:

    height_with_head         → aabb's mid axis (head width across the part)
    moon_height              → aabb's mid axis (semi-circular profile height)

Body length is measured from the profile sketch driving the Pad when that
sketch has a unique long, axis-parallel body edge.

Trigger: ``spec.name`` contains the token ``key`` but NOT ``keyway``
(``keyway`` parts are handled by KeywayCategory).
"""

from __future__ import annotations

import math

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_key_spec(spec: StructuredSpec) -> bool:
    name = (spec.name or "").lower()
    name_toks = name.split("_") if "_" in name else name.split()
    if "key" in name_toks and "keyway" not in name_toks:
        return True
    return False


def _aabb_sorted(bank: MeasurementBank) -> tuple[float, float, float] | None:
    g = bank.globals.get("aabb_sorted")
    if g is None or not isinstance(g.value, tuple) or len(g.value) != 3:
        return None
    return tuple(sorted(float(x) for x in g.value))  # type: ignore[return-value]


def _profile_body_length(bank: MeasurementBank) -> tuple[float, str] | None:
    """Measure the main body edge in a Pad-driving key profile."""
    profile_by_name = {profile.name: profile for profile in bank.sketch_profiles}
    hits: list[tuple[float, str]] = []
    for feature in bank.feature_tree:
        if feature.type_id != "PartDesign::Pad":
            continue
        for dependency in feature.dependencies:
            profile = profile_by_name.get(dependency)
            if profile is None or len(profile.line_segments) < 3:
                continue
            if len({round(segment.length, 6) for segment in profile.line_segments}) < 2:
                continue
            axis = max(profile.line_segments, key=lambda segment: segment.length)
            axis_vec = tuple(axis.end[i] - axis.start[i] for i in range(3))
            axis_len = math.sqrt(sum(value * value for value in axis_vec))
            if axis_len <= 1e-9:
                continue
            direction = tuple(value / axis_len for value in axis_vec)
            candidates: list[float] = []
            for segment in profile.line_segments:
                if segment.index == axis.index:
                    continue
                delta = tuple(segment.end[i] - segment.start[i] for i in range(3))
                projection = abs(sum(delta[i] * direction[i] for i in range(3)))
                if projection / max(segment.length, 1e-9) >= 0.99:
                    candidates.append(projection)
            if candidates:
                hits.append((max(candidates), f"{profile.name}.axis_parallel_body_edge"))
    if not hits:
        return None
    first = hits[0][0]
    if any(abs(value - first) / max(value, first, 1e-9) > 1e-3 for value, _ in hits):
        return None
    return hits[0]


def _classify(key: str) -> str | None:
    toks = _tokens(key)
    if "moon" in toks and "height" in toks:
        return "moon_height"
    if "height" in toks and "with" in toks and "head" in toks:
        return "height_with_head"
    if "length" in toks:
        return "length"
    return None


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    if not _is_key_spec(spec):
        return {}
    out: dict[str, tuple[float, str]] = {}
    aabb = _aabb_sorted(bank)
    body_length = _profile_body_length(bank)

    for source in (spec.scalars, spec.counts):
        for key in source:
            kind = _classify(key)
            if kind is None:
                continue

            if kind == "length" and body_length is not None:
                out[key] = body_length

            elif kind == "moon_height" and aabb is not None:
                # Half-moon key: the chord-to-arc-tip rise is the aabb's
                # second-smallest axis (perpendicular to both extrusion and
                # chord direction).
                out[key] = (aabb[1], "key.aabb[mid]")

            elif kind == "height_with_head" and aabb is not None:
                # Total in-plane height including head — aabb's mid axis
                # since the longest axis is the body length and the smallest
                # is the extrusion depth.
                out[key] = (aabb[1], "key.aabb[mid]")
    return out


# ---------------------------------------------------------------------------


class KeyCategory(Category):
    name = "key"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)
