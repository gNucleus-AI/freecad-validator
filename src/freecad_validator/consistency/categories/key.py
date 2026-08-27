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

    length / body_length     → closest sketch LineLength to the spec value
    height_with_head         → aabb's mid axis (head width across the part)
    moon_height              → aabb's mid axis (semi-circular profile height)
    width                    → Extrude.Length (matches generic; harmless echo)

Trigger: ``spec.name`` contains the token ``key`` but NOT ``keyway``
(``keyway`` parts are handled by KeywayCategory).
"""

from __future__ import annotations

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


def _sketch_line_lengths(bank: MeasurementBank) -> list[tuple[float, str]]:
    """Return [(line_length, feature_ref), ...] from every sketch."""
    out = []
    for ft in bank.feature_tree:
        if ft.type_id != "Sketcher::SketchObject":
            continue
        for prop, val in ft.properties.items():
            if "LineLength" in prop and isinstance(val, (int, float)):
                out.append((float(val), f"{ft.name}.{prop}"))
    return out


def _classify(key: str) -> str | None:
    toks = _tokens(key)
    if "moon" in toks and "height" in toks:
        return "moon_height"
    if "height" in toks and "with" in toks and "head" in toks:
        return "height_with_head"
    if "length" in toks:
        return "length"
    if "thickness" in toks:
        return "thickness"
    return None


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    if not _is_key_spec(spec):
        return {}
    out: dict[str, tuple[float, str]] = {}
    aabb = _aabb_sorted(bank)
    line_lengths = _sketch_line_lengths(bank)

    for source in (spec.scalars, spec.counts):
        for key, spec_val in source.items():
            try:
                spec_v = float(spec_val)
            except (TypeError, ValueError):
                continue
            kind = _classify(key)
            if kind is None:
                continue

            if kind == "length" and line_lengths:
                # Pick the sketch line whose length is closest to the spec —
                # disambiguates body length (e.g. 14) from total length (18)
                # in compound profiles like gib-head keys.
                best_val, best_ref = min(line_lengths, key=lambda lr: abs(lr[0] - spec_v))
                out[key] = (best_val, f"key.sketch({best_ref})")

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
