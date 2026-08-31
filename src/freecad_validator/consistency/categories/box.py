"""Box category — compound prismatic parts (car, couch, drawer, oval table)
where multiple `<noun>_length` / `<noun>_width` / `<noun>_height` /
`<noun>_depth` / `<noun>_thickness` keys describe nested sub-bodies.

The generic checker collapses every length/width key onto the first
Extrude's Length, which is wrong for compound parts (e.g., a car spec
has body_length=4000 AND cabin_length=2000 — these are TWO different
sketches, not one).

Trigger: the spec has at least two distinct `<noun>` prefixes that each
own a length-like key (`<noun>_length`, `<noun>_width`, `<noun>_depth`,
`<noun>_height`, or `<noun>_thickness`).

Algorithm:

  * Pair each `Sketcher::SketchObject` with the next `PartDesign::Pad`
    in feature_tree order — that's the (sketch, extrude) tuple that
    builds one sub-body.
  * For each pair compute three candidate dimensions:
        sketch_max_line  — the longer side of the rectangle sketch
        sketch_min_line  — the shorter side
        extrude_length   — the axial extent
  * Select a pair by CAD/profile semantics only. A single rectangular
    profile can serve bare box dimensions; repeated rectangle profiles
    identify rungs, and duplicated single rectangles identify rails.
    Ambiguous compound profiles are left to the generic report.
"""

from __future__ import annotations

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

_LENGTHLIKE_TOKENS: frozenset[str] = frozenset(
    {
        "length",
        "width",
        "height",
        "depth",
        "thickness",
    }
)

# Nouns that are "body parts" of the whole part (sub-bodies in a
# compound prismatic shape), as opposed to FEATURES on a single body
# (head/hub/shaft/chamfer/groove). BoxCategory should NOT claim
# `<feature>_<dim>` keys — those are owned by feature-specific
# categories (hex hub, flange head, etc.) or fall back to the generic
# checker.
_FEATURE_NOUNS: frozenset[str] = frozenset(
    {
        "head",
        "hub",
        "shaft",
        "neck",
        "groove",
        "chamfer",
        "fillet",
        "rim",
        "bore",
        "hole",
        "lip",
        "boss",
        "stud",
        "tooth",
        "thread",
        "helix",
        "spline",
        "key",
        "keyway",
        "tip",
        "root",
        "pitch",
        "wall",
        "pin",
    }
)


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _classify_dim(key: str) -> str | None:
    """Return 'length' | 'width' | 'height' | 'depth' | 'thickness' or None."""
    toks = _tokens(key)
    for t in ("length", "width", "depth", "height", "thickness"):
        if t in toks:
            return t
    return None


def _noun_prefix(key: str) -> str | None:
    """Return the noun prefix of `<noun>_<dim>` when the noun looks like
    a sub-body (car_body, cabin, seat). Returns None when the key has
    no noun prefix OR when the noun is a single-body feature word
    (head, hub, chamfer) — those belong to feature-specific categories,
    not to BoxCategory."""
    parts = key.split("_")
    if len(parts) < 2:
        return None
    if parts[-1] not in _LENGTHLIKE_TOKENS:
        return None
    noun = "_".join(parts[:-1])
    if any(tok in _FEATURE_NOUNS for tok in noun.split("_")):
        return None
    return noun


def _is_box_spec(spec: StructuredSpec) -> bool:
    """Trigger on either:

    (a) ≥ 2 distinct nouns each owning a length-like key (compound box —
        car, couch); or
    (b) bare ``length`` / ``width`` / ``depth`` plus ``height`` /
        ``thickness`` keys with no noun prefix (simple box — drawer,
        crate). Two-of-three is enough; a single dim alone could fit
        many non-prismatic shapes and we don't want to claim those.
    """
    nouns = set()
    bare_dims = set()
    for source in (spec.scalars, spec.counts):
        for key in source:
            n = _noun_prefix(key)
            if n is not None:
                nouns.add(n)
                continue
            d = _classify_dim(key)
            if d is not None and key == d:  # bare key, exactly the dim word
                bare_dims.add(d)
    if len(nouns) >= 2:
        return True
    planar = bare_dims & {"length", "width", "depth"}
    axial = bare_dims & {"height", "thickness"}
    return len(planar) >= 1 and len(axial) >= 1 and len(bare_dims) >= 2


def _unique_sorted_desc(values, eps_rel: float = 1e-3):
    """Collapse near-duplicate values (rectangles repeat each side
    twice) and return them sorted descending."""
    out = []
    for v in sorted(values, reverse=True):
        if not out:
            out.append(v)
            continue
        denom = max(abs(v), abs(out[-1]), 1e-9)
        if abs(v - out[-1]) / denom > eps_rel:
            out.append(v)
    return out


def _sketch_extrude_pairs(bank: MeasurementBank):
    """Yield (sketch_name, unique_line_lengths_desc, extrude_name,
    extrude_length) by walking feature_tree and pairing each sketch
    with the next pad. Line lengths are de-duplicated so a rectangle's
    two equal sides collapse to one entry — that way the "second
    largest" value is the OTHER side, not a repeat of the longest."""
    pairs = []
    entries = {entry.name: entry for entry in bank.feature_tree}
    for feature in bank.feature_tree:
        if feature.type_id != "PartDesign::Pad":
            continue
        sketches = [
            entries[name]
            for name in feature.dependencies
            if name in entries and entries[name].type_id == "Sketcher::SketchObject"
        ]
        if len(sketches) != 1:
            continue
        sketch = sketches[0]
        raw = [
            float(value)
            for key, value in sketch.properties.items()
            if "LineLength" in key and isinstance(value, (int, float))
        ]
        ex_len = float(feature.properties["Length"]) if "Length" in feature.properties else None
        pairs.append((sketch.name, _unique_sorted_desc(raw), feature.name, ex_len))
    return pairs


_AXIAL = {"height", "thickness"}


def _candidate_from_pair(pair, dim: str):
    sketch_name, line_lengths, extrude_name, ex_len = pair
    if dim == "length" and line_lengths:
        return line_lengths[0], f"{sketch_name}.LineLength[max]"
    if dim in {"width", "depth"} and line_lengths:
        return line_lengths[-1], f"{sketch_name}.LineLength[min]"
    if dim in _AXIAL and ex_len is not None:
        return ex_len, f"{extrude_name}.Length"
    return None


def _profile_rectangle_count(bank: MeasurementBank, sketch_name: str) -> int | None:
    profile = next((item for item in bank.sketch_profiles if item.name == sketch_name), None)
    if profile is None or len(profile.line_segments) < 4:
        return None
    if len(profile.line_segments) % 4 != 0:
        return None
    return len(profile.line_segments) // 4


def _same_pair_dimensions(left, right) -> bool:
    left_dims = [*left[1], left[3]]
    right_dims = [*right[1], right[3]]
    if len(left_dims) != len(right_dims):
        return False
    return all(
        a is not None and b is not None and abs(a - b) / max(abs(a), abs(b), 1e-9) <= 1e-3
        for a, b in zip(left_dims, right_dims, strict=True)
    )


def _semantic_pair(bank: MeasurementBank, pairs, key: str):
    """Select a profile without consulting the expected numeric value."""
    if key in _LENGTHLIKE_TOKENS and len(pairs) == 1:
        return pairs[0]

    parts = key.split("_")
    if not parts or parts[-1] not in _LENGTHLIKE_TOKENS:
        return None
    qualifier_tokens = set(parts[:-1])

    if "rung" in qualifier_tokens or "rungs" in qualifier_tokens:
        if {"total", "span", "spacing"} & qualifier_tokens:
            return None
        matches = [pair for pair in pairs if (_profile_rectangle_count(bank, pair[0]) or 0) > 1]
        return matches[0] if len(matches) == 1 else None

    if ("rail" in qualifier_tokens or "rails" in qualifier_tokens) and not {
        "outer",
        "span",
    } & qualifier_tokens:
        singles = [pair for pair in pairs if _profile_rectangle_count(bank, pair[0]) == 1]
        groups: list[list] = []
        for pair in singles:
            for group in groups:
                if _same_pair_dimensions(pair, group[0]):
                    group.append(pair)
                    break
            else:
                groups.append([pair])
        repeated = [group for group in groups if len(group) >= 2]
        return repeated[0][0] if len(repeated) == 1 else None

    return None


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    if not _is_box_spec(spec):
        return {}
    out: dict[str, tuple[float, str]] = {}
    # Don't fire on parts with rich cylindrical content (retaining
    # rings, gears, anything with ≥ 3 clusters). BoxCategory targets
    # purely prismatic compounds (cars, couches, drawers) — those
    # have 0–2 clusters at most. A retaining ring with 5 clusters is
    # not a prismatic part and should fall through to other categories.
    if len(bank.cylinder_clusters) > 2:
        return out
    pairs = _sketch_extrude_pairs(bank)
    if not pairs:
        return out
    for source in (spec.scalars, spec.counts):
        for key in source:
            dim = _classify_dim(key)
            if dim is None:
                continue
            # Skip feature-noun keys: hub_width, head_length, etc. are
            # owned by feature categories (hex/flange/pin), not by box.
            parts = key.split("_")
            noun_parts = parts[:-1] if parts[-1] in _LENGTHLIKE_TOKENS else parts
            if any(tok in _FEATURE_NOUNS for tok in noun_parts):
                continue
            pair = _semantic_pair(bank, pairs, key)
            if pair is None:
                continue
            hit = _candidate_from_pair(pair, dim)
            if hit is None:
                continue
            value, ref = hit
            out[key] = (value, f"box.{ref}")
    return out


# ---------------------------------------------------------------------------


class BoxCategory(Category):
    name = "box"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)
