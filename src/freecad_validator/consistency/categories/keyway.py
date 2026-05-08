"""Keyway / key category helpers.

Handles shaft keyways and their mating key pieces. Spec keys observed
in the abundant dataset's ``shaft_with_keyway`` cases:

    keyway_width    keyway_depth    keyway_height    keyway_length
    num_keyway

And, for specs that include the mating key piece:

    key_width       key_depth       key_height       key_length
    num_key

Since keys and keyways mate, their width / depth / length dimensions
are physically the same number — so the same derivation machinery
covers both, and the category triggers on either token once a
`keyway` token appears somewhere in the spec (avoiding false fires
from the very common word "key").

CAD measurement strategy:

    width, depth       → closest `plane_pair.offset`  (slot-side and
                                                       floor-to-surface gaps)
    height, length     → closest axial length (Extrude.Length or
                                                plane_pair.offset)
    num_keyway, num_key→ closest cluster / circular-pattern count

This is opportunistic rather than rigorously correct (the generic
closest-value matcher can still pick a wrong feature when multiple
plane pairs have similar offsets), but it routes keyway keys to the
**right candidate pool** — planes for width/depth, lengths for
height — which the generic `kind` dispatcher can't distinguish.

Uses token-based key classification like gear / spline categories.
"""
from __future__ import annotations

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# The category fires only when at least one spec key contains `keyway`
# as a token — `key` alone would over-trigger (spec sometimes uses
# `shaft_key_dimension_*` or similar in unrelated contexts).
_TRIGGER_TOKEN = "keyway"

# Once triggered, any key with `keyway` OR `key` token is claimable —
# `key_*` keys describe the mating key piece whose dimensions match
# the shaft's keyway by fit.
_CATEGORY_TOKENS: frozenset[str] = frozenset({"keyway", "key"})


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_keyway_spec(spec: StructuredSpec) -> bool:
    for source in (spec.scalars, spec.counts):
        for key in source:
            if _TRIGGER_TOKEN in _tokens(key):
                return True
    return False


def _classify_key(key: str) -> str | None:
    """Map a spec key to a canonical keyway dimension via token presence.
    Returns one of {"width", "depth", "length", "count"} or None."""
    toks = _tokens(key)
    if not (toks & _CATEGORY_TOKENS):
        return None
    if "width" in toks:
        return "width"
    if "depth" in toks:
        return "depth"
    # `height` and `length` both denote the slot's axial extent in this
    # dataset (keyway_height is used interchangeably with keyway_length
    # across specs) — collapse into one canonical.
    if "height" in toks or "length" in toks:
        return "length"
    if "num" in toks or "count" in toks or "number" in toks:
        return "count"
    return None


def _closest(candidates: list[tuple[float, str]], value: float) -> tuple[float, str] | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c[0] - value))


def derived_candidates(
    bank: MeasurementBank, spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Return ``{spec_key: (value, feature_ref)}`` for every keyway-related
    spec key, sourcing candidate values from the bank's plane-pair
    offsets (widths/depths), feature-tree lengths, and cluster/pattern
    counts. Returns empty if no spec key contains `keyway`.
    """
    if not _is_keyway_spec(spec):
        return {}

    plane_pair_cands: list[tuple[float, str]] = [
        (pp.offset, f"{pp.id}.offset") for pp in bank.plane_pairs
    ]

    # Keyway pockets are drawn as rectangles inside their pocket sketch;
    # the rectangle's two side lengths are exactly (width, depth). When
    # plane_pair detection collapses one of those sides — e.g. the
    # depth face joins the shaft cylinder so isn't paired with another
    # plane — we miss the value. Pull line lengths from `sketch_profiles`
    # to cover that gap. Only sketches with ≥ 2 distinct line lengths
    # are useful here (a single repeated length is a square or strip).
    for sp in bank.sketch_profiles:
        # Collapse near-duplicates so a 4-sided rectangle yields 2 entries.
        unique: list[float] = []
        for ln in sp.line_lengths:
            if not unique or abs(ln - unique[-1]) / max(abs(ln), abs(unique[-1]), 1e-9) > 1e-3:
                unique.append(ln)
        if len(unique) < 2:
            continue
        for ln in unique:
            plane_pair_cands.append((ln, f"{sp.name}.LineLength={ln:.3f}"))

    # For axial length: feature-tree `Length` properties plus plane-pair
    # offsets (an Extrude may carry the slot's axial length, but so can a
    # plane-pair between the two slot end-caps).
    length_cands: list[tuple[float, str]] = list(plane_pair_cands)
    for entry in bank.feature_tree:
        for prop in ("Length", "Length2", "Height", "Depth", "Width"):
            if prop in entry.properties:
                length_cands.append((entry.properties[prop], f"{entry.name}.{prop}"))

    count_cands: list[tuple[float, str]] = [
        (float(c.count), f"{c.id}.count") for c in bank.cylinder_clusters
    ]
    count_cands.extend(
        (float(cp.count), f"{cp.id}.count") for cp in bank.circular_patterns
    )
    # `Occurrences` on PartDesign pattern features is the direct count
    # for e.g. 2 keyways made via PolarPattern. Circular-pattern
    # detection requires N≥3 and cluster counts can't always tell how
    # many slots the shaft has, so this is often the only source for
    # num_keyway via pattern features.
    for entry in bank.feature_tree:
        if "Occurrences" in entry.properties:
            count_cands.append(
                (float(entry.properties["Occurrences"]), f"{entry.name}.Occurrences")
            )
    # Rectangular-slot count: a keyway typically appears in a sketch as
    # a 4-sided rectangle. When multiple keyways share one sketch
    # (generators often emit both keyways as siblings in sketch_1),
    # line count / 4 gives the keyway count. Only emit when the sketch
    # has line segments (not arcs/circles) — those won't form slot
    # rectangles.
    for entry in bank.feature_tree:
        n_lines = sum(1 for k in entry.properties if "LineLength" in k)
        if n_lines >= 4 and n_lines % 4 == 0 and n_lines <= 32:
            # Cap at 32 (8 rectangles) — more would be noise from a
            # non-keyway sketch (e.g. gear tooth profile polylines).
            count_cands.append((
                float(n_lines // 4),
                f"{entry.name} rectangle count ({n_lines} lines ÷ 4)",
            ))

    out: dict[str, tuple[float, str]] = {}
    for source in (spec.scalars, spec.counts):
        for spec_key, spec_val in source.items():
            canonical = _classify_key(spec_key)
            if canonical is None:
                continue

            if canonical in ("width", "depth"):
                pool = plane_pair_cands
            elif canonical == "length":
                pool = length_cands
            elif canonical == "count":
                pool = count_cands
            else:
                continue

            best = _closest(pool, float(spec_val))
            if best is None:
                continue
            val, feat = best
            out[spec_key] = (float(val), f"keyway.derived_from_cad({feat})")
    return out


# ---------------------------------------------------------------------------
# Category subclass.
# ---------------------------------------------------------------------------


class KeywayCategory(Category):
    name = "keyway"

    def derived_candidates(
        self, bank: MeasurementBank, spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)
