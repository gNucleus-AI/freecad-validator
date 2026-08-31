"""Keyway / key category helpers.

Handles shaft keyways and their mating key pieces. Spec keys observed
in the reference corpus's ``shaft_with_keyway`` cases:

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

    width, depth       → the two side lengths of the unique rectangular
                         keyway sketch profile
    height, length     → Length of the Pad/Pocket that directly depends
                         on that sketch
    num_keyway, num_key→ pattern Occurrences, otherwise rectangle count

The expected numeric values are never used to select among measurements.
Ambiguous profiles or feature links remain unverified.

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


def _keyway_profile(bank: MeasurementBank):
    """Return the one sketch profile that consists of slot rectangles."""
    matches = []
    for profile in bank.sketch_profiles:
        count = len(profile.line_lengths)
        unique = sorted({round(value, 6) for value in profile.line_lengths})
        if count >= 4 and count % 4 == 0 and 1 <= len(unique) <= 2:
            matches.append((profile, unique))
    if len(matches) != 1:
        return None
    return matches[0]


def _profile_feature_length(
    bank: MeasurementBank, profile_name: str
) -> tuple[float, str, str] | None:
    """Read Length from the feature directly driven by the slot sketch."""
    hits = []
    for entry in bank.feature_tree:
        if profile_name not in entry.dependencies or entry.type_id not in {
            "PartDesign::Pad",
            "PartDesign::Pocket",
        }:
            continue
        if "Length" in entry.properties:
            hits.append((float(entry.properties["Length"]), f"{entry.name}.Length", entry.name))
    if len(hits) != 1:
        return None
    return hits[0]


def _pattern_count(bank: MeasurementBank, source_feature: str | None) -> tuple[float, str] | None:
    if source_feature is None:
        return None
    hits = [
        (float(entry.properties["Occurrences"]), f"{entry.name}.Occurrences")
        for entry in bank.feature_tree
        if "Pattern" in entry.type_id
        and "Occurrences" in entry.properties
        and source_feature in entry.dependencies
    ]
    if not hits:
        return None
    first = hits[0][0]
    if any(value != first for value, _ in hits):
        return None
    return first, ",".join(ref for _, ref in hits)


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Derive keyway dimensions from the CAD slot profile and its feature."""
    if not _is_keyway_spec(spec):
        return {}
    profile_hit = _keyway_profile(bank)
    if profile_hit is None:
        return {}
    profile, dimensions = profile_hit
    width = dimensions[-1]
    depth = dimensions[0]
    feature_length = _profile_feature_length(bank, profile.name)
    length = feature_length[:2] if feature_length is not None else None
    source_feature = feature_length[2] if feature_length is not None else None
    count = _pattern_count(bank, source_feature) or (
        float(len(profile.line_lengths) // 4),
        f"{profile.name}.rectangle_count",
    )

    out: dict[str, tuple[float, str]] = {}
    for source in (spec.scalars, spec.counts):
        for spec_key in source:
            canonical = _classify_key(spec_key)
            if canonical is None:
                continue

            if canonical == "width":
                hit = (width, f"{profile.name}.rectangle_width")
            elif canonical == "depth":
                hit = (depth, f"{profile.name}.rectangle_depth")
            elif canonical == "length":
                hit = length
            elif canonical == "count":
                hit = count
            else:
                continue
            if hit is None:
                continue
            val, feat = hit
            out[spec_key] = (float(val), f"keyway.derived_from_cad({feat})")
    return out


# ---------------------------------------------------------------------------
# Category subclass.
# ---------------------------------------------------------------------------


class KeywayCategory(Category):
    name = "keyway"

    def derived_candidates(
        self,
        bank: MeasurementBank,
        spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)
