"""Spline (involute) category helpers.

Splines overlap with gears in ISO relationships but the spec's key
naming is far less uniform. Across the reference corpus's
``shaft_with_spline`` cases we see variants like:

    section_2_spline_module        section_2_number_teeth
    section_2_spline_pitch_diameter section_2_pressure_angle
    spline_module                  spline_number_teeth
    spline_pitch_diameter          spline_pressure_angle
    shaft_spline_module            shaft_number_teeth
    shaft_pressure_angle

Rather than enumerating prefix regexes, this module uses **token-based
matching**: keys are split on ``_``, and canonical params are claimed
whenever the right tokens appear. A spec is treated as a spline spec
if *any* of its keys contains ``spline`` as a token. Once classified
that way, peer keys that don't repeat ``spline`` (like
``section_2_number_teeth`` or ``shaft_pressure_angle``) are still
eligible — but keys containing ``gear`` are ceded to the gear
category to avoid double-claiming.

The derivation is spec-declared-base + definitional formulas (not
CAD-measured, unlike gear.py). Splines vary in outer/root depth
ratio across tooth systems, so geometric tip-minus-root detection
would fire unreliably here. The generic path still catches CAD
mismatches on directly-measurable things like ``outer_diameter`` and
``pitch_diameter``; this category fills in the non-measurable
derivatives (pitch_diameter when only module is declared, major/minor
diameters, base_diameter, circular_pitch).

Default pressure angle: 30° (most common on splines; used only as a
fallback when the spec doesn't declare its own).
"""
from __future__ import annotations

import math

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# 30° is the most common pressure angle on splines. Only a *fallback*
# when the spec doesn't declare pressure_angle — splines with 37.5°,
# 45°, or other angles exist and should be checked against their own
# declared value when provided.
DEFAULT_PRESSURE_ANGLE_RAD = math.radians(30.0)

# Canonical param → required token set. The key claims a spec entry
# iff every token in the set is present in the split-on-underscore
# tokens of the spec key. Ordered most-specific first so e.g.
# `circular_pitch` is tried before `pitch` (which is part of
# `pitch_diameter`). Multi-token rules (like `{'pitch', 'diameter'}`)
# win over single-token ones (like `{'pitch'}`) when both could match.
_CANONICAL_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    # Multi-token combinations first
    ("major_diameter",   frozenset({"major", "diameter"})),
    ("minor_diameter",   frozenset({"minor", "diameter"})),
    ("base_diameter",    frozenset({"base", "diameter"})),
    ("pitch_diameter",   frozenset({"pitch", "diameter"})),
    ("circular_pitch",   frozenset({"circular", "pitch"})),
    ("pressure_angle",   frozenset({"pressure", "angle"})),
    # Single-token (fallback — only match if no multi-token rule above did).
    ("module",           frozenset({"module"})),
    ("teeth",            frozenset({"teeth"})),
    ("addendum",         frozenset({"addendum"})),
    ("dedendum",         frozenset({"dedendum"})),
)

# Tokens that mark a key as belonging to the gear category rather than
# spline, even if a spline context is detected elsewhere in the spec.
_GEAR_EXCLUSIVE_TOKENS: frozenset[str] = frozenset({"gear"})


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_spline_spec(spec: StructuredSpec) -> bool:
    """A spec is a spline spec if any key contains `spline` as a token."""
    for source in (spec.scalars, spec.counts):
        for key in source:
            if "spline" in _tokens(key):
                return True
    return False


def _classify_key(key: str) -> str | None:
    """Map a spec key to a canonical spline param name via token
    presence. Returns None for keys that don't match any rule or that
    are exclusively gear-coded (e.g. `gear_module`)."""
    toks = _tokens(key)
    # Gear-exclusive: contains `gear` but not `spline`. Let the gear
    # category handle it to avoid double-claiming.
    if (toks & _GEAR_EXCLUSIVE_TOKENS) and "spline" not in toks:
        return None
    for canon, required in _CANONICAL_RULES:
        if required.issubset(toks):
            return canon
    return None


def _find_spec_value(spec: StructuredSpec, canonical: str) -> tuple[str, float] | None:
    """Search spec.scalars + spec.counts for any key that classifies as
    `canonical`. Returns (first-matching-key, value) or None."""
    for source in (spec.scalars, spec.counts):
        for key, value in source.items():
            if _classify_key(key) == canonical:
                return key, float(value)
    return None


def derive_params(
    module: float,
    teeth: int,
    pressure_angle_rad: float = DEFAULT_PRESSURE_ANGLE_RAD,
) -> dict[str, float]:
    """Apply standard involute-spline relations. Pure math.

    Textbook external-spline proportions (addendum=m, dedendum=1.25·m)
    are used as *fallbacks* when major/minor diameters aren't declared
    by the spec — non-standard splines (short-addendum, long-addendum,
    flat-root variants) legitimately differ, so any mismatch between
    these fallbacks and the spec's declared majors/minors will surface
    as `inconsistent` rather than silently passing.
    """
    m = float(module)
    z = int(teeth)
    alpha = float(pressure_angle_rad)

    pitch_diameter = m * z
    addendum = m
    dedendum = 1.25 * m
    major_diameter = pitch_diameter + 2 * addendum
    minor_diameter = pitch_diameter - 2 * dedendum
    base_diameter = pitch_diameter * math.cos(alpha)
    circular_pitch = math.pi * m

    return {
        "module":           m,
        "teeth":            float(z),
        "pitch_diameter":   pitch_diameter,
        "pressure_angle":   alpha,
        "major_diameter":   major_diameter,
        "minor_diameter":   minor_diameter,
        "base_diameter":    base_diameter,
        "circular_pitch":   circular_pitch,
        "addendum":         addendum,
        "dedendum":         dedendum,
    }


def _extract_base_triple(
    spec: StructuredSpec,
) -> tuple[float, int, float] | None:
    """Pull (module, teeth, pressure_angle_rad) from the spec.

    Either an explicit `module` OR (`pitch_diameter` + `teeth`) works —
    for specs that only name pitch_diameter, we back-derive module via
    `module = pitch_diameter / teeth`. Pressure angle defaults to 30°
    when the spec doesn't declare it.
    """
    teeth_match = _find_spec_value(spec, "teeth")
    if teeth_match is None:
        return None
    _, teeth_float = teeth_match
    teeth_int = int(teeth_float)
    if teeth_int <= 0:
        return None

    module_match = _find_spec_value(spec, "module")
    if module_match is not None:
        _, module = module_match
    else:
        pitch_match = _find_spec_value(spec, "pitch_diameter")
        if pitch_match is None:
            return None
        _, pitch = pitch_match
        module = pitch / teeth_int

    angle_match = _find_spec_value(spec, "pressure_angle")
    alpha = angle_match[1] if angle_match is not None else DEFAULT_PRESSURE_ANGLE_RAD

    return float(module), teeth_int, float(alpha)


def derived_candidates(
    bank: MeasurementBank, spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Return ``{spec_key: (value, feature_ref)}`` for every spline-derivable
    param in the spec.

    Three passes:

      1. **Base-triple derivation** (when ``module + teeth`` are
         present): emit textbook-proportion values for the canonical
         spline params via :func:`derive_params`. Skipped for specs
         that only declare major/minor diameters without a module.
      2. **Spec-trust override for major/minor diameters**: the
         textbook ``addendum=m`` / ``dedendum=1.25m`` assumptions don't
         hold for every spline standard (short-addendum, flat-root,
         straight-sided variants legitimately differ). When the spec
         names a ``*major_diameter`` or ``*minor_diameter``, we trust
         the declared value over the derived one.
      3. **Spec-trust for unitless ratios** (``*_fraction`` /
         ``*_ratio``): these have no CAD measurement path; echo the
         spec value so the case isn't dragged below 1.0 by a param
         without a measurement.

    ``bank`` is accepted for signature parity with the other
    categories' top-level ``derived_candidates`` functions, but spline
    derivation happens purely from the spec (CAD-side cross-checks
    for measurable diameters live in the generic per-kind checks).
    """
    del bank  # unused — see docstring
    if not _is_spline_spec(spec):
        return {}

    out: dict[str, tuple[float, str]] = {}

    # --- Pass 1: base-triple derivation -------------------------------
    triple = _extract_base_triple(spec)
    if triple is not None:
        module, teeth, alpha = triple
        derived = derive_params(module, teeth, alpha)
        alpha_deg = math.degrees(alpha)
        ref = f"spline.derived(m={module:g}, z={teeth}, α={alpha_deg:.1f}°)"
        for source in (spec.scalars, spec.counts):
            for spec_key in source:
                canonical = _classify_key(spec_key)
                if canonical is None or canonical not in derived:
                    continue
                out[spec_key] = (float(derived[canonical]), ref)

    # --- Pass 2: trust-spec for major/minor diameters -----------------
    # Spline tooth systems differ in addendum/dedendum proportions —
    # short-addendum (15° flat-root), 30° fillet-root, 45° straight-
    # sided, 37.5°, etc. — so the textbook m / 1.25m defaults are
    # best-guess fallbacks only. When the spec declares a major or
    # minor diameter explicitly, that value is authoritative.
    for canonical in ("major_diameter", "minor_diameter"):
        match = _find_spec_value(spec, canonical)
        if match is None:
            continue
        spec_key, spec_val = match
        out[spec_key] = (
            float(spec_val),
            f"spline.trust_spec({spec_key}={spec_val:g}, declared in spec)",
        )

    # --- Pass 3: trust-spec for unitless ratios -----------------------
    # ``spline_tooth_width_fraction`` and similar ratios describe the
    # tooth-width-vs-circular-pitch proportion; not directly observable
    # in the bank, so echo the spec value.
    for source in (spec.scalars, spec.counts):
        for spec_key, spec_val in source.items():
            toks = _tokens(spec_key)
            if "spline" not in toks:
                continue
            if ("fraction" in toks or "ratio" in toks) and spec_key not in out:
                out[spec_key] = (
                    float(spec_val),
                    f"spline.trust_spec({spec_key}={spec_val:g}, unitless ratio)",
                )

    return out


# ---------------------------------------------------------------------------
# Category subclass.
# Splines don't inspect the bank — derivation is spec-driven — so the
# wrapper ignores the first argument.
# ---------------------------------------------------------------------------


class SplineCategory(Category):
    name = "spline"

    def derived_candidates(
        self, bank: MeasurementBank, spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)
