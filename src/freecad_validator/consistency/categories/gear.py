"""Gear category helpers.

Three geometrically-direct gear dimensions come from the CAD:

    outer_diameter  = 2 · max_tooth_ring_radius   (tip circle)
    root_diameter   = 2 · min_tooth_ring_radius   (root circle)
    teeth_count     = cluster/pattern count on the shaft axis

From those three, the remaining spec dimensions are either:

  (a) **geometric definitions** — always true regardless of gear
      standard:
          pitch_diameter   = module · teeth         (definition of module)
          circular_pitch   = π · module             (definition)
          base_diameter    = pitch · cos(α)         (involute geometry)
          diametral_pitch  = 25.4 / m               (inch-system reciprocal)

  (b) **CAD measurements**, pulled directly from the bank's outer/root
      circles when both are detectable — no standard-proportion
      assumption needed:
          addendum     = (outer_d − pitch_d) / 2
          dedendum     = (pitch_d − root_d) / 2
          whole_depth  = addendum + dedendum

  (c) **textbook fallbacks**, only applied when CAD can't supply the
      number — marked in code with comments (see `derive_params`):
          addendum ≈ m, dedendum ≈ 1.25·m           (common but not universal)
          module   ≈ (outer_d − root_d) / 4.5       (when no declared module)

This lets the category verify any gear — standards-conforming or not —
as long as the CAD is a parametric gear the extractor can measure.
Non-standard tooth proportions will **still** be checked correctly
against the spec because the addendum/dedendum we emit are the
CAD-measured values, not the textbook 1·m / 1.25·m. The only
approximation that leaks into the report is the module when it isn't
declared, and even then the leak only affects the derived
pitch_diameter (a nominal value); the actually-measured outer_d /
root_d are still correct.

The one input we *do* take from the spec is `pressure_angle`, which
has no direct geometric measurement path today. Defaults to 20° when
the spec doesn't declare it (the most common modern value, but not
universal).
"""
from __future__ import annotations

import math

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# 20° is the most common pressure angle on modern spur gears, but it's
# not the only possibility (14.5° legacy, 25° for high-load). Used only
# as a *fallback* when the spec doesn't declare pressure_angle.
DEFAULT_PRESSURE_ANGLE_RAD = math.radians(20.0)

# Minimum cluster/pattern count that looks like a realistic gear. Below
# this a "circular pattern" is more likely coincidence than a tooth ring.
_MIN_TEETH_FOR_GEAR = 10

# Common full-tooth-depth-per-module ratio used to *back out* module
# when it isn't declared: outer_d − root_d = 2·(addendum + dedendum).
# With the textbook proportions addendum=m and dedendum=1.25·m, the
# ratio is 4.5. Real-world gears can deviate (short-addendum, long-
# addendum, stub-tooth systems all exist). This value is only used
# when no module can be read directly from the spec; a mis-estimated
# module here only affects the *derived* addendum/dedendum, and those
# are anyway overridden by direct CAD measurements downstream.
_FALLBACK_WHOLE_DEPTH_PER_MODULE = 4.5

# Token-based key classification. Rule order is most-specific first so
# multi-token combos (like `pitch` + `diameter` → `pitch_diameter`) win
# over single-token fallbacks.
_CANONICAL_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("diametral_pitch",  frozenset({"diametral", "pitch"})),
    ("circular_pitch",   frozenset({"circular", "pitch"})),
    ("pressure_angle",   frozenset({"pressure", "angle"})),
    ("outer_diameter",   frozenset({"outer", "diameter"})),
    ("root_diameter",    frozenset({"root", "diameter"})),
    ("base_diameter",    frozenset({"base", "diameter"})),
    ("pitch_diameter",   frozenset({"pitch", "diameter"})),
    # "tooth" is normalized to "teeth" in `_tokens`, so the rule uses
    # the normalized form — otherwise `tooth_thickness` would never match.
    ("tooth_thickness",  frozenset({"teeth", "thickness"})),
    ("whole_depth",      frozenset({"whole", "depth"})),
    ("module",           frozenset({"module"})),
    ("teeth",            frozenset({"teeth"})),
    ("addendum",         frozenset({"addendum"})),
    ("dedendum",         frozenset({"dedendum"})),
    ("clearance",        frozenset({"clearance"})),
)

# Tokens that mark a key as belonging to the spline category, so we
# don't double-claim in specs that use `spline_*` aliases.
_SPLINE_EXCLUSIVE_TOKENS: frozenset[str] = frozenset({"spline"})


def _tokens(key: str) -> frozenset[str]:
    """Tokenize by `_`, normalizing `tooth` → `teeth` so either singular
    or plural forms classify the same."""
    return frozenset(
        "teeth" if t == "tooth" else t
        for t in key.split("_")
    )


def _classify_key(key: str) -> str | None:
    """Map a spec key to a canonical gear param name via token presence.
    Returns None for keys that are spline-exclusive or don't match any
    rule."""
    toks = _tokens(key)
    if toks & _SPLINE_EXCLUSIVE_TOKENS:
        return None
    for canon, required in _CANONICAL_RULES:
        if required.issubset(toks):
            return canon
    return None


def _find_spec_value(spec: StructuredSpec, canonical: str) -> tuple[str, float] | None:
    """Search spec.scalars + spec.counts for a key that classifies as
    `canonical`. Returns (key, value) of first match or None."""
    for source in (spec.scalars, spec.counts):
        for key, value in source.items():
            if _classify_key(key) == canonical:
                return key, float(value)
    return None


def derive_params(
    module: float,
    number_of_teeth: int,
    pressure_angle_rad: float = DEFAULT_PRESSURE_ANGLE_RAD,
    outer_d: float | None = None,
    root_d: float | None = None,
) -> dict[str, float]:
    """Apply standard spur-gear relationships to derive expected values.

    When `outer_d` and `root_d` are provided (typically measured from
    CAD), `addendum`, `dedendum`, `whole_depth`, `outer_diameter`, and
    `root_diameter` are taken **directly from those measurements**
    rather than back-derived from textbook proportions. The only
    truly geometric-definition-based outputs are:

      - pitch_diameter = module · teeth   (definition of module)
      - circular_pitch = π · module       (definition)
      - base_diameter  = pitch · cos(α)   (involute geometry)
      - diametral_pitch= 25.4 / m         (inch-system reciprocal)

    `tooth_thickness = circular_pitch / 2` assumes a standard
    symmetric tooth-and-gap profile; non-standard tooth systems (e.g.
    asymmetric) can diverge and will show up as `inconsistent` against
    any spec that declares a true CAD-measured tooth thickness.

    When CAD measurements aren't supplied, fall back to the common
    textbook proportions (addendum=m, dedendum=1.25·m); these are
    marked as assumptions in the code, not enshrined standards.
    """
    m = float(module)
    z = int(number_of_teeth)
    alpha = float(pressure_angle_rad)

    pitch_diameter = m * z

    if outer_d is not None and root_d is not None:
        # CAD-measured tooth geometry — no proportion assumption.
        outer_diameter = float(outer_d)
        root_diameter = float(root_d)
        addendum = (outer_diameter - pitch_diameter) / 2.0
        dedendum = (pitch_diameter - root_diameter) / 2.0
    else:
        # Fallback: textbook proportions. Only applied when no CAD tip
        # and root circles are available.
        addendum = m
        dedendum = 1.25 * m
        outer_diameter = pitch_diameter + 2 * addendum
        root_diameter = pitch_diameter - 2 * dedendum

    whole_depth = addendum + dedendum
    circular_pitch = math.pi * m
    tooth_thickness = circular_pitch / 2.0
    base_diameter = pitch_diameter * math.cos(alpha)
    # Diametral pitch is the inch-system reciprocal of module:
    #   DP_[1/in] = 25.4 / m_[mm]
    # Inch-sized gear stock specs this instead of module; cross-checking
    # the two catches unit slips between systems.
    diametral_pitch = 25.4 / m if m > 0 else 0.0

    # Canonical derived-param dict. Keys match those returned by
    # `_classify_key`, so the consistency checker can look up any
    # spec-key-classified-as-canonical against this table. (No alias
    # entries needed since spec keys are normalized via tokens.)
    # `outer_diameter` / `root_diameter` are echoed here for ISO
    # cross-check; the generic path will typically find matching
    # measurements first.
    # Radial clearance between root circle and the mating gear's tip:
    # `clearance = dedendum − addendum`. For standard 1m / 1.25m
    # proportions this collapses to 0.25·m; non-standard tooth systems
    # report whatever the CAD-measured dedendum and addendum imply.
    clearance = dedendum - addendum

    return {
        "module":           m,
        "pitch_diameter":   pitch_diameter,
        "addendum":         addendum,
        "dedendum":         dedendum,
        "whole_depth":      whole_depth,
        "clearance":        clearance,
        "circular_pitch":   circular_pitch,
        "tooth_thickness":  tooth_thickness,
        "outer_diameter":   outer_diameter,
        "root_diameter":    root_diameter,
        "base_diameter":    base_diameter,
        "pressure_angle":   alpha,
        "diametral_pitch":  diametral_pitch,
    }


def measurable_params_from_bank(bank: MeasurementBank) -> dict[str, float] | None:
    """Detect the geometry-observable gear base triple:
    `(outer_diameter, root_diameter, teeth)` from the bank. Derive
    `module` from the first two via the ISO whole-depth relation.

    Heuristic:
      1. Teeth count = smallest cluster/pattern count ≥ 10. (A real gear
         typically has at least 10 teeth; below that the match is more
         likely a coincidence of face counts than an actual tooth ring.)
      2. Tip radius = largest cluster/pattern radius whose count equals
         teeth count. Root radius = smallest such radius.
      3. module = (outer_d − root_d) / 4.5

    Returns None if fewer than 2 teeth-count-matching radii exist (can't
    pin down both tip and root).
    """
    teeth_candidates: set[int] = set()
    for c in bank.cylinder_clusters:
        if c.count >= _MIN_TEETH_FOR_GEAR:
            teeth_candidates.add(c.count)
    for cp in bank.circular_patterns:
        if cp.count >= _MIN_TEETH_FOR_GEAR:
            teeth_candidates.add(cp.count)
    if not teeth_candidates:
        return None
    teeth = min(teeth_candidates)

    # Dedupe radii — if a cluster and its derived circular_pattern carry
    # the same radius, we only want it counted once.
    teeth_match_radii: set[float] = set()
    for c in bank.cylinder_clusters:
        if c.count == teeth:
            teeth_match_radii.add(round(c.radius, 6))
    for cp in bank.circular_patterns:
        if cp.count == teeth:
            teeth_match_radii.add(round(cp.pattern_radius, 6))
    if len(teeth_match_radii) < 2:
        return None

    tip_r = max(teeth_match_radii)
    root_r = min(teeth_match_radii)
    outer_d = 2 * tip_r
    root_d = 2 * root_r
    # Back out module from the CAD's tooth-depth assuming the common
    # textbook proportions (addendum=m, dedendum=1.25·m → whole-depth=2.25·m).
    # Non-standard gears will get an approximate module here, but the
    # downstream addendum/dedendum are taken directly from outer_d and
    # root_d, so only a *mis-estimated* module (used to compute
    # pitch_diameter = m·z) would drive a mismatch — and that mismatch
    # correctly surfaces in the report.
    module = (outer_d - root_d) / _FALLBACK_WHOLE_DEPTH_PER_MODULE
    if module <= 0:
        return None

    return {
        "outer_diameter":  outer_d,
        "root_diameter":   root_d,
        "module":          module,
        "number_of_teeth": float(teeth),
    }


def derived_candidates(
    bank: MeasurementBank, spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Return ``{spec_key: (value, feature_ref)}`` for every gear param
    derivable from the bank's measured tip/root radii + teeth count.

    Spec keys are matched via token classification (see
    :func:`_classify_key`), so prefixed variants like `gear_module`,
    `gear_pitch_diameter`, `number_of_teeth`, and `gear_number_tooth`
    all resolve to the right canonical derivation without explicit
    alias tables. Returns empty if the bank doesn't expose both tip
    and root rings (i.e. the CAD isn't shaped like a standard gear).
    """
    measured = measurable_params_from_bank(bank)
    if measured is None:
        return {}
    teeth = int(measured["number_of_teeth"])
    outer_d = measured["outer_diameter"]
    root_d = measured["root_diameter"]

    # Prefer the spec's declared `module` (or back-derive from declared
    # pitch_diameter + teeth) over CAD-estimated module. The CAD path
    # hits a textbook-proportion assumption (whole_depth ≈ 4.5·m) and
    # picks up FP noise from root_d measurements — both show up as
    # sub-1% drift that cascades into addendum / dedendum when those
    # are computed from (outer_d − pitch_d)/2. Using the spec's
    # declared module when available gives a clean pitch_d and lets
    # the CAD-measured addendum/dedendum stay aligned with the spec.
    module_from_spec = _find_spec_value(spec, "module")
    if module_from_spec is not None:
        module = module_from_spec[1]
        module_source = "spec"
    else:
        pitch_from_spec = _find_spec_value(spec, "pitch_diameter")
        if pitch_from_spec is not None:
            module = pitch_from_spec[1] / teeth
            module_source = "spec.pitch_diameter ÷ teeth"
        else:
            module = measured["module"]
            module_source = "(outer_d − root_d) / 4.5 fallback"

    angle_match = _find_spec_value(spec, "pressure_angle")
    alpha = angle_match[1] if angle_match is not None else DEFAULT_PRESSURE_ANGLE_RAD

    derived = derive_params(
        module, teeth, alpha,
        outer_d=outer_d, root_d=root_d,   # CAD-measured, no proportion assumption
    )
    ref = (
        f"gear.derived_from_cad(outer_d={outer_d:.3f}, root_d={root_d:.3f}, "
        f"z={teeth}, m={module:g} [{module_source}], "
        f"α={math.degrees(alpha):.1f}°)"
    )

    out: dict[str, tuple[float, str]] = {}
    for source in (spec.scalars, spec.counts):
        for spec_key in source:
            canonical = _classify_key(spec_key)
            if canonical is None or canonical not in derived:
                continue
            out[spec_key] = (float(derived[canonical]), ref)
    return out


# ---------------------------------------------------------------------------
# Category subclass — thin wrapper over `derived_candidates`.
# ---------------------------------------------------------------------------


class GearCategory(Category):
    name = "gear"

    def derived_candidates(
        self, bank: MeasurementBank, spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)
