"""Helical-gear category helpers.

Mirrors `gear.py` (which is spur-only) but uses the helical-gear
geometry where the transverse pitch diameter scales by ``1 / cos β``
and the involute is built in the transverse plane on the transverse
pressure angle ``α_t = atan(tan α_n / cos β)``.

Conventions for helical specs:

  * ``gear_module`` / ``module`` is the **normal** module ``m_n``
    (perpendicular to the tooth's helix).
  * ``outer_diameter`` / ``root_diameter`` / ``pitch_diameter`` are
    **transverse** measurements (taken in the plane perpendicular to
    the gear axis — what a caliper reads).
  * ``pressure_angle`` is the **normal** pressure angle ``α_n``;
    ``transverse_pressure_angle`` is exposed separately as a derived
    quantity for specs that declare it.
  * Addendum (= m_n) and dedendum (= 1.25·m_n) are radial offsets
    measured in the normal module — same as spur.

Standard helical-gear relations applied here:

    pitch_diameter (transverse) = N · m_n / cos β
    addendum (radial)           = m_n
    dedendum (radial)           = 1.25 · m_n
    outer_diameter (transverse) = pitch + 2·m_n = m_n · (N / cos β + 2)
    root_diameter  (transverse) = pitch − 2.5·m_n
    base_diameter  (transverse) = pitch · cos(α_t)
    transverse_pressure_angle   = atan(tan α_n / cos β)
    transverse_module           = m_n / cos β
    circular_pitch (normal)     = π · m_n
    diametral_pitch             = 25.4 / m_n   (inch-system reciprocal)
    lead (axial advance per rev)= π · pitch_diameter / tan β

When β = 0 every formula collapses to the spur case, but this category
*only triggers* when the spec declares ``helix_angle`` — so spur gear
specs continue to be handled by `gear.py` and there's no double-claim
overlap.
"""

from __future__ import annotations

import math

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# 20° is the most common normal pressure angle on modern helical gears.
# Used only as a fallback when the spec doesn't declare its own.
DEFAULT_PRESSURE_ANGLE_RAD = math.radians(20.0)

# Mirror of `gear._MIN_TEETH_FOR_GEAR`: bank-side tooth ring detection
# only fires for cluster/pattern counts at or above this threshold.
_MIN_TEETH_FOR_GEAR = 10

# Same back-out factor as spur: addendum (= m_n) plus dedendum (= 1.25·m_n)
# is 2.25·m_n radial → tip-minus-root diameter is 4.5·m_n. The factor is
# in the *normal* module and is unaffected by helix angle because the
# radial offsets aren't scaled by cos β.
_FALLBACK_WHOLE_DEPTH_PER_MODULE = 4.5

# Token-based key classification — same rule set as `gear.py` plus the
# helical-specific keys. Order is most-specific first so multi-token
# combos win over single-token fallbacks.
_CANONICAL_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("transverse_pressure_angle", frozenset({"transverse", "pressure", "angle"})),
    ("transverse_module", frozenset({"transverse", "module"})),
    ("normal_pressure_angle", frozenset({"normal", "pressure", "angle"})),
    ("normal_module", frozenset({"normal", "module"})),
    ("helix_angle", frozenset({"helix", "angle"})),
    ("helix_hand", frozenset({"helix", "hand"})),
    ("lead", frozenset({"lead"})),
    ("diametral_pitch", frozenset({"diametral", "pitch"})),
    ("circular_pitch", frozenset({"circular", "pitch"})),
    ("pressure_angle", frozenset({"pressure", "angle"})),
    ("outer_diameter", frozenset({"outer", "diameter"})),
    ("root_diameter", frozenset({"root", "diameter"})),
    ("base_diameter", frozenset({"base", "diameter"})),
    ("pitch_diameter", frozenset({"pitch", "diameter"})),
    ("tooth_thickness", frozenset({"teeth", "thickness"})),
    ("whole_depth", frozenset({"whole", "depth"})),
    ("module", frozenset({"module"})),
    ("teeth", frozenset({"teeth"})),
    ("addendum", frozenset({"addendum"})),
    ("dedendum", frozenset({"dedendum"})),
    ("clearance", frozenset({"clearance"})),
)

# Cede spline-prefixed keys to `SplineCategory` (some helical-gear cases
# also carry an internal spline; we don't want both categories to claim
# the spline keys).
_SPLINE_EXCLUSIVE_TOKENS: frozenset[str] = frozenset({"spline"})


def _tokens(key: str) -> frozenset[str]:
    """Tokenize by `_`, normalizing `tooth` → `teeth` so either singular
    or plural forms classify the same."""
    return frozenset("teeth" if t == "tooth" else t for t in key.split("_"))


def _classify_key(key: str) -> str | None:
    """Map a spec key to a canonical helical-gear param via token presence.
    Returns None for spline-exclusive keys or keys that match no rule."""
    toks = _tokens(key)
    if toks & _SPLINE_EXCLUSIVE_TOKENS:
        return None
    for canon, required in _CANONICAL_RULES:
        if required.issubset(toks):
            return canon
    return None


def _is_helical_spec(spec: StructuredSpec) -> bool:
    """A spec is a helical-gear spec iff it declares ``helix_angle``.
    No other heuristic is reliable — the presence of ``gear`` tokens
    alone could equally be a spur gear."""
    for source in (spec.scalars, spec.counts):
        for key in source:
            if _classify_key(key) == "helix_angle":
                return True
    return False


def _find_spec_value(spec: StructuredSpec, canonical: str) -> tuple[str, float] | None:
    """Search spec.scalars + spec.counts for a key that classifies as
    ``canonical``. Returns (key, value) of first match or None."""
    for source in (spec.scalars, spec.counts):
        for key, value in source.items():
            if _classify_key(key) == canonical:
                return key, float(value)
    return None


def derive_params(
    module_n: float,
    teeth: int,
    helix_angle_rad: float,
    normal_pressure_angle_rad: float = DEFAULT_PRESSURE_ANGLE_RAD,
    outer_d: float | None = None,
    root_d: float | None = None,
) -> dict[str, float]:
    """Apply standard helical-gear relations.

    When ``outer_d`` and ``root_d`` come from CAD measurements, the
    radial addendum/dedendum are taken directly from those values
    rather than back-derived from textbook proportions — same trick
    `gear.derive_params` uses.

    All ``*_diameter`` outputs are **transverse**. Module-style outputs
    (``module``, ``circular_pitch``, ``diametral_pitch``,
    ``tooth_thickness``) are in **normal** module units; the transverse
    versions are exposed under their own keys.
    """
    m_n = float(module_n)
    z = int(teeth)
    beta = float(helix_angle_rad)
    alpha_n = float(normal_pressure_angle_rad)

    cos_b = math.cos(beta)
    transverse_module = m_n / cos_b
    transverse_pressure_angle = math.atan(math.tan(alpha_n) / cos_b)
    pitch_diameter = m_n * z / cos_b

    if outer_d is not None and root_d is not None:
        outer_diameter = float(outer_d)
        root_diameter = float(root_d)
        addendum = (outer_diameter - pitch_diameter) / 2.0
        dedendum = (pitch_diameter - root_diameter) / 2.0
    else:
        # Textbook radial proportions in the *normal* module — same as
        # spur because addendum / dedendum aren't scaled by cos β.
        addendum = m_n
        dedendum = 1.25 * m_n
        outer_diameter = pitch_diameter + 2 * addendum
        root_diameter = pitch_diameter - 2 * dedendum

    whole_depth = addendum + dedendum
    clearance = dedendum - addendum
    # Normal-module-side quantities (the "true" tooth size primitives).
    circular_pitch = math.pi * m_n
    tooth_thickness = circular_pitch / 2.0
    diametral_pitch = 25.4 / m_n if m_n > 0 else 0.0
    base_diameter = pitch_diameter * math.cos(transverse_pressure_angle)
    # Axial distance the tooth advances per full revolution. For β=0
    # tan(β)=0 → lead is infinite (spur gear), so guard against it.
    lead = math.pi * pitch_diameter / math.tan(beta) if abs(beta) > 1e-12 else float("inf")

    return {
        "module": m_n,
        "normal_module": m_n,
        "transverse_module": transverse_module,
        "teeth": float(z),
        "pitch_diameter": pitch_diameter,
        "addendum": addendum,
        "dedendum": dedendum,
        "whole_depth": whole_depth,
        "clearance": clearance,
        "circular_pitch": circular_pitch,
        "tooth_thickness": tooth_thickness,
        "outer_diameter": outer_diameter,
        "root_diameter": root_diameter,
        "base_diameter": base_diameter,
        "pressure_angle": alpha_n,
        "normal_pressure_angle": alpha_n,
        "transverse_pressure_angle": transverse_pressure_angle,
        "helix_angle": beta,
        "diametral_pitch": diametral_pitch,
        "lead": lead,
    }


def measurable_params_from_bank(bank: MeasurementBank) -> dict[str, float] | None:
    """Same heuristic as the spur ``gear.measurable_params_from_bank``:
    find the smallest cluster/pattern count ≥ 10 (= teeth), pull the
    largest matching radius (= tip) and smallest (= root), and back out
    the *normal* module from ``(outer_d − root_d) / 4.5``.

    The 4.5 factor is unaffected by helix angle: tip and root are both
    offset radially from the pitch circle by the *normal* addendum /
    dedendum, so their difference is ``4.5·m_n`` for either spur or
    helical. Returns None if fewer than two matching radii are found.
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
    module_n = (outer_d - root_d) / _FALLBACK_WHOLE_DEPTH_PER_MODULE
    if module_n <= 0:
        return None

    return {
        "outer_diameter": outer_d,
        "root_diameter": root_d,
        "module": module_n,
        "number_of_teeth": float(teeth),
    }


def _resolve_module(
    spec: StructuredSpec,
    teeth: int,
    helix_angle_rad: float,
    measured: dict[str, float] | None,
) -> tuple[float | None, str | None]:
    """Pick the most authoritative normal module ``m_n`` for the case.

    Priority order (each step is the most direct derivation when the
    inputs are present):

      1. Spec declares ``module`` / ``normal_module``     → trust it.
      2. Spec declares ``pitch_diameter``                  → m_n = pitch · cos(β) / z.
      3. Spec declares ``outer_diameter``                  → m_n = outer / (z/cos β + 2).
      4. Spec declares ``root_diameter``                   → m_n = root  / (z/cos β − 2.5).
      5. Bank found a tooth ring (textbook 4.5·m fallback) → m_n from measured.

    Returns ``(module_n, source_label)`` or ``(None, None)`` if no
    derivation is possible.
    """
    module_match = _find_spec_value(spec, "module") or _find_spec_value(spec, "normal_module")
    if module_match is not None:
        return module_match[1], "spec"

    cos_b = math.cos(helix_angle_rad)

    pitch_match = _find_spec_value(spec, "pitch_diameter")
    if pitch_match is not None:
        return (pitch_match[1] * cos_b / teeth, "spec.pitch_diameter · cos β / teeth")

    outer_match = _find_spec_value(spec, "outer_diameter")
    if outer_match is not None:
        return (outer_match[1] / (teeth / cos_b + 2.0), "spec.outer_diameter ÷ (z/cos β + 2)")

    root_match = _find_spec_value(spec, "root_diameter")
    if root_match is not None:
        denom = teeth / cos_b - 2.5
        if denom > 0:
            return (root_match[1] / denom, "spec.root_diameter ÷ (z/cos β − 2.5)")

    if measured is not None and measured.get("module", 0) > 0:
        return measured["module"], "(outer_d − root_d) / 4.5 bank fallback"

    return None, None


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Return ``{spec_key: (value, feature_ref)}`` for every helical-gear
    param derivable from the spec (and optionally the bank's tooth ring).

    No-op for non-helical specs (no ``helix_angle`` key declared). When
    the spec is helical, we drive the derivation from
    ``(m_n, z, β, α_n)`` and emit canonical values for every spec key
    that classifies onto a derived param.
    """
    if not _is_helical_spec(spec):
        return {}

    # Helix angle is required by definition of "helical spec". The spec
    # may declare it in either radians or degrees; the upstream parser
    # normalizes ``*_angle`` values to radians by the time they hit
    # ``spec.scalars``, so no conversion is needed here.
    helix_match = _find_spec_value(spec, "helix_angle")
    if helix_match is None:
        return {}
    beta = helix_match[1]

    teeth_from_spec = _find_spec_value(spec, "teeth")
    measured = measurable_params_from_bank(bank)
    outer_d: float | None
    root_d: float | None

    if teeth_from_spec is not None:
        teeth = int(teeth_from_spec[1])
        # Same anti-mismatch guard as spur: if the bank found a tooth
        # ring with a different count (e.g. an internal spline's 12
        # teeth on a part whose gear has 100 teeth), the bank's
        # outer_d / root_d belong to that other feature — drop them
        # for the helical-gear derivation.
        if measured is not None and int(measured["number_of_teeth"]) == teeth:
            outer_d = measured["outer_diameter"]
            root_d = measured["root_diameter"]
        else:
            outer_d = None
            root_d = None
    elif measured is not None:
        teeth = int(measured["number_of_teeth"])
        outer_d = measured["outer_diameter"]
        root_d = measured["root_diameter"]
    else:
        return {}

    module_n, module_source = _resolve_module(spec, teeth, beta, measured)
    if module_n is None:
        return {}

    angle_match = _find_spec_value(spec, "pressure_angle") or _find_spec_value(
        spec, "normal_pressure_angle"
    )
    alpha_n = angle_match[1] if angle_match is not None else DEFAULT_PRESSURE_ANGLE_RAD

    derived = derive_params(
        module_n=module_n,
        teeth=teeth,
        helix_angle_rad=beta,
        normal_pressure_angle_rad=alpha_n,
        outer_d=outer_d,
        root_d=root_d,
    )

    if outer_d is not None and root_d is not None:
        ref = (
            f"helical_gear.derived_from_cad(outer_d={outer_d:.3f}, "
            f"root_d={root_d:.3f}, z={teeth}, m_n={module_n:g} [{module_source}], "
            f"β={math.degrees(beta):.1f}°, α_n={math.degrees(alpha_n):.1f}°)"
        )
    else:
        ref = (
            f"helical_gear.derived_from_spec(z={teeth}, m_n={module_n:g} "
            f"[{module_source}], β={math.degrees(beta):.1f}°, "
            f"α_n={math.degrees(alpha_n):.1f}°)"
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


def _measure_helix_hand_from_fcstd(fcstd_path: str) -> str | None:
    """Open ``fcstd_path``, locate the gear's tip cylinder, and infer the
    helix hand by majority vote over edges that span ``z_min`` to ``z_max``
    (or shorter axial strips, when the gear was built by loft instead of
    sweep) along the tip cylinder.

    Algorithm:
      1. Open the PartDesign Body's tip shape; let ``r_tip`` be half of
         the maximum X/Y bbox extent (the gear's tip diameter).
      2. Find every 2-vertex edge whose endpoints both sit near
         ``r_tip`` and lie at different ``z`` (i.e. tip-line edges of
         the gear flank).
      3. For each such edge, compute ``sign(Δθ × Δz)``. Right-hand → +,
         left-hand → −. Take the majority vote.

    Returns "right" / "left" / "spur" / None (couldn't measure).
    Imports FreeCAD lazily so the category file stays importable in
    environments without FreeCAD (tests, CLIs, distribution checks).
    """
    try:
        import FreeCAD  # type: ignore
    except ImportError:
        return None

    try:
        doc = FreeCAD.openDocument(fcstd_path)
    except Exception:
        return None
    try:
        bodies = [o for o in doc.Objects if str(getattr(o, "TypeId", "")) == "PartDesign::Body"]
        body = next(
            (
                b
                for b in bodies
                if getattr(b, "Shape", None) is not None
                and not b.Shape.isNull()
                and float(getattr(b.Shape, "Volume", 0.0) or 0.0) > 0.0
            ),
            None,
        )
        if body is None:
            return None
        shape = body.Shape
        bb = shape.BoundBox
        r_tip = max(bb.XLength, bb.YLength) / 2.0
        if r_tip <= 0.0:
            return None

        right_votes = 0
        left_votes = 0
        for e in shape.Edges:
            verts = e.Vertexes
            if len(verts) != 2:
                continue
            a, b = verts[0], verts[1]
            if abs(math.hypot(a.X, a.Y) - r_tip) > 1.0:
                continue
            if abs(math.hypot(b.X, b.Y) - r_tip) > 1.0:
                continue
            dz = b.Z - a.Z
            if abs(dz) < 0.5:
                continue
            dtheta = (math.atan2(b.Y, b.X) - math.atan2(a.Y, a.X) + math.pi) % (
                2.0 * math.pi
            ) - math.pi
            if abs(dtheta) < 1e-6:
                continue
            if dtheta * dz > 0:
                right_votes += 1
            else:
                left_votes += 1

        if right_votes == 0 and left_votes == 0:
            return None
        return "right" if right_votes > left_votes else "left"
    finally:
        FreeCAD.closeDocument(doc.Name)


def _apply_helix_hand_check(
    report,
    spec: StructuredSpec,
    tol_scalar: float,
) -> None:
    """Compare ``spec.strings['helix_hand']`` against the FCStd-measured
    hand and append a consistent/inconsistent finding to ``report``.

    No-op when the spec doesn't declare ``helix_hand``, or when the
    measurement returns None (FCStd unreadable, no tip vertices, etc).
    """
    spec_hand = (
        (spec.strings.get("helix_hand") or "").strip().lower() if hasattr(spec, "strings") else ""
    )
    if spec_hand not in ("left", "right"):
        return

    measured_hand = _measure_helix_hand_from_fcstd(report.fcstd_path)
    if measured_hand is None:
        return

    del tol_scalar  # exact string match; no tolerance

    from freecad_validator.consistency.compare import (
        make_consistent_finding,
        make_inconsistent_finding,
    )

    if measured_hand == spec_hand:
        report.consistent.append(
            make_consistent_finding(
                param="helix_hand",
                spec_value=spec_hand,
                measured_value=measured_hand,
                unit="",
                feature="helical_gear.measured_hand(tip-vertex Δθ×Δz sign)",
            )
        )
    else:
        report.inconsistent.append(
            make_inconsistent_finding(
                param="helix_hand",
                spec_value=spec_hand,
                measured_value=measured_hand,
                unit="",
                feature="helical_gear.measured_hand(tip-vertex Δθ×Δz sign)",
                rel_diff=1.0,
                reason=(
                    f"helix hand mismatch: spec says {spec_hand!r}, CAD twists "
                    f"{measured_hand!r} (sign of Δθ × Δz at tip-cylinder vertices)"
                ),
            )
        )


class HelicalGearCategory(Category):
    name = "helical_gear"

    def derived_candidates(
        self,
        bank: MeasurementBank,
        spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)

    def apply(self, report, bank, spec, tol_scalar) -> None:
        # Numeric derivations (m_n, pitch_d, outer_d, …) → reclassify
        # via the base-class machinery.
        super().apply(report, bank, spec, tol_scalar)
        # String-valued helix_hand needs a separate path because the
        # base reclassify logic only handles float-valued findings.
        _apply_helix_hand_check(report, spec, tol_scalar)
