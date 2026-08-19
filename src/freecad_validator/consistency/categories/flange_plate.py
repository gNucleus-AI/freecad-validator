"""Flange-plate category helpers.

Handles square / rectangular flange-mount specs whose key dimensions
are plate-and-lug geometry. Observed in the reference corpus across
``square_flange_mount/``:

    base_plate_width        plate_width        plate_height
    base_plate_height       lug_side_edge_length
    bolt_hole_quantity      number_of_bolt_holes

Geometric anchors used for derivation:

    plate width / height  → the two larger components of ``aabb_sorted``
                            (the smallest is the plate's thickness)
    lug edge length       → the ``LineLength`` value in any sketch that
                            best matches the spec — lug profiles repeat
                            the same edge length across the N-fold
                            circular pattern (4× for square mounts)
    bolt-hole count       → the smallest-radius concave cylinder cluster

Trigger: any spec key contains one of the token groups ``{plate}`` or
``{lug}``. The narrower spec language for square_flange_mount uses both,
so either one is sufficient.
"""
from __future__ import annotations

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec

# Tokens that mark a spec as flange-plate-related. Any one triggers.
_TRIGGER_TOKENS: frozenset[str] = frozenset({"plate", "lug"})

# Tokens that resolve a key to a plate in-plane dimension. We treat
# width / length / height the same way — for square plates they're
# equal anyway; for rectangular plates the difference between aabb's
# 2nd- and 1st-largest axis is what distinguishes them.
_PLATE_DIM_TOKENS: frozenset[str] = frozenset({"width", "length", "height", "size"})
_PLATE_THICK_TOKENS: frozenset[str] = frozenset({"thickness", "thick"})


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_flange_plate_spec(spec: StructuredSpec) -> bool:
    for source in (spec.scalars, spec.counts):
        for key in source:
            if _TRIGGER_TOKENS & _tokens(key):
                return True
    return False


def _classify_plate_key(key: str) -> str | None:
    """Return 'width', 'height', 'thickness', 'lug_edge', 'bolt_count' or None."""
    toks = _tokens(key)
    if "lug" in toks:
        # Any lug-side / lug-edge / lug_side_edge_length etc.
        if {"side", "edge"} & toks or "length" in toks:
            return "lug_edge"
        return None
    if "bolt" in toks and "hole" in toks:
        # number_of_bolt_holes, bolt_hole_quantity, etc.
        if {"quantity", "count", "number", "holes", "of"} & toks:
            return "bolt_count"
        return None
    if "plate" in toks:
        if _PLATE_THICK_TOKENS & toks:
            return "thickness"
        if _PLATE_DIM_TOKENS & toks:
            # width and height map to different aabb axes when distinguishable;
            # we tag both 'plate_dim' and let the deriver hand out the right axis.
            if "height" in toks:
                return "plate_height"
            return "plate_width"
    return None


def _aabb_sorted(bank: MeasurementBank) -> tuple[float, float, float] | None:
    """Return (min, mid, max) ascending from the global aabb_sorted measurement,
    or None if not present."""
    g = bank.globals.get("aabb_sorted")
    if g is None:
        return None
    val = g.value
    if not isinstance(val, tuple) or len(val) != 3:
        return None
    return tuple(sorted(float(x) for x in val))  # type: ignore[return-value]


def _best_line_length_match(
    bank: MeasurementBank, target: float, tol: float = 0.5,
) -> tuple[float, str] | None:
    """Find the sketch LineLength property whose value is closest to `target`.
    Returns (value, "<sketch_name>.<prop_key>") or None if no LineLength
    entry comes within `tol` (relative)."""
    best: tuple[float, str, float] | None = None  # (value, ref, abs_err)
    for ft in bank.feature_tree:
        for prop_key, val in ft.properties.items():
            if "LineLength" not in prop_key:
                continue
            err = abs(val - target)
            if best is None or err < best[2]:
                best = (val, f"{ft.name}.{prop_key}", err)
    if best is None:
        return None
    val, ref, err = best
    if target > 0 and err / target > tol:
        return None
    return val, ref


def _bolt_hole_count(bank: MeasurementBank) -> tuple[int, str] | None:
    """Pick the cluster that looks like the bolt-hole pattern: concave
    (a hole), count >= 2, and the smallest such radius (bolts are usually
    the smallest holes — the central bore is concave too but bigger and
    has count=1)."""
    holes = [c for c in bank.cylinder_clusters if not c.convex and c.count > 1]
    if not holes:
        return None
    # Smallest-radius wins; ties broken by highest count.
    holes.sort(key=lambda c: (c.radius, -c.count))
    return holes[0].count, f"{holes[0].id}.count"


def derived_candidates(
    bank: MeasurementBank, spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    """Return ``{spec_key: (value, feature_ref)}`` for every flange-plate
    spec param the category can derive. Empty if the spec isn't
    flange-plate-related."""
    if not _is_flange_plate_spec(spec):
        return {}

    aabb = _aabb_sorted(bank)
    out: dict[str, tuple[float, str]] = {}

    for source in (spec.scalars, spec.counts):
        for spec_key, spec_val in source.items():
            kind = _classify_plate_key(spec_key)
            if kind is None:
                continue

            if kind in ("plate_width", "plate_height", "thickness"):
                if aabb is None:
                    continue
                # aabb sorted ascending: [thickness, mid, max]
                if kind == "thickness":
                    out[spec_key] = (aabb[0], "flange_plate.aabb[min]")
                elif kind == "plate_width":
                    out[spec_key] = (aabb[2], "flange_plate.aabb[max]")
                else:  # plate_height
                    out[spec_key] = (aabb[1], "flange_plate.aabb[mid]")

            elif kind == "lug_edge":
                hit = _best_line_length_match(bank, float(spec_val))
                if hit is not None:
                    val, ref = hit
                    out[spec_key] = (val, f"flange_plate.lug_edge({ref})")

            elif kind == "bolt_count":
                hit = _bolt_hole_count(bank)
                if hit is not None:
                    n, ref = hit
                    out[spec_key] = (float(n), f"flange_plate.bolt_holes({ref})")

    return out


# ---------------------------------------------------------------------------
# Category subclass.
# ---------------------------------------------------------------------------


class FlangePlateCategory(Category):
    name = "flange_plate"

    def derived_candidates(
        self, bank: MeasurementBank, spec: StructuredSpec,
    ) -> dict[str, tuple[float, str]]:
        return derived_candidates(bank, spec)
