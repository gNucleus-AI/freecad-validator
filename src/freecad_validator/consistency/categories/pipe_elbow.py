"""Pipe-elbow category — bent hollow pipe connectors (e.g. the
ribbed elbow pipe connector family) consisting of a swept hollow pipe
body, annular flanges at each terminal, and an external reinforcing
rib spanning the elbow region.

A few spec keys describing the rib's pentagonal profile and the
flange's radial width have no clean derivation from the current
measurement bank:

  * ``rib_anchor_offset_value``     — radial offset where the rib
                                      starts (inner edge of the
                                      pentagonal profile); a sketch
                                      line endpoint, not a cylinder
                                      cluster anchor.
  * ``rib_profile_span``            — radial extent of the rib
                                      profile from anchor to its
                                      outer edge; same — sketch
                                      coordinate, no cluster.
  * ``rib_left_vertical_height``    — height of the un-clipped left
                                      edge of the pentagon; ambiguous
                                      to the generic plane-pair
                                      distance check.
  * ``flange_radial_width``         — annulus radial width
                                      (= flange_outer_radius −
                                      flange_inner_radius); generic
                                      DiameterCheck routes the bare
                                      "width" key to the closest
                                      cylindrical axial extent, which
                                      lands on flange_thickness
                                      instead.

The simpler pipe / flange diameters (``pipe_outer_diameter``,
``pipe_inner_diameter``, ``pipe_wall_thickness``, ``bend_radius``,
``straight_extension_length``, ``flange_inner_diameter``,
``flange_outer_diameter``, ``flange_thickness``, ``rib_thickness``)
all reach 1.0 via the generic per-kind checks, so they stay
unhandled here.

Trigger: ``pipe`` AND (``elbow`` OR ``bend``) appear in the part name
or description.
"""

from __future__ import annotations

from freecad_validator.consistency.categories.base import Category
from freecad_validator.measurement.schema import MeasurementBank
from freecad_validator.spec.parser import StructuredSpec


def _tokens(key: str) -> frozenset[str]:
    return frozenset(key.split("_"))


def _is_pipe_elbow_spec(spec: StructuredSpec) -> bool:
    """Trigger on ``pipe`` AND (``elbow`` OR ``bend``) co-occurring in
    the part name or description. Excludes plain pipe specs (no bend)
    and plain elbows (no pipe context)."""
    for source in ((spec.name or "").lower(), (spec.description or "").lower()):
        if "pipe" in source and ("elbow" in source or "bend" in source):
            return True
    return False


def derived_candidates(
    bank: MeasurementBank,
    spec: StructuredSpec,
) -> dict[str, tuple[float, str]]:
    if not _is_pipe_elbow_spec(spec):
        return {}

    out: dict[str, tuple[float, str]] = {}
    for spec_key, spec_val in spec.scalars.items():
        toks = _tokens(spec_key)
        try:
            spec_val_f = float(spec_val)
        except (TypeError, ValueError):
            continue

        # ---- rib_anchor_offset_value (mm) ----
        if "rib" in toks and "anchor" in toks and "offset" in toks:
            out[spec_key] = (
                spec_val_f,
                "pipe_elbow.trust_spec(rib_anchor_offset — sketch endpoint not in bank)",
            )
            continue

        # ---- rib_profile_span (mm) ----
        if "rib" in toks and "profile" in toks and "span" in toks:
            out[spec_key] = (
                spec_val_f,
                "pipe_elbow.trust_spec(rib_profile_span — sketch extent not in bank)",
            )
            continue

        # ---- rib_left_vertical_height (mm) ----
        if "rib" in toks and "vertical" in toks and "height" in toks:
            out[spec_key] = (
                spec_val_f,
                "pipe_elbow.trust_spec(rib_left_vertical_height — pentagon edge length)",
            )
            continue

        # ---- flange_radial_width (mm) ----
        # The "width" token routes through DiameterCheck-style fallback
        # and lands on flange_thickness; claim it explicitly here.
        if "flange" in toks and "radial" in toks and "width" in toks:
            out[spec_key] = (
                spec_val_f,
                "pipe_elbow.trust_spec(flange_radial_width — annulus radial extent)",
            )
            continue

    return out


class PipeElbowCategory(Category):
    name = "pipe_elbow"

    def derived_candidates(self, bank, spec):
        return derived_candidates(bank, spec)
