"""FCStd → MeasurementBank.

Opens a FreeCAD document, picks the representative solid, and builds a
generic, category-agnostic bank of measurements the classifier can
pull from:

  - globals (volume, area, AABB, OBB-via-PCA, centroid)
  - face stats (counts by surface type)
  - cylinder clusters (grouped by radius + axis + convex/concave)
  - parallel plane-face pairs (thickness candidates)
  - feature tree (named scalar properties + sketch Geometry reads)
  - linear-pattern and circular-pattern detection (derived from above)

Per-edge primitives are deferred — not yet needed by the classifier
paths exercised on the current test cases.

Env requirement: FreeCAD importable (set FREECAD_LIB / PYTHONPATH via
`source dev_setup.sh` at the repo root).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

# ``FreeCAD`` is imported lazily inside ``extract()`` so the package
# can be imported on hosts that haven't installed FreeCAD yet — the
# import error only surfaces when the user actually tries to score
# a case.

from .common import pick_representative_shape
from .detectors import DEFAULT_BANK_DETECTORS, BankDetector
from .extractors import DEFAULT_SHAPE_EXTRACTORS, ShapeExtractor
from .schema import MeasurementBank

log = logging.getLogger(__name__)


class MeasurementBankBuilder:
    """Runs a configured set of extractors and detectors over one document."""

    def __init__(
        self,
        extractors: Optional[List[ShapeExtractor]] = None,
        detectors: Optional[List[BankDetector]] = None,
    ):
        self.extractors = list(extractors) if extractors is not None else list(DEFAULT_SHAPE_EXTRACTORS)
        self.detectors = list(detectors) if detectors is not None else list(DEFAULT_BANK_DETECTORS)

    def build(self, doc) -> MeasurementBank:
        shape, chosen = pick_representative_shape(doc)
        if shape is None:
            return MeasurementBank(solid_count=0)
        owner_label = getattr(chosen, "Label", None) if chosen is not None else None
        bank = MeasurementBank(
            solid_count=len(getattr(shape, "Solids", [])) or 1,
        )
        for extractor in self.extractors:
            extractor.extract(shape=shape, doc=doc, owner_label=owner_label, bank=bank)
        for detector in self.detectors:
            detector.detect(bank=bank)
        return bank


def extract(fcstd_path: str | Path) -> MeasurementBank:
    """Open `fcstd_path`, recompute, build the bank, close the doc.

    Raises whatever FreeCAD raises on malformed input; the caller is
    responsible for catching and mapping that to the report's top-level
    `error` field.

    ``FreeCAD`` is imported lazily here so that callers can ``import
    freecad_validator`` without FreeCAD on the path; the import error
    is deferred to the moment a real ``.FCStd`` actually needs to be
    opened. ``import_freecad`` auto-detects common install locations
    (macOS Homebrew, apt, conda) so users typically don't need to
    set ``PYTHONPATH`` manually.
    """
    from freecad_validator._freecad_loader import import_freecad
    FreeCAD = import_freecad()

    fcstd_path = str(fcstd_path)
    doc = FreeCAD.openDocument(fcstd_path)
    try:
        doc.recompute()
        return MeasurementBankBuilder().build(doc)
    finally:
        FreeCAD.closeDocument(doc.Name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _format_value(v) -> str:
    if isinstance(v, tuple):
        return "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fcstd", type=Path, help="Path to a .FCStd file.")
    parser.add_argument("--json", action="store_true", help="Emit the bank as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bank = extract(args.fcstd)
    if args.json:
        log.info(bank.model_dump_json(indent=2))
        return 0

    log.info("solid_count: %d", bank.solid_count)
    log.info("globals:")
    for k, m in bank.globals.items():
        log.info("  %-12s = %s %s", k, _format_value(m.value), m.unit)
    log.info("face_stats:  %s", dict(sorted(bank.face_stats.items())))
    log.info("cylinder_clusters (%d):", len(bank.cylinder_clusters))
    for c in bank.cylinder_clusters:
        log.info(
            "  %s: count=%d radius=%.3f axis=%s %s axial_extent=%.3f",
            c.id, c.count, c.radius, _format_value(c.axis),
            "convex" if c.convex else "concave", c.axial_extent,
        )
    log.info("plane_pairs (%d):", len(bank.plane_pairs))
    for p in bank.plane_pairs:
        log.info(
            "  %s: offset=%.3f normal=%s min_area=%.3f",
            p.id, p.offset, _format_value(p.normal), p.min_area,
        )
    log.info("linear_pattern (%d):", len(bank.grids))
    for g in bank.grids:
        log.info(
            "  %s: %dx%d (%d pts) spacing=(%.3f, %.3f) origin=%s source=%s",
            g.id, g.rows, g.columns, g.count,
            g.spacing_rows, g.spacing_cols, _format_value(g.origin), g.source,
        )
    log.info("circular_patterns (%d):", len(bank.circular_patterns))
    for cp in bank.circular_patterns:
        log.info(
            "  %s: %d-fold r=%.3f pitch=%.4f rad center=%s axis=%s source=%s",
            cp.id, cp.count, cp.pattern_radius, cp.angular_pitch,
            _format_value(cp.center), _format_value(cp.axis), cp.source,
        )
    log.info("feature_tree (%d):", len(bank.feature_tree))
    for e in bank.feature_tree:
        props_str = ", ".join(f"{k}={_format_value(v)}" for k, v in e.properties.items())
        vecs_str = ", ".join(f"{k}={_format_value(v)}" for k, v in e.vectors.items())
        details = "; ".join(s for s in (props_str, vecs_str) if s)
        log.info("  %-18s (%-28s) %s", e.name, e.type_id, details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
