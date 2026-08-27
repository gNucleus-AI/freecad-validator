"""Measurement extraction against real FreeCAD geometry.

Covers builder/extractors/detectors, which the FreeCAD-free suite
cannot reach: every value here comes from opening a real document and
interrogating its shape.
"""

from __future__ import annotations

import pytest

from freecad_validator.measurement.builder import extract

pytestmark = pytest.mark.needs_freecad


def test_box_globals_match_known_geometry(box_10x5x3):
    """A 10x5x3 box has volume 150 and surface area 190 — arithmetic the
    extractor must reproduce exactly, not approximately."""
    bank = extract(str(box_10x5x3))
    assert bank.globals["volume"].value == pytest.approx(150.0)
    assert bank.globals["area"].value == pytest.approx(2 * (10 * 5 + 10 * 3 + 5 * 3))


def test_box_is_a_single_solid(box_10x5x3):
    assert extract(str(box_10x5x3)).solid_count == 1


def test_box_plane_pairs_are_the_three_dimensions(box_10x5x3):
    """Opposite faces of a box are parallel pairs 10, 5 and 3 apart."""
    offsets = sorted(round(pp.offset, 6) for pp in extract(str(box_10x5x3)).plane_pairs)
    assert offsets == [3.0, 5.0, 10.0]


def test_box_has_no_cylindrical_clusters(box_10x5x3):
    """A box has no curved faces, so nothing should be detected as one."""
    assert extract(str(box_10x5x3)).cylinder_clusters == []


def test_cylinder_radius_is_detected(cylinder_r4_h12):
    """The curved face of an r=4 cylinder must surface as a cluster at
    that radius — the detector path a box never exercises."""
    bank = extract(str(cylinder_r4_h12))
    assert any(c.radius == pytest.approx(4.0) for c in bank.cylinder_clusters), (
        f"no r=4 cluster in {[c.radius for c in bank.cylinder_clusters]}"
    )


def test_cylinder_volume_matches_formula(cylinder_r4_h12):
    import math

    bank = extract(str(cylinder_r4_h12))
    assert bank.globals["volume"].value == pytest.approx(math.pi * 4**2 * 12, rel=1e-6)


def test_feature_tree_records_the_parametric_features(box_10x5x3):
    """Extraction must see the PartDesign feature tree, not just the
    final shape — that is what makes the spec pass parametric."""
    entries = extract(str(box_10x5x3)).feature_tree
    assert entries, "feature tree is empty"
    assert any("AdditiveBox" in e.type_id for e in entries)


def test_feature_tree_carries_named_dimensions(box_10x5x3):
    """The box's driving parameters are readable off the feature tree."""
    props = {}
    for entry in extract(str(box_10x5x3)).feature_tree:
        props.update(entry.properties)
    assert props.get("Length") == pytest.approx(10.0)
    assert props.get("Width") == pytest.approx(5.0)
    assert props.get("Height") == pytest.approx(3.0)


def test_extraction_is_deterministic(box_10x5x3):
    """Two extractions of one unchanged document agree exactly. Guards
    against ordering or float instability leaking into scores."""
    a, b = extract(str(box_10x5x3)), extract(str(box_10x5x3))
    assert a.model_dump() == b.model_dump()
