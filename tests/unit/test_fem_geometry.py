"""Real topology regression for Boolean Fragments without face subdivisions."""

import pytest

try:
    import FreeCAD
    import Part
except ImportError:
    pytest.skip("FreeCAD is required for geometry extraction tests", allow_module_level=True)

from BOPTools import SplitAPI

from freecad_validator.fem.geometry import geometry_facts
from freecad_validator.fem.step_interface import (
    ExtractionError,
    _boolean_gate_report,
    _preprocessing_gate_report,
)
from freecad_validator.fem.topology_compare import container_mismatches

pytestmark = pytest.mark.needs_freecad


def test_shared_interfaces_distinguish_fragments_from_raw_compound(tmp_path):
    left = Part.makeBox(1, 1, 1)
    right = Part.makeBox(1, 1, 1, FreeCAD.Vector(1, 0, 0))
    source = geometry_facts([Part.makeCompound([left, right])])
    fragments, _ = left.generalFuse([right])
    # Persist and reload, as the scorer does; topology sharing must survive.
    brep = tmp_path / "fragments.brep"
    fragments.exportBrep(str(brep))
    reference = geometry_facts([Part.read(str(brep))])

    assert source["regions"] == reference["regions"]
    assert source["num_solids"] == reference["num_solids"] == 2
    assert source["shape_types"] == reference["shape_types"] == ["Compound"]
    assert (source["num_faces"], source["num_edges"]) == (12, 24)
    assert (reference["num_faces"], reference["num_edges"]) == (11, 20)
    assert _boolean_gate_report(source, reference, reference) is None
    assert _boolean_gate_report(source, reference, source).overall_score == 0


def test_single_solid_passthrough_still_cannot_satisfy_boolean_requirement():
    source = geometry_facts([Part.makeBox(1, 1, 1)])
    assert (source["num_faces"], source["num_edges"]) == (6, 12)
    with pytest.raises(ExtractionError, match="indistinguishable"):
        _boolean_gate_report(source, source, source)


@pytest.mark.parametrize("wrapper_depth", [1, 3])
def test_compsolid_wrappers_pass_boolean_gate(tmp_path, wrapper_depth):
    left = Part.makeBox(1, 1, 1)
    right = Part.makeBox(1, 1, 1, FreeCAD.Vector(1, 0, 0))
    source = geometry_facts([Part.makeCompound([left, right])])
    fragments = SplitAPI.booleanFragments([left, right], "CompSolid")
    assert fragments.ShapeType == "Compound"
    bare = fragments.childShapes()[0]
    assert bare.ShapeType == "CompSolid"
    for _ in range(wrapper_depth - 1):
        fragments = Part.makeCompound([fragments])
    brep = tmp_path / "wrapped.brep"
    fragments.exportBrep(str(brep))
    reference = geometry_facts([bare])
    candidate = geometry_facts([Part.read(str(brep))])

    assert candidate == reference
    for label, submission in ((reference, candidate), (candidate, reference)):
        assert _boolean_gate_report(source, label, submission) is None


def test_wrapping_raw_geometry_does_not_satisfy_boolean_requirement():
    left = Part.makeBox(1, 1, 1)
    right = Part.makeBox(1, 1, 1, FreeCAD.Vector(1, 0, 0))
    raw = Part.makeCompound([left, right])
    source = geometry_facts([raw])
    fragments, _ = left.generalFuse([right])
    reference = geometry_facts([fragments])
    candidate = geometry_facts([Part.makeCompound([raw])])

    assert candidate == source
    assert _boolean_gate_report(source, reference, candidate).overall_score == 0
    assert _preprocessing_gate_report(source, candidate).gates_triggered == [
        {"reason": "PREPROCESSING_NOT_PERFORMED"}
    ]


def test_real_container_and_region_differences_are_preserved():
    left = Part.makeBox(1, 1, 1)
    right = Part.makeBox(1, 1, 1, FreeCAD.Vector(1, 0, 0))
    fragments = SplitAPI.booleanFragments([left, right], "CompSolid")
    reference = geometry_facts([fragments])
    plain_compound = geometry_facts([Part.makeCompound(fragments.Solids)])
    extra_solid = geometry_facts(
        [Part.makeCompound([fragments, Part.makeBox(1, 1, 1, FreeCAD.Vector(5, 0, 0))])]
    )

    assert plain_compound["shape_types"] == extra_solid["shape_types"] == ["Compound"]
    assert plain_compound["num_compsolids"] == 0
    assert extra_solid["num_solids"] == 3
    assert any("num_compsolids" in item for item in container_mismatches(reference, plain_compound))
    assert any("regions" in item for item in container_mismatches(reference, extra_solid))


def test_unwrapping_preserves_world_placement():
    solid = Part.makeBox(1, 2, 3)
    solid.Placement = FreeCAD.Placement(
        FreeCAD.Vector(2, 3, 4), FreeCAD.Rotation(FreeCAD.Vector(1, 0, 0), 20)
    )
    placement = FreeCAD.Placement(
        FreeCAD.Vector(4, 5, 6), FreeCAD.Rotation(FreeCAD.Vector(0, 0, 1), 30)
    )
    wrapped = Part.makeCompound([Part.makeCompound([solid])])
    wrapped.Placement = placement
    moved = solid.copy()
    moved.Placement = placement.multiply(solid.Placement)
    anchor = Part.makeBox(1, 1, 1, FreeCAD.Vector(-10, -10, -10))

    # The anchor makes aggregate bounds sensitive to lost translations as well
    # as rotations; a single shape's bounding-box lengths ignore translation.
    actual = geometry_facts([wrapped, anchor])
    expected = geometry_facts([moved, anchor])
    # Equivalent OCCT transformations can differ by roundoff across platforms.
    # Keep topology exact and allow only tight tolerances on measurements.
    for key in ("volume_mm3", "surface_area_mm2", "characteristic_length_mm", "bbox_mm"):
        assert actual.pop(key) == pytest.approx(expected.pop(key), rel=1e-12, abs=1e-12)
    actual_regions = actual.pop("regions")
    expected_regions = expected.pop("regions")
    assert len(actual_regions) == len(expected_regions)
    for actual_region, expected_region in zip(actual_regions, expected_regions, strict=True):
        for key in ("volume_mm3", "surface_area_mm2"):
            assert actual_region.pop(key) == pytest.approx(
                expected_region.pop(key), rel=1e-12, abs=1e-12
            )
        assert actual_region == expected_region
    assert actual == expected
    assert wrapped.ShapeType == "Compound"
