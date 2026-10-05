"""Geometry regressions; run with pytest in the FEM FreeCAD Python environment."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_freecad

FreeCAD = pytest.importorskip("FreeCAD")
if not getattr(FreeCAD, "__file__", None):
    pytest.skip("Requires real FreeCAD bindings", allow_module_level=True)
pytest.importorskip("manifold3d")
import Part  # noqa: E402

from freecad_validator.fem.pre_process import scorer as scorer_module  # noqa: E402
from freecad_validator.fem.pre_process.errors import EvaluationError  # noqa: E402
from freecad_validator.fem.pre_process.geometry import (  # noqa: E402
    export_geometry,
    read_geometry,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (  # noqa: E402
    DiffConfig,
)
from freecad_validator.fem.pre_process.scorer import PreProcessScorer  # noqa: E402


@pytest.fixture
def scorer():
    return PreProcessScorer(DiffConfig(region_sample_count=1024))


@pytest.fixture
def hole_case(tmp_path):
    box = Part.makeBox(10, 10, 10)
    first = Part.makeCylinder(1.5, 10, FreeCAD.Vector(3, 3, 0))
    second = Part.makeCylinder(1.5, 10, FreeCAD.Vector(7, 7, 0))
    shapes = {
        "raw": box.cut(first).cut(second),
        "reference": box.cut(second),
        "wrong": box.cut(first),
        "extra": box,
    }
    paths = {}
    for name, shape in shapes.items():
        paths[name] = tmp_path / f"{name}.FCStd"
        export_geometry(shape, paths[name])
    return paths


def test_reference_and_noop(scorer, hole_case):
    raw, reference = hole_case["raw"], hole_case["reference"]
    assert scorer.score(raw, reference, reference) == 1
    assert scorer.score(raw, raw, raw) is None
    result = scorer.score_detailed(raw, reference, raw)
    assert result.score == 0
    assert set(result.subscores) == {
        "global_geometry",
        "pointcloud_chamfer",
        "region_edit_overlap",
    }


def test_wrong_edit_despite_equal_volume_and_area(scorer, hole_case):
    result = scorer.score_detailed(
        hole_case["raw"],
        hole_case["reference"],
        hole_case["wrong"],
    )
    assert result.subscores["global_geometry"] == pytest.approx(1)
    assert result.subscores["region_edit_overlap"] == 0
    assert result.score == 0


def test_extra_filled_hole_has_lower_precision(scorer, hole_case):
    result = scorer.score_detailed(
        hole_case["raw"],
        hole_case["reference"],
        hole_case["extra"],
    )
    region = result.details["component_subscores"]["region_edit_overlap"]
    assert 0.3 < region["precision"] < 0.7
    assert region["recall"] == 1
    assert result.score == 0
    assert 0 < result.details["geometry_score"] < 1


def test_compound_partitions_and_split_faces_do_not_reduce_score(scorer, tmp_path):
    box = Part.makeBox(10, 10, 10)
    split = Part.makeCompound(
        [
            Part.makeBox(5, 10, 10),
            Part.makeBox(5, 10, 10, FreeCAD.Vector(5, 0, 0)),
        ]
    )
    reference, candidate = tmp_path / "reference.FCStd", tmp_path / "candidate.FCStd"
    export_geometry(box, reference)
    export_geometry(split, candidate)
    raw = tmp_path / "raw.FCStd"
    export_geometry(box.cut(Part.makeCylinder(1, 10, FreeCAD.Vector(3, 3, 0))), raw)
    assert len(split.Faces) != len(box.Faces)
    assert scorer.score(raw, reference, candidate) == 1
    assert scorer.score(reference, candidate, reference) is None


def test_cylinder_seam_rotation_is_not_a_geometry_error(scorer, tmp_path):
    cylinder = Part.makeCylinder(5, 10)
    rotated = cylinder.copy()
    rotated.rotate(FreeCAD.Vector(), FreeCAD.Vector(0, 0, 1), 43)
    reference, candidate = tmp_path / "reference.FCStd", tmp_path / "candidate.FCStd"
    export_geometry(cylinder, reference)
    export_geometry(rotated, candidate)
    raw = tmp_path / "raw.FCStd"
    export_geometry(cylinder.cut(Part.makeCylinder(1, 10)), raw)
    assert scorer.score(raw, reference, candidate) == 1
    assert scorer.score(reference, candidate, reference) is None


def test_step_all_solids_and_world_placement(tmp_path):
    compound = Part.makeCompound(
        [
            Part.makeBox(2, 2, 2),
            Part.makeBox(2, 2, 2, FreeCAD.Vector(10, 0, 0)),
        ]
    )
    step = tmp_path / "assembly.step"
    compound.exportStep(str(step))
    assert read_geometry(step).Volume == pytest.approx(16)

    path = tmp_path / "placement.FCStd"
    doc = FreeCAD.newDocument("PlacementTest")
    try:
        parent = doc.addObject("App::Part", "Assembly")
        parent.Placement.Base = FreeCAD.Vector(100, 0, 0)
        obj = doc.addObject("Part::Feature", "Clean")
        obj.Shape = Part.makeBox(2, 2, 2)
        obj.Placement.Base = FreeCAD.Vector(3, 0, 0)
        parent.addObject(obj)
        doc.recompute()
        doc.saveAs(str(path))
    finally:
        FreeCAD.closeDocument(doc.Name)
    assert read_geometry(path).BoundBox.XMin == pytest.approx(103)


def test_ambiguous_document_requires_selection(tmp_path):
    path = tmp_path / "ambiguous.FCStd"
    doc = FreeCAD.newDocument("SelectionTest")
    try:
        for name, length in (("Raw", 10), ("Clean", 12)):
            obj = doc.addObject("Part::Feature", name)
            obj.Shape = Part.makeBox(length, 10, 10)
        doc.recompute()
        doc.saveAs(str(path))
    finally:
        FreeCAD.closeDocument(doc.Name)
    with pytest.raises(EvaluationError, match="specify an object"):
        read_geometry(path)
    assert read_geometry(path, "Clean").Volume == pytest.approx(1200)


@pytest.mark.parametrize("link_parent", [False, True])
def test_mesh_link_selects_clean_instead_of_raw(tmp_path, link_parent):
    path = tmp_path / "analysis.FCStd"
    doc = FreeCAD.newDocument("MeshSelectionTest")
    try:
        raw = doc.addObject("Part::Feature", "Raw")
        raw.Shape = Part.makeBox(10, 10, 10)
        clean = doc.addObject("Part::Feature", "Clean")
        clean.Shape = Part.makeBox(12, 10, 10)
        parent = doc.addObject("App::Part", "Assembly")
        parent.Placement.Base = FreeCAD.Vector(100, 0, 0)
        parent.addObject(clean)
        mesh = doc.addObject("Fem::FemMeshShapeNetgenObject", "Mesh")
        mesh.Shape = parent if link_parent else clean
        doc.recompute()
        doc.saveAs(str(path))
    finally:
        FreeCAD.closeDocument(doc.Name)
    assert read_geometry(path).Volume == pytest.approx(1200)
    assert read_geometry(path).BoundBox.XMin == pytest.approx(100)


def test_input_error_is_not_a_zero_score(scorer, hole_case):
    with pytest.raises(EvaluationError, match="does not exist"):
        scorer.score(Path("/nonexistent/raw.step"), hole_case["reference"], hole_case["raw"])


def test_repeated_path_reads_updated_geometry(scorer, hole_case):
    raw, reference = hole_case["raw"], hole_case["reference"]
    assert scorer.score(raw, reference, reference) == 1
    export_geometry(read_geometry(raw), reference)
    assert scorer.score(raw, hole_case["extra"], reference) == 0


def test_missing_boolean_dependency_is_an_evaluation_error(monkeypatch):
    monkeypatch.setattr(scorer_module, "find_spec", lambda name: None)
    with pytest.raises(EvaluationError, match="requires manifold3d"):
        PreProcessScorer()


def test_unchanged_reference_still_penalizes_changed_answer(scorer, hole_case):
    raw = hole_case["raw"]
    result = scorer.score_detailed(raw, raw, hole_case["extra"])
    assert result.score == 0
    assert result.details["extra_change"]
    assert not result.details["reference_changed"]
    with pytest.raises(EvaluationError, match="does not exist"):
        scorer.score_detailed(raw, raw, "/nonexistent/candidate.FCStd")


def test_whole_body_deletions(scorer, hole_case):
    raw = hole_case["raw"]
    assert scorer.score(raw, None, None) == 1
    assert scorer.score(raw, None, raw) == 0
    unwanted = scorer.score_detailed(raw, raw, None)
    assert unwanted.details["extra_change"]
