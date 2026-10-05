"""Automatic mapping ignores labels and Boolean region counts, not geometry edits."""

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.needs_freecad

FreeCAD = pytest.importorskip("FreeCAD")
if not getattr(FreeCAD, "__file__", None):
    pytest.skip("Requires real FreeCAD bindings", allow_module_level=True)
pytest.importorskip("manifold3d")
import Part  # noqa: E402

from freecad_validator.fem.pre_process import assembly as assembly_module  # noqa: E402
from freecad_validator.fem.pre_process.assembly import (  # noqa: E402
    BodyGeometry,
    overlap_matrix,
    read_clean_bodies,
    regroup_bodies,
    score_assembly,
)
from freecad_validator.fem.pre_process.errors import (  # noqa: E402
    BodyCorrespondenceError,
    CandidateGeometryError,
    EvaluationError,
)
from freecad_validator.fem.pre_process.geometry import refine_geometry  # noqa: E402
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (  # noqa: E402
    DiffConfig,
)
from freecad_validator.fem.pre_process.score import main  # noqa: E402
from freecad_validator.fem.pre_process.scorer import PreProcessScorer  # noqa: E402
from freecad_validator.fem.schema import ScoringReport  # noqa: E402
from freecad_validator.fem.step_interface import ExtractionError  # noqa: E402
from freecad_validator.fem.step_interface import _extract as extract  # noqa: E402


def save_analysis(path, shapes, history=None):
    doc = FreeCAD.newDocument("AssemblyTest")
    try:
        geometry = doc.addObject("Part::Feature", "RenamedAnalysis")
        geometry.Shape = Part.makeCompound(shapes)
        mesh = doc.addObject("Fem::FemMeshShapeBaseObjectPython", "Mesh")
        mesh.Shape = geometry
        if history is not False:
            feature = doc.addObject("Part::Feature", "RenamedBoolean")
            feature.addProperty("App::PropertyLinkList", "Objects")
            feature.addProperty("App::PropertyString", "Mode")
            objects = []
            for shape in shapes if history is None else history:
                part = doc.addObject("Part::Feature", "Input")
                part.Shape = shape
                objects.append(part)
            feature.Objects = objects
            feature.Mode = "CompSolid"
            feature.Shape = geometry.Shape
            mesh.Shape = feature
        doc.recompute()
        doc.saveAs(str(path))
    finally:
        FreeCAD.closeDocument(doc.Name)


@pytest.fixture
def assembly_case(tmp_path):
    solid = Part.makeBox(10, 10, 10)
    raw = solid.cut(Part.makeCylinder(1.5, 10, FreeCAD.Vector(3, 3, 0)))
    unchanged = Part.makeBox(5, 5, 5, FreeCAD.Vector(30, 0, 0))
    raw_path = tmp_path / "raw.step"
    Part.makeCompound([raw, unchanged]).exportStep(str(raw_path))
    ref = tmp_path / "reference.FCStd"
    save_analysis(ref, [solid, unchanged])
    return raw_path, ref, solid, raw, unchanged


def test_automatic_self_oracle_and_unchanged_body_skip(assembly_case):
    raw, ref, *_ = assembly_case
    result = score_assembly(raw, ref, ref, PreProcessScorer(DiffConfig(region_sample_count=256)))
    assert result["score"] == 1
    assert result["reference_changed_body_count"] == 1
    assert list(result["bodies"].values()).count(None) == 1


def test_default_freecad_worker_uses_automatic_correspondence(assembly_case, tmp_path):
    freecad = shutil.which("freecadcmd")
    if freecad is None:
        pytest.skip("FreeCAD command-line worker not installed")
    raw, ref, *_ = assembly_case
    worker = Path(assembly_module.__file__).parents[1] / "adapters" / "pre_process.py"
    report = extract(
        freecad,
        str(worker),
        str(raw),
        str(tmp_path / "score.json"),
        extra_args=[str(ref), str(ref)],
    )
    assert report["score"] == 1
    assert report["reference_changed_body_count"] == 1
    with pytest.raises(ExtractionError, match="Raw STEP does not exist"):
        extract(
            freecad,
            str(worker),
            str(tmp_path / "missing.step"),
            str(tmp_path / "error.json"),
            extra_args=[str(ref), str(ref)],
        )


def test_cli_combines_verified_fem_report(assembly_case, tmp_path, monkeypatch):
    raw_path, ref, solid, raw, unchanged = assembly_case
    other_raw, other_clean = raw.copy(), solid.copy()
    other_raw.translate(FreeCAD.Vector(60, 0, 0))
    other_clean.translate(FreeCAD.Vector(60, 0, 0))
    Part.makeCompound([raw, unchanged, other_raw]).exportStep(str(raw_path))
    save_analysis(ref, [solid, unchanged, other_clean])
    answer = tmp_path / "answer.FCStd"
    save_analysis(answer, [solid, unchanged, other_raw])
    fem = tmp_path / "fem.json"
    ScoringReport("test", 82.5, "good").save(str(fem))
    output = tmp_path / "combined.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--assembly",
            "--raw",
            str(raw_path),
            "--reference",
            str(ref),
            "--candidate",
            str(answer),
            "--fem-report",
            str(fem),
            "--out",
            str(output),
        ],
    )
    main()
    combined = json.loads(output.read_text())
    assert combined["geometry"]["score"] == 0.5
    assert combined["geometry"]["reference_changed_body_count"] == 2
    assert combined["overall_score"] == 41.25
    assert combined["reward"] == 0.4125
    assert combined["fem_report"]["grade"] == "poor"


def test_added_body_is_deducted_without_increasing_denominator(assembly_case, tmp_path):
    raw, ref, solid, _, unchanged = assembly_case
    answer = tmp_path / "answer.FCStd"
    save_analysis(answer, [unchanged, solid, Part.makeBox(2, 2, 2, FreeCAD.Vector(80, 0, 0))])
    result = score_assembly(raw, ref, answer, PreProcessScorer(DiffConfig(region_sample_count=256)))
    assert result["reference_changed_body_count"] == 1
    assert result["correct_body_count"] == 1
    assert result["extra_changed_body_count"] == 1
    assert result["score"] == 0


def test_changed_unchanged_body_is_deducted(assembly_case, tmp_path):
    raw, ref, solid, _, unchanged = assembly_case
    answer = tmp_path / "answer.FCStd"
    altered = unchanged.cut(Part.makeCylinder(1, 5, FreeCAD.Vector(32, 2, 0)))
    save_analysis(answer, [solid, altered])
    result = score_assembly(raw, ref, answer, PreProcessScorer(DiffConfig(region_sample_count=256)))
    assert result["extra_changed_body_count"] == 1
    assert result["score"] == 0


def test_deletions_are_original_bodies(assembly_case, tmp_path):
    raw, _, solid, *_ = assembly_case
    ref = tmp_path / "deleted.FCStd"
    save_analysis(ref, [solid])
    result = score_assembly(raw, ref, ref, PreProcessScorer(DiffConfig(region_sample_count=256)))
    assert result["reference_changed_body_count"] == 2
    assert result["correct_body_count"] == 2
    assert result["score"] == 1


def test_split_faces_and_regions_do_not_replace_saved_inputs(assembly_case, tmp_path):
    raw, ref, solid, _, unchanged = assembly_case
    answer = tmp_path / "split.FCStd"
    half = Part.makeBox(5, 10, 10)
    save_analysis(
        answer, [unchanged, solid.cut(half), solid.common(half)], history=[solid, unchanged]
    )
    result = score_assembly(raw, ref, answer, PreProcessScorer(DiffConfig(region_sample_count=256)))
    assert result["score"] == 1
    assert result["reference_changed_body_count"] == 1


def test_requested_move_is_matched_but_wrong_position_still_fails(tmp_path):
    raw_shape = Part.makeBox(10, 10, 10)
    moved = raw_shape.copy()
    moved.translate(FreeCAD.Vector(40, 0, 0))
    raw = tmp_path / "raw.step"
    raw_shape.exportStep(str(raw))
    ref = tmp_path / "moved.FCStd"
    answer = tmp_path / "unmoved.FCStd"
    save_analysis(ref, [moved])
    save_analysis(answer, [raw_shape])
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    assert score_assembly(raw, ref, ref, scorer)["score"] == 1
    result = score_assembly(raw, ref, answer, scorer)
    assert result["reference_changed_body_count"] == 1
    assert result["score"] == 0


def test_repeated_parts_are_matched_by_position():
    one = Part.makeBox(10, 10, 10)
    two = Part.makeBox(10, 10, 10, FreeCAD.Vector(40, 0, 0))
    raw = [BodyGeometry(one, "original1"), BodyGeometry(two, "original2")]
    mapped, extras, ownership = regroup_bodies(
        raw,
        [BodyGeometry(two, "renamed1"), BodyGeometry(one, "renamed2")],
        False,
    )
    assert ownership == [[1], [0]]
    assert not extras
    assert mapped[0].shape.CenterOfMass.x == pytest.approx(5)


def test_selected_boolean_inputs_are_used_without_reading_result(tmp_path):
    left = Part.makeBox(10, 10, 10)
    right = Part.makeBox(10, 10, 10, FreeCAD.Vector(10, 0, 0))
    path = tmp_path / "history.FCStd"
    save_analysis(path, [left.fuse(right)], history=[left, right])
    doc = FreeCAD.openDocument(str(path))
    doc.getObject("RenamedBoolean").Shape = Part.Shape()
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    parts, fragmented = read_clean_bodies(path)
    assert not fragmented
    assert [body.shape.Volume for body in parts] == pytest.approx([1000, 1000])


def test_saved_inputs_are_found_through_a_wrapper_link(assembly_case):
    _, path, solid, _, unchanged = assembly_case
    doc = FreeCAD.openDocument(str(path))
    wrapper = doc.addObject("Part::Feature", "Wrapper")
    wrapper.addProperty("App::PropertyLink", "Source")
    wrapper.Source = doc.getObject("RenamedBoolean")
    # The wrapper has no result shape: only the saved object links are relevant.
    doc.getObject("Mesh").Shape = wrapper
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    parts, fragmented = read_clean_bodies(path)
    assert not fragmented
    assert [body.shape.Volume for body in parts] == pytest.approx([solid.Volume, unchanged.Volume])


def test_fused_hole_fill_is_attributed_to_original_body(tmp_path):
    first = Part.makeBox(10, 10, 10)
    second = Part.makeBox(10, 10, 10, FreeCAD.Vector(10, 0, 0))
    raw = tmp_path / "raw.step"
    hole = Part.makeCylinder(1, 10, FreeCAD.Vector(3, 3, 0))
    Part.makeCompound([first.cut(hole), second]).exportStep(str(raw))
    ref = tmp_path / "fused.FCStd"
    save_analysis(ref, [first.fuse(second)], history=[first, second])
    result = score_assembly(raw, ref, ref, PreProcessScorer(DiffConfig(region_sample_count=256)))
    assert result["score"] == 1
    assert result["reference_changed_body_count"] == 1


def test_ambiguous_added_bridge_is_an_evaluation_error():
    raw = [
        BodyGeometry(Part.makeBox(10, 10, 10), "a"),
        BodyGeometry(Part.makeBox(10, 10, 10, FreeCAD.Vector(12, 0, 0)), "b"),
    ]
    fused = BodyGeometry(Part.makeBox(22, 10, 10), "bridge")
    with pytest.raises(EvaluationError, match="Post-Boolean"):
        regroup_bodies(raw, [fused], True)


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_correspondence_failure_identifies_input_without_claiming_invalid_geometry(role):
    raw = [
        BodyGeometry(Part.makeBox(10, 10, 10), "a"),
        BodyGeometry(Part.makeBox(10, 10, 10, FreeCAD.Vector(12, 0, 0)), "b"),
    ]
    fused = BodyGeometry(Part.makeBox(22, 10, 10), "bridge")
    with pytest.raises(BodyCorrespondenceError, match=f"{role} body correspondence") as error:
        regroup_bodies(raw, [fused], True, raw if role == "candidate" else None)
    assert error.value.input_role == role
    assert not isinstance(error.value, CandidateGeometryError)


def test_correspondence_kernel_failure_retains_candidate_context(monkeypatch):
    monkeypatch.setattr(
        assembly_module,
        "_regroup_bodies",
        Mock(side_effect=Part.OCCError("Cannot reconstruct original body")),
    )
    with pytest.raises(BodyCorrespondenceError, match="candidate body correspondence") as error:
        regroup_bodies([], [], True, [])
    assert "Cannot reconstruct original body" in str(error.value)


def test_correspondence_worker_emits_typed_error_without_score(tmp_path, monkeypatch):
    output = tmp_path / "error.json"
    monkeypatch.setattr(
        "freecad_validator.fem.pre_process.score.score_assembly",
        Mock(side_effect=BodyCorrespondenceError("ambiguous ownership", "candidate")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--assembly",
            "--raw",
            "raw.step",
            "--reference",
            "reference.FCStd",
            "--candidate",
            "answer.FCStd",
            "--out",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert report["status"] == "evaluation_error"
    assert report["input_role"] == "candidate"
    assert report["error_type"] == "body_correspondence"
    assert "score" not in report


def test_failed_refinement_uses_cad_overlap_without_discarding_valid_geometry():
    shape = Part.makeBox(10, 10, 10)
    failing = Mock(wraps=shape)
    failing.BoundBox = shape.BoundBox
    failing.removeSplitter.side_effect = Part.OCCError("Removing splitter failed")
    body = BodyGeometry(failing, "valid but unrefinable")
    assert body.mesh is None
    assert overlap_matrix([body], [BodyGeometry(shape, "other")])[0, 0] == pytest.approx(1000)
    assert refine_geometry(failing).Volume == pytest.approx(1000)
    failing.isValid.return_value = False
    with pytest.raises(Part.OCCError):
        refine_geometry(failing)


def test_failed_tessellation_uses_cad_overlap(monkeypatch):
    monkeypatch.setattr(
        assembly_module, "tessellate_shape", Mock(side_effect=Part.OCCError("Bnd_Box is void"))
    )
    body = BodyGeometry(Part.makeBox(10, 10, 10), "valid")
    assert body.mesh is None
    assert overlap_matrix([body], [body])[0, 0] == pytest.approx(1000)


def test_union_uses_saved_inputs_and_does_not_reconstruct_baked_candidate(tmp_path):
    left = Part.makeBox(10, 10, 10)
    right = Part.makeBox(10, 10, 10, FreeCAD.Vector(12, 0, 0))
    extended_left = Part.makeBox(12, 10, 10)
    path = tmp_path / "union.FCStd"
    save_analysis(path, [extended_left.fuse(right)], history=[extended_left, right])
    doc = FreeCAD.openDocument(str(path))
    doc.getObject("RenamedBoolean").Mode = "Union"
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    parts, fragmented = read_clean_bodies(path)
    assert not fragmented
    assert [body.shape.Volume for body in parts] == pytest.approx([1200, 1000])
    raw = tmp_path / "raw.step"
    Part.makeCompound([left, right]).exportStep(str(raw))
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    assert score_assembly(raw, path, path, scorer)["score"] == 1
    answer = tmp_path / "baked.FCStd"
    save_analysis(answer, [extended_left.fuse(right)], history=False)
    with pytest.raises(BodyCorrespondenceError, match="Pre-Boolean input objects"):
        score_assembly(raw, path, answer, scorer)
    save_analysis(answer, [left, right])
    assert score_assembly(raw, path, answer, scorer)["score"] == 0


def test_missing_candidate_is_distinct_from_bad_reference(assembly_case, tmp_path):
    raw, ref, *_ = assembly_case
    missing = tmp_path / "missing.FCStd"
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    with pytest.raises(CandidateGeometryError, match="does not exist"):
        score_assembly(raw, ref, missing, scorer)
    with pytest.raises(EvaluationError, match="does not exist") as error:
        score_assembly(raw, missing, ref, scorer)
    assert not isinstance(error.value, CandidateGeometryError)


def test_missing_input_links_are_an_error_for_both_roles(tmp_path):
    path = tmp_path / "baked.FCStd"
    save_analysis(path, [Part.makeBox(10, 10, 10)], history=False)
    for candidate, role in ((False, "reference"), (True, "candidate")):
        with pytest.raises(BodyCorrespondenceError, match="Pre-Boolean input objects") as error:
            read_clean_bodies(path, candidate=candidate)
        assert error.value.input_role == role


@pytest.mark.parametrize("invalid", ["empty", "multiple_meshes"])
def test_invalid_candidate_analysis_has_typed_failure(assembly_case, tmp_path, invalid):
    raw, ref, solid, *_ = assembly_case
    candidate = tmp_path / "invalid.FCStd"
    save_analysis(candidate, [solid])
    doc = FreeCAD.openDocument(str(candidate))
    if invalid == "empty":
        doc.getObject("Input").Shape = Part.Shape()
    else:
        doc.addObject("Fem::FemMeshShapeBaseObjectPython", "SecondMesh")
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    with pytest.raises(CandidateGeometryError):
        score_assembly(raw, ref, candidate, PreProcessScorer(DiffConfig(region_sample_count=256)))


def test_invalid_candidate_worker_emits_zero_not_evaluation_error(
    assembly_case, tmp_path, monkeypatch
):
    raw, ref, *_ = assembly_case
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--assembly",
            "--raw",
            str(raw),
            "--reference",
            str(ref),
            "--candidate",
            str(tmp_path / "missing.FCStd"),
            "--out",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text())
    assert report["status"] == "candidate_invalid"
    assert report["score"] == 0
    assert "does not exist" in report["error"]
