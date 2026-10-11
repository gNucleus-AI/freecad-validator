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
import Part  # noqa: E402

from freecad_validator.fem.pre_process import assembly as assembly_module  # noqa: E402
from freecad_validator.fem.pre_process.assembly import (  # noqa: E402
    BodyGeometry,
    match_clean_bodies,
    read_clean_bodies,
    score_assembly,
)
from freecad_validator.fem.pre_process.errors import (  # noqa: E402
    BodyCorrespondenceError,
    CandidateGeometryError,
    EvaluationError,
    MissingCleanBodiesError,
)
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
    mapped, extras, ownership = match_clean_bodies(
        raw, [BodyGeometry(two, "renamed1"), BodyGeometry(one, "renamed2")]
    )
    assert ownership == [[1], [0]]
    assert not extras
    assert mapped[0].shape.CenterOfMass.x == pytest.approx(5)


def test_saved_inputs_do_not_replace_missing_analysis_geometry(tmp_path):
    left = Part.makeBox(10, 10, 10)
    right = Part.makeBox(10, 10, 10, FreeCAD.Vector(10, 0, 0))
    path = tmp_path / "history.FCStd"
    save_analysis(path, [left.fuse(right)], history=[left, right])
    doc = FreeCAD.openDocument(str(path))
    doc.getObject("RenamedBoolean").Shape = Part.Shape()
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    with pytest.raises(EvaluationError, match="No solid geometry"):
        read_clean_bodies(path)


def test_saved_inputs_are_found_through_a_wrapper_link(assembly_case):
    _, path, solid, _, unchanged = assembly_case
    doc = FreeCAD.openDocument(str(path))
    wrapper = doc.addObject("Part::Feature", "Wrapper")
    wrapper.addProperty("App::PropertyLink", "Source")
    wrapper.Source = doc.getObject("RenamedBoolean")
    wrapper.Shape = doc.getObject("RenamedBoolean").Shape.copy()
    doc.getObject("Mesh").Shape = wrapper
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    parts = read_clean_bodies(path)
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


def test_correspondence_kernel_failure_retains_candidate_context(monkeypatch):
    monkeypatch.setattr(
        assembly_module,
        "_match_clean_bodies",
        Mock(side_effect=Part.OCCError("Cannot reconstruct original body")),
    )
    with pytest.raises(BodyCorrespondenceError, match="candidate body correspondence") as error:
        match_clean_bodies([], [], [])
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


def test_missing_candidate_is_distinct_from_bad_reference(assembly_case, tmp_path):
    raw, ref, *_ = assembly_case
    missing = tmp_path / "missing.FCStd"
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    with pytest.raises(CandidateGeometryError, match="does not exist"):
        score_assembly(raw, ref, missing, scorer)
    with pytest.raises(EvaluationError, match="does not exist") as error:
        score_assembly(raw, missing, ref, scorer)
    assert not isinstance(error.value, CandidateGeometryError)


def save_single_part_analysis(path, prepared, original):
    doc = FreeCAD.newDocument("SinglePartTest")
    try:
        imported = doc.addObject("Part::Feature", "Original")
        imported.Shape = original
        container = doc.addObject("App::Part", "AnalysisContainer")
        geometry = doc.addObject("Part::Feature", "Prepared")
        geometry.Shape = prepared
        container.addObject(geometry)
        mesh = doc.addObject("Fem::FemMeshShapeBaseObjectPython", "Mesh")
        mesh.Shape = geometry
        doc.recompute()
        doc.saveAs(str(path))
    finally:
        FreeCAD.closeDocument(doc.Name)


def test_single_part_uses_prepared_solid_instead_of_detached_raw(tmp_path):
    clean = Part.makeBox(10, 10, 10)
    raw = clean.cut(Part.makeCylinder(1.5, 10, FreeCAD.Vector(3, 3, 0)))
    raw_path = tmp_path / "raw.step"
    raw.exportStep(str(raw_path))
    ref = tmp_path / "reference.FCStd"
    save_single_part_analysis(ref, clean, raw)
    before = ref.read_bytes()
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    result = score_assembly(raw_path, ref, ref, scorer)
    assert result["reference_changed_body_count"] == 1
    assert result["correct_body_count"] == 1
    assert result["score"] == 1
    assert ref.read_bytes() == before
    answer = tmp_path / "unprocessed.FCStd"
    save_single_part_analysis(answer, raw, raw)
    result = score_assembly(raw_path, ref, answer, scorer)
    assert result["reference_changed_body_count"] == 1
    assert result["score"] == 0


def save_detached_inputs(path, clean, raw=()):
    save_analysis(path, clean, history=False)
    doc = FreeCAD.openDocument(str(path))
    try:
        group = doc.addObject("App::Part", "UninformativeGroup")
        for shape in raw:
            obj = doc.addObject("Part::Feature", "UninformativeOriginal")
            obj.Shape = shape
            group.addObject(obj)
        for shape in clean:
            obj = doc.addObject("Part::Feature", "UninformativePrepared")
            obj.Shape = shape
            obj.Visibility = False
        doc.recompute()
        doc.save()
    finally:
        FreeCAD.closeDocument(doc.Name)


@pytest.mark.parametrize("storage", ["detached", "linked"])
@pytest.mark.parametrize("edit", ["correct", "unprocessed", "extra_cut"])
def test_saved_fused_input_is_not_reconstructed_into_original_bodies(tmp_path, storage, edit):
    first = Part.makeBox(10, 10, 10)
    second = Part.makeBox(1, 10, 10, FreeCAD.Vector(10, 0, 0))
    unchanged = Part.makeBox(3, 3, 3, FreeCAD.Vector(30, 0, 0))
    raw_first = first.cut(Part.makeCylinder(1, 10, FreeCAD.Vector(3, 3, 0)))
    raw = tmp_path / "raw.step"
    Part.makeCompound([raw_first, second, unchanged]).exportStep(str(raw))
    ref = tmp_path / "reference.FCStd"
    save_analysis(ref, [first.fuse(second), unchanged], history=[first, second, unchanged])
    fused = (raw_first if edit == "unprocessed" else first).fuse(second).removeSplitter()
    if edit == "extra_cut":
        fused = fused.cut(Part.makeCylinder(0.3, 10, FreeCAD.Vector(10.5, 3, 0)))
    candidate = tmp_path / "candidate.FCStd"
    if storage == "detached":
        save_detached_inputs(candidate, [fused, unchanged])
    else:
        save_analysis(candidate, [fused, unchanged])
    result = score_assembly(
        raw, ref, candidate, PreProcessScorer(DiffConfig(region_sample_count=256))
    )
    assert result["score"] == 0
    assert result["reference_changed_body_count"] == 1
    mapping = result["correspondence"]["candidate"]
    assert mapping[2] == [1]
    assert sorted(mapping[:2]) == [[], [0]]


def test_contained_fitting_does_not_turn_a_saved_body_into_a_fusion():
    housing = BodyGeometry(Part.makeBox(10, 10, 10), "housing")
    fitting = BodyGeometry(Part.makeBox(1, 1, 1, FreeCAD.Vector(2, 2, 2)), "fitting")
    rebuilt, extras, mapping = match_clean_bodies([housing, fitting], [housing], [housing, fitting])
    assert not extras
    assert mapping == [[0], []]
    assert rebuilt[1] is None


@pytest.mark.parametrize("with_raw_history", [False, True])
def test_legacy_detached_clean_inputs_score_without_rewriting_file(
    assembly_case, tmp_path, with_raw_history
):
    raw_path, ref, solid, raw, unchanged = assembly_case
    answer = tmp_path / "legacy.FCStd"
    save_detached_inputs(answer, [solid, unchanged], [raw, unchanged] if with_raw_history else [])
    before = answer.read_bytes()
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    result = score_assembly(raw_path, ref, answer, scorer)
    assert result["score"] == 1
    assert result["reference_changed_body_count"] == 1
    assert score_assembly(raw_path, answer, answer, scorer)["score"] == 1
    assert answer.read_bytes() == before


def test_detached_history_preserves_deletions_and_penalizes_extra_changes(assembly_case, tmp_path):
    raw_path, _, solid, raw, unchanged = assembly_case
    reference = tmp_path / "legacy_ref.FCStd"
    answer = tmp_path / "legacy_answer.FCStd"
    save_detached_inputs(reference, [solid], [raw, unchanged])
    save_detached_inputs(answer, [solid, unchanged], [raw, unchanged])
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    result = score_assembly(raw_path, reference, answer, scorer)
    assert result["reference_changed_body_count"] == 2
    assert result["score"] == 0.5
    extra = Part.makeBox(2, 2, 2, FreeCAD.Vector(80, 0, 0))
    save_detached_inputs(answer, [solid, extra], [raw, unchanged])
    result = score_assembly(raw_path, reference, answer, scorer)
    assert result["extra_changed_body_count"] == 1
    assert result["score"] == 0.5


def test_detached_raw_geometry_is_not_automatically_full_credit(assembly_case, tmp_path):
    raw_path, ref, _, raw, unchanged = assembly_case
    answer = tmp_path / "unchanged.FCStd"
    save_detached_inputs(answer, [raw, unchanged])
    result = score_assembly(
        raw_path, ref, answer, PreProcessScorer(DiffConfig(region_sample_count=256))
    )
    assert result["score"] == 0


@pytest.mark.parametrize("association", ["link", "group"])
def test_detached_recovery_verifies_analysis_without_scoring_linked_fragments(
    assembly_case, tmp_path, monkeypatch, association
):
    _, _, solid, _, unchanged = assembly_case
    path = tmp_path / "detached.FCStd"
    save_detached_inputs(path, [solid, unchanged])
    doc = FreeCAD.openDocument(str(path))
    fragment = doc.addObject("Part::Feature", "AnalysisFragment")
    fragment.Shape = Part.makeBox(1, 1, 1)
    if association == "link":
        fragment.addProperty("App::PropertyLink", "Source")
        fragment.Source = doc.getObject("RenamedAnalysis")
    else:
        group = doc.addObject("App::Part", "AnalysisContainer")
        group.addObject(doc.getObject("RenamedAnalysis"))
        group.addObject(fragment)
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    original = assembly_module.world_shape
    reader = Mock(side_effect=original)
    monkeypatch.setattr(assembly_module, "world_shape", reader)
    parts = read_clean_bodies(path)
    assert len(parts) == 2
    assert reader.call_count == 3
    assert [body.shape.Volume for body in parts] == pytest.approx([solid.Volume, unchanged.Volume])


def test_explicit_empty_input_set_does_not_fall_back_to_detached_parts(assembly_case, tmp_path):
    _, _, solid, _, unchanged = assembly_case
    path = tmp_path / "empty_links.FCStd"
    save_detached_inputs(path, [solid, unchanged])
    doc = FreeCAD.openDocument(str(path))
    geometry = doc.getObject("RenamedAnalysis")
    geometry.addProperty("App::PropertyLinkList", "PreprocessingInputs")
    geometry.PreprocessingInputs = []
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    with pytest.raises(MissingCleanBodiesError, match="Pre-Boolean input objects"):
        read_clean_bodies(path)


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


def test_disconnected_boolean_history_is_found_without_rewriting_file(assembly_case, tmp_path):
    raw, ref, solid, _, unchanged = assembly_case
    answer = tmp_path / "disconnected.FCStd"
    save_analysis(answer, [solid, unchanged])
    doc = FreeCAD.openDocument(str(answer))
    doc.getObject("Mesh").Shape = doc.getObject("RenamedAnalysis")
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    before = answer.read_bytes()
    result = score_assembly(raw, ref, answer, PreProcessScorer(DiffConfig(region_sample_count=256)))
    assert result["score"] == 1
    assert all(result["correspondence"]["candidate"])
    assert answer.read_bytes() == before


def test_plain_snapshot_requires_clean_history(assembly_case, tmp_path):
    raw_path, ref, solid, raw, unchanged = assembly_case
    answer = tmp_path / "plain.FCStd"
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    save_analysis(answer, [solid, unchanged], history=False)
    result = score_assembly(raw_path, ref, answer, scorer)
    assert result["score"] == 0
    assert result["status"] == "missing_clean_bodies"
    save_analysis(answer, [raw, unchanged], history=False)
    assert score_assembly(raw_path, ref, answer, scorer)["score"] == 0


def test_union_prefers_saved_inputs_and_rejects_baked_candidate(tmp_path):
    left = Part.makeBox(10, 10, 10)
    right = Part.makeBox(10, 10, 10, FreeCAD.Vector(12, 0, 0))
    extended_left = Part.makeBox(12, 10, 10)
    path = tmp_path / "union.FCStd"
    save_analysis(path, [extended_left.fuse(right)], history=[extended_left, right])
    doc = FreeCAD.openDocument(str(path))
    doc.getObject("RenamedBoolean").Mode = "Union"
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    parts = read_clean_bodies(path)
    assert [body.shape.Volume for body in parts] == pytest.approx([1200, 1000])
    raw = tmp_path / "raw.step"
    Part.makeCompound([left, right]).exportStep(str(raw))
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    assert score_assembly(raw, path, path, scorer)["score"] == 1
    answer = tmp_path / "baked.FCStd"
    save_analysis(answer, [extended_left.fuse(right)], history=False)
    result = score_assembly(raw, path, answer, scorer)
    assert result["score"] == 0
    assert result["status"] == "missing_clean_bodies"
    save_analysis(answer, [left, right])
    assert score_assembly(raw, path, answer, scorer)["score"] == 0


def test_missing_input_links_require_saved_clean_bodies_for_both_roles(tmp_path):
    path = tmp_path / "baked.FCStd"
    save_analysis(path, [Part.makeBox(10, 10, 10)], history=False)
    for candidate in (False, True):
        with pytest.raises(MissingCleanBodiesError):
            read_clean_bodies(path, candidate=candidate)


def test_single_part_compound_is_not_unwrapped_as_prepared_solid(tmp_path):
    raw = Part.makeBox(10, 10, 10)
    path = tmp_path / "compound.FCStd"
    save_analysis(path, [raw], history=False)
    with pytest.raises(MissingCleanBodiesError):
        read_clean_bodies(path, raw=[BodyGeometry(raw, "raw")])


def test_multi_original_fused_solid_does_not_use_single_part_path(tmp_path):
    left = Part.makeBox(10, 10, 10)
    right = Part.makeBox(10, 10, 10, FreeCAD.Vector(10, 0, 0))
    path = tmp_path / "fused.FCStd"
    save_analysis(path, [left], history=False)
    doc = FreeCAD.openDocument(str(path))
    doc.getObject("RenamedAnalysis").Shape = left.fuse(right).removeSplitter()
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    originals = [BodyGeometry(left, "a"), BodyGeometry(right, "b")]
    with pytest.raises(MissingCleanBodiesError):
        read_clean_bodies(path, raw=originals)


def test_detached_alternative_versions_are_not_chosen_by_reference_score(assembly_case, tmp_path):
    raw_path, ref, solid, raw, unchanged = assembly_case
    answer = tmp_path / "ambiguous.FCStd"
    save_detached_inputs(answer, [solid, raw, unchanged])
    doc = FreeCAD.openDocument(str(answer))
    doc.getObject("RenamedAnalysis").Shape = Part.makeCompound([raw, unchanged])
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    result = score_assembly(
        raw_path, ref, answer, PreProcessScorer(DiffConfig(region_sample_count=256))
    )
    assert result["status"] == "missing_clean_bodies"
    assert result["score"] == 0


@pytest.mark.parametrize("storage", ["linked", "wrapper", "detached", "disconnected"])
def test_decoy_inputs_produce_zero_worker_reward(assembly_case, tmp_path, monkeypatch, storage):
    raw_path, ref, solid, raw, unchanged = assembly_case
    answer = tmp_path / "decoy.FCStd"
    if storage == "detached":
        save_detached_inputs(answer, [solid, unchanged])
    else:
        save_analysis(answer, [solid, unchanged])
    doc = FreeCAD.openDocument(str(answer))
    geometry = doc.getObject("Mesh").Shape
    if storage == "wrapper":
        wrapper = doc.addObject("Part::Feature", "Wrapper")
        wrapper.addProperty("App::PropertyLink", "Source")
        wrapper.Source = geometry
        doc.getObject("Mesh").Shape = wrapper
        geometry = wrapper
    geometry.Shape = Part.makeCompound([raw, unchanged])
    if storage == "disconnected":
        doc.getObject("RenamedAnalysis").Shape = geometry.Shape
        doc.getObject("Mesh").Shape = doc.getObject("RenamedAnalysis")
    doc.save()
    FreeCAD.closeDocument(doc.Name)
    scorer = PreProcessScorer(DiffConfig(region_sample_count=256))
    result = score_assembly(raw_path, ref, answer, scorer)
    assert result["score"] == 0
    assert result["status"] == "missing_clean_bodies"
    assert result["input_role"] == "candidate"
    reference_result = score_assembly(raw_path, answer, ref, scorer)
    assert reference_result["score"] == 0
    assert reference_result["input_role"] == "reference"
    fem = tmp_path / "fem.json"
    ScoringReport("test", 100.0, "excellent").save(str(fem))
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
    report = json.loads(output.read_text())
    assert report["geometry"]["status"] == "missing_clean_bodies"
    assert report["geometry"]["score"] == 0
    assert report["reward"] == 0
    assert report["overall_score"] == 0


@pytest.mark.parametrize("missing_material", [False, True])
def test_saved_input_occupancy_check_does_not_build_geometry(monkeypatch, missing_material):
    analyzed = Part.makeBox(10, 10, 10)
    prepared = Part.makeBox(5 if missing_material else 10, 10, 10)
    bodies = [BodyGeometry(prepared, "prepared")]
    monkeypatch.setattr(
        assembly_module.Part,
        "makeCompound",
        Mock(side_effect=AssertionError("Must not build verification geometry")),
    )
    if missing_material:
        with pytest.raises(EvaluationError, match="do not match"):
            assembly_module.verify_clean_inputs(bodies, analyzed, "reference")
    else:
        assembly_module.verify_clean_inputs(bodies, analyzed, "reference")


@pytest.mark.parametrize("extra_width", [0.0, 0.25])
def test_sampled_input_union_uses_five_percent_tolerance(extra_width):
    first = Part.makeBox(3, 2, 2)
    second = Part.makeBox(3, 2, 2, FreeCAD.Vector(2, 0, 0))
    actual = Part.makeBox(5 + extra_width, 2, 2)
    bodies = [BodyGeometry(first, "first"), BodyGeometry(second, "overlapping")]
    analyzed = [BodyGeometry(actual, "analysis")]
    assert assembly_module.history_matches_analysis(bodies, analyzed)
    # Overlap must use OR occupancy; a missing input still leaves exposed material.
    assert not assembly_module.history_matches_analysis(bodies[:1], analyzed)


def test_input_verification_failure_does_not_become_candidate_zero(monkeypatch):
    body = BodyGeometry(Part.makeBox(1, 1, 1), "prepared")
    monkeypatch.setattr(
        assembly_module,
        "_inside",
        Mock(side_effect=Part.OCCError("injected verification failure")),
    )
    with pytest.raises(BodyCorrespondenceError, match="injected verification failure"):
        assembly_module.verify_clean_inputs([body], Part.makeBox(1, 1, 1), "candidate")
