"""Diff failures must reach the worker as errors, never geometry credit."""

import json
import sys
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.needs_freecad

FreeCAD = pytest.importorskip("FreeCAD")
if not getattr(FreeCAD, "__file__", None):
    pytest.skip("Requires real FreeCAD bindings", allow_module_level=True)
pytest.importorskip("manifold3d")
import Part  # noqa: E402

from freecad_validator.fem.pre_process.errors import EvaluationError  # noqa: E402
from freecad_validator.fem.pre_process.geometry import export_geometry  # noqa: E402
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.methods.sampled_assignment import (  # noqa: E402
    SampledAssignmentMethod,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (  # noqa: E402
    DiffConfig,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.region_edit_overlap_scorer import (  # noqa: E402
    RegionEditOverlapScorer,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.scorer_base import (  # noqa: E402
    cached_diff,
    clear_doc_cache,
    load_doc,
)
from freecad_validator.fem.pre_process.score import main  # noqa: E402
from freecad_validator.fem.schema import ScoringReport  # noqa: E402


@pytest.fixture
def shapes():
    box = Part.makeBox(10, 10, 10)
    left = Part.makeCylinder(1.5, 10, FreeCAD.Vector(3, 3, 0))
    right = Part.makeCylinder(1.5, 10, FreeCAD.Vector(7, 7, 0))
    return box.cut(left).cut(right), box.cut(right), box.cut(left)


@pytest.fixture
def docs(tmp_path, shapes):
    clear_doc_cache()
    documents = []
    for role, shape in zip(("raw", "reference", "candidate"), shapes, strict=True):
        path = tmp_path / f"{role}.FCStd"
        export_geometry(shape, path)
        documents.append(load_doc(path, DiffConfig()))
    yield documents
    clear_doc_cache()


@pytest.mark.parametrize("failed_call", [1, 2])
@pytest.mark.parametrize("error_class", [RuntimeError, TypeError, Part.OCCError])
def test_region_diff_failure_never_returns_credit(docs, monkeypatch, failed_call, error_class):
    method = SampledAssignmentMethod()
    successful = method.diff(docs[0], docs[1])
    failure = error_class("injected diff failure")
    run = Mock(side_effect=[successful] * (failed_call - 1) + [failure])
    monkeypatch.setattr(SampledAssignmentMethod, "run", run)
    with pytest.raises(EvaluationError, match="injected diff failure") as caught:
        RegionEditOverlapScorer()._score_docs(*docs)
    assert caught.value.__cause__ is failure
    assert run.call_count == failed_call


def test_failed_diff_is_not_cached_as_empty_success(docs, monkeypatch):
    method = SampledAssignmentMethod()
    success = method.diff(docs[0], docs[1])
    run = Mock(side_effect=[RuntimeError("transient diff failure"), success])
    monkeypatch.setattr(method, "run", run)
    with pytest.raises(EvaluationError, match="transient diff failure"):
        cached_diff(method, docs[0], docs[1])
    recovered = cached_diff(method, docs[0], docs[1])
    assert recovered.added or recovered.removed or recovered.modified
    assert cached_diff(method, docs[0], docs[1]) is recovered
    assert run.call_count == 2


def test_successful_empty_diff_retains_full_region_credit(docs):
    result = RegionEditOverlapScorer()._score_docs(docs[0], docs[0], docs[0])
    assert result.score == 1.0


def test_diff_failure_reaches_worker_without_fem_reward(tmp_path, shapes, monkeypatch):
    raw = tmp_path / "raw.step"
    shapes[0].exportStep(str(raw))
    paths = []
    for role, shape in zip(("reference", "candidate"), shapes[1:], strict=True):
        path = tmp_path / f"{role}.FCStd"
        doc = FreeCAD.newDocument("DiffFailureTest")
        try:
            prepared = doc.addObject("Part::Feature", "Prepared")
            prepared.Shape = shape
            geometry = doc.addObject("Part::Feature", "Geometry")
            geometry.Shape = shape.copy()
            geometry.addProperty("App::PropertyLinkList", "PreprocessingInputs")
            geometry.PreprocessingInputs = [prepared]
            mesh = doc.addObject("Fem::FemMeshShapeBaseObjectPython", "Mesh")
            mesh.Shape = geometry
            doc.recompute()
            doc.saveAs(str(path))
        finally:
            FreeCAD.closeDocument(doc.Name)
        paths.append(path)
    fem = tmp_path / "fem.json"
    ScoringReport("test", 100.0, "excellent").save(str(fem))
    original_fem = fem.read_bytes()
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        SampledAssignmentMethod,
        "run",
        Mock(side_effect=RuntimeError("injected worker diff failure")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "score",
            "--assembly",
            "--raw",
            str(raw),
            "--reference",
            str(paths[0]),
            "--candidate",
            str(paths[1]),
            "--fem-report",
            str(fem),
            "--out",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 1
    report = json.loads(output.read_text())
    assert report["status"] == "evaluation_error"
    assert "injected worker diff failure" in report["error"]
    assert not {"score", "overall_score", "reward", "fem_report"}.intersection(report)
    assert fem.read_bytes() == original_fem
