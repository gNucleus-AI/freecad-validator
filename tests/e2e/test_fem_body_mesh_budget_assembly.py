"""Saved assembly meshes follow preprocessing ownership, not labels."""

import hashlib
import json
import math
import runpy
import sys
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.needs_freecad

FreeCAD = pytest.importorskip("FreeCAD")
if not getattr(FreeCAD, "__file__", None):
    pytest.skip("Requires real FreeCAD bindings", allow_module_level=True)
import Fem  # noqa: E402
import ObjectsFem  # noqa: E402
import Part  # noqa: E402

from freecad_validator.fem.errors import EvaluationError  # noqa: E402
from freecad_validator.fem.mesh.local_budget import (  # noqa: E402
    count_mesh_nodes,
    element_overlaps_bounds,
    target_geometry,
    verify_target_geometry,
)
from freecad_validator.fem.pre_process.assembly import (  # noqa: E402
    BodyGeometry,
    match_clean_bodies,
    read_clean_bodies,
    score_assembly,
)
from freecad_validator.fem.pre_process.errors import MissingCleanBodiesError  # noqa: E402
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (  # noqa: E402
    DiffConfig,
)
from freecad_validator.fem.pre_process.scorer import PreProcessScorer  # noqa: E402
from freecad_validator.fem.pre_process.spatial import SpatialShape  # noqa: E402
from freecad_validator.fem.step_interface import MESH_BUDGET_WORKER  # noqa: E402


def two_tetrahedra():
    mesh = Fem.FemMesh()
    for i, point in enumerate([(0, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (1, 0, 0)], 1):
        mesh.addNode(*point, i)
    mesh.addVolume([1, 2, 3, 4], 1)
    mesh.addVolume([1, 3, 2, 5], 2)
    return mesh


def test_shared_nodes_and_split_pieces_are_unioned():
    left = Part.makeBox(1, 1, 1, FreeCAD.Vector(-1, 0, 0))
    right = Part.makeBox(1, 1, 1)
    mesh = two_tetrahedra()
    bodies = {"Body1": BodyGeometry(left, "arbitrary"), "Body2": BodyGeometry(right, "arbitrary")}
    assert count_mesh_nodes(mesh, bodies, EvaluationError) == {"Body1": 4, "Body2": 4}
    joined = BodyGeometry(SpatialShape([left, right]), "split")
    assert count_mesh_nodes(mesh, {"Body1": joined}, EvaluationError) == {"Body1": 5}
    assert count_mesh_nodes(mesh, {"Body1": joined, "Body2": joined}, EvaluationError) == {
        "Body1": 5,
        "Body2": 5,
    }


def test_quadratic_nodes_are_included():
    mesh = Fem.FemMesh()
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    points = corners + [
        tuple((a + b) / 2 for a, b in zip(corners[i], corners[j], strict=True)) for i, j in edges
    ]
    for i, point in enumerate(points, 1):
        mesh.addNode(*point, i)
    mesh.addVolume(list(range(1, 11)), 1)
    body = BodyGeometry(Part.makeBox(1, 1, 1), "unused label")
    assert count_mesh_nodes(mesh, {"Body1": body}, EvaluationError) == {"Body1": 10}


@pytest.mark.parametrize("angle", [1.0, 1.4])
def test_curved_thin_wall_element_uses_quadratic_geometry(angle):
    # Constructed annular-wall fixture: the corner centroid lies in the hole.
    # At the larger angle even the quadratic center lies just inside the hole,
    # while the interior quadrature points supply consistent wall ownership.
    polar = [(10, 0, 0), (10, angle, 0), (10, 0, 1), (10.2, 0, 0)]
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    polar += [tuple((a + b) / 2 for a, b in zip(polar[i], polar[j], strict=True)) for i, j in edges]
    mesh = Fem.FemMesh()
    for index, (radius, theta, z) in enumerate(polar, 1):
        mesh.addNode(radius * math.cos(theta), radius * math.sin(theta), z, index)
    mesh.addVolume(list(range(1, 11)), 1)
    wall = Part.makeCylinder(10.2, 1).cut(Part.makeCylinder(10, 1))
    body = BodyGeometry(wall, "annular wall")
    assert count_mesh_nodes(mesh, {"Body1": body}, EvaluationError) == {"Body1": 10}


def test_unassigned_elements_and_orphan_nodes_cannot_pass():
    mesh = two_tetrahedra()
    body = BodyGeometry(Part.makeBox(1, 1, 1), "only right side")
    with pytest.raises(EvaluationError, match="Cannot assign"):
        count_mesh_nodes(mesh, {"Body1": body}, EvaluationError)
    mesh.addNode(50, 50, 50, 99)
    with pytest.raises(EvaluationError, match="cover exactly"):
        count_mesh_nodes(mesh, {"Body1": body}, EvaluationError)


def test_local_budget_ignores_only_provably_distant_unassigned_elements():
    mesh = Fem.FemMesh()
    points = [
        (0, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (10, 0, 0),
        (10, 1, 0),
        (10, 0, 1),
        (11, 0, 0),
    ]
    for index, point in enumerate(points, 1):
        mesh.addNode(*point, index)
    mesh.addVolume([1, 2, 3, 4], 1)
    mesh.addVolume([5, 6, 7, 8], 2)
    body = BodyGeometry(Part.makeBox(1, 1, 1, FreeCAD.Vector(-1, 0, 0)), "target")
    with pytest.raises(EvaluationError, match="Cannot assign"):
        count_mesh_nodes(mesh, {"Body1": body}, EvaluationError)
    assert count_mesh_nodes(mesh, {"Body1": body}, EvaluationError, target_body="Body1") == {
        "Body1": 4
    }
    # The unresolved adjacent tetrahedron touches the target's bounds: keep the failure.
    with pytest.raises(EvaluationError, match="Cannot assign"):
        count_mesh_nodes(two_tetrahedra(), {"Body1": body}, EvaluationError, target_body="Body1")


def test_curved_element_control_hull_is_used_instead_of_nodal_box():
    corners = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0)]
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
    points = corners + [
        tuple((a + b) / 2 for a, b in zip(corners[i], corners[j], strict=True)) for i, j in edges
    ]
    points[4] = (0.4, 0.5, 0)
    nodes = {i: FreeCAD.Vector(*p) for i, p in enumerate(points, 1)}
    bounds = FreeCAD.BoundBox(0.6, 0, 0, 1, 1, 1)
    assert max(p[0] for p in points) < bounds.XMin
    assert element_overlaps_bounds(nodes, tuple(nodes), bounds)


def test_geometry_locator_rejects_wrong_or_ambiguous_reference_body():
    body = BodyGeometry(Part.makeBox(1, 2, 3), "arbitrary label")
    other = BodyGeometry(Part.makeBox(1, 2, 3, FreeCAD.Vector(10, 0, 0)), "same label")
    geometry = target_geometry(body)
    verify_target_geometry("Body1", geometry, {"Body1": body, "Body2": other})
    with pytest.raises(EvaluationError, match="uniquely"):
        verify_target_geometry("Body1", geometry, {"Body1": other, "Body2": body})
    with pytest.raises(EvaluationError, match="uniquely"):
        verify_target_geometry("Body1", geometry, {"Body1": body, "Body2": body})


def save_budget_assembly(path, reverse=False, history=True):
    """Fixture construction only; grading never creates or recomputes geometry."""
    doc = FreeCAD.newDocument("BudgetFixture")
    try:
        shapes = [Part.makeBox(1, 1, 1, FreeCAD.Vector(-1, 0, 0)), Part.makeBox(1, 1, 1)]
        clean = []
        if history:
            for shape in reversed(shapes) if reverse else shapes:
                body = doc.addObject("Part::Feature", "Renamed")
                body.Shape = shape
                clean.append(body)
        final = doc.addObject("Part::Feature", "Assembly")
        final.Shape = Part.makeCompound(
            list(reversed(shapes)) if reverse and not history else shapes
        )
        if history:
            final.addProperty("App::PropertyLinkList", "PreprocessingInputs")
            final.PreprocessingInputs = clean
        mesh = doc.addObject("Fem::FemMeshShapeBaseObjectPython", "Source")
        mesh.Shape = final
        mesh.FemMesh = two_tetrahedra()
        analysis = doc.addObject("Fem::FemAnalysis", "Analysis")
        solver = doc.addObject("Fem::FemSolverObjectPython", "Solver")
        result = ObjectsFem.makeResultMechanical(doc, "Result")
        result.NodeNumbers = list(range(1, 6))
        result.DisplacementLengths = [0.0] * 5
        analysis.Group = [mesh, solver, result]
        doc.recompute()
        doc.saveAs(str(path))
    finally:
        FreeCAD.closeDocument(doc.Name)


def test_case_budget_reuses_reordered_preprocessing_match_read_only(tmp_path):
    raw = tmp_path / "raw.step"
    Part.makeCompound(
        [Part.makeBox(1, 1, 1, FreeCAD.Vector(-1, 0, 0)), Part.makeBox(1, 1, 1)]
    ).exportStep(str(raw))
    reference = tmp_path / "reference.FCStd"
    candidate = tmp_path / "candidate.FCStd"
    save_budget_assembly(reference)
    save_budget_assembly(candidate, reverse=True)
    target = BodyGeometry(Part.makeBox(1, 1, 1, FreeCAD.Vector(-1, 0, 0)), "reference")
    budget = {"local_max": {"Body1": {"max_node_count": 4, "geometry": target_geometry(target)}}}
    (tmp_path / "mesh_budget.json").write_text(json.dumps(budget))
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (raw, reference, candidate)]
    report = score_assembly(
        raw, reference, candidate, PreProcessScorer(DiffConfig(region_sample_count=256))
    )
    assert report["correspondence"]["reference"] == [[0], [1]]
    assert report["correspondence"]["candidate"] == [[1], [0]]
    assert report["mesh_budget"]["score"] == 100
    assert report["mesh_budget"]["candidate_node_count"] == 4
    assert report["mesh_budget"]["reference_geometry_verified"] is True
    assert before == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (raw, reference, candidate)
    ]


@pytest.mark.parametrize("history", [False, True])
def test_plain_case_reuses_matcher_without_preprocessing_credit(tmp_path, history, monkeypatch):
    raw = tmp_path / "raw.step"
    Part.makeCompound(
        [Part.makeBox(1, 1, 1, FreeCAD.Vector(-1, 0, 0)), Part.makeBox(1, 1, 1)]
    ).exportStep(str(raw))
    reference = tmp_path / "reference.FCStd"
    candidate = tmp_path / "candidate.FCStd"
    save_budget_assembly(reference, history=history)
    save_budget_assembly(candidate, reverse=True, history=history)
    (tmp_path / "mesh_budget.json").write_text(json.dumps({"local_max": {"Body1": 4}}))
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (raw, reference, candidate)]
    scorer = Mock()
    with patch(
        "freecad_validator.fem.pre_process.assembly.match_clean_bodies", wraps=match_clean_bodies
    ) as matcher:
        report = score_assembly(raw, reference, candidate, scorer, require_preprocessing=False)
    assert matcher.call_count == 2
    scorer._score_shapes.assert_not_called()
    assert report["score"] is None
    assert report["bodies"] == {}
    assert report["correspondence"]["candidate"] == [[1], [0]]
    assert report["mesh_budget"]["score"] == 100
    assert before == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (raw, reference, candidate)
    ]
    if not history:
        with pytest.raises(MissingCleanBodiesError):
            read_clean_bodies(reference)
        output = tmp_path / "budget_report.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [MESH_BUDGET_WORKER, str(raw), str(reference), str(candidate), str(output)],
        )
        with patch.object(
            PreProcessScorer,
            "_score_shapes",
            side_effect=AssertionError("Unexpected geometry scoring"),
        ):
            runpy.run_path(MESH_BUDGET_WORKER, run_name="__main__")
        worker_report = json.loads(output.read_text())
        assert worker_report["mesh_budget"] == report["mesh_budget"]
        assert worker_report["correspondence"] == report["correspondence"]
        assert before == [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in (raw, reference, candidate)
        ]
