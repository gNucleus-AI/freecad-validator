"""Exercise the native OBB backend without requiring a FreeCAD installation."""

from __future__ import annotations

import io
import math

import pytest

pytest.importorskip("OCP")

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeTorus
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

from freecad_validator.comparators.geometry import GeometryTolerances
from freecad_validator.comparators.occt_bbox import OBBMeasurementError, oriented_bbox_dimensions


def _brep(shape):
    stream = io.BytesIO()
    BRepTools.Write_s(shape, stream)
    return stream.getvalue().decode("utf-8")


@pytest.mark.parametrize("dimensions", [(100, 10, 5), (10, 10, 10), (20, 20, 5)])
def test_native_obb_recovers_rotated_box_dimensions(dimensions):
    shape = BRepPrimAPI_MakeBox(*dimensions).Shape()
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(1, 2, 3)), math.radians(37))
    rotated = BRepBuilderAPI_Transform(shape, transform, True).Shape()
    assert oriented_bbox_dimensions(_brep(rotated)) == pytest.approx(sorted(dimensions), abs=1e-9)


def test_native_obb_detects_scale_without_three_face_centers():
    reference = oriented_bbox_dimensions(_brep(BRepPrimAPI_MakeCone(5, 0, 20).Shape()))
    candidate = oriented_bbox_dimensions(_brep(BRepPrimAPI_MakeCone(50, 0, 200).Shape()))
    error = max(abs(a - b) / max(a, b) for a, b in zip(reference, candidate, strict=True))
    assert error == pytest.approx(0.9)


@pytest.mark.parametrize(
    "major,minor,degrees,axis,expected_error,rejected",
    [
        (30, 8, 20, (3, 1, 2), 0.0952535127, False),
        (50, 5, 20, (3, 1, 2), 0.0988746545, False),
        (50, 5, 75, (1, 3, 7), 0.1046807810, True),
    ],
)
def test_native_obb_known_torus_rotation_issue(
    major, minor, degrees, axis, expected_error, rejected
):
    """Record upstream OCCT pose sensitivity, so kernel changes get reviewed.

    These are observations for specific congruent tori, not desired scoring
    behavior or a universal error bound. A kernel fix should update this test.
    """
    shape = BRepPrimAPI_MakeTorus(major, minor).Shape()
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(*axis)), math.radians(degrees))
    rotated = BRepBuilderAPI_Transform(shape, transform, True).Shape()
    reference = oriented_bbox_dimensions(_brep(shape))
    candidate = oriented_bbox_dimensions(_brep(rotated))
    error = max(abs(a - b) / max(a, b) for a, b in zip(reference, candidate, strict=True))
    # Allow 0.01 percentage points of platform variation, while detecting drift.
    assert error == pytest.approx(expected_error, abs=1e-4)
    assert (error >= GeometryTolerances().bbox_far_rel_tol) is rejected


def test_native_obb_rejects_empty_brep():
    with pytest.raises(OBBMeasurementError, match="empty BREP"):
        oriented_bbox_dimensions("")


def test_native_meshing_exception_is_a_measurement_error(monkeypatch):
    brep = _brep(BRepPrimAPI_MakeBox(1, 2, 3).Shape())
    failure = RuntimeError("native meshing failed")

    def fail(*args):
        raise failure

    monkeypatch.setattr("OCP.BRepMesh.BRepMesh_IncrementalMesh", fail)
    with pytest.raises(OBBMeasurementError, match="native meshing failed") as exc:
        oriented_bbox_dimensions(brep)
    assert exc.value.__cause__ is failure
