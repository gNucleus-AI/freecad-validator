"""Exercise the native OBB backend without requiring a FreeCAD installation."""

from __future__ import annotations

import io
import math

import pytest

pytest.importorskip("OCP")

from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

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
