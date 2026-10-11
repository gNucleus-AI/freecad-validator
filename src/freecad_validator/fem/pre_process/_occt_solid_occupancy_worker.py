"""Isolated, read-only OCCT solid classification worker."""

import io
import json
import sys

from OCP.BRep import BRep_Builder
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
from OCP.TopoDS import TopoDS_Shape


def classify_solids(
    breps: list[str],
    points: list[tuple[float, float, float]],
    tolerance: float,
) -> list[list[bool]]:
    """Load each solid once and reuse its classifier for every query point."""
    query_points = [gp_Pnt(*point) for point in points]
    occupancies = []
    for brep in breps:
        shape = TopoDS_Shape()
        BRepTools.Read_s(shape, io.BytesIO(brep.encode("utf-8")), BRep_Builder())
        if shape.IsNull():
            raise ValueError("Cannot classify an empty solid")
        classifier = BRepClass3d_SolidClassifier(shape)
        occupied = []
        for point in query_points:
            classifier.Perform(point, tolerance)
            occupied.append(classifier.State() in (TopAbs_IN, TopAbs_ON))
        occupancies.append(occupied)
    return occupancies


if __name__ == "__main__":
    request = json.load(sys.stdin)
    json.dump(
        classify_solids(request["breps"], request["points"], request["tolerance"]),
        sys.stdout,
    )
