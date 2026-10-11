"""Read solid geometry for preprocessing comparisons without modifying its BRep."""

from pathlib import Path

import FreeCAD
import Part

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.pre_process.spatial import SpatialShape


def read_geometry(path: str | Path, object_name: str | None = None):
    """Read world-space geometry, never implicitly pick the first assembly object."""
    path = Path(path).resolve()
    if not path.is_file():
        raise EvaluationError(f"Geometry input does not exist: {path}")
    if path.suffix.lower() in (".step", ".stp"):
        if object_name is not None:
            raise EvaluationError("Object selection is only supported for FCStd inputs")
        shape = Part.Shape()
        shape.read(str(path))
    elif path.suffix.lower() == ".fcstd":
        doc = FreeCAD.openDocument(str(path))
        try:
            if object_name is not None:
                obj = doc.getObject(object_name)
                if obj is None or not hasattr(getattr(obj, "Shape", None), "Solids"):
                    raise EvaluationError(f"No shape object {object_name!r} in {path}")
            else:
                meshes = [obj for obj in doc.Objects if "FemMeshShape" in obj.TypeId]
                if len(meshes) == 1:
                    obj = getattr(meshes[0], "Shape", None) or getattr(meshes[0], "Part", None)
                    if obj is None:
                        raise EvaluationError(f"Mesh has no linked geometry in {path}")
                elif meshes:
                    raise EvaluationError(f"Multiple meshed shapes in {path}; specify an object")
                else:
                    objects = [
                        obj
                        for obj in doc.Objects
                        if hasattr(getattr(obj, "Shape", None), "Solids") and obj.Shape.Solids
                    ]
                    dependencies = {dep.Name for obj in objects for dep in obj.OutList}
                    roots = [obj for obj in objects if obj.Name not in dependencies]
                    if len(roots) != 1:
                        raise EvaluationError(
                            f"Expected one terminal shape in {path}, found {len(roots)}; "
                            "specify an object or export the clean geometry"
                        )
                    obj = roots[0]
            if isinstance(obj, Part.Shape):
                shape = obj
            else:
                shape = Part.getShape(
                    obj,
                    mat=obj.getGlobalPlacement().Matrix,
                    transform=False,
                )
        finally:
            FreeCAD.closeDocument(doc.Name)
    else:
        raise EvaluationError(f"Unsupported geometry format: {path.suffix}")

    if shape.isNull() or not shape.Solids or not shape.isValid() or shape.Volume <= 0:
        raise EvaluationError(f"Input must contain valid, nonempty solid geometry: {path}")
    solids = shape.Solids
    return SpatialShape(solids) if len(solids) > 1 else solids[0]
