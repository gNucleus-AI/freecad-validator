"""Read saved analysis setup and surface geometry without recomputing the document."""

import math

from femmesh.meshtools import sub_shape_at_global_placement

from freecad_validator.fem.freecad_io import (
    _is_result_object,
    extract_material_body_counts,
    find_replay_context,
    force_direction,
    geometry_facts,
    linked_mesh_shape,
    qty,
)


class UnsupportedSetupError(RuntimeError):
    """The evaluator cannot interpret this setup; candidate validity is unknown."""


def _surface(shape):
    box = shape.BoundBox
    if shape.ShapeType == "Vertex":
        center = list(shape.Point)
    elif shape.ShapeType == "Compound":
        faces = shape.Faces
        edges = shape.Edges
        if faces and shape.Area > 0:
            center = [
                sum(face.Area * getattr(face.CenterOfMass, axis) for face in faces) / shape.Area
                for axis in ("x", "y", "z")
            ]
        elif edges and shape.Length > 0:
            center = [
                sum(edge.Length * getattr(edge.CenterOfMass, axis) for edge in edges) / shape.Length
                for axis in ("x", "y", "z")
            ]
        elif shape.Vertexes:
            center = [
                sum(getattr(vertex.Point, axis) for vertex in shape.Vertexes) / len(shape.Vertexes)
                for axis in ("x", "y", "z")
            ]
        else:
            raise ValueError("Constraint reference contains no measurable geometry")
    else:
        center = list(shape.CenterOfMass)
    descriptor = {
        "area_mm2": float(shape.Area),
        "centroid": center,
        "bbox_min": [box.XMin, box.YMin, box.ZMin],
        "bbox_max": [box.XMax, box.YMax, box.ZMax],
    }
    if shape.ShapeType == "Face":
        u0, u1, v0, v1 = shape.ParameterRange
        descriptor["normal"] = list(shape.normalAt((u0 + u1) / 2, (v0 + v1) / 2))
    return descriptor


def _references(obj):
    surfaces = []
    for parent, subnames in obj.References:
        names = [subnames] if isinstance(subnames, str) else (subnames or [""])
        for name in names:
            # Include containing App::Part placements when measuring references.
            shape = sub_shape_at_global_placement(parent, name)
            if shape is None or shape.isNull():
                raise ValueError("Constraint reference cannot be resolved")
            surfaces.append(_surface(shape))
    if not surfaces:
        raise ValueError("Constraint has no geometric references")
    return surfaces


def _contacts(objects):
    contacts = []
    for obj in objects:
        kind = getattr(getattr(obj, "Proxy", None), "Type", obj.TypeId)
        if kind not in ("Fem::ConstraintContact", "Fem::ConstraintTie"):
            continue
        surfaces = _references(obj)
        if len(surfaces) < 2:
            raise ValueError("Contact/Tie requires slave and master surfaces")
        if kind == "Fem::ConstraintTie":
            parameters = {
                "tolerance_mm": qty(obj.Tolerance, "mm"),
                "adjust": bool(obj.Adjust),
                "cyclic_symmetry": bool(obj.CyclicSymmetry),
            }
            if obj.CyclicSymmetry:
                parameters.update(
                    sectors=int(obj.Sectors),
                    connected_sectors=int(obj.ConnectedSectors),
                    symmetry_axis_base=list(obj.SymmetryAxis.Base),
                    symmetry_axis_rotation=list(obj.SymmetryAxis.Rotation.Q),
                )
        else:
            behavior = str(obj.SurfaceBehavior)
            parameters = {
                "surface_behavior": behavior,
                "adjust_mm": qty(obj.Adjust, "mm"),
                "friction": bool(obj.Friction),
            }
            if behavior in ("Linear", "Tied"):
                parameters["slope_MPa_per_mm"] = qty(obj.Slope, "MPa/mm")
            if obj.Friction:
                parameters.update(
                    friction_coefficient=float(obj.FrictionCoefficient),
                    stick_slope_MPa_per_mm=qty(obj.StickSlope, "MPa/mm"),
                )
            if getattr(obj, "EnableThermalContact", False):
                raise UnsupportedSetupError("Thermal contact is outside the supported static scope")
        contacts.append(
            {
                "type": "tie" if kind.endswith("Tie") else "contact",
                "slave": surfaces[:-1],
                "master": surfaces[-1:],
                "parameters": parameters,
            }
        )
    return contacts


def _conditions(objects):
    bcs, loads = [], []
    for obj in objects:
        kind = getattr(getattr(obj, "Proxy", None), "Type", obj.TypeId)
        if kind in (
            "Fem::ConstraintFixed",
            "Fem::ConstraintDisplacement",
            "Fem::ConstraintBearing",
        ):
            item = {
                "type": {
                    "Fem::ConstraintFixed": "fixed",
                    "Fem::ConstraintDisplacement": "displacement",
                    "Fem::ConstraintBearing": "support",
                }[kind],
                "surfaces": _references(obj),
            }
            if kind == "Fem::ConstraintDisplacement":
                item["parameters"] = {}
                for axis in "xyz":
                    item["parameters"][axis + "Fix"] = bool(getattr(obj, axis + "Fix"))
                    item["parameters"][axis + "Free"] = bool(getattr(obj, axis + "Free"))
                    if not getattr(obj, axis + "Free"):
                        item["parameters"][axis + "Displacement"] = qty(
                            getattr(obj, axis + "Displacement"), "mm"
                        )
            bcs.append(item)
        elif kind == "Fem::ConstraintForce":
            loads.append(
                {
                    "type": "force",
                    "magnitude_N": qty(obj.Force, "N"),
                    "direction": force_direction(obj),
                    "surfaces": _references(obj),
                }
            )
        elif kind == "Fem::ConstraintPressure":
            loads.append(
                {
                    "type": "pressure",
                    "magnitude_Pa": qty(obj.Pressure, "Pa"),
                    "reversed": bool(obj.Reversed),
                    "surfaces": _references(obj),
                }
            )
        elif kind == "Fem::ConstraintSelfWeight":
            loads.append(
                {
                    "type": "self_weight",
                    "direction": list(obj.GravityDirection),
                    "parameters": {"acceleration_mm_s2": qty(obj.GravityAcceleration, "mm/s^2")},
                }
            )
        elif kind == "Fem::ConstraintSectionPrint":
            # An output request, not a restraint or load. Replay still receives it.
            continue
        elif kind.startswith("Fem::Constraint") and kind not in (
            "Fem::ConstraintContact",
            "Fem::ConstraintTie",
        ):
            raise UnsupportedSetupError(f"Unsupported static constraint: {kind}")
    return bcs, loads


def _snapshot(result):
    fields = {}
    for name in ("DisplacementLengths", "vonMises", "MaxShear"):
        values = list(getattr(result, name, []) or [])
        if values:
            fields[name] = [float(value) for value in values]
    vectors = list(getattr(result, "DisplacementVectors", []) or [])
    if vectors:
        fields["DisplacementVectors"] = [tuple(value) for value in vectors]
    nodes = [int(value) for value in result.NodeNumbers]
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("Result node identifiers must be unique and nonempty")
    for name, values in fields.items():
        flat = [x for vector in values for x in vector] if name == "DisplacementVectors" else values
        if len(values) != len(nodes) or not all(math.isfinite(x) for x in flat):
            raise ValueError("Stored result fields must be finite and aligned with result nodes")
    if any(
        name not in fields for name in ("DisplacementLengths", "DisplacementVectors", "vonMises")
    ):
        raise ValueError("Stored static result is missing displacement or stress fields")
    return {"node_numbers": nodes, "fields": fields}


def _result_values(snapshot):
    fields = snapshot["fields"]
    results = {
        "max_displacement_mm": max(fields["DisplacementLengths"]),
        "max_von_mises_MPa": max(fields["vonMises"]),
    }
    if fields.get("MaxShear"):
        results["max_shear_MPa"] = max(fields["MaxShear"])
    return results


def _extract_document(doc):
    results = [obj for obj in doc.Objects if _is_result_object(obj)]
    if len(results) != 1:
        raise ValueError("Static scoring requires exactly one saved result object")
    result = results[0]
    snapshot = _snapshot(result)
    analysis, solver, mesh = find_replay_context(doc, result, len(snapshot["node_numbers"]))
    if str(solver.AnalysisType) != "static":
        raise ValueError("Static scoring requires a static solver")
    shape = linked_mesh_shape(mesh)
    if shape.isNull() or not shape.isValid() or not shape.Solids:
        raise ValueError("Solved mesh must link to valid solid geometry")
    if set(snapshot["node_numbers"]) != set(mesh.FemMesh.Nodes):
        raise ValueError("Stored result node identifiers differ from the saved source mesh")
    # Match FreeCAD's femtools.membertools.get_member suppression semantics.
    objects = [
        obj
        for obj in analysis.Group
        if not (obj.hasExtension("App::SuppressibleExtension") and obj.Suppressed)
    ]
    materials = extract_material_body_counts(objects, len(shape.Solids))
    bcs, loads = _conditions(objects)
    geometry = geometry_facts([shape])
    solids = shape.Solids
    volume = sum(solid.Volume for solid in solids)
    center = [
        sum(solid.Volume * getattr(solid.CenterOfMass, axis) for solid in solids) / volume
        for axis in ("x", "y", "z")
    ]
    geometry.update(
        centroid=center,
        bbox_min=[shape.BoundBox.XMin, shape.BoundBox.YMin, shape.BoundBox.ZMin],
        bbox_max=[shape.BoundBox.XMax, shape.BoundBox.YMax, shape.BoundBox.ZMax],
    )
    payload = {
        "analysis_type": "static",
        "geometry": geometry,
        "materials": materials,
        "units": {"length": "mm", "force": "N", "stress": "MPa"},
        "boundary_conditions": bcs,
        "loads": loads,
        "contacts": _contacts(objects),
        "mesh": {"num_nodes": mesh.FemMesh.NodeCount, "num_elements": mesh.FemMesh.VolumeCount},
        "results": _result_values(snapshot),
    }
    return payload, snapshot, analysis, solver
