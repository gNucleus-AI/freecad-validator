"""Read saved static analysis facts without changing geometry."""

import FreeCAD
from femmesh.meshtools import sub_shape_at_global_placement
from FreeCAD import Units

from freecad_validator.fem.material_counts import (
    grouped_material_counts,
    material_signature,
)


def qty(value, unit):
    """Convert a FreeCAD quantity/string/number to a float in `unit`."""
    try:
        return float(Units.Quantity(value).getValueAs(unit))
    except (ValueError, TypeError):
        return float(value)


def _is_result_object(obj):
    """Identify a saved mechanical FEM result with nodal fields."""
    return obj.TypeId == "Fem::FemResultObjectPython" and getattr(obj, "NodeNumbers", None)


def material_values(obj):
    material = dict(obj.Material)
    out = {"name": material.get("Name", "")}
    for source, target, unit in (
        ("YoungsModulus", "E_MPa", "MPa"),
        ("Density", "rho_kg_m3", "kg/m^3"),
    ):
        if material.get(source):
            out[target] = qty(material[source], unit)
    if material.get("PoissonRatio"):
        out["nu"] = float(material["PoissonRatio"])
    return out


def _add_distinct_solids(buckets, solids):
    """Deduplicate local topology references; no geometric/body correspondence."""
    for solid in solids:
        bucket = buckets.setdefault(solid.hashCode(), [])
        if not any(solid.isSame(previous) for previous in bucket):
            bucket.append(solid)


def extract_material_body_counts(objects, total_solids):
    """Count referenced solids per parameter card in the selected analysis.

    A single material covers every solid, even when it has explicit References.
    Otherwise empty References denotes the default for the remaining solids.
    Only local topology identity is used to deduplicate repeated references;
    this does not compare where materials occur in reference/candidate models.
    """
    materials = [
        obj for obj in objects if "Material" in obj.TypeId and getattr(obj, "Material", None)
    ]
    if len(materials) == 1:
        return grouped_material_counts(
            [{**material_values(materials[0]), "body_count": total_solids}]
        )
    groups, assigned = {}, {}
    default = None
    for obj in materials:
        values = material_values(obj)
        signature = material_signature(values)
        group = groups.setdefault(signature, {"values": values, "solids": {}})
        if not obj.References:
            if default is not None and default != signature:
                raise ValueError("Multiple materials have empty/default References")
            default = signature
            continue
        for parent, subnames in obj.References:
            for subname in subnames or ("",):
                # Keep the referenced topology: transforming a copy would give
                # repeated references different identities and inflate counts.
                shape = parent.getSubObject(subname)
                solids = shape.Solids
                if not solids:
                    raise ValueError(
                        f"Material reference {parent.Name}/{subname} contains no solids"
                    )
                _add_distinct_solids(group["solids"], solids)
                _add_distinct_solids(assigned, solids)
    assigned_count = sum(len(bucket) for bucket in assigned.values())
    if assigned_count > total_solids:
        raise ValueError("Material references contain more solids than the analysed geometry")
    counts = []
    for signature, group in groups.items():
        count = sum(len(bucket) for bucket in group["solids"].values())
        if signature == default:
            count += total_solids - assigned_count
        counts.append({**group["values"], "body_count": count})
    return grouped_material_counts(counts)


def linked_mesh_shape(mesh):
    """Read the selected meshed shape in world coordinates, including parent placements."""
    link = getattr(mesh, "Shape", None) or getattr(mesh, "Part", None)
    if hasattr(link, "getGlobalPlacement"):
        return sub_shape_at_global_placement(link, "")
    return link


def _unwrap_single_child_compounds(shape):
    """Ignore packaging while retaining the child's accumulated placement."""
    while shape.ShapeType == "Compound":
        children = shape.childShapes()
        if len(children) != 1:
            break
        shape = children[0]
    return shape


def geometry_facts(shapes):
    if not shapes:
        return {}
    # BooleanFragments Mode=CompSolid returns a Compound of CompSolids, even
    # for a single connected group. Its wrapper is not a geometry difference.
    shapes = [_unwrap_single_child_compounds(shape) for shape in shapes]
    solids = [solid for shape in shapes for solid in shape.Solids]
    volume = sum(s.Volume for s in shapes)
    surface_area = sum(s.Area for s in shapes)
    regions = sorted(
        (
            {
                "volume_mm3": solid.Volume,
                "surface_area_mm2": solid.Area,
                "num_faces": len(solid.Faces),
                "num_edges": len(solid.Edges),
                "num_shells": len(solid.Shells),
            }
            for solid in solids
        ),
        key=lambda region: (
            region["volume_mm3"],
            region["surface_area_mm2"],
            region["num_faces"],
            region["num_edges"],
        ),
    )
    bbs = [s.BoundBox for s in shapes]
    xs = [b.XMin for b in bbs] + [b.XMax for b in bbs]
    ys = [b.YMin for b in bbs] + [b.YMax for b in bbs]
    zs = [b.ZMin for b in bbs] + [b.ZMax for b in bbs]
    dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    return {
        "characteristic_length_mm": (dx * dx + dy * dy + dz * dz) ** 0.5,
        "bbox_mm": [dx, dy, dz],
        "volume_mm3": volume,
        "surface_area_mm2": surface_area,
        "num_solids": len(solids),
        "num_compsolids": sum(len(shape.CompSolids) for shape in shapes),
        "shape_types": sorted({str(shape.ShapeType) for shape in shapes}),
        # Count shared interfaces once in each analyzed shape. Summing the
        # regions double-counts imprinted faces/edges and hides Boolean
        # Fragments that join coincident interfaces without splitting faces.
        "num_faces": sum(len(shape.Faces) for shape in shapes),
        "num_edges": sum(len(shape.Edges) for shape in shapes),
        "regions": regions,
    }


def find_replay_context(doc, result, solved_node_count):
    analyses = [obj for obj in doc.Objects if obj.TypeId == "Fem::FemAnalysis"]
    matching_analyses = []
    for analysis in analyses:
        group = list(getattr(analysis, "Group", []) or [])
        if result in group or getattr(result, "Mesh", None) in group:
            matching_analyses.append(analysis)
    if len(matching_analyses) != 1:
        raise RuntimeError(
            "stored FEM result is not owned by exactly one analysis "
            f"(found {len(matching_analyses)})"
        )

    analysis = matching_analyses[0]
    group = list(getattr(analysis, "Group", []) or [])
    solvers = [obj for obj in group if "Solver" in obj.TypeId or "Ccx" in obj.TypeId]
    if len(solvers) != 1:
        raise RuntimeError(
            f"analysis must contain exactly one solver for replay (found {len(solvers)})"
        )

    source_meshes = []
    for obj in group:
        fem_mesh = getattr(obj, "FemMesh", None)
        node_count = getattr(fem_mesh, "NodeCount", 0) if fem_mesh is not None else 0
        if "FemMeshShape" in obj.TypeId and node_count == solved_node_count:
            source_meshes.append(obj)
    if len(source_meshes) != 1:
        raise RuntimeError(
            "analysis must contain exactly one source mesh matching the stored result "
            f"({solved_node_count} nodes; found {len(source_meshes)})"
        )
    return analysis, solvers[0], source_meshes[0]


def force_direction(o):
    """Unit direction the force actually points, or None.

    Lets the scorer check WHICH WAY the load points, not just its magnitude.
    DirectionVector is already the FINAL applied direction: the CalculiX writer
    (Mod/Fem/femsolver/calculix/write_constraint_force.py) forms the nodal load
    straight from DirectionVector and never references `Reversed`, so we must NOT
    negate it here. A load written as (DirectionVector, Reversed=True) is
    physically identical to the same DirectionVector with Reversed=False; the
    solved displacement field confirms CalculiX ignores `Reversed`."""
    dv = getattr(o, "DirectionVector", None)
    if dv is None or getattr(dv, "Length", 0) < 1e-9:
        return None
    u = FreeCAD.Vector(dv)
    u.normalize()
    return [u.x, u.y, u.z]
