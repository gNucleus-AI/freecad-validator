"""Geometry summaries shared by the STEP and FCStd adapters."""


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
