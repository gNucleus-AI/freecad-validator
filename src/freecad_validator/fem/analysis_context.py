"""Locate the saved static analysis and its meshed geometry."""


def _is_result_object(obj):
    """A loaded mechanical FEM result (has nodal fields). Single source of truth
    for what counts as a 'result', shared by discovery, the pre-replay purge and
    replay-output selection so they can never disagree (a disagreement would be a
    hole: a result invisible to the purge but visible to selection)."""
    return obj.TypeId == "Fem::FemResultObjectPython" and getattr(obj, "NodeNumbers", None)


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


def linked_mesh_shape(mesh):
    """Read the selected meshed shape in world coordinates, including parent placements."""
    from femmesh.meshtools import sub_shape_at_global_placement

    link = getattr(mesh, "Shape", None) or getattr(mesh, "Part", None)
    if hasattr(link, "getGlobalPlacement"):
        return sub_shape_at_global_placement(link, "")
    return link
