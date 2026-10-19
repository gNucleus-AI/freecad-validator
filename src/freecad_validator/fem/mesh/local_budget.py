"""Read-only node attribution using the bodies already matched by preprocessing."""

import logging
from pathlib import Path

import FreeCAD
import numpy as np

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.freecad_io import _is_result_object, find_replay_context
from freecad_validator.fem.mesh.budget import score_local_node_budget
from freecad_validator.fem.pre_process.errors import CandidateGeometryError
from freecad_validator.fem.pre_process.geometry_compare.sampling import _inside_many


def body_node_counts(path, bodies, role, *, target_body=None):
    """Assign every tetrahedron by mapped interior points and union its node IDs.

    Geometry comes from the existing correspondence, in world coordinates.
    All tetrahedra are checked, regardless of submitted mesh-group labels.
    Shared volume regions and interface nodes count in each owning body.
    """
    error_type = CandidateGeometryError if role == "candidate" else EvaluationError
    doc = FreeCAD.openDocument(str(Path(path).resolve()))
    try:
        result = next((obj for obj in doc.Objects if _is_result_object(obj)), None)
        if result is None:
            raise error_type("Body mesh budget requires a saved mechanical result")
        _, _, mesh_object = find_replay_context(doc, result, len(result.DisplacementLengths))
        return count_mesh_nodes(mesh_object.FemMesh, bodies, error_type, target_body=target_body)
    finally:
        FreeCAD.closeDocument(doc.Name)


def count_mesh_nodes(mesh, bodies, error_type, *, target_body=None):
    """Inspect saved Tet4/Tet10 connectivity without modifying the mesh."""
    nodes = mesh.Nodes
    elements = [tuple(mesh.getElementNodes(element)) for element in mesh.Volumes]
    if not elements:
        raise error_type("Body mesh budget requires volume elements")
    if any(
        len(element) not in (4, 10) or len(set(element)) != len(element) for element in elements
    ):
        raise error_type("Body mesh budget supports nondegenerate Tet4/Tet10 connectivity only")
    used = set().union(*elements)
    if used != set(nodes):
        raise error_type("Saved volume connectivity must cover exactly the source mesh nodes")
    centers = np.array([mapped_point(nodes, element, (0.25,) * 4) for element in elements])
    if not np.isfinite(centers).all():
        raise error_type("Nonfinite volume element coordinates")
    covered = np.zeros(len(elements), dtype=bool)
    node_sets = {body_id: set() for body_id, body in bodies.items() if body is not None}
    for body_id, body in bodies.items():
        if body is None:
            continue
        # Analytical boxes avoid cached display triangulation bounds.
        box = body.bounds
        lower = np.array([box.XMin, box.YMin, box.ZMin])
        upper = np.array([box.XMax, box.YMax, box.ZMax])
        indices = np.flatnonzero(((centers >= lower) & (centers <= upper)).all(axis=1))
        occupied = _inside_many(body.shape, centers[indices].tolist(), 0.0)
        owned = indices[np.asarray(occupied, dtype=bool)]
        covered[owned] = True
        for index in owned:
            node_sets[body_id].update(elements[index])
    # Curved boundary elements approximate CAD surfaces; even their mapped
    # centers can fall outside the exact BRep. Use the four standard interior
    # tetrahedral quadrature points for these elements. Accept only consistent,
    # nonempty ownership evidence, never nearest-body guesses or larger tolerances.
    for index in np.flatnonzero(~covered):
        a = (5 + 3 * np.sqrt(5)) / 20
        b = (5 - np.sqrt(5)) / 20
        points = [
            mapped_point(nodes, elements[index], tuple(a if j == i else b for j in range(4)))
            for i in range(4)
        ]
        owner_sets = [set() for _ in points]
        for body_id, body in bodies.items():
            if body is None:
                continue
            for owners, inside in zip(
                owner_sets, _inside_many(body.shape, points, 0.0), strict=True
            ):
                if inside:
                    owners.add(body_id)
        evidence = [owners for owners in owner_sets if owners]
        if evidence and all(owners == evidence[0] for owners in evidence):
            covered[index] = True
            for body_id in evidence[0]:
                node_sets[body_id].update(elements[index])
    unresolved = np.flatnonzero(~covered)
    if target_body is not None and len(unresolved):
        target = bodies.get(target_body)
        if target is None:
            return {}
        nearby = [
            index
            for index in unresolved
            if element_overlaps_bounds(nodes, elements[index], target.bounds)
        ]
        logging.info(
            "Local mesh target %s: %d unresolved elements provably outside target bounds",
            target_body,
            len(unresolved) - len(nearby),
        )
        unresolved = nearby
    if len(unresolved):
        raise error_type(f"Cannot assign {len(unresolved)} volume elements to matched clean bodies")
    return {body_id: len(ids) for body_id, ids in node_sets.items()}


def element_overlaps_bounds(nodes, element, bounds):
    """Conservative Tet4/Tet10 bounds, using the quadratic Bezier control hull.

    A curved tetrahedron may extend beyond its nodal box. Bernstein control
    points bound the entire element, so outside elements can be excluded
    without silently discarding unresolved elements near the target.
    """
    points = np.array([tuple(nodes[node]) for node in element])
    controls = list(points[:4])
    if len(element) == 10:
        edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        controls.extend(
            2 * points[k + 4] - (points[i] + points[j]) / 2 for k, (i, j) in enumerate(edges)
        )
    controls = np.asarray(controls)
    lower = np.array([bounds.XMin, bounds.YMin, bounds.ZMin])
    upper = np.array([bounds.XMax, bounds.YMax, bounds.ZMax])
    return bool(np.all(controls.max(axis=0) >= lower) and np.all(controls.min(axis=0) <= upper))


def target_geometry(body):
    """Measure the trusted body's locator without modifying its saved shape."""
    box = body.bounds
    return {
        "volume_mm3": float(body.shape.Volume),
        "center_mm": list(body.shape.CenterOfMass),
        "bounds_mm": [box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax],
    }


def verify_target_geometry(body_id, geometry, bodies):
    """Require the saved locator to identify exactly the configured reference body.

    These tight numerical comparisons validate trusted metadata only. Candidate
    geometry is matched by the existing preprocessing correspondence instead.
    """
    matches = []
    for name, body in bodies.items():
        if body is None:
            continue
        box = body.bounds
        bounds = [box.XMin, box.YMin, box.ZMin, box.XMax, box.YMax, box.ZMax]
        if not np.allclose(bounds, geometry["bounds_mm"], rtol=1e-8, atol=1e-6):
            continue
        measured = target_geometry(body)
        if all(
            np.allclose(measured[key], value, rtol=1e-8, atol=1e-6)
            for key, value in geometry.items()
        ):
            matches.append(name)
    if matches != [body_id]:
        raise EvaluationError(
            "Target geometry does not uniquely identify the configured reference body"
        )


def mapped_point(nodes, element, barycentric):
    """Evaluate the saved tetrahedron's isoparametric geometry, including midnodes."""
    weights = list(barycentric)
    if len(element) == 10:
        weights = [value * (2 * value - 1) for value in barycentric]
        # FreeCAD/SMESH Tet10 edge order, also used by its volume-face tables.
        edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        weights.extend(4 * barycentric[i] * barycentric[j] for i, j in edges)
    return tuple(np.asarray(weights) @ np.array([tuple(nodes[node]) for node in element]))


def evaluate_local_mesh_budget(budget, reference_path, candidate_path, targets, answer, extras):
    """Use the current preprocessing match; never perform a second body match."""
    reference_bodies = {f"Body{i + 1}": body for i, body in enumerate(targets)}
    candidate_bodies = {f"Body{i + 1}": body for i, body in enumerate(answer)}
    candidate_bodies.update({f"ExtraBody{i + 1}": body for i, body in enumerate(extras)})
    body_id, entry = next(iter(budget["local_max"].items()))
    geometry = entry.get("geometry") if isinstance(entry, dict) else None
    if geometry is not None:
        verify_target_geometry(body_id, geometry, reference_bodies)
    reference_counts = body_node_counts(
        reference_path, reference_bodies, "reference", target_body=body_id
    )
    # Validate the trusted target and cap even if candidate attribution fails.
    score_local_node_budget(budget, reference_counts, {})
    try:
        candidate_counts = body_node_counts(
            candidate_path, candidate_bodies, "candidate", target_body=body_id
        )
    except CandidateGeometryError as exc:
        return {"score": 0.0, "status": "candidate_invalid", "error": str(exc)}
    evaluation = score_local_node_budget(budget, reference_counts, candidate_counts)
    if geometry is not None:
        evaluation["reference_geometry"] = geometry
        evaluation["reference_geometry_verified"] = True
    return evaluation
