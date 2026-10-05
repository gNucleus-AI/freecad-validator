"""Automatic original-body correspondence for saved preprocessing assemblies.

Score saved inputs before Boolean operations, never their resulting fragments.
Verify those inputs against the actual analysis geometry before scoring.
Names, labels and visibility are not correspondence evidence. Match whole input
bodies in world coordinates against the raw originals.
"""

import logging
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import FreeCAD
import numpy as np
import Part
from scipy.optimize import linear_sum_assignment

from freecad_validator.fem.analysis_context import (
    _is_result_object,
    find_replay_context,
)
from freecad_validator.fem.pre_process.errors import (
    BodyCorrespondenceError,
    CandidateGeometryError,
    EvaluationError,
)
from freecad_validator.fem.pre_process.geometry import export_geometry, refine_geometry
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.methods.mesh_boolean import (
    tessellate_shape,
    to_manifold,
)
from freecad_validator.fem.pre_process.policy import aggregate_body_scores, score_body_edit


@dataclass
class BodyGeometry:
    shape: object
    source: str

    @cached_property
    def mesh(self):
        # This coarse mesh finds correspondence only; body scoring uses its own
        # established tolerances and changed-region samples, not these overlaps.
        try:
            # OCCT refinement can mutate shared input surfaces even when the
            # returned shape is valid. Preserve the body used for later scoring.
            mesh = to_manifold(*tessellate_shape(self.shape.copy().removeSplitter(), 0.05))
        except Part.OCCError:
            # Refinement/tessellation can fail on valid saved Boolean regions.
            # Correspondence can still use exact CAD intersections below.
            return None
        if mesh.is_empty() or mesh.volume() <= 0:
            return None
        return mesh


def world_shape(obj):
    shape = getattr(obj, "Shape", None)
    if isinstance(shape, Part.Shape) and shape.isNull():
        return shape.copy()
    return Part.getShape(obj, mat=obj.getGlobalPlacement().Matrix, transform=False).copy()


def solid_bodies(shape, source, error_type=EvaluationError):
    if not isinstance(shape, Part.Shape) or shape.isNull() or not shape.Solids:
        raise error_type(f"No solid geometry in {source}")
    bodies = []
    for index, solid in enumerate(shape.Solids):
        if not solid.isValid() or solid.Volume <= 0:
            raise error_type(f"Invalid solid {index + 1} in {source}")
        bodies.append(BodyGeometry(solid.copy(), f"{source}/Solid{index + 1}"))
    return bodies


def read_raw_bodies(path):
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"Raw STEP does not exist: {path}")
    shape = Part.Shape()
    shape.read(str(path))
    return solid_bodies(shape, str(path))


def boolean_input_objects(feature):
    """Read input links only; do not inspect or validate the Boolean result."""
    properties = set(feature.PropertiesList)
    if "PreprocessingInputs" in properties:
        return list(feature.PreprocessingInputs)
    if {"Objects", "Mode"} <= properties:
        return list(feature.Objects)
    if feature.TypeId == "Part::MultiFuse" and "Shapes" in properties:
        return list(feature.Shapes)
    if feature.TypeId in {"Part::Fuse", "Part::MultiCommon", "Part::Common"}:
        if "Shapes" in properties:
            return list(feature.Shapes)
        if {"Base", "Tool"} <= properties:
            return [feature.Base, feature.Tool]
    return None


def linked_boolean_inputs(feature, role):
    """Follow saved links through wrappers, stopping at the nearest input set."""
    pending = [feature]
    visited = set()
    while pending:
        found = {}
        following = []
        for obj in pending:
            if obj.Name in visited:
                continue
            visited.add(obj.Name)
            inputs = boolean_input_objects(obj)
            if inputs is not None:
                key = tuple(sorted(item.Name for item in inputs if item is not None))
                found[key] = inputs
            else:
                following.extend(obj.OutList)
        if len(found) > 1:
            raise BodyCorrespondenceError(
                "Multiple pre-Boolean input sets are linked to the analysis feature",
                role,
            )
        if found:
            return next(iter(found.values()))
        pending = following
    return None


def same_solid_geometry(left, right):
    """Recognize raw import history geometrically, independent of face splits."""
    tolerance = max(left.Volume, right.Volume, 1.0) * 1e-8
    if abs(left.Volume - right.Volume) > tolerance:
        return False
    if (left.CenterOfMass - right.CenterOfMass).Length > 1e-6:
        return False
    return left.cut(right).Volume + right.cut(left).Volume <= tolerance


def mesh_inputs_match_analysis(bodies, analyzed):
    """Independently check material when OCC differences are inconclusive.

    Some valid coincident surfaces produce inconsistent CAD Boolean results.
    Check each input and analysis solid separately so a large assembly cannot
    hide a missing small body in an assembly-wide relative tolerance.
    """
    analysis_bodies = [BodyGeometry(solid.copy(), "analysis") for solid in analyzed.Solids]
    if any(body.mesh is None for body in [*bodies, *analysis_bodies]):
        return False
    prepared = bodies[0].mesh
    for body in bodies[1:]:
        prepared = prepared + body.mesh
    actual = analysis_bodies[0].mesh
    for body in analysis_bodies[1:]:
        actual = actual + body.mesh
    return all(
        (body.mesh - actual).volume() <= max(1e-5, body.shape.Volume * 1e-8) for body in bodies
    ) and all(
        (body.mesh - prepared).volume() <= max(1e-5, body.shape.Volume * 1e-8)
        for body in analysis_bodies
    )


def verify_clean_inputs(bodies, analyzed, role):
    """Check occupied material only; analyzed regions never become scored bodies."""
    try:
        mismatch = any(
            body.shape.cut(analyzed).Volume > max(1e-5, body.shape.Volume * 1e-8) for body in bodies
        )
        if not mismatch:
            prepared = Part.makeCompound([body.shape for body in bodies])
            mismatch = analyzed.cut(prepared).Volume > max(1e-5, analyzed.Volume * 1e-8)
    except Part.OCCError as exc:
        if mesh_inputs_match_analysis(bodies, analyzed):
            return
        raise BodyCorrespondenceError(
            f"Cannot verify saved clean inputs against analysis geometry: {exc}",
            role,
        ) from exc
    if mismatch and not mesh_inputs_match_analysis(bodies, analyzed):
        error_type = CandidateGeometryError if role == "candidate" else EvaluationError
        raise error_type("Saved clean inputs do not match the actual analysis geometry")


def detached_clean_bodies(doc, analysis, raw, path, role, error_type):
    """Recover independent saved parts when a baked analysis lost input links.

    Never read the analysis shape or its dependencies/dependents. A complete
    raw import group can be identified against STEP and omitted when a separate
    prepared set exists. No choice is based on similarity to the reference.
    """
    excluded = {
        obj.Name for obj in [analysis, *analysis.OutListRecursive, *analysis.InListRecursive]
    }
    # Baked analysis containers often also store individual result regions as
    # siblings, without links from those regions to the compound snapshot.
    for parent in analysis.InListRecursive:
        if parent.TypeId in {"App::Part", "App::DocumentObjectGroup"}:
            excluded.update(obj.Name for obj in parent.OutListRecursive)
    # Recognizable Boolean outputs elsewhere in the document are not inputs.
    for obj in doc.Objects:
        if boolean_input_objects(obj) is not None:
            excluded.update(item.Name for item in [obj, *obj.InListRecursive])
    objects = [
        obj
        for obj in doc.Objects
        if obj.Name not in excluded and obj.isDerivedFrom("Part::Feature")
    ]
    names = {obj.Name for obj in objects}
    # Retain terminal features, not intermediate construction history or tips
    # already represented by a PartDesign Body.
    objects = [
        obj for obj in objects if not any(parent.Name in names for parent in obj.InListRecursive)
    ]
    saved = {}
    for obj in objects:
        shape = world_shape(obj)
        if shape.isNull() or not shape.Solids:
            continue  # Direction lines and other non-solid helper objects.
        saved[obj.Name] = solid_bodies(shape, f"{path}:{obj.Name}", error_type)
    if not saved:
        return []

    raw_groups = set()
    if raw:
        for group in doc.Objects:
            if group.TypeId not in {"App::Part", "App::DocumentObjectGroup"}:
                continue
            members = frozenset(obj.Name for obj in group.OutListRecursive if obj.Name in saved)
            parts = [body for name in members for body in saved[name]]
            if len(parts) != len(raw) or len(members) == len(saved):
                continue
            unmatched = list(raw)
            for body in parts:
                matches = [
                    i
                    for i, original in enumerate(unmatched)
                    if same_solid_geometry(body.shape, original.shape)
                ]
                if len(matches) != 1:
                    break
                unmatched.pop(matches[0])
            else:
                raw_groups.add(members)
    if len(raw_groups) > 1:
        raise BodyCorrespondenceError("Multiple detached raw import histories", role)
    if raw_groups:
        for name in next(iter(raw_groups)):
            del saved[name]
    bodies = [body for parts in saved.values() for body in parts]
    # Alternative saved versions cannot be selected by which one scores better.
    # Touching parts are allowed; substantially overlapping detached versions
    # need input links to resolve their provenance.
    for index, body in enumerate(bodies):
        for other in bodies[:index]:
            if not body.shape.BoundBox.intersect(other.shape.BoundBox):
                continue
            overlap = body.shape.common(other.shape).Volume
            if overlap > 0.5 * min(body.shape.Volume, other.shape.Volume):
                raise BodyCorrespondenceError(
                    "Ambiguous overlapping detached pre-Boolean objects",
                    role,
                )
    logging.info("Recovered %d detached pre-Boolean bodies from %s", len(bodies), path)
    return bodies


def read_clean_bodies(path, *, candidate=False, raw=None):
    """Read saved pre-Boolean inputs, including legacy detached clean objects.

    Input links take precedence; detached parts provide a legacy fallback.
    Missing or ambiguous inputs remain an evaluation error.
    The selected inputs must occupy the same material as the analysis geometry.
    Analysis regions are used for verification only, never body correspondence.
    """
    path = Path(path)
    role = "candidate" if candidate else "reference"
    error_type = CandidateGeometryError if candidate else EvaluationError
    if not path.is_file():
        raise error_type(f"Clean FCStd does not exist: {path}")
    doc = FreeCAD.openDocument(str(path))
    try:
        meshes = [obj for obj in doc.Objects if "FemMeshShape" in obj.TypeId]
        result = next((obj for obj in doc.Objects if _is_result_object(obj)), None)
        if result is not None:
            _, _, source_mesh = find_replay_context(doc, result, len(result.DisplacementLengths))
            meshes = [source_mesh]
        if len(meshes) != 1:
            raise error_type(f"Expected one source FEM mesh in {path}, found {len(meshes)}")
        link = getattr(meshes[0], "Shape", None) or getattr(meshes[0], "Part", None)
        if isinstance(link, tuple):
            link = link[0]
        if link is None or not hasattr(link, "PropertiesList"):
            raise BodyCorrespondenceError("Missing link to pre-Boolean input objects", role)
        analyzed = world_shape(link)
        analyzed_bodies = solid_bodies(analyzed, f"{path}:{link.Name}", error_type)
        # A native single-body model has no assembly Boolean result to unwrap.
        if link.TypeId in {
            "Part::Box",
            "Part::Cylinder",
            "Part::Sphere",
            "Part::Cone",
            "Part::Torus",
        } or link.isDerivedFrom("PartDesign::Body"):
            return analyzed_bodies, False
        inputs = linked_boolean_inputs(link, role)
        if inputs is None:
            # Single-part preprocessing may store the prepared solid directly
            # on the mesh feature while retaining the unmodified STEP import.
            # Use that whole solid, not the detached raw copy. Do not unwrap a
            # Compound/CompSolid or extend this to a multi-original assembly.
            if (
                raw is not None
                and len(raw) == 1
                and link.TypeId == "Part::Feature"
                and not link.OutList
            ):
                if analyzed.ShapeType == "Solid":
                    return analyzed_bodies, False
            bodies = detached_clean_bodies(doc, link, raw, path, role, error_type)
            if bodies:
                verify_clean_inputs(bodies, analyzed, role)
                return bodies, False
        if not inputs:
            raise BodyCorrespondenceError(
                "Pre-Boolean input objects are not saved on the selected analysis feature; "
                "post-Boolean geometry is not used for preprocessing scoring",
                role,
            )
        bodies = []
        for obj in inputs:
            if obj is None or obj == link:
                raise BodyCorrespondenceError("Invalid pre-Boolean input link", role)
            bodies.extend(solid_bodies(world_shape(obj), f"{path}:{obj.Name}", error_type))
        verify_clean_inputs(bodies, analyzed, role)
        return bodies, False
    finally:
        FreeCAD.closeDocument(doc.Name)


def overlap_matrix(anchors, parts):
    overlaps = np.zeros((len(anchors), len(parts)))
    for i, anchor in enumerate(anchors):
        if anchor is None:
            continue
        for j, part in enumerate(parts):
            if anchor.shape.BoundBox.intersect(part.shape.BoundBox):
                if anchor.mesh is not None and part.mesh is not None:
                    volume = float((anchor.mesh ^ part.mesh).volume())
                else:
                    # Some valid STEP solids have non-watertight cached facets.
                    # Use the CAD intersection for correspondence in that case.
                    volume = anchor.shape.common(part.shape).Volume
                overlaps[i, j] = max(0.0, volume)
    return overlaps


def descriptor_distance(left, right):
    """Position-independent fallback for moved parts or failed CAD intersections.

    Used only for unmatched originals/solids, never to award geometry credit.
    Actual location/orientation is still compared by PreProcessScorer.
    """
    a, b = left.shape, right.shape
    extent_a = sorted((a.BoundBox.XLength, a.BoundBox.YLength, a.BoundBox.ZLength))
    extent_b = sorted((b.BoundBox.XLength, b.BoundBox.YLength, b.BoundBox.ZLength))
    extent = (
        sum(
            abs(math.log(max(x, 1e-9) / max(y, 1e-9)))
            for x, y in zip(extent_a, extent_b, strict=False)
        )
        / 3
    )
    return (
        0.4 * abs(math.log(a.Volume / b.Volume))
        + 0.2 * abs(math.log(a.Area / b.Area))
        + 0.4 * extent
    )


def regroup_bodies(raw, parts, fragmented, reference=None):
    """Keep correspondence failures distinct from invalid submitted geometry."""
    role = "reference" if reference is None else "candidate"
    try:
        return _regroup_bodies(raw, parts, fragmented, reference)
    except (EvaluationError, Part.OCCError) as exc:
        raise BodyCorrespondenceError(str(exc), role) from exc


def _regroup_bodies(raw, parts, fragmented, reference=None):
    """Match saved whole input bodies one-to-one without rebuilding fragments."""
    if fragmented:
        raise EvaluationError("Post-Boolean fragments are not preprocessing inputs")
    overlaps = overlap_matrix(raw, parts)
    if reference is not None:
        overlaps = np.maximum(overlaps, overlap_matrix(reference, parts))
    volumes = np.array([body.shape.Volume for body in parts])
    raw_volumes = np.array([body.shape.Volume for body in raw])
    if reference is not None:
        raw_volumes = np.maximum(
            raw_volumes, [body.shape.Volume if body else 0 for body in reference]
        )
    ownership = [[] for _ in raw]
    contributions = [[] for _ in raw]
    assigned = set()
    if parts:
        quality = overlaps / np.maximum(np.minimum(raw_volumes[:, None], volumes[None, :]), 1e-12)
        rows, columns = linear_sum_assignment(-quality)
        for i, j in zip(rows, columns, strict=False):
            if quality[i, j] >= 0.1:
                ownership[i].append(int(j))
                contributions[i].append(parts[j].shape)
                assigned.add(int(j))
    missing = [i for i, indices in enumerate(ownership) if not indices]
    remaining = [j for j in range(len(parts)) if j not in assigned]
    if missing and remaining:
        costs = np.array(
            [
                [
                    min(
                        descriptor_distance(raw[i], parts[j]),
                        descriptor_distance(reference[i], parts[j])
                        if reference is not None and reference[i] is not None
                        else float("inf"),
                    )
                    for j in remaining
                ]
                for i in missing
            ]
        )
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns, strict=False):
            cost = costs[row, column]
            alternatives = np.delete(costs[:, column], row)
            if cost > 0.3 or np.any(alternatives <= cost + 0.02):
                continue
            i, j = missing[row], remaining[column]
            ownership[i].append(j)
            contributions[i].append(parts[j].shape)
            assigned.add(j)
    rebuilt = []
    for i, indices in enumerate(ownership):
        if not indices:
            rebuilt.append(None)
            continue
        shapes = contributions[i]
        shape = shapes[0].copy()
        shape = refine_geometry(shape)
        if shape.isNull() or not shape.isValid() or not shape.Solids:
            raise EvaluationError(f"Cannot reconstruct original body {i + 1}")
        rebuilt.append(BodyGeometry(shape, f"Body{i + 1}"))
    return rebuilt, [part for j, part in enumerate(parts) if j not in assigned], ownership


def score_assembly(raw_path, reference_path, candidate_path, scorer):
    raw = read_raw_bodies(raw_path)
    ref_parts, ref_fragmented = read_clean_bodies(reference_path, raw=raw)
    reference, extra_reference, ref_map = regroup_bodies(raw, ref_parts, ref_fragmented)
    if extra_reference:
        raise EvaluationError(
            f"Reference has {len(extra_reference)} solids without an original-body correspondence"
        )
    answer_parts, answer_fragmented = read_clean_bodies(candidate_path, candidate=True, raw=raw)
    answer, extra_answer, answer_map = regroup_bodies(
        raw, answer_parts, answer_fragmented, reference
    )
    results = {}
    with TemporaryDirectory(prefix="fem-original-bodies-") as directory:
        for i, (original, target, candidate) in enumerate(
            zip(raw, reference, answer, strict=False)
        ):
            paths = []
            for role, body in zip(
                ("raw", "reference", "candidate"), (original, target, candidate), strict=False
            ):
                path = Path(directory) / f"{i + 1}_{role}.FCStd"
                if body is not None:
                    export_geometry(body.shape, path)
                paths.append(path if body is not None else None)
            results[f"Body{i + 1}"] = scorer.score_detailed(*paths)
            logging.info("Preprocessing: scored original body %d/%d", i + 1, len(raw))
    for i in range(len(extra_answer)):
        results[f"ExtraBody{i + 1}"] = score_body_edit(False, True)
    return {
        **aggregate_body_scores(results),
        "correspondence": {
            "original_body_count": len(raw),
            "reference_fragmented": ref_fragmented,
            "candidate_fragmented": answer_fragmented,
            "reference": ref_map,
            "candidate": answer_map,
            "extra_candidate_body_count": len(extra_answer),
        },
    }
