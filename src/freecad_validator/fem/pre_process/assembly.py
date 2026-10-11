"""Automatic original-body correspondence for saved preprocessing assemblies.

Require verified saved pre-Boolean clean bodies. Never reconstruct bodies from
analysis fragments or fused results. Names, labels and visibility are not
correspondence evidence; all geometry stays in world coordinates.
"""

import logging
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import FreeCAD
import numpy as np
import Part
from scipy.optimize import linear_sum_assignment

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.freecad_io import (
    _is_result_object,
    find_replay_context,
)
from freecad_validator.fem.pre_process.errors import (
    BodyCorrespondenceError,
    CandidateGeometryError,
    MissingCleanBodiesError,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    sample_tolerance,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import DiffConfig
from freecad_validator.fem.pre_process.geometry_compare.sampling import (
    _halton_points,
    _inside,
    _inside_many,
    point_in_bounds,
)
from freecad_validator.fem.pre_process.policy import aggregate_body_scores, score_body_edit
from freecad_validator.fem.pre_process.spatial import (
    SpatialShape,
    sampled_surfaces_match,
    shape_bounds,
)


@dataclass
class BodyGeometry:
    shape: object
    source: str
    _probe_batches: list = field(default_factory=list, init=False, repr=False, compare=False)

    @cached_property
    def bounds(self):
        return shape_bounds([self.shape])

    def _matching_batches(self):
        """Extend one deterministic probe sequence, reusing it across body pairs."""
        box = self.bounds
        bounds = ((box.XMin, box.YMin, box.ZMin), (box.XMax, box.YMax, box.ZMax))
        previous, count, index = 0, 256, 0
        while count <= 65536:
            if index == len(self._probe_batches):
                points = _halton_points(count, bounds)[previous:]
                occupied = [
                    point
                    for point, inside in zip(
                        points,
                        _inside_many(self.shape, points, 0.0),
                        strict=True,
                    )
                    if inside
                ]
                self._probe_batches.append(occupied)
            yield self._probe_batches[index], count
            previous, count, index = count, count * 2, index + 1


def world_shape(obj):
    """Retain saved topology for read-only queries; deep copies can change validity."""
    shape = getattr(obj, "Shape", None)
    if isinstance(shape, Part.Shape) and shape.isNull():
        return shape
    return Part.getShape(obj, mat=obj.getGlobalPlacement().Matrix, transform=False)


def solid_bodies(shape, source, error_type=EvaluationError):
    if not isinstance(shape, Part.Shape) or shape.isNull() or not shape.Solids:
        raise error_type(f"No solid geometry in {source}")
    bodies = []
    for index, solid in enumerate(shape.Solids):
        if not solid.isValid() or solid.Volume <= 0:
            raise error_type(f"Invalid solid {index + 1} in {source}")
        bodies.append(BodyGeometry(solid, f"{source}/Solid{index + 1}"))
    return bodies


def read_raw_bodies(path):
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"Raw STEP does not exist: {path}")
    shape = Part.Shape()
    shape.read(str(path))
    return solid_bodies(shape, str(path))


def saved_clean_input_objects(feature):
    """Read saved clean-input links without executing the feature."""
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


def linked_clean_inputs(feature, role):
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
            inputs = saved_clean_input_objects(obj)
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
    scale = max(left.BoundBox.DiagonalLength, right.BoundBox.DiagonalLength, 1.0)
    config = DiffConfig()
    return sampled_surfaces_match(left, right, config, sample_tolerance(config, scale))


def history_matches_analysis(inputs, analyzed):
    """Coarsely verify saved inputs with whole-assembly sampled IoU >= 0.95.

    Both occupied unions are queried on the same Halton points. Overlapping
    solids count once; neither a Boolean shape nor a per-body score is built.
    Small local differences may pass this precheck by design. Insufficient
    occupied samples remain an evaluation error, never an assumed match.
    """
    if not inputs or not analyzed:
        return False
    prepared = SpatialShape(body.shape for body in inputs)
    actual = SpatialShape(body.shape for body in analyzed)
    box = shape_bounds((prepared, actual))
    bounds = ((box.XMin, box.YMin, box.ZMin), (box.XMax, box.YMax, box.ZMax))
    config = DiffConfig()
    occupied, intersection, previous = 0, 0, 0
    count = config.region_sample_count
    while True:
        for point in _halton_points(count, bounds)[previous:]:
            in_prepared = _inside(prepared, point, 0.0)
            in_actual = _inside(actual, point, 0.0)
            occupied += in_prepared or in_actual
            intersection += in_prepared and in_actual
        if occupied >= 64:
            break
        if count == 65536:
            raise ValueError(f"Insufficient occupied probes for saved-input IoU: {occupied}")
        previous, count = count, min(count * 4, 65536)
    iou = intersection / occupied
    logging.info(
        "Saved-input sampled IoU: %.6f (%d intersection / %d union; %d probes)",
        iou,
        intersection,
        occupied,
        count,
    )
    return iou >= 0.95


def verify_clean_inputs(bodies, analyzed, role):
    """Check occupied material only; analyzed regions never become scored bodies."""
    try:
        actual = [BodyGeometry(solid, "analysis") for solid in analyzed.Solids]
        mismatch = not history_matches_analysis(bodies, actual)
    except (Part.OCCError, ValueError) as exc:
        raise BodyCorrespondenceError(
            f"Cannot verify saved clean inputs against analysis geometry: {exc}",
            role,
        ) from exc
    if mismatch:
        raise MissingCleanBodiesError(
            "Saved clean inputs do not match the actual analysis geometry",
            role,
        )


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
        if saved_clean_input_objects(obj) is not None:
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
    return [body for parts in saved.values() for body in parts]


def read_clean_bodies(path, *, candidate=False, raw=None):
    """Read saved clean bodies; missing history never falls back to analysis regions."""
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
            raise MissingCleanBodiesError("Missing link to saved clean bodies", role)
        analyzed = world_shape(link)
        analyzed_bodies = solid_bodies(analyzed, f"{path}:{link.Name}", error_type)
        # A native single-body model has no assembly Boolean result to unwrap.
        if (
            raw is not None
            and len(raw) == 1
            and analyzed.ShapeType == "Solid"
            and (
                link.TypeId
                in {"Part::Box", "Part::Cylinder", "Part::Sphere", "Part::Cone", "Part::Torus"}
                or link.isDerivedFrom("PartDesign::Body")
            )
        ):
            return analyzed_bodies
        inputs = linked_clean_inputs(link, role)
        if inputs is None:
            for feature in doc.Objects:
                history = saved_clean_input_objects(feature)
                if not history or any(obj is None or obj == link for obj in history):
                    continue
                bodies = [
                    body
                    for obj in history
                    for body in solid_bodies(world_shape(obj), f"{path}:{obj.Name}", error_type)
                ]
                if history_matches_analysis(bodies, analyzed_bodies):
                    return bodies
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
                    return analyzed_bodies
            bodies = verified_detached_bodies(doc, link, analyzed, raw, path, role, error_type)
            if bodies is not None:
                return bodies
            raise MissingCleanBodiesError(
                "No verified independent pre-Boolean clean bodies are saved",
                role,
            )
        if not inputs:
            raise MissingCleanBodiesError(
                "Pre-Boolean input objects are explicitly empty on the selected analysis feature",
                role,
            )
        bodies = []
        for obj in inputs:
            if obj is None or obj == link:
                raise MissingCleanBodiesError("Invalid pre-Boolean input link", role)
            bodies.extend(solid_bodies(world_shape(obj), f"{path}:{obj.Name}", error_type))
        verify_clean_inputs(bodies, analyzed, role)
        return bodies
    finally:
        FreeCAD.closeDocument(doc.Name)


def verified_detached_bodies(doc, link, analyzed, raw, path, role, error_type):
    """Use detached history only when unambiguous and consistent with analysis."""
    try:
        bodies = detached_clean_bodies(doc, link, raw, path, role, error_type)
        if bodies:
            # Reject stale construction history before its costly all-pairs
            # ownership check. Both checks still apply to accepted history.
            verify_clean_inputs(bodies, analyzed, role)
            # Touching parts are allowed. Substantially overlapping alternative
            # versions need explicit links to establish their provenance.
            overlaps = overlap_matrix(bodies, bodies)
            for index, body in enumerate(bodies):
                for other_index, other in enumerate(bodies[:index]):
                    if overlaps[index, other_index] > 0.5 * min(
                        body.shape.Volume, other.shape.Volume
                    ):
                        raise MissingCleanBodiesError(
                            "Ambiguous overlapping detached pre-Boolean objects",
                            role,
                        )
            logging.info(
                "Recovered %d verified detached pre-Boolean bodies from %s", len(bodies), path
            )
            return bodies
    except MissingCleanBodiesError as exc:
        # Detached objects can be stale history, rather than the saved clean set.
        logging.info("Detached input history unavailable in %s: %s", path, exc)
        return None
    return None


def overlap_matrix(anchors, parts):
    overlaps = np.zeros((len(anchors), len(parts)))
    for i, anchor in enumerate(anchors):
        if anchor is None:
            continue
        if isinstance(anchor.shape, SpatialShape):
            # Saved disjoint pieces are sparse inside their combined box.
            # Match their native solids separately, retaining one score owner.
            pieces = [BodyGeometry(shape, anchor.source) for shape in anchor.shape.parts]
            overlaps[i] = np.minimum(
                overlap_matrix(pieces, parts).sum(axis=0), [part.shape.Volume for part in parts]
            )
            continue
        for j, part in enumerate(parts):
            if anchors is parts and j < i:
                overlaps[i, j] = overlaps[j, i]
                continue
            if anchor is part:
                overlaps[i, j] = anchor.shape.Volume
                continue
            if anchor.bounds.intersect(part.bounds):
                source, target = (
                    (anchor, part) if anchor.shape.Volume < part.shape.Volume else (part, anchor)
                )
                volume = source.shape.Volume
                # Only refine near decisions still made by the consumers:
                # correspondence ownership and detached-history ambiguity.
                thresholds = {
                    fraction * body.shape.Volume / volume
                    for fraction in (0.1, 0.5)
                    for body in (anchor, part)
                }
                fraction, _ = sampled_overlap_fraction(source, target, thresholds)
                overlaps[i, j] = volume * fraction
    return overlaps


def sampled_overlap_fraction(source, target, thresholds):
    """Incrementally refine correspondence near decision thresholds.

    A conservative uncertainty band avoids exhaustive queries for obvious empty
    or full overlaps. It is a finite-sampling heuristic, not an equivalence proof;
    final geometry scoring and saved-input verification use their full budgets.
    """
    thresholds = [value for value in thresholds if 0 < value < 1]
    hits, count = 0, 0
    for points, tested in source._matching_batches():
        batch = [point for point in points if point_in_bounds(point, target.bounds, 0.0)]
        hits += sum(_inside_many(target.shape, batch, 0.0))
        count += len(points)
        if count < 64:
            continue
        center = (hits + 8) / (count + 16)
        radius = 4 * math.sqrt(hits * (count - hits) / count + 4) / (count + 16)
        if not any(center - radius <= value <= center + radius for value in thresholds):
            return hits / count, True
        # Keep the existing detailed budget; sparse solids may need more box
        # probes to reach the required number of occupied points.
        if tested >= 8192:
            return hits / count, False
    raise ValueError("Insufficient occupied probes for body correspondence")


def descriptor_distance(left, right):
    """Position-independent fallback for moved parts with no sampled overlap.

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


def match_clean_bodies(raw, parts, reference=None):
    """Match saved whole bodies without creating or recovering geometry."""
    role = "reference" if reference is None else "candidate"
    try:
        return _match_clean_bodies(raw, parts, reference)
    except (EvaluationError, Part.OCCError, ValueError) as exc:
        raise BodyCorrespondenceError(str(exc), role) from exc


def _match_clean_bodies(raw, parts, reference=None):
    """Return saved bodies grouped by original, extras, and saved-input indices."""
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
    assigned = set()
    if parts:
        quality = overlaps / np.maximum(np.minimum(raw_volumes[:, None], volumes[None, :]), 1e-12)
        rows, columns = linear_sum_assignment(-quality)
        for i, j in zip(rows, columns, strict=False):
            if quality[i, j] >= 0.1:
                ownership[i].append(int(j))
                assigned.add(int(j))
    # A cut may leave several saved, disjoint pieces of the same original.
    # Retain the existing solids as an occupancy view; never rebuild a BRep.
    for j, part in enumerate(parts):
        if j in assigned:
            continue
        owners = np.flatnonzero(overlaps[:, j] / max(volumes[j], 1e-12) > 0.8)
        if len(owners) != 1:
            continue
        i = int(owners[0])
        if not ownership[i]:
            continue
        fraction, _ = sampled_overlap_fraction(part, raw[i], (0.8,))
        if fraction <= 0.8:
            continue
        siblings = [parts[k] for k in ownership[i]]
        if np.any(overlap_matrix([part], siblings) > max(1e-6, volumes[j] * 1e-8)):
            continue
        ownership[i].append(j)
        assigned.add(j)
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
            # Repeated parts can share descriptors after a requested move.
            # Keep the one-to-one assignment; final geometry scoring checks
            # the candidate against the reference in world coordinates.
            if cost > 0.3:
                continue
            i, j = missing[row], remaining[column]
            ownership[i].append(j)
            assigned.add(j)
    matched = [
        (
            BodyGeometry(SpatialShape(parts[j].shape for j in indices), "saved pieces")
            if len(indices) > 1
            else parts[indices[0]]
        )
        if indices
        else None
        for indices in ownership
    ]
    return matched, [part for j, part in enumerate(parts) if j not in assigned], ownership


def score_assembly(raw_path, reference_path, candidate_path, scorer):
    raw = read_raw_bodies(raw_path)
    try:
        ref_parts = read_clean_bodies(reference_path, raw=raw)
        reference, extra_reference, ref_map = match_clean_bodies(raw, ref_parts)
        answer_parts = read_clean_bodies(candidate_path, candidate=True, raw=raw)
        anchors = raw + extra_reference
        targets = reference + extra_reference
        originals = raw + [None] * len(extra_reference)
        answer, extra_answer, answer_map = match_clean_bodies(anchors, answer_parts, targets)
    except MissingCleanBodiesError as exc:
        return {
            "status": "missing_clean_bodies",
            "score": 0.0,
            "input_role": exc.input_role,
            "error": str(exc),
        }
    results = {}
    for i, (original, target, candidate) in enumerate(zip(originals, targets, answer, strict=True)):
        results[f"Body{i + 1}"] = scorer._score_shapes(
            original.shape if original else None,
            target.shape if target else None,
            candidate.shape if candidate else None,
        )
        logging.info("Preprocessing: scored body %d/%d", i + 1, len(originals))
    for i in range(len(extra_answer)):
        results[f"ExtraBody{i + 1}"] = score_body_edit(False, True)
    return {
        **aggregate_body_scores(results),
        "correspondence": {
            "original_body_count": len(raw),
            "added_reference_body_count": len(extra_reference),
            "reference": ref_map,
            "candidate": answer_map,
            "extra_candidate_body_count": len(extra_answer),
        },
    }
