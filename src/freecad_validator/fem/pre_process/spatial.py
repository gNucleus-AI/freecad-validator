"""Read-only occupancy and surface comparison for saved solids.

Regions retain their source solids. Union is a point predicate,
not new BReps: no Boolean, refinement, document write or geometry repair occurs.
"""

from dataclasses import replace
from functools import cached_property

import FreeCAD
import numpy as np

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.extractor import (
    _face_record,
    _face_samples,
    document_from_shape,
    runtime_versions,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    area_uniform_samples,
    linear_tolerance,
    sample_distance_stats,
    sample_tolerance,
    surface_distance_stats,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    BrepDocument,
    DiffConfig,
)
from freecad_validator.fem.pre_process.geometry_compare.sampling import (
    _inside,
    _occupied_probes,
    point_in_bounds,
)


def shape_surface_stats(target, candidate, config):
    """Reuse deterministic area-uniform point-to-surface comparison without Booleans."""
    vt, tt = target.tessellate(config.tessellation_deflection)
    vc, tc = candidate.tessellate(config.tessellation_deflection)
    stats = surface_distance_stats(
        np.asarray([tuple(v) for v in vt]),
        np.asarray(tt),
        np.asarray([tuple(v) for v in vc]),
        np.asarray(tc),
        6000,
        0,
    )
    if stats is None:
        raise EvaluationError("No surface samples for geometric comparison")
    return stats


def sampled_surfaces_match(target, candidate, config, tolerance):
    """Use point-cloud distances as a safe upper bound before exact surface queries."""
    meshes = []
    clouds = []
    for shape in (target, candidate):
        vertices, triangles = shape.tessellate(config.tessellation_deflection)
        mesh = (np.asarray([tuple(v) for v in vertices]), np.asarray(triangles))
        meshes.append(mesh)
        clouds.append(area_uniform_samples(*mesh, 6000, 0))
    if any(len(cloud) == 0 for cloud in clouds):
        raise EvaluationError("No surface samples for geometric comparison")
    # Every point in the opposite cloud lies on its sampled surface. Therefore
    # this maximum bounds the same probes' exact point-to-surface maximum above.
    if sample_distance_stats(*(cloud.tolist() for cloud in clouds))["sample_max"] <= tolerance:
        return True
    stats = surface_distance_stats(*meshes[0], *meshes[1], 6000, 0)
    return stats["sample_max"] <= tolerance


def shape_bounds(shapes):
    box = FreeCAD.BoundBox()
    for shape in shapes:
        box.add(
            shape.BoundBox
            if isinstance(shape, SpatialShape)
            else shape.optimalBoundingBox(False, False)
        )
    return box


def boundary_samples(shape, config):
    """Existing per-face sampler, with analytic surface normals for occupancy tests."""
    samples = []
    for face in shape.Faces:
        samples.extend(boundary_samples_of_face(face, config))
    return samples


def boundary_offset(shapes, tolerance):
    """Place side probes beyond the saved BRep's ON-classification skin."""
    pending = list(shapes)
    precision = 0.0
    while pending:
        shape = pending.pop()
        if isinstance(shape, SpatialShape):
            pending.extend(shape.parts)
        else:
            precision = max(precision, shape.getTolerance(1))
    # A candidate's metadata cannot move probes arbitrarily far from a surface.
    return max(tolerance * 4, min(precision * 4, sample_tolerance(DiffConfig(), 1.0)), 1e-7)


def boundary_agrees(source, target, tolerance):
    """Probe either side of each sampled surface, ignoring seams and internal faces."""
    config = DiffConfig()
    samples = (
        source.surface_samples
        if isinstance(source, SpatialShape)
        else boundary_samples(source, config)
    )
    offset = boundary_offset((source, target), tolerance)
    for point, normal in samples:
        for sign in (-1, 1):
            shifted = tuple(p + sign * offset * n for p, n in zip(point, normal, strict=False))
            source_inside = _inside(source, shifted, 0.0)
            target_inside = _inside(target, shifted, 0.0)
            if source_inside != target_inside:
                missing = target if source_inside else source
                if not _inside(missing, shifted, tolerance):
                    # A probe can lie on a second face at an edge or seam. OCC's
                    # boundary classification there is not stable across splits.
                    # Accept only a tolerance-sized occupied neighbourhood.
                    nearby = [
                        tuple(
                            value + (direction * tolerance if axis == i else 0)
                            for i, value in enumerate(shifted)
                        )
                        for axis in range(3)
                        for direction in (-1, 1)
                    ]
                    if not any(_inside(missing, point, 0.0) for point in nearby):
                        return False
    return True


class SpatialShape:
    """Read-only union occupancy for the standalone whole-shape comparison API."""

    def __init__(self, parts):
        self.parts = tuple(parts)
        self.part_boxes = [shape_bounds([part]) for part in self.parts]
        self.BoundBox = shape_bounds(self.parts)

    def isInside(self, point, tolerance, on_surface):
        values = (point.x, point.y, point.z)
        if not point_in_bounds(values, self.BoundBox, tolerance):
            return False
        return any(
            point_in_bounds(values, box, tolerance) and _inside(part, values, tolerance)
            for part, box in zip(self.parts, self.part_boxes, strict=False)
        )

    @cached_property
    def probes(self):
        return _occupied_probes(self, 8192)

    @cached_property
    def Volume(self):
        intersections = [
            a.BoundBox.intersected(b.BoundBox)
            for i, a in enumerate(self.parts)
            for b in self.parts[:i]
        ]
        if all(min(box.XLength, box.YLength, box.ZLength) <= 0 for box in intersections):
            return sum(part.Volume for part in self.parts)
        points, count = self.probes
        box = self.BoundBox
        return box.XLength * box.YLength * box.ZLength * len(points) / count

    @cached_property
    def CenterOfMass(self):
        return FreeCAD.Vector(*np.asarray(self.probes[0]).mean(axis=0))

    @cached_property
    def boundary(self):
        config = DiffConfig()
        tolerance = boundary_offset(
            (self,), linear_tolerance(config, max(self.BoundBox.DiagonalLength, 1.0))
        )
        rows = []
        for shape in self.parts:
            if isinstance(shape, SpatialShape):
                candidates = shape.boundary
            else:
                candidates = [
                    (face, boundary_samples_of_face(face, config)) for face in shape.Faces
                ]
            for face, samples in candidates:
                kept = []
                for point, normal in samples:
                    low = tuple(p - tolerance * n for p, n in zip(point, normal, strict=False))
                    high = tuple(p + tolerance * n for p, n in zip(point, normal, strict=False))
                    if not (
                        point_in_bounds(low, self.BoundBox, tolerance)
                        or point_in_bounds(high, self.BoundBox, tolerance)
                    ):
                        continue
                    if _inside(self, low, 0.0) != _inside(self, high, 0.0):
                        kept.append((point, normal))
                if kept:
                    rows.append((face, kept))
        return rows

    @cached_property
    def surface_samples(self):
        return [sample for _, samples in self.boundary for sample in samples]

    @property
    def Area(self):
        return sum(
            face.Area * len(points) / max(len(_face_samples(face, DiffConfig())), 1)
            for face, points in self.boundary
        )


def boundary_samples_of_face(face, config):
    samples = []
    for point in _face_samples(face, config):
        u, v = face.Surface.parameter(FreeCAD.Vector(*point))
        normal = face.normalAt(u, v)
        samples.append((point, (normal.x, normal.y, normal.z)))
    return samples


def spatial_document(shape, name, config):
    versions = runtime_versions()
    if not isinstance(shape, SpatialShape):
        return document_from_shape(
            shape,
            path=name,
            object_name=name,
            freecad_version=versions[0],
            occt_version=versions[1],
            gate_reason=None,
            config=config,
        )
    faces = []
    for face, sampled in shape.boundary:
        points = tuple(point for point, _ in sampled)
        record = _face_record(face, len(faces) + 1, {}, config)
        faces.append(
            replace(
                record,
                samples=points,
                measure=face.Area * len(points) / max(len(record.samples), 1),
                bbox_min=tuple(min(p[a] for p in points) for a in range(3)),
                bbox_max=tuple(max(p[a] for p in points) for a in range(3)),
            )
        )
    if not faces:
        raise EvaluationError("No occupied boundary samples for original body")
    box = shape.BoundBox
    return BrepDocument(
        path=name,
        object_name=name,
        freecad_version=versions[0],
        occt_version=versions[1],
        bbox_min=(box.XMin, box.YMin, box.ZMin),
        bbox_max=(box.XMax, box.YMax, box.ZMax),
        volume=shape.Volume,
        area=sum(face.measure for face in faces),
        faces=tuple(faces),
        edges=(),
        vertices=(),
        _shape=shape,
    )
