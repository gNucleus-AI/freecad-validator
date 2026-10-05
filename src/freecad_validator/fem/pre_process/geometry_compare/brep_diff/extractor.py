"""Extract JSON-safe BREP descriptors from a FreeCAD document.

Runs under FreeCAD's interpreter (`freecadcmd`). Opens an FCStd, selects the
representative solid, and emits a `BrepDocument` of frozen per-subshape records
(geometry type, measure, centroid, bbox, orientation, analytic scalars, sampled
points, and face/edge/vertex adjacency). No FreeCAD object escapes this module.
"""

from __future__ import annotations

import math
import os
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

import FreeCAD
import numpy as np

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    SUBSHAPE_KINDS,
    BrepDocument,
    DiffConfig,
    SubshapeRecord,
)
from freecad_validator.fem.pre_process.geometry_compare.loaders.loaded_part import (
    LoadedPart,
)

_DRIVE_LETTER = re.compile(r"^[A-Za-z]:")

SCALAR_ATTRS = (
    "Radius",
    "Radius1",
    "Radius2",
    "MajorRadius",
    "MinorRadius",
    "Height",
    "Angle",
    "SemiMajorAxis",
    "SemiMinorAxis",
    "FirstParameter",
    "LastParameter",
)


def runtime_versions() -> tuple[str, str]:
    """`(freecad_version, occt_version)` of the running interpreter.

    The format-neutral way for any input backend to stamp a `BrepDocument`'s
    provenance without reaching into the private version helpers below.
    """

    return _freecad_version(FreeCAD), _occt_version(FreeCAD)


def document_from_shape(
    shape: Any,
    *,
    path: str,
    object_name: str,
    freecad_version: str,
    occt_version: str,
    gate_reason: str | None,
    config: DiffConfig,
) -> BrepDocument:
    """Extract numeric descriptors, samples and adjacency from a detached solid."""
    faces = tuple(shape.Faces)
    edges = tuple(shape.Edges)
    vertices = tuple(shape.Vertexes)
    face_adjacency, edge_adjacency, vertex_adjacency = _build_adjacency(faces, edges, vertices)
    return BrepDocument(
        path=path,
        object_name=object_name,
        freecad_version=freecad_version,
        occt_version=occt_version,
        bbox_min=_bbox_min(shape),
        bbox_max=_bbox_max(shape),
        volume=float(getattr(shape, "Volume", 0.0) or 0.0),
        area=float(getattr(shape, "Area", 0.0) or 0.0),
        faces=tuple(
            _face_record(face, index, face_adjacency[index], config)
            for index, face in enumerate(faces, start=1)
        ),
        edges=tuple(
            _edge_record(edge, index, edge_adjacency[index], config)
            for index, edge in enumerate(edges, start=1)
        ),
        vertices=tuple(
            _vertex_record(vertex, index, vertex_adjacency[index])
            for index, vertex in enumerate(vertices, start=1)
        ),
        gate_reason=gate_reason,
    )


def _reject_fcstd_path_traversal(fcstd_path: Path) -> None:
    """Reject archive entries that can escape their extraction directory."""
    try:
        with zipfile.ZipFile(str(fcstd_path)) as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, OSError):
        return
    for info in infos:
        name = info.filename
        normalized = name.replace("\\", "/")
        if (
            "\x00" in name
            or _DRIVE_LETTER.match(normalized)
            or normalized.startswith("/")
            or os.path.isabs(normalized)
            or any(part == ".." for part in normalized.split("/"))
        ):
            raise ValueError(
                f"FCStd archive entry escapes the extraction directory (zip-slip): {name!r}"
            )
        # Refuse explicit special-file types; tolerate absent type bits.
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type and file_type not in (stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(
                "FCStd archive contains a non-regular member "
                f"(symlink/device/FIFO/socket), which can redirect writes: {name!r}"
            )


def open_fcstd_part(fcstd_path: Path, config: DiffConfig) -> LoadedPart:
    """Read a detached geometry export without recomputing or inspecting feature history."""

    _reject_fcstd_path_traversal(fcstd_path)
    doc = FreeCAD.openDocument(str(fcstd_path))
    try:
        shape_obj = _select_representative_shape_object(doc)
        shape = shape_obj.Shape
        document = document_from_shape(
            shape,
            path=str(fcstd_path),
            object_name=str(getattr(shape_obj, "Name", "Shape")),
            freecad_version=_freecad_version(FreeCAD),
            occt_version=_occt_version(FreeCAD),
            gate_reason=None,
            config=config,
        )
        return LoadedPart(document=document, shape=shape.copy())
    finally:
        FreeCAD.closeDocument(doc.Name)


def _freecad_version(freecad_module: Any) -> str:
    values = [str(value) for value in freecad_module.Version()]
    if len(values) >= 3:
        return ".".join(values[:3])
    return " ".join(values)


def _occt_version(freecad_module: Any) -> str:
    values = [str(value) for value in freecad_module.Version()]
    for index, value in enumerate(values):
        if "OpenCASCADE" in value or value.upper() == "OCC":
            if index + 1 < len(values):
                return values[index + 1]
            return value
    for value in values:
        if value.startswith("7.") or value.startswith("8."):
            return value
    return ""


def _valid_shape(obj: Any) -> bool:
    shape = getattr(obj, "Shape", None)
    if shape is None:
        return False
    if hasattr(shape, "isNull") and shape.isNull():
        return False
    return bool(getattr(shape, "Faces", []) and float(getattr(shape, "Area", 0.0) or 0.0) > 0.0)


def _select_representative_shape_object(doc: Any) -> Any:
    bodies = [
        obj
        for obj in doc.Objects
        if getattr(obj, "TypeId", "") == "PartDesign::Body"
        and _valid_shape(obj)
        and float(getattr(obj.Shape, "Volume", 0.0) or 0.0) > 0.0
    ]
    if bodies:
        return bodies[0]
    shape_objects = [
        obj
        for obj in doc.Objects
        if _valid_shape(obj) and float(getattr(obj.Shape, "Volume", 0.0) or 0.0) > 0.0
    ]
    if not shape_objects:
        raise RuntimeError(
            f"no representative solid shape found in {getattr(doc, 'FileName', doc.Name)}"
        )
    dependency_names = {
        getattr(dep, "Name", "")
        for obj in shape_objects
        for dep in (getattr(obj, "OutList", []) or [])
    }
    top_level = [obj for obj in shape_objects if getattr(obj, "Name", "") not in dependency_names]
    return (top_level or shape_objects)[0]


def _vector_tuple(vector: Any) -> tuple[float, float, float]:
    return (float(vector.x), float(vector.y), float(vector.z))


def _bbox_min(shape: Any) -> tuple[float, float, float]:
    box = shape.BoundBox
    return (float(box.XMin), float(box.YMin), float(box.ZMin))


def _bbox_max(shape: Any) -> tuple[float, float, float]:
    box = shape.BoundBox
    return (float(box.XMax), float(box.YMax), float(box.ZMax))


def _point_bbox(
    point: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return point, point


def _geometry_type(geometry: Any) -> str:
    if geometry is None:
        return "Unknown"
    name = type(geometry).__name__
    if name and name != "NoneType":
        return name
    return str(geometry).split()[0]


def _float_attr(obj: Any, attr: str) -> float | None:
    if not hasattr(obj, attr):
        return None
    value = getattr(obj, attr)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scalar_attrs(obj: Any) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for attr in SCALAR_ATTRS:
        value = _float_attr(obj, attr)
        if value is not None and math.isfinite(value):
            scalars[attr] = value
    return scalars


def _subshape_tolerance(shape: Any) -> float:
    value = _float_attr(shape, "Tolerance")
    return 0.0 if value is None else value


def _uv_grid_samples(face: Any, target: int) -> list[tuple[float, float, float]]:
    """Sample the face on a UV grid clipped to its trimmed domain. Flat faces
    tessellate to only a couple of triangles, so the tessellation centroids are far
    too sparse for ICP / sampled matching; a UV grid gives even, dense coverage."""
    u_min, u_max, v_min, v_max = face.ParameterRange
    n = max(2, int(math.ceil(math.sqrt(target))) + 1)
    points: list[tuple[float, float, float]] = []
    for u in np.linspace(u_min, u_max, n):
        for v in np.linspace(v_min, v_max, n):
            point = face.valueAt(float(u), float(v))
            if face.isInside(point, 1e-6, True):
                points.append((float(point.x), float(point.y), float(point.z)))
    return points


def _safe_uv_grid_samples(face: Any, target: int) -> list[tuple[float, float, float]]:
    try:
        return _uv_grid_samples(face, target)
    except Exception:
        return []


def _tessellation_samples(face: Any, deflection: float) -> list[tuple[float, float, float]]:
    try:
        raw_vertices, raw_triangles = face.tessellate(deflection)
    except Exception:
        return []
    vertex_points = [_vector_tuple(v) for v in raw_vertices]
    points = list(vertex_points)
    for tri in raw_triangles:
        if len(tri) == 3:
            centroid = (
                np.asarray(vertex_points[tri[0]])
                + np.asarray(vertex_points[tri[1]])
                + np.asarray(vertex_points[tri[2]])
            ) / 3.0
            points.append((float(centroid[0]), float(centroid[1]), float(centroid[2])))
    return points


def _face_samples(face: Any, config: DiffConfig) -> tuple[tuple[float, float, float], ...]:
    points = _safe_uv_grid_samples(face, config.sample_points_per_face)
    points.extend(_tessellation_samples(face, config.tessellation_deflection))
    points.append(_vector_tuple(face.CenterOfMass))
    return _downsample_points(points, config.sample_points_per_face)


def _edge_samples(edge: Any, config: DiffConfig) -> tuple[tuple[float, float, float], ...]:
    try:
        points = [
            _vector_tuple(point) for point in edge.discretize(Number=config.sample_points_per_edge)
        ]
    except Exception:
        points = [_vector_tuple(vertex.Point) for vertex in edge.Vertexes]
    if not points:
        points = [_vector_tuple(edge.CenterOfMass)]
    return tuple(points)


def _downsample_points(
    points: list[tuple[float, float, float]],
    target_count: int,
) -> tuple[tuple[float, float, float], ...]:
    if target_count <= 0 or len(points) <= target_count:
        return tuple(points)
    indices = np.linspace(0, len(points) - 1, target_count, dtype=np.int64)
    return tuple(points[int(index)] for index in indices)


def _average_face_normal(face: Any, config: DiffConfig) -> tuple[float, float, float]:
    try:
        vertices, triangles = face.tessellate(config.tessellation_deflection)
    except Exception:
        return (0.0, 0.0, 0.0)
    points = np.asarray([_vector_tuple(vertex) for vertex in vertices], dtype=np.float64)
    normal = np.zeros(3, dtype=np.float64)
    for tri in triangles:
        if len(tri) != 3:
            continue
        cross = np.cross(points[tri[1]] - points[tri[0]], points[tri[2]] - points[tri[0]])
        normal += cross
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    normal /= norm
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def _safe_curve(edge: Any) -> Any:
    # Degenerate edges (sphere/cone pole seams) raise "undefined curve type".
    try:
        return edge.Curve
    except Exception:
        return None


def _edge_direction(edge: Any) -> tuple[float, float, float]:
    try:
        samples = [_vector_tuple(point) for point in edge.discretize(Number=2)]
    except Exception:
        samples = [_vector_tuple(v.Point) for v in edge.Vertexes]
    if len(samples) < 2:
        return (0.0, 0.0, 0.0)
    direction = np.asarray(samples[-1], dtype=np.float64) - np.asarray(samples[0], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    direction /= norm
    return (float(direction[0]), float(direction[1]), float(direction[2]))


def _face_record(
    face: Any,
    index: int,
    adjacency: dict[str, tuple[str, ...]],
    config: DiffConfig,
) -> SubshapeRecord:
    surface = getattr(face, "Surface", None)
    return SubshapeRecord(
        id=f"F{index}",
        kind="face",
        index=index,
        geometry_type=_geometry_type(surface),
        measure=float(getattr(face, "Area", 0.0) or 0.0),
        center=_vector_tuple(face.CenterOfMass),
        bbox_min=_bbox_min(face),
        bbox_max=_bbox_max(face),
        orientation=_average_face_normal(face, config),
        tolerance=_subshape_tolerance(face),
        scalars=_scalar_attrs(surface),
        samples=_face_samples(face, config),
        adjacency=adjacency,
    )


def _edge_record(
    edge: Any,
    index: int,
    adjacency: dict[str, tuple[str, ...]],
    config: DiffConfig,
) -> SubshapeRecord:
    curve = _safe_curve(edge)
    return SubshapeRecord(
        id=f"E{index}",
        kind="edge",
        index=index,
        geometry_type=_geometry_type(curve),
        measure=float(getattr(edge, "Length", 0.0) or 0.0),
        center=_vector_tuple(edge.CenterOfMass),
        bbox_min=_bbox_min(edge),
        bbox_max=_bbox_max(edge),
        orientation=_edge_direction(edge),
        tolerance=_subshape_tolerance(edge),
        scalars=_scalar_attrs(curve),
        samples=_edge_samples(edge, config),
        adjacency=adjacency,
    )


def _vertex_record(
    vertex: Any,
    index: int,
    adjacency: dict[str, tuple[str, ...]],
) -> SubshapeRecord:
    point = _vector_tuple(vertex.Point)
    bbox_min, bbox_max = _point_bbox(point)
    return SubshapeRecord(
        id=f"V{index}",
        kind="vertex",
        index=index,
        geometry_type="Point",
        measure=0.0,
        center=point,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        orientation=(0.0, 0.0, 0.0),
        tolerance=_subshape_tolerance(vertex),
        scalars={},
        samples=(point,),
        adjacency=adjacency,
    )


def _build_adjacency(
    faces: tuple[Any, ...],
    edges: tuple[Any, ...],
    vertices: tuple[Any, ...],
) -> tuple[
    dict[int, dict[str, tuple[str, ...]]],
    dict[int, dict[str, tuple[str, ...]]],
    dict[int, dict[str, tuple[str, ...]]],
]:
    face_edges: dict[int, set[str]] = {index: set() for index in range(1, len(faces) + 1)}
    face_vertices: dict[int, set[str]] = {index: set() for index in range(1, len(faces) + 1)}
    edge_faces: dict[int, set[str]] = {index: set() for index in range(1, len(edges) + 1)}
    edge_vertices: dict[int, set[str]] = {index: set() for index in range(1, len(edges) + 1)}
    vertex_faces: dict[int, set[str]] = {index: set() for index in range(1, len(vertices) + 1)}
    vertex_edges: dict[int, set[str]] = {index: set() for index in range(1, len(vertices) + 1)}

    vertex_lookup = _SameShapeLookup(vertices)
    edge_lookup = _SameShapeLookup(edges)

    for face_index, face in enumerate(faces, start=1):
        for local_edge in face.Edges:
            edge_index = edge_lookup.find(local_edge)
            if edge_index is None:
                continue
            face_edges[face_index].add(f"E{edge_index}")
            edge_faces[edge_index].add(f"F{face_index}")
        for local_vertex in face.Vertexes:
            vertex_index = vertex_lookup.find(local_vertex)
            if vertex_index is None:
                continue
            face_vertices[face_index].add(f"V{vertex_index}")
            vertex_faces[vertex_index].add(f"F{face_index}")

    for edge_index, edge in enumerate(edges, start=1):
        for local_vertex in edge.Vertexes:
            vertex_index = vertex_lookup.find(local_vertex)
            if vertex_index is None:
                continue
            edge_vertices[edge_index].add(f"V{vertex_index}")
            vertex_edges[vertex_index].add(f"E{edge_index}")

    face_adjacency = {
        index: _adjacency_dict(
            faces=(),
            edges=face_edges[index],
            vertices=face_vertices[index],
        )
        for index in range(1, len(faces) + 1)
    }
    edge_adjacency = {
        index: _adjacency_dict(
            faces=edge_faces[index],
            edges=(),
            vertices=edge_vertices[index],
        )
        for index in range(1, len(edges) + 1)
    }
    vertex_adjacency = {
        index: _adjacency_dict(
            faces=vertex_faces[index],
            edges=vertex_edges[index],
            vertices=(),
        )
        for index in range(1, len(vertices) + 1)
    }
    return face_adjacency, edge_adjacency, vertex_adjacency


class _SameShapeLookup:
    """Map a sub-subshape back to its global 1-based index.

    `Shape.isSame` is the OCCT TShape identity test, which is reliable *within*
    one document. A spatial hash on the centroid keeps the lookup near O(1)
    instead of O(n) per query, which matters on the helical gear (100s of faces).
    """

    def __init__(self, shapes: tuple[Any, ...]) -> None:
        self._shapes = shapes
        self._buckets: dict[tuple[int, int, int], list[int]] = {}
        for index, shape in enumerate(shapes, start=1):
            self._buckets.setdefault(self._key(shape), []).append(index)

    @staticmethod
    def _key(shape: Any) -> tuple[int, int, int]:
        center = shape.CenterOfMass if hasattr(shape, "CenterOfMass") else shape.Point
        return (round(center.x * 100), round(center.y * 100), round(center.z * 100))

    def find(self, shape: Any) -> int | None:
        for index in self._buckets.get(self._key(shape), ()):
            try:
                if shape.isSame(self._shapes[index - 1]):
                    return index
            except Exception:
                continue
        return None


def _adjacency_dict(
    faces: set[str] | tuple[str, ...],
    edges: set[str] | tuple[str, ...],
    vertices: set[str] | tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    values = {
        "face": tuple(sorted(faces)),
        "edge": tuple(sorted(edges)),
        "vertex": tuple(sorted(vertices)),
    }
    return {key: value for key, value in values.items() if key in SUBSHAPE_KINDS}
