"""Geometry descriptors, optional read-only source shapes and comparison results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SUBSHAPE_KINDS = ("face", "edge", "vertex")
CHANGE_CLASSES = ("removed", "added", "modified", "unchanged")
SCORED_CHANGE_CLASSES = ("removed", "added", "modified")


@dataclass(frozen=True)
class DiffConfig:
    """Numeric tolerances and deterministic sampling controls for geometry comparisons."""

    method: str = "sampled_assignment"
    linear_tolerance: float = 1e-5
    relative_tolerance: float = 1e-6
    angular_tolerance_deg: float = 1.0
    tessellation_deflection: float = 0.1
    match_threshold: float = 0.58
    unchanged_score_threshold: float = 0.965
    sample_points_per_face: int = 48
    sample_points_per_edge: int = 16
    # Each Halton probe classifies raw, reference and candidate occupancy.
    region_sample_count: int = 8192

    @property
    def angular_tolerance_rad(self) -> float:
        return math.radians(self.angular_tolerance_deg)


@dataclass(frozen=True)
class SubshapeRecord:
    """Descriptor for one face, edge, or vertex in a representative solid.

    `id` is the stable within-document identifier (`F<i>` / `E<i>` / `V<i>`,
    1-based in OCCT subshape order). Subshape order is deterministic for a given
    saved document, so ids are stable within a document; the diff problem is to
    recover the cross-document correspondence between A-ids and B-ids.
    """

    id: str
    kind: str
    index: int
    geometry_type: str
    measure: float
    center: tuple[float, float, float]
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    orientation: tuple[float, float, float]
    tolerance: float
    scalars: dict[str, float] = field(default_factory=dict)
    samples: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    adjacency: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def bbox_size(self) -> tuple[float, float, float]:
        return tuple(max(0.0, self.bbox_max[i] - self.bbox_min[i]) for i in range(3))

    def degree(self, kind: str) -> int:
        return len(self.adjacency.get(kind, ()))

    def public_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "index": self.index,
            "geometry_type": self.geometry_type,
            "measure": self.measure,
            "center": list(self.center),
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "tolerance": self.tolerance,
            "scalars": dict(self.scalars),
            "adjacency": {key: list(value) for key, value in sorted(self.adjacency.items())},
        }


@dataclass(frozen=True)
class BrepDocument:
    """Extracted BREP document representation used by matchers."""

    path: str
    object_name: str
    freecad_version: str
    occt_version: str
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    volume: float
    area: float
    faces: tuple[SubshapeRecord, ...]
    edges: tuple[SubshapeRecord, ...]
    vertices: tuple[SubshapeRecord, ...]
    gate_reason: str | None = None
    _shape: Any = field(default=None, repr=False, compare=False)

    def entities(self, kind: str) -> tuple[SubshapeRecord, ...]:
        if kind == "face":
            return self.faces
        if kind == "edge":
            return self.edges
        if kind == "vertex":
            return self.vertices
        raise ValueError(f"unknown subshape kind: {kind}")

    def entity_map(self, kind: str) -> dict[str, SubshapeRecord]:
        return {entity.id: entity for entity in self.entities(kind)}

    def bbox_size(self) -> tuple[float, float, float]:
        return tuple(max(0.0, self.bbox_max[i] - self.bbox_min[i]) for i in range(3))

    def bbox_diagonal(self) -> float:
        size = self.bbox_size()
        return math.sqrt(sum(value * value for value in size))


@dataclass(frozen=True)
class MatchRecord:
    """One established A-to-B correspondence."""

    kind: str
    a_id: str
    b_id: str
    score: float
    reason: str
    deltas: dict[str, float] = field(default_factory=dict)

    def key(self) -> tuple[str, str, str]:
        return (self.kind, self.a_id, self.b_id)


@dataclass(frozen=True)
class DiffResult:
    """Classified change set plus enough context for audit/debugging."""

    method: str
    config: dict[str, Any]
    reference: str
    candidate: str
    freecad_version: str
    occt_version: str
    summary: dict[str, Any]
    removed: tuple[dict[str, Any], ...]
    added: tuple[dict[str, Any], ...]
    modified: tuple[dict[str, Any], ...]
    unchanged: tuple[dict[str, Any], ...]
    matches: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    runtime_s: float = 0.0


def make_empty_summary() -> dict[str, dict[str, int]]:
    return {change_class: {kind: 0 for kind in SUBSHAPE_KINDS} for change_class in CHANGE_CLASSES}


@dataclass(frozen=True)
class ScoreResult:
    """One scorer's verdict on a (base, target, candidate) triple."""

    score: float
    reason: str
    subscores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def clamped(self) -> ScoreResult:
        value = max(0.0, min(1.0, float(self.score)))
        if value == self.score:
            return self
        return ScoreResult(value, self.reason, self.subscores, self.details)
