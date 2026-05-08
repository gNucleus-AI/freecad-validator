"""Pydantic data models for the MeasurementBank.

Single source of truth for the bank schema — every extractor and
detector writes into these types, and every consistency check reads
from them. Kept in one file on purpose: the models are tightly coupled
and total less than one screen of structure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class Measurement(BaseModel):
    id: str
    value: float | Tuple[float, ...]
    unit: str
    source: str                     # "global" | "face" | "edge" | "cluster" | "feature_tree"
    tags: List[str] = Field(default_factory=list)
    feature_name: Optional[str] = None


class ClusterSummary(BaseModel):
    id: str
    count: int
    radius: float
    axis: Tuple[float, float, float]
    convex: bool                    # vs concave (inner cylinder = hole)
    centroids: List[Tuple[float, float, float]]
    axial_extent: float


class PlanePairSummary(BaseModel):
    """A pair of parallel plane faces whose offset is a thickness
    candidate. Pairs are de-duplicated on (offset, |normal|)
    so e.g. four identical 1.6 mm wall pairs collapse into one entry."""

    id: str
    normal: Tuple[float, float, float]    # shared plane normal (|·|·direction)
    offset: float                         # perpendicular distance between planes (mm)
    min_area: float                       # area of the smaller of the two faces


class FeatureTreeEntry(BaseModel):
    name: str                       # FreeCAD object Name (unique within doc)
    type_id: str                    # e.g. "PartDesign::Pad", "Part::Cylinder"
    label: str                      # user-visible Label
    properties: Dict[str, float] = Field(default_factory=dict)
    vectors: Dict[str, Tuple[float, float, float]] = Field(default_factory=dict)


class GridSummary(BaseModel):
    """Rectangular/linear pattern of points.

    `rows` is the smaller dimension and `columns` is the larger one, so a
    2×4 LEGO stud grid always materializes as rows=2 / columns=4 regardless
    of how PCA ordered the axes. Spacings are paired to the dimensions:
    `spacing_rows` is along the shorter direction, `spacing_cols` the
    longer. A pure 1D line (e.g. 3 tubes) has rows=1 and spacing_rows=0.0.
    """

    id: str
    source: str                     # "sketch:<Name>@r=<radius>" or "cluster:<id>"
    count: int                      # total points in the pattern
    rows: int
    columns: int
    spacing_rows: float
    spacing_cols: float
    origin: Tuple[float, float, float]    # position of one anchor point (3D, original frame)


class CircularPatternSummary(BaseModel):
    """N-fold circular pattern of points around a symmetry axis.

    Used for splines, gear teeth, any rotationally-symmetric feature
    group. `pattern_radius` is the distance from `center` to each point
    in the pattern; `angular_pitch = 2π / count` radians.
    """

    id: str
    source: str                     # "sketch:<Name>@r=<r>" or "cluster:<id>"
    count: int                      # N-fold symmetry
    pattern_radius: float           # distance from axis to each point (mm)
    center: Tuple[float, float, float]    # center of the circle (3D)
    axis: Tuple[float, float, float]      # symmetry axis direction
    angular_pitch: float            # 2π/count, in radians


class ConicSurface(BaseModel):
    """Summary of one Cone face. A chamfer is a conic frustum: the
    surface is a `Cone` with a finite parameter range whose two ends
    have different radii — the larger is at the chamfer's base
    cylinder, the smaller at the chamfer tip.
    """

    id: str
    axis: Tuple[float, float, float]
    apex: Tuple[float, float, float]
    semi_angle: float                 # radians, signed (sign = direction)
    radius_min: float                 # smaller end of the frustum
    radius_max: float                 # larger end
    axial_extent: float               # axial distance between the two ends


class SketchProfile(BaseModel):
    """Structured view of one Sketcher::SketchObject's profile geometry.

    The raw geometry is also written into the matching FeatureTreeEntry
    as `Geometry[k].LineLength` / `Geometry[k].CircleRadius` keys, but
    those are unwieldy to consume. SketchProfile exposes the same data
    in lists ranked by magnitude so categories can pull "the longest
    line" or "the smallest circle radius" without re-parsing keys.

    All lists are sorted descending. Lines and circles/arcs come from
    the sketch's profile elements only — construction geometry is
    excluded by the upstream extractor.
    """

    name: str
    line_lengths: List[float] = Field(default_factory=list)        # mm, sorted desc
    line_angles: List[float] = Field(default_factory=list)         # rad, sorted desc
    circle_radii: List[float] = Field(default_factory=list)        # mm, sorted desc
    arc_radii: List[float] = Field(default_factory=list)           # mm, sorted desc
    constraint_angles: List[float] = Field(default_factory=list)   # rad, sorted desc


class MeasurementBank(BaseModel):
    solid_count: int = 0
    globals: Dict[str, Measurement] = Field(default_factory=dict)
    face_stats: Dict[str, int] = Field(default_factory=dict)
    cylinder_clusters: List[ClusterSummary] = Field(default_factory=list)
    plane_pairs: List[PlanePairSummary] = Field(default_factory=list)
    grids: List[GridSummary] = Field(default_factory=list)
    circular_patterns: List[CircularPatternSummary] = Field(default_factory=list)
    feature_tree: List[FeatureTreeEntry] = Field(default_factory=list)
    sketch_profiles: List[SketchProfile] = Field(default_factory=list)
    conic_surfaces: List[ConicSurface] = Field(default_factory=list)
