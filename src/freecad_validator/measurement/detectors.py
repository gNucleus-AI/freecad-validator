"""BankDetector ABC + all derived detectors + ordered registry.

Detectors run AFTER every ShapeExtractor so they can mine already-extracted
structures (sketch circles, cylinder cluster centroids) rather than hitting
the FreeCAD shape directly. Two concrete detectors today:

  - `LinearPatternDetector` — rectangular / 1-D grids
  - `CircularPatternDetector` — N-fold rotational symmetry
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import round_angle, round_length
from .schema import (
    CircularPatternSummary,
    GridSummary,
    MeasurementBank,
    SketchProfile,
)


class BankDetector(ABC):
    """Populate a derived section of the bank from already-extracted data."""

    #: Short stable identifier used in logs and error messages.
    name: str = ""

    @abstractmethod
    def detect(self, *, bank: MeasurementBank) -> None:
        """Mutate `bank` in place by filling in this detector's section."""


# ---------------------------------------------------------------------------
# Linear-pattern detection (rectangular / 1-D grids)
# ---------------------------------------------------------------------------

# Relative tolerance for "uniformly spaced": max deviation from mean spacing
# must be under this fraction of the mean. 5% is generous enough to absorb
# FreeCAD numeric drift but tight enough to reject accidentally near-uniform
# clouds of unrelated points.
_GRID_SPACING_REL_TOL = 0.05
# Bin-merge tolerance as a fraction of the projected range. Points within
# this fraction of each other on the projected axis count as the same bin.
_GRID_BIN_FRAC_TOL = 0.05


def _bin_projections(values: np.ndarray, tol: float) -> Optional[Tuple[int, float, float]]:
    """Group sorted 1D values into bins (within `tol`). Return
    (num_bins, mean_spacing, min_center) or None if the bin spacing is
    not approximately uniform. Single-bin input returns spacing 0.0."""
    if values.size == 0:
        return None
    vals = np.sort(values)
    spread = float(vals[-1] - vals[0])
    if spread < tol:
        return 1, 0.0, float(vals[0])

    bins: List[List[float]] = [[float(vals[0])]]
    for v in vals[1:]:
        if float(v) - bins[-1][-1] < tol:
            bins[-1].append(float(v))
        else:
            bins.append([float(v)])

    if len(bins) == 1:
        return 1, 0.0, bins[0][0]

    centers = [sum(b) / len(b) for b in bins]
    spacings = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    mean_spacing = sum(spacings) / len(spacings)
    if mean_spacing <= 0:
        return None
    max_dev = max(abs(s - mean_spacing) for s in spacings) / mean_spacing
    if max_dev > _GRID_SPACING_REL_TOL:
        return None
    return len(centers), float(mean_spacing), float(centers[0])


def _detect_grid_from_points(
    points: List[Tuple[float, float, float]],
) -> Optional[Tuple[int, int, float, float, Tuple[float, float, float]]]:
    """Return (rows, cols, spacing_rows, spacing_cols, origin_xyz) if the
    given 3D points form a rectangular grid (including the 1×N degenerate
    case). `rows <= cols` always; spacing_rows is the spacing along the
    smaller dimension. ``origin_xyz`` is the input point closest to the
    min-projection corner of the pattern (in the original frame)."""
    n = len(points)
    if n < 2:
        return None

    arr = np.asarray(points, dtype=float)
    centered = arr - arr.mean(axis=0)

    try:
        cov = np.cov(centered, rowvar=False) if n >= 3 else np.outer(centered[0], centered[0])
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    order = np.argsort(eigvals)[::-1]
    axis_major = eigvecs[:, order[0]]
    axis_minor = eigvecs[:, order[1]] if eigvecs.shape[1] > 1 else np.array([0.0, 0.0, 0.0])

    proj_major = centered @ axis_major
    proj_minor = centered @ axis_minor

    range_major = float(proj_major.max() - proj_major.min())
    range_minor = float(proj_minor.max() - proj_minor.min())
    tol = _GRID_BIN_FRAC_TOL * max(range_major, range_minor, 1e-6)

    bin_major = _bin_projections(proj_major, tol)
    bin_minor = _bin_projections(proj_minor, tol)
    if bin_major is None or bin_minor is None:
        return None
    n_major, sp_major, _ = bin_major
    n_minor, sp_minor, _ = bin_minor
    if n_major * n_minor != n:
        return None

    if n_major >= n_minor:
        cols, spacing_cols = n_major, sp_major
        rows, spacing_rows = n_minor, sp_minor
    else:
        cols, spacing_cols = n_minor, sp_minor
        rows, spacing_rows = n_major, sp_major

    min_major = proj_major.min()
    min_minor = proj_minor.min()
    dists = np.hypot(proj_major - min_major, proj_minor - min_minor)
    anchor = arr[int(np.argmin(dists))]
    origin = tuple(float(x) for x in anchor)

    return rows, cols, round_length(spacing_rows), round_length(spacing_cols), round_length(origin)


class LinearPatternDetector(BankDetector):
    name = "grids"

    def detect(self, *, bank: MeasurementBank) -> None:
        grids: List[GridSummary] = []
        grid_idx = 0

        # Sketch-level: group CircleCenter/CircleRadius pairs by radius.
        for entry in bank.feature_tree:
            circles: List[Tuple[int, float, Tuple[float, float, float]]] = []
            for prop_key, radius in entry.properties.items():
                if "CircleRadius" not in prop_key:
                    continue
                idx_str = prop_key.split("[", 1)[1].split("]", 1)[0]
                center_key = f"Geometry[{idx_str}].CircleCenter"
                center = entry.vectors.get(center_key)
                if center is None:
                    continue
                try:
                    circles.append((int(idx_str), float(radius), tuple(center)))
                except (TypeError, ValueError):
                    continue
            if not circles:
                continue

            by_radius: Dict[float, List[Tuple[float, float, float]]] = {}
            for _idx, r, c in circles:
                key = round(r, 6)
                by_radius.setdefault(key, []).append(c)

            for radius, centers in by_radius.items():
                if len(centers) < 2:
                    continue
                detection = _detect_grid_from_points(centers)
                if detection is None:
                    continue
                rows, cols, sp_rows, sp_cols, origin = detection
                grids.append(
                    GridSummary(
                        id=f"grid_{grid_idx}",
                        source=f"sketch:{entry.name}@r={radius:g}",
                        count=len(centers),
                        rows=rows, columns=cols,
                        spacing_rows=sp_rows, spacing_cols=sp_cols,
                        origin=origin,
                    )
                )
                grid_idx += 1

        # Cluster-level: run on each cluster's centroids.
        for cluster in bank.cylinder_clusters:
            if cluster.count < 2:
                continue
            detection = _detect_grid_from_points(cluster.centroids)
            if detection is None:
                continue
            rows, cols, sp_rows, sp_cols, origin = detection
            grids.append(
                GridSummary(
                    id=f"grid_{grid_idx}",
                    source=f"cluster:{cluster.id}",
                    count=cluster.count,
                    rows=rows, columns=cols,
                    spacing_rows=sp_rows, spacing_cols=sp_cols,
                    origin=origin,
                )
            )
            grid_idx += 1

        bank.grids = grids


# ---------------------------------------------------------------------------
# Circular-pattern detection (N-fold rotational symmetry)
# ---------------------------------------------------------------------------

# Points are considered to sit on a common circle if all radii are
# within this fraction of the mean radius.
_CIRCULAR_RADIUS_REL_TOL = 0.05
# Angular gaps between sorted θ are considered uniform within this
# fraction of the mean gap (and 2π/N).
_CIRCULAR_PITCH_REL_TOL = 0.05


def _detect_circular_from_points(
    points: List[Tuple[float, float, float]],
    axis_hint: Optional[Tuple[float, float, float]] = None,
) -> Optional[Tuple[int, float, Tuple[float, float, float], Tuple[float, float, float], float]]:
    """Return (count, pattern_radius, center, axis, angular_pitch_rad)
    if `points` form an N-fold circular pattern (N ≥ 3). Else None.

    When `axis_hint` is provided (e.g. from a cylinder cluster's shared
    axis), it's used directly; otherwise the smallest-eigenvalue PCA
    direction is taken as the axis.
    """
    n = len(points)
    if n < 3:
        return None

    arr = np.asarray(points, dtype=float)

    if axis_hint is not None:
        axis = np.asarray(axis_hint, dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return None
        axis = axis / axis_norm
    else:
        centered = arr - arr.mean(axis=0)
        try:
            cov = np.cov(centered, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return None
        axis = eigvecs[:, int(np.argmin(eigvals))]

    if abs(axis[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, tmp)
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm < 1e-9:
        return None
    e1 = e1 / e1_norm
    e2 = np.cross(axis, e1)

    mean_pt = arr.mean(axis=0)
    radii: List[float] = []
    thetas: List[float] = []
    for p in arr:
        v = p - mean_pt
        perp = v - np.dot(v, axis) * axis
        r = float(np.linalg.norm(perp))
        if r < 1e-9:
            # Point sits on the axis → no well-defined θ. Reject the
            # whole pattern rather than emit garbage.
            return None
        radii.append(r)
        thetas.append(float(math.atan2(np.dot(perp, e2), np.dot(perp, e1))))

    mean_r = float(np.mean(radii))
    if mean_r < 1e-6:
        return None
    if max(abs(r - mean_r) for r in radii) / mean_r > _CIRCULAR_RADIUS_REL_TOL:
        return None

    thetas_sorted = sorted(thetas)
    gaps: List[float] = []
    for i in range(n):
        j = (i + 1) % n
        gap = thetas_sorted[j] - thetas_sorted[i]
        if gap <= 0:
            gap += 2 * math.pi
        gaps.append(gap)
    mean_gap = sum(gaps) / n
    if mean_gap <= 0:
        return None
    if max(abs(g - mean_gap) for g in gaps) / mean_gap > _CIRCULAR_PITCH_REL_TOL:
        return None

    expected_pitch = 2 * math.pi / n
    if abs(mean_gap - expected_pitch) / expected_pitch > _CIRCULAR_PITCH_REL_TOL:
        return None

    center = tuple(float(x) for x in mean_pt)
    axis_tuple = tuple(float(x) for x in axis)
    return n, mean_r, center, axis_tuple, float(mean_gap)


class CircularPatternDetector(BankDetector):
    name = "circular_patterns"

    def detect(self, *, bank: MeasurementBank) -> None:
        patterns: List[CircularPatternSummary] = []
        idx = 0

        for entry in bank.feature_tree:
            circles: List[Tuple[int, float, Tuple[float, float, float]]] = []
            for prop_key, radius in entry.properties.items():
                if "CircleRadius" not in prop_key:
                    continue
                idx_str = prop_key.split("[", 1)[1].split("]", 1)[0]
                center = entry.vectors.get(f"Geometry[{idx_str}].CircleCenter")
                if center is None:
                    continue
                try:
                    circles.append((int(idx_str), float(radius), tuple(center)))
                except (TypeError, ValueError):
                    continue
            if not circles:
                continue

            by_radius: Dict[float, List[Tuple[float, float, float]]] = {}
            for _i, r, c in circles:
                by_radius.setdefault(round(r, 6), []).append(c)

            for radius, centers in by_radius.items():
                if len(centers) < 3:
                    continue
                detection = _detect_circular_from_points(centers)
                if detection is None:
                    continue
                count, pat_r, center, axis, pitch = detection
                patterns.append(
                    CircularPatternSummary(
                        id=f"circular_pattern_{idx}",
                        source=f"sketch:{entry.name}@r={radius:g}",
                        count=count,
                        pattern_radius=round_length(pat_r),
                        center=round_length(center),
                        axis=round_length(axis),
                        angular_pitch=round_angle(pitch),
                    )
                )
                idx += 1

        for cluster in bank.cylinder_clusters:
            if cluster.count < 3:
                continue
            detection = _detect_circular_from_points(cluster.centroids, axis_hint=cluster.axis)
            if detection is None:
                continue
            count, pat_r, center, axis, pitch = detection
            patterns.append(
                CircularPatternSummary(
                    id=f"circular_pattern_{idx}",
                    source=f"cluster:{cluster.id}",
                    count=count,
                    pattern_radius=round_length(pat_r),
                    center=round_length(center),
                    axis=round_length(axis),
                    angular_pitch=round_angle(pitch),
                )
            )
            idx += 1

        bank.circular_patterns = patterns


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Order: grid first (a rectangular cloud is a stricter structure, prefer
# naming it that way when both fit), then circular.
# ---------------------------------------------------------------------------
# Sketch-profile aggregation
# ---------------------------------------------------------------------------


class SketchProfileDetector(BankDetector):
    """Aggregate per-sketch profile geometry from feature_tree into a
    structured ``bank.sketch_profiles`` list.

    The underlying data (``Geometry[k].LineLength``, ``CircleRadius``,
    ``ArcRadius``, ``LineAngle[i,j]``, ``Constraint[i].Angle``) is
    already populated on each ``Sketcher::SketchObject`` entry by
    ``FeatureTreeExtractor``. This detector pulls those keys out and
    presents them as ranked lists (sorted desc) so consumers can ask
    for "the longest line" or "the smallest circle radius" without
    re-parsing keys.
    """

    name = "sketch_profile"

    @staticmethod
    def _values(props, prefix: str, suffix: str) -> List[float]:
        out = [v for k, v in props.items()
               if k.startswith(prefix) and k.endswith(suffix)
               and isinstance(v, (int, float))]
        return sorted((float(v) for v in out), reverse=True)

    @staticmethod
    def _line_angles(props) -> List[float]:
        out = [v for k, v in props.items()
               if k.startswith("LineAngle[") and isinstance(v, (int, float))]
        return sorted((float(v) for v in out), reverse=True)

    @staticmethod
    def _constraint_angles(props) -> List[float]:
        out = [v for k, v in props.items()
               if k.startswith("Constraint[") and k.endswith(".Angle")
               and isinstance(v, (int, float))]
        return sorted((float(v) for v in out), reverse=True)

    def detect(self, *, bank: MeasurementBank) -> None:
        for entry in bank.feature_tree:
            if entry.type_id != "Sketcher::SketchObject":
                continue
            p = entry.properties
            profile = SketchProfile(
                name=entry.name,
                line_lengths=self._values(p, "Geometry[", ".LineLength"),
                circle_radii=self._values(p, "Geometry[", ".CircleRadius"),
                arc_radii=self._values(p, "Geometry[", ".ArcRadius"),
                line_angles=self._line_angles(p),
                constraint_angles=self._constraint_angles(p),
            )
            bank.sketch_profiles.append(profile)


DEFAULT_BANK_DETECTORS: List[BankDetector] = [
    LinearPatternDetector(),
    CircularPatternDetector(),
    SketchProfileDetector(),
]
