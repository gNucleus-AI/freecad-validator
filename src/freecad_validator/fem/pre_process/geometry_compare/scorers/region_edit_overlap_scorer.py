"""Deterministic occupancy sampling of intended and produced geometry edits."""

from __future__ import annotations

from typing import Any

import FreeCAD

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    document_scale,
    linear_tolerance,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.methods.sampled_assignment import (
    SampledAssignmentMethod,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    BrepDocument,
    DiffResult,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.scorer_base import (
    BaseScorer,
    ScoreResult,
    cached_diff,
    load_shape,
)

BBOX_PAD_FRACTION = 0.01


def _changed_rows(result: DiffResult) -> list[dict[str, Any]]:
    return list(result.removed) + list(result.added) + list(result.modified)


def _row_bbox_points(row: dict[str, Any]) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for side in ("a", "b"):
        summary = row.get(side)
        if not summary:
            continue
        bbox_min = summary.get("bbox_min")
        bbox_max = summary.get("bbox_max")
        if bbox_min and bbox_max:
            points.append(tuple(float(value) for value in bbox_min))
            points.append(tuple(float(value) for value in bbox_max))
    return points


def _bbox_from_diff_results(
    results: list[DiffResult],
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    points: list[tuple[float, float, float]] = []
    for result in results:
        for row in _changed_rows(result):
            points.extend(_row_bbox_points(row))
    if not points:
        return None
    mins = tuple(min(point[axis] for point in points) for axis in range(3))
    maxs = tuple(max(point[axis] for point in points) for axis in range(3))
    return mins, maxs


def _bbox_from_documents(
    documents: tuple[BrepDocument, ...],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins = tuple(min(doc.bbox_min[axis] for doc in documents) for axis in range(3))
    maxs = tuple(max(doc.bbox_max[axis] for doc in documents) for axis in range(3))
    return mins, maxs


def _pad_bbox(
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
    pad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins, maxs = bbox
    padded_min = []
    padded_max = []
    for axis in range(3):
        low = mins[axis]
        high = maxs[axis]
        if high - low < pad:
            center = 0.5 * (low + high)
            low = center - 0.5 * pad
            high = center + 0.5 * pad
        padded_min.append(low - pad)
        padded_max.append(high + pad)
    return tuple(padded_min), tuple(padded_max)


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _halton_points(
    count: int,
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    mins, maxs = bbox
    sizes = tuple(maxs[axis] - mins[axis] for axis in range(3))
    points: list[tuple[float, float, float]] = []
    for index in range(1, count + 1):
        unit = (
            _van_der_corput(index, 2),
            _van_der_corput(index, 3),
            _van_der_corput(index, 5),
        )
        points.append(tuple(mins[axis] + sizes[axis] * unit[axis] for axis in range(3)))
    return points


def _inside(shape: Any, point: tuple[float, float, float], tolerance: float) -> bool:
    return bool(shape.isInside(FreeCAD.Vector(*point), tolerance, True))


def _occupancy_change_counts(
    points: list[tuple[float, float, float]],
    base_shape: Any,
    target_shape: Any,
    candidate_shape: Any,
    tolerance: float,
) -> dict[str, int]:
    intended = 0
    produced = 0
    overlap = 0
    union = 0
    for point in points:
        base_inside = _inside(base_shape, point, tolerance)
        intended_changed = base_inside != _inside(target_shape, point, tolerance)
        produced_changed = base_inside != _inside(candidate_shape, point, tolerance)
        intended += int(intended_changed)
        produced += int(produced_changed)
        overlap += int(intended_changed and produced_changed)
        union += int(intended_changed or produced_changed)
    return {
        "intended": intended,
        "produced": produced,
        "overlap": overlap,
        "union": union,
    }


def _safe_ratio(numerator: int, denominator: int, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


class RegionEditOverlapScorer(BaseScorer):
    name = "region_edit_overlap"

    def _score_docs(
        self,
        base_doc: BrepDocument | None,
        target_doc: BrepDocument,
        candidate_doc: BrepDocument,
    ) -> ScoreResult:
        if base_doc is None:
            return ScoreResult(
                0.0,
                "region edit overlap requires a valid base solid",
                {"iou": 0.0, "precision": 0.0, "recall": 0.0},
            )

        method = SampledAssignmentMethod(self.config)
        d_t = cached_diff(method, base_doc, target_doc)
        d_c = cached_diff(method, base_doc, candidate_doc)
        intended_rows = _changed_rows(d_t)
        produced_rows = _changed_rows(d_c)
        if not intended_rows and not produced_rows:
            return ScoreResult(
                1.0,
                "region edit overlap: no intended or candidate changed region",
                {"iou": 1.0, "precision": 1.0, "recall": 1.0},
                {"d_t_summary": d_t.summary, "d_c_summary": d_c.summary},
            )

        scale = max(
            document_scale(base_doc, target_doc),
            document_scale(base_doc, candidate_doc),
            1.0,
        )
        bbox = _bbox_from_diff_results([d_t, d_c])
        if bbox is None:
            bbox = _bbox_from_documents((base_doc, target_doc, candidate_doc))
        pad = max(BBOX_PAD_FRACTION * scale, linear_tolerance(self.config, scale))
        bbox = _pad_bbox(bbox, pad)

        base_shape = load_shape(base_doc.path)
        target_shape = load_shape(target_doc.path)
        candidate_shape = load_shape(candidate_doc.path)
        if base_shape is None or target_shape is None or candidate_shape is None:
            return ScoreResult(
                0.0,
                "region edit overlap could not load one or more solid shapes",
                {"iou": 0.0, "precision": 0.0, "recall": 0.0},
            )

        tolerance = max(linear_tolerance(self.config, scale), 1e-7)
        sample_count = self.config.region_sample_count
        points = _halton_points(sample_count, bbox)
        counts = _occupancy_change_counts(
            points,
            base_shape,
            target_shape,
            candidate_shape,
            tolerance,
        )
        iou = _safe_ratio(counts["overlap"], counts["union"], 0.0)
        precision = _safe_ratio(counts["overlap"], counts["produced"], 0.0)
        recall = _safe_ratio(counts["overlap"], counts["intended"], 0.0)
        intended_fraction = counts["intended"] / sample_count
        produced_fraction = counts["produced"] / sample_count
        reason = (
            "region edit overlap (uses base): "
            f"IoU={iou:.3f}, P={precision:.3f}, R={recall:.3f}, "
            f"intended={counts['intended']}/{sample_count} ({intended_fraction:.2%}), "
            f"candidate={counts['produced']}/{sample_count} ({produced_fraction:.2%})"
        )
        if counts["union"] == 0 and (intended_rows or produced_rows):
            reason += "; no occupancy-detected changed volume inside diff-bounded region"
        return ScoreResult(
            score=iou,
            reason=reason,
            subscores={"iou": iou, "precision": precision, "recall": recall},
            details={
                "counts": counts,
                "sample_count": sample_count,
                "bbox_min": bbox[0],
                "bbox_max": bbox[1],
                "d_t_summary": d_t.summary,
                "d_c_summary": d_c.summary,
            },
        )
