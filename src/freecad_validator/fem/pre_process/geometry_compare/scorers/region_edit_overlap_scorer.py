"""Deterministic occupancy sampling of intended and produced geometry edits."""

from __future__ import annotations

from typing import Any

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
from freecad_validator.fem.pre_process.geometry_compare.sampling import (
    _halton_points,
    _inside_many,
    point_in_bounds,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.scorer_base import (
    BaseScorer,
    ScoreResult,
    cached_diff,
    load_shape,
)
from freecad_validator.fem.pre_process.occt_solid_occupancy import solid_occupancies
from freecad_validator.fem.pre_process.spatial import shape_bounds

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
    shapes = (base_shape, target_shape, candidate_shape)
    boxes = [shape_bounds([shape]) if shape is not None else None for shape in shapes]
    if len(points) >= 4096 and all(hasattr(shape, "exportBrepToString") for shape in shapes):
        occupancies = solid_occupancies(
            [shape.exportBrepToString() for shape in shapes], points, tolerance
        )
    else:
        occupancies = [
            (_inside_many(shape, points, tolerance) if shape is not None else [False] * len(points))
            for shape in shapes
        ]
    for point, states in zip(points, zip(*occupancies, strict=True), strict=True):
        base_inside, target_inside, candidate_inside = [
            box is not None and point_in_bounds(point, box, tolerance) and inside
            for inside, box in zip(states, boxes, strict=True)
        ]
        intended_changed = base_inside != target_inside
        produced_changed = base_inside != candidate_inside
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
            scale = max(document_scale(target_doc, candidate_doc), 1.0)
            tolerance = max(linear_tolerance(self.config, scale), 1e-7)
            bbox = _pad_bbox(
                _bbox_from_documents((target_doc, candidate_doc)),
                max(BBOX_PAD_FRACTION * scale, tolerance),
            )
            shapes = [
                doc._shape if doc._shape is not None else load_shape(doc.path, self.config)
                for doc in (target_doc, candidate_doc)
            ]
            if any(shape is None for shape in shapes):
                raise ValueError("Missing geometry for required addition comparison")
            points = _halton_points(self.config.region_sample_count, bbox)
            counts = _occupancy_change_counts(points, None, *shapes, tolerance)
            iou = _safe_ratio(counts["overlap"], counts["union"], 0.0)
            return ScoreResult(
                iou,
                "Whole-body addition sampled occupied-volume IoU",
                {"iou": iou},
                {
                    "counts": counts,
                    "sample_count": len(points),
                    "bbox_min": bbox[0],
                    "bbox_max": bbox[1],
                },
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

        base_shape, target_shape, candidate_shape = [
            doc._shape if doc._shape is not None else load_shape(doc.path, self.config)
            for doc in (base_doc, target_doc, candidate_doc)
        ]
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
