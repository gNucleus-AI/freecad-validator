"""Bidirectional surface distances in the original coordinate frame."""

from __future__ import annotations

import math

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    all_samples,
    document_scale,
    sample_distance_stats,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    BrepDocument,
    DiffConfig,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.scorer_base import (
    BaseScorer,
    ScoreResult,
    load_shape,
)
from freecad_validator.fem.pre_process.spatial import SpatialShape, shape_surface_stats

# Scale-normalized decay constants: the score reaches ~0.37 (~= 1/e) at a mean deviation of TAU_MEAN
# (0.5% of the bbox diagonal) and at a p95 deviation of TAU_P95 (2% of the diagonal). The mean
# term is tighter because, on a local edit, the unchanged majority of samples keeps the mean
# small; the p95 term carries the moved-region magnitude.
TAU_MEAN = 0.005
TAU_P95 = 0.02
W_MEAN = 0.5
W_P95 = 0.5


def _surface_stats(
    target_doc: BrepDocument, candidate_doc: BrepDocument, config: DiffConfig
) -> dict[str, float] | None:
    """Read-only surface distances; spatial regions use sampled boundaries instead."""
    target, candidate = [
        doc._shape if doc._shape is not None else load_shape(doc.path, config)
        for doc in (target_doc, candidate_doc)
    ]
    if target is None or candidate is None:
        raise ValueError("Missing geometry for surface-distance comparison")
    if isinstance(target, SpatialShape) or isinstance(candidate, SpatialShape):
        return None
    return shape_surface_stats(target, candidate, config)


class PointCloudChamferScorer(BaseScorer):
    name = "pointcloud_chamfer"

    def _score_docs(
        self,
        base_doc: BrepDocument | None,
        target_doc: BrepDocument,
        candidate_doc: BrepDocument,
    ) -> ScoreResult:
        scale = document_scale(target_doc, candidate_doc)
        stats = _surface_stats(target_doc, candidate_doc, self.config)
        mode = "surface"
        if stats is None:
            stats = sample_distance_stats(all_samples(target_doc), all_samples(candidate_doc))
            mode = "uv-grid"
        mean_norm = stats["sample_mean"] / scale
        p95_norm = stats["sample_p95"] / scale
        max_norm = stats["sample_max"] / scale
        chamfer_score = math.exp(-mean_norm / TAU_MEAN)
        hausdorff_score = math.exp(-p95_norm / TAU_P95)
        overall = W_MEAN * chamfer_score + W_P95 * hausdorff_score
        subscores = {"chamfer_mean": chamfer_score, "hausdorff_p95": hausdorff_score}
        reason = (
            f"surface distance [{mode}] (base unused): mean={mean_norm:.4f}·D ->{chamfer_score:.3f}, "
            f"p95={p95_norm:.4f}·D ->{hausdorff_score:.3f}, max={max_norm:.4f}·D"
        )
        return ScoreResult(
            score=overall,
            reason=reason,
            subscores=subscores,
            details={
                "sample_mean_norm": mean_norm,
                "sample_p95_norm": p95_norm,
                "sample_max_norm": max_norm,
                "scale": scale,
            },
        )
