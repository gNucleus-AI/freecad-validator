"""Sampled-point optimal assignment for geometric correspondence.

Global min-cost (Hungarian) matching per subshape kind on a fused cost that mixes the
analytic descriptor with a symmetric sampled-point Hausdorff residual, so a matched
pair whose surface moved is detected as `modified`. Classification additionally uses a
boundary-change-fraction overlap test (each subshape's samples vs the other model's
full point cloud) to localize trim changes.

"""

from __future__ import annotations

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    document_scale,
    optimal_assignment,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.methods.base import (
    BrepDiffMethod,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    SUBSHAPE_KINDS,
    BrepDocument,
    DiffResult,
    MatchRecord,
)


class SampledAssignmentMethod(BrepDiffMethod):
    name = "sampled_assignment"

    def run(self, reference: BrepDocument, candidate: BrepDocument) -> DiffResult:
        scale = document_scale(reference, candidate)
        matches: list[MatchRecord] = []
        threshold = max(0.5, self.config.match_threshold - 0.05)
        for kind in SUBSHAPE_KINDS:
            matches.extend(
                optimal_assignment(
                    list(reference.entities(kind)),
                    list(candidate.entities(kind)),
                    scale,
                    kind,
                    "sampled_hausdorff",
                    min_score=threshold,
                )
            )
        # accepted matches carry sampled-Hausdorff deltas; overlap_override adds the
        # boundary-change-fraction test used to localize trim changes.
        return self.classify(reference, candidate, tuple(matches), overlap_override=True)
