"""Geometry-only raw/reference/answer scoring; independent of static FEM grading."""

from pathlib import Path

from freecad_validator.fem.errors import EvaluationError
from freecad_validator.fem.pre_process.geometry import read_geometry
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import DiffConfig
from freecad_validator.fem.pre_process.geometry_compare.scorers.global_geometry_scorer import (
    GlobalGeometryScorer,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.pointcloud_chamfer_scorer import (
    PointCloudChamferScorer,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.region_edit_overlap_scorer import (
    RegionEditOverlapScorer,
)
from freecad_validator.fem.pre_process.geometry_compare.scorers.scorer_base import (
    BaseScorer,
    ScoreResult,
    clear_doc_cache,
    documents_equivalent,
)
from freecad_validator.fem.pre_process.policy import (
    GEOMETRY_PASS_THRESHOLD,
    score_body_edit,
)
from freecad_validator.fem.pre_process.spatial import spatial_document


class PreProcessScorer(BaseScorer):
    """Compare changed material and final shape, with no topology score or FEM solve.

    Continuous geometry = region IoU * (0.75 * surface + 0.25 * global geometry).
    A required edit earns one point iff continuous geometry > 0.95, else zero.
    Skip only when neither reference nor answer changes raw. Unrequested edits
    are flagged for a one-point task penalty. Use bodies before Boolean Fragments.
    Invalid inputs raise EvaluationError; explicit None clean inputs mean deletion.
    """

    name = "pre_process"

    def __init__(self, config: DiffConfig | None = None) -> None:
        self.config = config or DiffConfig()
        if self.config.region_sample_count <= 0 or self.config.tessellation_deflection <= 0:
            raise ValueError("Sampling count and tessellation deflection must be positive")

    def score(self, base: str, target: str, candidate: str) -> float | None:
        result = self.score_detailed(base, target, candidate)
        return None if result is None else result.score

    def score_detailed(
        self,
        raw: str | Path | None,
        reference: str | Path | None,
        candidate: str | Path | None,
        *,
        raw_object: str | None = None,
        reference_object: str | None = None,
        candidate_object: str | None = None,
    ) -> ScoreResult | None:
        """Return binary body credit; None means both clean shapes left raw unchanged.

        Optional selectors are FCStd object Names, not Labels.
        A None reference/candidate denotes a deleted source body, not a missing file.
        A None raw with a nonempty reference denotes a required new body.
        """
        inputs = ((raw, raw_object), (reference, reference_object), (candidate, candidate_object))
        shapes = []
        for role, (source, object_name) in zip(
            ("raw", "reference", "candidate"), inputs, strict=False
        ):
            if source is None:
                if object_name is not None or (role == "raw" and reference is None):
                    raise EvaluationError(
                        "Absent shapes cannot select objects; additions require a reference"
                    )
                shapes.append(None)
            else:
                shapes.append(read_geometry(source, object_name))
        return self._score_shapes(*shapes)

    def _score_shapes(self, raw, reference, candidate):
        """Evaluate detached read-only regions directly, without exporting CAD documents."""
        clear_doc_cache()
        try:
            docs = [
                spatial_document(shape, role, self.config) if shape is not None else None
                for role, shape in zip(
                    ("raw", "reference", "candidate"), (raw, reference, candidate), strict=False
                )
            ]
            raw_doc, reference_doc, candidate_doc = docs
            if raw_doc is None:
                if reference_doc is None:
                    raise EvaluationError("Additions require a reference body")
                geometry = (
                    ScoreResult(0.0, "Required new body is missing")
                    if candidate_doc is None
                    else self._score_docs(*docs)
                )
                return score_body_edit(True, candidate_doc is not None, geometry)
            reference_changed = reference_doc is None or not documents_equivalent(
                raw_doc,
                reference_doc,
                self.config,
            )
            answer_changed = candidate_doc is None or not documents_equivalent(
                raw_doc,
                candidate_doc,
                self.config,
            )
            if not reference_changed:
                return score_body_edit(False, answer_changed)
            if reference_doc is None or candidate_doc is None:
                geometry = ScoreResult(
                    float(reference_doc is candidate_doc),
                    "Whole-body deletion comparison",
                )
            else:
                geometry = self._score_docs(*docs)
            return score_body_edit(True, answer_changed, geometry)
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, EvaluationError):
                raise
            raise EvaluationError(f"Preprocessing geometry evaluation failed: {exc}") from exc
        finally:
            # The geometry loader caches by filename; never reuse stale geometry across requests.
            clear_doc_cache()

    def _score_docs(self, raw, reference, candidate) -> ScoreResult:
        names = ("global_geometry", "pointcloud_chamfer", "region_edit_overlap")
        if documents_equivalent(reference, candidate, self.config):
            return ScoreResult(
                1.0,
                "Candidate and reference geometry are equivalent",
                dict.fromkeys(names, 1.0),
                {"geometry_equivalent": True},
            )

        global_result = GlobalGeometryScorer(self.config)._score_docs(raw, reference, candidate)
        surface = PointCloudChamferScorer(self.config)._score_docs(raw, reference, candidate)
        fidelity = 0.75 * surface.score + 0.25 * global_result.score
        candidate_is_raw = raw is not None and documents_equivalent(raw, candidate, self.config)
        if not candidate_is_raw and fidelity <= GEOMETRY_PASS_THRESHOLD:
            # Region IoU is at most one, so no region result can earn body credit.
            # Report the bound, not a fabricated continuous score or region score.
            return ScoreResult(
                0.0,
                f"Geometry score is at most {fidelity:.4f}; body credit requires "
                f"> {GEOMETRY_PASS_THRESHOLD}. Region sampling was not needed.",
                {"global_geometry": global_result.score, "pointcloud_chamfer": surface.score},
                {
                    "geometry_equivalent": False,
                    "geometry_score_upper_bound": fidelity,
                    "geometry_fidelity": fidelity,
                    "region_evaluated": False,
                    "surface_distances": surface.details,
                    "global_differences": global_result.details,
                },
            )
        if candidate_is_raw:
            region = ScoreResult(0.0, "Candidate is unchanged raw geometry; preprocessing required")
        else:
            region = RegionEditOverlapScorer(self.config)._score_docs(raw, reference, candidate)
        components = dict(zip(names, (global_result, surface, region), strict=False))
        return ScoreResult(
            fidelity * region.score,
            f"Geometry fidelity={fidelity:.4f} × changed-region IoU={region.score:.4f}",
            {name: result.score for name, result in components.items()},
            {
                "geometry_equivalent": False,
                "geometry_fidelity": fidelity,
                "component_reasons": {name: result.reason for name, result in components.items()},
                "component_subscores": {
                    name: result.subscores for name, result in components.items()
                },
                "surface_distances": surface.details,
                "global_differences": global_result.details,
                "region_sampling": {
                    key: region.details[key]
                    for key in ("counts", "sample_count", "bbox_min", "bbox_max")
                    if key in region.details
                },
            },
        ).clamped()
