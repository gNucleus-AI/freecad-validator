"""Geometry-only raw/reference/answer scoring; independent of static FEM grading."""

from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

from freecad_validator.fem.pre_process.errors import EvaluationError
from freecad_validator.fem.pre_process.geometry import export_geometry, read_geometry
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
    load_doc,
)
from freecad_validator.fem.pre_process.policy import score_body_edit


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
        if find_spec("manifold3d") is None:
            raise EvaluationError(
                "pre_process requires manifold3d for seam-independent geometry equality; "
                "install gnucleus-freecad-validator[preprocess] "
                "in FreeCAD's Python environment"
            )

    def score(self, base: str, target: str, candidate: str) -> float | None:
        result = self.score_detailed(base, target, candidate)
        return None if result is None else result.score

    def score_detailed(
        self,
        raw: str | Path,
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
        """
        inputs = ((raw, raw_object), (reference, reference_object), (candidate, candidate_object))
        clear_doc_cache()
        try:
            with TemporaryDirectory(prefix="fem-pre-process-") as directory:
                docs = []
                for role, (source, object_name) in zip(
                    ("raw", "reference", "candidate"), inputs, strict=False
                ):
                    if source is None:
                        if role == "raw" or object_name is not None:
                            raise EvaluationError(
                                "Raw must exist; deleted shapes cannot select objects"
                            )
                        docs.append(None)
                        continue
                    path = Path(directory) / f"{role}.FCStd"
                    export_geometry(read_geometry(source, object_name), path)
                    doc = load_doc(path, self.config)
                    if doc is None:
                        raise EvaluationError(f"Could not extract {role} geometry from {source}")
                    docs.append(doc)
                raw_doc, reference_doc, candidate_doc = docs
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
        if documents_equivalent(raw, candidate, self.config):
            region = ScoreResult(0.0, "Candidate is unchanged raw geometry; preprocessing required")
        else:
            region = RegionEditOverlapScorer(self.config)._score_docs(raw, reference, candidate)
        fidelity = 0.75 * surface.score + 0.25 * global_result.score
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
