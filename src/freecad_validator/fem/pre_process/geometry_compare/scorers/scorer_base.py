"""Shared geometry results, cached solid loading and equivalence checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    document_scale,
    relative_abs_delta,
    sample_tolerance,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    BrepDocument,
    DiffConfig,
    ScoreResult,
)
from freecad_validator.fem.pre_process.geometry_compare.loaders.loaded_part import (
    LoadedPart,
)
from freecad_validator.fem.pre_process.geometry_compare.loaders.loader import load_part
from freecad_validator.fem.pre_process.geometry_compare.sampling import (
    _inside,
    _occupied_probes,
)
from freecad_validator.fem.pre_process.spatial import (
    SpatialShape,
    boundary_agrees,
    sampled_surfaces_match,
)

EQUAL_MEASURE_REL_TOL = 1e-5
_PART_CACHE: dict[str, LoadedPart | None] = {}
_DIFF_CACHE: dict[tuple[str, str, str], Any] = {}


def _load_part_cached(path: str | Path, config: DiffConfig) -> LoadedPart | None:
    key = str(Path(path).resolve())
    if key not in _PART_CACHE:
        part = load_part(path, config) if Path(path).is_file() else None
        _PART_CACHE[key] = part if part is not None and part.document.faces else None
    return _PART_CACHE[key]


def load_doc(path: str | Path, config: DiffConfig) -> BrepDocument | None:
    part = _load_part_cached(path, config)
    return part.document if part is not None else None


def load_shape(path: str | Path, config: DiffConfig | None = None) -> Any | None:
    part = _load_part_cached(path, config or DiffConfig())
    return part.shape if part is not None else None


def clear_doc_cache() -> None:
    _PART_CACHE.clear()
    _DIFF_CACHE.clear()


def cached_diff(method: Any, a_doc: BrepDocument, b_doc: BrepDocument) -> Any:
    """Cache successful diffs only; evaluation errors propagate to the caller."""
    key = (a_doc.path, b_doc.path, method.name)
    if key not in _DIFF_CACHE:
        _DIFF_CACHE[key] = method.diff(a_doc, b_doc)
    return _DIFF_CACHE[key]


def documents_equivalent(a: BrepDocument, b: BrepDocument, config: DiffConfig) -> bool:
    """Compare volume, area, position and occupied material within geometric tolerances."""
    left = a._shape if a._shape is not None else load_shape(a.path, config)
    right = b._shape if b._shape is not None else load_shape(b.path, config)
    if left is None or right is None:
        raise ValueError("Missing occupied geometry for equivalence comparison")
    sampled = isinstance(left, SpatialShape) or isinstance(right, SpatialShape)
    # Virtual regions have sampled measures. Do not treat integration noise or
    # internal split faces as a physical edit; check their actual occupancy below.
    if not sampled:
        if relative_abs_delta(a.volume, b.volume) > EQUAL_MEASURE_REL_TOL:
            return False
        if relative_abs_delta(a.area, b.area) > EQUAL_MEASURE_REL_TOL:
            return False
    scale = document_scale(a, b)
    # Bounding boxes must agree too: equal volume and area alone do not pin down
    # position or orientation (a translated copy matches both).
    bbox_eps = max(EQUAL_MEASURE_REL_TOL * scale, 1e-7)
    if not sampled:
        # Saved reference surfaces can have tolerance-sized bounding envelopes
        # beyond the occupied material. Permit that envelope only as a prefilter;
        # the bidirectional boundary test below still checks actual occupancy.
        # Candidate metadata cannot widen this reference-controlled allowance.
        bbox_eps = max(bbox_eps, min(left.getTolerance(1), sample_tolerance(config, scale)))
        for lo_a, lo_b in zip(a.bbox_min, b.bbox_min, strict=False):
            if abs(lo_a - lo_b) > bbox_eps:
                return False
        for hi_a, hi_b in zip(a.bbox_max, b.bbox_max, strict=False):
            if abs(hi_a - hi_b) > bbox_eps:
                return False
        # Either surface agreement or boundary occupancy establishes equivalence
        # after the same measure/position gates. Try the existing surface test
        # first to avoid thousands of solid classifications for matching parts.
        if sampled_surfaces_match(left, right, config, sample_tolerance(config, scale)):
            return True
    tolerance = max(config.linear_tolerance, config.relative_tolerance * scale)
    for source, target in ((left, right), (right, left)):
        # Native solids already have exact global measures and all face boundaries.
        # Spatial views also need their occupied probes because their measures and
        # boundary clipping are sampled. Probe each saved part so distant thin
        # pieces do not exhaust the budget on empty space between them.
        if isinstance(source, SpatialShape):
            for part in source.parts:
                if any(
                    not _inside(target, point, tolerance)
                    for point in _occupied_probes(part, 8192)[0]
                ):
                    return False
        if not boundary_agrees(source, target, tolerance):
            return False
    return True


class BaseScorer(ABC):
    """Interface for geometric components evaluated on extracted documents."""

    name: str = ""

    def __init__(self, config: DiffConfig | None = None) -> None:
        self.config = config or DiffConfig()

    @abstractmethod
    def _score_docs(self, base_doc, target_doc, candidate_doc) -> ScoreResult:
        """Score geometric similarity between extracted documents."""
