"""Shared geometry results, cached solid loading and equivalence checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.geometry_ops import (
    all_samples,
    document_scale,
    relative_abs_delta,
    sample_distance_stats,
    sample_tolerance,
)
from freecad_validator.fem.pre_process.geometry_compare.brep_diff.methods import (
    mesh_boolean,
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

EQUAL_MEASURE_REL_TOL = 1e-5
EQUAL_VOLUME_SYMDIFF_FRAC = 1e-3
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
    """Run (and memoize) `method.diff(a_doc, b_doc)`. `method` is a `BrepDiffMethod`
    instance; `method.diff` is exception-safe and returns a `DiffResult`."""
    key = (a_doc.path, b_doc.path, method.name)
    if key not in _DIFF_CACHE:
        _DIFF_CACHE[key] = method.diff(a_doc, b_doc)
    return _DIFF_CACHE[key]


def _symmetric_difference_volume(
    a: BrepDocument, b: BrepDocument, config: DiffConfig
) -> float | None:
    """Return symmetric-difference volume, or None if tessellation is unavailable."""
    try:
        va, ta = mesh_boolean.tessellate_solid(a.path, config.tessellation_deflection, config)
        vb, tb = mesh_boolean.tessellate_solid(b.path, config.tessellation_deflection, config)
        man_a, man_b = mesh_boolean.to_manifold(va, ta), mesh_boolean.to_manifold(vb, tb)
        return float((man_a - man_b).volume() + (man_b - man_a).volume())
    except Exception:
        return None


def documents_equivalent(a: BrepDocument, b: BrepDocument, config: DiffConfig) -> bool:
    """Compare volume, area, position and occupied material within geometric tolerances."""
    # Different sketch seams can change face, edge and vertex counts without
    # changing occupied material. Compare geometric measures, bounding boxes and
    # symmetric-difference volume without requiring matching topology counts.
    if relative_abs_delta(a.volume, b.volume) > EQUAL_MEASURE_REL_TOL:
        return False
    if relative_abs_delta(a.area, b.area) > EQUAL_MEASURE_REL_TOL:
        return False
    scale = document_scale(a, b)
    # Bounding boxes must agree too: equal volume and area alone do not pin down
    # position or orientation (a translated copy matches both).
    bbox_eps = max(EQUAL_MEASURE_REL_TOL * scale, 1e-7)
    for lo_a, lo_b in zip(a.bbox_min, b.bbox_min, strict=False):
        if abs(lo_a - lo_b) > bbox_eps:
            return False
    for hi_a, hi_b in zip(a.bbox_max, b.bbox_max, strict=False):
        if abs(hi_a - hi_b) > bbox_eps:
            return False
    symdiff = _symmetric_difference_volume(a, b, config)
    if symdiff is not None:
        # vol_eps mirrors the mesh_boolean "material changed" threshold so equality here
        # and the diff method's change-gate agree — but bbox_diagonal**3 is far too loose
        # for a thin or hollow part (a plate's bbox cube dwarfs its own volume, so a real
        # change could hide under the epsilon). Take the tighter of the bbox-scaled
        # threshold and a fraction of the solids' own volume.
        vol_eps = min(
            1e-4 * (scale**3),
            EQUAL_VOLUME_SYMDIFF_FRAC * max(a.volume, b.volume, 1e-9),
        )
        return symdiff <= vol_eps
    stats = sample_distance_stats(all_samples(a), all_samples(b))
    return stats["sample_p95"] <= sample_tolerance(config, scale)


class BaseScorer(ABC):
    """Interface for geometric components evaluated on extracted documents."""

    name: str = ""

    def __init__(self, config: DiffConfig | None = None) -> None:
        self.config = config or DiffConfig()

    @abstractmethod
    def _score_docs(self, base_doc, target_doc, candidate_doc) -> ScoreResult:
        """Score geometric similarity between extracted documents."""
