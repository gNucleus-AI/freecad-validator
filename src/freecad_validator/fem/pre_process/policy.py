"""Binary credit and extra-edit penalties for pre-Boolean source bodies."""

import math
from dataclasses import asdict, replace

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    ScoreResult,
)

GEOMETRY_PASS_THRESHOLD = 0.95


def score_body_edit(
    reference_changed: bool,
    answer_changed: bool,
    geometry: ScoreResult | None = None,
) -> ScoreResult | None:
    if not reference_changed:
        if not answer_changed:
            return None
        return ScoreResult(
            0.0,
            "Unrequested geometry change: subtract one point from the task total",
            details={"reference_changed": False, "extra_change": True, "geometry_score": None},
        )
    if geometry is None or not math.isfinite(geometry.score) or not 0 <= geometry.score <= 1:
        raise ValueError("A changed reference body requires a finite geometry score in [0, 1]")
    return replace(
        geometry,
        score=float(geometry.score > GEOMETRY_PASS_THRESHOLD),
        details={
            **geometry.details,
            "reference_changed": True,
            "extra_change": False,
            "geometry_score": geometry.score,
        },
    )


def aggregate_body_scores(results: dict[str, ScoreResult | None]) -> dict:
    """One entry per original body, before Boolean Fragments; skipped entries add nothing."""
    scored = [result for result in results.values() if result is not None]
    changed = [result for result in scored if result.details["reference_changed"]]
    correct = sum(result.score == 1 for result in changed)
    extra = sum(result.details["extra_change"] for result in scored)
    total = max(0.0, (correct - extra) / len(changed)) if changed else (0.0 if extra else None)
    return {
        "score": total,
        "reference_changed_body_count": len(changed),
        "correct_body_count": correct,
        "extra_changed_body_count": extra,
        "bodies": {
            name: asdict(result) if result is not None else None for name, result in results.items()
        },
    }
