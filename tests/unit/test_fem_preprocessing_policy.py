"""Task policy independent of the FreeCAD runtime."""

import pytest

from freecad_validator.fem.pre_process.geometry_compare.brep_diff.models import (
    ScoreResult,
)
from freecad_validator.fem.pre_process.policy import aggregate_body_scores, score_body_edit


@pytest.mark.parametrize("geometry,credit", [(0.0, 0), (0.94, 0), (0.95, 0), (0.950001, 1), (1, 1)])
def test_strict_geometry_threshold(geometry, credit):
    result = score_body_edit(True, True, ScoreResult(geometry, "test"))
    assert result.score == credit
    assert result.details["geometry_score"] == geometry


def test_fixed_reference_denominator_and_extra_penalties():
    results = {
        f"required_{i}": score_body_edit(True, True, ScoreResult(1 if i < 8 else 0.9, "test"))
        for i in range(10)
    }
    results.update({f"untouched_{i}": None for i in range(50)})
    results.update({f"extra_{i}": score_body_edit(False, True) for i in range(2)})
    report = aggregate_body_scores(results)
    assert report["score"] == 0.6
    assert report["reference_changed_body_count"] == 10
    assert report["correct_body_count"] == 8
    assert report["extra_changed_body_count"] == 2
    results.update({f"more_extra_{i}": score_body_edit(False, True) for i in range(10)})
    assert aggregate_body_scores(results)["score"] == 0


def test_no_required_edits():
    assert score_body_edit(False, False) is None
    assert aggregate_body_scores({"unchanged": None})["score"] is None
    assert aggregate_body_scores({"extra": score_body_edit(False, True)})["score"] == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_geometry_score_is_not_credited(invalid):
    with pytest.raises(ValueError):
        score_body_edit(True, True, ScoreResult(invalid, "invalid"))
