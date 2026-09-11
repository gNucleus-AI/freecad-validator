"""Geometry tolerances must define an ordered, finite scoring interval."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from freecad_validator.comparators.geometry import GeometryTolerances

THRESHOLD_PAIRS = [
    ("volume_matched_rel_tol", "volume_far_rel_tol"),
    ("area_matched_rel_tol", "area_far_rel_tol"),
    ("bbox_matched_rel_tol", "bbox_far_rel_tol"),
    ("surface_types_exact_tol", "surface_types_zero_score"),
    ("surface_types_matched_rel_tol", "surface_types_far_rel_tol"),
    ("principal_moments_matched_rel_tol", "principal_moments_far_rel_tol"),
]


@pytest.mark.parametrize(("matched_field", "far_field"), THRESHOLD_PAIRS)
@pytest.mark.parametrize("matched", [0.01, 0.02], ids=["equal", "inverted"])
def test_rejects_unordered_thresholds(matched_field, far_field, matched):
    with pytest.raises(ValidationError) as exc:
        GeometryTolerances(**{matched_field: matched, far_field: 0.01})

    message = str(exc.value)
    assert matched_field in message
    assert far_field in message
    assert "must be less than" in message


@pytest.mark.parametrize(("matched_field", "far_field"), THRESHOLD_PAIRS)
@pytest.mark.parametrize("override_matched", [True, False])
def test_single_override_is_checked_against_the_other_default(
    matched_field, far_field, override_matched
):
    defaults = GeometryTolerances()
    field, value = (
        (matched_field, getattr(defaults, far_field))
        if override_matched
        else (far_field, getattr(defaults, matched_field))
    )

    with pytest.raises(ValidationError, match="must be less than"):
        GeometryTolerances(**{field: value})


@pytest.mark.parametrize(("matched_field", "far_field"), THRESHOLD_PAIRS)
def test_valid_pair_can_override_both_defaults(matched_field, far_field):
    tolerances = GeometryTolerances(**{matched_field: 0.02, far_field: 0.03})
    assert getattr(tolerances, matched_field) == 0.02
    assert getattr(tolerances, far_field) == 0.03


@pytest.mark.parametrize("field", ["volume_matched_rel_tol", "volume_far_rel_tol"])
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_rejects_nonfinite_thresholds(field, value):
    with pytest.raises(ValidationError):
        GeometryTolerances(**{field: value})
