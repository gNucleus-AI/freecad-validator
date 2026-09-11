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


@pytest.mark.parametrize("matched", [0.005, 0.01])
def test_v2_factory_rejects_explicit_unordered_bbox_pair(matched):
    with pytest.raises(ValidationError, match="bbox_matched_rel_tol.*must be less than"):
        GeometryTolerances.for_scorer("v2", bbox_matched_rel_tol=matched, bbox_far_rel_tol=0.005)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_factory_preserves_explicit_bbox_pair(version):
    tolerances = GeometryTolerances.for_scorer(
        version, bbox_matched_rel_tol=0.002, bbox_far_rel_tol=0.005
    )
    assert tolerances.bbox_matched_rel_tol == 0.002
    assert tolerances.bbox_far_rel_tol == 0.005


def test_factory_keeps_v1_strict_and_default_values_unchanged():
    with pytest.raises(ValidationError, match="bbox_matched_rel_tol.*must be less than"):
        GeometryTolerances.for_scorer("v1", bbox_far_rel_tol=0.005)
    for version in ("v1", "v2"):
        assert GeometryTolerances.for_scorer(version) == GeometryTolerances()


@pytest.mark.parametrize("value", [0, -0.01, float("inf"), float("nan"), None, "invalid"])
def test_v2_factory_rejects_invalid_bbox_gate(value):
    with pytest.raises(ValidationError):
        GeometryTolerances.for_scorer("v2", bbox_far_rel_tol=value)


def test_v2_factory_accepts_numeric_strings_like_the_model():
    tolerances = GeometryTolerances.for_scorer("v2", bbox_far_rel_tol="0.005")
    assert tolerances.bbox_far_rel_tol == 0.005
    assert tolerances.bbox_matched_rel_tol == pytest.approx(0.0005)


def test_factory_rejects_unknown_scorer():
    with pytest.raises(ValueError, match="unknown scorer version"):
        GeometryTolerances.for_scorer("v3", bbox_far_rel_tol=0.005)


@pytest.mark.parametrize("version", [None, "v1", "v2"])
@pytest.mark.parametrize("value", [None, 0.005])
def test_unknown_fields_cannot_silently_leave_defaults_in_place(version, value):
    with pytest.raises(ValidationError) as exc:
        if version is None:
            GeometryTolerances(bbox_far_rel_to=value)
        else:
            GeometryTolerances.for_scorer(version, bbox_far_rel_to=value)
    assert exc.value.errors()[0]["type"] == "extra_forbidden"
    assert exc.value.errors()[0]["loc"] == ("bbox_far_rel_to",)
