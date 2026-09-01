"""Regression tests for safe, unambiguous geometry rejection messages."""

from __future__ import annotations

import pytest

from freecad_validator import _freecad_loader
from freecad_validator.comparators import geometry


@pytest.fixture
def comparator(monkeypatch):
    monkeypatch.setattr(_freecad_loader, "import_freecad", lambda: object())
    return geometry.GeometryComparator()


@pytest.mark.parametrize(
    ("extracted", "expected_side"),
    [
        ([{"_gate_reason": "rejected"}, {}], "reference"),
        ([{}, {"_gate_reason": "rejected"}], "candidate"),
    ],
)
def test_gate_reason_identifies_side_without_parent_path(
    monkeypatch, comparator, extracted, expected_side
):
    reference = "/private/tmp/run/reference/model.FCStd"
    candidate = "/private/tmp/run/candidate/model.FCStd"
    values = iter(extracted)
    monkeypatch.setattr(
        geometry,
        "_select_shape_and_features",
        lambda _path, **_kwargs: next(values),
    )

    result = comparator.compare(reference, candidate)

    assert result.reason == f"rejected in {expected_side} model 'model.FCStd'"
    assert "/private/tmp" not in result.reason


@pytest.mark.parametrize(
    ("extracted", "expected_side"),
    [
        ([None, {}], "reference"),
        ([{}, None], "candidate"),
    ],
)
def test_no_solid_reason_identifies_side_without_parent_path(
    monkeypatch, comparator, extracted, expected_side
):
    reference = "/private/tmp/run/reference/model.FCStd"
    candidate = "/private/tmp/run/candidate/model.FCStd"
    values = iter(extracted)
    monkeypatch.setattr(
        geometry,
        "_select_shape_and_features",
        lambda _path, **_kwargs: next(values),
    )

    result = comparator.compare(reference, candidate)

    assert result.reason == f"No solid shape found in {expected_side} model 'model.FCStd'"
    assert "/private/tmp" not in result.reason
