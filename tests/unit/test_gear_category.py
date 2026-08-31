"""Regression coverage for CAD-anchored gear derivation."""

from __future__ import annotations

import pytest

from freecad_validator.consistency.categories.gear import derived_candidates
from freecad_validator.measurement.schema import ClusterSummary, MeasurementBank
from freecad_validator.spec.parser import parse_spec


def _gear_spec(*, teeth: int = 20):
    return parse_spec(
        {
            "name": "spur gear",
            "key_parameters": (
                "gear_module = 1 mm\n"
                f"number_of_teeth = {teeth}\n"
                "pitch_diameter = 20 mm\n"
                "outer_diameter = 22 mm\n"
                "root_diameter = 17.5 mm\n"
                "addendum = 1 mm\n"
                "dedendum = 1.25 mm\n"
                "whole_depth = 2.25 mm\n"
                "circular_pitch = 3.142 mm"
            ),
        }
    )


def _gear_bank(*, teeth: int) -> MeasurementBank:
    common = {
        "count": teeth,
        "axis": (0.0, 0.0, 1.0),
        "convex": True,
        "centroids": [],
        "axial_extent": 10.0,
    }
    return MeasurementBank(
        cylinder_clusters=[
            ClusterSummary(id="gear_tip", radius=11.0, **common),
            ClusterSummary(id="gear_root", radius=8.75, **common),
        ]
    )


def test_spec_values_cannot_replace_a_missing_cad_gear_anchor():
    assert derived_candidates(MeasurementBank(), _gear_spec()) == {}


def test_spec_values_cannot_replace_a_disagreeing_cad_tooth_count():
    assert derived_candidates(_gear_bank(teeth=15), _gear_spec(teeth=20)) == {}


def test_matching_cad_anchor_still_supports_gear_derivations():
    derived = derived_candidates(_gear_bank(teeth=20), _gear_spec(teeth=20))

    expected = {
        "gear_module": 1.0,
        "pitch_diameter": 20.0,
        "outer_diameter": 22.0,
        "root_diameter": 17.5,
        "addendum": 1.0,
        "dedendum": 1.25,
        "whole_depth": 2.25,
        "circular_pitch": pytest.approx(3.141592653589793),
    }
    assert {key: value for key, (value, _ref) in derived.items()} == expected
    assert all("derived_from_cad" in ref for _value, ref in derived.values())
