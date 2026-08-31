"""Categories must not turn expected spec values into CAD measurements."""

from __future__ import annotations

from freecad_validator.consistency.categories.helical_gear import (
    derived_candidates as helical_gear_candidates,
)
from freecad_validator.consistency.categories.spline import (
    derived_candidates as spline_candidates,
)
from freecad_validator.measurement.schema import ClusterSummary, MeasurementBank
from freecad_validator.spec.parser import parse_spec


def _spec(name: str, key_parameters: str):
    return parse_spec({"name": name, "key_parameters": key_parameters})


def _tooth_ring_bank() -> MeasurementBank:
    common = {
        "count": 20,
        "axis": (0.0, 0.0, 1.0),
        "convex": True,
        "centroids": [],
        "axial_extent": 10.0,
    }
    return MeasurementBank(
        cylinder_clusters=[
            ClusterSummary(id="tip", radius=11.0, **common),
            ClusterSummary(id="root", radius=8.75, **common),
        ]
    )


def test_helical_gear_does_not_use_spec_angles_as_measurements():
    spec = _spec(
        "helical_gear",
        "helix_angle = 20 deg\n"
        "gear_module = 2 mm\n"
        "number_of_teeth = 20\n"
        "pitch_diameter = 42.567 mm",
    )

    assert helical_gear_candidates(MeasurementBank(), spec) == {}
    assert helical_gear_candidates(_tooth_ring_bank(), spec) == {}


def test_spline_does_not_derive_candidate_values_from_spec():
    spec = _spec(
        "splined_shaft",
        "spline_module = 2 mm\n"
        "spline_number_teeth = 15\n"
        "spline_pitch_diameter = 30 mm\n"
        "spline_major_diameter = 34 mm\n"
        "spline_tooth_width_ratio = 0.5",
    )

    assert spline_candidates(MeasurementBank(), spec) == {}
