"""Regression tests for spec parsing and trusted param-check discovery."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from freecad_validator.consistency.checker import ConsistencyChecker
from freecad_validator.measurement.extractors import FeatureTreeExtractor
from freecad_validator.measurement.schema import ClusterSummary, MeasurementBank
from freecad_validator.spec.parser import parse_key_parameters, parse_spec


def test_parser_rejects_expressions_and_handles_supported_numeric_forms():
    scalars, vectors, counts, strings = parse_key_parameters(
        "gear_ratio = 100/25\n"
        "tooth_angle = 2*pi/20 rad\n"
        "clearance = 5e-3 mm\n"
        "Overall_Length = 100mm\n"
        "positions = (−5mm, 3mm)"
    )

    assert "gear_ratio" not in scalars
    assert "tooth_angle" not in scalars
    assert scalars["clearance"] == pytest.approx(0.005)
    assert scalars["overall_length"] == 100.0
    assert vectors["positions"] == (-5.0, 3.0)
    assert counts == {}
    assert strings == {}


def test_parser_accepts_structured_key_parameters_object():
    parsed = parse_spec(
        {
            "name": "structured",
            "key_parameters": {
                "Overall_Length": {"value": 10, "unit": "cm"},
                "num_teeth": 20,
                "position": {"value": ["−5 mm", "3 mm"]},
                "cut_angle": {"value": 90, "unit": "deg"},
                "helix_hand": "right",
            },
        }
    )

    assert parsed.scalars["overall_length"] == 100.0
    assert parsed.scalars["cut_angle"] == pytest.approx(math.pi / 2)
    assert parsed.counts["num_teeth"] == 20
    assert parsed.vectors["position"] == (-5.0, 3.0)
    assert parsed.strings["helix_hand"] == "right"


class _Quantity:
    def __init__(self, value: float):
        self.Value = value


def test_feature_tree_normalizes_object_angle_properties_to_radians():
    obj = SimpleNamespace(
        Name="Cone",
        Label="Cone",
        TypeId="Part::Cone",
        Angle=_Quantity(90.0),
        TaperAngle=_Quantity(30.0),
        OutList=[],
    )
    bank = MeasurementBank()

    FeatureTreeExtractor().extract(
        shape=None,
        doc=SimpleNamespace(Objects=[obj]),
        owner_label=None,
        bank=bank,
    )

    assert bank.feature_tree[0].properties["Angle"] == pytest.approx(math.pi / 2)
    assert bank.feature_tree[0].properties["TaperAngle"] == pytest.approx(math.pi / 6)


def test_checker_never_loads_param_check_from_candidate_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "freecad_validator.consistency.checker.extract_bank",
        lambda _path: MeasurementBank(solid_count=1),
    )
    candidate_dir = tmp_path / "candidate"
    spec_dir = tmp_path / "case"
    candidate_dir.mkdir()
    spec_dir.mkdir()
    candidate = candidate_dir / "answer.FCStd"
    candidate.write_bytes(b"not opened by patched extractor")
    (candidate_dir / "param_check.py").write_text(
        "def apply(report, bank, spec, tol_scalar):\n"
        "    report.error = 'candidate checker executed'\n",
        encoding="utf-8",
    )
    trusted_check = spec_dir / "param_check.py"
    trusted_check.write_text(
        "def apply(report, bank, spec, tol_scalar):\n"
        "    report.error = 'trusted checker executed'\n",
        encoding="utf-8",
    )
    spec = {"name": "case", "key_parameters": "length = 10 mm"}
    spec_path = spec_dir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    checker = ConsistencyChecker()

    without_explicit_path = checker.check(spec, candidate)
    with_trusted_path = checker.check(spec_path, candidate)

    assert without_explicit_path.error is None
    assert with_trusted_path.error == "trusted checker executed"


def test_trusted_category_refines_cad_anchored_gear_without_partdesign_history(
    tmp_path, monkeypatch
):
    """A minimal alternative feature history still yields positive evidence.

    The bank has only the repeated CAD tooth-ring surfaces, not a Sketch/Pad
    feature history. That is enough for GearCategory to verify module and
    pitch diameter through the trusted, case-controlled checker.
    """
    ring = {
        "count": 20,
        "axis": (0.0, 0.0, 1.0),
        "convex": True,
        "centroids": [],
        "axial_extent": 10.0,
    }
    bank = MeasurementBank(
        solid_count=1,
        cylinder_clusters=[
            ClusterSummary(id="tip", radius=11.0, **ring),
            ClusterSummary(id="root", radius=8.75, **ring),
        ],
    )
    monkeypatch.setattr("freecad_validator.consistency.checker.extract_bank", lambda _path: bank)
    candidate = tmp_path / "answer.FCStd"
    candidate.write_bytes(b"not opened by patched extractor")
    trusted_check = tmp_path / "param_check.py"
    trusted_check.write_text(
        "from freecad_validator.consistency.categories.gear import GearCategory\n"
        "def apply(report, bank, spec, tol_scalar):\n"
        "    GearCategory().apply(report, bank, spec, tol_scalar)\n",
        encoding="utf-8",
    )
    spec = {
        "name": "alternate_gear_history",
        "key_parameters": "number_of_teeth = 20\ngear_module = 1 mm\ngear_pitch_diameter = 20 mm",
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    report = ConsistencyChecker().check(spec_path, candidate)

    assert report.summary is not None
    assert report.summary.consistent == 3
    assert report.summary.inconsistent == 0
    assert report.summary.not_found == 0


def test_checker_reports_when_spec_has_zero_parseable_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "freecad_validator.consistency.checker.extract_bank",
        lambda _path: MeasurementBank(solid_count=1),
    )
    candidate = tmp_path / "answer.FCStd"
    candidate.write_bytes(b"not opened by patched extractor")

    report = ConsistencyChecker().check(
        {"name": "bad", "key_parameters": "gear_ratio = 100/25"},
        candidate,
    )

    assert report.summary is not None
    assert report.summary.total_params == 0
    assert report.error == "spec yielded zero parseable measurable parameters"


def test_checker_keeps_unmeasured_string_parameters_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "freecad_validator.consistency.checker.extract_bank",
        lambda _path: MeasurementBank(solid_count=1),
    )
    candidate = tmp_path / "answer.FCStd"
    candidate.write_bytes(b"not opened by patched extractor")

    report = ConsistencyChecker().check(
        {"name": "helical", "key_parameters": "helix_hand = right"},
        candidate,
    )

    assert report.summary is not None
    assert report.summary.total_params == 1
    assert report.summary.not_found == 1
    assert report.not_found[0].param == "helix_hand"


def test_trusted_param_check_is_registered_while_it_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "freecad_validator.consistency.checker.extract_bank",
        lambda _path: MeasurementBank(solid_count=1),
    )
    candidate = tmp_path / "answer.FCStd"
    candidate.write_bytes(b"not opened by patched extractor")
    param_check = tmp_path / "param_check.py"
    param_check.write_text(
        "import sys\n"
        "MODULE_WAS_REGISTERED = __name__ in sys.modules\n"
        "def apply(report, bank, spec, tol_scalar):\n"
        "    report.error = 'registered' if MODULE_WAS_REGISTERED else 'missing'\n",
        encoding="utf-8",
    )

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps({"name": "case", "key_parameters": "length = 10 mm"}),
        encoding="utf-8",
    )
    report = ConsistencyChecker().check(spec_path, candidate)

    assert report.error == "registered"
