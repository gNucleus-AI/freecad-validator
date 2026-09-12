"""Spec configuration and measurement errors never become candidate scores."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from freecad_validator.cli.main import main as cli_main
from freecad_validator.comparators.base import ComparisonResult
from freecad_validator.measurement.schema import FeatureTreeEntry, MeasurementBank
from freecad_validator.measurement.spatial import SpatialMeasurementError
from freecad_validator.scorers.geometry_v2 import HeuristicGeometryScorerV2
from freecad_validator.validator import main as validator_main


def write_case(directory, *, stale=False, spatial=False):
    directory.mkdir(parents=True)
    for name in ("candidate.FCStd", "reference.FCStd"):
        (directory / name).write_bytes(b"fixture")
    spec = {
        "key_parameters": "length = 10 mm" + ("\nwidth = 4 mm" if stale else ""),
        "geometry_bindings": {
            "version": 1,
            "parameters": {"length": {"mode": "legacy", "reason": "Construction length"}},
        },
    }
    if spatial:
        spec["geometry_bindings"].update(
            datum={
                "center": (0, 0, 0),
                "diagonal": 10,
                "frames": [((1, 0, 0), (0, 1, 0), (0, 0, 1))],
            },
            parameters={
                "length": {
                    "mode": "geometry",
                    "quantity": "length",
                    "reason": "Corresponding straight edge",
                    "witnesses": [
                        {"kind": "line", "position": (0, 0, 0), "direction": (1, 0, 0), "scale": 10}
                    ],
                }
            },
        )
    (directory / "spec.json").write_text(json.dumps(spec))
    return [str(directory / name) for name in ("candidate.FCStd", "reference.FCStd", "spec.json")]


@pytest.fixture
def matched_geometry(monkeypatch):
    monkeypatch.setattr(
        HeuristicGeometryScorerV2,
        "score",
        lambda *args: ComparisonResult(score=1.0, reason="Matched geometry"),
    )


@pytest.mark.parametrize("entrypoint", [cli_main, validator_main])
def test_stale_v2_binding_is_a_cli_error(
    tmp_path, monkeypatch, capsys, matched_geometry, entrypoint
):
    paths = write_case(tmp_path / "case", stale=True)

    def unexpected_extraction(*args, **kwargs):
        pytest.fail("Invalid binding configuration must be rejected before measurement")

    monkeypatch.setattr("freecad_validator.consistency.checker.extract_bank", unexpected_extraction)
    args = (["validate"] if entrypoint is cli_main else []) + paths + ["--json"]
    assert entrypoint(args) == 1
    captured = capsys.readouterr()
    assert "missing=['width']" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out


@pytest.mark.parametrize("entrypoint", [cli_main, validator_main])
def test_spatial_failure_is_a_cli_error(
    tmp_path, monkeypatch, capsys, matched_geometry, entrypoint
):
    paths = write_case(tmp_path / "case", spatial=True)

    def fail(*args, **kwargs):
        assert kwargs["include_spatial"]
        raise SpatialMeasurementError("Native wall measurement failed")

    monkeypatch.setattr("freecad_validator.consistency.checker.extract_bank", fail)
    args = (["validate"] if entrypoint is cli_main else []) + paths + ["--json"]
    assert entrypoint(args) == 1
    captured = capsys.readouterr()
    assert "Native wall measurement failed" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out


@pytest.mark.parametrize("failure", ["stale", "spatial"])
def test_batch_records_spec_errors_and_scores_remaining_cases(
    tmp_path, monkeypatch, matched_geometry, failure
):
    write_case(tmp_path / "data" / "a_bad", stale=failure == "stale", spatial=failure == "spatial")
    write_case(tmp_path / "data" / "b_good")

    def extract(path, **kwargs):
        if Path(path).parent.name == "a_bad":
            assert kwargs["include_spatial"]
            raise SpatialMeasurementError("Native wall measurement failed")
        return MeasurementBank(
            solid_count=1,
            feature_tree=[
                FeatureTreeEntry(
                    name="Pad", label="Pad", type_id="PartDesign::Pad", properties={"Length": 10}
                )
            ],
        )

    monkeypatch.setattr("freecad_validator.consistency.checker.extract_bank", extract)
    assert cli_main(["batch", "--sample-data-dir", str(tmp_path)]) == 0
    with (tmp_path / "validation_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    bad, good = rows
    assert bad["case_id"] == "a_bad" and bad["error"]
    assert all(
        bad[key] == "" for key in ("geometry_similarity", "cad_spec_consistency", "combined")
    )
    error_type = "GeometryBindingError" if failure == "stale" else "SpatialMeasurementError"
    assert error_type in bad["error"]
    assert good["case_id"] == "b_good" and not good["error"]
    assert float(good["combined"]) == 1.0
    summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert summary["cases_errored"] == summary["cases_validated"] == 1
    assert summary["combined"]["mean"] == 1.0
