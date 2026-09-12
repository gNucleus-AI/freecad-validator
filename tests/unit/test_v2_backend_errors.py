"""Backend failures must remain errors, with usable CLI and batch behavior."""

import builtins
import csv
import json
from pathlib import Path

import pytest

from freecad_validator import Validator
from freecad_validator.cli.main import main as cli_main
from freecad_validator.comparators import geometry, occt_bbox
from freecad_validator.comparators.base import ComparisonResult
from freecad_validator.comparators.occt_bbox import OBBMeasurementError, OCCTUnavailableError
from freecad_validator.scorers.geometry_v2 import HeuristicGeometryScorerV2
from freecad_validator.scorers.geometry_v2 import main as v2_main
from freecad_validator.validator import main as validator_main


@pytest.fixture
def missing_ocp(monkeypatch):
    original = builtins.__import__
    attempts = []

    def unavailable(name, *args, **kwargs):
        if name == "OCP" or name.startswith("OCP."):
            attempts.append(name)
            raise ImportError("libGL.so.1: cannot open shared object file")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    return attempts


def test_v2_constructor_checks_backend_but_v1_does_not(missing_ocp):
    Validator(scorer_version="v1")
    assert missing_ocp == []
    for factory in (Validator, HeuristicGeometryScorerV2):
        with pytest.raises(OCCTUnavailableError, match="libGL.so.1"):
            factory()
    assert len(missing_ocp) == 2


ENTRYPOINTS = [
    (cli_main, ["validate", "candidate.FCStd", "reference.FCStd", "spec.json"]),
    (validator_main, ["candidate.FCStd", "reference.FCStd", "spec.json"]),
    (v2_main, ["reference.FCStd", "candidate.FCStd"]),
]


@pytest.mark.parametrize("entrypoint,args", ENTRYPOINTS)
def test_missing_backend_cli_exits_without_traceback(missing_ocp, capsys, entrypoint, args):
    assert entrypoint(args) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "gnucleus-freecad-validator[v2]" in output.err
    assert "libGL.so.1" in output.err
    assert "Traceback" not in output.err
    assert "usage:" not in output.err


def _make_case(root, name):
    case = root / "data" / name
    case.mkdir(parents=True)
    for filename in ("candidate.FCStd", "reference.FCStd", "spec.json"):
        (case / filename).touch()
    return case


def test_missing_backend_batch_fails_once_before_processing(missing_ocp, capsys, tmp_path):
    _make_case(tmp_path, "first")
    _make_case(tmp_path, "second")
    assert cli_main(["batch", "--sample-data-dir", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert len(missing_ocp) == 1
    assert "validating" not in output.out
    assert "libGL.so.1" in output.err
    assert not (tmp_path / "validation_results.csv").exists()
    assert not (tmp_path / "validation_summary.json").exists()


@pytest.fixture
def fake_geometry(monkeypatch):
    """Supply already-extracted geometry to exercise scoring and error propagation."""
    monkeypatch.setattr("freecad_validator.scorers.geometry_v2.ensure_ocp_available", lambda: None)
    monkeypatch.setattr("freecad_validator._freecad_loader.import_freecad", lambda: object())
    monkeypatch.setattr(
        geometry,
        "_select_shape_and_features",
        lambda path, **kwargs: dict(
            solid_count=1,
            n_faces=6,
            n_vertices=8,
            volume=1.0,
            area=6.0,
            bbox_sorted_mm=[1.0, 1.0, 1.0],
            surface_area_by_type={"Plane": 6.0},
            principal_moments_normalized=[1.0, 1.0, 1.0],
            brep=path,
        ),
    )
    for target in (
        "freecad_validator.comparators.icp.FaceCenterICPComparator.compare",
        "freecad_validator.scorers.spec_consistency.HeuristicSpecConsistencyScorer.score",
    ):
        monkeypatch.setattr(target, lambda *args: ComparisonResult(score=1.0, reason="matched"))


@pytest.mark.parametrize("role", ["reference", "candidate"])
def test_measurement_error_identifies_input_without_scoring_zero(fake_geometry, monkeypatch, role):
    def measure(brep):
        if brep == f"{role}.FCStd":
            raise OBBMeasurementError("meshing did not complete")
        return [1.0, 1.0, 1.0]

    monkeypatch.setattr(occt_bbox, "oriented_bbox_dimensions", measure)
    with pytest.raises(OBBMeasurementError, match=f"{role} model '{role}.FCStd'"):
        Validator().validate("candidate.FCStd", "reference.FCStd", "spec.json")


@pytest.mark.parametrize("entrypoint,args", ENTRYPOINTS)
def test_measurement_failure_cli_is_an_error(fake_geometry, monkeypatch, capsys, entrypoint, args):
    def measure(_brep):
        raise OBBMeasurementError("meshing did not complete")

    monkeypatch.setattr(occt_bbox, "oriented_bbox_dimensions", measure)
    assert entrypoint(args) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "reference model 'reference.FCStd'" in output.err
    assert "meshing did not complete" in output.err
    assert "Traceback" not in output.err
    assert "usage:" not in output.err


def test_batch_continues_after_measurement_error_without_polluting_averages(
    fake_geometry, monkeypatch, tmp_path
):
    _make_case(tmp_path, "first_bad")
    _make_case(tmp_path, "second_good")

    def measure(brep):
        path = Path(brep)
        if path.parent.name == "first_bad" and path.name == "candidate.FCStd":
            raise OBBMeasurementError("meshing did not complete")
        return [1.0, 1.0, 1.0]

    monkeypatch.setattr(occt_bbox, "oriented_bbox_dimensions", measure)
    assert cli_main(["batch", "--sample-data-dir", str(tmp_path)]) == 0
    with (tmp_path / "validation_results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "candidate model 'candidate.FCStd'" in rows[0]["error"]
    for field in ("geometry_similarity", "cad_spec_consistency", "combined"):
        assert rows[0][field] == ""
        assert float(rows[1][field]) == 1.0
    summary = json.loads((tmp_path / "validation_summary.json").read_text())
    assert summary["cases_errored"] == 1
    assert summary["cases_validated"] == 1
    assert summary["combined"]["n"] == 1
    assert summary["combined"]["mean"] == 1.0
