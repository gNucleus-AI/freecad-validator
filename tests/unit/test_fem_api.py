"""FreeCAD-free tests for the public FEM API and CLI wiring."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from freecad_validator.cli.main import main
from freecad_validator.fem import FEMScoringReport, FEMValidator, ScoringReport


def _report() -> ScoringReport:
    return ScoringReport(
        case_id="example",
        overall_score=91.0,
        grade="excellent",
        confidence=1.0,
        reproducibility_status="reproducible",
    )


def test_fem_scoring_report_is_the_public_report_type() -> None:
    assert FEMScoringReport is ScoringReport


def test_validator_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="displacement_tolerance must be positive"):
        FEMValidator(displacement_tolerance=0.0)
    with pytest.raises(ValueError, match="mesh_budget_zero_ratio"):
        FEMValidator(mesh_budget_zero_ratio=1.0)
    with pytest.raises(ValueError, match="timeout_seconds"):
        FEMValidator(timeout_seconds=0.0)


def test_validator_forwards_configuration() -> None:
    validator = FEMValidator(
        freecad_cmd="/fake/freecadcmd",
        extract_dir="/tmp/extract",
        require_boolean=True,
        timeout_seconds=45.0,
    )
    with patch("freecad_validator.fem.validator.score_step_fcstd", return_value=_report()) as score:
        report = validator.validate(
            step_path="source.step",
            reference_fcstd="reference.FCStd",
            candidate_fcstd="candidate.FCStd",
        )

    assert report.overall_score == 91.0
    assert score.call_args.kwargs["require_boolean"] is True
    assert score.call_args.kwargs["freecad_cmd"] == "/fake/freecadcmd"
    assert score.call_args.kwargs["timeout_seconds"] == 45.0


def test_fem_score_cli_emits_json(capsys) -> None:
    with patch("freecad_validator.cli.main.FEMValidator.validate", return_value=_report()):
        code = main(
            [
                "fem-score",
                "source.step",
                "reference.FCStd",
                "candidate.FCStd",
                "--json",
                "--require-boolean",
                "--timeout",
                "45",
            ]
        )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_score"] == 91.0
    assert payload["grade"] == "excellent"
