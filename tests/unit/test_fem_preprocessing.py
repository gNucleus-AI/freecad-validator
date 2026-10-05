"""Static FEM multiplication preserves gates and never rewards geometry errors."""

from pathlib import Path
from unittest.mock import patch

import pytest
from test_fem_step_interface import LABEL, STEP, _cand, resolve_fake_freecad_command  # noqa: F401

from freecad_validator.fem.pre_process.errors import EvaluationError
from freecad_validator.fem.pre_process.integration import apply_preprocessing_score
from freecad_validator.fem.schema import ScoringReport, grade_from_score
from freecad_validator.fem.step_interface import (
    ExtractionError,
    score_step_fcstd,
    score_trusted_payloads,
)


@pytest.mark.parametrize("geometry,expected", [(3 / 4, 61.875), (2 / 3, 55.0), (1, 82.5), (0, 0)])
def test_multiplication_and_grade(geometry, expected):
    original = ScoringReport("test", 82.5, "good")
    report = apply_preprocessing_score(original, geometry)
    assert report.overall_score == pytest.approx(expected)
    assert report.overall_score <= original.overall_score
    assert report.grade == grade_from_score(expected)
    assert original.overall_score == 82.5
    assert original.subscores_details == {}
    assert report.subscores_details["preprocessing"]["weight"] == 0


def test_no_preprocessing_is_unchanged():
    original = ScoringReport("test", 82.5, "good")
    assert apply_preprocessing_score(original, None) is original
    assert (
        score_trusted_payloads(STEP, LABEL, _cand()).to_dict()
        == score_trusted_payloads(
            STEP,
            LABEL,
            _cand(),
            preprocessing_score=None,
        ).to_dict()
    )


def test_existing_gate_stays_zero():
    bad = _cand(material={"E_MPa": 0, "nu": 0.3})
    original = score_trusted_payloads(STEP, LABEL, bad)
    adjusted = score_trusted_payloads(STEP, LABEL, bad, preprocessing_score=0.8)
    assert original.overall_score == adjusted.overall_score == 0
    assert original.gates_triggered == adjusted.gates_triggered
    assert original.failure_modes_detected == adjusted.failure_modes_detected


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("nan"), float("inf"), "0.8", True])
def test_invalid_geometry_is_an_evaluation_error(invalid):
    with pytest.raises(EvaluationError):
        score_trusted_payloads(STEP, LABEL, _cand(), preprocessing_score=invalid)


def test_no_double_multiplication():
    adjusted = apply_preprocessing_score(ScoringReport("test", 82.5, "good"), 0.8)
    with pytest.raises(EvaluationError, match="already"):
        apply_preprocessing_score(adjusted, 0.8)


def test_existing_flag_alone_enables_automatic_multiplier(tmp_path):
    raw = {**STEP, "volume_mm3": 2e6, "surface_area_mm2": 1e5}
    candidate = _cand()
    prepared = {"geometry": {**candidate["geometry"], "surface_area_mm2": 5e4}}
    geometry = {
        "score": 0.75,
        "reference_changed_body_count": 4,
        "correct_body_count": 3,
        "extra_changed_body_count": 0,
    }
    with patch(
        "freecad_validator.fem.step_interface._extract",
        side_effect=[raw, prepared, LABEL, candidate, geometry],
    ) as extract:
        report = score_step_fcstd(
            "raw.step",
            "reference.FCStd",
            "candidate.FCStd",
            extract_dir=str(tmp_path),
            require_preprocessing=True,
        )
    baseline = score_trusted_payloads(LABEL["geometry"], LABEL, candidate)
    assert report.overall_score == pytest.approx(baseline.overall_score * 0.75)
    assert extract.call_args.args[2] == "raw.step"
    assert extract.call_args.kwargs["extra_args"] == [
        str(Path("reference.FCStd").resolve()),
        str(Path("candidate.FCStd").resolve()),
    ]
    assert extract.call_args.kwargs["timeout_seconds"] == 1800.0


def test_invalid_candidate_geometry_is_zero_with_reason(tmp_path):
    raw = {**STEP, "volume_mm3": 2e6, "surface_area_mm2": 1e5}
    prepared = {"geometry": {**LABEL["geometry"], "surface_area_mm2": 5e4}}
    with patch(
        "freecad_validator.fem.step_interface._extract",
        side_effect=[
            raw,
            prepared,
            LABEL,
            _cand(),
            {"status": "candidate_invalid", "score": 0.0, "error": "No analysis solids"},
        ],
    ):
        report = score_step_fcstd(
            "raw.step",
            "reference.FCStd",
            "candidate.FCStd",
            extract_dir=str(tmp_path),
            require_preprocessing=True,
        )
    assert report.overall_score == 0
    assert report.subscores_details["preprocessing"]["multiplier"] == 0
    assert any(
        "preprocessing_candidate_invalid: No analysis solids" in text for text in report.evidence
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": "evaluation_error", "error": "ambiguous"},
        {"score": None, "reference_changed_body_count": 1, "extra_changed_body_count": 0},
    ],
)
def test_missing_worker_score_is_not_full_credit(tmp_path, payload):
    raw = {**STEP, "volume_mm3": 2e6, "surface_area_mm2": 1e5}
    prepared = {"geometry": {**LABEL["geometry"], "surface_area_mm2": 5e4}}
    with patch(
        "freecad_validator.fem.step_interface._extract",
        side_effect=[raw, prepared, LABEL, _cand(), payload],
    ):
        with pytest.raises(ExtractionError):
            score_step_fcstd(
                "raw.step",
                "reference.FCStd",
                "candidate.FCStd",
                extract_dir=str(tmp_path),
                require_preprocessing=True,
            )
