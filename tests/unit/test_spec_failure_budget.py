"""Unit coverage for failure-budget spec scoring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freecad_validator import Validator
from freecad_validator.consistency.report import (
    ConsistencyReport,
    ParamFinding,
    compute_summary,
)
from freecad_validator.scorers.spec_consistency import (
    DEFAULT_FAILURE_BUDGET,
    HeuristicSpecConsistencyScorer,
    _failure_budget_score,
)
from freecad_validator.validator import DEFAULT_V2_FAILURE_BUDGET


@pytest.mark.parametrize(
    ("total_params", "failures", "expected"),
    [
        (0, 0, 0.0),
        (4, 0, 1.0),
        (4, 1, 0.75),
        (4, 4, 0.0),
        (10, 1, 0.9),
        (20, 1, 0.95),
        (20, 5, 0.75),
        (20, 10, 0.5),
        (100, 10, 0.9),
        (2000, 10, 0.995),
        (2000, 1000, 0.5),
        (3, 1, 0.6667),
    ],
)
def test_failure_budget_score(total_params, failures, expected):
    assert _failure_budget_score(total_params=total_params, failures=failures) == expected


def test_custom_failure_budget():
    assert _failure_budget_score(total_params=40, failures=5, failure_budget=10) == 0.5
    assert _failure_budget_score(total_params=40, failures=10, failure_budget=10) == 0.0


def test_large_failure_budget_does_not_round_a_failure_to_perfect():
    score = _failure_budget_score(total_params=20_000, failures=1, failure_budget=20_000)

    assert score == pytest.approx(1.0 - 1.0 / 20_000)
    assert score < 1.0


@pytest.mark.parametrize("failure_budget", [0, -1, 1.5, True])
def test_invalid_failure_budget_is_rejected(failure_budget):
    with pytest.raises(ValueError, match="positive integer"):
        HeuristicSpecConsistencyScorer(failure_budget=failure_budget)  # type: ignore[arg-type]


def _finding(index: int) -> ParamFinding:
    return ParamFinding(param=f"param_{index}", spec_value=index)


def test_scorer_counts_inconsistent_and_not_found_as_failures(tmp_path):
    spec = tmp_path / "spec.json"
    candidate = tmp_path / "candidate.FCStd"
    spec.write_text("{}", encoding="utf-8")
    candidate.write_bytes(b"")

    report = ConsistencyReport(spec_name="test", fcstd_path=str(candidate))
    report.consistent = [_finding(index) for index in range(12)]
    report.inconsistent = [_finding(12), _finding(13)]
    report.not_found = [_finding(14)]
    report.summary = compute_summary(report)

    scorer = HeuristicSpecConsistencyScorer()
    scorer._checker = SimpleNamespace(check=lambda _spec, _candidate, **_kwargs: report)
    result = scorer.score(str(spec), str(candidate))

    assert result.score == 0.8
    assert result.details["failures"] == 3
    assert result.details["failure_denominator"] == 15
    assert result.details["failure_budget"] is None
    assert result.details["raw_consistency_rate"] == 0.8
    assert "failure_budget=disabled" in result.reason


def test_scorer_applies_configured_failure_budget(tmp_path):
    spec = tmp_path / "spec.json"
    candidate = tmp_path / "candidate.FCStd"
    spec.write_text("{}", encoding="utf-8")
    candidate.write_bytes(b"")

    report = ConsistencyReport(spec_name="test", fcstd_path=str(candidate))
    report.consistent = [_finding(index) for index in range(12)]
    report.inconsistent = [_finding(12), _finding(13)]
    report.not_found = [_finding(14)]
    report.summary = compute_summary(report)

    scorer = HeuristicSpecConsistencyScorer(failure_budget=10)
    scorer._checker = SimpleNamespace(check=lambda _spec, _candidate, **_kwargs: report)
    result = scorer.score(str(spec), str(candidate))

    assert result.score == 0.7
    assert result.details["failure_denominator"] == 10
    assert result.details["failure_budget"] == 10
    assert "failure_budget=10" in result.reason


def test_validator_exposes_default_and_custom_failure_budget():
    # The default budget is scorer-version dependent: v2 (the default
    # scorer) applies DEFAULT_V2_FAILURE_BUDGET, v1 keeps the legacy
    # disabled default so it reproduces pre-0.4.0 numbers exactly.
    assert Validator().spec_failure_budget == DEFAULT_V2_FAILURE_BUDGET
    assert Validator(scorer_version="v1").spec_failure_budget == DEFAULT_FAILURE_BUDGET
    assert Validator(spec_failure_budget=20).spec_failure_budget == 20
    assert Validator(scorer_version="v1", spec_failure_budget=20).spec_failure_budget == 20
