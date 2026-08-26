"""Regression tests for trusted FEM replay field comparison."""

import math

from freecad_validator.fem.replay_compare import compare_result_snapshots, select_scored_results


def _snapshot(scale=1.0):
    return {
        "node_numbers": [1, 2, 3, 4],
        "fields": {
            "DisplacementLengths": [value * scale for value in (0.0, 1.0, 2.0, 4.0)],
            "DisplacementVectors": [(0.0, 0.0, value * scale) for value in (0.0, 1.0, 2.0, 4.0)],
            "vonMises": [value * scale for value in (0.0, 10.0, 20.0, 40.0)],
            "MaxShear": [value * scale for value in (0.0, 5.0, 10.0, 20.0)],
        },
    }


def test_identical_replayed_fields_pass():
    result = compare_result_snapshots(_snapshot(), _snapshot(), "static")
    assert result["passed"] is True
    assert result["status"] == "verified"


def test_one_percent_replay_difference_passes():
    result = compare_result_snapshots(_snapshot(1.01), _snapshot(), "static")
    assert result["passed"] is True


def test_three_percent_replay_difference_is_accepted_with_warning():
    result = compare_result_snapshots(_snapshot(1.03), _snapshot(), "static")
    assert result["passed"] is True
    assert result["status"] == "accepted_mismatch"
    assert any("exceeds 2%" in warning for warning in result["warnings"])


def test_accepted_mismatch_scores_stored_results():
    scored, source = select_scored_results(
        {"max_displacement_mm": 1.1},
        {"max_displacement_mm": 1.0},
        {"status": "accepted_mismatch"},
    )
    assert scored == {"max_displacement_mm": 1.1}
    assert source == "stored"


def test_verified_replay_scores_replayed_results():
    scored, source = select_scored_results(
        {"max_displacement_mm": 1.01},
        {"max_displacement_mm": 1.0},
        {"status": "verified"},
    )
    assert scored == {"max_displacement_mm": 1.0}
    assert source == "replayed"


def test_more_than_ten_percent_rms_difference_fails():
    result = compare_result_snapshots(_snapshot(1.11), _snapshot(), "static")
    assert result["passed"] is False
    assert result["status"] == "mismatch"
    assert any("exceeds 10%" in failure for failure in result["failures"])


def test_more_than_ten_percent_peak_difference_fails():
    stored = _snapshot()
    stored["fields"]["vonMises"][-1] *= 1.11
    result = compare_result_snapshots(stored, _snapshot(), "static")
    assert result["passed"] is False
    assert any(
        "peak error" in failure and "exceeds 10%" in failure for failure in result["failures"]
    )


def test_matching_peak_with_fabricated_constant_field_fails():
    fabricated = _snapshot()
    fabricated["fields"]["DisplacementLengths"] = [4.0, 4.0, 4.0, 4.0]
    result = compare_result_snapshots(fabricated, _snapshot(), "static")

    assert result["passed"] is False
    comparison = result["field_comparisons"]["DisplacementLengths"]
    assert comparison["peak_relative_error"] == 0.0
    assert comparison["normalized_rms_error"] > 0.10


def test_missing_required_static_field_fails():
    stored = _snapshot()
    del stored["fields"]["DisplacementVectors"]
    result = compare_result_snapshots(stored, _snapshot(), "static")
    assert result["passed"] is False
    assert any(
        "missing required field DisplacementVectors" in failure for failure in result["failures"]
    )


def test_changed_node_order_and_nonfinite_field_fail():
    stored = _snapshot()
    stored["node_numbers"] = [4, 3, 2, 1]
    stored["fields"]["vonMises"][2] = math.nan
    result = compare_result_snapshots(stored, _snapshot(), "static")

    assert result["passed"] is False
    assert any("node ordering" in failure for failure in result["failures"])
    assert result["field_comparisons"]["vonMises"]["finite"] is False
